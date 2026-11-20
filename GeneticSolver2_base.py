import gymnasium
from neural_controller import get_next_action
import numpy as np
from custom_maps import *
from gymnasium.envs.toy_text.frozen_lake import generate_random_map


INPUT_SIZE = 1
NUMBER_OF_HIDDEN_UNITS = 20
OUTPUT_SIZE = 4

MAP = generate_random_map(size=4, seed=0, p=0.8)

rng = np.random.default_rng(seed=0)
model_size = (INPUT_SIZE * NUMBER_OF_HIDDEN_UNITS) + (NUMBER_OF_HIDDEN_UNITS * OUTPUT_SIZE) + (NUMBER_OF_HIDDEN_UNITS + OUTPUT_SIZE)
model = rng.random(model_size, dtype=float)

# print(model)

env = gymnasium.make(
        'FrozenLake-v1',
        #render_mode='human',
        desc=MAP,
        map_name=None,
        is_slippery=False,
        success_rate=1.0,
        reward_schedule=(1000, -1000, 0), # goal, hole, normal
        max_episode_steps=200,
        )

def evaluate(model, _print=False):
    init, _ = env.reset(seed=0)
    init = [init]
    score = 0

    action = get_next_action(model, 
                    init,
                    INPUT_SIZE,
                    NUMBER_OF_HIDDEN_UNITS,
                    OUTPUT_SIZE)
    if _print: print(action)
    visited = set()
    while True:
        if _print: print(action)
        observation, reward, terminated, truncated, info = env.step(action)
        if not terminated and observation not in visited: # reward new observed tiles
            reward = 1

        visited.add(observation)
        
        observation = [observation]
        score += float(reward)
        action = get_next_action(model, 
                    observation,
                    INPUT_SIZE,
                    NUMBER_OF_HIDDEN_UNITS,
                    OUTPUT_SIZE)
        #print(action)
        #print(observation)

        if terminated:
            #print("terminated!")
            break

        if truncated:
            #print("truncated!")
            break
    #print("\t"+str(score))
    return score

from cmaes import CMA

def get_best(early_return=False):
    best_x = None
    best_value = float("inf")

    optimizer = CMA(mean=np.zeros(model_size), sigma=0.1, population_size=20)
    for generation in range(80):
        solutions = []
        for _ in range(optimizer.population_size):
            x = optimizer.ask()
            value = -evaluate(x)

            solutions.append((x, value))
            if value < best_value:
                best_value = value
                best_x = x.copy()
            print(f"#{generation} {value} [best={best_value}]") # (x={x})")
            if early_return and best_value < -1000: return best_x, best_value
        optimizer.tell(solutions)

    return best_x, best_value

model, score = get_best(early_return = True)

env = gymnasium.make(
        'FrozenLake-v1',
        render_mode='human',
        desc=MAP,
        map_name=None,
        is_slippery=False,
        success_rate=1.0,
        max_episode_steps=10000,
        )

print(f"evaluating: score: {score}")
evaluate(model, False)