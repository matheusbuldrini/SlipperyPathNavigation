import gymnasium
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from typing import Callable
from AStarSolver import AStarSolver
from AStarWeightedSolver import AStarWeightedSolver
from BFSSolver import BFSSolver
from HumanSolver import HumanSolver
from DijkstraSolver import DijkstraSolver
from Solver import Solver
from custom_maps import *

RENDER = True

def init_env(map, is_slippery = False, success_rate = 1.0/3.0, seed = 0):
    env = gymnasium.make(
            'FrozenLake-v1',
            render_mode='human' if RENDER else None,
            desc=map,
            map_name=None,
            is_slippery=is_slippery,
            success_rate=success_rate,
            reward_schedule=(1, 0, 0),
            max_episode_steps=100000
            )
    env.reset(seed=seed)
    env.unwrapped.metadata["render_fps"] = 40
    return env


def run(solver: Callable[..., Solver], map, success_rate, seed, **kwargs):
    steps = 0
    env = init_env(map, is_slippery=int(success_rate) == 1, success_rate=success_rate, seed=seed)
    solver_instance = solver(env, **kwargs)
    while True:
        if RENDER: env.render()
        observation, reward, terminated, truncated, info = env.step(solver_instance.get_next_action())
        steps += 1
        if float(reward) > 0:
            # print("reward received!")
            # print(f"Total steps taken: {steps}")
            env.close()
            return True, steps, solver_instance.cost

        if terminated:
            print("terminated!")
            print(f"Total steps taken: {steps}")
            env.close()
            return False, steps, solver_instance.cost

        if truncated:
            print("truncated!")
            print(f"Total steps taken: {steps}")
            env.close()
            return False, steps, solver_instance.cost

solvers = [
    # (HumanSolver, {}),
    (BFSSolver, {}),
    (DijkstraSolver, {}),
    (AStarSolver, {}),
    (AStarWeightedSolver, {"weight": 1.5}),
    (AStarWeightedSolver, {"weight": 2.0}),
    (AStarWeightedSolver, {"weight": 2.5}),
    (AStarWeightedSolver, {"weight": 3.0}),
    ]

for i in range(10):
    map = generate_random_map(size=30, seed=None, p=0.8)
    #map = map_DijkstraVsBFS_2
    for solver, kwargs in solvers:
        print(f"Running {solver.__name__} with {kwargs}...")
        success, steps, cost = run(solver, map=map, success_rate=1.0, seed=0, **kwargs)
        print(f"Success: {success}, steps: {steps}, cost: {cost}")
    print("--------------------------------------------------")
        