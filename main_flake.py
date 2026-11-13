import gymnasium
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from typing import Callable
from BFSSolver import BFSSolver
from HumanSolver import HumanSolver
from Solver import Solver


def init_env(map_size = 80, is_slippery = True, success_rate = 1.0, seed = 0):
    env = gymnasium.make(
            'FrozenLake-v1',
            render_mode="human",
            desc=generate_random_map(size=map_size, seed=seed),
            map_name=None,
            is_slippery=is_slippery,
            success_rate=success_rate,
            reward_schedule=(1, 0, 0),
            max_episode_steps=100000
            )
    env.reset(seed=seed)
    env.unwrapped.metadata["render_fps"] = 240
    return env


def run(solver: Callable[..., Solver] = HumanSolver):
    cost = 0
    env = init_env()
    solver_instance = solver(env)
    while True:
        env.render()
        observation, reward, terminated, truncated, info = env.step(solver_instance.get_next_action())
        cost += 1
        if float(reward) > 0:
            print("reward received!")
            print(f"Total steps taken: {cost}")
            env.close()
            return True, cost

        if terminated:
            print("terminated!")
            print(f"Total steps taken: {cost}")
            env.close()
            return False, cost

        if truncated:
            print("truncated!")
            print(f"Total steps taken: {cost}")
            env.close()
            return False, cost

success, cost = run(BFSSolver)
print(f"Success: {success}, Cost: {cost}")
