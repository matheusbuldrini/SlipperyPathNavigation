import gymnasium
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from typing import Callable
from AStarSolver import AStarSolver
from AStarWeightedSolver import AStarWeightedSolver
from AndOrSolver import AndOrSolver
from BFSSolver import BFSSolver
from HumanSolver import HumanSolver
from DijkstraSolver import DijkstraSolver
from Solver import Solver
from custom_maps import *


RENDER = True

def run(solver: Callable[..., Solver], map, solver_args, env_args):
    steps = 0
    env = gymnasium.make(
        'FrozenLake-v1',
        render_mode='human' if RENDER else None,
        desc=map,
        map_name=None,
        is_slippery=env_args.get("is_slippery") or False,
        success_rate=env_args.get("success_rate") or 1.0,
        reward_schedule=(1, 0, 0),
        max_episode_steps=100000
        )
    env.reset(seed=env_args.get("seed"))
    env.unwrapped.metadata["render_fps"] = 40

    solver_instance = solver(env, **solver_args)
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
    (AndOrSolver, {}, {"is_slippery": False}),
    #(AndOrSolver, {}, {"is_slippery": True, "success_rate": 0.8}),
    ]

for i in range(1):
    map = generate_random_map(size=40, seed=None, p=0.8)
    #map = map_bad_for_AndOrSolver
    #map = map_DijkstraVsBFS_2
    for solver, solver_args, env_args in solvers:
        print(f"Running {solver.__name__} with {solver_args}...")
        success, steps, cost = run(solver, map=map, solver_args=solver_args, env_args=env_args)
        print(f"Success: {success}, steps: {steps}, cost: {cost}")
    print("--------------------------------------------------")
        