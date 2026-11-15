import gymnasium
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from typing import Callable
from AStarSolver import AStarSolver
from AStarWeightedSolver import AStarWeightedSolver
from AndOrSolver import AndOrSolver
from BFSSolver import BFSSolver
from HumanSolver import HumanSolver
from DijkstraSolver import DijkstraSolver
from GeneticSolver import GeneticSolver
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
        reward_schedule=(1, -1, 0), # goal, hole, normal
        max_episode_steps=100000
        )
    env.reset(seed=env_args.get("seed") or 0)
    env.unwrapped.metadata["render_fps"] = 40

    solver_instance = solver(env, **solver_args)

    env.reset(seed=env_args.get("seed") or 0)
    while True:
        if RENDER: env.render()
        observation, reward, terminated, truncated, info = env.step(solver_instance.get_next_action())
        solver_instance.last_action_reward = float(reward)
        steps += 1
        if float(reward) > 0:
            # print("reward received!")
            # print(f"Total steps taken: {steps}")
            env.close()
            return True, steps, solver_instance.cost

        if terminated or float(reward) < 0:
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
    (BFSSolver, {}, {"is_slippery": False}),
    (DijkstraSolver, {}, {"is_slippery": False}),
    (AStarSolver, {}, {"is_slippery": False}),
    (AStarWeightedSolver, {"weight": 1.5}, {"is_slippery": False}),
    (AStarWeightedSolver, {"weight": 2.0}, {"is_slippery": False}),
    (AStarWeightedSolver, {"weight": 2.5}, {"is_slippery": False}),
    (AStarWeightedSolver, {"weight": 3.0}, {"is_slippery": False}),
    (AndOrSolver, {}, {"is_slippery": False}),
    #(AndOrSolver, {}, {"is_slippery": True, "success_rate": 0.8}),
    (GeneticSolver, {}, {"is_slippery": True, "success_rate": 0.8, "seed":0}),
    ]

for i in range(1):
    map = generate_random_map(size=40, seed=0, p=0.8)
    #map = map_bad_for_AndOrSolver
    #map = map_DijkstraVsBFS_2
    for solver, solver_args, env_args in solvers:
        print(f"Running {solver.__name__} with {solver_args}...")
        success, steps, cost = run(solver, map=map, solver_args=solver_args, env_args=env_args)
        print(f"Success: {"YES" if success else "!!!FAILURE!!!"}, steps: {steps}, cost: {cost}")
    print("--------------------------------------------------")
        