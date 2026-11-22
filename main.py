from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from AStarSolver import AStarSolver
from AStarWeightedSolver import AStarWeightedSolver
from AndOrSolver import AndOrSolver
from BFSSolver import BFSSolver
from HumanSolver import HumanSolver
from DijkstraSolver import DijkstraSolver
from GeneticSolver import GeneticSolver
from GreedySolver import GreedySolver
from DFSSolver import DFSSolver
from IDSSolver import IDSSolver
import pprint
from datetime import datetime


RENDER = False

# This is the main file, that will run and collect all 
# metrics for all algorithms


# first, we need to generate all maps to be used. 
# This will be a fixed list for all agents 
MAPS_GENERATOR_SEEDS = [0] #, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
MAPS_SIZE = [4, 6] #, 8, 10]
MAPS_PROB_FROZEN_TILE = [0.6] #, 0.8, 0.9]


# generate all possible maps with the above configuration
MAPS_DICT = dict()

for seed in MAPS_GENERATOR_SEEDS:
    for size in MAPS_SIZE:
        for p in MAPS_PROB_FROZEN_TILE:
            map = generate_random_map(size,p,seed)
            MAPS_DICT[(size, p, seed)] = map

print(f"Generated {len(MAPS_DICT)} maps.")

# env config
SUCCESS_RATES = [0.8] #, 0.6, 0.3]  # only important if SLIPPERY=True
MAX_EPISODE_STEPS = [100] #, 200]

ENV_CONFIGS = []

# for each map, associate an env config
for map_name, map_desc in MAPS_DICT.items():
            # deterministic
            ENV_CONFIGS.append({
                "map_name": map_name,
                "desc": map_desc,
                "is_slippery": False,
                "success_rate": 1.0,
                "max_episode_steps": 200, # not used/important for the deterministic case
                "seed": 12345,
            })
            # slippery
            for max_episode_steps in MAX_EPISODE_STEPS:
                for success_rate in SUCCESS_RATES:
                    ENV_CONFIGS.append({
                        "map_name": map_name,
                        "desc": map_desc,
                        "is_slippery": True,
                        "success_rate": success_rate,
                        "max_episode_steps": max_episode_steps,
                        "seed": 12345,
                    })


print(f"Generated {len(ENV_CONFIGS)} configurations.")

solvers = [
    # (HumanSolver, {}),
    (BFSSolver, {}),
    (DFSSolver, {}),
    (IDSSolver, {}),
    (DijkstraSolver, {}),
    (AStarSolver, {}),
    (AStarWeightedSolver, {"weight": 1.5}),
    (AStarWeightedSolver, {"weight": 2.0}),
    (AStarWeightedSolver, {"weight": 2.5}),
    (AStarWeightedSolver, {"weight": 3.0}),
    (GreedySolver, {}),
    (AndOrSolver, {}),
    (GeneticSolver, {}),
    ]

def _get_solver_name(solver, args):
    if args:
        args_str = ", ".join(f"{key}={value}" for key, value in args.items())
        return f"{solver.__name__}({args_str})"

    return solver.__name__

RESULTS = []
for env_config in ENV_CONFIGS:
    for solver, solver_args in solvers:
        solver_name = _get_solver_name(solver, solver_args)
        print(f"Running {solver_name}")
        solver_instance = solver(env_config, **solver_args)
        metrics = solver_instance.evaluate(env_config)
        result={
            "solver": solver_name,
            "env_config": env_config,
            "metrics": metrics
        }
        # print(result)
        RESULTS.append(result)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"results_{timestamp}.txt"

with open(filename, "w", encoding="utf-8") as results_file:
    pprint.pprint(RESULTS, stream=results_file)

print(f"Results saved in {filename}")