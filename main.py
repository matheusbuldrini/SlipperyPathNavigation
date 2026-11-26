from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from AStarSolver import AStarSolver
from AStarWeightedSolver import AStarWeightedSolver
from AndOrSolver import AndOrSolver
from BFSSolver import BFSSolver
from GeneticSolverNeural import GeneticSolverNeural
from GeneticSolverNeuralWithHoles import GeneticSolverNeuralWithHoles
from HumanSolver import HumanSolver
from DijkstraSolver import DijkstraSolver
from GeneticSolver import GeneticSolver
from GreedySolver import GreedySolver
from DFSSolver import DFSSolver
from IDSSolver import IDSSolver
from ValueIterationSolver import ValueIterationSolver
from ReplanningAStarSolver import ReplanningAStarSolver
from HillClimbingSolver import HillClimbingSolver
from SimulatedAnnealingSolver import SimulatedAnnealingSolver
from datetime import datetime
from time import perf_counter
from tqdm import tqdm
import json

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
MAX_EPISODE_STEPS = [200] #[50, 100, 200]

# how many games we play per config to measure the win rate
# (the normal lake gives the same result every time, so 1 game is enough there)
EVALUATION_GAMES = 1000
EVALUATION_SEED = 1000  # different from the planning seed (0)

ENV_CONFIGS = []

# for each map, associate an env config
for map_name, map_desc in MAPS_DICT.items():
            # deterministic
            ENV_CONFIGS.append({
                "map_name": map_name,
                "desc": map_desc,
                "is_slippery": False,
                "success_rate": 1.0,
                "max_episode_steps": max(MAX_EPISODE_STEPS), # not used/important for the deterministic case
                "seed": 0,
                "evaluation_games": 1,
                "evaluation_seed": EVALUATION_SEED,
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
                        "seed": 0,
                        "evaluation_games": EVALUATION_GAMES,
                        "evaluation_seed": EVALUATION_SEED,
                    })


print(f"Generated {len(ENV_CONFIGS)} configurations.")

solvers = [
    # (HumanSolver, {}),
    (BFSSolver, {}),
    (DFSSolver, {}),
    (IDSSolver, {}),
    (DijkstraSolver, {}),
    (AStarSolver, {}),
    (ReplanningAStarSolver, {}),
    (AStarWeightedSolver, {"weight": 1.5}),
    (AStarWeightedSolver, {"weight": 2.0}),
    (AStarWeightedSolver, {"weight": 2.5}),
    (AStarWeightedSolver, {"weight": 3.0}),
    (GreedySolver, {}),
    (AndOrSolver, {}),
    (GeneticSolver, {}),
    (GeneticSolverNeural, {"number_of_hidden_units": 100, "population_size": 200, "max_gens":100,
                           "mutation_rate":0.05,"mutation_strength":0.1,"random_gen_deviation":0.1,
                           "early_return":True}),
    (GeneticSolverNeuralWithHoles, {"number_of_hidden_units": 100, "population_size": 80, "max_gens":20,
                           "mutation_rate":0.5,"mutation_strength":0.1,"random_gen_deviation":10.0,
                           "early_return":True}),
    (ValueIterationSolver, {}),
    (HillClimbingSolver, {}),
    (SimulatedAnnealingSolver, {}),
    ]

def _get_solver_name(solver, args):
    if args:
        args_str = ", ".join(f"{key}={value}" for key, value in args.items())
        return f"{solver.__name__}({args_str})"

    return solver.__name__

RESULTS = []
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"results_{timestamp}.jsonl"
with open(filename, "a", encoding="utf-8") as results_file:
    with tqdm(total=len(ENV_CONFIGS) * len(solvers), desc="Progress") as progress_bar:
        for env_config in ENV_CONFIGS:
            for solver, solver_args in solvers:
                solver_name = _get_solver_name(solver, solver_args)
                progress_bar.desc = f"{solver_name} map={env_config["map_name"]}"
                progress_bar.update(0)
                planning_start = perf_counter()
                solver_instance = solver(env_config, **solver_args)
                solver_instance.metrics["planning_time_seconds"] = perf_counter() - planning_start
                metrics = solver_instance.evaluate(env_config)
                result={
                    "solver": solver_name,
                    "env_config": env_config,
                    "metrics": metrics
                }
                # print(result)
                RESULTS.append(result)
                results_file.write(json.dumps(result, ensure_ascii=False) + "\n")
                progress_bar.update(1)
            results_file.flush()

print(f"Results saved in {filename}")