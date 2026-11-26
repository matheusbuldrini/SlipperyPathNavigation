from Solver import Solver
from typing import Literal, cast
import random

GOAL = b'G'
HOLE = b'H'


class HillClimbingSolver(Solver):
    # Hill climbing on a policy table (one move for every tile), for the slippery lake.
    # Unlike value iteration it does NOT use the slip probabilities:
    # it only plays games with a table and looks at the score.
    # Start from a random table, change the move on one tile, keep the change
    # if the score doesn't go down, undo it if it does. Repeat.

    def __init__(self, env_config, iterations=1000, training_games=30):
        super().__init__(env_config)
        self.iterations = iterations
        self.training_games = training_games
        self.random = random.Random(self.env_config["seed"])
        self.n_states = self.map.size
        self.max_distance = (self.map_height - 1) + (self.map_width - 1)

        # tiles where the move matters (on holes and on the goal the game is already over)
        self.free_tiles = [s for s in range(self.n_states) if self.map.flat[s] not in (GOAL, HOLE)]

        # we always score a table on the same games (same seeds), so two tables
        # are compared on the same slips. these seeds are different from the evaluation ones (1000+)
        self.training_seeds = [self.env_config["seed"] + i for i in range(self.training_games)]

        self.policy, self.training_score = self.search()

        self.metrics["planning_succeeded"] = True  # we always end with a table to play
        self.metrics["nodes_expanded"] = None
        self.metrics["training_score"] = self.training_score

    def get_next_action(self) -> Literal[0, 1, 2, 3]:
        # same as value iteration: look at the current tile and use the table
        return cast(Literal[0, 1, 2, 3], self.policy[self.gym_env.unwrapped.s])

    def random_policy(self):
        return [self.random.randint(0, 3) for _ in range(self.n_states)]

    def neighbor(self, policy):
        # small change: pick one tile and give it a different move
        new_policy = list(policy)
        tile = self.random.choice(self.free_tiles)
        new_policy[tile] = self.random.choice([a for a in range(4) if a != policy[tile]])
        return new_policy

    def score(self, policy):
        # play the training games with this table.
        # a win counts 1. a lost game still gets a small bonus for getting close to the goal,
        # otherwise at the start every random table scores 0 and we don't know which change helps
        env = self.gym_env.unwrapped  # without the wrappers, it's faster and we count the steps ourselves
        total = 0.0
        for seed in self.training_seeds:
            self.metrics["planning_simulation_count"] += 1
            state, _ = env.reset(seed=seed)
            won = False
            closest = self.distance_to_goal(state)
            for _ in range(self.env_config["max_episode_steps"]):
                state, reward, terminated, truncated, info = env.step(policy[state])
                closest = min(closest, self.distance_to_goal(state))
                if self.map.flat[state] == GOAL:
                    won = True
                    break
                if terminated:
                    break
            if won:
                total += 1.0
            else:
                # we use the closest the agent got, not where it ended:
                # a table that walks in circles near the goal is still better than one stuck at the start
                total += 0.1 * (1 - closest / self.max_distance)
        return total / len(self.training_seeds)

    def distance_to_goal(self, state):
        x, y = state % self.map_width, state // self.map_width
        return abs(x - self.goal_x) + abs(y - self.goal_y)

    # hill climbing: keep the change if the score doesn't go down
    # (equal is ok, so we can walk across flat parts instead of getting stuck there)
    def accept(self, new_score, current_score, iteration):
        return new_score >= current_score

    def search(self):
        current = self.random_policy()
        current_score = self.score(current)
        best, best_score = current, current_score

        for iteration in range(self.iterations):
            candidate = self.neighbor(current)
            candidate_score = self.score(candidate)
            if self.accept(candidate_score, current_score, iteration):
                current, current_score = candidate, candidate_score
                if current_score > best_score:
                    best, best_score = current, current_score
        return best, best_score
