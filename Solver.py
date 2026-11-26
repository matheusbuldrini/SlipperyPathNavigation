from abc import ABC, abstractmethod
from typing import Literal
import gymnasium
import numpy as np

GOAL = b'G'

class Solver(ABC):
    DEBUG = False

    def __init__(self, env_config):
        self.env_config = env_config

        self.metrics = {
            "planning_succeeded": False,
            "planning_time_seconds": 0.0,
            "nodes_expanded": None,
            "planning_simulation_count": 0,
        }
        self.gym_env = gymnasium.make(
            'FrozenLake-v1',
            reward_schedule=(1, -1, 0), # goal, hole, normal
            map_name= env_config["map_name"],
            desc= env_config["desc"],
            is_slippery= env_config["is_slippery"],
            success_rate= env_config["success_rate"],
            max_episode_steps=env_config["max_episode_steps"],
        )

        self.map = self.gym_env.unwrapped.desc
        self.map_height, self.map_width = self.map.shape
        goal_y, goal_x = np.where(self.map == GOAL)
        self.goal_x = goal_x[0]
        self.goal_y = goal_y[0]

        self.gym_env.reset(seed=self.env_config["seed"])

    # plays several games with the same solver (we plan only once, in __init__)
    # and returns the averages. On the slippery lake one game is just luck,
    # so we use env_config["evaluation_games"] games, each with its own seed.
    # The seeds start at env_config["evaluation_seed"], different from the
    # planning seed, so a solver can't win just by memorizing the slips it trained on.
    def evaluate(self, env_config, **solver_args) -> dict:
        number_of_games = env_config.get("evaluation_games", 1)
        first_seed = env_config.get("evaluation_seed", env_config["seed"])

        # the search solvers empty their plan while playing (pop), so we keep a copy
        # of the original plan and give it back at the start of every game
        self.initial_actions = list(self.actions) if hasattr(self, "actions") else None

        games = []
        for i in range(number_of_games):
            self.reset_for_new_game()
            games.append(self.play_one_game(first_seed + i, env_config, solver_args))
        self.gym_env.close()
        self.after_all_games(number_of_games)

        wins = [g for g in games if g["success"]]
        metrics = {
            "games": number_of_games,
            "wins": len(wins),
            "success_rate": len(wins) / number_of_games,
            "avg_number_of_actions": float(np.mean([g["number_of_actions"] for g in games])),
            # how long the winning games took (None if it never won)
            "avg_number_of_actions_when_won": float(np.mean([g["number_of_actions"] for g in wins])) if wins else None,
            "avg_total_reward": float(np.mean([g["total_reward"] for g in games])),
            "avg_distance_to_goal": float(np.mean([g["distance_to_goal"] for g in games])),
        }
        return {**metrics, **self.metrics}

    # one game, same logic as before
    def play_one_game(self, seed, env_config, solver_args) -> dict:
        metrics = {
            "success": False,
            "number_of_actions": 0,
            "total_reward": 0.0,
            "distance_to_goal": -1
        }

        observation, _ = self.gym_env.reset(seed=seed)

        for _ in range(env_config["max_episode_steps"]):
            try:
                observation, reward, terminated, truncated, info = self.gym_env.step(self.get_next_action())
            except EOFError as e:
                if str(e) != "No actions!":
                    print(env_config, solver_args)
                    raise
                metrics["success"]= False
                break
            except Exception as e:
                print(env_config, solver_args)
                raise

            metrics["number_of_actions"] += 1
            metrics["total_reward"]  += float(reward)

            if float(reward) > 0: #reached goal
                metrics["success"]= True
                break

            if terminated or float(reward) < 0 or truncated:
                metrics["success"]= False
                break

        x = int(observation % self.map_width)
        y = int(observation / self.map_width)
        dist = abs(x - self.goal_x) + abs(y - self.goal_y)

        metrics["distance_to_goal"] = int(dist)
        return metrics

    # called before every game. by default: give back the original plan (search solvers)
    # and forget the AND-OR plan position. a solver can override this if it needs more
    def reset_for_new_game(self):
        if self.initial_actions is not None:
            self.actions = list(self.initial_actions)
        if hasattr(self, "current_plan"):
            self.current_plan = None

    # called once after all the games, if a solver wants to turn totals into averages
    def after_all_games(self, number_of_games):
        pass

    @abstractmethod
    def get_next_action(self) -> Literal[0, 1, 2, 3]:
        raise NotImplementedError