from abc import ABC, abstractmethod
from typing import Literal
import gymnasium


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
        self.gym_env.reset(seed=self.env_config["seed"])

    def evaluate(self, env_config, **solver_args) -> dict:
        metrics = {
            "success": False,
            "number_of_actions": 0,
            "total_reward": 0.0,
        }

        if not self.metrics["planning_succeeded"]:
            return {**metrics, **self.metrics}

        observation, _ = self.gym_env.reset(seed=env_config["seed"])

        for _ in range(env_config["max_episode_steps"]):
            try:
                observation, reward, terminated, truncated, info = self.gym_env.step(self.get_next_action())
            except EOFError as e:
                if str(e) != "No actions!":
                    print(env_config, solver_args)
                    raise
                self.gym_env.close() 
                metrics["success"]= False
                break
            except Exception as e:
                print(env_config, solver_args)
                raise
            
            metrics["number_of_actions"] += 1
            metrics["total_reward"]  += float(reward)

            if float(reward) > 0: #reached goal
                self.gym_env.close() 
                metrics["success"]= True
                break

            if terminated or float(reward) < 0 or truncated:
                self.gym_env.close()
                metrics["success"]= False
                break
        
        return {**metrics, **self.metrics}

    @abstractmethod
    def get_next_action(self) -> Literal[0, 1, 2, 3]:
        raise NotImplementedError
