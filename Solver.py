from abc import ABC, abstractmethod
from typing import Literal, Optional


class Solver(ABC):

    cost: int = -1
    last_action_reward: Optional[float] = None
    
    def __init__(self, gym_env):
        self.gym_env = gym_env

    @abstractmethod
    def get_next_action(self) -> Literal[0, 1, 2, 3]:
        raise NotImplementedError
