from abc import ABC, abstractmethod
from typing import Literal


class Solver(ABC):
    def __init__(self, gym_env):
        self.gym_env = gym_env

    @abstractmethod
    def get_next_action(self) -> Literal[0, 1, 2, 3]:
        raise NotImplementedError
