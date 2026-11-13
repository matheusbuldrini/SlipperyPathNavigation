from Solver import Solver
from typing import Literal, cast

class HumanSolver(Solver):

    def __init__(self, gym_env):
        super().__init__(gym_env)
        print("HumanSolver initialized")

    def get_next_action(self) -> Literal[0, 1, 2, 3]:
        action = int(input("Enter next action (0=left, 1=down, 2=right, 3=up): "))
        while action < 0 or action > 3:
            print("Invalid action. Please enter a number between 0 and 3.")
            action = int(input("Enter next action (0=left, 1=down, 2=right, 3=up): "))
        return cast(Literal[0, 1, 2, 3], action)
