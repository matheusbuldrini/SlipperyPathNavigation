from Solver import Solver
from typing import Literal, cast
import numpy as np

DEBUG = False

GOAL = b'G'
HOLE = b'H'

class AndOrSolver(Solver):
    
    map: np.ndarray
    map_height: int
    map_width: int
    actions: list[Literal[0, 1, 2, 3]]
    cost: int = -1

    def get_position(self):
        return self.gym_env.unwrapped.s

    def __init__(self, gym_env):
        super().__init__(gym_env)
        self.nodes = {}
        self.nodes_expanded = 0
        self.current_action_index = 0
        self.map = self.gym_env.unwrapped.desc
        self.map_height, self.map_width = self.map.shape

        if DEBUG: print(f"{self.__class__.__name__} initialized with map:")
        if DEBUG: print("map height:", self.map_height)
        if DEBUG: print("map width:", self.map_width)
        if DEBUG: print("initial position:", self.get_position())
        if DEBUG: print(self.map)
        if DEBUG: print("solving...")
        self.plan = self.and_or_search()
        self.current_plan = None
        if DEBUG: print("solution:")
        if DEBUG: print(self.plan)

    def get_next_action(self) -> Literal[0, 1, 2, 3]:
        if self.current_plan is None:
            self.current_plan = self.plan
            if self.current_plan is None:
                raise EOFError("No actions!")
        else:
            previous_action = list(self.current_plan.keys())[0]
            current_state = self.get_position()
            self.current_plan = self.current_plan[previous_action][current_state]

        action = list(self.current_plan.keys())[0]
        return cast(Literal[0, 1, 2, 3], action)

    def is_goal(self, state):
        x = state % self.map_width
        y = int(state / self.map_width)
        return self.map[y][x] == GOAL
    
    def is_hole(self, state):
        x = state % self.map_width
        y = int(state / self.map_width)
        return self.map[y][x] == HOLE

    def get_possible_outcomes(self, state, action):
        self.cost += 1
        outcomes = self.gym_env.unwrapped.P[state][action]
        return outcomes

    def is_cycle(self, state, path):
        return state in path

    def get_actions(self, state):
        return [2, 1, 0, 3] # [0,1,2,3] action order affects the number of recursive calls (gets stuck sometimes...)

    # plan = {action: {outcome_state: sub_plan}}

    def and_or_search(self):
        return self.or_search(self.get_position(), set())

    def or_search(self, state, path):
        if self.is_goal(state):
            return {}
        if self.is_cycle(state, path):
            return None
        if self.is_hole(state):
            return None
        for action in self.get_actions(state):
            result = self.get_possible_outcomes(state, action)
            path_plus_state = path.union({state})
            plan = self.and_search(result, path_plus_state)
            if plan is not None:
                return {action: plan}
        return None

    def and_search(self, states, path):
        plans = {}
        if not states:
            return None
        for prob, s, reward, terminated  in states:
            plan = self.or_search(s, path)
            if plan is None:
                return None
            plans[s] = plan
        return plans
