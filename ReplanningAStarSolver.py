from AStarSolver import AStarSolver
from typing import Literal, cast

DEBUG = False


class ReplanningAStarSolver(AStarSolver):
    # A* that looks where it is before every move (execution monitoring).
    # We keep the plan and remember which tile we expect to land on.
    # If we slipped and ended up somewhere else, we run A* again from the new tile.
    # It doesn't know the slip chances (no model), it just recovers after a slip.

    def __init__(self, env_config):
        # AStarSolver.__init__ already makes the first plan from the start tile
        super().__init__(env_config)
        self.expected_position = None  # no move done yet, nothing to check
        self.replans = 0
        self.metrics["number_of_replans"] = 0

    def get_next_action(self) -> Literal[0, 1, 2, 3]:
        position = self.get_position()

        # we are not where the plan said -> we slipped, plan again from here
        slipped = self.expected_position is not None and position != self.expected_position
        if slipped or len(self.actions) == 0:
            self.replan()

        if len(self.actions) == 0:
            raise EOFError("No actions!")  # no path to the goal from this tile

        action = self.actions.pop(0)
        self.expected_position = self.next_position(position, action)
        return cast(Literal[0, 1, 2, 3], action)

    def replan(self):
        solution = self.AStar()  # AStar() always starts from the current tile (get_position)
        if solution is None:
            self.actions = []
        else:
            self.actions = self.path_to_actions(self.get_path_to_node(solution))
        self.replans += 1
        if DEBUG: print(f"replan #{self.replans} from {self.get_position()}")
        # nodes_expanded keeps adding up, so this is the total for the whole game
        self.metrics["nodes_expanded"] = self.nodes_expanded
        self.metrics["number_of_replans"] = self.replans

    def next_position(self, position, action):
        # where we land if we don't slip (walking into the edge = we stay)
        x, y = position
        if action == 0: x = max(x - 1, 0)                    # left
        elif action == 1: y = min(y + 1, self.map_height - 1)  # down
        elif action == 2: x = min(x + 1, self.map_width - 1)   # right
        elif action == 3: y = max(y - 1, 0)                    # up
        return (x, y)
