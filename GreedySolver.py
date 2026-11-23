from AStarSolver import AStarSolver
import heapq


class GreedySolver(AStarSolver):
    """
    Greedy best-first search.

    Same as A*, but the frontier is sorted only by the heuristic h(n)
    (Manhattan distance to the goal). The cost so far g(n) is ignored
    when choosing the next node, so the search runs straight toward the goal.

    - Complete on a finite map (we never add the same tile twice).
    - Not optimal: the path can be longer or more costly than A*'s.

    The path cost is still computed (with the same tile costs as UCS and A*),
    so we can compare it with the other solvers in the experiments.
    """

    def __init__(self, gym_env):
        # AStarSolver.__init__ reads the map, finds the goal and calls
        # self.AStar(). We override AStar() below, so the greedy search runs.
        super().__init__(gym_env)

    def AStar(self):
        return self.greedy_best_first()

    def greedy_best_first(self):
        self.nodes.clear()
        frontier = []
        reached = set()
        counter = 0  # tie-breaker: when two nodes have the same h, take the older one

        start = self.get_or_create_node(*self.get_position(), parent=None, cost=0)
        if self.is_goal(start):
            return start

        heapq.heappush(frontier, (self.heuristic(start), counter, start))
        reached.add(start)

        while len(frontier) > 0:
            h, _, node = heapq.heappop(frontier)

            if self.is_goal(node):
                return node
            self.nodes_expanded += 1

            for child in self.expand(node):
                # Greedy adds each tile only once: the first time it sees it.
                # It never looks for a cheaper way to reach a tile.
                if child not in reached:
                    reached.add(child)
                    child.parent = node
                    child.cost = node.cost + self.get_tile_cost(child.x, child.y)
                    counter += 1
                    heapq.heappush(frontier, (self.heuristic(child), counter, child))
        return None
