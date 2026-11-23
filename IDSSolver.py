from BFSSolver import BFSSolver
from Node import Node


class IDSSolver(BFSSolver):
    # Iterative deepening: DFS with a depth limit (fuel), and we keep giving it
    # more fuel (0, 1, 2, ...) until it finds the goal.
    # The first round that finds it gives the shortest path, same as BFS,
    # but memory stays small like DFS (only the current branch).

    # BFSSolver.__init__ calls self.BFS(), so we just swap the search here
    def BFS(self):
        return self.IDS()

    # the same tile can show up on many different routes, each with its own parent,
    # so here we always create a new node instead of reusing the cached one
    def get_or_create_node(self, x, y, parent=None):
        return Node(x, y, parent=parent)

    def IDS(self):
        start = Node(*self.get_position())
        max_fuel = self.map_width * self.map_height  # no path can be longer than this
        for fuel in range(max_fuel + 1):
            # new round = empty notebook
            # notebook[(x, y)] = most fuel we had when we got to that tile in this round
            self.notebook = {}
            result = self.depth_limited(start, fuel)
            if result is not None:
                self.depth_found = fuel
                return result
        return None

    def depth_limited(self, node, fuel):
        if self.is_goal(node):
            return node
        if fuel == 0:
            return None

        # been here before with the same or more fuel -> nothing new to find from here, skip.
        # this also stops us walking in circles (going back always means less fuel)
        tile = (node.x, node.y)
        if self.notebook.get(tile, -1) >= fuel:
            return None
        self.notebook[tile] = fuel
        self.nodes_expanded += 1

        for child in self.expand(node):
            result = self.depth_limited(child, fuel - 1)
            if result is not None:
                return result
        return None