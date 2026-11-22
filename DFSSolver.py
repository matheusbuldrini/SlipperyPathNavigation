from BFSSolver import BFSSolver


class DFSSolver(BFSSolver):
    """
    Depth-first search.

    Same as BFS, but the frontier is a stack (last in, first out) instead of
    a queue (first in, first out). The search goes as deep as it can along
    one branch, and only backs up when it reaches a dead end.

    - Complete on a finite map (we never add the same tile twice).
    - Not optimal: the path can be much longer than the BFS path.
    - Uses less memory than BFS: the stack holds one branch, not a whole ring.
    """

    def __init__(self, gym_env):
        # BFSSolver.__init__ reads the map and calls self.BFS().
        # We override BFS() below, so the depth-first search runs.
        super().__init__(gym_env)

    def BFS(self):
        return self.DFS()

    def DFS(self):
        self.nodes.clear()
        frontier = []  # a Python list used as a stack: append() pushes, pop() takes the last one
        reached = set()

        node = self.get_or_create_node(*self.get_position())
        if self.is_goal(node):
            return node
        frontier.append(node)
        reached.add(node)

        while len(frontier) > 0:
            node = frontier.pop()  # BFS uses popleft() here: that is the only real difference
            for child in self.expand(node):
                if self.is_goal(child):
                    return child
                if child not in reached:
                    reached.add(child)
                    frontier.append(child)
        return None
