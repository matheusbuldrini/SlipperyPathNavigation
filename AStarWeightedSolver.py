from AStarSolver import AStarSolver

class AStarWeightedSolver(AStarSolver):
    
    WEIGHT = 1.0

    def __init__(self, gym_env, weight=1.0):
        self.WEIGHT = weight
        super().__init__(gym_env)

    def heuristic(self, node):
        return self.WEIGHT * super().heuristic(node)
