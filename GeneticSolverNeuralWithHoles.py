import numpy as np

from GeneticSolverNeural import GeneticSolverNeural, HOLE

class GeneticSolverNeuralWithHoles(GeneticSolverNeural):
    """GeneticSolverNeural but input includes the holes location too"""

    def get_input_size(self):
        # one-hot for current position + another for the holes
        return 2 * self.map_height * self.map_width

    def get_nn_input(self, position):
        # one-hot for current position + another for the holes
        positions = np.zeros(self.map_height * self.map_width)
        positions[position] = 1

        holes = np.array([1 if tile == HOLE else 0 for tile in self.map.flatten()])
        return np.concatenate((positions, holes))
