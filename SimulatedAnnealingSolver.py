from HillClimbingSolver import HillClimbingSolver
import math


class SimulatedAnnealingSolver(HillClimbingSolver):
    # Same as hill climbing (same table, same small changes, same score),
    # but sometimes it also accepts a worse change, to get out of small hills.
    # How often depends on the temperature: hot at the start (accepts a lot),
    # cold at the end (almost never, so it ends like hill climbing).

    def __init__(self, env_config, iterations=1000, training_games=30,
                 start_temperature=0.02, end_temperature=0.0002):
        self.start_temperature = start_temperature
        self.end_temperature = end_temperature
        # we multiply the temperature by this every step, so it goes from start to end
        self.cooling = (end_temperature / start_temperature) ** (1 / iterations)
        super().__init__(env_config, iterations, training_games)

    def temperature(self, iteration):
        return self.start_temperature * (self.cooling ** iteration)

    def accept(self, new_score, current_score, iteration):
        if new_score >= current_score:
            return True
        # worse change: accept it with chance e^(difference / temperature)
        # small drop + hot = likely, big drop or cold = unlikely
        chance = math.exp((new_score - current_score) / self.temperature(iteration))
        return self.random.random() < chance
