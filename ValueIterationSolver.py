from Solver import Solver
from typing import Literal, cast
import numpy as np

GOAL = b'G'
HOLE = b'H'


class ValueIterationSolver(Solver):
    # Value iteration (MDP). Our benchmark for the slippery lake.
    # value[tile] = chance of reaching the goal from that tile if we play perfectly.
    # We keep updating all values until they stop changing, then on every tile
    # we pick the move with the best value -> that's the policy.
    # Needs the model (slip probabilities), we read it from env.unwrapped.P

    def __init__(self, env_config, gamma=0.99, theta=1e-4):
        super().__init__(env_config)
        self.gamma = gamma    # < 1 so reaching the goal sooner is worth a bit more
        self.theta = theta    # stop when the biggest change in a round is smaller than this
        self.map = self.gym_env.unwrapped.desc
        self.P = self.gym_env.unwrapped.P
        self.n_states = self.map.size
        self.start_state = self.gym_env.unwrapped.s
        self.rounds = 0
        self.nodes_expanded = 0   # here = number of tile updates, so we can compare with the search solvers

        self.values = self.value_iteration()
        self.policy = [self.best_action(s, self.values) for s in range(self.n_states)]

        # if the start value is 0 the goal can't be reached at all from the start
        self.metrics["planning_succeeded"] = bool(self.values[self.start_state] > 0)
        self.metrics["nodes_expanded"] = self.nodes_expanded
        if self.DEBUG: print(f"{self.__class__.__name__}: rounds: {self.rounds}, "
                             f"value of start tile: {self.values[self.start_state]:.3f}")

    def get_next_action(self) -> Literal[0, 1, 2, 3]:
        # no fixed path here: just look at where we are now and use the policy
        state = self.gym_env.unwrapped.s
        return cast(Literal[0, 1, 2, 3], self.policy[state])

    def is_terminal(self, state):
        tile = self.map.flat[state]
        return tile == GOAL or tile == HOLE

    def action_value(self, state, action, values):
        # average of where we can land: probability * (reward + gamma * value of next tile)
        total = 0.0
        for prob, next_state, reward, terminated in self.P[state][action]:
            if prob == 0:
                continue  # the env also lists impossible slips with prob 0, skip them
            # reward 1 only for the goal. we don't use the env reward here because
            # Solver gives -1 for holes, and we want values to stay a chance between 0 and 1
            reward = 1.0 if self.map.flat[next_state] == GOAL else 0.0
            if self.is_terminal(next_state):
                total += prob * reward  # game over, nothing comes after
            else:
                total += prob * (reward + self.gamma * values[next_state])
        return total

    def best_action(self, state, values):
        return max(range(4), key=lambda a: self.action_value(state, a, values))

    def value_iteration(self):
        values = np.zeros(self.n_states)  # goal and holes stay at 0, we never update them
        while True:
            self.rounds += 1
            new_values = values.copy()
            for s in range(self.n_states):
                if self.is_terminal(s):
                    continue
                self.nodes_expanded += 1
                new_values[s] = max(self.action_value(s, a, values) for a in range(4))
            biggest_change = np.max(np.abs(new_values - values))
            values = new_values
            if biggest_change < self.theta:
                return values
