from Solver import Solver
from typing import Literal, cast
import numpy as np

GOAL = b'G'
HOLE = b'H'


class ValueIterationSolver(Solver):
    # Value iteration (MDP). Our benchmark for the slippery lake.
    # Needs the model (slip probabilities), we read it from env.unwrapped.P
    #
    # The game stops after max_episode_steps, so "best policy" means: the best chance
    # to reach the goal within that many steps. So we do it with a horizon:
    #   win[t][tile] = best chance to reach the goal from this tile with t steps left
    # and the best move can change with the steps left (with few steps left it takes more risk).
    # This gives the exact best possible win rate, so no solver can beat it on average.
    #
    # We also keep the old discounted values (gamma) only to break ties:
    # when two moves have the same chance, we take the one that reaches the goal sooner.

    def __init__(self, env_config, gamma=0.99, theta=1e-4):
        super().__init__(env_config)
        self.gamma = gamma
        self.theta = theta
        self.P = self.gym_env.unwrapped.P
        self.n_states = self.map.size
        self.start_state = self.gym_env.unwrapped.s
        self.horizon = self.env_config["max_episode_steps"]
        self.nodes_expanded = 0   # here = number of tile updates, so we can compare with the search solvers
        self.steps_taken = 0

        self.values = self.discounted_values()        # only for tie-breaking
        self.policy_by_steps_left = self.finite_horizon()

        # exact chance to win from the start (if it's 0 the goal can't be reached in time)
        self.metrics["planning_succeeded"] = bool(self.win_chance[self.start_state] > 0)
        self.metrics["exact_win_probability"] = float(self.win_chance[self.start_state])
        self.metrics["nodes_expanded"] = self.nodes_expanded

    def get_next_action(self) -> Literal[0, 1, 2, 3]:
        # look at the current tile and how many steps are left, and use the table for that
        steps_left = max(self.horizon - self.steps_taken, 1)
        self.steps_taken += 1
        state = self.gym_env.unwrapped.s
        return cast(Literal[0, 1, 2, 3], self.policy_by_steps_left[steps_left][state])

    def reset_for_new_game(self):
        super().reset_for_new_game()
        self.steps_taken = 0

    def is_terminal(self, state):
        tile = self.map.flat[state]
        return tile == GOAL or tile == HOLE

    def outcomes(self, state, action):
        # the env also lists impossible slips with prob 0, skip them
        return [(prob, next_state) for prob, next_state, _, _ in self.P[state][action] if prob > 0]

    def finite_horizon(self):
        # win with 0 steps left = 0 everywhere. then add one step at a time:
        # win[t][s] = best over moves of: sum of prob * (1 if goal, 0 if hole, win[t-1][next] otherwise)
        win = np.zeros(self.n_states)
        policy_by_steps_left = [None] * (self.horizon + 1)
        for steps_left in range(1, self.horizon + 1):
            new_win = np.zeros(self.n_states)
            policy = [0] * self.n_states
            for s in range(self.n_states):
                if self.is_terminal(s):
                    continue
                self.nodes_expanded += 1
                chances = []
                for a in range(4):
                    chance = 0.0
                    for prob, next_state in self.outcomes(s, a):
                        if self.map.flat[next_state] == GOAL:
                            chance += prob
                        elif not self.is_terminal(next_state):
                            chance += prob * win[next_state]
                    chances.append(chance)
                best = max(chances)
                # moves that are as good as the best one -> take the fastest one (discounted value)
                ties = [a for a in range(4) if chances[a] >= best - 1e-12]
                policy[s] = max(ties, key=lambda a: self.discounted_action_value(s, a, self.values))
                new_win[s] = best
            win = new_win
            policy_by_steps_left[steps_left] = policy
        self.win_chance = win  # with the full number of steps
        return policy_by_steps_left

    def discounted_action_value(self, state, action, values):
        # probability * (1 if goal + gamma * value of next tile)
        total = 0.0
        for prob, next_state in self.outcomes(state, action):
            if self.map.flat[next_state] == GOAL:
                total += prob
            elif not self.is_terminal(next_state):
                total += prob * self.gamma * values[next_state]
        return total

    def discounted_values(self):
        # the classic value iteration with gamma, repeat until the values stop changing
        values = np.zeros(self.n_states)
        while True:
            new_values = values.copy()
            for s in range(self.n_states):
                if self.is_terminal(s):
                    continue
                self.nodes_expanded += 1
                new_values[s] = max(self.discounted_action_value(s, a, values) for a in range(4))
            biggest_change = np.max(np.abs(new_values - values))
            values = new_values
            if biggest_change < self.theta:
                return values