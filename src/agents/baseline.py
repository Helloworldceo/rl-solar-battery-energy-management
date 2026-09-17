"""Simple baselines."""

import numpy as np


class RandomAgent:
    def __init__(self, n_actions: int = 3):
        self.n_actions = n_actions

    def act(self, obs):
        return np.random.randint(0, self.n_actions)


class RuleBasedAgent:
    """
    Simple heuristic:
    - If PV > Load and SOC not full → charge
    - If Load > PV and price high and SOC not empty → discharge
    - Else idle
    """
    def act(self, obs):
        soc, pv, load, price, hour = obs
        if pv > load + 0.2 and soc < 0.9:
            return 0  # charge
        if load > pv + 0.2 and price > 0.18 and soc > 0.2:
            return 2  # discharge
        return 1  # idle
