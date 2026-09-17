"""Tabular Q-Learning with state discretization."""

from __future__ import annotations
import numpy as np
from collections import defaultdict


class QLearningAgent:
    def __init__(
        self,
        n_actions: int = 3,
        learning_rate: float = 0.1,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.q = defaultdict(lambda: np.zeros(n_actions))

    def _discretize(self, obs):
        soc, pv, load, price, hour = obs
        # Coarse bins
        s = int(soc * 10)
        p = int(pv * 2)
        l = int(load * 2)
        pr = int(price * 20)
        h = int(hour * 8)
        return (s, p, l, pr, h)

    def act(self, obs, explore: bool = True):
        state = self._discretize(obs)
        if explore and np.random.rand() < self.epsilon:
            return np.random.randint(0, self.n_actions)
        return int(np.argmax(self.q[state]))

    def update(self, obs, action, reward, next_obs, done):
        state = self._discretize(obs)
        next_state = self._discretize(next_obs)
        best_next = 0.0 if done else np.max(self.q[next_state])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q[state][action]
        self.q[state][action] += self.lr * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
