"""
Simplified PPO implementation for discrete actions.
Sufficient for demonstration and learning the concepts.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int = 5, n_actions: int = 3, hidden: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        features = self.shared(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value


class PPOAgent:
    def __init__(
        self,
        state_dim: int = 5,
        n_actions: int = 3,
        lr: float = 3e-4,
        gamma: float = 0.95,
        clip_epsilon: float = 0.2,
        epochs: int = 4,
        batch_size: int = 64,
    ):
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic(state_dim, n_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def act(self, obs):
        state = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.model(state)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return int(action.item()), log_prob.item(), value.item()

    def store(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def update(self):
        if len(self.states) == 0:
            return

        # Compute returns (simple Monte-Carlo)
        returns = []
        R = 0
        for r, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                R = 0
            R = r + self.gamma * R
            returns.insert(0, R)

        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        values = torch.FloatTensor(self.values).to(self.device)

        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.epochs):
            logits, new_values = self.model(states)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(new_values.squeeze(), returns)
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Clear buffer
        self.states, self.actions, self.log_probs = [], [], []
        self.rewards, self.dones, self.values = [], [], []
