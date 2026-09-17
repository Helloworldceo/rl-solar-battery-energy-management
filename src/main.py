"""
Main script: run baselines, train Q-Learning, quick DQN & PPO demos, evaluate.
"""

from __future__ import annotations
import numpy as np
from pathlib import Path

from src.env.energy_env import EnergyManagementEnv
from src.agents.baseline import RandomAgent, RuleBasedAgent
from src.agents.q_learning import QLearningAgent
from src.agents.dqn import DQNAgent
from src.agents.ppo import PPOAgent


def evaluate(agent, env, n_episodes: int = 10, name: str = "Agent"):
    costs = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=100 + ep)
        done = False
        total_reward = 0.0
        while not done:
            if hasattr(agent, "act"):
                if isinstance(agent, PPOAgent):
                    action, _, _ = agent.act(obs)
                else:
                    action = agent.act(obs, explore=False) if hasattr(agent, "epsilon") else agent.act(obs)
            else:
                action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
        costs.append(env.total_cost)
    mean_cost = np.mean(costs)
    print(f"{name:20s} | Mean cost over {n_episodes} eps: {mean_cost:.3f} $")
    return mean_cost


def train_q_learning(env, episodes: int = 500):
    agent = QLearningAgent()
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(obs, action, reward, next_obs, done)
            obs = next_obs
        agent.decay_epsilon()
        if (ep + 1) % 100 == 0:
            print(f"Q-Learning episode {ep+1}, epsilon={agent.epsilon:.3f}")
    return agent


def train_dqn(env, episodes: int = 300):
    agent = DQNAgent()
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.remember(obs, action, reward, next_obs, done)
            agent.update()
            obs = next_obs
        agent.decay_epsilon()
        if (ep + 1) % 50 == 0:
            print(f"DQN episode {ep+1}, epsilon={agent.epsilon:.3f}")
    return agent


def train_ppo(env, episodes: int = 200):
    agent = PPOAgent()
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, log_prob, value = agent.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store(obs, action, log_prob, reward, done, value)
            obs = next_obs
        agent.update()
        if (ep + 1) % 50 == 0:
            print(f"PPO episode {ep+1}")
    return agent


def main():
    env = EnergyManagementEnv(seed=42)

    print("=== Baselines ===")
    evaluate(RandomAgent(), env, name="Random")
    evaluate(RuleBasedAgent(), env, name="Rule-Based")

    print("\n=== Training Q-Learning ===")
    q_agent = train_q_learning(env, episodes=400)
    evaluate(q_agent, env, name="Q-Learning")

    print("\n=== Training DQN (short demo) ===")
    dqn_agent = train_dqn(env, episodes=200)
    evaluate(dqn_agent, env, name="DQN")

    print("\n=== Training PPO (short demo) ===")
    ppo_agent = train_ppo(env, episodes=150)
    evaluate(ppo_agent, env, name="PPO")

    print("\nDone. You can increase episode counts for better performance.")


if __name__ == "__main__":
    main()
