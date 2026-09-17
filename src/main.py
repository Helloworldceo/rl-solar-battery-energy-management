"""Improved main: baselines + training with metrics and learning curves."""

from __future__ import annotations
from pathlib import Path
import numpy as np

from src.env.energy_env import EnergyManagementEnv
from src.agents.baseline import RandomAgent, RuleBasedAgent
from src.agents.q_learning import QLearningAgent
from src.agents.dqn import DQNAgent
from src.agents.ppo import PPOAgent
from src.utils.metrics import evaluate_agent, summarize
from src.utils.plotting import plot_learning_curve

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)


def train_q_learning(env, episodes: int = 600):
    agent = QLearningAgent()
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(obs, action, reward, next_obs, done)
            obs = next_obs
            ep_ret += reward
        agent.decay_epsilon()
        returns.append(ep_ret)
        if (ep + 1) % 100 == 0:
            print(f"  Q-Learning ep {ep+1:4d} | ε={agent.epsilon:.3f} | return={ep_ret:.2f}")
    plot_learning_curve(returns, "Q-Learning", RESULTS / "q_learning_curve.png")
    return agent


def train_dqn(env, episodes: int = 350):
    agent = DQNAgent()
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.remember(obs, action, reward, next_obs, float(done))
            agent.update()
            obs = next_obs
            ep_ret += reward
        agent.decay_epsilon()
        returns.append(ep_ret)
        if (ep + 1) % 50 == 0:
            print(f"  DQN ep {ep+1:4d} | ε={agent.epsilon:.3f} | return={ep_ret:.2f}")
    plot_learning_curve(returns, "DQN", RESULTS / "dqn_curve.png")
    return agent


def train_ppo(env, episodes: int = 250):
    agent = PPOAgent()
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            action, log_prob, value = agent.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.store(obs, action, log_prob, reward, done, value)
            obs = next_obs
            ep_ret += reward
        agent.update()
        returns.append(ep_ret)
        if (ep + 1) % 50 == 0:
            print(f"  PPO ep {ep+1:4d} | return={ep_ret:.2f}")
    plot_learning_curve(returns, "PPO", RESULTS / "ppo_curve.png")
    return agent


def main():
    env = EnergyManagementEnv(seed=42)

    print("=== Baselines ===")
    summarize(evaluate_agent(RandomAgent(), env, n_episodes=8), "Random")
    summarize(evaluate_agent(RuleBasedAgent(), env, n_episodes=8), "Rule-Based")

    print("\n=== Training Q-Learning ===")
    q_agent = train_q_learning(env)
    summarize(evaluate_agent(q_agent, env, n_episodes=8), "Q-Learning")

    print("\n=== Training DQN ===")
    dqn_agent = train_dqn(env)
    summarize(evaluate_agent(dqn_agent, env, n_episodes=8), "DQN")

    print("\n=== Training PPO ===")
    ppo_agent = train_ppo(env)
    summarize(evaluate_agent(ppo_agent, env, n_episodes=8), "PPO")

    print("\nAll done. Learning curves saved in results/")


if __name__ == "__main__":
    main()
