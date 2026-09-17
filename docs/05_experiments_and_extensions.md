# Suggested Experiments & Extensions

## Experiments you can run immediately

1. Increase training episodes and observe how the learning curves improve.
2. Change the price profile (make the evening peak higher) and see whether the agent learns to discharge more aggressively.
3. Reduce battery capacity and observe the effect on cost and self-consumption.
4. Compare the final policies of Q-Learning vs DQN vs PPO on the same test days.
5. Disable the self-consumption bonus and see how the policy changes.

## Natural extensions

- Continuous action space (power set-point instead of three discrete actions).
- Multi-day episodes or continuing tasks.
- More realistic PV / load data (real measured profiles).
- Battery degradation model inside the reward.
- Multi-agent version (several houses sharing a community battery) → links to Project 3.
- Replace the simple DQN with Double DQN, Dueling DQN or Rainbow.
- Use a professional library (Stable-Baselines3) once you understand the algorithms from scratch.
