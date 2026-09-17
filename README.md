# Reinforcement Learning for Solar + Battery Energy Management

Complete project: intelligent agent that learns to control a battery in a PV + Load + Grid system.

## System

```
PV → Battery → Load
         ↓
        Grid
```

The agent decides at each step: **Charge / Idle / Discharge** (or continuous power).

## Features

- Custom Gymnasium-compatible environment
- Realistic components: PV profile, load profile, time-varying electricity price, battery efficiency & SOC limits
- Reward designed for low cost + high self-consumption + constraint satisfaction
- Baselines: Random & simple rule-based controller
- Tabular Q-Learning
- Deep Q-Network (DQN) with PyTorch
- PPO agent
- Evaluation metrics: electricity cost, peak grid demand, battery throughput, PV utilization, constraint violations
- Clean modular code

## Quick Start

```bash
git clone https://github.com/Helloworldceo/rl-solar-battery-energy-management.git
cd rl-solar-battery-energy-management
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run baselines + tabular Q-Learning demo
python -m src.main
```

## Project Structure

```
src/
├── env/
│   ├── energy_env.py          # Main environment
│   └── profiles.py            # PV, load, price profiles
├── agents/
│   ├── baseline.py
│   ├── q_learning.py
│   ├── dqn.py
│   └── ppo.py
├── training/
├── evaluation/
└── main.py
```

## MDP Formulation (summary)

**State:** `[SOC, PV, Load, Price, Hour]`  
**Actions:** 0 = Charge, 1 = Idle, 2 = Discharge  
**Reward:** `-cost + bonuses - penalties`

See `docs/` for full mathematical description.
