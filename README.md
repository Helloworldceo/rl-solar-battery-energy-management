# Reinforcement Learning for Solar + Battery Energy Management

Intelligent agent that learns optimal charge / idle / discharge decisions for a home battery paired with solar PV.

## System

```
PV generation → Battery → Household Load
                    ↓
                   Grid (buy/sell)
```

## Features

- Gymnasium-compatible environment with SOC limits, efficiencies, time-of-use prices
- Synthetic but realistic PV / load / price profiles
- Baselines: Random + Rule-based heuristic
- Tabular Q-Learning with state discretization
- DQN (PyTorch) with experience replay + target network
- PPO (Actor-Critic)
- Rich evaluation: cost, peak import, battery throughput, self-consumption
- Learning-curve plots and model checkpointing
- Configurable hyperparameters

## Quick Start

```bash
git clone https://github.com/Helloworldceo/rl-solar-battery-energy-management.git
cd rl-solar-battery-energy-management
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.main
```

## Key Metrics Reported

- Mean electricity cost ($)
- Peak grid import (kW)
- Battery throughput (kWh) – proxy for cycling
- PV self-consumption ratio

## Structure

```
src/
├── env/           # EnergyManagementEnv + profiles
├── agents/        # baselines, Q-Learning, DQN, PPO
├── utils/         # metrics, plotting, checkpointing
└── main.py
```
