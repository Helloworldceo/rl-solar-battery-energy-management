# Reinforcement Learning Fundamentals

## Core concepts

| Concept | Intuition |
|---------|-----------|
| **Agent** | The decision maker (our battery controller) |
| **Environment** | Everything the agent interacts with (PV, load, prices, battery physics) |
| **State** | What the agent observes at a given moment |
| **Action** | What the agent can do (Charge / Idle / Discharge) |
| **Reward** | Scalar feedback telling the agent how good the last action was |
| **Policy** | Mapping from states to actions (the “controller” the agent learns) |
| **Episode** | One complete trajectory (here usually one day) |
| **Return** | Sum (or discounted sum) of rewards over an episode |
| **Value function** | Expected return from a state (or state-action pair) |
| **Q-function** | Expected return of taking a specific action in a state and then following the policy |
| **Exploration vs Exploitation** | Trying new actions vs using what currently looks best |

## Why RL for energy management?

Classical control / optimization needs an accurate model and often a fixed objective.  
RL can learn a good policy directly from interaction when:

- The dynamics are complex or partially unknown
- Prices, PV and load are stochastic
- The objective is a long-term cumulative cost rather than a single-step error

In this project the agent learns when to charge and discharge the battery to minimize electricity cost while respecting SOC limits and preferring self-consumption of solar energy.
