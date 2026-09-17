# MDP Formulation

## Markov Decision Process

An MDP is defined by the tuple (S, A, P, R, γ):

- **S** – state space
- **A** – action space
- **P** – transition probabilities P(s′ | s, a)
- **R** – reward function R(s, a, s′)
- **γ** – discount factor (0 ≤ γ ≤ 1)

The **Markov property** says that the future depends only on the current state and action, not on the full history.

## Our concrete MDP

**State** (continuous, later discretized for tabular Q-Learning):

```
[SOC, PV_generation, Load, Electricity_price, Normalized_hour]
```

**Actions** (discrete):

- 0 = Charge at maximum rate
- 1 = Idle
- 2 = Discharge at maximum rate

**Transition**: deterministic given the current profiles + battery dynamics (efficiency, SOC limits).

**Reward** (intuition):

```
reward = − electricity_cost
         + small bonus for self-consumption of PV
         − penalty when SOC approaches hard limits
```

This encourages low bill, high local use of solar, and safe battery operation.

**Discount factor γ**: typically 0.95–0.99. It balances immediate cost versus future cost (e.g. saving battery energy for the expensive evening peak).
