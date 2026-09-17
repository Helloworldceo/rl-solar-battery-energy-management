# MDP Formulation & Reward Design

## State

```
[SOC, PV_generation, Load, Electricity_price, Normalized_hour]
```

## Actions (discrete)

- 0: Charge at maximum rate
- 1: Idle
- 2: Discharge at maximum rate

## Transition

Deterministic given the profiles + battery dynamics with efficiency and SOC limits.

## Reward (intuition)

Primary objective: minimize electricity cost paid to the grid.

```
reward = -cost + small_self_consumption_bonus - SOC_boundary_penalty
```

This encourages:
- Buying from the grid when price is low (or not at all if PV/battery can cover)
- Using the battery to avoid expensive peak periods
- Staying away from hard SOC limits
- Preferring local PV consumption
