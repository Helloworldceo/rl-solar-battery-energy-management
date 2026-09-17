# Reward Design & Evaluation Metrics

## Reward design principles

A good reward should:

1. Reflect the true business objective (here: minimize electricity cost).
2. Softly encourage secondary goals (high PV self-consumption).
3. Penalize constraint violations (SOC limits) without making learning impossible.
4. Be dense enough that the agent receives useful feedback every step.

In this project:

```
reward = −step_cost
         + 0.01 * min(PV, Load)          # self-consumption bonus
         − 0.05 if SOC near limits       # soft constraint
```

You can (and should) experiment with different weightings.

## Evaluation metrics we report

| Metric | Why it matters |
|--------|----------------|
| Mean electricity cost | Primary economic objective |
| Peak grid import | Important for demand charges and grid friendliness |
| Battery throughput | Proxy for cycling / degradation |
| PV self-consumption ratio | How much solar is used locally instead of exported |

Always evaluate on held-out days (different seeds) and compare against strong baselines (rule-based heuristic, not only random).
