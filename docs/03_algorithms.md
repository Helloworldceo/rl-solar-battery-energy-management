# Algorithms Used in This Project

## 1. Tabular Q-Learning

We maintain a table Q(s, a) that estimates the expected return of taking action a in state s.

**Update rule (Bellman):**

```
Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') − Q(s,a) ]
```

- α = learning rate
- γ = discount factor
- The term in brackets is the temporal-difference (TD) error

Because the true state is continuous we **discretize** it into bins. This works only for low-dimensional state spaces.

**Exploration**: ε-greedy (with probability ε take a random action).

## 2. Deep Q-Network (DQN)

When the state space is large we approximate Q(s, a) with a neural network.

Key ideas that stabilize learning:

- **Experience replay**: store transitions in a buffer and sample random mini-batches (breaks temporal correlation).
- **Target network**: a slowly updated copy of the Q-network used to compute the TD target (reduces moving-target problem).

Loss:

```
L = ( r + γ max_a' Q_target(s', a') − Q_policy(s, a) )²
```

## 3. PPO (Proximal Policy Optimization)

Value-based methods (Q-Learning / DQN) learn an action-value function and derive a policy from it.  
Policy-gradient methods directly optimize the policy parameters.

PPO is a stable policy-gradient algorithm that:

- Collects trajectories with the current policy
- Computes advantages (how much better an action was than expected)
- Updates the policy with a clipped objective so that the new policy does not deviate too far from the old one

It is often more sample-efficient and stable than vanilla policy gradient, and handles both discrete and continuous actions well.
