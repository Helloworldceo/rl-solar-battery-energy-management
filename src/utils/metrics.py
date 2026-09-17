"""Evaluation metrics for energy management agents."""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class EpisodeStats:
    cost: float
    peak_import: float
    throughput: float
    self_consumption: float
    final_soc: float


def evaluate_agent(agent, env, n_episodes: int = 10, explore: bool = False) -> list[EpisodeStats]:
    stats = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=200 + ep)
        done = False
        peak = 0.0
        while not done:
            if hasattr(agent, "act"):
                if type(agent).__name__ == "PPOAgent":
                    action, _, _ = agent.act(obs)
                else:
                    try:
                        action = agent.act(obs, explore=explore)
                    except TypeError:
                        action = agent.act(obs)
            else:
                action = 1
            obs, reward, terminated, truncated, info = env.step(action)
            peak = max(peak, info.get("grid_import", 0.0))
            done = terminated or truncated

        sc = env.total_pv_used / (np.sum(env.pv) * env.dt + 1e-6)
        stats.append(EpisodeStats(
            cost=env.total_cost,
            peak_import=peak,
            throughput=env.battery_throughput,
            self_consumption=float(sc),
            final_soc=env.soc,
        ))
    return stats


def summarize(stats: list[EpisodeStats], name: str = ""):
    costs = [s.cost for s in stats]
    peaks = [s.peak_import for s in stats]
    thrs = [s.throughput for s in stats]
    scs = [s.self_consumption for s in stats]
    print(f"{name:18s} | Cost {np.mean(costs):6.3f}±{np.std(costs):.3f} $ | "
          f"Peak {np.mean(peaks):5.2f} kW | Throughput {np.mean(thrs):5.2f} kWh | "
          f"Self-cons {np.mean(scs)*100:5.1f}%")
    return {
        "name": name,
        "mean_cost": float(np.mean(costs)),
        "std_cost": float(np.std(costs)),
        "mean_peak": float(np.mean(peaks)),
        "mean_throughput": float(np.mean(thrs)),
        "mean_self_consumption": float(np.mean(scs)),
    }
