"""Learning curve and comparison plots."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curve(rewards_history: list[float], title: str, save_path: Path | None = None):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(rewards_history, alpha=0.4, label="Episode return")
    # Moving average
    window = max(1, len(rewards_history) // 20)
    if len(rewards_history) >= window:
        ma = np.convolve(rewards_history, np.ones(window)/window, mode="valid")
        ax.plot(range(window-1, len(rewards_history)), ma, lw=2, label=f"MA-{window}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.close(fig)
