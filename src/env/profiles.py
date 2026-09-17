"""Simple synthetic PV, load and price profiles for demonstration."""

import numpy as np


def generate_pv_profile(hours: int = 24, peak_power: float = 5.0, seed: int = 42) -> np.ndarray:
    """Generate a simple daily PV profile (kW)."""
    rng = np.random.default_rng(seed)
    t = np.arange(hours)
    # Simple bell-shaped PV around midday
    pv = peak_power * np.exp(-0.5 * ((t - 12) / 3.5) ** 2)
    pv = np.clip(pv + rng.normal(0, 0.15, hours), 0, None)
    return pv


def generate_load_profile(hours: int = 24, seed: int = 42) -> np.ndarray:
    """Residential-like load profile (kW)."""
    rng = np.random.default_rng(seed)
    base = np.array([
        0.8, 0.7, 0.6, 0.6, 0.7, 1.0, 1.5, 2.0,
        1.8, 1.5, 1.4, 1.6, 1.7, 1.5, 1.4, 1.6,
        2.2, 2.8, 3.0, 2.7, 2.2, 1.8, 1.4, 1.0
    ])
    load = base + rng.normal(0, 0.1, hours)
    return np.clip(load, 0.3, None)


def generate_price_profile(hours: int = 24) -> np.ndarray:
    """Time-of-use style price ($/kWh)."""
    price = np.ones(hours) * 0.12
    price[7:10] = 0.22   # morning peak
    price[17:21] = 0.28  # evening peak
    price[0:6] = 0.08    # night valley
    price[22:] = 0.09
    return price
