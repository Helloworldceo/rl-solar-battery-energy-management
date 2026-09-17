"""
Solar + Battery + Load + Grid environment.
Compatible with the Gymnasium API.
"""

from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .profiles import generate_pv_profile, generate_load_profile, generate_price_profile


class EnergyManagementEnv(gym.Env):
    """
    Discrete-action energy management environment.

    Actions:
        0 = Charge at max rate
        1 = Idle
        2 = Discharge at max rate
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        battery_capacity: float = 10.0,      # kWh
        max_charge_rate: float = 3.0,        # kW
        max_discharge_rate: float = 3.0,     # kW
        eta_charge: float = 0.95,
        eta_discharge: float = 0.95,
        soc_min: float = 0.1,
        soc_max: float = 0.95,
        dt: float = 1.0,                     # hours
        episode_hours: int = 24,
        seed: int | None = 42,
    ):
        super().__init__()
        self.capacity = battery_capacity
        self.max_charge = max_charge_rate
        self.max_discharge = max_discharge_rate
        self.eta_c = eta_charge
        self.eta_d = eta_discharge
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.dt = dt
        self.episode_hours = episode_hours

        self.action_space = spaces.Discrete(3)
        # State: SOC, PV, Load, Price, Hour_normalized
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 10.0, 10.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.rng = np.random.default_rng(seed)
        self._generate_profiles()
        self.reset(seed=seed)

    def _generate_profiles(self):
        self.pv = generate_pv_profile(self.episode_hours, seed=int(self.rng.integers(1e6)))
        self.load = generate_load_profile(self.episode_hours, seed=int(self.rng.integers(1e6)))
        self.price = generate_price_profile(self.episode_hours)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self._generate_profiles()

        self.t = 0
        self.soc = 0.5
        self.total_cost = 0.0
        self.total_grid_import = 0.0
        self.total_pv_used = 0.0
        self.battery_throughput = 0.0

        return self._get_obs(), {}

    def _get_obs(self):
        # Safe index: when episode is over, return last valid observation
        t = min(self.t, self.episode_hours - 1)
        return np.array([
            self.soc,
            self.pv[t],
            self.load[t],
            self.price[t],
            t / self.episode_hours,
        ], dtype=np.float32)

    def step(self, action: int):
        t = min(self.t, self.episode_hours - 1)
        pv = self.pv[t]
        load = self.load[t]
        price = self.price[t]

        # Desired battery power (positive = charge)
        if action == 0:      # charge
            p_batt = self.max_charge
        elif action == 2:    # discharge
            p_batt = -self.max_discharge
        else:
            p_batt = 0.0

        # Respect SOC limits
        if p_batt > 0:  # charging
            max_possible = (self.soc_max - self.soc) * self.capacity / (self.eta_c * self.dt)
            p_batt = min(p_batt, max_possible)
            energy_to_batt = p_batt * self.dt * self.eta_c
        elif p_batt < 0:  # discharging
            max_possible = (self.soc - self.soc_min) * self.capacity / self.dt
            p_batt = max(p_batt, -max_possible)
            energy_to_batt = p_batt * self.dt / self.eta_d   # negative
        else:
            energy_to_batt = 0.0

        self.soc += energy_to_batt / self.capacity
        self.soc = float(np.clip(self.soc, self.soc_min, self.soc_max))
        self.battery_throughput += abs(energy_to_batt)

        # Power balance
        net = load + p_batt - pv          # positive = need from grid
        grid_import = max(net, 0.0)
        grid_export = max(-net, 0.0)

        cost = grid_import * price * self.dt
        revenue = grid_export * price * 0.3 * self.dt
        step_cost = cost - revenue

        self.total_cost += step_cost
        self.total_grid_import += grid_import * self.dt
        self.total_pv_used += min(pv, load + max(p_batt, 0)) * self.dt

        # Reward
        reward = -step_cost
        if self.soc <= self.soc_min + 0.02 or self.soc >= self.soc_max - 0.02:
            reward -= 0.05
        reward += 0.01 * min(pv, load)

        self.t += 1
        terminated = self.t >= self.episode_hours
        truncated = False

        info = {
            "cost": step_cost,
            "soc": self.soc,
            "grid_import": grid_import,
        }

        return self._get_obs(), float(reward), terminated, truncated, info
