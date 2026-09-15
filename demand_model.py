"""Synthetic settlement demand + RandomForestRegressor (no real Mars city data exists)."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor

_RNG = np.random.default_rng(42)
_N = 500


def _make_training_data(n: int = _N):
    hour = _RNG.integers(0, 24, size=n)
    # Greenhouses run more often in daylight, but not exclusively.
    daylight = ((hour >= 6) & (hour <= 20)).astype(int)
    greenhouse_p = 0.25 + 0.50 * daylight
    greenhouse = (_RNG.random(n) < greenhouse_p).astype(int)
    life_support = _RNG.uniform(44.0, 56.0, size=n)

    lighting = np.where((hour < 6) | (hour > 21), 10.0, 4.0)
    habitat_cycle = 12.0 * np.sin((hour - 7) / 24.0 * 2 * np.pi) ** 2
    gh_load = greenhouse * (24.0 + 0.35 * hour)
    noise = _RNG.normal(0.0, 2.2, size=n)
    demand = life_support + lighting + habitat_cycle + gh_load + noise
    demand = np.clip(demand, 35.0, 130.0)
    return hour, greenhouse, life_support, demand


_hour, _gh, _life, _demand = _make_training_data()
_X = np.column_stack([_hour, _gh, _life])
_model = RandomForestRegressor(
    n_estimators=80,
    max_depth=8,
    random_state=42,
    n_jobs=1,
)
_model.fit(_X, _demand)
_mean_life = float(np.mean(_life))


def predict_demand(hour_of_day: float, greenhouse_active: int | bool) -> float:
    """Predicted settlement demand in kW."""
    gh = 1 if greenhouse_active else 0
    hour = float(np.clip(hour_of_day, 0.0, 23.99))
    x = np.array([[hour, gh, _mean_life]])
    return float(_model.predict(x)[0])
