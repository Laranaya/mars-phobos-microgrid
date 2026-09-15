"""Risk classifier: transit depth + demand → Normal / Soft Curtailment / Critical Buffer."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

_RNG = np.random.default_rng(7)
_N = 900
RISK_LABELS = {
    0: "Normal",
    1: "Soft Curtailment",
    2: "Critical Buffer",
}


def _label(percent_blocked: np.ndarray, demand_kw: np.ndarray) -> np.ndarray:
    """Rule used only to synthesize labels; the live decision is the trained forest."""
    risk = np.zeros(percent_blocked.shape, dtype=int)
    risk[(percent_blocked >= 6.0) | (demand_kw >= 88.0)] = 1
    critical = (percent_blocked >= 16.0) & (demand_kw >= 72.0)
    critical |= percent_blocked >= 24.0
    critical |= (percent_blocked >= 12.0) & (demand_kw >= 95.0)
    risk[critical] = 2
    return risk


_blocked = np.concatenate(
    [
        _RNG.uniform(0.0, 5.0, 300),
        _RNG.uniform(5.0, 18.0, 300),
        _RNG.uniform(18.0, 40.0, 300),
    ]
)
_demand = _RNG.uniform(40.0, 120.0, _N)
_y = _label(_blocked, _demand)
_X = np.column_stack([_blocked, _demand])

_model = RandomForestClassifier(
    n_estimators=120,
    max_depth=6,
    random_state=42,
    n_jobs=1,
    class_weight="balanced",
)
_model.fit(_X, _y)


def predict_risk(percent_blocked: float, predicted_demand_kw: float) -> tuple[int, float, dict]:
    """Return (risk_level, confidence, class_probabilities).

    Levels: 0 = Normal, 1 = Soft Curtailment (dim habitat lights),
    2 = Critical Buffer (activate batteries).
    """
    x = np.array([[float(percent_blocked), float(predicted_demand_kw)]])
    level = int(_model.predict(x)[0])
    proba = _model.predict_proba(x)[0]
    classes = list(_model.classes_)
    mapping = {int(c): float(p) for c, p in zip(classes, proba)}
    for k in (0, 1, 2):
        mapping.setdefault(k, 0.0)
    confidence = float(mapping[level])
    return level, confidence, mapping
