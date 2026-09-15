"""Phobos solar-transit geometry from a fixed equatorial Mars site (lat 0, lon 0)."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import spiceypy as spice

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
KERNEL_FILES = (
    "naif0012.tls",
    "pck00011.tpc",
    "de440s.bsp",
    "mar099s.bsp",
)

SITE_LAT_DEG = 0.0
SITE_LON_DEG = 0.0
R_SUN_KM = 696_000.0
R_PHOBOS_KM = 11.1  # mean radius; Phobos is irregular
AU_KM = 149_597_870.7

# SPICE-confirmed daylight transit at this site (see get_transit_info).
TRANSIT_PEAK_UTC = datetime(2026, 9, 26, 12, 24, 26, tzinfo=timezone.utc)

_kernels_loaded = False
_observer_iau = None
_up_iau = None


def load_kernels() -> None:
    global _kernels_loaded, _observer_iau, _up_iau
    if _kernels_loaded:
        return
    for name in KERNEL_FILES:
        spice.furnsh(str(KERNEL_DIR / name))
    radii = spice.bodvrd("MARS", "RADII", 3)[1]
    re, rp = float(radii[0]), float(radii[2])
    flattening = (re - rp) / re
    _observer_iau = spice.georec(
        math.radians(SITE_LON_DEG),
        math.radians(SITE_LAT_DEG),
        0.0,
        re,
        flattening,
    )
    _up_iau = spice.vhat(_observer_iau)
    _kernels_loaded = True


def _to_et(when: datetime) -> float:
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return spice.datetime2et(when)


def _disk_overlap_area(radius_a: float, radius_b: float, separation: float) -> float:
    """Intersection area of two circles (angular radii, angular separation)."""
    if separation >= radius_a + radius_b:
        return 0.0
    if separation <= abs(radius_a - radius_b):
        return math.pi * min(radius_a, radius_b) ** 2
    a = math.acos(
        max(-1.0, min(1.0, (separation**2 + radius_a**2 - radius_b**2) / (2 * separation * radius_a)))
    )
    b = math.acos(
        max(-1.0, min(1.0, (separation**2 + radius_b**2 - radius_a**2) / (2 * separation * radius_b)))
    )
    k = max(
        0.0,
        (-separation + radius_a + radius_b)
        * (separation + radius_a - radius_b)
        * (separation - radius_a + radius_b)
        * (separation + radius_a + radius_b),
    )
    return radius_a**2 * a + radius_b**2 * b - 0.5 * math.sqrt(k)


def _vectors_at(et: float):
    load_kernels()
    sun_mars, _ = spice.spkpos("SUN", et, "IAU_MARS", "LT+S", "MARS")
    pho_mars, _ = spice.spkpos("PHOBOS", et, "IAU_MARS", "LT+S", "MARS")
    vec_sun = spice.vsub(sun_mars, _observer_iau)
    vec_pho = spice.vsub(pho_mars, _observer_iau)
    return vec_sun, vec_pho, sun_mars


def get_sun_geometry(when: datetime) -> dict:
    """Sun elevation, zenith angle, and Mars-Sun distance for the settlement site."""
    load_kernels()
    et = _to_et(when)
    vec_sun, _, sun_mars = _vectors_at(et)
    dist_sun = spice.vnorm(vec_sun)
    zenith = spice.vsep(vec_sun, _up_iau)
    elevation = math.pi / 2.0 - zenith
    r_au = spice.vnorm(sun_mars) / AU_KM
    return {
        "zenith_deg": math.degrees(zenith),
        "elevation_deg": math.degrees(elevation),
        "r_au": r_au,
        "distance_sun_km": dist_sun,
    }


def get_transit_info(when: datetime) -> tuple[bool, float]:
    """Return (is_transiting, percent of the solar disk blocked) at `when` (UTC)."""
    load_kernels()
    et = _to_et(when)
    vec_sun, vec_pho, _ = _vectors_at(et)
    dist_sun = spice.vnorm(vec_sun)
    dist_pho = spice.vnorm(vec_pho)
    if dist_sun <= R_SUN_KM or dist_pho <= R_PHOBOS_KM:
        return False, 0.0

    sep = spice.vsep(vec_sun, vec_pho)
    ang_sun = math.asin(min(1.0, R_SUN_KM / dist_sun))
    ang_pho = math.asin(min(1.0, R_PHOBOS_KM / dist_pho))
    overlapping = sep < (ang_sun + ang_pho)
    if not overlapping:
        return False, 0.0

    blocked_area = _disk_overlap_area(ang_sun, ang_pho, sep)
    sun_area = math.pi * ang_sun**2
    percent_blocked = 100.0 * blocked_area / sun_area if sun_area > 0 else 0.0
    return True, float(min(100.0, percent_blocked))
