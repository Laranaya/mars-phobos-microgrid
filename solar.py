"""Expected Mars solar output (percent of overhead clear-sky) using pvlib + SPICE geometry."""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
from pvlib.atmosphere import get_relative_airmass
from pvlib.irradiance import get_extra_radiation

from transit import get_sun_geometry

# Typical background dust optical depth at an equatorial settlement (order-of-magnitude).
MARS_TAU = 0.45
SOLAR_CONSTANT = 1366.1  # W/m^2 at 1 AU


def expected_solar_output_pct(when: datetime) -> float:
    """Believable 0-100% solar availability for the lat-0 / lon-0 site at `when` (UTC)."""
    geo = get_sun_geometry(when)
    zenith = geo["zenith_deg"]
    if zenith >= 90.0:
        return 0.0

    # pvlib I0 at 1 AU, then scale by the actual Mars-Sun range from SPICE.
    doy = when.timetuple().tm_yday
    i0 = float(get_extra_radiation(doy, solar_constant=SOLAR_CONSTANT))
    mars_s0 = i0 / (geo["r_au"] ** 2)

    am = get_relative_airmass(min(zenith, 87.0))
    am = 12.0 if am is None or np.any(np.isnan(am)) else float(np.asarray(am).reshape(-1)[0])
    am = float(np.clip(am, 1.0, 20.0))

    cos_z = max(0.0, math.cos(math.radians(zenith)))
    ghi = mars_s0 * cos_z * math.exp(-MARS_TAU * am)
    ghi_overhead = mars_s0 * math.exp(-MARS_TAU)  # zenith=0, AM=1
    pct = 100.0 * ghi / ghi_overhead if ghi_overhead > 0 else 0.0
    return float(np.clip(pct, 0.0, 100.0))
