import spiceypy as spice

# Load all four kernel files
spice.furnsh("kernels/naif0012.tls")
spice.furnsh("kernels/pck00011.tpc")
spice.furnsh("kernels/de440s.bsp")
spice.furnsh("kernels/mar099s.bsp")

# Pick a date/time to check (year-month-day hour:minute:second, UTC)
utc_time = "2026-09-15T18:30:00"

# Convert that human-readable time into SPICE's internal time format
et = spice.str2et(utc_time)

# Get Phobos's position relative to Mars, as seen from the Sun's direction
# (this is a simplified check just to prove positions can be calculated)
pos, light_time = spice.spkpos("PHOBOS", et, "J2000", "NONE", "MARS")

print(f"At {utc_time}:")
print(f"Phobos position relative to Mars (km): {pos}")

spice.kclear()