import urllib.request
import os

os.makedirs("kernels", exist_ok=True)

files = {
 "naif0012.tls": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls",
 "pck00011.tpc": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00011.tpc",
 "de440s.bsp": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp",
 "mar099s.bsp": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/mar099s.bsp",
}

for name, url in files.items():
 path = os.path.join("kernels", name)
 print(f"Downloading {name}...")
 urllib.request.urlretrieve(url, path)
 print(f" done: {os.path.getsize(path)} bytes")

print("All kernels downloaded.")