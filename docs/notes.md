# Step-by-step: Intel RealSense D435i on Raspberry Pi 5

## Hardware

- Raspberry Pi 5 (8GB RAM)
- Intel RealSense D435i
- Debian GNU/Linux 12 (Bookworm), aarch64
- Python 3.11.2
- Docker 29.3.0

---

## Step 1 — Verify hardware detection

Plug the D435i into the **USB 3.0 blue port** on the RPi5.

```bash
lsusb | grep -i intel
```

Expected output:
```
Bus 004 Device 002: ID 8086:0b3a Intel Corp. Intel(R) RealSense(TM) Depth Camera 435i
```

`Bus 004` confirms USB 3.0. ✓

---

## Step 2 — Install udev rules

Required so the camera can be accessed without sudo.

```bash
wget https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules
sudo cp 99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
rm 99-realsense-libusb.rules
```

Verify:
```bash
ls /etc/udev/rules.d/ | grep realsense
# expected: 99-realsense-libusb.rules
```

✓

---

## Step 3 — Create Python virtual environment

Debian 12 blocks system-wide pip installs (PEP 668). Use a venv.

```bash
python3 -m venv ~/realsense-env
source ~/realsense-env/bin/activate
pip install --upgrade pip
```

✓

---

## Step 4 — Install pyrealsense2

### What was tried and why it failed

**A) `pip install pyrealsense2`** — FAILED
```
ERROR: Could not find a version that satisfies the requirement pyrealsense2 (from versions: none)
```
Intel publishes no ARM64 wheel on PyPI. piwheels.org also has none.

**B) Intel apt repo (`jammy` on Debian Bookworm)** — FAILED

The `jammy` repo uses a different GPG key (`FB0B24895113F120`) than the one
Intel provides at `librealsense.intel.com/Debian/librealsense.pgp`.
Fetching from keyserver also failed (`dirmngr` not installed on this system).

### What works: build from source

Install build dependencies (minimal — no GUI/OpenGL needed):

```bash
sudo apt-get install -y \
    cmake build-essential \
    libssl-dev libusb-1.0-0-dev \
    pkg-config python3-dev
```

Clone and configure:

```bash
git clone --depth 1 https://github.com/IntelRealSense/librealsense.git ~/librealsense-src
cd ~/librealsense-src && mkdir build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DFORCE_RSUSB_BACKEND=ON \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DPYTHON_EXECUTABLE=$HOME/realsense-env/bin/python3 \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_GRAPHICAL_EXAMPLES=OFF
```

Key flags:
- `-DFORCE_RSUSB_BACKEND=ON` — uses libusb instead of kernel module (no kernel patching needed on Debian)
- `-DBUILD_PYTHON_BINDINGS=ON` — builds the `pyrealsense2` Python module
- `-DPYTHON_EXECUTABLE=...` — points to the venv so bindings install there
- `BUILD_EXAMPLES=OFF` + `BUILD_GRAPHICAL_EXAMPLES=OFF` — skips OpenGL/GTK, much faster build

Build and install:

```bash
make -j$(nproc)
sudo make install
sudo ldconfig
```

Build time on RPi5 (4 cores, 8GB RAM): ~40 min  
Started: 2026-06-22 ~19:20 — completed ~20:00

Build output (relevant lines):
```
[100%] Linking CXX shared library ../../Release/pyrealsense2.cpython-311-aarch64-linux-gnu.so
[100%] Built target pyrealsense2
-- Installing: /home/user/realsense-env/lib/python3.11/site-packages/pyrealsense2/pyrealsense2.cpython-311-aarch64-linux-gnu.so
-- Installing: /home/user/realsense-env/lib/python3.11/site-packages/pyrealsense2/__init__.py
```

The Python bindings install directly to the venv's site-packages because
`-DPYTHON_EXECUTABLE=$HOME/realsense-env/bin/python3` was specified in cmake.

Compiler warnings (`-Wstringop-overflow=`) from `hdr-config.cpp` are cosmetic —
known issue with GCC 12 and this file. Build succeeds regardless. ✓

---

## Step 5 — Verify pyrealsense2

```bash
cd ~/rpi5-realsense
source ~/realsense-env/bin/activate
python -c "import pyrealsense2; print(pyrealsense2.__version__)"
python setup/verify.py
```

Output:
```
2.58.2
Found 1 device(s):
  Name:     Intel RealSense D435I
  Serial:   233522078548
  Firmware: 5.17.0.10

Testing pipeline (5 frames)...
  Frame 1: OK
  Frame 2: OK
  Frame 3: OK
  Frame 4: OK
  Frame 5: OK

RealSense OK
```

✓

---

## Step 6 — Run green ball example

### OpenCV display troubleshooting

**Problem A**: `opencv-python` (pip) uses Qt — window fails on Debian Bookworm Wayland:
```
qt.qpa.plugin: Could not find the Qt platform plugin "wayland"
QFontDatabase: Cannot find font directory .../cv2/qt/fonts
```

**Fix**: Use system OpenCV (GTK-based) instead of pip opencv-python:
```bash
sudo apt-get install -y python3-opencv
pip uninstall opencv-python -y
echo "/usr/lib/python3/dist-packages" \
    > ~/realsense-env/lib/python3.11/site-packages/system.pth
```

**Problem B**: NumPy version conflict — system OpenCV 4.6.0 compiled with NumPy 1.x,
venv had NumPy 2.4.6:
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.4.6
```

**Fix**: Downgrade NumPy in the venv:
```bash
pip install "numpy<2"
```

Note: pyrealsense2 uses pybind11 2.13.6 which supports both NumPy 1.x and 2.x — no issue.

**Warning** `libpng warning: iCCP: known incorrect sRGB profile` is cosmetic, harmless.

### Running the example

```bash
cd ~/rpi5-realsense
source ~/realsense-env/bin/activate
python examples/green_ball.py
```

Three windows open: **RGB**, **Depth** (JET colormap), **Mask**.

**To track an object:**
1. Point the camera at the object
2. **Click on it** in the RGB window
3. The script samples a 20×20 pixel region around the click, computes the
   mean HSV value, and sets a ±tolerance range automatically
4. The mask window shows what is being detected in white

Terminal output on click:
```
Clicked (423,201) — HSV mean: H=52 S=88 V=61
  Range: lower=[37 28  1]  upper=[67 148 121]
```

When an object is detected:
```
  Object (423,201)  depth=0.82 m
```

Press `q` to quit cleanly. ✓

### Known issue: filter breaks get_distance()

After applying `spatial.process()` or `temporal.process()`, the returned object
is a generic `frame` — `get_distance()` is not available on it.

Fix: cast back to `depth_frame` explicitly:
```python
depth_frame = spatial.process(depth_frame).as_depth_frame()
depth_frame = temporal.process(depth_frame).as_depth_frame()
```

### Calibrated HSV presets (measured on RPi5, June 2026)

| Preset | Key | H mean | S mean | V mean | Range lower | Range upper |
|--------|-----|--------|--------|--------|-------------|-------------|
| verde musgo | `1` | 82 | 92 | 29 | [62, 10, 0] | [102, 210, 130] |
| verde folha | `2` | 61 | 141 | 58 | [41, 35, 0] | [86, 255, 180] |

**Distance test results** (indoor, low light):
- verde folha: tracked reliably up to **3m+**, object became very small but mask remained solid white
- verde musgo: harder to track at distance — darker color (V=29) loses contrast faster under poor lighting

**Recommendation**: use verde folha or brighter objects for longer-range demos.
For verde musgo, better room lighting significantly improves range.

### Color tuning tips

For other colors, click on the object in the RGB window.

If the mask is noisy (detecting too much):
- Click again on a better-lit area of the object
- Or increase the erosion iterations in the code

If the mask is empty (detecting nothing):
- The object may be too dark or too saturated for the tolerance window
- Increase `tol` values in `on_mouse_click()` (default: `[15, 80, 80]`)

### Final working stack

| Component | Version | Source |
|-----------|---------|--------|
| Python | 3.11.2 | system |
| pyrealsense2 | 2.58.2 | built from source (librealsense) |
| opencv | 4.6.0 | system apt (`python3-opencv`, GTK) |
| numpy | 1.26.4 | pip (downgraded for opencv compat) |
| pybind11 | 2.13.6 | fetched by cmake during build |

---

## Step 7 — Install additional dependencies (corridor tracker)

```bash
source ~/realsense-env/bin/activate
pip install pyyaml matplotlib
```

---

## Step 8 — Generate and print ArUco markers

Run on any machine that has OpenCV (PC or RPi5):

```bash
# PC: pip install opencv-contrib-python (if needed)
# RPi5: system opencv already includes aruco
python tools/generate_aruco.py
```

Output: `tools/aruco_prints/` — one A4 PNG per marker.

| Corridor | Marker IDs |
|----------|-----------|
| reto | 0, 1 |
| S_esq_dir | 10, 11, 12, 13 |
| U_esq | 20, 21, 22, 23 |

**Print instructions:**
- Print at **100% scale** (disable "fit to page")
- Physical marker size must be **190 mm** — the config depends on this
- Each PNG has a label at the bottom with the ID and corridor name

---

## Step 9 — Configure corridor dimensions

Edit `config/corridors.yaml` after building each corridor.
Fields marked `# measure after building` need real measurements (in meters).

Confirmed widths:
- `reto`: 1.4 m
- `S_esq_dir`, `U_esq`: 1.2 m

Marker placement per corridor:

**reto** — marker on start wall (ID 0, facing N) and end wall (ID 1, facing S).

**S_esq_dir** — marker at start (ID 10), right wall before first turn (ID 11),
left wall after first turn (ID 12), end wall (ID 13).

**U_esq** — top of first leg (ID 20), top center (ID 21),
top of second leg (ID 22), end wall (ID 23).

After updating `corridors.yaml`, the tracker uses the new values automatically.

---

## Step 10 — Run the corridor tracker

Launch once per session. Run multiple trials back-to-back without restarting.

```bash
cd ~/rpi5-realsense
source ~/realsense-env/bin/activate

# Basic session
python examples/corridor_tracker.py --corridor reto --conditions bengala colete

# With video recording
python examples/corridor_tracker.py --corridor reto --conditions bengala colete --record
```

**Keys during session:**

| Key | Action |
|-----|--------|
| `1`, `2`, `3`... | Select condition (shown in top bar) |
| `SPACE` | Start recording |
| `SPACE` | Stop — auto-saves CSV (and MP4 if `--record`) |
| `Q` | Quit and show session summary |

**Output files** — one per trial, auto-incremented:
```
data/reto_bengala_001.csv
data/reto_bengala_001.mp4   (only with --record)
data/reto_colete_001.csv
data/reto_bengala_002.csv   (second trial of same condition)
```

**CSV columns:** `t` (seconds from trial start), `x`, `y` (meters in corridor
coords), `yaw_deg` (head orientation from IMU gyro), `speed_mps`.

**Session summary** shown on quit:
```
── Session summary ──────────────────────────────────
  reto_bengala_001.csv     bengala    18.3s  550 frames
  reto_colete_001.csv      colete     15.1s  453 frames
─────────────────────────────────────────────────────
```

---

## Step 11 — Plot trial results

```bash
python examples/plot_trial.py \
  data/reto_bengala_001.csv \
  data/reto_colete_001.csv \
  --corridor reto \
  --out results/reto_trial1.png
```

The figure has four panels (similar to Peng et al. Fig. 5):
1. **Trajectory** — top-view corridor map with user path
2. **Head orientation** — yaw (degrees) vs time
3. **Walking speed** — m/s vs time
4. **Summary bars** — total time, total distance, average speed

Each CSV passed = one colored line/condition in the plot.
For single-trial plots (one representative trial per condition), this is
consistent with how Peng et al. present their results.

---

## Troubleshooting

### "No RealSense devices were found"

1. Confirm USB 3.0: `lsusb | grep -i intel`
2. Check udev rules: `ls /etc/udev/rules.d/ | grep realsense`
3. Reload udev: `sudo udevadm control --reload-rules && sudo udevadm trigger`
4. Replug the cable

### pip install fails — no ARM64 wheel

pyrealsense2 on PyPI has no ARM64 wheel (confirmed June 2026).
Must build from source — see Step 4.

### Intel apt repo fails on Debian Bookworm

Intel's `jammy` repo uses key `FB0B24895113F120` which is not in the `.pgp` file
they publish. The repo works on Ubuntu 22.04 but not easily on Debian 12.
Use source build instead.
