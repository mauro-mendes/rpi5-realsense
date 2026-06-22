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

Build time on RPi5 (4 cores): ~30–40 min  
Started: 2026-06-22 ~19:20

*(section to be updated when build completes)*

---

## Step 5 — Verify pyrealsense2

*(to be completed)*

```bash
source ~/realsense-env/bin/activate
python setup/verify.py
```

---

## Step 6 — Run green ball example

*(to be completed)*

```bash
source ~/realsense-env/bin/activate
python examples/green_ball.py
```

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
