# Notes — rpi5-realsense

## Hardware

- Raspberry Pi 5 (8GB)
- Intel RealSense D435i
- Debian GNU/Linux 12 (Bookworm), aarch64
- Docker 29.3.0

## Setup log

### 2026-06-22

- D435i detected on Bus 004 (USB 3.0 blue port): `ID 8086:0b3a Intel Corp. Intel(R) RealSense(TM) Depth Camera 435i`
- udev rules installed: `/etc/udev/rules.d/99-realsense-libusb.rules`
- pyrealsense2: **no ARM64 wheel on PyPI** — must build from source
- Built librealsense from source with `-DBUILD_PYTHON_BINDINGS=ON -DPYTHON_EXECUTABLE=~/realsense-env/bin/python3`
- Python 3.11.2, pip 23.0.1

### pyrealsense2 ARM64 build notes

`pip install pyrealsense2` fails on Debian Bookworm aarch64 — Intel publishes no ARM64 wheel.
Solution: build librealsense from source pointing cmake to the venv Python.
The Python bindings (`.so`) install to the venv's `site-packages` automatically.

## Troubleshooting

### "No RealSense devices were found"

1. Confirm USB 3.0 (blue port): `lsusb | grep -i intel`
2. Check udev rules: `ls /etc/udev/rules.d/ | grep realsense`
3. Reload udev: `sudo udevadm control --reload-rules && sudo udevadm trigger`
4. Try replug the cable

### pip install fails with "externally-managed-environment"

Debian 12 protects system Python. Use venv:
```bash
python3 -m venv ~/realsense-env
~/realsense-env/bin/pip install pyrealsense2
```
