# rpi5-realsense

Intel RealSense D435i on Raspberry Pi 5 — setup, verification and examples.

Tested on **Debian Bookworm (12), aarch64**, with the D435i on a USB 3.0 (blue) port.

---

## Requirements

- Raspberry Pi 5 (any RAM)
- Debian Bookworm (12), aarch64
- Intel RealSense D435i connected to **USB 3.0 (blue port)**
- Python 3.11+

---

## Setup

```bash
bash setup/install.sh
source ~/realsense-env/bin/activate
python setup/verify.py
```

`install.sh` does three things:
1. Installs Intel udev rules (required for USB access without sudo)
2. Creates a Python virtual environment at `~/realsense-env`
3. Installs `pyrealsense2`, `opencv-python` and `numpy`

---

## Examples

### Green ball tracker

Tracks the largest green object in the frame and displays its real-time depth:

```bash
source ~/realsense-env/bin/activate
python examples/green_ball.py
```

Shows a side-by-side view: RGB with bounding circle + distance label | depth colormap.

---

## Notes

See [docs/notes.md](docs/notes.md) for the full setup log and troubleshooting.

---

## Related

- [rpi5-realsense-orbslam3](https://github.com/mauro-mendes/rpi5-realsense-orbslam3) — ORB-SLAM3 in Docker with RealSense D435i on RPi5
