#!/bin/bash
# RealSense D435i setup — Raspberry Pi 5, Debian Bookworm (12)
set -e

echo "[1/3] Installing udev rules..."
wget -q https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules
sudo cp 99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
rm 99-realsense-libusb.rules
echo "  Done."

echo "[2/3] Creating virtual environment..."
python3 -m venv ~/realsense-env
echo "  venv: ~/realsense-env"

echo "[3/3] Installing pyrealsense2..."
~/realsense-env/bin/pip install --quiet pyrealsense2 opencv-python numpy
echo "  Done."

echo ""
echo "Setup complete. Activate with:"
echo "  source ~/realsense-env/bin/activate"
echo "Verify with:"
echo "  python setup/verify.py"
