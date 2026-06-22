#!/bin/bash
# RealSense D435i setup — Raspberry Pi 5, Debian Bookworm (12)
#
# NOTE: pyrealsense2 has no ARM64 wheel on PyPI.
# This script builds librealsense from source with Python bindings.
# Build time: ~30–60 min on RPi5.
set -e

echo "[1/4] Installing udev rules..."
wget -q https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules
sudo cp 99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
rm 99-realsense-libusb.rules
echo "  Done."

echo "[2/4] Creating virtual environment..."
python3 -m venv ~/realsense-env
echo "  venv: ~/realsense-env"

echo "[3/4] Installing build dependencies..."
sudo apt-get install -y \
    cmake build-essential \
    libssl-dev libusb-1.0-0-dev pkg-config \
    libgtk-3-dev libglfw3-dev \
    python3-dev
echo "  Done."

echo "[4/4] Building librealsense from source (30–60 min)..."
git clone --depth 1 https://github.com/IntelRealSense/librealsense.git ~/librealsense-src
cd ~/librealsense-src && mkdir -p build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DPYTHON_EXECUTABLE="$HOME/realsense-env/bin/python3" \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_GRAPHICAL_EXAMPLES=OFF

make -j$(nproc)
sudo make install
sudo ldconfig
echo "  Done."

echo ""
echo "Setup complete. Activate with:"
echo "  source ~/realsense-env/bin/activate"
echo "Verify with:"
echo "  python setup/verify.py"
