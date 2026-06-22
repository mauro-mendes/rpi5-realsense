"""Verify Intel RealSense D435i is connected and accessible."""
import sys

try:
    import pyrealsense2 as rs
except ImportError:
    print("ERROR: pyrealsense2 not found. Run setup/install.sh first.")
    sys.exit(1)

ctx = rs.context()
devices = ctx.query_devices()

if len(devices) == 0:
    print("ERROR: No RealSense devices found.")
    print("  - Check USB 3.0 (blue port)")
    print("  - Verify: ls /etc/udev/rules.d/ | grep realsense")
    sys.exit(1)

print(f"Found {len(devices)} device(s):")
for d in devices:
    print(f"  Name:     {d.get_info(rs.camera_info.name)}")
    print(f"  Serial:   {d.get_info(rs.camera_info.serial_number)}")
    print(f"  Firmware: {d.get_info(rs.camera_info.firmware_version)}")

print("\nTesting pipeline (5 frames)...")
pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(cfg)

for i in range(5):
    frames = pipeline.wait_for_frames()
    c = frames.get_color_frame()
    d = frames.get_depth_frame()
    status = "OK" if (c and d) else "MISSING"
    print(f"  Frame {i + 1}: {status}")

pipeline.stop()
print("\nRealSense OK")
