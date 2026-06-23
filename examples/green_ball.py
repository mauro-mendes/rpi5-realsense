"""
Green ball tracker with real-time depth — RealSense D435i on RPi5.

Tracks the largest green object in the RGB frame and reads its
distance from the aligned depth stream.

Usage:
    source ~/realsense-env/bin/activate
    python examples/green_ball.py
"""
import numpy as np
import cv2
import pyrealsense2 as rs

pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
align = rs.align(rs.stream.color)

profile = pipeline.start(cfg)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
print(f"Depth scale: {depth_scale:.6f} m/unit  |  Press 'q' to quit")

# Filters to reduce depth noise
temporal = rs.temporal_filter()
spatial  = rs.spatial_filter()

try:
    while True:
        frames = align.process(pipeline.wait_for_frames())
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        # Apply filters — cast back to depth_frame to keep get_distance()
        depth_frame = spatial.process(depth_frame).as_depth_frame()
        depth_frame = temporal.process(depth_frame).as_depth_frame()

        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())

        # Green detection in HSV
        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([40, 60, 60]), np.array([80, 255, 255]))
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 500:
                (cx, cy), radius = cv2.minEnclosingCircle(c)
                cx, cy = int(cx), int(cy)
                dist_m = depth_frame.get_distance(cx, cy)
                label = f"{dist_m:.2f} m" if dist_m > 0 else "---"
                cv2.circle(color, (cx, cy), int(radius), (0, 255, 0), 2)
                cv2.circle(color, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(color, label, (cx + 12, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                print(f"\r  Ball ({cx},{cy})  depth={label}    ", end="", flush=True)

        # Normalize depth to 0.1–4.0m range for full color use
        depth_m = depth * depth_scale
        depth_clipped = np.clip(depth_m, 0.1, 4.0)
        depth_8bit = ((depth_clipped - 0.1) / 3.9 * 255).astype(np.uint8)
        depth_viz = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)
        cv2.imshow("RGB", color)
        cv2.imshow("Depth", depth_viz)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("\nStopped.")
