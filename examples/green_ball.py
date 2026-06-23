"""
Color object tracker with real-time depth — RealSense D435i on RPi5.

Click on any object in the RGB window to track it by color.
The HSV range is auto-detected from a 20x20 pixel sample around the click.

Usage:
    source ~/realsense-env/bin/activate
    python examples/green_ball.py
"""
import numpy as np
import cv2
import pyrealsense2 as rs

# --- Color presets (press key to switch, or click to auto-detect) ---
PRESETS = {
    ord('1'): ("verde musgo",  np.array([67, 32,  0]), np.array([97, 152,  89])),
    ord('2'): ("verde folha",  np.array([46, 81,  0]), np.array([76, 201, 118])),
}

hsv_lower = PRESETS[ord('1')][1].copy()
hsv_upper = PRESETS[ord('1')][2].copy()
tracking_color = PRESETS[ord('1')][0]

def on_mouse_click(event, x, y, flags, param):
    global hsv_lower, hsv_upper, tracking_color
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    frame = param["frame"]
    if frame is None:
        return
    h, w = frame.shape[:2]
    # Sample 20x20 region around click
    r = 10
    x1, x2 = max(0, x - r), min(w, x + r)
    y1, y2 = max(0, y - r), min(h, y + r)
    roi = frame[y1:y2, x1:x2]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mean = hsv_roi.mean(axis=(0, 1))
    std  = hsv_roi.std(axis=(0, 1))
    tol = np.array([15, 60, 60])
    hsv_lower = np.clip(mean - tol, 0, [179, 255, 255]).astype(np.uint8)
    hsv_upper = np.clip(mean + tol, 0, [179, 255, 255]).astype(np.uint8)
    tracking_color = f"H={mean[0]:.0f} S={mean[1]:.0f} V={mean[2]:.0f}"
    print(f"\nClicked ({x},{y}) — HSV mean: {tracking_color}")
    print(f"  Range: lower={hsv_lower}  upper={hsv_upper}")

# --- Pipeline ---
pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
align = rs.align(rs.stream.color)

profile = pipeline.start(cfg)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
print(f"Depth scale: {depth_scale:.6f} m/unit")
print("Keys: 1=verde musgo  2=verde folha  click=auto-detect  q=quit")

temporal = rs.temporal_filter()
spatial  = rs.spatial_filter()

frame_data = {"frame": None}
cv2.namedWindow("RGB")
cv2.setMouseCallback("RGB", on_mouse_click, frame_data)

try:
    while True:
        frames = align.process(pipeline.wait_for_frames())
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        depth_frame = spatial.process(depth_frame).as_depth_frame()
        depth_frame = temporal.process(depth_frame).as_depth_frame()

        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())
        frame_data["frame"] = color.copy()

        # Object detection
        hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
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
                print(f"\r  Object ({cx},{cy})  depth={label}    ", end="", flush=True)

        # HUD
        cv2.putText(color, f"Tracking: {tracking_color}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(color, "1=musgo  2=folha  click=auto  q=quit", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Depth viz
        depth_m = depth * depth_scale
        depth_clipped = np.clip(depth_m, 0.1, 4.0)
        depth_8bit = ((depth_clipped - 0.1) / 3.9 * 255).astype(np.uint8)
        depth_viz = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)

        cv2.imshow("RGB", color)
        cv2.imshow("Depth", depth_viz)
        cv2.imshow("Mask", mask)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key in PRESETS:
            tracking_color, hsv_lower, hsv_upper = PRESETS[key]
            hsv_lower = hsv_lower.copy()
            hsv_upper = hsv_upper.copy()
            print(f"\nPreset: {tracking_color}  lower={hsv_lower}  upper={hsv_upper}")
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("\nStopped.")
