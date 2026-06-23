"""
Corridor tracker using ArUco markers + RealSense D435i IMU.

Records user position, head orientation and walking speed to CSV.
Run once per trial; press SPACE to start/stop recording, Q to quit.

Usage:
    source ~/realsense-env/bin/activate
    python examples/corridor_tracker.py --corridor reto
    python examples/corridor_tracker.py --corridor S_esq_dir
    python examples/corridor_tracker.py --corridor U_esq

Output:
    data/<corridor>_<timestamp>.csv
"""
import argparse
import csv
import math
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--corridor", required=True,
                    choices=["reto", "S_esq_dir", "U_esq"],
                    help="Which corridor to track")
parser.add_argument("--config", default="config/corridors.yaml")
parser.add_argument("--label", default="",
                    help="Trial label (e.g. bengala, colete)")
args = parser.parse_args()

# ── Config ───────────────────────────────────────────────────────────────────
with open(args.config) as f:
    cfg = yaml.safe_load(f)

corridor = cfg["corridors"][args.corridor]
marker_size = cfg["marker_size_m"]

marker_world = {}   # id -> (x, y) in corridor coords
for m in corridor["markers"]:
    marker_world[m["id"]] = np.array([m["x"], m["y"]], dtype=float)

# ── Output ───────────────────────────────────────────────────────────────────
out_dir = Path("data")
out_dir.mkdir(exist_ok=True)
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
label = f"_{args.label}" if args.label else ""
csv_path = out_dir / f"{args.corridor}{label}_{timestamp_str}.csv"

# ── ArUco ────────────────────────────────────────────────────────────────────
try:
    aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector     = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    USE_NEW_API  = True
except AttributeError:
    aruco_dict   = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters_create()
    USE_NEW_API  = False

# ── RealSense pipeline ────────────────────────────────────────────────────────
pipeline = rs.pipeline()
rs_cfg   = rs.config()
rs_cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8,  30)
rs_cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f)
rs_cfg.enable_stream(rs.stream.gyro,  rs.format.motion_xyz32f)

profile    = pipeline.start(rs_cfg)
color_prof = profile.get_stream(rs.stream.color).as_video_stream_profile()
intr       = color_prof.get_intrinsics()
cam_matrix = np.array([[intr.fx, 0, intr.ppx],
                        [0, intr.fy, intr.ppy],
                        [0, 0, 1]], dtype=float)
dist_coeffs = np.array(intr.coeffs, dtype=float)

print(f"Camera: {intr.width}x{intr.height}  fx={intr.fx:.1f} fy={intr.fy:.1f}")
print(f"Corridor: {args.corridor}  |  Markers: {list(marker_world.keys())}")
print(f"Output: {csv_path}")
print("SPACE = start/stop recording   Q = quit")

# ── State ─────────────────────────────────────────────────────────────────────
recording  = False
yaw_deg    = 0.0          # integrated gyro yaw
prev_gyro_t = None
pos_history = deque(maxlen=10)   # (t, x, y) for speed smoothing
pos_x = pos_y = None
speed_mps  = 0.0
rows       = []

# ── Helpers ──────────────────────────────────────────────────────────────────
def estimate_position(corners, ids, frame):
    """Return (x, y) in corridor coords from best visible marker, or None."""
    if ids is None:
        return None
    best = None
    best_dist = float("inf")
    for i, mid in enumerate(ids.flatten()):
        if mid not in marker_world:
            continue
        c = corners[i]
        ok, rvec, tvec = cv2.solvePnP(
            np.array([[-marker_size/2,  marker_size/2, 0],
                      [ marker_size/2,  marker_size/2, 0],
                      [ marker_size/2, -marker_size/2, 0],
                      [-marker_size/2, -marker_size/2, 0]], dtype=float),
            c[0].astype(float), cam_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if not ok:
            continue
        dist = float(np.linalg.norm(tvec))
        if dist < best_dist:
            best_dist = dist
            # Camera position relative to marker (invert tvec; ignore z=height)
            R, _ = cv2.Rodrigues(rvec)
            cam_in_marker = -R.T @ tvec.flatten()
            wx, wy = marker_world[mid]
            # cam_in_marker: x=right, y=up, z=toward marker
            # corridor coord: +x=right, +y=forward → use cam x and -z
            best = (wx + cam_in_marker[0], wy - cam_in_marker[2])
            cv2.drawFrameAxes(frame, cam_matrix, dist_coeffs, rvec, tvec, 0.1)
    return best

def draw_topview(pos, corridor_cfg, recording):
    """Render a simple top-down corridor map with user dot."""
    scale = 100          # pixels per meter
    pad   = 60
    w_m   = corridor_cfg.get("width_m", corridor_cfg.get("corridor_width_m", 1.2))
    l_m   = corridor_cfg.get("length_m", 6.0)
    W = int(w_m * scale) + 2 * pad + 200
    H = int(l_m * scale) + 2 * pad
    canvas = np.ones((H, W, 3), dtype=np.uint8) * 30

    def to_px(x, y):
        return (int(x * scale) + pad + 100, H - pad - int(y * scale))

    # Walls
    for wall in corridor_cfg.get("walls", []):
        if len(wall) == 2:
            p1 = to_px(*wall[0])
            p2 = to_px(*wall[1])
            cv2.line(canvas, p1, p2, (180, 180, 180), 3)

    # Markers
    for m in corridor_cfg["markers"]:
        px = to_px(m["x"], m["y"])
        cv2.circle(canvas, px, 6, (0, 200, 255), -1)
        cv2.putText(canvas, str(m["id"]), (px[0]+8, px[1]+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

    # Start / goal
    sx, sy = corridor_cfg["start"]
    gx, gy = corridor_cfg["goal"]
    cv2.circle(canvas, to_px(sx, sy), 8, (0, 255, 0), -1)
    cv2.putText(canvas, "Start", (to_px(sx,sy)[0]+10, to_px(sx,sy)[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.drawMarker(canvas, to_px(gx, gy), (0,0,255),
                   cv2.MARKER_CROSS, 15, 2)

    # Trail
    pts = list(pos_history)
    for i in range(1, len(pts)):
        cv2.line(canvas, to_px(pts[i-1][1], pts[i-1][2]),
                          to_px(pts[i][1],   pts[i][2]),
                 (255, 140, 0), 2)

    # Current position
    if pos:
        cv2.circle(canvas, to_px(*pos), 10, (0, 255, 255), -1)

    # Recording indicator
    color = (0, 0, 220) if recording else (80, 80, 80)
    cv2.putText(canvas, "REC" if recording else "---",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return canvas

# ── Main loop ────────────────────────────────────────────────────────────────
try:
    while True:
        frames = pipeline.wait_for_frames()

        # IMU
        accel_frame = frames.first_or_default(rs.stream.accel)
        gyro_frame  = frames.first_or_default(rs.stream.gyro)
        if gyro_frame:
            gdata = gyro_frame.as_motion_frame().get_motion_data()
            gt    = gyro_frame.get_timestamp() / 1000.0
            if prev_gyro_t is not None:
                dt = gt - prev_gyro_t
                yaw_deg += math.degrees(gdata.y) * dt   # gyro.y = yaw axis
            prev_gyro_t = gt

        # Color
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        color = np.asanyarray(color_frame.get_data())
        t_now = color_frame.get_timestamp() / 1000.0

        # ArUco detection
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        if USE_NEW_API:
            corners, ids, _ = detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(color, corners, ids)

        pos = estimate_position(corners, ids, color)
        if pos:
            pos_x, pos_y = pos
            pos_history.append((t_now, pos_x, pos_y))
            if len(pos_history) >= 2:
                dt = pos_history[-1][0] - pos_history[0][0]
                dx = pos_history[-1][1] - pos_history[0][1]
                dy = pos_history[-1][2] - pos_history[0][2]
                if dt > 0:
                    speed_mps = math.sqrt(dx**2 + dy**2) / dt

        # HUD overlay
        hud = f"Pos: ({pos_x:.2f},{pos_y:.2f})m" if pos_x is not None else "Pos: ---"
        cv2.putText(color, hud,          (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)
        cv2.putText(color, f"Yaw: {yaw_deg:.1f}deg", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)
        cv2.putText(color, f"Speed: {speed_mps:.2f} m/s", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)
        rec_color = (0, 0, 220) if recording else (120, 120, 120)
        cv2.putText(color, "● REC" if recording else "SPACE=rec  Q=quit",
                    (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, rec_color, 2)

        if recording and pos_x is not None:
            rows.append({
                "t":         round(t_now, 4),
                "x":         round(pos_x, 4),
                "y":         round(pos_y, 4),
                "yaw_deg":   round(yaw_deg, 2),
                "speed_mps": round(speed_mps, 4),
            })

        topview = draw_topview((pos_x, pos_y) if pos_x is not None else None,
                               corridor, recording)

        cv2.imshow("Camera", color)
        cv2.imshow("Top view", topview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord(' '):
            recording = not recording
            if recording:
                rows.clear()
                yaw_deg = 0.0
                pos_history.clear()
                print("Recording started…")
            else:
                print(f"Recording stopped — {len(rows)} frames")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

# ── Save CSV ──────────────────────────────────────────────────────────────────
if rows:
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["t", "x", "y", "yaw_deg", "speed_mps"])
        writer.writeheader()
        writer.writerows(rows)
    # Normalize t to start at 0
    t0 = rows[0]["t"]
    lines = open(csv_path).readlines()
    with open(csv_path, "w") as f:
        f.write(lines[0])
        for line in lines[1:]:
            parts = line.strip().split(",")
            parts[0] = str(round(float(parts[0]) - t0, 4))
            f.write(",".join(parts) + "\n")
    print(f"Saved: {csv_path}")
else:
    print("No data recorded.")
