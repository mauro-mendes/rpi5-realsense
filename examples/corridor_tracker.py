"""
Corridor tracker — ArUco localization + IMU yaw + walking speed → CSV.

Launch once per session. Run multiple trials back-to-back without restarting.

Usage:
    python examples/corridor_tracker.py --corridor reto --conditions bengala colete
    python examples/corridor_tracker.py --corridor S_esq_dir --conditions bengala colete vest

Keys:
    1, 2, 3...  select condition (shown on screen)
    SPACE       start / stop recording (auto-saves on stop)
    Q           quit and show session summary

Output files:
    data/<corridor>_<condition>_001.csv
    data/<corridor>_<condition>_002.csv   (increments if same condition repeated)
    ...
"""
import argparse
import csv
import math
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--corridor", required=True,
                    choices=["reto", "S_esq_dir", "U_esq"])
parser.add_argument("--conditions", nargs="+", default=["trial"],
                    help="Condition names, e.g. bengala colete vest")
parser.add_argument("--config", default="config/corridors.yaml")
args = parser.parse_args()

# ── Config ────────────────────────────────────────────────────────────────────
with open(args.config) as f:
    cfg = yaml.safe_load(f)

corridor    = cfg["corridors"][args.corridor]
marker_size = cfg["marker_size_m"]

marker_world = {m["id"]: np.array([m["x"], m["y"]], dtype=float)
                for m in corridor["markers"]}

out_dir = Path("data")
out_dir.mkdir(exist_ok=True)

def next_csv_path(corridor_key, condition):
    """Auto-increment: reto_bengala_001.csv, reto_bengala_002.csv ..."""
    i = 1
    while True:
        p = out_dir / f"{corridor_key}_{condition}_{i:03d}.csv"
        if not p.exists():
            return p
        i += 1

# ── ArUco ─────────────────────────────────────────────────────────────────────
try:
    aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector     = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    USE_NEW_API  = True
except AttributeError:
    aruco_dict   = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters_create()
    USE_NEW_API  = False

# ── RealSense ─────────────────────────────────────────────────────────────────
pipeline = rs.pipeline()
rs_cfg   = rs.config()
rs_cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
rs_cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f)
rs_cfg.enable_stream(rs.stream.gyro,  rs.format.motion_xyz32f)

profile  = pipeline.start(rs_cfg)
color_vp = profile.get_stream(rs.stream.color).as_video_stream_profile()
intr     = color_vp.get_intrinsics()
cam_mtx  = np.array([[intr.fx, 0, intr.ppx],
                      [0, intr.fy, intr.ppy],
                      [0, 0, 1]], dtype=float)
dist_c   = np.array(intr.coeffs, dtype=float)

# ── Session state ─────────────────────────────────────────────────────────────
conditions       = args.conditions
cond_idx         = 0           # currently selected condition
recording        = False
saved_trials     = []          # [(condition, path, n_frames, duration)]

# Per-trial state (reset on each SPACE-start)
rows             = []
yaw_deg          = 0.0
prev_gyro_t      = None
pos_history      = deque(maxlen=10)
pos_x = pos_y    = None
speed_mps        = 0.0
trail            = []          # [(x, y)] for top-view

print(f"\nCorridor : {args.corridor}")
print(f"Conditions: {conditions}")
print(f"Keys: {' '.join(f'{i+1}={c}' for i,c in enumerate(conditions))}")
print("      SPACE=start/stop   Q=quit\n")

# ── Helpers ───────────────────────────────────────────────────────────────────
MARKER_OBJ = np.array([[-marker_size/2,  marker_size/2, 0],
                        [ marker_size/2,  marker_size/2, 0],
                        [ marker_size/2, -marker_size/2, 0],
                        [-marker_size/2, -marker_size/2, 0]], dtype=float)

def estimate_position(corners, ids, frame):
    if ids is None:
        return None
    best_pos, best_dist = None, float("inf")
    for i, mid in enumerate(ids.flatten()):
        if mid not in marker_world:
            continue
        ok, rvec, tvec = cv2.solvePnP(MARKER_OBJ, corners[i][0].astype(float),
                                       cam_mtx, dist_c,
                                       flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            continue
        dist = float(np.linalg.norm(tvec))
        if dist < best_dist:
            best_dist = dist
            R, _ = cv2.Rodrigues(rvec)
            cam_in_marker = -R.T @ tvec.flatten()
            wx, wy = marker_world[mid]
            best_pos = (wx + cam_in_marker[0], wy - cam_in_marker[2])
            cv2.drawFrameAxes(frame, cam_mtx, dist_c, rvec, tvec, 0.1)
    return best_pos

def save_trial(condition):
    if not rows:
        print("  (no data, not saved)")
        return
    path = next_csv_path(args.corridor, condition)
    t0   = rows[0]["t"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t", "x", "y", "yaw_deg", "speed_mps"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "t": round(r["t"] - t0, 4)})
    duration = rows[-1]["t"] - rows[0]["t"]
    saved_trials.append((condition, path, len(rows), duration))
    print(f"  Saved: {path.name}  ({len(rows)} frames, {duration:.1f}s)")

def draw_topview():
    scale = 80
    pad   = 50
    w_m   = corridor.get("width_m", corridor.get("corridor_width_m", 1.2))
    l_m   = corridor.get("length_m", 6.0)
    W     = int((w_m + 2) * scale) + 2 * pad
    H     = int(l_m * scale) + 2 * pad
    img   = np.full((H, W, 3), 30, dtype=np.uint8)

    def px(x, y):
        return (int(x * scale) + pad + int(scale), H - pad - int(y * scale))

    for wall in corridor.get("walls", []):
        if len(wall) == 2:
            cv2.line(img, px(*wall[0]), px(*wall[1]), (180, 180, 180), 3)

    for m in corridor["markers"]:
        p = px(m["x"], m["y"])
        cv2.circle(img, p, 5, (0, 200, 255), -1)
        cv2.putText(img, str(m["id"]), (p[0]+7, p[1]+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

    sx, sy = corridor["start"]
    gx, gy = corridor["goal"]
    cv2.circle(img, px(sx, sy), 8, (0, 255, 100), -1)
    cv2.drawMarker(img, px(gx, gy), (0, 60, 255),
                   cv2.MARKER_CROSS, 14, 2)

    for i in range(1, len(trail)):
        cv2.line(img, px(*trail[i-1]), px(*trail[i]), (255, 140, 0), 2)

    if pos_x is not None:
        cv2.circle(img, px(pos_x, pos_y), 9, (0, 255, 255), -1)

    return img

def draw_hud(frame):
    cond  = conditions[cond_idx]
    color = (0, 60, 220) if recording else (60, 60, 60)

    # Condition selector bar at top
    bar = np.full((36, frame.shape[1], 3), 20, dtype=np.uint8)
    for i, c in enumerate(conditions):
        sel = (i == cond_idx)
        txt_color = (0, 255, 180) if sel else (140, 140, 140)
        prefix = f"[{i+1}] " if not sel else f">{i+1}< "
        cv2.putText(bar, prefix + c, (10 + i * 160, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, txt_color, 2 if sel else 1)

    frame_with_bar = np.vstack([bar, frame])

    pos_txt = f"({pos_x:.2f}, {pos_y:.2f}) m" if pos_x is not None else "no marker"
    lines = [
        (f"{'● REC  ' if recording else 'SPACE=start'}", color),
        (f"Pos: {pos_txt}", (0, 255, 255)),
        (f"Yaw: {yaw_deg:+.1f} deg", (0, 255, 255)),
        (f"Speed: {speed_mps:.2f} m/s", (0, 255, 255)),
        (f"Trials saved: {len(saved_trials)}", (180, 180, 180)),
    ]
    for i, (txt, col) in enumerate(lines):
        cv2.putText(frame_with_bar, txt,
                    (10, 36 + 28 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)

    return frame_with_bar

# ── Main loop ─────────────────────────────────────────────────────────────────
try:
    while True:
        frames = pipeline.wait_for_frames()

        # Gyro integration for yaw
        gyro_frame = frames.first_or_default(rs.stream.gyro)
        if gyro_frame:
            g  = gyro_frame.as_motion_frame().get_motion_data()
            gt = gyro_frame.get_timestamp() / 1000.0
            if prev_gyro_t is not None:
                yaw_deg += math.degrees(g.y) * (gt - prev_gyro_t)
            prev_gyro_t = gt

        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        t_now = color_frame.get_timestamp() / 1000.0
        color = np.asanyarray(color_frame.get_data())

        # ArUco
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        if USE_NEW_API:
            corners, ids, _ = detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, aruco_dict, parameters=aruco_params)
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
            if recording:
                trail.append((pos_x, pos_y))
                rows.append({"t": t_now, "x": round(pos_x, 4),
                             "y": round(pos_y, 4),
                             "yaw_deg": round(yaw_deg, 2),
                             "speed_mps": round(speed_mps, 4)})

        cv2.imshow("Camera", draw_hud(color))
        cv2.imshow("Top view", draw_topview())

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        # Condition selector: keys 1..9
        if ord('1') <= key <= ord('9'):
            idx = key - ord('1')
            if idx < len(conditions):
                if recording:
                    print("  Stop recording first (SPACE)")
                else:
                    cond_idx = idx
                    print(f"  Condition: {conditions[cond_idx]}")

        if key == ord(' '):
            if not recording:
                # Start
                rows.clear()
                trail.clear()
                yaw_deg = 0.0
                pos_history.clear()
                recording = True
                cond = conditions[cond_idx]
                print(f"\nRecording: {cond} …  (SPACE to stop)")
            else:
                # Stop
                recording = False
                cond = conditions[cond_idx]
                print(f"Stopped.")
                save_trial(cond)
                print(f"Ready. Select condition (keys) then SPACE to record again.")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()

# ── Session summary ───────────────────────────────────────────────────────────
print("\n── Session summary ──────────────────────────────────")
if saved_trials:
    for cond, path, n, dur in saved_trials:
        print(f"  {path.name:<40} {cond:<12} {dur:.1f}s  {n} frames")
else:
    print("  No trials saved.")
print("─────────────────────────────────────────────────────")
