"""
record_trajectory.py — Ground truth: bola verde → trajectória top-view

Fases:
  1. Calibração ArUco automática (mesma lógica de test_aruco_pose.py)
     → aguarda CAM LOCKED (~5 depth válidos)
  2. Após lock: detecta bola verde por HSV + depth, grava posições no mundo
  3. Q=sair → salva CSV + gera PNG top-view (corredor + trajectória)

Uso (RPi5):
    conda activate rpi5-realsense
    python tools/record_trajectory.py
    python tools/record_trajectory.py --corridor reto_esq
    python tools/record_trajectory.py --headless --corridor S_esq

Teclas:
    Q — sair e gerar plot
    R — apaga trajectória (recomeça gravação)
    B — toggle overlay da máscara HSV (debug da bola)
    S — salva frame actual

Ajuste da bola:
    Se a bola não for detectada, editar BALL_HSV_LOW / BALL_HSV_HIGH abaixo.
    Primir B para ver pixels detectados em tempo real.
"""

import argparse, csv, os, time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml

# ── Matplotlib (backend Agg: salva PNG sem precisar de display) ───────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
YAML_PATH = Path(__file__).parent.parent / "config" / "corridors.yaml"
OUT_DIR   = Path(__file__).parent.parent / "output"

# ── ArUco ─────────────────────────────────────────────────────────────────────
DICT_TYPE   = cv2.aruco.DICT_4X4_50
MIN_MARKERS = 4
LOCK_AFTER  = 5          # depth válidos consecutivos para fixar pose

# ── Detecção da bola (HSV verde) ──────────────────────────────────────────────
# Valores lidos de config/ball_hsv.yaml (gerado por calibrate_ball_hsv.py).
# Se o ficheiro não existir, usa estes defaults.
_HSV_CFG = Path(__file__).parent.parent / "config" / "ball_hsv.yaml"
if _HSV_CFG.exists():
    with open(_HSV_CFG) as _f:
        _h = yaml.safe_load(_f).get("ball_hsv", {})
    BALL_HSV_LOW  = np.array([_h.get("h_low", 35),  _h.get("s_low", 50),
                               _h.get("v_low", 50)],  dtype=np.uint8)
    BALL_HSV_HIGH = np.array([_h.get("h_high", 85), _h.get("s_high", 255),
                               _h.get("v_high", 255)], dtype=np.uint8)
else:
    BALL_HSV_LOW  = np.array([35,  50,  50], dtype=np.uint8)
    BALL_HSV_HIGH = np.array([85, 255, 255], dtype=np.uint8)

MIN_BALL_AREA = 80       # px² — bola 150mm: ~254px² a 5m, ~177px² a 6m
RECORD_HZ     = 10       # pontos/segundo a gravar (máximo)

# ── Rotação da câmara (horizontal, apontada em +Y) ───────────────────────────
# X_cam = X_world  |  Y_cam = -Z_world  |  Z_cam = Y_world
R_INIT = np.array([[1, 0,  0],
                   [0, 0, -1],
                   [0, 1,  0]], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
def _has_display() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def load_yaml(path: Path):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    markers: dict[int, np.ndarray] = {}
    for corr in cfg["corridors"].values():
        for m in corr.get("markers", []):
            if m["id"] not in markers:
                markers[m["id"]] = np.array(m["pos"], dtype=np.float64)
    size = float(cfg.get("marker_size_m", 0.19))
    return markers, size, cfg


# ── Calibração: posição da câmara por depth ───────────────────────────────────
def depth_cam_estimate(ids, corners_list, markers, half,
                       depth_f, intr) -> "np.ndarray | None":
    ests = []
    for mid, corners in zip(ids, [c[0] for c in corners_list]):
        if mid not in markers:
            continue
        cx_px, cy_px = corners.mean(axis=0)
        samples = []
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                xi = int(np.clip(cx_px + dx, 0, 639))
                yi = int(np.clip(cy_px + dy, 0, 479))
                d  = depth_f.get_distance(xi, yi)
                if 0.3 < d < 12.0:
                    samples.append(d)
        if not samples:
            continue
        d     = float(np.median(samples))
        p_cam = np.array(rs.rs2_deproject_pixel_to_point(
            intr, [float(cx_px), float(cy_px)], d))
        ests.append(markers[mid] - R_INIT.T @ p_cam)
    return np.median(ests, axis=0) if ests else None


# ── Detecção da bola ──────────────────────────────────────────────────────────
def detect_ball(frame_bgr):
    """Devolve (cx, cy, raio, mask) ou None."""
    hsv  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BALL_HSV_LOW, BALL_HSV_HIGH)
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    best = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(best) < MIN_BALL_AREA:
        return None
    (bx, by), radius = cv2.minEnclosingCircle(best)
    if radius < 4:
        return None
    return int(bx), int(by), int(radius), mask


def ball_world_pos(px, py, depth_f, cam_pos, intr) -> "np.ndarray | None":
    """Pixel + depth → posição (x,y,z) no mundo."""
    samples = []
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            xi = int(np.clip(px + dx, 0, 639))
            yi = int(np.clip(py + dy, 0, 479))
            d  = depth_f.get_distance(xi, yi)
            if 0.1 < d < 10.0:
                samples.append(d)
    if not samples:
        return None
    d     = float(np.median(samples))
    p_cam = np.array(rs.rs2_deproject_pixel_to_point(
        intr, [float(px), float(py)], d))
    return cam_pos + R_INIT.T @ p_cam


# ── Plot final ────────────────────────────────────────────────────────────────
def generate_plot(trajectory: list, traj_times: list, cam_pos: np.ndarray,
                  corridor_key: str, cfg: dict, out_dir: Path) -> Path:
    corridor = cfg["corridors"].get(corridor_key, {})
    sg       = cfg.get("shared_geometry", {})

    total_w = corridor.get("total_width_m",  sg.get("total_width_m",  3.5))
    # U_esq só vai até u_depth_m; reto/S vão até total_depth_m
    total_d = corridor.get("total_depth_m",
              corridor.get("u_depth_m", sg.get("total_depth_m", 5.5)))

    fig, ax = plt.subplots(figsize=(6, 10))
    ax.set_aspect("equal")
    ax.set_title(f"Trajectória — {corridor_key}", fontsize=13)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, alpha=0.25)

    # Paredes
    for wall in corridor.get("walls", []):
        ax.plot([wall[0][0], wall[1][0]], [wall[0][1], wall[1][1]],
                "k-", linewidth=2)
    mw = corridor.get("movable_wall", {}).get("segment")
    if mw:
        ax.plot([mw[0][0], mw[1][0]], [mw[0][1], mw[1][1]],
                "b--", linewidth=2, label="Parede móvel")

    # Marcadores
    for m in corridor.get("markers", []):
        ax.plot(m["pos"][0], m["pos"][1], "rs", markersize=7, zorder=3)
        ax.text(m["pos"][0] + 0.05, m["pos"][1], str(m["id"]),
                fontsize=7, color="darkred", va="center")

    # Câmara
    ax.plot(cam_pos[0], cam_pos[1], "D", color="darkorange",
            markersize=11, zorder=4,
            label=f"Câmara ({cam_pos[0]:.2f}, {cam_pos[1]:.2f})")

    # Trajectória (cores por tempo em segundos)
    if trajectory:
        xs = [p[0] for p in trajectory]
        ys = [p[1] for p in trajectory]
        sc = ax.scatter(xs, ys, c=traj_times, cmap="plasma",
                        s=10, zorder=4, label="Trajectória")
        plt.colorbar(sc, ax=ax, label="Tempo (s)", shrink=0.6)
        ax.plot(xs[0],  ys[0],  "go", markersize=11, zorder=5, label="Início")
        ax.plot(xs[-1], ys[-1], "ro", markersize=11, zorder=5, label="Fim")

    ax.set_xlim(-0.3, total_w + 0.3)
    ax.set_ylim(cam_pos[1] - 0.4, total_d + 0.4)
    ax.legend(loc="upper right", fontsize=8)

    # Linha de entrada do corredor
    ax.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax.text(total_w / 2, -0.15, "entrada", ha="center",
            fontsize=8, color="gray")

    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = out_dir / f"trajectory_{corridor_key}_{ts}.png"
    fig.tight_layout()
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    return out


def _find_stable_start(pts: np.ndarray,
                       window: int = 5, max_spread: float = 0.25) -> int:
    """Devolve o índice do primeiro bloco de `window` pontos consecutivos
    com dispersão espacial ≤ max_spread. Descarta ruído inicial de HSV."""
    for i in range(len(pts) - window):
        chunk = pts[i : i + window]
        spread = float(np.max(np.linalg.norm(chunk - chunk.mean(axis=0), axis=1)))
        if spread <= max_spread:
            return i
    return 0   # fallback: usa desde o início


def _filter_outliers(pts: np.ndarray, max_jump: float = 0.6) -> np.ndarray:
    """Remove pontos com salto > max_jump metros em relação ao ponto anterior aceite.
    Se o resultado tiver < 20% dos pontos originais, devolve os pontos originais."""
    if len(pts) < 2:
        return pts
    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) <= max_jump:
            keep.append(i)
    filtered = pts[keep]
    # fallback: se filtrou demasiado, usa originais
    if len(filtered) < max(2, len(pts) // 5):
        return pts
    return filtered


def _smooth_path(pts: np.ndarray, window: int = 7) -> np.ndarray:
    """Média móvel centrada, preserva os extremos."""
    if len(pts) < window:
        return pts
    half = window // 2
    out  = pts.copy().astype(float)
    for i in range(half, len(pts) - half):
        out[i] = pts[i - half : i + half + 1].mean(axis=0)
    return out


def _resample_path(trajectory: list, n_pts: int = 200) -> "tuple | None":
    """Filtra outliers, suaviza e reamosta por distância acumulada (arco).
    Devolve (x_res, y_res) ou None se pontos insuficientes."""
    pts = np.array([[p[0], p[1]] for p in trajectory])
    # 1. descarta ruído inicial de HSV (antes da bola entrar a sério)
    start = _find_stable_start(pts)
    pts = pts[start:]
    # 2. remove saltos grandes
    pts = _filter_outliers(pts)
    if len(pts) >= 7:
        pts = _smooth_path(pts, window=7)
    if len(pts) < 2:
        return None
    segs  = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cumul = np.concatenate([[0.0], np.cumsum(segs)])
    total = cumul[-1]
    if total < 0.1:   # menos de 10cm — sem movimento útil
        return None
    d_uni = np.linspace(0.0, total, n_pts)
    x_res = np.interp(d_uni, cumul, pts[:, 0])
    y_res = np.interp(d_uni, cumul, pts[:, 1])
    return x_res, y_res


def generate_path_plot(trajectory: list, cam_pos: np.ndarray,
                       corridor_key: str, cfg: dict,
                       out_dir: Path, ts: str) -> Path:
    """Plot com caminho reamostrado por distância (pontos equidistantes em arco)."""
    corridor = cfg["corridors"].get(corridor_key, {})
    sg       = cfg.get("shared_geometry", {})

    total_w = corridor.get("total_width_m", sg.get("total_width_m", 3.5))
    total_d = corridor.get("total_depth_m",
              corridor.get("u_depth_m", sg.get("total_depth_m", 5.5)))

    fig, ax = plt.subplots(figsize=(6, 10))
    ax.set_aspect("equal")
    ax.set_title(f"Caminho (reamostrado) — {corridor_key}", fontsize=13)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.grid(True, alpha=0.25)

    # Paredes
    for wall in corridor.get("walls", []):
        ax.plot([wall[0][0], wall[1][0]], [wall[0][1], wall[1][1]],
                "k-", linewidth=2)
    mw = corridor.get("movable_wall", {}).get("segment")
    if mw:
        ax.plot([mw[0][0], mw[1][0]], [mw[0][1], mw[1][1]],
                "b--", linewidth=2, label="Parede móvel")

    # Marcadores
    for m in corridor.get("markers", []):
        ax.plot(m["pos"][0], m["pos"][1], "rs", markersize=7, zorder=3)
        ax.text(m["pos"][0] + 0.05, m["pos"][1], str(m["id"]),
                fontsize=7, color="darkred", va="center")

    # Câmara
    ax.plot(cam_pos[0], cam_pos[1], "D", color="darkorange",
            markersize=11, zorder=4,
            label=f"Câmara ({cam_pos[0]:.2f}, {cam_pos[1]:.2f})")

    # Caminho reamostrado
    res = _resample_path(trajectory) if trajectory else None
    if res is not None:
        x_res, y_res = res
        ax.plot(x_res, y_res, color="steelblue", linewidth=1.5,
                zorder=4, alpha=0.7)
        ax.scatter(x_res[::10], y_res[::10], color="steelblue",
                   s=18, zorder=5, label="Caminho (equidist.)")
        ax.plot(x_res[0],  y_res[0],  "go", markersize=11, zorder=6, label="Início")
        ax.plot(x_res[-1], y_res[-1], "ro", markersize=11, zorder=6, label="Fim")
    else:
        ax.text(0.5, 0.5, "pontos insuficientes", transform=ax.transAxes,
                ha="center", va="center", color="gray", fontsize=10)

    ax.set_xlim(-0.3, total_w + 0.3)
    ax.set_ylim(cam_pos[1] - 0.4, total_d + 0.4)
    ax.legend(loc="upper right", fontsize=8)

    ax.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax.text(total_w / 2, -0.15, "entrada", ha="center",
            fontsize=8, color="gray")

    out = out_dir / f"path_{corridor_key}_{ts}.png"
    fig.tight_layout()
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    return out


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corridor", default="U_esq",
                        choices=["U_esq", "reto_esq", "S_esq"],
                        help="Tipo de corredor para o plot final")
    parser.add_argument("--headless", action="store_true",
                        help="Sem janela — grava frames em output/")
    parser.add_argument("--record", action="store_true",
                        help="Grava vídeo da câmara em output/video_<corredor>_<ts>.avi")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    markers, marker_size, cfg = load_yaml(YAML_PATH)
    marker_half = marker_size / 2.0
    print(f"Corredor : {args.corridor}")
    hsv_src = f"(de {_HSV_CFG.name})" if _HSV_CFG.exists() else "(default)"
    print(f"Bola HSV : H=[{BALL_HSV_LOW[0]},{BALL_HSV_HIGH[0]}]"
          f"  S=[{BALL_HSV_LOW[1]},{BALL_HSV_HIGH[1]}]"
          f"  V=[{BALL_HSV_LOW[2]},{BALL_HSV_HIGH[2]}]  {hsv_src}")
    print(f"Marcadores conhecidos: {sorted(markers)}")

    # ── ArUco ─────────────────────────────────────────────────────────────────
    try:
        aruco_dict   = cv2.aruco.getPredefinedDictionary(DICT_TYPE)
        aruco_params = cv2.aruco.DetectorParameters()
        detector     = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        use_new_api  = True
    except AttributeError:
        aruco_dict   = cv2.aruco.Dictionary_get(DICT_TYPE)
        aruco_params = cv2.aruco.DetectorParameters_create()
        use_new_api  = False

    # ── RealSense ──────────────────────────────────────────────────────────────
    pipeline = rs.pipeline()
    cfg_rs   = rs.config()
    cfg_rs.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg_rs.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)
    profile  = pipeline.start(cfg_rs)
    align    = rs.align(rs.stream.color)
    print("[OK] RealSense 640×480 @ 30fps + depth")

    intr = (profile.get_stream(rs.stream.color)
            .as_video_stream_profile().get_intrinsics())
    print(f"     fx={intr.fx:.1f}  fy={intr.fy:.1f}"
          f"  cx={intr.ppx:.1f}  cy={intr.ppy:.1f}")

    # ── Janela ────────────────────────────────────────────────────────────────
    WIN = "Ground Truth — Q=sair  R=reset  B=mask  S=salvar"
    if not args.headless and _has_display():
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, 960, 540)
    else:
        args.headless = True

    # ── Estado ────────────────────────────────────────────────────────────────
    lock_count  = 0
    pose_locked = False
    cam_pos     = np.zeros(3)
    depth_hist: list[np.ndarray] = []

    trajectory:  list[np.ndarray] = []
    traj_times:  list[float]      = []   # segundos desde início da gravação
    t0_record:   float            = 0.0  # timestamp do primeiro ponto
    show_mask  = False
    save_idx   = 0
    frame_cnt  = 0
    last_record = 0.0
    last_log    = 0.0
    last_save   = 0.0
    RECORD_DT   = 1.0 / RECORD_HZ

    # ── Vídeo opcional ────────────────────────────────────────────────────────
    video_writer: "cv2.VideoWriter | None" = None
    video_path:   "Path | None"            = None
    if args.record:
        _vts      = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = OUT_DIR / f"video_{args.corridor}_{_vts}.avi"
        fourcc     = cv2.VideoWriter_fourcc(*"MJPG")
        video_writer = cv2.VideoWriter(str(video_path), fourcc, 30, (640, 480))
        print(f"[REC]  A gravar vídeo → {video_path.name}")

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError as e:
                print(f"[WARN] {e}")
                continue

            aligned = align.process(frames)
            color_f = aligned.get_color_frame()
            depth_f = aligned.get_depth_frame()
            if not color_f:
                continue

            frame = np.asanyarray(color_f.get_data()).copy()
            frame_cnt += 1
            now = time.time()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ── FASE 1: calibração ArUco ───────────────────────────────────
            if not pose_locked:
                if use_new_api:
                    corners_list, ids_raw, _ = detector.detectMarkers(gray)
                else:
                    corners_list, ids_raw, _ = cv2.aruco.detectMarkers(
                        gray, aruco_dict, parameters=aruco_params)

                n_known = 0
                if ids_raw is not None:
                    cv2.aruco.drawDetectedMarkers(frame, corners_list, ids_raw)
                    det_ids = ids_raw.flatten().tolist()
                    n_known = sum(1 for i in det_ids if i in markers)

                    if n_known >= MIN_MARKERS and depth_f:
                        est = depth_cam_estimate(
                            det_ids, corners_list, markers,
                            marker_half, depth_f, intr)
                        if est is not None and est[1] < 0 and est[2] > 1.0:
                            depth_hist.append(est)
                            if len(depth_hist) > 20:
                                depth_hist.pop(0)
                            lock_count += 1
                            if lock_count >= LOCK_AFTER:
                                cam_pos = np.mean(depth_hist, axis=0)
                                # Y físico medido com fita (ArUco depth subestima
                                # distância da câmara à entrada → erro sistemático em Y)
                                _sg = cfg.get("shared_geometry", {})
                                _cam_y_yaml = _sg.get("camera_y_m")
                                if _cam_y_yaml is not None:
                                    cam_pos[1] = float(_cam_y_yaml)
                                pose_locked = True
                                print(f"\n[LOCKED] cam=({cam_pos[0]:+.3f},"
                                      f"{cam_pos[1]:+.3f},{cam_pos[2]:+.3f})"
                                      f"  [Y=YAML]")
                                print("         A detectar bola...\n")

                # HUD calibração
                clr = (80, 200, 80) if n_known >= MIN_MARKERS else (80, 80, 255)
                cv2.putText(frame,
                            f"CALIBRANDO — calib {lock_count}/{LOCK_AFTER} "
                            f"mkr={n_known}",
                            (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, clr, 2)
                cv2.putText(frame,
                            "Aguarda marcadores ArUco visiveis e depth valido...",
                            (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (180, 180, 180), 1)

            # ── FASE 2: detecção da bola + gravação ────────────────────────
            else:
                result = detect_ball(frame)
                ball_world = None

                if result is not None:
                    bx, by, brad, ball_mask = result
                    if depth_f:
                        ball_world = ball_world_pos(bx, by, depth_f, cam_pos, intr)

                    # overlay bola
                    cv2.circle(frame, (bx, by), brad, (0, 255, 0), 2)
                    cv2.drawMarker(frame, (bx, by), (0, 255, 0),
                                   cv2.MARKER_CROSS, 20, 2)

                    if ball_world is not None:
                        lbl = (f"x={ball_world[0]:+.2f}  "
                               f"y={ball_world[1]:+.2f}  "
                               f"z={ball_world[2]:+.2f} m")
                        cv2.putText(frame, lbl, (bx + brad + 5, by),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                    (0, 255, 0), 2)
                        # grava com limite de taxa
                        if now - last_record >= RECORD_DT:
                            if not trajectory:
                                t0_record = now
                            trajectory.append(ball_world.copy())
                            traj_times.append(now - t0_record)
                            last_record = now

                    if show_mask:
                        frame[ball_mask > 0] = [0, 200, 0]

                # HUD estado
                cv2.rectangle(frame, (0, 0), (480, 38), (0, 55, 0), -1)
                cv2.putText(frame,
                            f"CAM LOCKED ({cam_pos[0]:+.2f},{cam_pos[1]:+.2f},"
                            f"{cam_pos[2]:+.2f})",
                            (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                            (80, 255, 80), 2)

                ball_str = "BOLA OK" if result else "sem bola"
                ball_clr = (80, 255, 80) if result else (80, 80, 255)
                cv2.putText(frame,
                            f"pts={len(trajectory):4d}  {ball_str}"
                            f"  B=mask  R=reset  Q=plot+sair",
                            (6, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                            ball_clr, 1)

                # log terminal
                if now - last_log >= 3.0:
                    bs = (f"bola=({ball_world[0]:+.2f},{ball_world[1]:+.2f})"
                          if ball_world is not None else "bola=N/A")
                    print(f"[f={frame_cnt:05d}] pts={len(trajectory):4d}  {bs}")
                    last_log = now

            # ── Grava frame no vídeo ──────────────────────────────────────
            if video_writer is not None:
                video_writer.write(frame)

            # ── Janela / headless ──────────────────────────────────────────
            if args.headless:
                if now - last_save >= 2.0:
                    cv2.imwrite(str(OUT_DIR / f"traj_{save_idx:04d}.png"), frame)
                    last_save = now
                    save_idx += 1
            else:
                try:
                    cv2.imshow(WIN, frame)
                except cv2.error:
                    args.headless = True
                    continue

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    trajectory.clear()
                    traj_times.clear()
                    print("Trajectória limpa — a gravar de novo.")
                elif key == ord("b"):
                    show_mask = not show_mask
                    print(f"Máscara HSV: {'ON' if show_mask else 'OFF'}")
                elif key == ord("s"):
                    save_idx += 1
                    fn = OUT_DIR / f"traj_frame_{save_idx:04d}.png"
                    cv2.imwrite(str(fn), frame)
                    print(f"Salvo: {fn.name}")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        if video_writer is not None:
            video_writer.release()
            print(f"[REC]  Vídeo guardado → {video_path.name}")
        print(f"\n{'─'*50}")

        # ── CSV ────────────────────────────────────────────────────────────
        if trajectory:
            ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = OUT_DIR / f"trajectory_{args.corridor}_{ts}.csv"
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["n", "t_s", "x_m", "y_m", "z_m"])
                for i, (p, t) in enumerate(zip(trajectory, traj_times)):
                    w.writerow([i, f"{t:.3f}",
                                f"{p[0]:.4f}", f"{p[1]:.4f}", f"{p[2]:.4f}"])
            print(f"[CSV]  {len(trajectory)} pontos → {csv_path.name}")

            # ── Plot PNG ───────────────────────────────────────────────────
            ts_str    = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = generate_plot(
                trajectory, traj_times, cam_pos, args.corridor, cfg, OUT_DIR)
            path_path = generate_path_plot(
                trajectory, cam_pos, args.corridor, cfg, OUT_DIR, ts_str)
            print(f"[PLOT] scatter → {plot_path.name}")
            print(f"[PLOT] caminho → {path_path.name}")
            print(f"       Transferir para PC: git pull após push do RPi5")
        else:
            print("[AVISO] Nenhum ponto gravado.")

        print("─" * 50)


if __name__ == "__main__":
    main()
