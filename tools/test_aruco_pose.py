"""
test_aruco_pose.py — Fase 1: deteção de ArUcos + estimação de pose via solvePnP

Fluxo por frame:
  1. Captura RGB da RealSense D435i
  2. Deteta marcadores DICT_4X4_50 presentes na cena
  3. Constrói pares (3D world pos do YAML, 2D pixel pos detetada)
  4. solvePnPRansac → posição estimada da câmara em coordenadas do mundo
  5. Mostra frame com IDs, corners, pose e erro de reprojeção

Uso (RPi5):
    conda activate rpi5-realsense
    python tools/test_aruco_pose.py
    python tools/test_aruco_pose.py --headless   # sem janela, grava frames

Teclas:
    Q — sair
    S — salva frame atual em output/aruco_pose_NNNN.png
    C — limpa histórico de estabilidade
"""

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml

def _has_display() -> bool:
    """Detecta se existe display disponível (X11 / Wayland)."""
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

# ── Config ────────────────────────────────────────────────────────────────────
YAML_PATH     = Path(__file__).parent.parent / "config" / "corridors.yaml"
OUT_DIR       = Path(__file__).parent.parent / "output"
DICT_TYPE     = cv2.aruco.DICT_4X4_50
MIN_MARKERS   = 4          # mínimo para solvePnP
HISTORY_LEN   = 30         # frames para cálculo de estabilidade
CAM_EXPECTED  = np.array([1.20, -0.55, 1.90])  # posição esperada (m)

# ── Carrega marcadores do YAML (IDs únicos) ────────────────────────────────
def load_marker_positions(yaml_path: Path) -> dict[int, np.ndarray]:
    with open(yaml_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    positions: dict[int, np.ndarray] = {}
    for corridor in cfg["corridors"].values():
        for m in corridor["markers"]:
            mid = m["id"]
            if mid not in positions:
                positions[mid] = np.array(m["pos"], dtype=np.float64)
    return positions


# ── Cálculo de erro de reprojeção ─────────────────────────────────────────
def reprojection_error(obj_pts, img_pts, rvec, tvec, K, dist):
    projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    projected = projected.reshape(-1, 2)
    errs = np.linalg.norm(img_pts - projected, axis=1)
    return float(errs.mean()), float(errs.max())


# ── Extrai posição da câmara no mundo ─────────────────────────────────────
def camera_world_pos(rvec, tvec) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec)
    return (-R.T @ tvec.flatten()).astype(np.float64)


# ── Overlay na frame ───────────────────────────────────────────────────────
def draw_overlay(frame, detected_ids, pose_valid, cam_pos, reproj_mean,
                 reproj_max, n_used, history):
    h, w = frame.shape[:2]

    def put(text, row, col=10, color=(220, 220, 220), scale=0.60, thick=1):
        cv2.putText(frame, text, (col, 30 + row * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    # fundo semitransparente (canto superior esquerdo)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (460, 210), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    ids_str = " ".join(str(i) for i in sorted(detected_ids)) if detected_ids else "—"
    put(f"Detetados : {ids_str}  ({len(detected_ids)} marcadores)", 0,
        color=(180, 220, 255))

    if pose_valid:
        ex, ey, ez = cam_pos - CAM_EXPECTED
        err_color = (80, 220, 80) if abs(ex) < 0.1 and abs(ey) < 0.1 else (80, 120, 255)
        put(f"Pos cam   : x={cam_pos[0]:+.3f}  y={cam_pos[1]:+.3f}  z={cam_pos[2]:+.3f} m",
            1, color=err_color)
        put(f"Esperado  : x={CAM_EXPECTED[0]:+.3f}  y={CAM_EXPECTED[1]:+.3f}"
            f"  z≥{CAM_EXPECTED[2]:+.3f} m", 2, color=(160, 160, 160))
        put(f"Erro      : Δx={ex:+.3f}  Δy={ey:+.3f}  Δz={ez:+.3f} m", 3,
            color=(200, 180, 80))
        rp_color = (80, 220, 80) if reproj_mean < 3.0 else (60, 100, 255)
        put(f"Reprojeção: mean={reproj_mean:.1f}px  max={reproj_max:.1f}px  "
            f"[{n_used} pts]", 4, color=rp_color)

        if len(history) >= 5:
            h_arr = np.array(history)
            std   = h_arr.std(axis=0)
            put(f"Estab.    : σx={std[0]:.4f}  σy={std[1]:.4f}  σz={std[2]:.4f} m",
                5, color=(160, 200, 200))
    else:
        put(f"solvePnP  : aguardando ≥{MIN_MARKERS} marcadores...", 1,
            color=(80, 80, 255))

    put("Q=sair  S=salvar  C=limpar histórico", 6,
        color=(120, 120, 120), scale=0.50)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true",
                        help="sem janela — grava 1 frame por segundo em output/")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    known = load_marker_positions(YAML_PATH)
    print(f"Marcadores conhecidos: {sorted(known)}")
    print(f"Posição esperada da câmara: {CAM_EXPECTED}")

    # ── ArUco API ──────────────────────────────────────────────────────────
    try:
        aruco_dict   = cv2.aruco.getPredefinedDictionary(DICT_TYPE)
        aruco_params = cv2.aruco.DetectorParameters()
        detector     = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        use_new_api  = True
    except AttributeError:
        aruco_dict   = cv2.aruco.Dictionary_get(DICT_TYPE)
        aruco_params = cv2.aruco.DetectorParameters_create()
        use_new_api  = False

    # ── RealSense ──────────────────────────────────────────────────────────
    pipeline = rs.pipeline()
    cfg_rs   = rs.config()
    cfg_rs.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile  = pipeline.start(cfg_rs)
    print("[OK] Pipeline RealSense iniciado (640x480 @ 30fps)")

    intr = (profile.get_stream(rs.stream.color)
            .as_video_stream_profile().get_intrinsics())
    K    = np.array([[intr.fx, 0, intr.ppx],
                     [0, intr.fy, intr.ppy],
                     [0, 0, 1]], dtype=np.float64)
    dist = np.array(intr.coeffs, dtype=np.float64)
    print(f"Intrínsecas: fx={intr.fx:.1f} fy={intr.fy:.1f} "
          f"cx={intr.ppx:.1f} cy={intr.ppy:.1f}")

    # janela de vídeo
    WIN = "ArUco Pose — Q=sair  S=salvar  C=limpar"
    if not args.headless:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN, 960, 540)
        print("[OK] Janela criada — aguarda primeiro frame...")

    history: list[np.ndarray] = []
    save_idx     = 0
    frame_count  = 0
    last_log     = time.time()
    last_save    = time.time()

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError as e:
                print(f"[WARN] wait_for_frames timeout: {e}")
                continue

            color_f = frames.get_color_frame()
            if not color_f:
                print("[WARN] frame de cor inválido — a ignorar")
                continue

            frame = np.asanyarray(color_f.get_data()).copy()
            frame_count += 1

            # ── Deteção ───────────────────────────────────────────────────
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if use_new_api:
                corners_list, ids_raw, _ = detector.detectMarkers(gray)
            else:
                corners_list, ids_raw, _ = cv2.aruco.detectMarkers(
                    gray, aruco_dict, parameters=aruco_params)

            detected_ids = (ids_raw.flatten().tolist()
                            if ids_raw is not None else [])

            if ids_raw is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners_list, ids_raw)

            # ── solvePnP ──────────────────────────────────────────────────
            obj_pts: list[np.ndarray] = []
            img_pts: list[np.ndarray] = []
            for mid, corners in zip(detected_ids,
                                    [c[0] for c in corners_list]):
                if mid in known:
                    obj_pts.append(known[mid])
                    img_pts.append(corners.mean(axis=0))

            pose_valid  = False
            cam_pos     = np.zeros(3)
            reproj_mean = reproj_max = 0.0

            if len(obj_pts) >= MIN_MARKERS:
                obj_arr = np.array(obj_pts, dtype=np.float64)
                img_arr = np.array(img_pts, dtype=np.float64)

                # RANSAC relaxado: 20px para aceitar erros de medição do YAML
                ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                    obj_arr, img_arr, K, dist,
                    iterationsCount=500, reprojectionError=20.0,
                    confidence=0.99, flags=cv2.SOLVEPNP_ITERATIVE)

                n_inliers = len(inliers.flatten()) if (ok and inliers is not None) else 0

                # fallback: solvePnP direto com todos os pontos
                if not ok or n_inliers < 4:
                    ok, rvec, tvec = cv2.solvePnP(
                        obj_arr, img_arr, K, dist,
                        flags=cv2.SOLVEPNP_ITERATIVE)
                    idx = np.arange(len(obj_arr))
                else:
                    idx = inliers.flatten()
                    # refina com inliers
                    ok, rvec, tvec = cv2.solvePnP(
                        obj_arr[idx], img_arr[idx], K, dist,
                        rvec, tvec, useExtrinsicGuess=True,
                        flags=cv2.SOLVEPNP_ITERATIVE)

                if ok:
                    pose_valid  = True
                    cam_pos     = camera_world_pos(rvec, tvec)
                    reproj_mean, reproj_max = reprojection_error(
                        obj_arr[idx], img_arr[idx], rvec, tvec, K, dist)
                    history.append(cam_pos.copy())
                    if len(history) > HISTORY_LEN:
                        history.pop(0)

            # ── Overlay na frame ──────────────────────────────────────────
            draw_overlay(frame, detected_ids, pose_valid, cam_pos,
                         reproj_mean, reproj_max, len(obj_pts), history)

            # ── Log no terminal a cada 2 s ────────────────────────────────
            now = time.time()
            if now - last_log >= 2.0:
                ids_str  = str(detected_ids) if detected_ids else "[]"
                pts_info = f"obj_pts={len(obj_pts)}"
                if pose_valid:
                    print(f"[f={frame_count:05d}] ids={ids_str}  {pts_info}  "
                          f"cam=({cam_pos[0]:+.3f},{cam_pos[1]:+.3f},"
                          f"{cam_pos[2]:+.3f})  reproj={reproj_mean:.1f}px")
                else:
                    print(f"[f={frame_count:05d}] ids={ids_str}  {pts_info}  "
                          f"solvePnP=FALHOU")
                last_log = now

            # ── Janela / headless ─────────────────────────────────────────
            if args.headless:
                if now - last_save >= 2.0:
                    fname = OUT_DIR / f"aruco_pose_{save_idx:04d}.png"
                    cv2.imwrite(str(fname), frame)
                    last_save = now
                    save_idx += 1
            else:
                try:
                    cv2.imshow(WIN, frame)
                except cv2.error as e:
                    print(f"[WARN] imshow falhou: {e}")
                    print("[INFO] A usar modo headless automático.")
                    args.headless = True
                    continue

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    save_idx += 1
                    fname = OUT_DIR / f"aruco_pose_{save_idx:04d}.png"
                    cv2.imwrite(str(fname), frame)
                    print(f"Salvo: {fname}")
                elif key == ord("c"):
                    history.clear()
                    print("Histórico de estabilidade limpo.")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
