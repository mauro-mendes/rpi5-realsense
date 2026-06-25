"""
calibrate_ball_hsv.py — Calibração interactiva do HSV da bola

Mostra frame RGB + máscara HSV lado a lado com 6 trackbars.
Ajusta até a bola aparecer branca e o fundo preto.

Uso:
    conda activate rpi5-realsense
    python tools/calibrate_ball_hsv.py

Teclas:
    S — guarda valores em config/ball_hsv.yaml e sai
    Q — sai sem guardar
"""

from pathlib import Path
import cv2
import numpy as np
import pyrealsense2 as rs
import yaml

YAML_OUT = Path(__file__).parent.parent / "config" / "ball_hsv.yaml"
WIN      = "HSV Calibration — S=guardar  Q=sair"

# ── RealSense (só cor, sem depth) ─────────────────────────────────────────────
pipeline = rs.pipeline()
cfg      = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(cfg)
print("[OK] RealSense iniciado")
print("     Aponta a câmara para a bola e ajusta os sliders.")

# ── Janela + trackbars ────────────────────────────────────────────────────────
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, 1280, 540)

# Tenta ler valores existentes do YAML para partir daí
if YAML_OUT.exists():
    with open(YAML_OUT) as f:
        saved = yaml.safe_load(f).get("ball_hsv", {})
    h_lo = saved.get("h_low",  35)
    h_hi = saved.get("h_high", 85)
    s_lo = saved.get("s_low",  50)
    s_hi = saved.get("s_high", 255)
    v_lo = saved.get("v_low",  50)
    v_hi = saved.get("v_high", 255)
    print(f"     Valores anteriores carregados de {YAML_OUT.name}")
else:
    h_lo, h_hi = 35, 85
    s_lo, s_hi = 50, 255
    v_lo, v_hi = 50, 255

cv2.createTrackbar("H low",  WIN, h_lo, 179, lambda x: None)
cv2.createTrackbar("H high", WIN, h_hi, 179, lambda x: None)
cv2.createTrackbar("S low",  WIN, s_lo, 255, lambda x: None)
cv2.createTrackbar("S high", WIN, s_hi, 255, lambda x: None)
cv2.createTrackbar("V low",  WIN, v_lo, 255, lambda x: None)
cv2.createTrackbar("V high", WIN, v_hi, 255, lambda x: None)

try:
    while True:
        frames  = pipeline.wait_for_frames(timeout_ms=5000)
        color_f = frames.get_color_frame()
        if not color_f:
            continue

        frame = np.asanyarray(color_f.get_data()).copy()

        # Lê trackbars
        hl = cv2.getTrackbarPos("H low",  WIN)
        hh = cv2.getTrackbarPos("H high", WIN)
        sl = cv2.getTrackbarPos("S low",  WIN)
        sh = cv2.getTrackbarPos("S high", WIN)
        vl = cv2.getTrackbarPos("V low",  WIN)
        vh = cv2.getTrackbarPos("V high", WIN)

        # Máscara HSV
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,
                           np.array([hl, sl, vl], dtype=np.uint8),
                           np.array([hh, sh, vh], dtype=np.uint8))

        # Limpeza morfológica
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        # Detecta maior blob e desenha círculo
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            best = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(best) > 100:
                (bx, by), r = cv2.minEnclosingCircle(best)
                cv2.circle(frame, (int(bx), int(by)), int(r), (0, 255, 0), 2)
                cv2.putText(frame, f"r={r:.0f}px  area={cv2.contourArea(best):.0f}",
                            (int(bx) - 60, int(by) - int(r) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        # HUD valores actuais
        info = (f"H:[{hl},{hh}]  S:[{sl},{sh}]  V:[{vl},{vh}]"
                f"   S=guardar  Q=sair")
        cv2.putText(frame, info, (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 220, 255), 2)

        # Máscara a cores (verde nos pixels detectados)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_bgr[mask > 0] = [0, 220, 0]

        combined = np.hstack([frame, mask_bgr])
        cv2.imshow(WIN, combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Saiu sem guardar.")
            break
        elif key == ord("s"):
            data = {
                "ball_hsv": {
                    "h_low":  int(hl), "h_high": int(hh),
                    "s_low":  int(sl), "s_high": int(sh),
                    "v_low":  int(vl), "v_high": int(vh),
                }
            }
            with open(YAML_OUT, "w") as f:
                yaml.dump(data, f, default_flow_style=False)
            print(f"\n[GUARDADO] {YAML_OUT}")
            print(f"  H:[{hl},{hh}]  S:[{sl},{sh}]  V:[{vl},{vh}]")
            print("  record_trajectory.py vai usar estes valores automaticamente.\n")
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
