"""
calibrate_ball_hsv.py — Calibração HSV por clique na bola

Uso:
  1. Clica na bola na imagem da esquerda (várias vezes se precisares)
  2. O HSV é amostrado automaticamente e a máscara atualiza à direita
  3. Afina com os trackbars se necessário
  4. S = guardar em config/ball_hsv.yaml   Q = sair sem guardar   R = reset

conda activate rpi5-realsense
python tools/calibrate_ball_hsv.py
"""

from pathlib import Path
import cv2
import numpy as np
import pyrealsense2 as rs
import yaml

YAML_OUT    = Path(__file__).parent.parent / "config" / "ball_hsv.yaml"
WIN         = "HSV  |  esq=clica na bola   S=guardar   R=reset   Q=sair"
SAMPLE_R    = 15    # raio de amostragem à volta do clique (px)
H_TOL       = 10    # tolerância extra de hue após amostragem
SV_MARGIN   = 40    # abaixa mínimo de S e V este valor abaixo do amostrado

# ── RealSense (só cor) ────────────────────────────────────────────────────────
pipeline = rs.pipeline()
cfg      = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(cfg)
print("[OK] RealSense pronto.")
print("     Clica na bola na metade ESQUERDA da janela.")

# ── Estado dos cliques ────────────────────────────────────────────────────────
all_h: list[int] = []
all_s: list[int] = []
all_v: list[int] = []
click_pts: list[tuple[int, int]] = []   # para desenhar no frame

# Valores correntes do range (começa com defaults ou YAML existente)
if YAML_OUT.exists():
    with open(YAML_OUT) as f:
        _h = yaml.safe_load(f).get("ball_hsv", {})
    h_lo = _h.get("h_low",  35);  h_hi = _h.get("h_high", 85)
    s_lo = _h.get("s_low",  50);  s_hi = _h.get("s_high", 255)
    v_lo = _h.get("v_low",  50);  v_hi = _h.get("v_high", 255)
    print(f"     Valores anteriores carregados de {YAML_OUT.name}")
else:
    h_lo, h_hi = 35, 85
    s_lo, s_hi = 50, 255
    v_lo, v_hi = 50, 255

# Frame "congelado" para amostrar (atualizado no loop)
frozen_hsv = None


def update_range_from_samples():
    """Recalcula h_lo…v_hi a partir de todos os pixels amostrados."""
    global h_lo, h_hi, s_lo, s_hi, v_lo, v_hi
    if not all_h:
        return
    h_lo = max(0,   int(np.min(all_h)) - H_TOL)
    h_hi = min(179, int(np.max(all_h)) + H_TOL)
    s_lo = max(0,   int(np.min(all_s)) - SV_MARGIN)
    s_hi = 255
    v_lo = max(0,   int(np.min(all_v)) - SV_MARGIN)
    v_hi = 255
    # Atualiza trackbars
    for name, val in [("H low", h_lo), ("H high", h_hi),
                      ("S low", s_lo), ("S high", s_hi),
                      ("V low", v_lo), ("V high", v_hi)]:
        cv2.setTrackbarPos(name, WIN, val)


def on_mouse(event, x, y, flags, param):
    """Clique na metade esquerda da janela (frame RGB) → amostra HSV."""
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if x >= 640 or frozen_hsv is None:   # ignorar cliques na máscara
        return
    # Amostra pixels num círculo de raio SAMPLE_R
    for dy in range(-SAMPLE_R, SAMPLE_R + 1):
        for dx in range(-SAMPLE_R, SAMPLE_R + 1):
            if dx * dx + dy * dy <= SAMPLE_R * SAMPLE_R:
                xi = int(np.clip(x + dx, 0, 639))
                yi = int(np.clip(y + dy, 0, 479))
                h, s, v = frozen_hsv[yi, xi]
                all_h.append(int(h))
                all_s.append(int(s))
                all_v.append(int(v))
    click_pts.append((x, y))
    update_range_from_samples()
    print(f"  clique ({x},{y})  →  H:[{h_lo},{h_hi}]"
          f"  S:[{s_lo},{s_hi}]  V:[{v_lo},{v_hi}]"
          f"  ({len(all_h)} pixels amostrados)")


# ── Janela ────────────────────────────────────────────────────────────────────
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, 1280, 560)
cv2.setMouseCallback(WIN, on_mouse)

cv2.createTrackbar("H low",  WIN, h_lo, 179, lambda x: None)
cv2.createTrackbar("H high", WIN, h_hi, 179, lambda x: None)
cv2.createTrackbar("S low",  WIN, s_lo, 255, lambda x: None)
cv2.createTrackbar("S high", WIN, s_hi, 255, lambda x: None)
cv2.createTrackbar("V low",  WIN, v_lo, 255, lambda x: None)
cv2.createTrackbar("V high", WIN, v_hi, 255, lambda x: None)

# ── Loop ──────────────────────────────────────────────────────────────────────
try:
    while True:
        frames  = pipeline.wait_for_frames(timeout_ms=5000)
        color_f = frames.get_color_frame()
        if not color_f:
            continue

        frame      = np.asanyarray(color_f.get_data()).copy()
        frozen_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Lê trackbars (permitem afinação manual depois dos cliques)
        h_lo = cv2.getTrackbarPos("H low",  WIN)
        h_hi = cv2.getTrackbarPos("H high", WIN)
        s_lo = cv2.getTrackbarPos("S low",  WIN)
        s_hi = cv2.getTrackbarPos("S high", WIN)
        v_lo = cv2.getTrackbarPos("V low",  WIN)
        v_hi = cv2.getTrackbarPos("V high", WIN)

        # Máscara
        mask = cv2.inRange(frozen_hsv,
                           np.array([h_lo, s_lo, v_lo], dtype=np.uint8),
                           np.array([h_hi, s_hi, v_hi], dtype=np.uint8))
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        # Detecta blob e desenha círculo no frame
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            best = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(best) > 100:
                (bx, by), r = cv2.minEnclosingCircle(best)
                cv2.circle(frame, (int(bx), int(by)), int(r), (0, 255, 0), 2)
                cv2.putText(frame, f"r={r:.0f}px",
                            (int(bx) - 25, int(by) - int(r) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Marca os cliques no frame
        for px, py in click_pts:
            cv2.drawMarker(frame, (px, py), (0, 80, 255),
                           cv2.MARKER_CROSS, 20, 2)

        # HUD
        n_pts = len(all_h)
        status = (f"{n_pts} px amostrados | H:[{h_lo},{h_hi}]"
                  f"  S:[{s_lo},{s_hi}]  V:[{v_lo},{v_hi}]"
                  if n_pts > 0 else "Clica na bola à esquerda →")
        cv2.putText(frame, status, (6, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
        cv2.putText(frame, "S=guardar  R=reset  Q=sair", (6, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1)

        # Máscara a cores (verde nos pixels detectados)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_bgr[mask > 0] = [0, 220, 0]
        cv2.putText(mask_bgr, "mascara HSV", (6, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow(WIN, np.hstack([frame, mask_bgr]))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Saiu sem guardar.")
            break

        elif key == ord("r"):
            all_h.clear(); all_s.clear(); all_v.clear()
            click_pts.clear()
            h_lo, h_hi = 35, 85
            s_lo, s_hi = 50, 255
            v_lo, v_hi = 50, 255
            for name, val in [("H low", h_lo), ("H high", h_hi),
                              ("S low", s_lo), ("S high", s_hi),
                              ("V low", v_lo), ("V high", v_hi)]:
                cv2.setTrackbarPos(name, WIN, val)
            print("Reset. Clica de novo na bola.")

        elif key == ord("s"):
            data = {"ball_hsv": {
                "h_low":  int(h_lo), "h_high": int(h_hi),
                "s_low":  int(s_lo), "s_high": int(s_hi),
                "v_low":  int(v_lo), "v_high": int(v_hi),
            }}
            with open(YAML_OUT, "w") as f:
                yaml.dump(data, f, default_flow_style=False)
            print(f"\n[GUARDADO] {YAML_OUT}")
            print(f"  H:[{h_lo},{h_hi}]  S:[{s_lo},{s_hi}]  V:[{v_lo},{v_hi}]")
            print("  record_trajectory.py vai usar estes valores.\n")
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
