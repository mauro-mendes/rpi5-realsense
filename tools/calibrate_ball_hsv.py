"""
calibrate_ball_hsv.py — Calibração HSV por clique na bola

Uso:
  1. Clica na bola na imagem da esquerda (várias vezes)
  2. A máscara (direita) atualiza — bola deve ficar branca, resto preto
  3. Afina com os trackbars se necessário
  4. S = guardar em config/ball_hsv.yaml   R = reset   Q = sair sem guardar

conda activate rpi5-realsense
python tools/calibrate_ball_hsv.py
"""

from pathlib import Path
import cv2
import numpy as np
import pyrealsense2 as rs
import yaml

YAML_OUT  = Path(__file__).parent.parent / "config" / "ball_hsv.yaml"
WIN       = "HSV Cal  |  clica na bola (esq)  S=guardar  R=reset  Q=sair"
SAMPLE_R  = 15   # raio de amostragem à volta do clique (px)
H_TOL     = 10   # tolerância extra de hue
SV_MARGIN = 40   # margem abaixo do mínimo amostrado para S e V

# ── RealSense ─────────────────────────────────────────────────────────────────
pipeline = rs.pipeline()
cfg      = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(cfg)
print("[OK] RealSense pronto — clica na bola na janela.")

# ── Carrega valores anteriores (ou defaults) ──────────────────────────────────
if YAML_OUT.exists():
    with open(YAML_OUT) as f:
        _h = yaml.safe_load(f).get("ball_hsv", {})
    h_lo = _h.get("h_low",  35);  h_hi = _h.get("h_high", 85)
    s_lo = _h.get("s_low",  50);  s_hi = _h.get("s_high", 255)
    v_lo = _h.get("v_low",  50);  v_hi = _h.get("v_high", 255)
    print(f"     Carregado de {YAML_OUT.name}")
else:
    h_lo, h_hi = 35, 85
    s_lo, s_hi = 50, 255
    v_lo, v_hi = 50, 255

# ── Estado partilhado entre callback e loop ───────────────────────────────────
all_h:      list[int]          = []
all_s:      list[int]          = []
all_v:      list[int]          = []
click_pts:  list[tuple]        = []
pending_click: list[tuple]     = []   # cliques pendentes para processar no loop
frozen_frame = None                   # frame capturado no momento do clique


def on_mouse(event, x, y, flags, param):
    """Só regista o clique — processamento feito no loop principal."""
    if event == cv2.EVENT_LBUTTONDOWN and x < 640:
        pending_click.append((x, y))


# ── Janela + trackbars ────────────────────────────────────────────────────────
cv2.namedWindow(WIN)          # sem flags: melhor compatibilidade com trackbars
cv2.setMouseCallback(WIN, on_mouse)

cv2.createTrackbar("H low",  WIN, h_lo, 179, lambda _: None)
cv2.createTrackbar("H high", WIN, h_hi, 179, lambda _: None)
cv2.createTrackbar("S low",  WIN, s_lo, 255, lambda _: None)
cv2.createTrackbar("S high", WIN, s_hi, 255, lambda _: None)
cv2.createTrackbar("V low",  WIN, v_lo, 255, lambda _: None)
cv2.createTrackbar("V high", WIN, v_hi, 255, lambda _: None)

need_set_bars = False   # flag: precisa de actualizar trackbars no próximo tick

# ── Loop ──────────────────────────────────────────────────────────────────────
try:
    while True:
        frames  = pipeline.wait_for_frames(timeout_ms=5000)
        color_f = frames.get_color_frame()
        if not color_f:
            continue

        frame       = np.asanyarray(color_f.get_data()).copy()
        frame_hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frozen_frame = frame.copy()

        # ── Processa cliques pendentes (no loop principal, thread segura) ──
        for (cx, cy) in pending_click:
            for dy in range(-SAMPLE_R, SAMPLE_R + 1):
                for dx in range(-SAMPLE_R, SAMPLE_R + 1):
                    if dx * dx + dy * dy <= SAMPLE_R * SAMPLE_R:
                        xi = int(np.clip(cx + dx, 0, 639))
                        yi = int(np.clip(cy + dy, 0, 479))
                        h, s, v = frame_hsv[yi, xi]
                        all_h.append(int(h))
                        all_s.append(int(s))
                        all_v.append(int(v))
            click_pts.append((cx, cy))
            # Recalcula range
            h_lo = max(0,   int(np.min(all_h)) - H_TOL)
            h_hi = min(179, int(np.max(all_h)) + H_TOL)
            s_lo = max(0,   int(np.min(all_s)) - SV_MARGIN)
            s_hi = 255
            v_lo = max(0,   int(np.min(all_v)) - SV_MARGIN)
            v_hi = 255
            need_set_bars = True
            print(f"  clique ({cx},{cy})  H:[{h_lo},{h_hi}]"
                  f"  S:[{s_lo},{s_hi}]  V:[{v_lo},{v_hi}]"
                  f"  ({len(all_h)} px)")
        pending_click.clear()

        # ── Actualiza trackbars se necessário (sempre no loop) ─────────────
        if need_set_bars:
            cv2.setTrackbarPos("H low",  WIN, h_lo)
            cv2.setTrackbarPos("H high", WIN, h_hi)
            cv2.setTrackbarPos("S low",  WIN, s_lo)
            cv2.setTrackbarPos("S high", WIN, s_hi)
            cv2.setTrackbarPos("V low",  WIN, v_lo)
            cv2.setTrackbarPos("V high", WIN, v_hi)
            need_set_bars = False
        else:
            # Lê afinação manual
            h_lo = cv2.getTrackbarPos("H low",  WIN)
            h_hi = cv2.getTrackbarPos("H high", WIN)
            s_lo = cv2.getTrackbarPos("S low",  WIN)
            s_hi = cv2.getTrackbarPos("S high", WIN)
            v_lo = cv2.getTrackbarPos("V low",  WIN)
            v_hi = cv2.getTrackbarPos("V high", WIN)

        # ── Máscara HSV ────────────────────────────────────────────────────
        mask = cv2.inRange(frame_hsv,
                           np.array([h_lo, s_lo, v_lo], dtype=np.uint8),
                           np.array([h_hi, s_hi, v_hi], dtype=np.uint8))
        k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        # Blob detection + círculo no frame
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            best = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(best) > 50:
                (bx, by), r = cv2.minEnclosingCircle(best)
                cv2.circle(frame, (int(bx), int(by)), int(r), (0, 255, 0), 2)

        # Marca cliques
        for (px, py) in click_pts:
            cv2.drawMarker(frame, (px, py), (0, 80, 255),
                           cv2.MARKER_CROSS, 20, 2)

        # HUD
        n = len(all_h)
        hud = (f"{n} px | H:[{h_lo},{h_hi}] S:[{s_lo},{s_hi}] V:[{v_lo},{v_hi}]"
               if n > 0 else "Clica na bola (metade esquerda)")
        cv2.putText(frame, hud, (6, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)

        # Máscara visível a verde
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_bgr[mask > 0] = [0, 220, 0]
        cv2.putText(mask_bgr, "mascara", (6, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow(WIN, np.hstack([frame, mask_bgr]))

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Saiu sem guardar.")
            break

        elif key == ord("r"):
            all_h.clear(); all_s.clear(); all_v.clear()
            click_pts.clear()
            h_lo, h_hi = 35, 85;  s_lo, s_hi = 50, 255;  v_lo, v_hi = 50, 255
            need_set_bars = True
            print("Reset.")

        elif key == ord("s"):
            data = {"ball_hsv": {
                "h_low": int(h_lo), "h_high": int(h_hi),
                "s_low": int(s_lo), "s_high": int(s_hi),
                "v_low": int(v_lo), "v_high": int(v_hi),
            }}
            with open(YAML_OUT, "w") as f:
                yaml.dump(data, f, default_flow_style=False)
            print(f"\n[GUARDADO] {YAML_OUT}")
            print(f"  H:[{h_lo},{h_hi}]  S:[{s_lo},{s_hi}]  V:[{v_lo},{v_hi}]\n")
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
