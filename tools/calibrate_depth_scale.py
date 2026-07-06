"""
calibrate_depth_scale.py — Calibra o modelo scale(d) = A + B*d do RealSense.

Usa uma trena laser para obter distâncias de referência precisas e compara
com as leituras brutas do sensor, ajustando um modelo linear de escala.

Com --update-markers, regista também a posição 3D de cada ArUco medido:
  y = camera_y + d_laser     (calculado automaticamente)
  x = x_borda_esq + 0.095m  (borda esquerda do marker → centro)
  z = z_base      + 0.095m  (base do marker → centro)

Procedimento (RPi5 com monitor):
  1. Colocar alvo (papel A4 branco) a distâncias Y conhecidas do corredor.
     Sugestão: centrar cada medição num ArUco para calibrar escala E posição.
  2. Mover cursor ao centro do alvo — profundidade actualiza em tempo real
  3. ESPAÇO ou clique esquerdo → amostra 30 frames (mediana)
  4. Terminal: introduzir distância medida pela trena laser em metros
     Com --update-markers: também pede ID do ArUco, x e z_base
  5. Repetir para 2-7 alvos (mínimo 2 para escala)
  6. F → calcula A, B, imprime resultado, pergunta se guarda no YAML
  7. Q → sair sem guardar

Uso:
    conda activate rpi5-realsense
    python tools/calibrate_depth_scale.py
    python tools/calibrate_depth_scale.py --save
    python tools/calibrate_depth_scale.py --update-markers
    python tools/calibrate_depth_scale.py --update-markers --save
"""

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml

YAML_PATH    = Path(__file__).parent.parent / "config" / "corridors.yaml"
SAMPLE_N     = 30    # frames por ponto de calibração
PATCH_HALF   = 3     # amostra patch 7×7 (pixels) à volta do cursor
MARKER_HALF  = 0.095 # metade de 190mm → centro do marker a partir da base

# ── YAML ──────────────────────────────────────────────────────────────────────
def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def update_yaml(path: Path, A: float, B: float) -> None:
    """Actualiza depth_scale_A e depth_scale_B no YAML preservando o resto."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    text = re.sub(r"(depth_scale_A:\s*)[\d.]+", f"\\g<1>{A:.4f}", text)
    text = re.sub(r"(depth_scale_B:\s*)[\d.]+", f"\\g<1>{B:.6f}", text)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def update_marker_positions(path: Path, measurements: dict) -> list:
    """
    Actualiza pos: [x, y, z] de cada marker medido em todos os corredores.
    measurements: {marker_id: [x, y, z]}
    Devolve lista de IDs actualizados.
    """
    with open(path, encoding="utf-8") as f:
        text = f.read()

    updated = []
    for mid, pos in measurements.items():
        x, y, z = pos
        # Substitui pos em todas as ocorrências do marker (3 corredores)
        pattern = rf"(- id:\s*{mid}[ \t]*\n[ \t]+pos:\s*\[)[^\]]*(\])"
        new_pos = rf"\g<1>{x:.3f}, {y:.3f}, {z:.3f}\2"
        new_text, n = re.subn(pattern, new_pos, text)
        if n > 0:
            text = new_text
            updated.append((mid, n, pos))

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return updated


# ── Cálculo ───────────────────────────────────────────────────────────────────
def get_depth(depth_frame, px: int, py: int) -> float:
    """Mediana de patch 7×7 à volta de (px, py). Devolve 0 se sem dados."""
    samples = []
    for dy in range(-PATCH_HALF, PATCH_HALF + 1):
        for dx in range(-PATCH_HALF, PATCH_HALF + 1):
            xi = int(np.clip(px + dx, 0, 639))
            yi = int(np.clip(py + dy, 0, 479))
            d  = depth_frame.get_distance(xi, yi)
            if 0.2 < d < 12.0:
                samples.append(d)
    return float(np.median(samples)) if samples else 0.0


def fit_model(pts: list) -> tuple:
    """
    pts: [(d_rs, d_laser), ...]
    Ajusta scale(d) = A + B*d por mínimos quadrados.
    Devolve (A, B, r2).
    """
    d_rs    = np.array([p[0] for p in pts])
    d_laser = np.array([p[1] for p in pts])
    scale   = d_laser / d_rs

    if len(pts) == 1:
        return float(scale[0]), 0.0, float("nan")

    X = np.column_stack([np.ones(len(pts)), d_rs])
    coef, *_ = np.linalg.lstsq(X, scale, rcond=None)
    A, B = float(coef[0]), float(coef[1])

    ss_res = float(np.sum((scale - (A + B * d_rs)) ** 2))
    ss_tot = float(np.sum((scale - scale.mean()) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return A, B, r2


# ── HUD ───────────────────────────────────────────────────────────────────────
_F     = cv2.FONT_HERSHEY_SIMPLEX
_WHITE = (255, 255, 255)
_GREEN = (80, 255, 80)
_CYAN  = (255, 220, 60)
_GRAY  = (160, 160, 160)
_DARK  = (0, 45, 0)


def _put(img, text, pos, scale=0.52, color=_WHITE, thick=1):
    cv2.putText(img, text, pos, _F, scale, color, thick, cv2.LINE_AA)


def draw_hud(frame, cursor, d_live, calib_pts, state, progress, A, B, r2,
             marker_meas=None):
    h, w = frame.shape[:2]
    cx, cy = cursor

    # Crosshair
    cv2.drawMarker(frame, (cx, cy), _CYAN, cv2.MARKER_CROSS, 28, 1)
    d_str = f"{d_live:.3f} m" if d_live > 0.01 else "sem depth"
    _put(frame, d_str, (cx + 16, cy - 10), scale=0.65, color=_CYAN, thick=2)

    # Barra topo
    cv2.rectangle(frame, (0, 0), (w, 38), _DARK, -1)
    if state == "LIVE":
        _put(frame,
             "ESPACO/clique=amostrar  |  F=finalizar e calcular  |  Q=sair",
             (8, 26), color=_GREEN)
    elif state == "SAMPLING":
        pct = int(progress * 100)
        _put(frame, f"A amostrar...  {pct}%  nao mover cursor", (8, 26),
             color=_CYAN, thick=2)
    elif state == "INPUT":
        _put(frame, "Introduzir distancia no TERMINAL  (janela pode ficar parada)",
             (8, 26), color=(120, 200, 255), thick=2)

    # Tabela de pontos
    y0 = 52
    _put(frame, "Pontos calibrados:", (8, y0), color=_WHITE, scale=0.48)
    if not calib_pts:
        _put(frame, "(nenhum — colocar alvo e premir ESPACO)", (12, y0 + 18),
             color=_GRAY, scale=0.44)
    for i, (d_rs, d_laser) in enumerate(calib_pts):
        sc_i   = d_laser / d_rs
        d_pred = d_rs * (A + B * d_rs) if not np.isnan(r2) else d_rs
        err_cm = (d_pred - d_laser) * 100
        col    = _GREEN if abs(err_cm) < 5 else _CYAN
        _put(frame,
             f"  {i+1}. d_sensor={d_rs:.3f}m  laser={d_laser:.3f}m"
             f"  scale={sc_i:.4f}  residuo={err_cm:+.1f}cm",
             (8, y0 + 18 * (i + 1)), scale=0.42, color=col)

    # Markers medidos
    if marker_meas:
        ids_str = " ".join(str(mid) for mid in sorted(marker_meas))
        _put(frame, f"Markers: {ids_str}",
             (8, h - 34), scale=0.44, color=_CYAN)

    # Modelo actual
    if len(calib_pts) >= 2:
        r2_s = f"{r2:.4f}" if not np.isnan(r2) else "n/a"
        _put(frame,
             f"scale(d) = {A:.4f} + {B:.6f}*d    R2={r2_s}",
             (8, h - 18), scale=0.50, color=_GREEN)
        _put(frame,
             f"scale @ 2.5m={A+B*2.5:.3f}  5m={A+B*5:.3f}  7.5m={A+B*7.5:.3f}",
             (8, h - 4), scale=0.40, color=_GRAY)

    # Barra de progresso (sampling)
    if state == "SAMPLING" and progress > 0:
        bw = int((w - 16) * progress)
        cv2.rectangle(frame, (8, h - 6), (8 + bw, h - 1), (80, 200, 255), -1)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true",
                        help="Guarda A/B (e posições de markers) sem confirmação")
    parser.add_argument("--update-markers", action="store_true",
                        help="Pede ID+x+z de cada ArUco medido e actualiza pos no YAML")
    args = parser.parse_args()

    # RealSense
    pipeline = rs.pipeline()
    cfg_rs   = rs.config()
    cfg_rs.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg_rs.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)
    pipeline.start(cfg_rs)
    align = rs.align(rs.stream.color)
    print("[OK] RealSense 640×480 @ 30fps")
    print("     Colocar alvo branco em frente à câmara e seguir instruções.\n")

    WIN = "Calibração depth_scale — ESPACO=amostrar  F=finalizar  Q=sair"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 960, 540)

    cfg_yaml = load_yaml(YAML_PATH)
    cam_y    = float(cfg_yaml.get("shared_geometry", {}).get("camera_y_m", -2.50))

    # Estado mutável
    cursor          = [320, 240]
    state           = ["LIVE"]        # lista para permitir mutação em closure
    depth_buf       : list[float] = []
    calib_pts       : list[tuple]  = []
    marker_meas     : dict[int, list[float]] = {}   # {id: [x, y, z]}
    A, B, r2        = 1.0, 0.0, float("nan")

    def on_mouse(event, x, y, _flags, _param):
        cursor[0], cursor[1] = x, y
        if event == cv2.EVENT_LBUTTONDOWN and state[0] == "LIVE":
            state[0] = "SAMPLING"
            depth_buf.clear()

    cv2.setMouseCallback(WIN, on_mouse)

    d_live = 0.0

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError:
                continue

            aligned = align.process(frames)
            color_f = aligned.get_color_frame()
            depth_f = aligned.get_depth_frame()
            if not color_f:
                continue

            frame = np.asanyarray(color_f.get_data()).copy()

            if depth_f:
                d_live = get_depth(depth_f, cursor[0], cursor[1])

            # ── SAMPLING ──────────────────────────────────────────────────
            if state[0] == "SAMPLING":
                if depth_f and d_live > 0.01:
                    depth_buf.append(d_live)
                progress = len(depth_buf) / SAMPLE_N
                if len(depth_buf) >= SAMPLE_N:
                    state[0] = "INPUT"
            else:
                progress = 0.0

            # ── INPUT: bloqueia para leitura do terminal ───────────────────
            if state[0] == "INPUT":
                draw_hud(frame, tuple(cursor), d_live, calib_pts,
                         "INPUT", 1.0, A, B, r2, marker_meas)
                cv2.imshow(WIN, frame)
                cv2.waitKey(1)

                d_rs_med = float(np.median(depth_buf))
                print(f"\n[Amostrado] profundidade RealSense: {d_rs_med:.4f} m"
                      f"  ({len(depth_buf)} amostras, mediana)")
                while True:
                    try:
                        raw = input("  Distância trena laser (m): ").strip().replace(",", ".")
                        d_laser = float(raw)
                        if d_laser > 0.1:
                            break
                        print("  → valor deve ser > 0")
                    except ValueError:
                        print("  → formato inválido, ex: 2.50")

                calib_pts.append((d_rs_med, d_laser))
                sc_i = d_laser / d_rs_med
                print(f"  scale = {d_laser:.4f} / {d_rs_med:.4f} = {sc_i:.4f}")

                if len(calib_pts) >= 2:
                    A, B, r2 = fit_model(calib_pts)
                    print(f"  Modelo: A={A:.4f}  B={B:.6f}  R²={r2:.4f}")

                # ── Posição do marker (opcional) ──────────────────────────
                if args.update_markers:
                    raw_id = input(
                        "  ArUco ID para registar posição (Enter=ignorar): "
                    ).strip()
                    if raw_id.isdigit():
                        mid = int(raw_id)
                        # y calculado da câmara + profundidade laser
                        y_m = cam_y + d_laser
                        print(f"  y calculado: {cam_y:.3f} + {d_laser:.3f} = {y_m:.3f} m")
                        while True:
                            try:
                                x_borda = float(
                                    input("  x da borda ESQUERDA do marker (m): ")
                                    .strip().replace(",", ".")
                                )
                                break
                            except ValueError:
                                print("  → ex: 1.185")
                        while True:
                            try:
                                z_base = float(
                                    input("  z do CHÃO até base do marker (m): ")
                                    .strip().replace(",", ".")
                                )
                                break
                            except ValueError:
                                print("  → ex: 1.805")
                        x_m = x_borda + MARKER_HALF
                        z_m = z_base  + MARKER_HALF
                        marker_meas[mid] = [x_m, y_m, z_m]
                        print(f"  → ID {mid}  pos=[{x_m:.3f}, {y_m:.3f}, {z_m:.3f}]"
                              f"  (x_borda={x_borda:.3f}+{MARKER_HALF},"
                              f" z_base={z_base:.3f}+{MARKER_HALF})")

                state[0] = "LIVE"
                depth_buf.clear()

            # ── Render ────────────────────────────────────────────────────
            draw_hud(frame, tuple(cursor), d_live, calib_pts,
                     state[0], progress, A, B, r2, marker_meas)
            cv2.imshow(WIN, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" ") and state[0] == "LIVE":
                state[0] = "SAMPLING"
                depth_buf.clear()
            elif key in (ord("q"), ord("Q")):
                print("\nSaindo sem guardar.")
                return
            elif key in (ord("f"), ord("F")):
                if len(calib_pts) < 2:
                    print("\n[AVISO] Precisa de pelo menos 2 pontos.")
                else:
                    break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    if len(calib_pts) < 2:
        return

    # ── Resultado ─────────────────────────────────────────────────────────────
    A, B, r2 = fit_model(calib_pts)
    cfg      = load_yaml(YAML_PATH)
    sg       = cfg.get("shared_geometry", {})
    cam_y    = float(sg.get("camera_y_m", -2.50))

    print(f"\n{'═'*58}")
    print(f"  RESULTADO DA CALIBRAÇÃO")
    print(f"{'─'*58}")
    print(f"  Modelo: scale(d) = {A:.4f} + {B:.6f} × d")
    r2_s = f"{r2:.4f}" if not np.isnan(r2) else "n/a"
    print(f"  R² = {r2_s}   ({len(calib_pts)} pontos)")
    print(f"{'─'*58}")
    print(f"  {'d_sensor':>10}  {'laser':>8}  {'scale':>8}  {'pred':>8}  {'erro':>8}")
    for d_rs, d_laser in calib_pts:
        sc_i   = d_laser / d_rs
        d_pred = d_rs * (A + B * d_rs)
        err    = (d_pred - d_laser) * 100
        print(f"  {d_rs:10.4f}  {d_laser:8.4f}  {sc_i:8.4f}  {d_pred:8.4f}  {err:+7.1f} cm")
    print(f"{'─'*58}")
    print(f"  scale(2.5m) = {A + B*2.5:.4f}")
    print(f"  scale(5.0m) = {A + B*5.0:.4f}")
    print(f"  scale(7.5m) = {A + B*7.5:.4f}")
    print(f"{'─'*58}")
    print(f"  YAML → shared_geometry:")
    print(f"    depth_scale_A: {A:.4f}")
    print(f"    depth_scale_B: {B:.6f}")
    print(f"{'═'*58}\n")

    # Posições corrigidas dos alvos (verificação)
    print("  Verificação — posição Y dos alvos com modelo calibrado:")
    for d_rs, d_laser in calib_pts:
        y_pred = cam_y + d_rs * (A + B * d_rs)
        y_true = cam_y + d_laser
        print(f"    d_sensor={d_rs:.3f}m  y_pred={y_pred:.3f}m  y_true={y_true:.3f}m"
              f"  diff={abs(y_pred-y_true)*100:.1f}cm")
    print()

    # ── Posições de markers medidas ───────────────────────────────────────────
    if marker_meas:
        print(f"\n{'─'*58}")
        print(f"  POSIÇÕES DE MARKERS MEDIDAS  ({len(marker_meas)} markers)")
        print(f"{'─'*58}")
        for mid, pos in sorted(marker_meas.items()):
            print(f"  ID {mid:2d}  →  pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        print(f"{'─'*58}")

    # Guardar
    if args.save:
        update_yaml(YAML_PATH, A, B)
        print(f"[SALVO] escala → {YAML_PATH.name}")
        if marker_meas:
            updated = update_marker_positions(YAML_PATH, marker_meas)
            for mid, n, pos in updated:
                print(f"[SALVO] marker {mid}  pos={[round(v,3) for v in pos]}"
                      f"  ({n} ocorrências)")
    else:
        resp = input("Guardar A/B em corridors.yaml? (s/n): ").strip().lower()
        if resp in ("s", "sim", "y", "yes"):
            update_yaml(YAML_PATH, A, B)
            print(f"[SALVO] escala → {YAML_PATH.name}")
        else:
            print("Não guardado — copiar manualmente os valores acima.")

        if marker_meas:
            resp2 = input("Guardar posições dos markers no YAML? (s/n): ").strip().lower()
            if resp2 in ("s", "sim", "y", "yes"):
                updated = update_marker_positions(YAML_PATH, marker_meas)
                for mid, n, pos in updated:
                    print(f"[SALVO] marker {mid}  pos={[round(v,3) for v in pos]}"
                          f"  ({n} ocorrências)")
            else:
                print("Posições não guardadas — copiar manualmente os valores acima.")


if __name__ == "__main__":
    main()
