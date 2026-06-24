"""
level_camera.py — Nivelamento ortogonal da câmera guiado pelo IMU da RealSense D435i.

O acelerômetro em repouso mede a gravidade. Quando a câmera está
perfeitamente nadir (apontando para o chão), a gravidade fica alinhada
com um eixo fixo. Esta ferramenta mostra o desvio em graus e uma
bolha de nível virtual para guiar o ajuste do suporte.

Uso:
    source ~/realsense-env/bin/activate
    python tools/level_camera.py

Teclas:
    C — define orientação atual como referência de "zero"
         (pressione quando estiver próximo do nadir desejado)
    Q — sai
    S — salva frame atual como imagem de referência

Workflow:
    1. Monte a câmera apontando aproximadamente para baixo
    2. Execute este script — a bolha mostra o desvio
    3. Ajuste o suporte até bolha centrada e desvio < 1°
    4. Pressione C para confirmar zero
    5. Mínimo ajuste fino posterior sem precisar abrir o script
"""

import cv2
import numpy as np
import pyrealsense2 as rs
from collections import deque
from pathlib import Path

# ── Configuração ──────────────────────────────────────────────────────────────
FILTER_LEN   = 30    # janela do filtro passa-baixa (frames de IMU ~200Hz → ~0.15s)
LEVEL_OK_DEG = 1.5   # desvio abaixo deste valor → "nivelada" (verde)
BUBBLE_RANGE = 60    # raio do display de bolha em pixels
SCALE_DEG    = 5.0   # ângulo que leva a bolha até a borda (graus)

# ── Pipeline RealSense ────────────────────────────────────────────────────────
pipeline  = rs.pipeline()
cfg       = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8,          30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,           30)
cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f)

profile   = pipeline.start(cfg)
colorizer = rs.colorizer()

# ── Estado ────────────────────────────────────────────────────────────────────
accel_buf   = deque(maxlen=FILTER_LEN)
ref_gravity = None   # gravidade de referência (nadir desejado)

print("Aponte a câmera para baixo e pressione C para definir o zero.")
print("Q = sair | S = salvar frame")


def orthonormal_basis(v):
    """Retorna dois vetores ortogonais a v (para projetar o erro de inclinação)."""
    v = v / np.linalg.norm(v)
    # escolhe eixo auxiliar não paralelo a v
    aux = np.array([1.0, 0.0, 0.0]) if abs(v[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1  = np.cross(v, aux);  e1 /= np.linalg.norm(e1)
    e2  = np.cross(v, e1);   e2 /= np.linalg.norm(e2)
    return e1, e2


def draw_level(frame, g_curr, g_ref):
    """Desenha bolha de nível e anotações de ângulo sobre o frame."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    g_curr_n = g_curr / np.linalg.norm(g_curr)
    g_ref_n  = g_ref  / np.linalg.norm(g_ref)

    # Inclinação total
    dot        = float(np.clip(np.dot(g_curr_n, g_ref_n), -1, 1))
    tilt_total = float(np.degrees(np.arccos(dot)))

    # Projeção do erro nas duas direções perpendiculares ao nadir
    e1, e2   = orthonormal_basis(g_ref_n)
    err      = g_curr_n - g_ref_n
    angle_e1 = float(np.degrees(np.arcsin(np.clip(np.dot(err, e1), -1, 1))))
    angle_e2 = float(np.degrees(np.arcsin(np.clip(np.dot(err, e2), -1, 1))))

    # Posição da bolha
    scale = BUBBLE_RANGE / SCALE_DEG
    bx    = int(cx + angle_e1 * scale)
    by    = int(cy + angle_e2 * scale)
    bx    = max(cx - BUBBLE_RANGE, min(cx + BUBBLE_RANGE, bx))
    by    = max(cy - BUBBLE_RANGE, min(cy + BUBBLE_RANGE, by))

    is_level  = tilt_total < LEVEL_OK_DEG
    ok_color  = (0, 220, 80)
    bad_color = (0, 80, 230)
    bubble_c  = ok_color if is_level else bad_color
    ring_c    = ok_color if is_level else (80, 80, 200)

    # Círculo externo e cruz
    cv2.circle(frame, (cx, cy), BUBBLE_RANGE, ring_c, 2)
    cv2.circle(frame, (cx, cy), 4, ring_c, -1)
    cv2.line(frame, (cx - BUBBLE_RANGE, cy), (cx + BUBBLE_RANGE, cy), ring_c, 1)
    cv2.line(frame, (cx, cy - BUBBLE_RANGE), (cx, cy + BUBBLE_RANGE), ring_c, 1)

    # Bolha
    cv2.circle(frame, (bx, by), 16, bubble_c, -1)
    cv2.circle(frame, (bx, by), 16, (255, 255, 255), 1)

    # HUD de texto
    status     = "NIVELADA ✓" if is_level else "AJUSTAR"
    status_clr = ok_color if is_level else bad_color
    info = [
        (f"Desvio total : {tilt_total:+.2f} graus", (220, 220, 220)),
        (f"Eixo lateral : {angle_e1:+.2f} graus",   (180, 200, 255)),
        (f"Eixo frontal : {angle_e2:+.2f} graus",   (255, 200, 140)),
        (f"Status       : {status}",                  status_clr),
        ("C=zero  S=salvar  Q=sair",                  (140, 140, 140)),
    ]
    for i, (txt, col) in enumerate(info):
        cv2.putText(frame, txt, (10, 28 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)

    return tilt_total, angle_e1, angle_e2


# ── Loop principal ────────────────────────────────────────────────────────────
save_dir = Path("output")
save_dir.mkdir(exist_ok=True)
save_idx  = 0

try:
    while True:
        frames = pipeline.wait_for_frames()

        # IMU (acelerômetro)
        accel_f = frames.first_or_default(rs.stream.accel)
        if accel_f:
            a = accel_f.as_motion_frame().get_motion_data()
            accel_buf.append(np.array([a.x, a.y, a.z]))

        color_f = frames.get_color_frame()
        depth_f = frames.get_depth_frame()
        if not color_f:
            continue

        color = np.asanyarray(color_f.get_data()).copy()
        depth = (np.asanyarray(colorizer.colorize(depth_f).get_data())
                 if depth_f else np.zeros_like(color))

        # Gravidade filtrada (média das últimas amostras)
        if len(accel_buf) >= 5:
            gravity = np.mean(accel_buf, axis=0)

            # Define referência automática na primeira leitura estável
            if ref_gravity is None:
                ref_gravity = gravity / np.linalg.norm(gravity)
                print(f"  Referência inicial automática: {ref_gravity}")

            draw_level(color, gravity, ref_gravity)
        else:
            cv2.putText(color, "Aguardando IMU...", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

        # Exibe RGB + depth lado a lado
        combined = np.hstack([color, depth])
        cv2.imshow("Level Camera — IMU RealSense", combined)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('c') and len(accel_buf) >= 5:
            gravity     = np.mean(accel_buf, axis=0)
            ref_gravity = gravity / np.linalg.norm(gravity)
            print(f"  Zero redefinido → gravidade de referência: {ref_gravity.round(4)}")

        elif key == ord('s'):
            save_idx += 1
            fname = save_dir / f"level_ref_{save_idx:03d}.png"
            cv2.imwrite(str(fname), combined)
            print(f"  Salvo: {fname}")

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
