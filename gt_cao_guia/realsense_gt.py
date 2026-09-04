#!/usr/bin/env python3
"""
realsense_gt.py - Ground truth com RealSense FIXA + ArUco: BOLA VERDE e/ou PESSOA (YOLO).

Versao corrigida do realsense_aruco_standalone.py. Rastreia os dois alvos na MESMA passada,
no mesmo referencial, para dar pra comparar o metodo da bola com o do bbox nos mesmos dados.

O QUE MUDOU EM RELACAO AO ORIGINAL
----------------------------------
Referencial:
  * POSE TRAVADA. A camera e fixa -> a pose do ArUco e estimada UMA vez (mediana de N
    amostras) e nunca mais recalculada. O original refazia todo frame, injetando ruido.
  * MARCADOR DE REFERENCIA EXPLICITO (--ref-id). O original usava rvecs[0] = "o primeiro
    que a deteccao devolveu", que podia trocar de marcador (e de origem) no meio da sessao.
  * --marker-size na linha de comando + CONFERENCIA COM A TRENA (--cam-height). Erro de
    escala aqui desloca TODA a trajetoria rigidamente; o script mede e diz o valor correto.
  * --level: com a camera nivelada, forca a vertical do mundo (remove o pior grau de
    liberdade de um marcador rasante no chao).
  * solvePnP + IPPE_SQUARE em vez de estimatePoseSingleMarkers (removido no OpenCV >= 4.7).

Ponto 3D da pessoa (era a maior fonte de erro):
  * O original lia UM pixel de profundidade no centro do bbox. Aqui: amostra uma grade na
    FAIXA DO TRONCO do bbox, fica com a superficie mais PROXIMA coerente (= a pessoa,
    rejeitando fundo por construcao) e usa a mediana + o centroide dos pixels aceitos.
  * Nao depende dos pes, que costumam estar ocluidos ou fora de quadro.

Tracking:
  * So grava tracks confirmados (>= --min-hits) e ATUALIZADOS no frame. O original gravava
    todos os tracks ativos todo frame, incluindo os "perdidos", congelados na ultima posicao.
  * Gate de velocidade: rejeita associacao que exija velocidade impossivel.
  * Associacao caixa<->track por dicionario (o original indexava um set pela posicao de um
    elemento em outro set -> desenhava a caixa no track errado).

CSV (colunas do original preservadas + o que faltava para reprocessar depois):
  * p_cam_x/y/z  - o ponto no frame da CAMERA, ANTES da transformacao para o mundo.
    Com isso, se a pose/escala do ArUco estiver errada, da pra recalcular tudo offline
    sem repetir a coleta. E a coluna mais importante que faltava.
  * u_px/v_px/depth_m/n_px/depth_std - de onde saiu a medida e o quao confiavel ela e.
  * bbox + clipped - permite refazer a extracao do ponto do zero, offline.
  * wall (epoch) - alinhamento com a camera do teto.
  * sidecar <trial>.meta.json com pose travada, intrinsecos e parametros da sessao.

Uso:
    python realsense_gt.py --ref-id 0 --marker-size 0.15 --cam-height 0.78 --level
    python realsense_gt.py --track ball --prefix P01
    python realsense_gt.py --track person --every-n 2
    python realsense_gt.py --bag sessao.bag          # grava tudo p/ reprocessar depois

Teclas:
    ESPACO - inicia/encerra uma PASSADA (1 CSV + 1 sidecar por passada)
    B      - overlay da mascara HSV (ajuste da bola)
    R      - refaz o travamento da pose
    Q      - sai
"""
import argparse, csv, json, os, time
from datetime import datetime

import cv2
import numpy as np
import pyrealsense2 as rs

DICT_TYPE = cv2.aruco.DICT_4X4_50
DEPTH_MIN, DEPTH_MAX = 0.2, 12.0
PATCH = 2                    # patch (2*PATCH+1)^2 para a mediana de profundidade da bola


def git_commit():
    """Commit do codigo que gerou os dados. No ciclo remoto (eu corrijo -> push -> voces
    pull -> testam) sem isso a gente depura a versao errada."""
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "?"


# ============================ ArUco / referencial ============================
def build_detector():
    try:
        d = cv2.aruco.getPredefinedDictionary(DICT_TYPE)
        p = cv2.aruco.DetectorParameters()
        return ("new", cv2.aruco.ArucoDetector(d, p))
    except AttributeError:
        d = cv2.aruco.Dictionary_get(DICT_TYPE)
        p = cv2.aruco.DetectorParameters_create()
        return ("old", (d, p))


def detect_markers(det, gray):
    kind, obj = det
    if kind == "new":
        c, i, _ = obj.detectMarkers(gray)
    else:
        d, p = obj
        c, i, _ = cv2.aruco.detectMarkers(gray, d, parameters=p)
    return c, i


def marker_object_points(L):
    """Cantos na ordem que o detectMarkers devolve. L = lado do QUADRADO PRETO (inclui a
    borda preta grossa; NAO inclui a margem branca de silencio)."""
    h = L / 2.0
    return np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]])


def solve_marker_pose(corners, L, K, D):
    img = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    flag = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE)
    try:
        ok, rvec, tvec = cv2.solvePnP(marker_object_points(L), img, K, D, flags=flag)
    except cv2.error:
        ok, rvec, tvec = cv2.solvePnP(marker_object_points(L), img, K, D,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
    return (rvec.reshape(3), tvec.reshape(3)) if ok else None


def level_rotation(R, up_cam=np.array([0.0, -1.0, 0.0])):
    """Camera NIVELADA -> o 'para cima' no frame da camera e conhecido (-Y). Forca o eixo Z
    do mundo a ser a vertical verdadeira, preservando a guinada e o sentido do marcador."""
    s = 1.0 if float(R[:, 2] @ up_cam) >= 0 else -1.0
    zc = s * up_cam
    x = R[:, 0] - float(R[:, 0] @ zc) * zc
    n = np.linalg.norm(x)
    if n < 1e-6:
        return R
    x = x / n
    return np.column_stack([x, np.cross(zc, x), zc])


# ================================= Alvos =====================================
def detect_ball(bgr, lo, hi, min_area, min_radius):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lo, hi)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    best = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(best) < min_area:
        return None
    (bx, by), radius = cv2.minEnclosingCircle(best)
    if radius < min_radius:
        return None
    return int(bx), int(by), float(radius), mask


def depth_median_patch(depth_frame, px, py, W, H):
    s = []
    for dy in range(-PATCH, PATCH + 1):
        for dx in range(-PATCH, PATCH + 1):
            xi = int(np.clip(px + dx, 0, W - 1))
            yi = int(np.clip(py + dy, 0, H - 1))
            d = depth_frame.get_distance(xi, yi)
            if DEPTH_MIN < d < DEPTH_MAX:
                s.append(d)
    if not s:
        return None
    return float(np.median(s)), len(s), float(np.std(s))


def person_point(depth_frame, box, W, H, step=4, band=(0.20, 0.60),
                 width_frac=0.5, tol=0.25, hint=None):
    """Ponto 3D robusto da pessoa, sem depender dos pes (que costumam estar ocluidos ou
    fora de quadro).

    Amostra uma grade no TRONCO SUPERIOR do bbox, separa as profundidades em superficies
    (clusters separados por um vao > `tol`) e fica com a MAIOR delas - a pessoa ocupa a
    maior parte do recorte. Isso rejeita ao mesmo tempo:
      - o FUNDO, que aparece nas bordas do bbox (superficie minoritaria, mais longe);
      - um MOVEL na frente cobrindo parte do corpo (superficie minoritaria, mais perto).
    Pegar a superficie "mais proxima" falhava no segundo caso: travava no movel.

    `band` alto (tronco superior) porque a camera e baixa: movel/cadeira ocluem a parte de
    baixo primeiro. `hint` (profundidade do frame anterior) desempata a favor da
    continuidade quando dois clusters tem tamanho parecido.

    Retorna (u, v, depth, n_px, std) ou None."""
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if bw < 6 or bh < 12:
        return None
    cx = (x1 + x2) / 2.0
    hw = max(2.0, bw * width_frac / 2.0)
    xa, xb = int(max(0, cx - hw)), int(min(W - 1, cx + hw))
    ya = int(max(0, y1 + bh * band[0]))
    yb = int(min(H - 1, y1 + bh * band[1]))
    us, vs, ds = [], [], []
    for vv in range(ya, yb + 1, step):
        for uu in range(xa, xb + 1, step):
            d = depth_frame.get_distance(uu, vv)
            if DEPTH_MIN < d < DEPTH_MAX:
                us.append(uu); vs.append(vv); ds.append(d)
    if len(ds) < 8:
        return None
    ds = np.asarray(ds, float)
    order = np.argsort(ds)
    ds_sorted = ds[order]
    cuts = np.where(np.diff(ds_sorted) > tol)[0]          # vao entre superficies
    groups = np.split(np.arange(len(ds_sorted)), cuts + 1)
    if hint is not None:
        near = [gi for gi in groups if abs(float(np.median(ds_sorted[gi])) - hint) <= 0.6]
        best = max(near, key=len) if near else max(groups, key=len)
    else:
        best = max(groups, key=len)
    sel = order[best]
    if len(sel) < 6:
        return None
    return (float(np.median(np.asarray(us)[sel])),
            float(np.median(np.asarray(vs)[sel])),
            float(np.median(ds[sel])),
            int(len(sel)),
            float(ds[sel].std()))


# ================================ Tracking ===================================
class Track:
    _next = 0

    def __init__(self, pos, wall):
        self.id = Track._next; Track._next += 1
        self.pos = np.asarray(pos, float)
        self.wall = wall
        self.hits = 1
        self.misses = 0

    def update(self, pos, wall):
        self.pos = np.asarray(pos, float)
        self.wall = wall
        self.hits += 1
        self.misses = 0

    @property
    def confirmed_at(self):
        return self.hits


def associate(tracks, dets, wall, gate_m, max_speed):
    """Vizinho mais proximo guloso com gate de DISTANCIA e de VELOCIDADE. Devolve
    {idx_deteccao: track} dos casados. Sem o gate de velocidade, um salto de profundidade
    (bbox pegando o fundo) era aceito como movimento real."""
    pairs = []
    for di, d in enumerate(dets):
        for tk in tracks:
            dist = float(np.linalg.norm(np.asarray(d) - tk.pos))
            dt = max(1e-3, wall - tk.wall)
            if dist <= gate_m and dist / dt <= max_speed:
                pairs.append((dist, di, tk))
    pairs.sort(key=lambda p: p[0])
    used_d, used_t, out = set(), set(), {}
    for dist, di, tk in pairs:
        if di in used_d or tk.id in used_t:
            continue
        tk.update(dets[di], wall)
        used_d.add(di); used_t.add(tk.id); out[di] = tk
    for di, d in enumerate(dets):
        if di not in used_d:
            nt = Track(d, wall)
            tracks.append(nt); out[di] = nt
    for tk in tracks:
        if tk.id not in used_t and tk.wall != wall:
            tk.misses += 1
    return out


# ================================== Main =====================================
CSV_COLS = ["timestamp_iso", "track_id", "x_world", "y_world", "z_world",
            "target", "wall", "t_s", "frame",
            "p_cam_x", "p_cam_y", "p_cam_z",
            "u_px", "v_px", "depth_m", "n_px", "depth_std",
            "conf", "rad_px", "bx1", "by1", "bx2", "by2", "clipped",
            "hits", "misses"]


def main():
    ap = argparse.ArgumentParser(description="GT RealSense fixa + ArUco travado: bola e/ou pessoa")
    ap.add_argument("--track", choices=["ball", "person", "both"], default="both")
    ap.add_argument("--ref-id", type=int, default=None,
                    help="ID do marcador de REFERENCIA. Default: menor ID visivel.")
    ap.add_argument("--marker-size", type=float, default=0.15,
                    help="lado do QUADRADO PRETO em metros (NAO a margem branca)")
    ap.add_argument("--cam-height", type=float, default=None,
                    help="altura REAL da camera (m, trena) - o script confere a escala")
    ap.add_argument("--cam-dist", type=float, default=None,
                    help="distancia horizontal REAL camera->marcador (m, trena) - 2a conferencia")
    ap.add_argument("--level", action="store_true",
                    help="camera NIVELADA: forca a vertical do mundo")
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--hsv", default="35,50,50,85,255,255")
    ap.add_argument("--min-area", type=float, default=60.0)
    ap.add_argument("--min-radius", type=float, default=3.0)
    ap.add_argument("--ball-height", type=float, default=None,
                    help="altura MEDIDA da bola no capacete DESTE participante (m, trena). "
                         "Varia por pessoa. O script confere o z_world contra ela em tempo real "
                         "-> checagem independente da pose/escala do ArUco, de graca.")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--every-n", type=int, default=1, help="roda o YOLO a cada N frames")
    ap.add_argument("--gate-m", type=float, default=1.0, help="gate de associacao (m)")
    ap.add_argument("--max-speed", type=float, default=3.0, help="gate de velocidade (m/s)")
    ap.add_argument("--min-hits", type=int, default=3, help="frames antes de gravar um track")
    ap.add_argument("--max-misses", type=int, default=10)
    ap.add_argument("--rate", type=float, default=10.0, help="Hz de gravacao")
    ap.add_argument("--prefix", default="trial")
    ap.add_argument("--out", default="trajetorias")
    ap.add_argument("--bag", default=None,
                    help="grava a sessao inteira em .bag (reprocessavel; ~1-2 GB/min)")
    a = ap.parse_args()

    v = [int(x) for x in a.hsv.split(",")]
    HSV_LO, HSV_HI = np.array(v[:3], np.uint8), np.array(v[3:], np.uint8)
    os.makedirs(a.out, exist_ok=True)
    RECORD_DT = 1.0 / a.rate
    want_ball = a.track in ("ball", "both")
    want_person = a.track in ("person", "both")

    model = None
    if want_person:
        from ultralytics import YOLO
        here = os.path.dirname(os.path.abspath(__file__))
        mp = a.model if os.path.isabs(a.model) else os.path.join(here, a.model)
        mp = mp if os.path.exists(mp) else a.model
        print(f"[YOLO] carregando {mp}")
        model = YOLO(mp)

    det = build_detector()
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    if a.bag:
        cfg.enable_record_to_file(a.bag)
        print(f"[BAG] gravando a sessao em {a.bag}")
    profile = pipeline.start(cfg)
    align = rs.align(rs.stream.color)
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], float)
    DIST = np.array(intr.coeffs, float)
    W, H = intr.width, intr.height
    vfov = 2 * np.degrees(np.arctan((H / 2.0) / intr.fy))
    print(f"[versao] commit {git_commit()}")
    print(f"[OK] {W}x{H}  fx={intr.fx:.1f} fy={intr.fy:.1f}  FOV vertical={vfov:.1f} deg")
    print(f"     alvo={a.track}  marcador={a.marker_size*100:.1f} cm  nivelado={'SIM' if a.level else 'nao'}")
    if want_ball and a.cam_height:
        hb = a.ball_height if a.ball_height else 1.78
        dz = abs(hb - a.cam_height)
        dmin = dz / np.tan(np.radians(vfov / 2.0))
        onde = "topo" if hb > a.cam_height else "base"
        print(f"     FOV: bola a {hb:.2f} m, camera a {a.cam_height:.2f} m (desnivel {dz:.2f} m)")
        print(f"          -> a bola sai pelo {onde} do quadro a menos de {dmin:.2f} m da camera")
        if dmin > 1.5:
            print(f"          *** SUBA A CAMERA. Na altura da bola ({hb:.2f} m) o problema some;")
            print(f"          *** no MEIO da faixa de alturas dos participantes ele fica minimo.")

    WIN = "GT - ESPACO=passada  B=mask  R=re-travar  Q=sair"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL); cv2.resizeWindow(WIN, 960, 720)

    locked, ref_id, samples = False, a.ref_id, []
    R_wc = t_wc = cam_pos = None
    lock_info = {}
    show_mask, rec, trial_n, frame_i = False, False, 0, 0
    rows, t0, last_rec, last_log = [], 0.0, 0.0, 0.0
    tracks = []
    last_person_d = None      # profundidade da pessoa no frame anterior (continuidade)

    def to_world(p_cam):
        return R_wc.T @ (p_cam - t_wc)

    def close_trial():
        nonlocal rec, rows
        rec = False
        if not rows:
            print("  (0 pontos - nada salvo)"); return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join(a.out, f"{a.prefix}_{trial_n:02d}_{ts}")
        with open(base + ".csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLS)
            w.writeheader(); w.writerows(rows)
        meta = dict(lock_info)
        meta.update({
            "code_commit": git_commit(),
            "trial": trial_n, "n_rows": len(rows), "wall_start": rows[0]["wall"],
            "wall_end": rows[-1]["wall"], "track_mode": a.track,
            "intrinsics": {"width": W, "height": H, "fx": intr.fx, "fy": intr.fy,
                           "ppx": intr.ppx, "ppy": intr.ppy, "coeffs": list(intr.coeffs)},
            "params": {"marker_size_m": a.marker_size, "ref_id": ref_id, "level": a.level,
                       "hsv": a.hsv, "conf": a.conf, "gate_m": a.gate_m,
                       "max_speed": a.max_speed, "min_hits": a.min_hits, "rate_hz": a.rate,
                       "ball_height_m": a.ball_height, "cam_height_m": a.cam_height,
                       "cam_dist_m": a.cam_dist},
            "bag": a.bag,
        })
        with open(base + ".meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        n_b = sum(1 for r in rows if r["target"] == "ball")
        print(f"  [CSV] {len(rows)} linhas (bola {n_b} / pessoa {len(rows)-n_b}) -> {base}.csv")
        print(f"  [META] {base}.meta.json")
        rows = []

    def add_row(target, tid, world, p_cam, u, v, d, npx, std, wall,
                conf="", rad="", box=None, clip="", hits="", miss=""):
        rows.append({
            "timestamp_iso": datetime.now().isoformat(), "track_id": tid,
            "x_world": f"{world[0]:.4f}", "y_world": f"{world[1]:.4f}", "z_world": f"{world[2]:.4f}",
            "target": target, "wall": f"{wall:.3f}", "t_s": f"{wall - t0:.3f}", "frame": frame_i,
            "p_cam_x": f"{p_cam[0]:.4f}", "p_cam_y": f"{p_cam[1]:.4f}", "p_cam_z": f"{p_cam[2]:.4f}",
            "u_px": f"{u:.1f}", "v_px": f"{v:.1f}", "depth_m": f"{d:.4f}",
            "n_px": npx, "depth_std": f"{std:.4f}",
            "conf": conf, "rad_px": rad,
            "bx1": box[0] if box else "", "by1": box[1] if box else "",
            "bx2": box[2] if box else "", "by2": box[3] if box else "",
            "clipped": clip, "hits": hits, "misses": miss,
        })

    try:
        while True:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            aligned = align.process(frames)
            color_f, depth_f = aligned.get_color_frame(), aligned.get_depth_frame()
            if not color_f or not depth_f:
                continue
            frame = np.asanyarray(color_f.get_data()).copy()
            now = time.time()
            frame_i += 1

            # ---------------- FASE 1: travar a pose ----------------
            if not locked:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids = detect_markers(det, gray)
                seen = None
                if ids is not None and len(ids):
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                    flat = ids.flatten().tolist()
                    if ref_id is None:
                        ref_id = min(flat)
                        print(f"[ref] marcador de referencia = ID {ref_id}")
                    if ref_id in flat:
                        seen = corners[flat.index(ref_id)][0]
                if seen is not None:
                    pose = solve_marker_pose(seen, a.marker_size, K, DIST)
                    if pose is not None:
                        samples.append(pose)
                        if len(samples) >= a.warmup:
                            rv = np.median(np.array([s[0] for s in samples]), axis=0)
                            tv = np.median(np.array([s[1] for s in samples]), axis=0)
                            R_wc, _ = cv2.Rodrigues(rv)
                            ang = np.degrees(np.arccos(np.clip(
                                abs(float(R_wc[:, 2] @ np.array([0.0, -1.0, 0.0]))), -1, 1)))
                            leveled = False
                            if a.level:
                                if ang > 20:
                                    print(f"[AVISO] eixo Z do marcador a {ang:.0f} deg da vertical "
                                          f"da camera -> --level IGNORADO (camera nao nivelada ou "
                                          f"marcador nao esta no chao).")
                                else:
                                    R_wc = level_rotation(R_wc); leveled = True
                                    print(f"[level] vertical forcada (corrigiu {ang:.1f} deg)")
                            t_wc = tv
                            cam_pos = -R_wc.T @ t_wc
                            locked = True
                            d_h = float(np.hypot(cam_pos[0], cam_pos[1]))
                            print("\n" + "=" * 74)
                            print(f"[TRAVADO] {len(samples)} amostras do marcador {ref_id}")
                            print(f"  CAMERA no frame do marcador: "
                                  f"({cam_pos[0]:+.3f}, {cam_pos[1]:+.3f}, {cam_pos[2]:+.3f}) m")
                            print(f"  -> distancia horizontal ate o marcador : {d_h:.3f} m")
                            print(f"  -> ALTURA da camera                    : {cam_pos[2]:+.3f} m")
                            s_scale = None
                            if a.cam_height:
                                s_scale = abs(cam_pos[2]) / a.cam_height
                                print(f"  trena altura {a.cam_height:.3f} m -> s = {s_scale:.3f}")
                            if a.cam_dist:
                                s2 = d_h / a.cam_dist
                                print(f"  trena distancia {a.cam_dist:.3f} m -> s = {s2:.3f}")
                                s_scale = s_scale if s_scale else s2
                            if s_scale:
                                if abs(s_scale - 1) > 0.05:
                                    desl = abs(1 - 1 / s_scale) * float(np.linalg.norm(cam_pos))
                                    print(f"  *** ESCALA ERRADA -> rode com "
                                          f"--marker-size {a.marker_size / s_scale:.4f}")
                                    print(f"  *** deslocamento atual de TODA a trajetoria: {desl*100:.0f} cm")
                                else:
                                    print("  OK - escala confere.")
                            else:
                                print("  >> passe --cam-height (e --cam-dist) p/ conferir a escala!")
                            print("=" * 74 + "\n")
                            lock_info = {
                                "ref_id": int(ref_id), "warmup": len(samples),
                                "rvec": rv.tolist(), "tvec": tv.tolist(),
                                "R_marker_from_cam": R_wc.tolist(),
                                "cam_pos_in_marker": cam_pos.tolist(),
                                "cam_dist_horizontal_m": d_h, "leveled": leveled,
                                "tilt_before_level_deg": float(ang),
                                "scale_check": s_scale, "locked_wall": now,
                            }
                cv2.putText(frame, f"TRAVANDO POSE {len(samples)}/{a.warmup}  ref={ref_id}",
                            (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (80, 200, 80) if seen is not None else (80, 80, 255), 2)

            # ---------------- FASE 2: alvos ----------------
            else:
                do_rec = rec and (now - last_rec) >= RECORD_DT
                hud = []

                if want_ball:
                    res = detect_ball(frame, HSV_LO, HSV_HI, a.min_area, a.min_radius)
                    if res is not None:
                        bx, by, brad, mask = res
                        if show_mask:
                            frame[mask > 0] = [0, 200, 0]
                        cv2.circle(frame, (bx, by), int(brad), (0, 255, 0), 2)
                        dm = depth_median_patch(depth_f, bx, by, W, H)
                        if dm is not None:
                            d, npx, std = dm
                            p_cam = np.array(rs.rs2_deproject_pixel_to_point(
                                intr, [float(bx), float(by)], d))
                            wpt = to_world(p_cam)
                            lbl = f"bola ({wpt[0]:+.2f},{wpt[1]:+.2f}) z={wpt[2]:+.2f}"
                            cor = (0, 255, 0)
                            if a.ball_height:
                                dz = wpt[2] - a.ball_height
                                lbl += f" dz={dz*100:+.0f}cm"
                                if abs(dz) > 0.10:
                                    cor = (0, 165, 255)      # z fora do esperado -> pose/escala suspeita
                            cv2.putText(frame, lbl, (bx + int(brad) + 6, by),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)
                            hud.append("BOLA")
                            if do_rec:
                                add_row("ball", 0, wpt, p_cam, bx, by, d, npx, std, now,
                                        rad=f"{brad:.1f}")

                if want_person and (frame_i % a.every_n == 0):
                    dets, meta_d = [], []
                    for r in model(frame, conf=a.conf, verbose=False):
                        for b in r.boxes:
                            if int(b.cls[0]) != 0:
                                continue
                            x1, y1, x2, y2 = map(int, b.xyxy[0])
                            pp = person_point(depth_f, (x1, y1, x2, y2), W, H,
                                              hint=last_person_d)
                            if pp is None:
                                continue
                            u, vv, d, npx, std = pp
                            p_cam = np.array(rs.rs2_deproject_pixel_to_point(intr, [u, vv], d))
                            dets.append(to_world(p_cam))
                            meta_d.append(dict(p_cam=p_cam, u=u, v=vv, d=d, npx=npx, std=std,
                                               conf=float(b.conf[0]), box=(x1, y1, x2, y2),
                                               clip=int(x1 <= 1 or y1 <= 1 or x2 >= W - 2 or y2 >= H - 2)))
                    if meta_d:
                        last_person_d = float(np.median([m['d'] for m in meta_d]))
                    matched = associate(tracks, dets, now, a.gate_m, a.max_speed)
                    tracks[:] = [t for t in tracks if t.misses < a.max_misses]
                    for di, tk in matched.items():
                        md = meta_d[di]
                        x1, y1, x2, y2 = md["box"]
                        col = [(0, 255, 0), (255, 128, 0), (0, 128, 255), (255, 0, 255)][tk.id % 4]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                        cv2.putText(frame, f"ID{tk.id} {md['conf']:.2f} n={md['npx']}",
                                    (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 2)
                        cv2.drawMarker(frame, (int(md["u"]), int(md["v"])), col, cv2.MARKER_CROSS, 14, 2)
                        # SO grava track CONFIRMADO e ATUALIZADO neste frame
                        if do_rec and tk.hits >= a.min_hits:
                            add_row("person", tk.id, tk.pos, md["p_cam"], md["u"], md["v"],
                                    md["d"], md["npx"], md["std"], now,
                                    conf=f"{md['conf']:.3f}", box=md["box"], clip=md["clip"],
                                    hits=tk.hits, miss=tk.misses)
                    if matched:
                        hud.append(f"PESSOA x{len(matched)}")

                if do_rec:
                    if not rows:
                        t0 = now
                    last_rec = now

                bar = (0, 0, 160) if rec else (40, 40, 40)
                cv2.rectangle(frame, (0, 0), (W, 34), bar, -1)
                cv2.putText(frame, f"{'REC ' + str(trial_n) if rec else 'pronto'}  "
                                   f"linhas={len(rows)}  {' '.join(hud) if hud else 'sem alvo'}  "
                                   f"cam=({cam_pos[0]:+.2f},{cam_pos[1]:+.2f},{cam_pos[2]:+.2f})",
                            (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                            (90, 255, 90) if hud else (90, 90, 255), 2)
                if rec and now - last_log >= 2.0:
                    print(f"  rec linhas={len(rows):5d}  {' '.join(hud) if hud else 'sem alvo'}")
                    last_log = now

            cv2.imshow(WIN, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("b"):
                show_mask = not show_mask
            elif key == ord("r"):
                locked, samples, rows, rec, tracks = False, [], [], False, []
                last_person_d = None
                print("[reset] re-travando a pose...")
            elif key == 32 and locked:
                if rec:
                    close_trial()
                else:
                    trial_n += 1
                    rows, rec, tracks = [], True, []
                    t0 = time.time()
                    print(f"\n[PASSADA {trial_n}] gravando... (ESPACO encerra)")
    finally:
        if rec:
            close_trial()
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
