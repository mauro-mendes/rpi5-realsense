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
import argparse, csv, json, os, sys, time
from datetime import datetime

import cv2
import numpy as np

try:                                  # import tolerante: --check tem que rodar mesmo
    import pyrealsense2 as rs         # sem a lib instalada (e um dos itens que ele checa)
    _RS_ERR = None
except Exception as _e:
    rs, _RS_ERR = None, _e

try:                                  # opcional: so existe se o cenario.py estiver junto
    import cenario as CEN             # (guardo o erro: se faltar, quero dizer QUAL e o motivo)
    _CEN_ERR = None
except Exception as _e:
    CEN, _CEN_ERR = None, _e

DICT_TYPE = cv2.aruco.DICT_4X4_50
DEPTH_MIN, DEPTH_MAX = 0.2, 12.0
_AQUI = os.path.dirname(os.path.abspath(__file__))
PATCH = 2                    # patch (2*PATCH+1)^2 para a mediana de profundidade da bola


def git_commit():
    """Identifica a versao EXATA do codigo que rodou. No ciclo remoto (eu corrijo -> push ->
    voces pull -> testam) sem isso a gente depura a versao errada.

    Devolve 'sha:<hash do arquivo> git:<commit>'.

    O `sha` e do PROPRIO ARQUIVO: funciona mesmo se o script for COPIADO para fora do repo.
    O `git` so aparece se ESTE arquivo estiver rastreado no repo da pasta onde ele esta -
    senao daria o commit do repo do colega (ex.: workspace ROS), que enganaria: a gente
    acharia que sabe a versao e estaria olhando outra coisa."""
    import hashlib, subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        with open(os.path.abspath(__file__), "rb") as f:
            sha = hashlib.sha1(f.read()).hexdigest()[:8]
    except Exception:
        sha = "?"
    git = "-"
    try:                       # universal_newlines (nao text=) p/ funcionar no py3.6
        subprocess.check_output(
            ["git", "ls-files", "--error-unmatch", os.path.basename(os.path.abspath(__file__))],
            cwd=here, universal_newlines=True, stderr=subprocess.DEVNULL)
        git = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=here, universal_newlines=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        pass                   # nao versionado aqui: fica so o sha, que basta
    return "sha:%s git:%s" % (sha, git)


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


def find_model(name):
    """Acha o .pt em disco SEM deixar a ultralytics tentar baixar. O laboratorio e offline:
    um download silencioso vira travamento sem explicacao. Se nao achar, falha dizendo o
    que fazer."""
    if os.path.isabs(name):
        if os.path.exists(name):
            return name
        raise SystemExit(f"[ERRO] modelo nao encontrado: {name}")
    here = os.path.dirname(os.path.abspath(__file__))
    cand = [os.path.join(here, name), os.path.join(os.getcwd(), name),
            os.path.join(here, "..", name), os.path.join(os.getcwd(), "..", name)]
    for c in cand:
        if os.path.exists(c):
            return os.path.abspath(c)
    linhas = [f"[ERRO] '{name}' nao encontrado. Procurei em:"]
    linhas += ["  " + os.path.abspath(c) for c in cand]
    linhas += ["",
               "NAO vou deixar a ultralytics baixar (offline). Opcoes:",
               f"  - copie o {name} que voce ja usa para esta pasta, ou",
               f"  - rode com --model /caminho/completo/para/{name}, ou",
               "  - rode com --track ball (nao usa YOLO nenhum)."]
    raise SystemExit("\n".join(linhas))


def check_env():
    """--check: relatorio do ambiente, sem tocar na camera. Rode uma vez apos o git pull."""
    print("=" * 66)
    print("python      :", sys.version.split()[0])
    try:
        cv2.aruco.ArucoDetector
        api = "API nova (>=4.7)"
    except AttributeError:
        api = "API antiga (<4.7) - suportada"
    print("opencv      :", cv2.__version__, " aruco:", api)
    print("numpy       :", np.__version__)
    print("solvePnP    :", "IPPE_SQUARE" if hasattr(cv2, "SOLVEPNP_IPPE_SQUARE")
          else "ITERATIVE (fallback, ok)")
    if rs is not None:
        print("pyrealsense2:", getattr(rs, "__version__", "ok"))
        try:
            devs = rs.context().query_devices()
            if len(devs):
                for d in devs:
                    print("camera      :", d.get_info(rs.camera_info.name),
                          "sn", d.get_info(rs.camera_info.serial_number))
            else:
                print("camera      : NENHUMA conectada")
        except Exception as e:
            print("camera      : erro ao consultar ->", e)
    else:
        print("pyrealsense2: AUSENTE ->", _RS_ERR)
    try:
        import matplotlib
        print("matplotlib  :", matplotlib.__version__)
    except Exception as e:
        print("matplotlib  : ausente ->", e, " (a planta sai em SVG, sem perder nada)")
    print("cenario.json:", "ok" if CEN and CEN.carrega_cenario()[0] else "NAO CARREGADO")
    try:
        import ultralytics
        print("ultralytics :", ultralytics.__version__)
    except Exception as e:
        print("ultralytics : ausente ->", e, " (so precisa p/ --track person|both)")
    try:
        print("modelo      :", find_model("yolov8n.pt"))
    except SystemExit:
        print("modelo      : NAO ENCONTRADO -> use --model <caminho> ou --track ball")
    print("script      :", os.path.abspath(__file__))
    print("saida ira p/:", os.path.abspath("trajetorias"), "(muda com --out)")
    print("codigo      :", git_commit())
    print("=" * 66)


# ================================== Main =====================================
CSV_COLS = ["timestamp_iso", "track_id", "x_world", "y_world", "z_world",
            "target", "wall", "t_s", "frame",
            "p_cam_x", "p_cam_y", "p_cam_z",
            "u_px", "v_px", "depth_m", "n_px", "depth_std",
            "conf", "rad_px", "bx1", "by1", "bx2", "by2", "clipped",
            "hits", "misses"]


def main():
    ap = argparse.ArgumentParser(description="GT RealSense fixa + ArUco travado: bola e/ou pessoa")
    ap.add_argument("--cenario", default=None,
                    help="caminho do cenario.json. Default: ao lado deste script. Util quando "
                         "o script foi COPIADO p/ outra pasta e o cenario ficou no repo.")
    ap.add_argument("--check", action="store_true",
                    help="so verifica o ambiente (versoes, modelo, camera) e sai")
    ap.add_argument("--track", choices=["ball", "person", "both"], default="both")
    ap.add_argument("--ref-id", type=int, default=None,
                    help="ID do marcador de REFERENCIA. Default: menor ID visivel.")
    ap.add_argument("--marker-size", type=float, default=0.15,
                    help="lado do QUADRADO PRETO em metros (NAO a margem branca)")
    ap.add_argument("--cam-height", type=float, default=None,
                    help="altura REAL da camera (m, trena) - o script confere a escala")
    ap.add_argument("--cam-dist", type=float, default=None,
                    help="distancia HORIZONTAL real camera->centro do marcador (m, trena) - "
                         "2a conferencia. Se voce mediu a linha DIRETA (lente -> marcador), "
                         "use --cam-dist-slant.")
    ap.add_argument("--cam-dist-slant", type=float, default=None,
                    help="distancia DIRETA lente->centro do marcador (m). Com o marcador no "
                         "CHAO, horizontal = sqrt(direta^2 - altura^2); o script converte. "
                         "Exige --cam-height.")
    ap.add_argument("--level", action="store_true",
                    help="camera NIVELADA: forca a vertical do mundo")
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--hsv", default="35,50,50,85,255,255")
    ap.add_argument("--min-area", type=float, default=60.0)
    ap.add_argument("--min-radius", type=float, default=3.0)
    ap.add_argument("--ball-height", type=float, default=None,
                    help="OPCIONAL. Altura medida da bola DESTE participante (m). Nao e "
                         "necessaria: a escala ja e conferida por --cam-height, que e do "
                         "SETUP e nao muda entre participantes. Use so se quiser cravar a "
                         "conferencia num participante especifico.")
    ap.add_argument("--ball-range", type=float, nargs=2, default=[1.40, 2.10],
                    metavar=("MIN", "MAX"),
                    help="faixa plausivel da altura da bola (m). Definida UMA vez p/ o estudo "
                         "inteiro, cobre todos os participantes. Pega bola errada e escala "
                         "grosseiramente errada sem trena por pessoa. Default 1.40 2.10")
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
    # ---- cenario.json vira o PADRAO (linha de comando ainda sobrepoe) ----
    cen, cen_path = ({}, None)
    _arg_cen = None
    for _i, _t in enumerate(sys.argv):
        if _t == "--cenario" and _i + 1 < len(sys.argv):
            _arg_cen = sys.argv[_i + 1]
        elif _t.startswith("--cenario="):
            _arg_cen = _t.split("=", 1)[1]
    # --cenario aponta p/ o JSON; o cenario.py mora na MESMA pasta. Se o script foi copiado
    # p/ fora do repo, e por aqui que ele reencontra o modulo - senao a flag nao resolveria
    # nada (sem o modulo nao ha codigo p/ ler o json nem p/ desenhar).
    global CEN, _CEN_ERR
    if CEN is None and _arg_cen:
        _d = os.path.dirname(os.path.abspath(_arg_cen))
        if _d and _d not in sys.path:
            sys.path.insert(0, _d)
        try:
            import cenario as _C
            CEN, _CEN_ERR = _C, None
            print(f"[cenario] modulo carregado de {_d}")
        except Exception as _e2:
            _CEN_ERR = _e2
    if CEN is not None:
        cen, cen_path = CEN.carrega_cenario(_arg_cen)
    if cen:
        rid, mref = CEN.marcador_ref(cen)
        d3, dh = CEN.dist_camera_marcador(cen)
        cam = cen.get("camera", {})
        padroes = {}
        if rid is not None:
            padroes.update(ref_id=rid, marker_size=float(mref.get("tamanho_m", 0.15)))
        if cam:
            padroes.update(cam_height=float(cam["z_m"]), level=bool(cam.get("nivelada")))
        if dh:
            padroes.update(cam_dist=round(dh, 4))
        b = cen.get("bola", {})
        if b.get("hsv"):
            padroes["hsv"] = ",".join(str(int(v)) for v in b["hsv"])
        if b.get("faixa_altura_m"):
            padroes["ball_range"] = [float(x) for x in b["faixa_altura_m"]]
        cap = cen.get("captura", {})
        if cap.get("alvo"):
            padroes["track"] = cap["alvo"]
        if cap.get("rate_hz"):
            padroes["rate"] = float(cap["rate_hz"])
        ap.set_defaults(**padroes)

    a = ap.parse_args()
    if not cen:
        print("[cenario] AUSENTE -> sem planta ao fim das passadas."
              + (f" (import falhou: {_CEN_ERR})" if CEN is None else f" (procurei em {cen_path})"))
        print("          copie cenario.py + cenario.json para junto do script, "
              "ou use --cenario <caminho>")
    if cen:
        print(f"[cenario] {os.path.basename(cen_path)}: recinto "
              f"{cen['recinto']['largura_m']}x{cen['recinto']['profundidade_m']} m, "
              f"ref=ID {a.ref_id}, marcador {a.marker_size*100:.1f} cm, "
              f"camera z={a.cam_height} dist_h={a.cam_dist}")
    if a.check:
        check_env()
        return

    # trena DIRETA (lente->marcador) -> HORIZONTAL. Marcador no chao (z=0), entao o
    # triangulo e reto e a conversao e exata. Sem isso, um numero medido apontando a trena
    # para o marcador acusa "escala errada" sem haver erro nenhum.
    if a.cam_dist_slant is not None:
        if not a.cam_height:
            raise SystemExit("[ERRO] --cam-dist-slant exige --cam-height (e o cateto vertical).")
        if a.cam_dist_slant <= a.cam_height:
            raise SystemExit(f"[ERRO] --cam-dist-slant ({a.cam_dist_slant}) tem que ser MAIOR "
                             f"que --cam-height ({a.cam_height}): e a hipotenusa.")
        a.cam_dist = float(np.sqrt(a.cam_dist_slant ** 2 - a.cam_height ** 2))
        print(f"[trena] direta {a.cam_dist_slant:.3f} m, altura {a.cam_height:.3f} m "
              f"-> HORIZONTAL {a.cam_dist:.3f} m")
    if rs is None:
        raise SystemExit(f"[ERRO] pyrealsense2 nao importa: {_RS_ERR}\n"
                         f"Rode 'python realsense_gt.py --check' para o relatorio do ambiente.")

    v = [int(x) for x in a.hsv.split(",")]
    HSV_LO, HSV_HI = np.array(v[:3], np.uint8), np.array(v[3:], np.uint8)
    os.makedirs(a.out, exist_ok=True)
    RECORD_DT = 1.0 / a.rate
    want_ball = a.track in ("ball", "both")
    want_person = a.track in ("person", "both")

    model = None
    if want_person:
        from ultralytics import YOLO
        mp = find_model(a.model)
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
    print(f"[versao] {git_commit()}")
    print(f"[OK] {W}x{H}  fx={intr.fx:.1f} fy={intr.fy:.1f}  FOV vertical={vfov:.1f} deg")
    print(f"     alvo={a.track}  marcador={a.marker_size*100:.1f} cm  nivelado={'SIM' if a.level else 'nao'}")
    if want_ball and a.cam_height:
        # pior caso da FAIXA (o participante mais alto e o mais baixo), nao de uma medida
        # individual: o enquadramento tem que servir p/ todo mundo sem remontar a camera.
        pior = max(abs(a.ball_range[1] - a.cam_height), abs(a.ball_range[0] - a.cam_height))
        alvo = a.ball_range[1] if abs(a.ball_range[1] - a.cam_height) >= \
            abs(a.ball_range[0] - a.cam_height) else a.ball_range[0]
        dmin = pior / np.tan(np.radians(vfov / 2.0))
        print(f"     FOV: camera a {a.cam_height:.2f} m, bolas de {a.ball_range[0]:.2f} a "
              f"{a.ball_range[1]:.2f} m (pior desnivel {pior:.2f} m @ {alvo:.2f} m)")
        print(f"          -> no pior caso a bola so entra em quadro alem de {dmin:.2f} m")
        if dmin > 1.5:
            meio = (a.ball_range[0] + a.ball_range[1]) / 2.0
            print(f"          *** SUBA A CAMERA para ~{meio:.2f} m (meio da faixa): assim o pior")
            print(f"          *** caso cai para {abs(a.ball_range[1]-meio)/np.tan(np.radians(vfov/2.0)):.2f} m e serve p/ todos os participantes.")

    WIN = "GT - ESPACO=passada  B=mask  R=re-travar  Q=sair"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL); cv2.resizeWindow(WIN, 960, 720)

    locked, ref_id, samples = False, a.ref_id, []
    yaw_rec = None            # rotacao marcador->recinto (deduzida no travamento)
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
                       "ball_height_m": a.ball_height, "ball_range_m": list(a.ball_range),
                       "cam_height_m": a.cam_height, "cam_dist_m": a.cam_dist},
            "bag": a.bag,
        })
        # A altura da bola sai MEDIDA da propria passada - ninguem precisa ir de trena em
        # cada participante. Se a escala esta certa (conferida por --cam-height, que e do
        # setup), este numero E a altura da bola daquela pessoa.
        zb = [float(r["z_world"]) for r in rows if r["target"] == "ball"]
        if zb:
            meta["ball_height_medido_m"] = round(float(np.median(zb)), 3)
        with open(base + ".meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        n_b = sum(1 for r in rows if r["target"] == "ball")
        print(f"  [CSV] {len(rows)} linhas (bola {n_b} / pessoa {len(rows)-n_b}) -> {base}.csv")
        if zb:
            z = float(np.median(zb))
            fora = "" if a.ball_range[0] <= z <= a.ball_range[1] else "   <-- FORA DA FAIXA!"
            print(f"  [BOLA] altura medida nesta passada: {z:.3f} m{fora}")
        print(f"  [META] {base}.meta.json")
        # PLANTA da passada: trajetoria desenhada dentro do recinto.
        # Cada motivo de NAO gerar e dito em voz alta - falhar em silencio aqui custou
        # uma ida e volta no laboratorio.
        if not cen:
            if CEN is None:
                print(f"  [PLANO] nao gerado: nao consegui importar o cenario.py ({_CEN_ERR})")
                print(f"          -> copie cenario.py e cenario.json para {_AQUI}")
                print(f"          -> ou rode com --cenario /caminho/para/cenario.json")
            else:
                print(f"  [PLANO] nao gerado: cenario.json nao encontrado em {cen_path}")
                print(f"          -> rode com --cenario /caminho/para/cenario.json")
        elif yaw_rec is None:
            print("  [PLANO] nao gerado: o yaw nao foi deduzido no travamento "
                  "(procure a linha '[cenario] yaw do marcador' no arranque)")
        elif not zb:
            print("  [PLANO] nao gerado: nenhuma amostra de BOLA nesta passada")
        else:
            try:
                Pm = np.array([[float(r["x_world"]), float(r["y_world"])]
                               for r in rows if r["target"] == "ball"], float)
                ts_b = np.array([float(r["t_s"]) for r in rows if r["target"] == "ball"], float)
                png = CEN.plota_plano(Pm, ts_b, yaw_rec, cen, base + "_plano.png",
                                      titulo=f"{a.prefix} passada {trial_n}  "
                                             f"({len(Pm)} pts, {ts_b[-1]:.0f} s)")
                print(f"  [PLANO] {png}")
            except Exception as e:
                print(f"  [PLANO] falhou (CSV esta salvo): {e}")
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
                            # Rotacao marcador->recinto DEDUZIDA: comparo a camera vista
                            # pelo ArUco com a posicao dela no cenario.json (trena). Nao
                            # precisa medir angulo nenhum.
                            if cen:
                                yw = CEN.yaw_do_marcador(cam_pos, cen)
                                if yw:
                                    yaw_rec = yw["yaw_deg"]
                                    print(f"  [cenario] yaw do marcador = {yaw_rec:+.0f} deg "
                                          f"(bruto {yw['yaw_bruto_deg']:+.1f}, "
                                          f"residuo {yw['residuo_deg']:.1f})")
                                    print(f"            distancia camera->marcador: "
                                          f"recinto {yw['dist_recinto_m']:.3f} m  x  "
                                          f"ArUco {yw['dist_marcador_m']:.3f} m  "
                                          f"(dif {yw['erro_escala_m']*100:.0f} cm)")
                                    if yw["residuo_deg"] > 20:
                                        print("            *** residuo alto: as bordas do "
                                              "marcador podem nao estar paralelas as paredes")
                                    if yw["erro_escala_m"] > 0.15:
                                        print("            *** os modulos discordam -> "
                                              "conferir --marker-size e as medidas do cenario")
                            lock_info = {
                                "ref_id": int(ref_id), "warmup": len(samples),
                                "yaw_recinto_deg": yaw_rec,
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
                            if a.ball_height:            # medida do participante, se houver
                                dz = wpt[2] - a.ball_height
                                lbl += f" dz={dz*100:+.0f}cm"
                                if abs(dz) > 0.10:
                                    cor = (0, 165, 255)
                            elif not (a.ball_range[0] <= wpt[2] <= a.ball_range[1]):
                                # sem medida do participante: so a FAIXA plausivel. Pega bola
                                # errada (verde qualquer no ambiente) e escala grosseiramente
                                # errada, sem exigir trena a cada pessoa.
                                lbl += " FORA DA FAIXA"
                                cor = (0, 165, 255)
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
                yaw_rec = None
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
