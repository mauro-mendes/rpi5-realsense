"""
collision_check.py — mede COLISÃO/near-miss por trial a partir do GT (bola verde).

Entra a trajetória (cabeça) do trial + a posição dos obstáculos (sidecar por trial ou
corridors.yaml) + as paredes do corredor, e devolve, por obstáculo e por parede:
  distância MÍNIMA da trajetória até ele + veredito (COLISAO / near-miss / ok).

Como a bola fica ~na vertical da cabeça, o (x,y) da trajetória é bom proxy da posição da
cabeça/corpo → a colisão AÉREA (placa na altura da cabeça) é medida direto pelo GT; a de
chão (cadeira) é medida pela passagem por cima do footprint. Não substitui a anotação humana
do vídeo, mas dá a contagem automática (métrica primária do reto) e concorda com ela.

Uso (RPi5, em ~/rpi5-realsense):
    python tools/collision_check.py --trial P03_reto_1
    python tools/collision_check.py --traj output/trajectory_X.csv --obst output/obstacles_X.json
    python tools/collision_check.py --trial P03_reto_1 --collision-m 0.20 --near-m 0.35

Limiares (ajustáveis): colisão = corpo/cabeça encostou; near-miss = passou raspando.
"""

import argparse
import csv
import glob
import json
import math
import os
from pathlib import Path

import yaml

OUT_DIR = Path(__file__).parent.parent / "output"
YAML_PATH = Path(__file__).parent.parent / "config" / "corridors.yaml"


def _load_traj(path):
    xs, ys, ts = [], [], []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            xs.append(float(r["x_m"])); ys.append(float(r["y_m"])); ts.append(float(r["t_s"]))
    return xs, ys, ts


def _dist_point_box(px, py, xl, xr, yl, yr):
    """Distância de um ponto a um retângulo alinhado aos eixos (0 se dentro)."""
    dx = max(xl - px, 0.0, px - xr)
    dy = max(yl - py, 0.0, py - yr)
    return math.hypot(dx, dy)


def _dist_point_seg(px, py, ax, ay, bx, by):
    """Distância de um ponto ao segmento [A,B]."""
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    if L2 == 0.0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / L2))
    return math.hypot(px - (ax + t * vx), py - (ay + t * vy))


def _obstacle_box(ob):
    """(xl,xr,yl,yr) do obstáculo. x = borda esq → ocupa [x, x+size]. Aéreo = barreira fina
    em y (±0.10); chão = footprint ±size/2 em y."""
    x, y = ob["pos"]; sz = float(ob.get("size_m", 0.4))
    xl, xr = x, x + sz
    hy = 0.10 if ob.get("type") == "aerial" else sz / 2.0
    return xl, xr, y - hy, y + hy


def check(traj_csv, obstacles, walls, collision_m, near_m, wall_m):
    xs, ys, ts = _load_traj(traj_csv)
    n = len(xs)
    print(f"trajetória: {n} pontos ({traj_csv.split(os.sep)[-1] if os.sep in traj_csv else traj_csv})")
    if n == 0:
        print("  (vazia)"); return

    # obstáculos
    for ob in obstacles:
        if "pos" not in ob:
            continue
        xl, xr, yl, yr = _obstacle_box(ob)
        best, bi = 1e9, -1
        for i in range(n):
            d = _dist_point_box(xs[i], ys[i], xl, xr, yl, yr)
            if d < best:
                best, bi = d, i
        verdict = "COLISAO" if best < collision_m else ("near-miss" if best < near_m else "ok (passou longe)")
        print(f"  {ob.get('id','obs'):8} [{ob.get('type','?'):6}] x[{xl:.2f},{xr:.2f}] y[{yl:.2f},{yr:.2f}]"
              f"  min={best*100:4.0f}cm @t={ts[bi]:.1f}s (pos {xs[bi]:.2f},{ys[bi]:.2f}) -> {verdict}")

    # paredes (aproximação mais perto)
    wbest, wi, wseg = 1e9, -1, None
    for (ax, ay), (bx, by) in walls:
        for i in range(n):
            d = _dist_point_seg(xs[i], ys[i], ax, ay, bx, by)
            if d < wbest:
                wbest, wi, wseg = d, i, ((ax, ay), (bx, by))
    wv = "ENCOSTOU" if wbest < wall_m else ("raspou" if wbest < near_m else "ok")
    print(f"  parede   [wall  ] mais perto={wbest*100:4.0f}cm @t={ts[wi]:.1f}s (pos {xs[wi]:.2f},{ys[wi]:.2f}) -> {wv}")


def main():
    ap = argparse.ArgumentParser(description="Colisão/near-miss por trial a partir do GT")
    ap.add_argument("--trial", help="trial_id (acha output/trajectory_<id>_*.csv + obstacles_<id>_*.json)")
    ap.add_argument("--traj", help="caminho explícito do trajectory_*.csv")
    ap.add_argument("--obst", help="caminho explícito do obstacles_*.json")
    ap.add_argument("--corridor", help="corredor no corridors.yaml (se não houver sidecar)")
    ap.add_argument("--cadeira", choices=["esquerda", "direita"], default="esquerda")
    ap.add_argument("--corridors", default=str(YAML_PATH))
    ap.add_argument("--collision-m", type=float, default=0.20, help="colisão se min < isso (m)")
    ap.add_argument("--near-m", type=float, default=0.35, help="near-miss se min < isso (m)")
    ap.add_argument("--wall-m", type=float, default=0.15, help="encostou na parede se min < isso (m)")
    a = ap.parse_args()

    # localiza trajetória + sidecar
    traj = a.traj
    obst_path = a.obst
    if a.trial and not traj:
        c = sorted(glob.glob(str(OUT_DIR / f"trajectory_{a.trial}_*.csv")))
        if not c:
            ap.error(f"não achei trajectory_{a.trial}_*.csv em {OUT_DIR}")
        traj = c[-1]
    if a.trial and not obst_path:
        c = sorted(glob.glob(str(OUT_DIR / f"obstacles_{a.trial}_*.json")))
        obst_path = c[-1] if c else None
    if not traj:
        ap.error("informe --trial ou --traj")

    cfg = yaml.safe_load(open(a.corridors, encoding="utf-8"))
    corridor_key = a.corridor
    obstacles = []
    if obst_path and os.path.exists(obst_path):
        sc = json.load(open(obst_path, encoding="utf-8"))
        corridor_key = sc.get("corridor", corridor_key)
        obstacles = sc.get("obstacles", [])
        print(f"obstáculos: sidecar {os.path.basename(obst_path)} (cadeira={sc.get('cadeira')})")
    else:
        # sem sidecar: resolve do yaml pelo --corridor + --cadeira
        cor = cfg["corridors"].get(corridor_key, {})
        for ob in cor.get("obstacles", []):
            o = dict(ob)
            if "pos" not in o and "x" in o and "y" in o:
                o["pos"] = [float(o["x"].get(a.cadeira, 0.0)), float(o["y"])]
            obstacles.append(o)
        print(f"obstáculos: {corridor_key} do yaml (cadeira={a.cadeira})")

    cor = cfg["corridors"].get(corridor_key, {})
    walls = [((w[0][0], w[0][1]), (w[1][0], w[1][1])) for w in cor.get("walls", [])]
    mw = cor.get("movable_wall", {}).get("segment")
    if mw:
        walls.append(((mw[0][0], mw[0][1]), (mw[1][0], mw[1][1])))

    print(f"limiares: colisão<{a.collision_m}m near<{a.near_m}m parede<{a.wall_m}m\n")
    check(traj, obstacles, walls, a.collision_m, a.near_m, a.wall_m)


if __name__ == "__main__":
    main()
