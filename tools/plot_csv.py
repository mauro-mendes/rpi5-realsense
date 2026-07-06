"""
plot_csv.py — Gera plots a partir de CSVs de trajectória já gravados.

Dois modelos de correcção de profundidade:

  Fixo  : p_corr = cam + (p_raw - cam) * depth_scale
  Variável: p_corr = cam + (p_raw - cam) * (A + B * d_y)
            onde d_y = p_raw[1] - cam_y  (profundidade Y medida pelo sensor)
            Parâmetros: depth_scale_A, depth_scale_B no YAML

O modelo variável compensa a não-linearidade do RealSense:
  - ~2.5m: scale≈1.025  (pouco erro perto)
  - ~4.8m: scale≈1.083  (zona de travessia)
  - ~6.8m: scale≈1.132  (fim do corredor)

Uso:
    python tools/plot_csv.py output/trajectory_S_esq_*.csv
    python tools/plot_csv.py output/trajectory_S_esq_20250101.csv --corridor S_esq
    python tools/plot_csv.py output/trajectory_*.csv --compare   # 3 colunas: raw | fixo | variável
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

# ── Paths ──────────────────────────────────────────────────────────────────────
YAML_PATH = Path(__file__).parent.parent / "config" / "corridors.yaml"
OUT_DIR   = Path(__file__).parent.parent / "output"


def load_yaml(path: Path):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_csv(path: Path) -> tuple[list[np.ndarray], list[float]]:
    """Lê CSV de trajectória. Devolve (trajectory, traj_times)."""
    trajectory, traj_times = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trajectory.append(np.array([float(row["x_m"]),
                                         float(row["y_m"]),
                                         float(row["z_m"])]))
            traj_times.append(float(row["t_s"]))
    return trajectory, traj_times


def apply_depth_scale(trajectory: list[np.ndarray],
                      cam_pos: np.ndarray,
                      depth_scale: float) -> list[np.ndarray]:
    """Escala fixa: p_corr = cam + (p_raw - cam) * depth_scale"""
    if abs(depth_scale - 1.0) < 1e-6:
        return trajectory
    return [cam_pos + (p - cam_pos) * depth_scale for p in trajectory]


def apply_depth_scale_variable(trajectory: list[np.ndarray],
                                cam_pos: np.ndarray,
                                A: float, B: float) -> list[np.ndarray]:
    """Escala variável: scale(d_y) = A + B*d_y
    d_y = profundidade Y medida (p[1] - cam_y) — o que o RealSense realmente mede.
    Compensa a não-linearidade do erro do sensor com a distância.
    """
    result = []
    for p in trajectory:
        v   = p - cam_pos
        d_y = v[1]          # profundidade ao longo do eixo Y (óptico para câmara horizontal)
        scale = A + B * max(d_y, 0.1)
        result.append(cam_pos + v * scale)
    return result


# ── Funções de plot (replicadas de record_trajectory.py) ──────────────────────
def _find_stable_start(pts: np.ndarray,
                       window: int = 5, max_spread: float = 0.25) -> int:
    for i in range(len(pts) - window):
        chunk  = pts[i : i + window]
        spread = float(np.max(np.linalg.norm(chunk - chunk.mean(axis=0), axis=1)))
        if spread <= max_spread:
            return i
    return 0


def _filter_outliers(pts: np.ndarray, max_jump: float = 0.6) -> np.ndarray:
    if len(pts) < 2:
        return pts
    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) <= max_jump:
            keep.append(i)
    filtered = pts[keep]
    if len(filtered) < max(2, len(pts) // 5):
        return pts
    return filtered


def _smooth_path(pts: np.ndarray, window: int = 7) -> np.ndarray:
    if len(pts) < window:
        return pts
    half = window // 2
    out  = pts.copy().astype(float)
    for i in range(half, len(pts) - half):
        out[i] = pts[i - half : i + half + 1].mean(axis=0)
    return out


def _resample_path(trajectory: list, n_pts: int = 200):
    pts   = np.array([[p[0], p[1]] for p in trajectory])
    start = _find_stable_start(pts)
    pts   = pts[start:]
    pts   = _filter_outliers(pts)
    if len(pts) >= 7:
        pts = _smooth_path(pts, window=7)
    if len(pts) < 2:
        return None
    segs  = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cumul = np.concatenate([[0.0], np.cumsum(segs)])
    total = cumul[-1]
    if total < 0.1:
        return None
    d_uni = np.linspace(0.0, total, n_pts)
    return np.interp(d_uni, cumul, pts[:, 0]), np.interp(d_uni, cumul, pts[:, 1])


def _draw_corridor(ax, corridor: dict, sg: dict):
    for wall in corridor.get("walls", []):
        ax.plot([wall[0][0], wall[1][0]], [wall[0][1], wall[1][1]],
                "k-", linewidth=2)
    mw = corridor.get("movable_wall", {}).get("segment")
    if mw:
        ax.plot([mw[0][0], mw[1][0]], [mw[0][1], mw[1][1]],
                "b--", linewidth=2, label="Parede móvel")
    for m in corridor.get("markers", []):
        ax.plot(m["pos"][0], m["pos"][1], "rs", markersize=6, zorder=3)
        ax.text(m["pos"][0] + 0.05, m["pos"][1], str(m["id"]),
                fontsize=6, color="darkred", va="center")


def plot_scatter(ax, trajectory, traj_times, cam_pos, corridor, sg,
                 title: str):
    total_w = corridor.get("total_width_m", sg.get("total_width_m", 3.5))
    total_d = corridor.get("total_depth_m",
              corridor.get("u_depth_m", sg.get("total_depth_m", 5.5)))

    _draw_corridor(ax, corridor, sg)
    ax.plot(cam_pos[0], cam_pos[1], "D", color="darkorange", markersize=9,
            zorder=4, label=f"Câmara ({cam_pos[0]:.2f},{cam_pos[1]:.2f})")

    if trajectory:
        xs = [p[0] for p in trajectory]
        ys = [p[1] for p in trajectory]
        sc = ax.scatter(xs, ys, c=traj_times, cmap="plasma", s=8, zorder=4)
        plt.colorbar(sc, ax=ax, label="Tempo (s)", shrink=0.55)
        ax.plot(xs[0],  ys[0],  "go", markersize=10, zorder=5, label="Início")
        ax.plot(xs[-1], ys[-1], "ro", markersize=10, zorder=5, label="Fim")

    ax.set_xlim(-0.3, total_w + 0.3)
    ax.set_ylim(cam_pos[1] - 0.4, total_d + 0.4)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.grid(True, alpha=0.25)
    ax.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax.text(total_w / 2, -0.15, "entrada", ha="center", fontsize=8, color="gray")
    ax.legend(loc="upper right", fontsize=7)


def plot_path(ax, trajectory, cam_pos, corridor, sg, title: str):
    total_w = corridor.get("total_width_m", sg.get("total_width_m", 3.5))
    total_d = corridor.get("total_depth_m",
              corridor.get("u_depth_m", sg.get("total_depth_m", 5.5)))

    _draw_corridor(ax, corridor, sg)
    ax.plot(cam_pos[0], cam_pos[1], "D", color="darkorange", markersize=9,
            zorder=4, label=f"Câmara ({cam_pos[0]:.2f},{cam_pos[1]:.2f})")

    res = _resample_path(trajectory) if trajectory else None
    if res is not None:
        x_res, y_res = res
        ax.plot(x_res, y_res, color="steelblue", linewidth=1.5, alpha=0.8)
        ax.scatter(x_res[::10], y_res[::10], color="steelblue", s=16, zorder=5)
        ax.plot(x_res[0],  y_res[0],  "go", markersize=10, zorder=6, label="Início")
        ax.plot(x_res[-1], y_res[-1], "ro", markersize=10, zorder=6, label="Fim")
    else:
        ax.text(0.5, 0.5, "pontos insuficientes", transform=ax.transAxes,
                ha="center", va="center", color="gray")

    ax.set_xlim(-0.3, total_w + 0.3)
    ax.set_ylim(cam_pos[1] - 0.4, total_d + 0.4)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.grid(True, alpha=0.25)
    ax.axhline(0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax.text(total_w / 2, -0.15, "entrada", ha="center", fontsize=8, color="gray")
    ax.legend(loc="upper right", fontsize=7)


# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Plota trajectórias de CSV com correcção de profundidade")
    parser.add_argument("csvs", nargs="+", type=Path,
                        help="Ficheiros CSV de trajectória")
    parser.add_argument("--corridor", default=None,
                        choices=["U_esq", "reto_esq", "S_esq"],
                        help="Tipo de corredor (detectado pelo nome do ficheiro se omitido)")
    parser.add_argument("--depth-scale", type=float, default=None,
                        help="Escala fixa (override do YAML)")
    parser.add_argument("--compare", action="store_true",
                        help="3 colunas: raw | escala fixa | escala variável")
    args = parser.parse_args()

    cfg = load_yaml(YAML_PATH)
    sg  = cfg.get("shared_geometry", {})

    cam_pos = np.array([
        float(sg.get("camera_x_m", 1.80)),
        float(sg.get("camera_y_m", -2.50)),
        float(sg.get("camera_z_m",  1.90)),
    ])
    fixed_scale = args.depth_scale if args.depth_scale is not None \
                  else float(sg.get("depth_scale", 1.0))
    A = float(sg.get("depth_scale_A", 1.0))
    B = float(sg.get("depth_scale_B", 0.0))

    print(f"cam_pos      : {cam_pos}")
    print(f"scale fixo   : {fixed_scale:.3f}")
    print(f"scale variável: A={A:.4f}  B={B:.4f}  "
          f"[scale(2.5m)={A+B*2.5:.3f}  scale(5m)={A+B*5:.3f}  scale(7m)={A+B*7:.3f}]")

    OUT_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for csv_path in args.csvs:
        if not csv_path.exists():
            print(f"[ERRO] não encontrado: {csv_path}")
            continue

        corridor_key = args.corridor
        if corridor_key is None:
            for k in ["U_esq", "reto_esq", "S_esq"]:
                if k in csv_path.stem:
                    corridor_key = k
                    break
        if corridor_key is None:
            corridor_key = "S_esq"
            print(f"[AVISO] corredor não detectado, usando {corridor_key}")

        corridor = cfg["corridors"].get(corridor_key, {})
        print(f"\n{csv_path.name}  →  corredor={corridor_key}")

        traj_raw,  traj_times = load_csv(csv_path)
        traj_fix  = apply_depth_scale(traj_raw, cam_pos, fixed_scale)
        traj_var  = apply_depth_scale_variable(traj_raw, cam_pos, A, B)

        def _yrange(traj):
            ys = [p[1] for p in traj]
            return f"[{min(ys):.2f}, {max(ys):.2f}]"

        print(f"  {len(traj_raw)} pontos")
        print(f"  y_raw      : {_yrange(traj_raw)}")
        print(f"  y_fixo     : {_yrange(traj_fix)}  (scale={fixed_scale:.3f})")
        print(f"  y_variável : {_yrange(traj_var)}  (A={A:.3f} B={B:.4f})")

        stem = csv_path.stem

        if args.compare:
            # ── 3 colunas × 2 linhas: scatter em cima, caminho em baixo
            fig, axes = plt.subplots(2, 3, figsize=(21, 20))
            fig.suptitle(f"{stem}  —  comparação de modelos de correcção",
                         fontsize=12)

            plot_scatter(axes[0, 0], traj_raw, traj_times, cam_pos, corridor, sg,
                         "Scatter — sem correcção (raw)")
            plot_scatter(axes[0, 1], traj_fix, traj_times, cam_pos, corridor, sg,
                         f"Scatter — escala fixa ({fixed_scale:.3f})")
            plot_scatter(axes[0, 2], traj_var, traj_times, cam_pos, corridor, sg,
                         f"Scatter — escala variável (A={A:.3f} B={B:.4f})")

            plot_path(axes[1, 0], traj_raw, cam_pos, corridor, sg,
                      "Caminho — sem correcção (raw)")
            plot_path(axes[1, 1], traj_fix, cam_pos, corridor, sg,
                      f"Caminho — escala fixa ({fixed_scale:.3f})")
            plot_path(axes[1, 2], traj_var, cam_pos, corridor, sg,
                      f"Caminho — escala variável (A={A:.3f} B={B:.4f})")

            fig.tight_layout()
            out = OUT_DIR / f"compare_{stem}_{ts}.png"
            fig.savefig(str(out), dpi=150)
            plt.close(fig)
            print(f"  [PLOT] comparação (3×2) → {out.name}")

        else:
            # modo padrão: usa escala variável se A/B configurados, senão fixa
            traj_out  = traj_var if B > 0 else traj_fix
            model_lbl = f"var(A={A:.3f},B={B:.4f})" if B > 0 else f"fixo({fixed_scale:.3f})"

            fig_s, ax_s = plt.subplots(figsize=(6, 10))
            plot_scatter(ax_s, traj_out, traj_times, cam_pos, corridor, sg,
                         f"Trajectória — {corridor_key}  [{model_lbl}]")
            fig_s.tight_layout()
            out_s = OUT_DIR / f"trajectory_{stem}_{ts}.png"
            fig_s.savefig(str(out_s), dpi=150)
            plt.close(fig_s)

            fig_p, ax_p = plt.subplots(figsize=(6, 10))
            plot_path(ax_p, traj_out, cam_pos, corridor, sg,
                      f"Caminho — {corridor_key}  [{model_lbl}]")
            fig_p.tight_layout()
            out_p = OUT_DIR / f"path_{stem}_{ts}.png"
            fig_p.savefig(str(out_p), dpi=150)
            plt.close(fig_p)

            print(f"  [PLOT] scatter → {out_s.name}")
            print(f"  [PLOT] caminho → {out_p.name}")

    print("\nConcluído.")


if __name__ == "__main__":
    main()
