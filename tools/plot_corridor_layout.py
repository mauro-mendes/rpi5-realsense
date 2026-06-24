"""
plot_corridor_layout.py — Planta baixa dos corredores com posições dos markers.

Lê corridors.yaml e gera um desenho em escala com:
  - paredes do corredor
  - posições dos ArUcos (ID e nota)
  - posição da câmera
  - cotas de dimensão

Uso:
    python tools/plot_corridor_layout.py              # todos os corredores
    python tools/plot_corridor_layout.py U_esq        # só o U
    python tools/plot_corridor_layout.py U_esq --save # salva PNG
"""

import sys
import math
from pathlib import Path

import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

CONFIG = Path(__file__).parent.parent / "config" / "corridors.yaml"

MARKER_COLOR  = "#e63946"
WALL_COLOR    = "#1d3557"
ISLAND_COLOR  = "#457b9d"
CAMERA_COLOR  = "#2a9d8f"
DIM_COLOR     = "#555555"
BG_COLOR      = "#f8f9fa"


def load_config():
    with open(CONFIG) as f:
        return yaml.safe_load(f)


def draw_u_esq(ax, cfg):
    c = cfg["corridors"]["U_esq"]
    W  = c["total_width_m"]      # 3.5
    H  = c["total_height_m"]     # 3.3
    cw = c["corridor_width_m"]   # 1.2
    iw = c["island_width_m"]     # 1.1
    il = c["island_length_m"]    # 2.1

    ax.set_facecolor(BG_COLOR)
    ax.set_aspect("equal")

    # ── Preenchimento do corredor (área caminhável) ───────────────────────────
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection

    # Braço esquerdo (arm1)
    arm1 = plt.Polygon([(0, 0), (cw, 0), (cw, H), (0, H)],
                       closed=True, fc="#d6eaf8", ec="none", zorder=1)
    # Braço direito (arm2)
    arm2 = plt.Polygon([(W - cw, 0), (W, 0), (W, H), (W - cw, H)],
                       closed=True, fc="#d6eaf8", ec="none", zorder=1)
    # Espaço de viragem (abaixo da ilha)
    turn = plt.Polygon([(0, il), (W, il), (W, H), (0, H)],
                       closed=True, fc="#d6eaf8", ec="none", zorder=1)
    # Ilha (opaco, cor diferente)
    island = plt.Polygon([(cw, 0), (W - cw, 0), (W - cw, il), (cw, il)],
                         closed=True, fc="#aed6f1", ec=ISLAND_COLOR, lw=1.5,
                         zorder=2)
    for p in [arm1, arm2, turn, island]:
        ax.add_patch(p)

    # ── Paredes externas ─────────────────────────────────────────────────────
    walls = c["walls"]
    for seg in walls:
        (x1, y1), (x2, y2) = seg
        ax.plot([x1, x2], [y1, y2], color=WALL_COLOR, lw=3, zorder=3,
                solid_capstyle="round")

    # ── Marcadores ───────────────────────────────────────────────────────────
    for m in c["markers"]:
        mid = m["id"]
        x, y, z = m["pos"]
        note = m.get("note", "")
        ax.plot(x, y, marker="s", ms=14, color=MARKER_COLOR,
                markeredgecolor="white", markeredgewidth=1.5, zorder=5)
        ax.text(x, y, str(mid), ha="center", va="center",
                fontsize=7, fontweight="bold", color="white", zorder=6)
        # legenda da nota ao lado
        offset_x = 0.12 if x < W / 2 else -0.12
        align = "left" if x < W / 2 else "right"
        ax.text(x + offset_x, y + 0.07, f"ID {mid}", ha=align, va="bottom",
                fontsize=7.5, color=MARKER_COLOR, fontweight="bold", zorder=6)
        ax.text(x + offset_x, y - 0.07, note, ha=align, va="top",
                fontsize=6, color="#555", zorder=6, style="italic")

    # ── Câmera ───────────────────────────────────────────────────────────────
    cam_x, cam_y = W / 2, -0.55
    ax.plot(cam_x, cam_y, marker="^", ms=16, color=CAMERA_COLOR,
            markeredgecolor="white", markeredgewidth=1.5, zorder=5)
    ax.text(cam_x, cam_y - 0.15, "Câmera\n(D435i)", ha="center", va="top",
            fontsize=8, color=CAMERA_COLOR, fontweight="bold")
    # FOV indicativo (87° horizontal)
    fov_half = math.radians(43.5)
    fov_len  = 4.2
    for sign in (+1, -1):
        dx = math.sin(fov_half) * sign
        dy = math.cos(fov_half)
        ax.annotate("", xy=(cam_x + dx * fov_len, cam_y + dy * fov_len),
                    xytext=(cam_x, cam_y),
                    arrowprops=dict(arrowstyle="-", color=CAMERA_COLOR,
                                   alpha=0.35, lw=1.2, linestyle="--"))

    # ── Entradas / saídas ─────────────────────────────────────────────────────
    ax.text(cw / 2, -0.08, "✕ entrada", ha="center", va="top",
            fontsize=8, color="#333")
    ax.text(W - cw / 2, -0.08, "• saída", ha="center", va="top",
            fontsize=8, color="#333")

    # ── Cotas ────────────────────────────────────────────────────────────────
    def dim_h(ax, x1, x2, y, label, above=True):
        dy = 0.12 if above else -0.12
        ax.annotate("", xy=(x2, y + dy), xytext=(x1, y + dy),
                    arrowprops=dict(arrowstyle="<->", color=DIM_COLOR, lw=1))
        ax.text((x1 + x2) / 2, y + dy + (0.08 if above else -0.08),
                label, ha="center", va="bottom" if above else "top",
                fontsize=7.5, color=DIM_COLOR)

    def dim_v(ax, y1, y2, x, label, left=True):
        dx = -0.12 if left else 0.12
        ax.annotate("", xy=(x + dx, y2), xytext=(x + dx, y1),
                    arrowprops=dict(arrowstyle="<->", color=DIM_COLOR, lw=1))
        ax.text(x + dx + (-0.07 if left else 0.07), (y1 + y2) / 2,
                label, ha="right" if left else "left", va="center",
                fontsize=7.5, color=DIM_COLOR, rotation=90)

    # Largura total
    dim_h(ax, 0, W,   -0.35, f"{W:.1f} m (total)")
    # Arm1 width
    dim_h(ax, 0, cw,   H + 0.25, f"{cw:.1f} m")
    # Island width
    dim_h(ax, cw, W - cw, H + 0.25, f"{iw:.1f} m (ilha)")
    # Arm2 width
    dim_h(ax, W - cw, W, H + 0.25, f"{cw:.1f} m")
    # Total height
    dim_v(ax, 0, H, -0.35, f"{H:.1f} m (total)")
    # Island length
    dim_v(ax, 0, il, W + 0.3, f"{il:.1f} m (ilha)", left=False)
    # Turning space
    dim_v(ax, il, H, W + 0.3, f"{H-il:.1f} m (viragem)", left=False)

    # ── Rótulo da ilha ────────────────────────────────────────────────────────
    ax.text(W / 2, il / 2, "ILHA", ha="center", va="center",
            fontsize=10, color=ISLAND_COLOR, fontweight="bold",
            alpha=0.7, zorder=3)

    # ── Seta de direção do user ───────────────────────────────────────────────
    ax.annotate("", xy=(cw / 2, H * 0.6), xytext=(cw / 2, 0.4),
                arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.5))
    ax.annotate("", xy=(W - cw / 2, 0.4), xytext=(W - cw / 2, H * 0.6),
                arrowprops=dict(arrowstyle="-|>", color="#888", lw=1.5))

    ax.set_xlim(-0.7, W + 0.9)
    ax.set_ylim(-1.0, H + 0.65)
    ax.set_title("Corredor U — posição dos ArUcos", fontsize=13,
                 fontweight="bold", pad=12)
    ax.set_xlabel("X (m)", fontsize=9)
    ax.set_ylabel("Y — profundidade no corredor (m)", fontsize=9)
    ax.invert_yaxis()   # y=0 no topo (entrada/câmera), y=3.3 em baixo (fundo)
    ax.grid(True, alpha=0.25, lw=0.5)


def main():
    args   = sys.argv[1:]
    save   = "--save" in args
    which  = [a for a in args if not a.startswith("--")]
    cfg    = load_config()

    targets = which if which else list(cfg["corridors"].keys())

    for name in targets:
        if name not in cfg["corridors"]:
            print(f"Corredor '{name}' não encontrado em corridors.yaml")
            continue

        fig, ax = plt.subplots(figsize=(7, 9))

        if name == "U_esq":
            draw_u_esq(ax, cfg)
        else:
            ax.set_title(f"{name} — layout ainda não implementado neste script")

        fig.tight_layout()

        if save:
            out = Path(f"output/layout_{name}.png")
            out.parent.mkdir(exist_ok=True)
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"Salvo: {out}")
        else:
            plt.show()


if __name__ == "__main__":
    main()
