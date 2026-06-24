"""
plot_corridor_layout.py — Planta baixa + vista lateral dos corredores.

Uso:
    python tools/plot_corridor_layout.py U_esq
    python tools/plot_corridor_layout.py U_esq --save
"""

import sys
import math
from pathlib import Path

import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

CONFIG = Path(__file__).parent.parent / "config" / "corridors.yaml"

MARKER_COLOR = "#e63946"
WALL_COLOR   = "#1d3557"
ISLAND_COLOR = "#457b9d"
CAMERA_COLOR = "#2a9d8f"
BALL_COLOR   = "#f4a261"
DIM_COLOR    = "#666666"
BG_COLOR     = "#f8f9fa"
WALK_COLOR   = "#d6eaf8"
LOS_COLOR    = "#2a9d8f"


def load_config():
    with open(CONFIG) as f:
        return yaml.safe_load(f)


# ── helpers de cotas ──────────────────────────────────────────────────────────

def dim_h(ax, x1, x2, y, label, above=True, fs=7.5):
    sign = 1 if above else -1
    off  = 0.10 * sign
    ax.annotate("", xy=(x2, y + off), xytext=(x1, y + off),
                arrowprops=dict(arrowstyle="<->", color=DIM_COLOR, lw=0.9))
    ax.text((x1+x2)/2, y + off + 0.07*sign, label,
            ha="center", va="bottom" if above else "top",
            fontsize=fs, color=DIM_COLOR)


def dim_v(ax, y1, y2, x, label, left=True, fs=7.5):
    sign = -1 if left else 1
    off  = 0.10 * sign
    ax.annotate("", xy=(x + off, y2), xytext=(x + off, y1),
                arrowprops=dict(arrowstyle="<->", color=DIM_COLOR, lw=0.9))
    ax.text(x + off + 0.06*sign, (y1+y2)/2, label,
            ha="right" if left else "left", va="center",
            fontsize=fs, color=DIM_COLOR, rotation=90)


# ── vista de cima (planta baixa) ─────────────────────────────────────────────

def draw_top_view(ax, c):
    W   = c["total_width_m"]
    H   = c["total_height_m"]
    cw  = c["corridor_width_m"]
    iw  = c["island_width_m"]
    il  = c["island_length_m"]
    cam_x = c.get("camera_x_m", cw / 2)

    ax.set_facecolor(BG_COLOR)
    ax.set_aspect("equal")

    # preenchimentos
    for poly, fc in [
        ([(0,0),(cw,0),(cw,H),(0,H)],             WALK_COLOR),
        ([(W-cw,0),(W,0),(W,H),(W-cw,H)],         WALK_COLOR),
        ([(0,il),(W,il),(W,H),(0,H)],              WALK_COLOR),
        ([(cw,0),(W-cw,0),(W-cw,il),(cw,il)],     "#aed6f1"),
    ]:
        ax.add_patch(plt.Polygon(poly, closed=True, fc=fc, ec="none", zorder=1))

    # ilha borda
    ax.add_patch(plt.Polygon([(cw,0),(W-cw,0),(W-cw,il),(cw,il)],
                             closed=True, fc="none", ec=ISLAND_COLOR, lw=2, zorder=2))

    # paredes
    for (x1,y1),(x2,y2) in c["walls"]:
        ax.plot([x1,x2],[y1,y2], color=WALL_COLOR, lw=3, zorder=3,
                solid_capstyle="round")

    # markers (visão de cima = posição XY)
    for m in c["markers"]:
        mid = m["id"]
        x, y, _ = m["pos"]
        ax.plot(x, y, marker="s", ms=13, color=MARKER_COLOR,
                markeredgecolor="white", markeredgewidth=1.4, zorder=5)
        ax.text(x, y, str(mid), ha="center", va="center",
                fontsize=7, fontweight="bold", color="white", zorder=6)
        side = "left" if x < W/2 else "right"
        ox = 0.13 if side == "left" else -0.13
        ax.text(x+ox, y, f"ID {mid}", ha=side, va="center",
                fontsize=7, color=MARKER_COLOR, fontweight="bold", zorder=6)

    # câmera (deslocada para frente do arm1)
    cam_y = -0.55
    ax.plot(cam_x, cam_y, marker="^", ms=15, color=CAMERA_COLOR,
            markeredgecolor="white", markeredgewidth=1.4, zorder=5)
    ax.text(cam_x, cam_y - 0.13, "Câmera", ha="center", va="top",
            fontsize=8, color=CAMERA_COLOR, fontweight="bold")

    # offset da câmera em relação ao centro do U
    u_cx = W / 2
    ax.annotate("", xy=(cam_x, cam_y + 0.08), xytext=(u_cx, cam_y + 0.08),
                arrowprops=dict(arrowstyle="<->", color=DIM_COLOR, lw=0.9))
    ax.text((cam_x + u_cx)/2, cam_y + 0.18,
            f"{u_cx - cam_x:.2g} m (offset)", ha="center", va="bottom",
            fontsize=7, color=DIM_COLOR)

    # FOV 87°
    fov_half = math.radians(43.5)
    fov_len  = 4.5
    for sign in (+1, -1):
        ax.annotate("", xy=(cam_x + math.sin(fov_half)*sign*fov_len,
                             cam_y + math.cos(fov_half)*fov_len),
                    xytext=(cam_x, cam_y),
                    arrowprops=dict(arrowstyle="-", color=CAMERA_COLOR,
                                   alpha=0.3, lw=1.2, linestyle="--"))

    # linhas de visão câmera → markers
    for m in c["markers"]:
        x, y, _ = m["pos"]
        ax.plot([cam_x, x], [cam_y, y], color=LOS_COLOR,
                lw=0.8, alpha=0.5, linestyle=":", zorder=4)

    # entradas/saídas
    ax.text(cw/2,   -0.06, "✕ entrada", ha="center", va="top", fontsize=8, color="#333")
    ax.text(W-cw/2, -0.06, "• saída",   ha="center", va="top", fontsize=8, color="#333")

    # setas de direção
    ax.annotate("", xy=(cw/2, H*0.58), xytext=(cw/2, 0.35),
                arrowprops=dict(arrowstyle="-|>", color="#aaa", lw=1.5))
    ax.annotate("", xy=(W-cw/2, 0.35), xytext=(W-cw/2, H*0.58),
                arrowprops=dict(arrowstyle="-|>", color="#aaa", lw=1.5))

    # cotas
    dim_h(ax, 0, W,      -0.38, f"{W:.1f} m")
    dim_h(ax, 0, cw,      H+0.22, f"{cw:.1f}")
    dim_h(ax, cw, W-cw,   H+0.22, f"{iw:.2g} m (ilha)")
    dim_h(ax, W-cw, W,    H+0.22, f"{cw:.1f}")
    dim_v(ax, 0, H,      -0.38, f"{H:.1f} m")
    dim_v(ax, 0, il,      W+0.28, f"{il:.2g} m ilha", left=False)
    dim_v(ax, il, H,      W+0.28, f"{H-il:.2g} m viragem", left=False)

    ax.text(W/2, il/2, "ILHA", ha="center", va="center",
            fontsize=11, color=ISLAND_COLOR, fontweight="bold", alpha=0.6, zorder=3)

    ax.set_xlim(-0.65, W + 0.85)
    ax.set_ylim(-1.05, H + 0.55)
    ax.invert_yaxis()
    ax.set_title("Vista de cima (planta baixa)", fontsize=11, fontweight="bold")
    ax.set_xlabel("X (m)", fontsize=9)
    ax.set_ylabel("Y — profundidade (m)", fontsize=9)
    ax.grid(True, alpha=0.2, lw=0.5)


# ── vista lateral (elevação frontal) ─────────────────────────────────────────

def draw_side_view(ax, c):
    """Corte transversal visto de cima → mostra alturas."""
    W      = c["total_width_m"]
    cw     = c["corridor_width_m"]
    il     = c["island_length_m"]
    wh     = c["wall_height_m"]        # 1.8
    mk_z   = c["markers"][0]["pos"][2] # 1.895
    ms     = c.get("marker_size_m",
             load_config().get("marker_size_m", 0.19))
    cam_x  = c.get("camera_x_m", cw/2)
    cam_z  = c.get("camera_z_min_m", 1.9)

    ax.set_facecolor(BG_COLOR)
    ax.set_aspect("equal")

    # chão
    ax.axhline(0, color=WALL_COLOR, lw=1.5, alpha=0.4)

    # paredes externas (blocos)
    for x0, x1 in [(0, 0.05), (W-0.05, W)]:
        ax.add_patch(plt.Polygon([(x0,0),(x1,0),(x1,wh),(x0,wh)],
                                 fc=WALL_COLOR, ec="none", alpha=0.6, zorder=2))

    # ilha (bloco central)
    ax.add_patch(plt.Polygon([(cw,0),(W-cw,0),(W-cw,wh),(cw,wh)],
                             fc=ISLAND_COLOR, ec=ISLAND_COLOR, alpha=0.5, zorder=2))
    ax.text(W/2, wh/2, "ILHA", ha="center", va="center",
            fontsize=9, color="white", fontweight="bold")

    # paredes arm1 e arm2 (lateral)
    for x0, x1 in [(0.05, cw), (W-cw, W-0.05)]:
        ax.add_patch(plt.Polygon([(x0,0),(x1,0),(x1,wh),(x0,wh)],
                                 fc=WALL_COLOR, ec="none", alpha=0.15, zorder=1))

    # marcadores no topo das paredes (quadradinhos vermelhos)
    for m in c["markers"]:
        mx, my, mz = m["pos"]
        mid = m["id"]
        base = mz - ms/2
        ax.add_patch(plt.Rectangle((mx - ms/2, base), ms, ms,
                                   fc=MARKER_COLOR, ec="white", lw=1, zorder=5))
        ax.text(mx, mz, str(mid), ha="center", va="center",
                fontsize=6, fontweight="bold", color="white", zorder=6)

    # câmera
    ax.plot(cam_x, cam_z, marker="<", ms=14, color=CAMERA_COLOR,
            markeredgecolor="white", markeredgewidth=1.2, zorder=6)
    ax.text(cam_x - 0.1, cam_z, "Câmera", ha="right", va="center",
            fontsize=8, color=CAMERA_COLOR, fontweight="bold")

    # linhas de visão câmera → markers
    for m in c["markers"]:
        mx, _, mz = m["pos"]
        ax.plot([cam_x, mx], [cam_z, mz], color=LOS_COLOR,
                lw=1, alpha=0.6, linestyle="--", zorder=4)

    # bola verde (pessoa na arm1 e arm2, bola acima da parede)
    ball_z = wh + 0.15   # bola ligeiramente acima das paredes
    for bx, label in [(cw/2, "bola\n(arm1)"), (W-cw/2, "bola\n(arm2)")]:
        ax.plot(bx, ball_z, "o", ms=14, color=BALL_COLOR,
                markeredgecolor="white", markeredgewidth=1.2, zorder=5)
        ax.text(bx, ball_z + 0.08, label, ha="center", va="bottom",
                fontsize=7, color=BALL_COLOR, fontweight="bold")
        # linha de visão câmera → bola
        ax.plot([cam_x, bx], [cam_z, ball_z], color=BALL_COLOR,
                lw=1, alpha=0.5, linestyle=":", zorder=4)

    # cotas de altura
    dim_v(ax, 0, wh,   -0.18, f"{wh:.1f} m\n(parede)")
    dim_v(ax, 0, mk_z, W+0.18, f"{mk_z:.3g} m\n(marker)", left=False)
    dim_v(ax, 0, cam_z, cam_x - 0.35, f"≥{cam_z:.2g} m\n(câmera)")

    # linha de referência da parede (pontilhada)
    ax.axhline(wh, color=WALL_COLOR, lw=1, linestyle="--", alpha=0.5)
    ax.text(W + 0.05, wh, f"topo\n{wh} m", va="center",
            fontsize=7, color=WALL_COLOR)

    ax.set_xlim(-0.55, W + 0.65)
    ax.set_ylim(-0.15, cam_z + 0.55)
    ax.set_title("Vista frontal (elevação)", fontsize=11, fontweight="bold")
    ax.set_xlabel("X (m)", fontsize=9)
    ax.set_ylabel("Z — altura (m)", fontsize=9)
    ax.grid(True, alpha=0.2, lw=0.5)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args    = sys.argv[1:]
    save    = "--save" in args
    which   = [a for a in args if not a.startswith("--")]
    cfg     = load_config()
    targets = which if which else list(cfg["corridors"].keys())

    for name in targets:
        if name not in cfg["corridors"]:
            print(f"Corredor '{name}' não encontrado."); continue

        c = cfg["corridors"][name]
        # injeta marker_size do nível raiz
        c["marker_size_m"] = cfg.get("marker_size_m", 0.19)

        if name == "U_esq":
            fig, (ax_top, ax_side) = plt.subplots(
                1, 2, figsize=(14, 8),
                gridspec_kw={"width_ratios": [1, 1]})
            fig.suptitle("Corredor U — posição dos ArUcos e câmera",
                         fontsize=14, fontweight="bold", y=1.01)
            draw_top_view(ax_top, c)
            draw_side_view(ax_side, c)
        else:
            fig, ax = plt.subplots(figsize=(7, 9))
            ax.set_title(f"{name} — layout ainda não implementado")

        fig.tight_layout()

        if save:
            out = Path("output") / f"layout_{name}.png"
            out.parent.mkdir(exist_ok=True)
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"Salvo: {out}")
        else:
            plt.show()


if __name__ == "__main__":
    main()
