"""
plot_corridor_layout.py — Planta baixa + vista frontal dos corredores.

Estrutura física única com parede móvel em 3 posições:
  pos 1 → Corredor U      (layout_U_esq.png)
  pos 2 → Corredor Reto   (layout_Reto.png)
  pos 3 → Corredor S      (layout_S.png)

Uso:
    python tools/plot_corridor_layout.py U_esq --save
    python tools/plot_corridor_layout.py reto_esq S_esq --save
    python tools/plot_corridor_layout.py --save   (todos os 3)
"""

import sys
import math
from pathlib import Path

import yaml
import matplotlib.pyplot as plt

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

# Cores e labels das 3 posições de parede móvel
MWALL_COLOR = {1: "#27ae60", 2: "#e67e22", 3: "#e74c3c"}
MWALL_LABEL = {1: "pos 1 → U", 2: "pos 2 → Reto", 3: "pos 3 → S"}

# Cor das setas de percurso por corredor
PATH_COLOR  = {"U_esq": "#1a6b9a", "reto_esq": "#e67e22", "S_esq": "#c0392b"}

# Nomes de ficheiro de saída
OUT_NAME    = {"U_esq": "layout_U_esq.png", "reto_esq": "layout_Reto.png",
               "S_esq": "layout_S.png"}


def load_config():
    with open(CONFIG, encoding="utf-8") as f:
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

def draw_top_view(ax, c, name):
    W    = c["total_width_m"]      # 3.5
    H    = c.get("u_depth_m", c.get("total_height_m", 3.3))  # 3.3
    ext  = c.get("extension_length_m", 0.0)  # 2.2
    cw   = c["corridor_width_m"]   # 1.2
    iw   = c["island_width_m"]     # 1.1
    il   = c["island_length_m"]    # 2.1
    full = H + ext                 # 5.5
    cam_x = c.get("camera_x_m", cw / 2)

    ax.set_facecolor(BG_COLOR)
    ax.set_aspect("equal")

    # ── áreas transitáveis ──────────────────────────────────
    for poly in [
        [(0, 0),   (cw, 0),    (cw, H),    (0, H)   ],   # arm1
        [(W-cw,0), (W, 0),     (W, full),  (W-cw,full)],  # arm2 + extensão
        [(0, il),  (W, il),    (W, H),     (0, H)   ],   # turning space
    ]:
        ax.add_patch(plt.Polygon(poly, closed=True,
                                 fc=WALK_COLOR, ec="none", zorder=1))

    # ilha (fundo diferente)
    ax.add_patch(plt.Polygon([(cw,0),(W-cw,0),(W-cw,il),(cw,il)],
                             closed=True, fc="#aed6f1", ec="none", zorder=1))
    # borda da ilha
    ax.add_patch(plt.Polygon([(cw,0),(W-cw,0),(W-cw,il),(cw,il)],
                             closed=True, fc="none", ec=ISLAND_COLOR, lw=2, zorder=2))

    # ── paredes fixas ───────────────────────────────────────
    fixed = [
        ((0,   0),   (0,    H)),    # parede externa esq (arm1)
        ((W,   0),   (W,    full)), # parede externa dir (arm2 + ext)
        ((0,   H),   (W-cw, H)),    # parede do fundo esq (fixa)
        ((W-cw,H),   (W-cw, full)), # parede esq da extensão
        ((W-cw,full),(W,    full)), # parede do fundo da extensão
        ((cw,  0),   (cw,   il)),   # ilha: face esq
        ((cw,  il),  (W-cw, il)),   # ilha: face inferior
        ((W-cw,il),  (W-cw, 0)),    # ilha: face dir
    ]
    for (x1, y1), (x2, y2) in fixed:
        ax.plot([x1, x2], [y1, y2], color=WALL_COLOR, lw=3, zorder=3,
                solid_capstyle="round")

    # ── parede móvel deste corredor ─────────────────────────
    mw = c.get("movable_wall", {})
    if mw:
        pos = mw.get("position", 0)
        seg = mw.get("segment", [])
        col = MWALL_COLOR.get(pos, "#999")
        lbl = MWALL_LABEL.get(pos, f"pos {pos}")
        if seg:
            (x1, y1), (x2, y2) = seg
            ax.plot([x1, x2], [y1, y2], color=col, lw=5, zorder=8,
                    solid_capstyle="butt")
            # número grande no centro da parede
            mx, my = (x1+x2)/2, (y1+y2)/2
            ax.text(mx, my - 0.15, lbl,
                    ha="center", va="top", fontsize=9,
                    color=col, fontweight="bold")

    # ── seta de percurso ────────────────────────────────────
    pc   = PATH_COLOR.get(name, "#888")
    akw  = dict(color=pc, lw=2)
    cx1  = cw / 2       # centro arm1 = 0.6
    cx2  = W - cw / 2   # centro arm2 = 2.9

    if name == "U_esq":
        ax.annotate("", xy=(cx1, H-0.25), xytext=(cx1, 0.5),
                    arrowprops=dict(arrowstyle="-|>", **akw))
        ax.annotate("", xy=(cx2, H-0.25), xytext=(cx1, H-0.25),
                    arrowprops=dict(arrowstyle="-|>", **akw))
        ax.annotate("", xy=(cx2, 0.5), xytext=(cx2, H-0.25),
                    arrowprops=dict(arrowstyle="-|>", **akw))

    elif name == "reto_esq":
        ax.annotate("", xy=(cx2, full-0.3), xytext=(cx2, 0.4),
                    arrowprops=dict(arrowstyle="-|>", **akw))

    elif name == "S_esq":
        ax.annotate("", xy=(cx1, il-0.1), xytext=(cx1, 0.4),
                    arrowprops=dict(arrowstyle="-|>", **akw))
        ax.annotate("", xy=(cx2, il+0.1), xytext=(cx1, il+0.1),
                    arrowprops=dict(arrowstyle="-|>", **akw))
        ax.annotate("", xy=(cx2, full-0.3), xytext=(cx2, il+0.2),
                    arrowprops=dict(arrowstyle="-|>", **akw))

    # ── marcadores (XY no plano) ────────────────────────────
    for m in c["markers"]:
        mid     = m["id"]
        x, y, _ = m["pos"]
        ax.plot(x, y, marker="s", ms=13, color=MARKER_COLOR,
                markeredgecolor="white", markeredgewidth=1.4, zorder=5)
        ax.text(x, y, str(mid), ha="center", va="center",
                fontsize=7, fontweight="bold", color="white", zorder=6)
        side = "left" if x < W/2 else "right"
        ox   = 0.14 if side == "left" else -0.14
        ax.text(x+ox, y, f"ID {mid}", ha=side, va="center",
                fontsize=7, color=MARKER_COLOR, fontweight="bold", zorder=6)

    # ── câmera ──────────────────────────────────────────────
    cam_y = -0.55
    ax.plot(cam_x, cam_y, marker="^", ms=15, color=CAMERA_COLOR,
            markeredgecolor="white", markeredgewidth=1.4, zorder=5)
    ax.text(cam_x, cam_y - 0.13, f"Câmera  x={cam_x:.2g} m",
            ha="center", va="top", fontsize=8,
            color=CAMERA_COLOR, fontweight="bold")

    # FOV 87°
    fov_half = math.radians(43.5)
    fov_len  = 7.0
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
                lw=0.8, alpha=0.4, linestyle=":", zorder=4)

    # ── labels de entrada/saída ─────────────────────────────
    ax.text(cx1, -0.06, "✕ arm1", ha="center", va="top", fontsize=8, color="#333")
    ax.text(cx2, -0.06, "• arm2", ha="center", va="top", fontsize=8, color="#333")

    # ── cotas ───────────────────────────────────────────────
    dim_h(ax, 0,    W,    -0.38,      f"{W:.1f} m")
    dim_h(ax, 0,    cw,    H+0.22,   f"{cw:.1f}")
    dim_h(ax, cw,   W-cw,  H+0.22,  f"{iw:.2g} m\n(ilha)")
    dim_h(ax, W-cw, W,     H+0.22,  f"{cw:.1f}")
    dim_v(ax, 0,  il, -0.38,         f"{il:.2g} m\nilha")
    dim_v(ax, il, H,  -0.38,         f"{H-il:.2g} m\nturning")
    dim_v(ax, 0,  H,   W+0.28,       f"{H:.1f} m (U)", left=False)
    if ext > 0:
        dim_v(ax, H, full, W+0.28,   f"{ext:.2g} m\next",  left=False)

    ax.text(W/2, il/2, "ILHA", ha="center", va="center",
            fontsize=11, color=ISLAND_COLOR, fontweight="bold", alpha=0.6, zorder=3)
    if ext > 0:
        ax.text(cx2, (H+full)/2, "ext",
                ha="center", va="center", fontsize=9, color="#888", alpha=0.8)

    ax.set_xlim(-0.65, W + 0.95)
    ax.set_ylim(-1.05, full + 0.55)
    ax.invert_yaxis()
    titles = {
        "U_esq":    "Vista de cima — Corredor U (parede móvel pos. 1)",
        "reto_esq": "Vista de cima — Corredor Reto (parede móvel pos. 2)",
        "S_esq":    "Vista de cima — Corredor S (parede móvel pos. 3)",
    }
    ax.set_title(titles.get(name, name), fontsize=11, fontweight="bold")
    ax.set_xlabel("X (m)", fontsize=9)
    ax.set_ylabel("Y — profundidade (m)", fontsize=9)
    ax.grid(True, alpha=0.2, lw=0.5)


# ── vista frontal (elevação) ─────────────────────────────────────────────────

def draw_side_view(ax, c):
    W      = c["total_width_m"]
    cw     = c["corridor_width_m"]
    wh     = c["wall_height_m"]
    mk_z   = c["markers"][0]["pos"][2]
    ms     = c.get("marker_size_m", 0.19)
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

    # arm1 e arm2 (transparente)
    for x0, x1 in [(0.05, cw), (W-cw, W-0.05)]:
        ax.add_patch(plt.Polygon([(x0,0),(x1,0),(x1,wh),(x0,wh)],
                                 fc=WALL_COLOR, ec="none", alpha=0.15, zorder=1))

    # marcadores no topo (quadrados vermelhos)
    for m in c["markers"]:
        mx, _, mz = m["pos"]
        base = mz - ms/2
        ax.add_patch(plt.Rectangle((mx - ms/2, base), ms, ms,
                                   fc=MARKER_COLOR, ec="white", lw=1, zorder=5))
        ax.text(mx, mz, str(m["id"]), ha="center", va="center",
                fontsize=6, fontweight="bold", color="white", zorder=6)

    # câmera com seta de anotação
    ax.plot(cam_x, cam_z, marker=">", ms=14, color=CAMERA_COLOR,
            markeredgecolor="white", markeredgewidth=1.2, zorder=6)
    ax.annotate(f"Câmera\nx={cam_x:.2g} m",
                xy=(cam_x, cam_z), xytext=(cam_x - 0.5, cam_z + 0.22),
                ha="center", va="bottom",
                fontsize=8, color=CAMERA_COLOR, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=CAMERA_COLOR, lw=0.9))

    # linhas de visão câmera → markers
    for m in c["markers"]:
        mx, _, mz = m["pos"]
        ax.plot([cam_x, mx], [cam_z, mz], color=LOS_COLOR,
                lw=1, alpha=0.6, linestyle="--", zorder=4)

    # bola verde (arm1 e arm2 acima da parede)
    ball_z = wh + 0.15
    for bx, lbl in [(cw/2, "bola\n(arm1)"), (W-cw/2, "bola\n(arm2)")]:
        ax.plot(bx, ball_z, "o", ms=14, color=BALL_COLOR,
                markeredgecolor="white", markeredgewidth=1.2, zorder=5)
        ax.text(bx, ball_z + 0.08, lbl, ha="center", va="bottom",
                fontsize=7, color=BALL_COLOR, fontweight="bold")
        ax.plot([cam_x, bx], [cam_z, ball_z], color=BALL_COLOR,
                lw=1, alpha=0.5, linestyle=":", zorder=4)

    # labels entrada/saída
    ax.text(cw/2,   -0.10, "✕ entrada\n(arm1)", ha="center", va="top",
            fontsize=7, color="#333")
    ax.text(W-cw/2, -0.10, "• saída\n(arm2)",   ha="center", va="top",
            fontsize=7, color="#333")

    # cotas de altura
    dim_v(ax, 0, wh,  -0.18, f"{wh:.1f} m\n(parede)")
    dim_v(ax, 0, mk_z, W+0.18, f"{mk_z:.3g} m\n(marker)", left=False)
    dim_v(ax, 0, cam_z, cam_x + 0.4, f"≥{cam_z:.2g} m\n(câmera)", left=False)

    # linha de referência do topo das paredes
    ax.axhline(wh, color=WALL_COLOR, lw=1, linestyle="--", alpha=0.5)
    ax.text(-0.05, wh, f"topo\n{wh} m", va="center",
            fontsize=7, color=WALL_COLOR, ha="right")

    ax.set_xlim(-0.55, W + 0.65)
    ax.set_ylim(-0.45, cam_z + 0.55)
    ax.invert_xaxis()
    ax.set_title("Vista frontal (elevação)", fontsize=11, fontweight="bold")
    ax.set_xlabel("X (m)  [eixo espelhado: arm1 à direita, arm2 à esquerda]", fontsize=9)
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
        c["marker_size_m"] = cfg.get("marker_size_m", 0.19)

        fig, (ax_top, ax_side) = plt.subplots(
            1, 2, figsize=(14, 10),
            gridspec_kw={"width_ratios": [1.1, 1]})

        sup_titles = {
            "U_esq":    "Corredor U — layout e ArUcos",
            "reto_esq": "Corredor Reto — layout e ArUcos",
            "S_esq":    "Corredor S — layout e ArUcos",
        }
        fig.suptitle(sup_titles.get(name, f"{name} — layout"),
                     fontsize=14, fontweight="bold", y=1.01)

        draw_top_view(ax_top, c, name)
        draw_side_view(ax_side, c)
        fig.tight_layout()

        if save:
            out  = Path("output") / OUT_NAME.get(name, f"layout_{name}.png")
            out.parent.mkdir(exist_ok=True)
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"Salvo: {out}")
        else:
            plt.show()


if __name__ == "__main__":
    main()
