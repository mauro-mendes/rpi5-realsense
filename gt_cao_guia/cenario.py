#!/usr/bin/env python3
"""
cenario.py - O recinto do laboratorio: le o cenario.json, converte o referencial do
marcador para o do recinto e desenha a planta com a trajetoria por cima.

POR QUE ISSO EXISTE
-------------------
A trajetoria sai em coordenadas do MARCADOR (o ArUco de referencia e a origem do mundo).
Para desenhar dentro da planta e preciso a transformacao marcador -> recinto: uma
translacao (a posicao do marcador, que esta no cenario.json) e uma ROTACAO.

A rotacao NAO precisa ser medida. O script ja calcula onde a camera esta no referencial
do MARCADOR (`cam_pos_in_marker`, do ArUco) e o cenario.json diz onde ela esta no
referencial do RECINTO (trena). O mesmo vetor visto de dois referenciais: a diferenca
entre os angulos E a rotacao. Como as bordas do marcador estao encostadas/paralelas as
paredes, o resultado e multiplo de 90 graus, e a gente arredonda p/ o mais proximo -
qualquer erro de trena some no arredondamento.

Se o modulo dos dois vetores discordar muito, e sinal de escala errada (o mesmo bug de
~60 cm que investigamos) e a funcao avisa.
"""
from __future__ import annotations

import json
import os

import numpy as np

PADRAO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cenario.json")


# ------------------------------------------------------------------ carregar
def carrega_cenario(path=None):
    """Devolve (cenario, caminho). Cenario vazio se o arquivo nao existir."""
    path = path or PADRAO
    if not os.path.exists(path):
        return {}, path
    with open(path, encoding="utf-8") as f:
        return json.load(f), path


def marcador_ref(cen):
    """(id, dados) do marcador marcado como referencia. None se nao houver cenario."""
    for mid, m in (cen.get("marcadores") or {}).items():
        if m.get("referencia"):
            return int(mid), m
    return None, None


def dist_camera_marcador(cen):
    """(direta_3d, horizontal) da camera ate o marcador de referencia, PELO CENARIO.
    E o que alimenta a conferencia de escala - o usuario nao precisa passar --cam-dist."""
    rid, m = marcador_ref(cen)
    c = cen.get("camera")
    if m is None or c is None:
        return None, None
    dx, dy = c["x_m"] - m["x_m"], c["y_m"] - m["y_m"]
    dz = c["z_m"] - m.get("z_m", 0.0)
    h = float(np.hypot(dx, dy))
    return float(np.sqrt(h * h + dz * dz)), h


# ------------------------------------------------------------------- rotacao
def yaw_do_marcador(cam_in_marker, cen):
    """Rotacao marcador -> recinto, deduzida da camera vista nos dois referenciais.

    cam_in_marker: [x, y, z] da camera no frame do marcador (o ArUco calcula).
    Retorna dict com o yaw bruto, o yaw arredondado p/ 90 graus (o que se usa) e a
    conferencia dos modulos. None se faltar cenario.
    """
    rid, m = marcador_ref(cen)
    c = cen.get("camera")
    if m is None or c is None:
        return None
    v_rec = np.array([c["x_m"] - m["x_m"], c["y_m"] - m["y_m"]], float)
    v_mar = np.array([cam_in_marker[0], cam_in_marker[1]], float)
    n_rec, n_mar = float(np.linalg.norm(v_rec)), float(np.linalg.norm(v_mar))
    if n_rec < 0.2 or n_mar < 0.2:
        return None
    a_rec = np.degrees(np.arctan2(v_rec[1], v_rec[0]))
    a_mar = np.degrees(np.arctan2(v_mar[1], v_mar[0]))
    bruto = (a_rec - a_mar + 180.0) % 360.0 - 180.0
    snap = float(round(bruto / 90.0) * 90.0)
    return {
        "yaw_bruto_deg": round(float(bruto), 2),
        "yaw_deg": snap,
        "residuo_deg": round(float(abs(bruto - snap)), 2),
        "dist_recinto_m": round(n_rec, 3),
        "dist_marcador_m": round(n_mar, 3),
        "erro_escala_m": round(abs(n_rec - n_mar), 3),
    }


def para_recinto(P, yaw_deg, cen):
    """Pontos (N,2 ou N,3) do frame do MARCADOR -> frame do RECINTO."""
    rid, m = marcador_ref(cen)
    if m is None:
        return np.asarray(P)[:, :2]
    th = np.radians(yaw_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    P = np.asarray(P, float)
    return (R @ P[:, :2].T).T + np.array([m["x_m"], m["y_m"]])


# -------------------------------------------------------------------- figura
def plota_plano(P_marcador, tempos, yaw_deg, cen, out_png, titulo=""):
    """Planta do recinto com a trajetoria por cima. P_marcador = pontos no frame do
    marcador; a conversao p/ o recinto acontece aqui."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    rec = cen.get("recinto", {})
    Wr = float(rec.get("largura_m", 2.16))
    Lr = float(rec.get("profundidade_m", 3.50))
    fig, ax = plt.subplots(figsize=(7.6, 9.0))

    ax.add_patch(Rectangle((0, 0), Wr, Lr, fill=False, lw=3.5, ec="#333", zorder=4))
    d = rec.get("divisoria")
    if d:
        ax.plot([d["x0_m"], d["x1_m"]], [d["y_m"], d["y_m"]], lw=7, color="#b03030",
                zorder=4, solid_capstyle="butt")
        ax.annotate("divisoria", ((d["x0_m"] + d["x1_m"]) / 2, d["y_m"] + 0.09),
                    ha="center", color="#b03030", fontsize=8, weight="bold")

    for mid, m in (cen.get("marcadores") or {}).items():
        chao = m.get("tipo") == "chao"
        cor = "k" if m.get("referencia") else "#1565c0"
        s = float(m.get("tamanho_m", 0.14))
        if chao:
            ax.add_patch(Rectangle((m["x_m"] - s/2, m["y_m"] - s/2), s, s,
                                   fc="w", ec=cor, lw=2, zorder=6))
            ax.add_patch(Rectangle((m["x_m"] - s/3, m["y_m"] - s/3), 2*s/3, 2*s/3,
                                   fc=cor, zorder=7))
        else:
            ax.plot([m["x_m"], m["x_m"]], [m["y_m"] - s/2, m["y_m"] + s/2],
                    lw=7, color=cor, zorder=6, solid_capstyle="butt")
        ax.annotate(f"ID {mid}" + ("  (origem)" if m.get("referencia") else ""),
                    (m["x_m"] + 0.10, m["y_m"]), fontsize=7.5, va="center", color=cor)

    c = cen.get("camera")
    if c:
        ax.plot(c["x_m"], c["y_m"], "o", color="#f2c200", ms=14, mec="k", mew=1.4, zorder=8)
        ax.annotate("camera", (c["x_m"], c["y_m"] - 0.26), fontsize=7.5, ha="center",
                    weight="bold")

    P = para_recinto(P_marcador, yaw_deg, cen)
    if len(P) > 1:
        sc = ax.scatter(P[:, 0], P[:, 1], c=tempos, cmap="plasma", s=9, zorder=9)
        ax.plot(P[:, 0], P[:, 1], lw=0.5, color="green", alpha=0.5, zorder=8)
        plt.colorbar(sc, ax=ax, label="t (s)", shrink=0.55)
        ax.plot(P[0, 0], P[0, 1], "o", color="#1b5e20", ms=11, zorder=10, label="inicio")
        ax.plot(P[-1, 0], P[-1, 1], "o", color="#b71c1c", ms=11, zorder=10, label="fim")
        ax.legend(loc="upper right", fontsize=8)

    lo = min(0.0, (c or {}).get("x_m", 0.0)) - 0.4
    hi = max(Wr, (c or {}).get("x_m", Wr)) + 0.4
    ax.set_xlim(lo, hi)
    ax.set_ylim(min(0.0, (c or {}).get("y_m", 0.0)) - 0.4, Lr + 0.4)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.grid(alpha=0.25)
    ax.set_title(titulo or "trajetoria no recinto", fontsize=11, weight="bold")
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)
    return out_png
