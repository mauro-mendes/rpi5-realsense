#!/usr/bin/env python3
"""
verifica_passada.py - Controle de qualidade de UMA passada, no laboratorio, na hora.

Le o CSV + o sidecar .meta.json que o realsense_gt.py gerou e responde tres coisas:

  1. ESCALA / POSE  - a altura da bola que o sistema calcula bate com a que voces mediram
     com trena? E a posicao da camera que o ArUco reporta bate com a trena? Erro de escala
     desloca TODA a trajetoria rigidamente, entao isso tem que fechar ANTES de coletar mais.

  2. RASTREAMENTO   - cobertura, lacunas, saltos impossiveis, ruido, inflacao do caminho.

  3. GEOMETRIA      - extensao, estabilidade do z, inclinacao residual, e uma triangulacao
     INDEPENDENTE da camera a partir do ruido radial da profundidade. Se ela bater com a
     pose do ArUco, e uma confirmacao cruzada forte de que o referencial esta certo.

Uso:
    python verifica_passada.py                       # pega o CSV mais recente em trajetorias/
    python verifica_passada.py trajetorias/P01_01_*.csv
    python verifica_passada.py --alvo person         # verifica as linhas da pessoa
"""
import argparse, csv, glob, json, math, os

import numpy as np

OK, AT, ERRO = "OK", "ATENCAO", "ERRO"


def marca(valor, bom, medio, invertido=False):
    """Classifica um numero. invertido=True quando MAIOR e melhor."""
    if invertido:
        return OK if valor >= bom else (AT if valor >= medio else ERRO)
    return OK if valor <= bom else (AT if valor <= medio else ERRO)


def linha(rotulo, texto, status=None):
    s = "" if status is None else ("   " + status)
    print(f"  {rotulo:<24}: {texto}{s}")


def medfilt(a, k=9):
    if len(a) < k:
        return np.asarray(a, float)
    pad = k // 2
    A = np.pad(np.asarray(a, float), ((pad, pad), (0, 0)) if a.ndim > 1 else (pad, pad),
               mode="edge")
    if a.ndim == 1:
        return np.array([np.median(A[i:i + k]) for i in range(len(a))])
    return np.array([np.median(A[i:i + k], axis=0) for i in range(len(a))])


def carrega(path, alvo):
    W, T, P, RAD, DEP, NPX = [], [], [], [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("target") and r["target"] != alvo:
                continue
            try:
                W.append(float(r["wall"]) if r.get("wall") else float(len(W)))
                T.append(float(r["t_s"]) if r.get("t_s") else 0.0)
                P.append([float(r["x_world"]), float(r["y_world"]), float(r["z_world"])])
                RAD.append(float(r["rad_px"]) if r.get("rad_px") else np.nan)
                DEP.append(float(r["depth_m"]) if r.get("depth_m") else np.nan)
                NPX.append(float(r["n_px"]) if r.get("n_px") else np.nan)
            except (ValueError, KeyError):
                continue
    return (np.array(W), np.array(T), np.array(P),
            np.array(RAD), np.array(DEP), np.array(NPX))


def em_movimento(F, W, v_min=0.05):
    """Mascara das amostras em que o alvo REALMENTE se move (velocidade filtrada > v_min).
    Varias metricas so fazem sentido aqui: com a pessoa parada, o caminho 'cru' acumula
    ruido contra um caminho filtrado ~zero (inflacao explode sem significar nada), e a
    triangulacao ganha varios raios repetidos do mesmo ponto, que nao acrescentam base."""
    if len(F) < 3:
        return np.zeros(len(F), bool)
    d = np.linalg.norm(np.diff(F, axis=0), axis=1)
    dt = np.diff(W)
    v = np.divide(d, dt, out=np.zeros_like(d), where=dt > 0)
    m = np.concatenate([[False], v > v_min])
    return m


def triangula_camera(P, F, mov):
    """Camera estimada pelo ruido RADIAL da profundidade: o eixo principal do residuo em
    cada trecho aponta para a camera; as retas se cruzam nela. Medida INDEPENDENTE da pose
    do ArUco. APROXIMADA - precisa de bastante deslocamento lateral p/ o leque de raios
    abrir; por isso e informativa, nunca criterio de reprovacao."""
    if mov.sum() < 80:
        return None, None
    P, F = P[mov], F[mov]
    R = P - F
    segs = np.array_split(np.arange(len(P)), 6)
    A, b, eixos = np.zeros((3, 3)), np.zeros(3), []
    for s in segs:
        if len(s) < 12:
            continue
        C = np.cov(R[s].T)
        w, V = np.linalg.eigh(C)
        if w.max() <= 0:
            continue
        u = V[:, int(np.argmax(w))]
        frac = w.max() / w.sum()
        eixos.append(frac)
        M = np.eye(3) - np.outer(u, u)
        A += M
        b += M @ F[s].mean(axis=0)
    if len(eixos) < 3 or np.linalg.matrix_rank(A) < 3:
        return None, None
    try:
        cam = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None, None
    return cam, float(np.mean(eixos))


def main():
    ap = argparse.ArgumentParser(description="QC de uma passada do realsense_gt.py")
    ap.add_argument("csv", nargs="?", default=None)
    ap.add_argument("--alvo", default="ball", choices=["ball", "person"])
    ap.add_argument("--dir", default="trajetorias")
    a = ap.parse_args()

    path = a.csv
    if not path:
        c = sorted(glob.glob(os.path.join(a.dir, "*.csv")), key=os.path.getmtime)
        if not c:
            raise SystemExit(f"[ERRO] nenhum CSV em {os.path.abspath(a.dir)}")
        path = c[-1]
    if not os.path.exists(path):
        raise SystemExit(f"[ERRO] nao achei {path}")

    meta = {}
    mp = path[:-4] + ".meta.json"
    if os.path.exists(mp):
        meta = json.load(open(mp, encoding="utf-8"))
    par = meta.get("params", {})

    W, T, P, RAD, DEP, NPX = carrega(path, a.alvo)
    n = len(P)
    print("=" * 70)
    print("VERIFICACAO DA PASSADA")
    print("=" * 70)
    linha("arquivo", os.path.basename(path))
    if meta:
        linha("codigo", str(meta.get("code_commit", "?")))
        linha("marcador / ref-id", f"{par.get('marker_size_m')} m  /  id {par.get('ref_id')}  "
                                   f"level={'SIM' if par.get('level') else 'nao'}")
    if n < 20:
        raise SystemExit(f"[ERRO] so {n} amostras de '{a.alvo}' - passada curta demais p/ avaliar")
    dur = float(T[-1] - T[0]) if T[-1] > T[0] else float(W[-1] - W[0])
    linha("alvo / amostras", f"{a.alvo}   {n} amostras em {dur:.1f} s")

    F = medfilt(P, 9)
    veredito = []

    # ---------------------------------------------------------------- ESCALA
    print("\n[ESCALA / POSE]")
    zmed = float(np.median(P[:, 2]))
    bh = par.get("ball_height_m") if a.alvo == "ball" else None
    linha("z mediano do alvo", f"{zmed:+.3f} m")
    if bh:
        dz = zmed - float(bh)
        st = marca(abs(dz), 0.10, 0.20)
        linha("altura medida (trena)", f"{float(bh):.3f} m")
        linha("diferenca", f"{dz*100:+.1f} cm", st)
        if st != OK:
            veredito.append("z do alvo nao bate com a trena -> escala/pose suspeitas")
    elif a.alvo == "ball":
        # Sem trena por participante: basta a FAIXA plausivel. A escala em si ja foi
        # conferida por --cam-height, que e do SETUP e nao muda entre pessoas; aqui a faixa
        # so pega bola errada (outro objeto verde) ou escala grosseiramente fora.
        lo, hi = par.get("ball_range_m", [1.40, 2.10])
        dentro = float(lo) <= zmed <= float(hi)
        linha("faixa plausivel", f"{float(lo):.2f} a {float(hi):.2f} m",
              OK if dentro else ERRO)
        if dentro:
            print(f"      -> altura da bola MEDIDA pelo sistema: {zmed:.3f} m "
                  f"(nao precisou de trena)")
        else:
            veredito.append(f"z={zmed:.2f} m fora da faixa - bola errada ou escala furada")

    cam = np.array(meta.get("cam_pos_in_marker", [np.nan] * 3), float)
    if np.isfinite(cam).all():
        linha("camera (ArUco)", f"({cam[0]:+.2f}, {cam[1]:+.2f}, {cam[2]:+.2f}) m  "
                                f"| horiz {math.hypot(cam[0], cam[1]):.2f} m")
    sc = meta.get("scale_check")
    if sc:
        st = marca(abs(float(sc) - 1), 0.05, 0.10)
        linha("conferencia do meta", f"s = {float(sc):.3f}", st)
        if st != OK:
            desl = abs(1 - 1 / float(sc)) * float(np.linalg.norm(cam))
            veredito.append(f"escala fora por {abs(float(sc)-1)*100:.0f}% "
                            f"-> desloca tudo {desl*100:.0f} cm")
    elif not par.get("cam_height_m"):
        linha("conferencia do meta", "sem --cam-height/--cam-dist: escala NAO conferida", AT)

    # ----------------------------------------------------------- RASTREAMENTO
    print("\n[RASTREAMENTO]")
    dt = np.diff(W)
    dt = dt[dt > 0]
    hz = 1.0 / float(np.median(dt)) if len(dt) else 0.0
    pedido = float(par.get("rate_hz", 10.0))
    linha("taxa efetiva", f"{hz:.1f} Hz  (pedido {pedido:.1f})",
          marca(abs(hz - pedido) / pedido, 0.2, 0.4))
    lim = 2.5 / pedido
    lac = dt[dt > lim]
    cob = 100.0 * (1 - lac.sum() / (W[-1] - W[0])) if W[-1] > W[0] else 100.0
    st = marca(cob, 90, 75, invertido=True)
    linha("cobertura", f"{cob:.0f}% do tempo com o alvo visivel", st)
    if st != OK:
        veredito.append(f"alvo perdido em {100-cob:.0f}% do tempo")
    if len(lac):
        i = int(np.argmax(dt))
        linha("maior lacuna", f"{dt[i]:.1f} s  @ t={T[i]-T[0]:.1f} s  ({len(lac)} lacunas)")

    d = np.linalg.norm(np.diff(P, axis=0), axis=1)
    v = d / np.diff(W)
    linha("salto entre amostras", f"mediana {np.median(d)*100:.1f} cm  "
                                  f"p90 {np.percentile(d,90)*100:.1f}  max {d.max()*100:.1f}")
    imp = 100.0 * float((v > 2.0).mean())
    st = marca(imp, 1, 5)
    linha("amostras impossiveis", f"{imp:.1f}% com v > 2 m/s", st)
    if st != OK:
        veredito.append(f"{imp:.0f}% das amostras exigem velocidade impossivel")

    res = float(np.linalg.norm(P - F, axis=1).std())
    st = marca(res * 100, 3, 6)
    linha("ruido residual (rms)", f"{res*100:.1f} cm", st)

    # inflacao SO no trecho em movimento: parado, o caminho cru acumula ruido contra um
    # filtrado ~zero e a razao explode sem querer dizer nada.
    mov = em_movimento(F, W)
    mv = mov[1:]
    cru = float(d[mv].sum())
    dfl = np.linalg.norm(np.diff(F, axis=0), axis=1)
    filt = float(dfl[mv].sum())
    linha("tempo em movimento", f"{100*mov.mean():.0f}% das amostras")
    # INFORMATIVA: a inflacao depende da VELOCIDADE (andar devagar infla mais com o mesmo
    # ruido), entao ela nao decide nada. Quem carrega o veredito e o ruido rms acima, que
    # independe da velocidade. Aqui so mostramos o numero e o contexto p/ interpretar.
    if filt > 0.3:
        infl = cru / filt
        vmed = filt / float(np.diff(W)[mv].sum()) if np.diff(W)[mv].sum() > 0 else 0.0
        linha("caminho (em movimento)",
              f"cru {cru:.1f} m -> filtrado {filt:.1f} m  (inflacao {infl:.2f}x  "
              f"a {vmed:.2f} m/s)")
        if infl > 1.3:
            print("      -> use a trajetoria FILTRADA para medir distancia; a crua "
                  "superestima.")
    else:
        linha("caminho (em movimento)", "alvo praticamente parado - metrica nao se aplica")

    if np.isfinite(RAD).any():
        linha("raio da bola (px)", f"mediana {np.nanmedian(RAD):.1f}  "
                                   f"min {np.nanmin(RAD):.1f}  max {np.nanmax(RAD):.1f}")

    # -------------------------------------------------------------- GEOMETRIA
    print("\n[GEOMETRIA]")
    linha("extensao", f"x {np.ptp(P[:,0]):.2f} m   y {np.ptp(P[:,1]):.2f} m")
    r = np.linalg.norm(P[:, :2], axis=1)
    linha("distancia da origem", f"{r.min():.2f} a {r.max():.2f} m")
    st = marca(P[:, 2].std() * 100, 5, 12)
    linha("z: desvio", f"{P[:,2].std()*100:.1f} cm", st)

    if np.isfinite(cam).all():
        dcam = np.linalg.norm(F[:, :2] - cam[:2], axis=1)
        if np.ptp(dcam) > 0.5:
            A = np.c_[dcam, np.ones(len(dcam))]
            sl = np.linalg.lstsq(A, F[:, 2], rcond=None)[0][0]
            ang = math.degrees(math.atan(abs(sl)))
            st = marca(ang, 2, 5)
            linha("z vs distancia", f"{ang:.1f} deg de inclinacao residual", st)

    # INFORMATIVA: e uma estimativa grosseira, sensivel a quanto o alvo se deslocou
    # lateralmente. Nunca entra no veredito - so confirma quando concorda.
    tri, frac = triangula_camera(P, F, mov)
    if tri is None:
        linha("camera triangulada", "movimento insuficiente p/ triangular (informativo)")
    else:
        linha("camera triangulada", f"({tri[0]:+.2f}, {tri[1]:+.2f})  "
                                    f"[ruido {frac*100:.0f}% radial]  (aproximada)")
        if np.isfinite(cam).all():
            e = float(np.linalg.norm(tri[:2] - cam[:2]))
            if e <= 0.40:
                print(f"      -> concorda com a pose do ArUco ({e*100:.0f} cm): "
                      f"confirmacao cruzada do referencial.")
            else:
                print(f"      -> {e*100:.0f} cm da pose do ArUco. Metodo e grosseiro; "
                      f"so investigue se a ESCALA acima tambem falhou.")

    # --------------------------------------------------------------- VEREDITO
    print("\n" + "=" * 70)
    if not veredito:
        print("VEREDITO: passada utilizavel. Nenhum problema detectado.")
    else:
        print("VEREDITO: revisar antes de coletar mais -")
        for x in veredito:
            print("  - " + x)
    print("=" * 70)

    # ----------------------------------------------------------------- FIGURA
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(17, 5.5))
        s = ax[0].scatter(P[:, 0], P[:, 1], c=W - W[0], cmap="plasma", s=9)
        ax[0].plot(P[:, 0], P[:, 1], lw=.4, alpha=.5, color="green")
        ax[0].plot(0, 0, "r*", ms=18, label="origem (marcador)")
        if np.isfinite(cam).all():
            ax[0].plot(cam[0], cam[1], "D", color="darkorange", ms=10, label="camera (ArUco)")
        if tri is not None:
            ax[0].plot(tri[0], tri[1], "s", color="royalblue", ms=9, label="camera (ruido)")
        ax[0].set_aspect("equal"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
        ax[0].set_title("planta (x,y)"); ax[0].set_xlabel("x (m)"); ax[0].set_ylabel("y (m)")
        plt.colorbar(s, ax=ax[0], label="t (s)")
        ax[1].plot(T - T[0], P[:, 2], lw=.8, label="z")
        if bh:
            ax[1].axhline(float(bh), color="r", ls="--", label=f"trena {float(bh):.2f} m")
        ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
        ax[1].set_title("altura do alvo"); ax[1].set_xlabel("t (s)"); ax[1].set_ylabel("z (m)")
        ax[2].plot(T[1:] - T[0], d * 100, lw=.7)
        ax[2].axhline(10, color="r", ls="--", label="10 cm")
        ax[2].set_yscale("log"); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
        ax[2].set_title("salto entre amostras"); ax[2].set_xlabel("t (s)"); ax[2].set_ylabel("cm")
        out = path[:-4] + "_qc.png"
        fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig)
        print("figura:", out)
    except ImportError:
        print("(matplotlib ausente - relatorio sem figura, o que ja basta p/ decidir)")


if __name__ == "__main__":
    main()
