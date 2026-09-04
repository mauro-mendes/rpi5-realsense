#!/usr/bin/env python3
"""
gerar_arucos.py - Gera folhas A4 (300 DPI) com marcadores ArUco DICT_4X4_50 prontos p/ imprimir.

Cada folha traz, impresso ao lado do marcador:
  * o ID;
  * o LADO DO QUADRADO PRETO em mm - que e exatamente o que o OpenCV quer em
    `--marker-size` (e o que NAO se deve confundir com a margem branca);
  * o valor pronto pra linha de comando (ex.: --marker-size 0.1400);
  * uma REGUA DE 100 mm para conferir com a trena depois de imprimir. Se a regua nao
    medir 100 mm, a impressora escalou a folha e TODO o tamanho mudou junto.

Imprimir SEM ajuste de escala: "Tamanho real" / "100%" / "Escala: nenhuma".
Nada de "Ajustar a pagina".

Uso:
    python gerar_arucos.py
    python gerar_arucos.py --ids 30 31 32 33 --size-mm 140
    python gerar_arucos.py --ids 40 41 --size-mm 120 --out arucos_pequenos
"""
import argparse, os

import cv2
import numpy as np

DPI = 300
MM = DPI / 25.4                      # pixels por mm
A4_W, A4_H = int(round(210 * MM)), int(round(297 * MM))
DICT_TYPE = cv2.aruco.DICT_4X4_50
CELLS = 6                            # 4x4 de dados + 1 celula de borda preta de cada lado


def gen_marker(dictionary, mid, px):
    """Imagem do marcador com o lado EXTERNO do quadrado preto medindo `px` pixels."""
    big = CELLS * 200
    try:
        img = cv2.aruco.generateImageMarker(dictionary, mid, big)
    except AttributeError:
        img = cv2.aruco.drawMarker(dictionary, mid, big)
    return cv2.resize(img, (px, px), interpolation=cv2.INTER_NEAREST)


def put(canvas, text, x, y, scale, thick=2, color=0):
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def sheet(dictionary, mid, size_mm):
    size_px = int(round(size_mm * MM))
    quiet_mm = (210 - size_mm) / 2.0
    canvas = np.full((A4_H, A4_W), 255, np.uint8)

    marker = gen_marker(dictionary, mid, size_px)
    x0 = (A4_W - size_px) // 2
    y0 = int(round(20 * MM))
    canvas[y0:y0 + size_px, x0:x0 + size_px] = marker

    L = int(22 * MM)                                    # margem esquerda do texto
    y = y0 + size_px + int(round(14 * MM))
    put(canvas, f"ArUco DICT_4X4_50   -   ID {mid}", L, y, 2.2, 5)

    # caixa com o numero que importa
    y += int(round(13 * MM))
    bx0, bx1 = int(18 * MM), A4_W - int(18 * MM)
    by0, by1 = y - int(11 * MM), y + int(15 * MM)
    cv2.rectangle(canvas, (bx0, by0), (bx1, by1), 0, 4)
    put(canvas, "LADO DO QUADRADO PRETO", bx0 + int(6 * MM), y, 1.5, 3)
    put(canvas, f"{size_mm:.1f} mm", bx1 - int(52 * MM), y + int(9 * MM), 3.2, 7)

    y = by1 + int(round(12 * MM))
    put(canvas, f"--marker-size {size_mm/1000:.4f}", L, y, 1.8, 4)
    y += int(round(9 * MM))
    put(canvas, "(o lado do QUADRADO PRETO, borda preta inclusa.", L, y, 1.0, 2)
    y += int(round(7 * MM))
    put(canvas, f" NAO inclui a margem branca de {quiet_mm:.0f} mm em volta.)", L, y, 1.0, 2)

    # regua de conferencia
    y += int(round(14 * MM))
    put(canvas, "CONFIRA COM A TRENA - esta linha tem exatamente 100 mm:", L, y, 1.1, 3)
    y += int(round(9 * MM))
    rx0 = L
    rx1 = rx0 + int(round(100 * MM))
    cv2.line(canvas, (rx0, y), (rx1, y), 0, 5)
    for xx in (rx0, rx1):
        cv2.line(canvas, (xx, y - int(4 * MM)), (xx, y + int(4 * MM)), 0, 5)
    for i in range(1, 10):
        xt = rx0 + int(round(i * 10 * MM))
        cv2.line(canvas, (xt, y - int(2 * MM)), (xt, y + int(2 * MM)), 0, 3)
    put(canvas, "100 mm", rx1 + int(6 * MM), y + int(3 * MM), 1.1, 3)

    y += int(round(12 * MM))
    put(canvas, "Se nao medir 100 mm a impressora ESCALOU: reimprima em", L, y, 1.0, 2)
    y += int(round(7 * MM))
    put(canvas, "'Tamanho real' / '100%' / sem 'ajustar a pagina'.", L, y, 1.0, 2)
    assert y < A4_H - int(10 * MM), f"texto passou do fim da pagina (y={y/MM:.0f}mm)"
    return canvas


def main():
    ap = argparse.ArgumentParser(description="Folhas A4 com ArUco DICT_4X4_50 p/ imprimir")
    ap.add_argument("--ids", type=int, nargs="+", default=[30, 31, 32, 33])
    ap.add_argument("--size-mm", type=float, default=140.0,
                    help="lado do QUADRADO PRETO em mm (max ~150 p/ sobrar margem branca em A4)")
    ap.add_argument("--out", default="arucos")
    a = ap.parse_args()

    if a.size_mm > 155:
        raise SystemExit("size-mm > 155 nao deixa margem branca suficiente em A4 (quiet zone).")
    quiet = (210 - a.size_mm) / 2.0
    cell = a.size_mm / CELLS
    if quiet < cell:
        print(f"[aviso] margem branca {quiet:.0f} mm < 1 celula ({cell:.0f} mm) - "
              f"a deteccao pode sofrer em angulo rasante.")

    os.makedirs(a.out, exist_ok=True)
    try:
        dictionary = cv2.aruco.getPredefinedDictionary(DICT_TYPE)
    except AttributeError:
        dictionary = cv2.aruco.Dictionary_get(DICT_TYPE)

    pages = []
    for mid in a.ids:
        img = sheet(dictionary, mid, a.size_mm)
        p = os.path.join(a.out, f"aruco_{mid:02d}_{a.size_mm:.0f}mm_A4.png")
        cv2.imwrite(p, img)
        pages.append(img)
        print(f"  {p}")

    try:                                        # PDF unico (opcional, se houver Pillow)
        from PIL import Image
        ims = [Image.fromarray(p).convert("RGB") for p in pages]
        pdf = os.path.join(a.out, f"arucos_{a.size_mm:.0f}mm_A4.pdf")
        ims[0].save(pdf, save_all=True, append_images=ims[1:], resolution=float(DPI))
        print(f"  {pdf}   <- imprima este (1 marcador por pagina)")
    except Exception as e:
        print(f"  (PDF nao gerado: {e} - imprima os PNG)")

    print(f"\nID(s): {a.ids}   lado do quadrado preto: {a.size_mm:.1f} mm   "
          f"margem branca: {quiet:.0f} mm")
    print(f"Use:  --marker-size {a.size_mm/1000:.4f}")
    print("IMPRIMA EM 'TAMANHO REAL' (100%). Confira a regua de 100 mm com a trena.")


if __name__ == "__main__":
    main()
