"""
Generate printable A4 ArUco markers (DICT_4X4_50).

Output:
  tools/aruco_prints/aruco_XX.png  (A4 300 DPI, para conferir)
  tools/aruco_prints/aruco_XX.pdf  (A4 PDF, para imprimir)

Ao imprimir o PDF: tamanho real / 100% / sem ajustar à página.
O marker impresso terá exatamente 190mm de lado.

Usage:
    python tools/generate_aruco.py
"""
import cv2
import numpy as np
from pathlib import Path
from PIL import Image as PILImage

# A4 portrait at 300 DPI
A4_W_PX = 2480
A4_H_PX = 3508

# Marker: 190mm = 2244px at 300 DPI  (leaves ~118px / ~10mm white border)
MARKER_PX = 2244
MARKER_SIZE_MM = 190   # printed physical size — used in YAML config

IDS_BY_CORRIDOR = {
    "reto":     [0, 1],
    "S_esq_dir": [10, 11, 12, 13],
    "U_esq":    [20, 21, 22, 23],
}

out_dir = Path(__file__).parent / "aruco_prints"
out_dir.mkdir(exist_ok=True)

# Support both old (4.x) and new (4.8+) OpenCV aruco API
try:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
except AttributeError:
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

for corridor, ids in IDS_BY_CORRIDOR.items():
    for marker_id in ids:
        try:
            marker_img = cv2.aruco.generateImageMarker(dictionary, marker_id, MARKER_PX)
        except AttributeError:
            marker_img = cv2.aruco.drawMarker(dictionary, marker_id, MARKER_PX)

        # Place marker centered on white A4 canvas
        canvas = np.ones((A4_H_PX, A4_W_PX), dtype=np.uint8) * 255
        x0 = (A4_W_PX - MARKER_PX) // 2
        y0 = (A4_H_PX - MARKER_PX) // 2
        canvas[y0:y0 + MARKER_PX, x0:x0 + MARKER_PX] = marker_img

        # Label at bottom
        label = f"ID {marker_id}  |  {corridor}  |  {MARKER_SIZE_MM}mm"
        cv2.putText(canvas, label, (A4_W_PX // 2 - 400, A4_H_PX - 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, 0, 5)

        png_path = out_dir / f"aruco_{marker_id:02d}_{corridor}.png"
        cv2.imwrite(str(png_path), canvas)

        # PDF A4 a 300 DPI — imprimir em tamanho real (100%)
        pdf_path = out_dir / f"aruco_{marker_id:02d}_{corridor}.pdf"
        pil_img = PILImage.fromarray(canvas)
        pil_img.save(str(pdf_path), "PDF", resolution=300)

        print(f"  {png_path.name}  +  {pdf_path.name}")

print(f"\nPDFs prontos em tools/aruco_prints/")
print(f"Imprimir com: Adobe Reader / SumatraPDF / navegador")
print(f"Opção: Tamanho real  |  100%  |  sem 'ajustar à página'")
print(f"Tamanho físico do marker: {MARKER_SIZE_MM}mm")
