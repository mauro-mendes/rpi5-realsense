"""
Generate printable A4 ArUco markers (DICT_4X4_50).

Output: tools/aruco_prints/aruco_XX.png  (A4 at 300 Dpi, portrait)
Marker is centered with white border. Print at 100% scale (no fit-to-page).

Usage:
    pip install opencv-contrib-python   # PC only, if needed
    python tools/generate_aruco.py
"""
import cv2
import numpy as np
from pathlib import Path

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

        fname = out_dir / f"aruco_{marker_id:02d}_{corridor}.png"
        cv2.imwrite(str(fname), canvas)
        print(f"  {fname.name}")

print(f"\nDone. Print all files at 100% scale (no fit-to-page).")
print(f"Physical marker size: {MARKER_SIZE_MM}mm — use this value in corridors.yaml")
