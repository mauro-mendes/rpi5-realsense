"""
Generate printable A4 ArUco markers (DICT_4X4_50) — IDs 20–26.

Output:
  tools/aruco_prints/aruco_XX.png          — A4 300 DPI, individual (para conferir)
  tools/aruco_prints/all_arucos.pdf        — PDF 7 páginas (UMA por marker) → IMPRIMIR ESTE
  tools/aruco_prints/all_arucos_preview.png — grade 2×4, visão geral

Ao imprimir all_arucos.pdf:
  Tamanho real / 100% / sem ajustar à página.
  Cada marker impresso terá exatamente 190mm de lado.

Usage:
    conda run -n rpi5-realsense python tools/generate_aruco.py
"""
import cv2
import numpy as np
from pathlib import Path
from PIL import Image as PILImage

# ── Tamanhos ──────────────────────────────────────────────────────────────────
A4_W_PX      = 2480   # A4 portrait 300 DPI
A4_H_PX      = 3508
MARKER_PX    = 2244   # 190mm @ 300 DPI  (borda branca de ~10mm em volta)
MARKER_MM    = 190    # tamanho físico impresso

# ── Descrição de cada marcador (para o label) ─────────────────────────────────
MARKER_INFO = {
    20: ("x=1.345, y=2.11", "ilha — canto esq (face inferior)"),
    21: ("x=2.220, y=2.11", "ilha — canto dir (face inferior)"),
    22: ("x=0.105, y=3.305", "parede fundo — sobre arm1"),
    23: ("x=2.215, y=3.305", "parede fundo — sobre arm2"),
    24: ("x=2.530, y=5.515", "extensão — fundo esq (arm2)"),
    25: ("x=3.375, y=5.515", "extensão — fundo dir (arm2)"),
    26: ("x=1.280, y=0.150", "entrada — parede esq da ilha (MEDIR e actualizar YAML!)"),
}

out_dir = Path(__file__).parent / "aruco_prints"
out_dir.mkdir(exist_ok=True)

# ── API ArUco ─────────────────────────────────────────────────────────────────
try:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
except AttributeError:
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)


def make_a4_canvas(marker_id: int) -> np.ndarray:
    """Gera canvas A4 branco com o marker centrado e label na base."""
    try:
        marker_img = cv2.aruco.generateImageMarker(dictionary, marker_id, MARKER_PX)
    except AttributeError:
        marker_img = cv2.aruco.drawMarker(dictionary, marker_id, MARKER_PX)

    canvas = np.ones((A4_H_PX, A4_W_PX), dtype=np.uint8) * 255
    x0 = (A4_W_PX - MARKER_PX) // 2
    y0 = (A4_H_PX - MARKER_PX) // 2
    canvas[y0:y0 + MARKER_PX, x0:x0 + MARKER_PX] = marker_img

    pos, note = MARKER_INFO.get(marker_id, ("?", ""))
    line1 = f"ID {marker_id}  |  {MARKER_MM}mm  |  DICT_4X4_50"
    line2 = f"{pos}  —  {note}"
    for i, txt in enumerate([line1, line2]):
        cv2.putText(canvas, txt, (A4_W_PX // 2 - 500, A4_H_PX - 130 + i * 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.2, 0, 4)
    return canvas


# ── Gera PNGs individuais + coleta imagens para PDF e preview ─────────────────
all_pages: list[PILImage.Image] = []
preview_cells: list[np.ndarray] = []

PREVIEW_SZ = 700  # px por célula na grade de preview

for mid in sorted(MARKER_INFO.keys()):
    canvas = make_a4_canvas(mid)

    # PNG individual A4
    png_path = out_dir / f"aruco_{mid:02d}.png"
    cv2.imwrite(str(png_path), canvas)
    print(f"  {png_path.name}")

    # acumula página para PDF
    all_pages.append(PILImage.fromarray(canvas))

    # célula de preview (proporção A4, PREVIEW_SZ de largura)
    ph = int(PREVIEW_SZ * A4_H_PX / A4_W_PX)
    cell = cv2.resize(canvas, (PREVIEW_SZ, ph), interpolation=cv2.INTER_AREA)
    preview_cells.append(cell)

# ── PDF combinado (1 marker por página) ──────────────────────────────────────
pdf_path = out_dir / "all_arucos.pdf"
all_pages[0].save(
    str(pdf_path), "PDF", resolution=300,
    save_all=True, append_images=all_pages[1:]
)
print(f"\n  PDF combinado → {pdf_path.name}  ({len(all_pages)} páginas)")
print(f"  Imprimir: Tamanho real | 100% | sem 'ajustar à página'")

# ── Preview grid 2 × 4 (7 markers, última célula vazia) ──────────────────────
rows, cols = 4, 2
ph = preview_cells[0].shape[0]
grid = np.ones((rows * ph, cols * PREVIEW_SZ), dtype=np.uint8) * 220
for idx, cell in enumerate(preview_cells):
    r, c = divmod(idx, cols)
    y0, x0 = r * ph, c * PREVIEW_SZ
    grid[y0:y0 + cell.shape[0], x0:x0 + PREVIEW_SZ] = cell

# borda entre células
for r in range(1, rows):
    grid[r * ph - 2:r * ph + 2, :] = 100
for c in range(1, cols):
    grid[:, c * PREVIEW_SZ - 2:c * PREVIEW_SZ + 2] = 100

prev_path = out_dir / "all_arucos_preview.png"
cv2.imwrite(str(prev_path), grid)
print(f"  Preview grid  → {prev_path.name}")
print(f"\nTamanho físico: {MARKER_MM}mm | Dicionário: DICT_4X4_50 | IDs: {sorted(MARKER_INFO)}")
