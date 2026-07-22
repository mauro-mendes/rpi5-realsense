"""
Helper de gravação de vídeo em H.264 pronto para WhatsApp, SEM dupla compressão.

Recebe frames BGR crus e os encoda direto via ffmpeg em H.264 + yuv420p +
faststart (CRF 18 = praticamente sem perda). Como o input é frame cru, não há
transcodificação em cima de um vídeo já comprimido → qualidade preservada.

Se o ffmpeg não for encontrado, cai para cv2.VideoWriter (mp4v) com aviso.

Cópia do tools/_ffwriter.py do colete (smart-vest) — mantido idêntico de propósito
para os dois dispositivos gravarem no mesmo formato H.264/mp4.
"""

from __future__ import annotations

import os
import shutil
import subprocess


def find_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    # fallbacks comuns (ffmpeg no PATH do sistema no RPi)
    for c in (
        r"C:\Users\User\.conda\envs\ocrpdf55\Library\bin\ffmpeg.exe",
        "/usr/bin/ffmpeg",
    ):
        if os.path.exists(c):
            return c
    return None


class FFmpegWriter:
    """Escritor de vídeo H.264 (WhatsApp-ready). API compatível: write()/release()."""

    def __init__(self, path, width, height, fps, crf=18, preset="medium"):
        self.path = path
        self._proc = None
        self._fallback = None
        ff = find_ffmpeg()
        fps = float(fps) if fps and fps > 0 else 30.0
        if ff is None:
            import cv2
            print("[ffwriter] ffmpeg não encontrado — gravando em mp4v (converta depois p/ WhatsApp)")
            self._fallback = cv2.VideoWriter(
                path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            return
        cmd = [
            ff, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "-",
            "-an",
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            path,
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame):
        if self._fallback is not None:
            self._fallback.write(frame)
            return
        self._proc.stdin.write(frame.tobytes())

    def release(self):
        if self._fallback is not None:
            self._fallback.release()
            return
        try:
            self._proc.stdin.close()
        finally:
            self._proc.wait()
