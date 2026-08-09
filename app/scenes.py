"""Efek visual untuk tiap gestur.

Semua state partikel dipegang instance, bukan global modul, supaya dua adegan
tidak pernah saling mencemari dan test bisa memulai dari kondisi bersih.
"""

import math
import os
import random

import cv2

from app import config
from app.draw import draw_heart, draw_note, draw_text, hsv_color

LABELS = {
    "PEACE": "FOTO KITA BLURRR",
    "HEART": "I LOVE YOU",
    "ILY": "GOKILL",
    "OK": "OKE",
    "KICAW": "KICAW MANIA!",
}

# Thumbnail contoh gestur yang ditempel di pojok saat efek Peace aktif.
BLUR_THUMB = None
if os.path.exists(config.IMG_BLUR):
    _img = cv2.imread(config.IMG_BLUR)
    if _img is not None:
        BLUR_THUMB = cv2.resize(_img, (160, 90))
else:
    print(f"[warn] gambar tidak ditemukan: {config.IMG_BLUR}")


class GestureScenes:
    """Menggambar efek besar di tengah layar sesuai gestur aktif."""

    def __init__(self):
        self.hearts = []    # [x, y, kecepatan, ukuran]
        self.confetti = []  # [x, y, vy, vx, warna, ukuran]

    def render(self, frame, gesture, t, effect_fn=None,
               prev_effect_fn=None, blend=1.0):
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        pulse = 1.0 + 0.12 * math.sin(t * 6.0)
        hue = (t * 0.25) % 1.0

        if gesture == "PEACE":
            self._peace(frame, cx, cy, w, h, pulse, effect_fn,
                        prev_effect_fn, blend)
        elif gesture == "HEART":
            self._heart(frame, cx, cy, w, h, pulse)
        elif gesture == "ILY":
            self._ily(frame, cx, cy, pulse, hue)
        elif gesture == "OK":
            self._ok(frame, cx, cy, pulse)
        elif gesture == "KICAW":
            self._kicaw(frame, cx, cy, w, h, t, pulse, hue)

    def _peace(self, frame, cx, cy, w, h, pulse, effect_fn,
               prev_effect_fn, blend):
        """Seluruh layar kena filter aktif, lalu judul di tengah."""
        if effect_fn is None:
            k = 45
            frame[:] = cv2.GaussianBlur(frame, (k, k), 0)
        else:
            from app.effects import apply_full_frame
            frame[:] = apply_full_frame(frame, effect_fn, prev_effect_fn, blend)
        draw_text(frame, LABELS["PEACE"], (cx, cy), 1.8 * pulse,
                  (255, 255, 255), 3)
        if BLUR_THUMB is not None:
            bh, bw = BLUR_THUMB.shape[:2]
            frame[h - bh - 15:h - 15, w - bw - 15:w - 15] = BLUR_THUMB

    def _heart(self, frame, cx, cy, w, h, pulse):
        """Partikel hati naik ke atas."""
        if random.random() < 0.4:
            self.hearts.append([random.randint(60, w - 60), h - 40,
                                random.uniform(2.5, 5.5), random.randint(14, 30)])
        for hp in self.hearts:
            hp[1] -= hp[2]
        self.hearts[:] = [hp for hp in self.hearts if hp[1] > -40]
        for x, y, _, s in self.hearts:
            draw_heart(frame, int(x), int(y), s, (180, 105, 255))
        draw_text(frame, LABELS["HEART"], (cx, cy), 1.7 * pulse, (180, 105, 255), 3)
        draw_heart(frame, cx, cy + 70, 40 * pulse, (180, 105, 255))

    def _ily(self, frame, cx, cy, pulse, hue):
        draw_text(frame, LABELS["ILY"], (cx, cy), 2.0 * pulse, hsv_color(hue), 3)
        draw_text(frame, "\\m/  ROCK ON  \\m/", (cx, cy + 70), 0.9, (0, 255, 255), 2)

    def _ok(self, frame, cx, cy, pulse):
        draw_text(frame, LABELS["OK"], (cx, cy), 2.2 * pulse, (0, 230, 0), 3)
        r = int(60 * pulse)
        cv2.circle(frame, (cx, cy + 90), r, (0, 230, 0), 4, cv2.LINE_AA)

    def _kicaw(self, frame, cx, cy, w, h, t, pulse, hue):
        """Konfeti jatuh + not balok beterbangan mengelilingi tulisan."""
        if random.random() < 0.9:
            self.confetti.append([
                random.randint(0, w), -10,
                random.uniform(3.0, 7.0), random.uniform(-1.5, 1.5),
                hsv_color(random.random()), random.randint(6, 13),
            ])
        for c in self.confetti:
            c[0] += c[3]
            c[1] += c[2]
        self.confetti[:] = [c for c in self.confetti if c[1] < h + 10]
        for x, y, _, _, col, s in self.confetti:
            cv2.rectangle(frame, (int(x), int(y)), (int(x + s), int(y + s)), col, -1)
        for i in range(7):
            nx = int(cx + math.sin(t * 3.0 + i) * (160 + i * 22))
            ny = int(cy + math.cos(t * 2.0 + i * 1.3) * 70 - 30)
            draw_note(frame, nx, ny, hsv_color((t * 0.5 + i * 0.14) % 1.0), 1.3)
        draw_text(frame, LABELS["KICAW"], (cx, cy), 1.9 * pulse, (0, 220, 255), 4)
        draw_text(frame, "~ cuit cuit cuit ~", (cx, cy + 70), 1.1, hsv_color(hue), 2)
