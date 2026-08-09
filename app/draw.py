"""Primitif menggambar di atas frame OpenCV.

Modul ini tidak tahu apa pun soal gestur atau mode — hanya cara menggambar.
Dipakai bersama oleh `app.scenes`, `app.portal`, dan fallback HUD.
"""

import math

import cv2
import numpy as np

from app.gestures import HAND_CONNECTIONS


def draw_text(img, text, center, scale, color, thickness=2,
              font=cv2.FONT_HERSHEY_DUPLEX):
    """Teks ter-align tengah dengan outline hitam supaya terbaca di latar apa pun."""
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = int(center[0] - tw / 2)
    y = int(center[1] + th / 2)
    cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), thickness + 4, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def hsv_color(hue):
    """hue 0..1 -> warna BGR cerah. Nilai di luar 0..1 berputar (wrap)."""
    c = np.uint8([[[int((hue % 1.0) * 179), 255, 255]]])
    bgr = cv2.cvtColor(c, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_note(img, x, y, color, s=1.0):
    """Gambar not balok sederhana (kepala + tangkai) untuk efek meriah."""
    cv2.ellipse(img, (x, y), (int(7 * s), int(5 * s)), 20, 0, 360, color, -1, cv2.LINE_AA)
    stem_x = x + int(6 * s)
    cv2.line(img, (stem_x, y), (stem_x, y - int(22 * s)), color, max(1, int(2 * s)), cv2.LINE_AA)
    cv2.line(img, (stem_x, y - int(22 * s)), (stem_x + int(12 * s), y - int(17 * s)),
             color, max(1, int(2 * s)), cv2.LINE_AA)


def draw_heart(img, cx, cy, size, color):
    pts = []
    for t in np.linspace(0, 2 * math.pi, 40):
        x = 16 * math.sin(t) ** 3
        y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
        pts.append([cx + x * size / 16.0, cy - y * size / 16.0])
    cv2.fillPoly(img, [np.array(pts, np.int32)], color, cv2.LINE_AA)


def draw_hand_skeleton(img, pts, color):
    """Gambar 'tulang' tangan: garis antar sendi + titik di tiap sendi."""
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, p, 5, color, -1, cv2.LINE_AA)
        cv2.circle(img, p, 5, (255, 255, 255), 1, cv2.LINE_AA)


def dim_color(color, f):
    """Redupkan warna BGR ke arah hitam sebesar faktor f (0..1)."""
    return tuple(max(0, min(255, int(round(c * f)))) for c in color)


def lerp_color(c1, c2, t):
    """Interpolasi linear antara dua warna BGR."""
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(c1, c2))


def make_vignette_layer(w, h, strength=0.30, inner=0.45):
    """Lapisan vignette radial siap-kali sebagai uint8 BGR (0 = hitam).

    Dihitung sekali lalu dipakai ulang tiap frame lewat cv2.multiply.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx / max(w - 1, 1)) * 2.0 - 1.0
    ny = (yy / max(h - 1, 1)) * 2.0 - 1.0
    d = np.sqrt(nx * nx + ny * ny)
    t = np.clip((d - inner) / (1.6 - inner), 0.0, 1.0)
    v = 1.0 - strength * t * t
    return (np.stack([v] * 3, axis=-1) * 255.0).astype(np.uint8)
