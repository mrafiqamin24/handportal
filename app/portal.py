"""Portal: jendela persegi di antara dua ujung telunjuk, isinya kena filter.

Aturan permanen modul ini: **isi portal tidak pernah di-warp perspektif.**
Wilayah kotak di-crop, filter diterapkan apa adanya, hasilnya ditempel kembali
di posisi yang sama lalu di-mask sesuai bentuk kotak. Warping-lah yang dulu
membuat wajah tampak gepeng.

Diadaptasi dari milan-kb/fancy-fingers (MIT).
"""

import math
import random

import cv2
import numpy as np

from app import config
from app.draw import dim_color


def build_box(tip_a, tip_b, min_size=20):
    """Kotak sejajar sumbu dari dua titik diagonal (ujung telunjuk tiap tangan).

    Persis seperti menarik kotak seleksi antara dua titik diagonal: tangan mana
    pun yang lebih tinggi otomatis membentuk sudut atas di sisinya sendiri.
    `min_size` mencegah kotak menciut jadi seiris garis saat kedua tangan
    sejajar. Return titik berurutan TL, TR, BR, BL.
    """
    x_min, x_max = sorted([float(tip_a[0]), float(tip_b[0])])
    y_min, y_max = sorted([float(tip_a[1]), float(tip_b[1])])

    half = min_size / 2.0
    if (y_max - y_min) < min_size:
        mid = (y_max + y_min) / 2.0
        y_min, y_max = mid - half, mid + half
    if (x_max - x_min) < min_size:
        mid = (x_max + x_min) / 2.0
        x_min, x_max = mid - half, mid + half

    return np.array([[x_min, y_min], [x_max, y_min],
                     [x_max, y_max], [x_min, y_max]], dtype=np.float32)


class QuadSmoother:
    """Rata-rata bergerak eksponensial untuk keempat sudut kotak.

    Menghilangkan getaran antar-frame dari landmark mentah. `alpha` kecil =
    lebih halus tapi lebih lambat; besar = lebih gesit tapi lebih bergetar.
    Perubahan jumlah tangan ditangani lewat `reset()`.
    """

    def __init__(self, alpha=0.35):
        self.alpha = alpha
        self.smoothed = None

    def update(self, quad_pts):
        if self.smoothed is None or self.smoothed.shape != quad_pts.shape:
            self.smoothed = quad_pts.copy()
        else:
            self.smoothed = (self.alpha * quad_pts
                             + (1 - self.alpha) * self.smoothed)
        return self.smoothed

    def reset(self):
        self.smoothed = None


def render_portal(frame, effect_fn, quad_pts, prev_effect_fn=None, blend=1.0,
                  alpha=1.0):
    """Tempelkan hasil `effect_fn` di dalam wilayah kotak. Mengubah `frame`.

    Crop dilebarkan sebesar FEATHER_PX supaya pita mask yang dilembutkan jatuh
    di DALAM crop — blur yang terpotong di batas crop akan mematikan tepi
    lembutnya. `prev_effect_fn` + `blend` melakukan crossfade saat ganti filter;
    `alpha` (0..1) memudarkan seluruh komposit saat portal muncul/hilang.

    Penjepitan x_min/x_max/y_min/y_max ke ukuran frame — plus keluar lebih awal
    kalau wilayahnya di bawah 5 px — adalah yang membuat portal separuh (atau
    sepenuhnya) di luar layar tetap aman. Jangan dihilangkan.
    """
    h, w = frame.shape[:2]
    pad = config.FEATHER_PX
    x_min = max(int(math.floor(np.min(quad_pts[:, 0]))) - pad, 0)
    x_max = min(int(math.ceil(np.max(quad_pts[:, 0]))) + pad, w)
    y_min = max(int(math.floor(np.min(quad_pts[:, 1]))) - pad, 0)
    y_max = min(int(math.ceil(np.max(quad_pts[:, 1]))) + pad, h)

    if x_max - x_min < 5 or y_max - y_min < 5:
        return

    # crop hanyalah view — setiap filter mengembalikan array baru, tidak
    # pernah menulisi masukannya
    crop = frame[y_min:y_max, x_min:x_max]
    processed = effect_fn(crop)
    if prev_effect_fn is not None and blend < 1.0:
        prev = prev_effect_fn(crop)
        processed = cv2.addWeighted(processed, float(blend), prev,
                                    1.0 - float(blend), 0)

    # mask ter-feather: isi bentuk kotak, lalu lembutkan tepinya beberapa pixel
    rh, rw = y_max - y_min, x_max - x_min
    mask = np.zeros((rh, rw), dtype=np.uint8)
    local = quad_pts.astype(np.float32) - np.array([x_min, y_min], np.float32)
    cv2.fillConvexPoly(mask, local.astype(np.int32), 255)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=config.FEATHER_SIGMA)

    if alpha < 1.0:
        mask = cv2.multiply(
            mask, np.array([int(round(255 * alpha))], dtype=np.uint8),
            scale=1.0 / 255.0)

    mask3 = cv2.merge([mask, mask, mask])
    region = frame[y_min:y_max, x_min:x_max]
    out = cv2.add(cv2.multiply(processed, mask3, scale=1.0 / 255.0),
                  cv2.multiply(region, 255 - mask3, scale=1.0 / 255.0))
    frame[y_min:y_max, x_min:x_max] = out


def draw_glow(frame, quad_pts, color, intensity):
    """Rim-glow aditif di sekeliling tepi portal.

    Halo Gaussian lebar plus pita tipis terang yang memeluk garis luar. Tidak
    ada garis tegas sama sekali — supaya terbaca sebagai jendela energi, bukan
    bingkai. `intensity` 0..1 menskalakan bloom-nya.
    """
    h, w = frame.shape[:2]
    pad = config.GLOW_PAD
    x_min = max(int(math.floor(np.min(quad_pts[:, 0]))) - pad, 0)
    x_max = min(int(math.ceil(np.max(quad_pts[:, 0]))) + pad, w)
    y_min = max(int(math.floor(np.min(quad_pts[:, 1]))) - pad, 0)
    y_max = min(int(math.ceil(np.max(quad_pts[:, 1]))) + pad, h)

    if x_max - x_min < 5 or y_max - y_min < 5:
        return

    rw, rh = x_max - x_min, y_max - y_min
    local = quad_pts.astype(np.float32) - np.array([x_min, y_min], np.float32)
    fill = np.zeros((rh, rw), dtype=np.uint8)
    cv2.fillConvexPoly(fill, local.astype(np.int32), 255)

    # Bloom lebar dihitung di skala 1/4 (radius fisik sama, biaya ~1/16), lalu
    # diperbesar lagi. rim = blur(fill) - fill = cahaya lembut DI LUAR tepi;
    # core = DoG dari rim = pita tipis terang yang memeluk garis luar.
    ds = 0.25
    sw, sh = max(1, int(round(rw * ds))), max(1, int(round(rh * ds)))
    small = cv2.resize(fill, (sw, sh), interpolation=cv2.INTER_AREA)
    halo_s = cv2.GaussianBlur(small, (0, 0),
                              sigmaX=max(1.0, config.GLOW_SIGMA * ds))
    rim_s = cv2.subtract(halo_s, small)
    core_s = cv2.subtract(rim_s, cv2.GaussianBlur(rim_s, (0, 0), sigmaX=1.3))
    rim = cv2.resize(rim_s, (rw, rh), interpolation=cv2.INTER_LINEAR)
    core = cv2.resize(core_s, (rw, rh), interpolation=cv2.INTER_LINEAR)

    # pewarnaan dalam aritmetika uint8 (menghindari promosi ke float64)
    col = np.array(color, dtype=np.uint8)
    rim3 = cv2.multiply(cv2.merge([rim, rim, rim]), col, scale=1.0 / 255.0)
    core3 = cv2.multiply(cv2.merge([core, core, core]), col, scale=1.0 / 255.0)
    glow = cv2.addWeighted(rim3, 0.55, core3, 1.0, 0)
    glow = cv2.multiply(
        glow, np.array([max(1, int(round(255 * intensity)))] * 3, np.uint8),
        scale=1.0 / 255.0)

    frame[y_min:y_max, x_min:x_max] = cv2.add(
        frame[y_min:y_max, x_min:x_max], glow)


def draw_corner_accents(frame, quad_pts, color, t, alpha=1.0, length=16):
    """Bracket sudut ala target AR di keempat sudut, menghadap ke tengah kotak,
    dengan kerlip halus per sudut. `alpha` memudarkannya bersama portal."""
    cx = float(np.mean(quad_pts[:, 0]))
    cy = float(np.mean(quad_pts[:, 1]))
    for i, pt in enumerate(quad_pts):
        x, y = int(round(pt[0])), int(round(pt[1]))
        dx = 1 if x < cx else -1
        dy = 1 if y < cy else -1
        shimmer = 0.78 + 0.22 * math.sin(t * 3.1 + i * 1.7)
        L = int(length * shimmer * alpha)
        if L < 3:
            continue
        cv2.circle(frame, (x, y), 5, dim_color(color, 0.30 * alpha), -1,
                   cv2.LINE_AA)
        cv2.line(frame, (x, y), (x + dx * L, y),
                 dim_color(color, 0.92 * alpha), 2, cv2.LINE_AA)
        cv2.line(frame, (x, y), (x, y + dy * L),
                 dim_color(color, 0.92 * alpha), 2, cv2.LINE_AA)


class Particle:
    __slots__ = ("x", "y", "vx", "vy", "life", "max_life", "size")

    def __init__(self, x, y, vx, vy, life, size):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.life = life
        self.max_life = life
        self.size = size


class ParticleField:
    """Partikel energi yang menyembur dari sudut portal.

    Murni dekorasi — tidak pernah menyentuh logika kotak atau mask.
    """

    def __init__(self, max_particles=config.MAX_PARTICLES):
        self.max_particles = max_particles
        self.particles = []

    def spawn(self, quad, count):
        if len(self.particles) >= self.max_particles:
            return
        for _ in range(count):
            c = quad[random.randrange(4)]
            ang = random.uniform(0, 2 * math.pi)
            spd = random.uniform(16, 70)
            self.particles.append(Particle(
                float(c[0]), float(c[1]),
                math.cos(ang) * spd, math.sin(ang) * spd,
                random.uniform(0.45, 1.0), random.uniform(0.25, 0.5)))

    def update(self, dt):
        drag = math.exp(-1.8 * dt)
        alive = []
        for p in self.particles:
            p.life -= dt
            if p.life <= 0:
                continue
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.vx *= drag
            p.vy *= drag
            alive.append(p)
        self.particles = alive

    def draw(self, frame, color):
        for p in self.particles:
            a = max(0.0, p.life / p.max_life)
            r = max(1, int(round(p.size * (2.2 + 3.0 * a))))
            cv2.circle(frame, (int(p.x), int(p.y)), r,
                       dim_color(color, 0.65 * a), -1, cv2.LINE_AA)
