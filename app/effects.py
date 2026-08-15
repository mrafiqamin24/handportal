"""Filter gambar. Setiap fungsi: ndarray BGR uint8 masuk, ndarray baru keluar.

Aturan yang berlaku untuk semua isi modul ini:

* input **tidak pernah** dimutasi — pemanggil boleh memakai ulang framenya;
* shape dan dtype keluaran sama persis dengan masukan;
* aman untuk potongan sekecil 1x1 pixel, karena filter yang sama dipakai untuk
  crop kecil (mis. isi portal) maupun layar penuh.

Delapan filter diadaptasi dari milan-kb/fancy-fingers (MIT); `fx_blur` baru,
mempertahankan tampilan ✌️ Peace yang lama.
"""

import cv2
import numpy as np

from app import config


class FilterTransition:
    """State machine crossfade yang aman saat pengguna mengganti efek cepat.

    Retarget di tengah transisi dimulai dari efek yang saat itu paling dominan,
    lalu memakai smoothstep agar kecepatan visual tidak patah di awal/akhir.
    """

    def __init__(self, current_idx=0, duration=config.CROSSFADE_S):
        self.current_idx = int(current_idx)
        self.previous_idx = None
        self.duration = max(float(duration), 1e-6)
        self.started_at = 0.0

    @staticmethod
    def _ease(p):
        p = max(0.0, min(1.0, float(p)))
        return p * p * (3.0 - 2.0 * p)

    def state(self, now):
        """Return (previous_idx, eased_blend) dan selesaikan state bila penuh."""
        if self.previous_idx is None:
            return None, 1.0
        linear = max(0.0, min(1.0, (float(now) - self.started_at)
                                  / self.duration))
        blend = self._ease(linear)
        previous = self.previous_idx
        if linear >= 1.0:
            self.previous_idx = None
            return None, 1.0
        return previous, blend

    def switch(self, new_idx, now):
        """Pilih target baru. Return indeks sumber untuk animasi label/efek."""
        new_idx = int(new_idx)
        if new_idx == self.current_idx:
            return None

        previous, blend = self.state(now)
        if previous is None:
            source = self.current_idx
        else:
            source = self.current_idx if blend >= 0.5 else previous

        self.current_idx = new_idx
        self.started_at = float(now)
        self.previous_idx = None if source == new_idx else source
        return source


def _odd_kernel(img, want):
    """Ukuran kernel ganjil yang tidak pernah melebihi gambar.

    Kernel yang lebih besar dari gambar membuat OpenCV melempar, dan itu
    persis yang terjadi kalau filter dipakai pada crop kecil.
    """
    limit = max(3, (min(img.shape[:2]) // 2) * 2 + 1)
    return min(want, limit)


def fx_blur(img):
    """Blur bawaan ✌️ Peace. Kernel menyesuaikan ukuran gambar supaya crop
    sekecil 1x1 pun tidak membuat OpenCV melempar."""
    h, w = img.shape[:2]
    k = min(45, max(3, (min(h, w) // 8) * 2 + 1))
    return cv2.GaussianBlur(img, (k, k), 0)


def fx_thermal(img):
    """Peta panas: luminance dipetakan ke colormap JET."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)


def fx_posterize_neon(img):
    """Warna dikuantisasi lalu saturasinya didorong -> kesan poster neon."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.8, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * 1.1, 0, 255)
    boosted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    div = 64
    quant = (boosted // div) * div + div // 2
    return quant.astype(np.uint8)


_K2 = np.ones((2, 2), np.uint8)  # kernel dilate, dibuat sekali saja


def fx_edge_mesh(img):
    """Hanya garis tepi putih di atas dasar gelap — kesan jaring kawat."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)
    edges = cv2.dilate(edges, _K2, iterations=1)
    base = np.full_like(img, (20, 20, 20))
    base[edges > 0] = (255, 255, 255)
    return base


def fx_invert_glitch(img):
    """Warna dibalik lalu kanal merah/biru digeser -> kesan glitch."""
    inv = 255 - img
    b, g, r = cv2.split(inv)
    shift = 6
    r = np.roll(r, shift, axis=1)
    b = np.roll(b, -shift, axis=1)
    return cv2.merge([b, g, r])


def fx_sketch(img):
    """Sketsa pensil: abu-abu dibagi versi negatif-blur dari dirinya."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    k = _odd_kernel(img, 21)
    blur = cv2.GaussianBlur(inv, (k, k), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)


def fx_chromatic(img):
    """Aberasi kromatik: kanal warna terpisah, saturasi sedikit dinaikkan."""
    b, g, r = cv2.split(img)
    shift = 8
    r = np.roll(r, shift, axis=1)
    b = np.roll(b, -shift, axis=1)
    out = cv2.merge([b, g, r])
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.25, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def fx_pixel_mosaic(img):
    """Mosaik kotak: diperkecil keras, dibesarkan nearest, plus garis nat."""
    h, w = img.shape[:2]
    block = 14
    small = cv2.resize(img, (max(1, w // block), max(1, h // block)),
                       interpolation=cv2.INTER_AREA)
    out = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    seam = np.array([184], np.uint8)  # 184/255 = 0,72
    out[::block, :] = cv2.multiply(out[::block, :], seam, scale=1.0 / 255.0)
    out[:, ::block] = cv2.multiply(out[:, ::block], seam, scale=1.0 / 255.0)
    return out


def _build_duotone_lut():
    """LUT BGR 256 entri untuk gradasi duotone (dibuat sekali saat impor)."""
    c0 = np.array([32, 24, 90], np.float32)     # BGR indigo pekat
    c1 = np.array([255, 196, 64], np.float32)   # BGR cyan listrik
    t = np.linspace(0.0, 1.0, 256, dtype=np.float32)[:, None]
    return (t * c1 + (1.0 - t) * c0).astype(np.uint8)


_DUOTONE_LUT = _build_duotone_lut()


def fx_duotone(img):
    """Duotone: luminance dipetakan ke gradasi indigo -> cyan."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return _DUOTONE_LUT[gray]


def fx_night_vision(img):
    """Night vision hijau dengan kontras luminance dan scanline lembut."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    out = np.zeros_like(img)
    out[..., 0] = gray // 8
    out[..., 1] = gray
    out[..., 2] = gray // 5
    if out.shape[0] > 1:
        out[1::4] = cv2.multiply(
            out[1::4], np.array([205], np.uint8), scale=1.0 / 255.0)
    return out


def fx_vhs(img):
    """Pemisahan kanal, tint analog, dan scanline ala kaset VHS."""
    b, g, r = cv2.split(img)
    shift = max(1, min(6, img.shape[1] // 40))
    out = cv2.merge([np.roll(b, -shift, axis=1), g,
                     np.roll(r, shift, axis=1)])
    tint = np.full_like(out, (12, 2, 18))
    out = cv2.add(out, tint)
    out[::3] = cv2.multiply(
        out[::3], np.array([190], np.uint8), scale=1.0 / 255.0)
    return out


def fx_comic_ink(img):
    """Warna komik terkuantisasi dengan kontur tinta hitam."""
    quantized = ((img // 48) * 48 + 24).astype(np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 70, 150)
    out = quantized.copy()
    out[edges > 0] = (8, 8, 8)
    return out


def fx_emboss_chrome(img):
    """Relief metalik dari gradien luminance, diberi colormap bone."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], np.float32)
    relief = cv2.filter2D(gray, cv2.CV_16S, kernel)
    relief = np.clip(relief + 128, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(relief, cv2.COLORMAP_BONE)


EFFECTS = [
    fx_blur, fx_thermal, fx_edge_mesh, fx_posterize_neon, fx_invert_glitch,
    fx_sketch, fx_chromatic, fx_pixel_mosaic, fx_duotone, fx_night_vision,
    fx_vhs, fx_comic_ink, fx_emboss_chrome,
]
EFFECT_NAMES = [
    "Blur", "Thermal", "Edge Mesh", "Posterize Neon", "Invert Glitch",
    "Sketch", "Chromatic", "Pixel Mosaic", "Duotone", "Night Vision",
    "VHS", "Comic Ink", "Emboss Chrome",
]


def apply_full_frame(frame, fn, prev_fn=None, blend=1.0,
                     scale=config.EFFECT_SCALE):
    """Terapkan filter ke seluruh frame, dihitung di resolusi lebih rendah.

    Filter layar-penuh di 1280x720 setiap frame mahal — terutama Posterize Neon
    yang bolak-balik konversi HSV. Menghitungnya di separuh resolusi lalu
    di-upscale nyaris tak terlihat bedanya untuk efek stilisasi, tapi jauh
    lebih murah. `prev_fn` + `blend` melakukan crossfade saat berganti filter.
    """
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))),
                       interpolation=cv2.INTER_AREA) if scale < 1.0 else frame
    out = fn(small)
    if prev_fn is not None and blend < 1.0:
        out = cv2.addWeighted(out, float(blend), prev_fn(small),
                              1.0 - float(blend), 0)
    if out.shape[:2] != (h, w):
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
    return out
