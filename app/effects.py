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


EFFECTS = [
    fx_blur, fx_thermal, fx_edge_mesh, fx_posterize_neon, fx_invert_glitch,
    fx_sketch, fx_chromatic, fx_pixel_mosaic, fx_duotone,
]
EFFECT_NAMES = [
    "Blur", "Thermal", "Edge Mesh", "Posterize Neon", "Invert Glitch",
    "Sketch", "Chromatic", "Pixel Mosaic", "Duotone",
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
