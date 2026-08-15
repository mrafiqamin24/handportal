"""HUD: panel-panel kecil bertipografi Pillow di atas frame OpenCV.

Pillow dipakai karena `cv2.putText` tidak bisa membuat sudut membulat, latar
semi-transparan, maupun huruf setebal ini. Kalau Pillow tidak terpasang, HUD
turun ke versi OpenCV sederhana — aplikasi tetap jalan.

Diadaptasi dari milan-kb/fancy-fingers (MIT).
"""

import os

import cv2
import numpy as np

from app import config

try:
    from PIL import Image, ImageDraw, ImageFont

    PIL_AVAILABLE = True
except Exception:  # noqa: BLE001
    PIL_AVAILABLE = False

_FONT_CANDIDATES = [
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
]


def _find_font_path():
    for p in _FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


FONT_PATH = _find_font_path()
_FONT_CACHE = {}

CONTROL_Y = 16
CONTROL_H = 34
MODE_SWITCH_W = 238
CAMERA_BUTTON_W = 144


def _font(size, bold=False):
    """Font ukuran `size`. Utamakan TTF asli; kalau tidak ada, pakai font
    bawaan Pillow yang bisa diskalakan (Pillow >= 10.1), supaya HUD tetap
    terbaca di sistem tanpa file font sama sekali."""
    if not PIL_AVAILABLE:
        return None
    key = (size, bold)
    f = _FONT_CACHE.get(key)
    if f is None:
        try:
            f = ImageFont.truetype(FONT_PATH, size)
        except Exception:  # noqa: BLE001
            try:
                f = ImageFont.load_default(size)
            except TypeError:  # Pillow lama tanpa dukungan ukuran
                f = ImageFont.load_default()
        _FONT_CACHE[key] = f
    return f


def _text_width(text, font):
    if not PIL_AVAILABLE or font is None:
        return int(8 * len(text))
    bbox = font.getbbox(text)
    return bbox[2] - bbox[0]


def _pil_rgb(color, a=255):
    """BGR -> tuple RGBA untuk PIL."""
    return (int(color[2]), int(color[1]), int(color[0]), int(a))


def blend_panel(frame, panel, x, y, alpha=1.0):
    """Tempel panel RGBA kecil ke frame BGR dengan alpha-compositing.

    `alpha` (0..1) menskalakan opasitas panel — dipakai untuk memudarkan hint.
    """
    if panel is None or alpha <= 0.0:
        return
    ph, pw = panel.shape[:2]
    x0 = max(int(x), 0)
    y0 = max(int(y), 0)
    x1 = min(int(x) + pw, frame.shape[1])
    y1 = min(int(y) + ph, frame.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    sx, sy = x0 - int(x), y0 - int(y)
    a = panel[sy:y1 - int(y), sx:x1 - int(x), 3:4].astype(np.float32) / 255.0
    a *= float(alpha)
    rgb = panel[sy:y1 - int(y), sx:x1 - int(x), :3].astype(np.float32)
    region = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (rgb * a + region * (1.0 - a)).astype(np.uint8)


def build_mode_switch(mode, accent, hovered=None):
    """Kontrol segmented untuk memilih Gestur atau Portal dengan mouse."""
    if not PIL_AVAILABLE:
        return None
    font = _font(12, bold=True)
    W, H = MODE_SWITCH_W, CONTROL_H
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((1, 1, W - 2, H - 2), radius=12,
                        fill=_pil_rgb(config.PANEL_FILL, 188),
                        outline=_pil_rgb(config.PANEL_EDGE, 130), width=1)
    labels = (("GESTUR", 0), ("PORTAL", 1))
    half = W // 2
    for label, idx in labels:
        selected = mode == label
        x0, x1 = idx * half + 3, (idx + 1) * half - 3
        if selected:
            d.rounded_rectangle((x0, 4, x1, H - 4), radius=9,
                                fill=_pil_rgb(accent, 225))
        elif hovered == f"mode:{label}":
            d.rounded_rectangle((x0, 4, x1, H - 4), radius=9,
                                fill=_pil_rgb(config.PANEL_EDGE, 150))
        color = config.PANEL_FILL if selected else config.TEXT_HI
        d.text(((x0 + x1) // 2, H // 2), label, font=font,
               fill=_pil_rgb(color, 255 if selected else 220), anchor="mm")
    return np.array(img)


def build_camera_button(camera_index, accent, hovered=False):
    """Tombol kamera runtime; label satu-based agar ramah pengguna."""
    if not PIL_AVAILABLE:
        return None
    W, H = CAMERA_BUTTON_W, CONTROL_H
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    fill = config.PANEL_EDGE if hovered else config.PANEL_FILL
    d.rounded_rectangle((1, 1, W - 2, H - 2), radius=12,
                        fill=_pil_rgb(fill, 188 if hovered else 168),
                        outline=_pil_rgb(accent, 155 if hovered else 95), width=1)
    font = _font(12, bold=True)
    d.ellipse((13, H // 2 - 4, 21, H // 2 + 4), fill=_pil_rgb(accent, 235))
    d.text((29, H // 2), f"KAMERA {camera_index + 1}", font=font,
           fill=_pil_rgb(config.TEXT_HI, 245), anchor="lm")
    d.text((W - 14, H // 2), "C", font=_font(10, bold=True),
           fill=_pil_rgb(config.TEXT_LO, 210), anchor="rm")
    return np.array(img)


def build_notice_panel(text, error=False):
    if not PIL_AVAILABLE or not text:
        return None
    font = _font(13, bold=True)
    W = max(220, _text_width(text, font) + 34)
    H = 34
    color = (80, 92, 255) if error else config.ACCENT
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((1, 1, W - 2, H - 2), radius=12,
                        fill=_pil_rgb(config.PANEL_FILL, 220),
                        outline=_pil_rgb(color, 170), width=1)
    d.text((W // 2, H // 2), text, font=font,
           fill=_pil_rgb(config.TEXT_HI, 255), anchor="mm")
    return np.array(img)


def build_effect_panel(effect_idx, name, from_name, anim_p, accent):
    """Nomor efek beraksen + nama efek, lengkap dengan animasi slide/fade
    saat berganti. Return array RGBA, atau None kalau Pillow tak ada."""
    if not PIL_AVAILABLE:
        return None
    font_num = _font(13, bold=True)
    font_name = _font(28, bold=True)
    name_w = _text_width(name, font_name)
    from_w = _text_width(from_name, font_name) if from_name else 0
    W = max(320, 70 + max(name_w, from_w))
    H = 62
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((1, 1, W - 2, H - 2), radius=13,
                        fill=_pil_rgb(config.PANEL_FILL, 178),
                        outline=_pil_rgb(config.PANEL_EDGE, 110), width=1)
    d.text((18, H // 2), f"{effect_idx + 1:02d}", font=font_num,
           fill=_pil_rgb(accent, 240), anchor="lm")
    y_name = H // 2
    if from_name is not None:
        fade = int(210 * (1.0 - anim_p))
        if fade > 2:
            d.text((52, y_name - int(34 * anim_p)), from_name,
                   font=font_name, fill=_pil_rgb(config.TEXT_HI, fade),
                   anchor="lm")
        d.text((52, y_name + int(34 * (1.0 - anim_p))), name,
               font=font_name,
               fill=_pil_rgb(config.TEXT_HI, int(255 * anim_p)), anchor="lm")
    else:
        d.text((52, y_name), name, font=font_name,
               fill=_pil_rgb(config.TEXT_HI, 255), anchor="lm")
    return np.array(img)


def build_fps_pill(fps):
    """Angka FPS kecil di pojok kanan atas."""
    if not PIL_AVAILABLE:
        return None
    txt = f"{fps:3.0f} FPS"
    font = _font(12, bold=True)
    W = _text_width(txt, font) + 28
    H = 24
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((1, 1, W - 2, H - 2), radius=12,
                        fill=_pil_rgb(config.PANEL_FILL, 135),
                        outline=_pil_rgb(config.PANEL_EDGE, 75), width=1)
    d.text((14, H // 2), txt, font=font,
           fill=_pil_rgb(config.TEXT_LO, 235), anchor="lm")
    return np.array(img)


def build_hint_panel():
    """Petunjuk tombol singkat; memudar beberapa detik setelah aplikasi mulai."""
    if not PIL_AVAILABLE:
        return None
    txt = ("PORTAL: bentuk L dengan 2 tangan, double-pinch: efek  |  "
           "1-9 / [ ]: filter  |  klik mode / TAB  |  kamera: klik / C  |  "
           "q: keluar")
    font = _font(12)
    W = _text_width(txt, font) + 36
    H = 28
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((1, 1, W - 2, H - 2), radius=14,
                        fill=_pil_rgb(config.PANEL_FILL, 150),
                        outline=_pil_rgb(config.PANEL_EDGE, 90), width=1)
    d.text((18, H // 2), txt, font=font,
           fill=_pil_rgb(config.TEXT_LO, 240), anchor="lm")
    return np.array(img)


def _hud_fallback(frame, mode, effect_name, fps, camera_index=0):
    """HUD minimal OpenCV, hanya dipakai kalau Pillow tidak terpasang."""
    w = frame.shape[1]
    status = effect_name if mode == "PORTAL" else "DETEKSI GESTUR"
    cv2.putText(frame, f"[{mode}]  {status}", (16, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, config.ACCENT, 2, cv2.LINE_AA)
    cv2.putText(frame, f"CAM {camera_index + 1} [C]", (w - 145, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, config.TEXT_HI, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{fps:.0f} FPS", (w - 118, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, config.TEXT_LO, 1, cv2.LINE_AA)


class Hud:
    """Semua elemen HUD dalam satu tempat.

    Panel hint dan kontrol mode/kamera di-cache karena isinya jarang berubah;
    panel efek dibangun tiap frame karena membawa animasi slide.
    """

    def __init__(self):
        self._hint = build_hint_panel()
        self._fps_pill = None
        self._fps_shown = None
        self._mode_control = None
        self._mode_key = None
        self._camera_control = None
        self._camera_key = None
        self._hovered = None
        self._notice = None
        self._notice_start = 0.0
        self._notice_error = False

    def hit_test(self, x, y, frame_width):
        """Return aksi kontrol HUD pada koordinat frame, atau None."""
        if CONTROL_Y <= y < CONTROL_Y + CONTROL_H:
            if 16 <= x < 16 + MODE_SWITCH_W:
                label = "GESTUR" if x < 16 + MODE_SWITCH_W // 2 else "PORTAL"
                return f"mode:{label}"
            cam_x = frame_width - CAMERA_BUTTON_W - 16
            if cam_x <= x < cam_x + CAMERA_BUTTON_W:
                return "camera:next"
        return None

    def pointer(self, x, y, frame_width):
        hovered = self.hit_test(x, y, frame_width)
        if hovered != self._hovered:
            self._hovered = hovered
            self._mode_key = None
            self._camera_key = None
        return hovered

    def notify(self, text, now, error=False):
        self._notice = build_notice_panel(text, error)
        self._notice_start = now
        self._notice_error = error

    def _mode_switch(self, mode, accent):
        key = (mode, accent, self._hovered)
        if key != self._mode_key:
            self._mode_control = build_mode_switch(mode, accent, self._hovered)
            self._mode_key = key
        return self._mode_control

    def _camera_button(self, camera_index, accent):
        key = (camera_index, accent, self._hovered == "camera:next")
        if key != self._camera_key:
            self._camera_control = build_camera_button(
                camera_index, accent, self._hovered == "camera:next")
            self._camera_key = key
        return self._camera_control

    def draw(self, frame, mode, effect_idx, effect_name, label_from, label_p,
             fps, accent, elapsed, camera_index=0, now=0.0):
        dw, dh = frame.shape[1], frame.shape[0]
        if not PIL_AVAILABLE:
            _hud_fallback(frame, mode, effect_name, fps, camera_index)
            return

        mode_control = self._mode_switch(mode, accent)
        blend_panel(frame, mode_control, 16, CONTROL_Y)

        camera = self._camera_button(camera_index, accent)
        blend_panel(frame, camera, dw - CAMERA_BUTTON_W - 16, CONTROL_Y)

        if mode == "PORTAL":
            panel = build_effect_panel(effect_idx, effect_name, label_from,
                                       label_p, accent)
            blend_panel(frame, panel, 16, 58)

        # FPS dibulatkan dulu: pill hanya digambar ulang saat angkanya berubah
        shown = int(round(fps))
        if shown != self._fps_shown:
            self._fps_shown = shown
            self._fps_pill = build_fps_pill(shown)
        if self._fps_pill is not None:
            blend_panel(frame, self._fps_pill,
                        dw - self._fps_pill.shape[1] - 16, 58)

        if self._notice is not None:
            age = max(0.0, now - self._notice_start)
            notice_alpha = max(0.0, min(1.0, 1.0 - max(0.0, age - 1.6) / 0.35))
            if notice_alpha > 0.01:
                blend_panel(frame, self._notice,
                            (dw - self._notice.shape[1]) // 2, 16,
                            alpha=notice_alpha)

        hint_alpha = 1.0 - max(0.0, (elapsed - config.HINT_SHOW_S)
                               / config.HINT_FADE_S)
        if self._hint is not None and hint_alpha > 0.02:
            blend_panel(frame, self._hint, (dw - self._hint.shape[1]) // 2,
                        dh - self._hint.shape[0] - 14, alpha=hint_alpha)
