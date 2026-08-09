# Foto-Kita-Blurrr × HandPortal — Rencana Implementasi

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Memecah `foto_kita_blurrr.py` menjadi paket `app/` bermodul, lalu menambahkan sembilan filter gambar, Mode Portal, HUD, dan pinch-tap sebagai kontrol ganti efek — sesuai spec `docs/superpowers/specs/2026-08-09-foto-kita-blurrr-handportal-merge-design.md`.

**Architecture:** *Extract-then-extend*. Task 1–5 memindahkan kode yang sudah ada ke paket bermodul tanpa mengubah satu pun perilaku, dengan test sebagai jaring pengaman. Task 6–9 menambah fitur baru di atas fondasi itu. Setiap task berakhir dengan aplikasi yang masih bisa dijalankan.

**Tech Stack:** Python 3.12, OpenCV, MediaPipe Tasks API, NumPy, pygame (audio), Pillow (HUD), pytest.

## Global Constraints

- Semua identifier dalam bahasa Inggris; semua komentar dan docstring dalam bahasa Indonesia — mengikuti gaya `foto_kita_blurrr.py` yang sudah ada.
- `app/effects.py` murni `ndarray → ndarray`: tidak mengimpor `app.gestures`, `app.portal`, atau `app.main`. Setiap `fx_*` mengembalikan array **baru** dan tidak pernah memutasi argumennya.
- `app/gestures.py` tidak mengimpor apa pun untuk menggambar dan **tidak pernah memanggil `time.time()` / `time.perf_counter()` sendiri** — setiap fungsi yang bergantung waktu menerima `now: float` sebagai argumen. Ini yang membuatnya bisa di-test tanpa `sleep`.
- Isi portal **tidak pernah** di-`cv2.warpPerspective`. Crop → filter lurus → tempel balik → mask. Aturan warisan dari HandPortal, permanen.
- Semua nilai dtype gambar adalah `uint8`.
- Referensi kode HandPortal ada di `.reference/portal_app.py` (gitignored). Rujuk baris persisnya saat mengadaptasi.
- Sumber kode lama yang diekstrak: `foto_kita_blurrr.py` pada commit `ee8ce71`. Kalau file itu sudah berubah, ambil dari `git show ee8ce71:foto_kita_blurrr.py`.
- Jalankan test dengan `python -m pytest -q` dari root project. Jumlah test yang
  ditulis di tiap langkah adalah hitungan yang diharapkan dari rencana ini —
  kalau kamu menambah test sendiri angkanya akan bergeser, dan itu tidak apa-apa.
  Yang tidak boleh bergeser: **nol kegagalan**.
- Setiap task diakhiri commit. Pesan commit bahasa Indonesia, imperatif, diakhiri baris `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.

---

## Peta File

| File | Tanggung jawab | Dibuat di task |
|------|----------------|----------------|
| `app/config.py` | Semua konstanta tunable + path aset. Tidak ada logika. | 1 |
| `app/gestures.py` | Landmark → label. Helper geometri, classifier, smoother, debouncer, pinch. | 1, 2, 6 |
| `app/draw.py` | Primitif menggambar OpenCV: teks beroutline, hati, not balok, skeleton, vignette. | 3 |
| `app/scenes.py` | Efek visual per-gestur + partikel hati/konfeti. | 3 |
| `app/audio.py` | `AudioPlayer` — pygame, degrade dengan anggun. | 4 |
| `app/models.py` | Auto-download model + factory HandLandmarker/FaceDetector + titik mulut. | 4 |
| `app/main.py` | Loop kamera, mesin mode, keyboard. Satu-satunya yang menyentuh kamera & jam. | 5 |
| `foto_kita_blurrr.py` | Entry point tipis → `app.main:main`. | 5 |
| `app/effects.py` | Sembilan filter gambar + penerap layar-penuh. | 7 |
| `app/hud.py` | Panel PIL + fallback OpenCV. | 8 |
| `app/portal.py` | Kotak, komposit feather, rim glow, corner bracket, partikel energi. | 9 |
| `tests/conftest.py` | Pembuat landmark sintetis + jam yang di-inject. | 1 |

---

### Task 1: Paket `app/`, konfigurasi, dan helper landmark

**Files:**
- Create: `app/__init__.py`, `app/config.py`, `app/gestures.py`
- Create: `tests/__init__.py`, `tests/conftest.py`, `tests/test_gestures.py`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: —
- Produces:
  - `app.config` — konstanta modul-level (daftar lengkap di Step 2)
  - `app.gestures.dist(a, b) -> float`
  - `app.gestures.hand_points(landmark_list, w: int, h: int) -> list[tuple[int, int]]`
  - `app.gestures.palm_scale(pts) -> float`
  - `app.gestures.fingers_extended(pts) -> list[bool]` (5 elemen: thumb, index, middle, ring, pinky)
  - Konstanta indeks: `WRIST, THUMB_TIP, THUMB_IP, THUMB_MCP, INDEX_TIP, INDEX_PIP, INDEX_MCP, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP, RING_TIP, RING_PIP, PINKY_TIP, PINKY_PIP, PINKY_MCP, TIP_IDS, HAND_CONNECTIONS`
  - `tests.conftest` fixture `make_hand` dan `FakeClock`

- [ ] **Step 1: Tambah pytest ke requirements**

Ganti isi `requirements.txt` menjadi:

```
# Aplikasi memakai MediaPipe Tasks API (HandLandmarker). Model diunduh
# otomatis saat pertama dijalankan (lihat README).
opencv-python>=4.8.0
mediapipe>=0.10.11
numpy>=1.26.0
pygame>=2.5.0
Pillow>=10.1.0
pytest>=8.0.0
```

- [ ] **Step 2: Tulis `app/config.py`**

```python
"""Semua konstanta yang bisa disetel + lokasi aset. Tidak ada logika di sini."""

import os

# ---------------------------------------------------------------- lokasi
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GESTUR_DIR = os.path.join(BASE_DIR, "assets", "gestur")
SOUND_DIR = os.path.join(BASE_DIR, "assets", "Sond")
MODELS_DIR = os.path.join(BASE_DIR, "models")
SHOTS_DIR = os.path.join(BASE_DIR, "shots")

SOUND_BLUR = os.path.join(SOUND_DIR, "foto kita blur.wav")
SOUND_KICAW = os.path.join(SOUND_DIR, "Kicaw Mania.wav")
IMG_BLUR = os.path.join(GESTUR_DIR, "Fotokitablurr.jpg")

MODEL_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
FACE_MODEL_PATH = os.path.join(MODELS_DIR, "blaze_face_short_range.tflite")
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)

# ---------------------------------------------------------------- kamera
CAM_INDEX = 0
FRAME_W, FRAME_H = 1280, 720
DETECT_WIDTH = 640        # deteksi tangan di frame diperkecil -> lebih responsif
EFFECT_SCALE = 0.5        # filter layar-penuh dihitung di separuh resolusi

# ---------------------------------------------------------------- gestur
HOLD_FRAMES = 6           # frame stabil sebelum sebuah gestur dianggap aktif
SMOOTH_ALPHA = 0.6        # besar = responsif, kecil = halus
SMOOTH_MATCH_DIST = 140.0  # jarak maksimum pencocokan tangan antar-frame

# ---------------------------------------------------------------- pinch
# Jarak ternormalisasi ujung jempol <-> ujung telunjuk. Dua ambang berbeda
# (histeresis): masuk lebih ketat daripada keluar, supaya jitter landmark di
# sekitar ambang tidak memicu tap beruntun.
PINCH_ENTER = 0.055
PINCH_EXIT = 0.075
PINCH_HOLD_S = 0.35       # <0,35 dtk = TAP (ganti efek); >=0,35 dtk = HOLD (gestur OK)
PINCH_STEP = 0.005        # langkah penyetelan sensitivitas lewat tombol +/-
PINCH_MIN, PINCH_MAX = 0.015, 0.20

# ---------------------------------------------------------------- transisi
CROSSFADE_S = 0.22        # lama crossfade antar-filter
LABEL_ANIM_S = 0.30       # lama animasi slide label nama efek
PORTAL_FADE_S = 0.30      # tetapan waktu fade in/out portal

# ---------------------------------------------------------------- HUD
HINT_SHOW_S = 5.0         # hint kontrol tampil selama ini...
HINT_FADE_S = 1.5         # ...lalu memudar selama ini

# ---------------------------------------------------------------- portal
FEATHER_PX = 8            # lebar pita blending di tepi portal
FEATHER_SIGMA = 2.0
GLOW_PAD = 26             # seberapa jauh glow melebar dari tepi kotak
GLOW_SIGMA = 9.0
MAX_PARTICLES = 150

# ---------------------------------------------------------------- warna (BGR)
ACCENT = (255, 170, 64)         # cyan listrik
ACCENT_VIOLET = (230, 70, 190)  # violet, untuk denyut warna lambat
TEXT_HI = (240, 244, 248)
TEXT_LO = (150, 160, 175)
PANEL_FILL = (24, 26, 38)
PANEL_EDGE = (58, 78, 100)

# ---------------------------------------------------------------- default runtime
FULLSCREEN = False
DETECT_DOWNSCALE = False
VIGNETTE = True
```

- [ ] **Step 3: Tulis test yang gagal untuk helper landmark**

Buat `app/__init__.py` dan `tests/__init__.py` sebagai file kosong.

`tests/conftest.py`:

```python
"""Perkakas test: landmark tangan sintetis + jam yang di-inject.

Tidak ada test di repo ini yang menyentuh webcam atau `time.sleep`.
"""

import pytest

# Indeks landmark MediaPipe, diurutkan per jari (4 sendi tiap jari).
# 0 = pergelangan; 1..4 jempol; 5..8 telunjuk; 9..12 tengah;
# 13..16 manis; 17..20 kelingking.
FINGER_SLOTS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}

# Arah tiap jari saat terbuka penuh, relatif terhadap pergelangan di (0, 0)
# dengan sumbu y menghadap ke bawah (konvensi pixel OpenCV): jari mengarah
# ke ATAS, jadi y negatif. Jempol melebar ke samping kiri.
FINGER_DIRS = {
    "thumb": (-0.85, -0.45),
    "index": (-0.22, -1.0),
    "middle": (0.0, -1.05),
    "ring": (0.22, -1.0),
    "pinky": (0.42, -0.85),
}


def make_hand(extended=("index", "middle"), origin=(320, 400), scale=100.0,
              thumb_index_gap=None):
    """Bangun 21 titik pixel untuk satu tangan.

    `extended` — nama jari yang lurus; sisanya digambar menekuk (ujung jari
    ditarik balik mendekat ke pergelangan sehingga `fingers_extended` melihat
    tip lebih dekat ke wrist daripada pip).

    `thumb_index_gap` — kalau diisi, ujung jempol dipindah supaya jaraknya ke
    ujung telunjuk persis sekian pixel. Dipakai untuk menguji pinch/OK.
    """
    ox, oy = origin
    pts = [(0, 0)] * 21
    pts[0] = (int(ox), int(oy))
    for name, slots in FINGER_SLOTS.items():
        dx, dy = FINGER_DIRS[name]
        straight = name in extended
        for i, idx in enumerate(slots, start=1):
            # sendi ke-i: 0,25 / 0,5 / 0,75 / 1,0 dari panjang jari
            reach = i / 4.0
            if not straight and i >= 3:
                # dua sendi terakhir melipat balik ke telapak
                reach = 0.5 - (i - 2) * 0.18
            pts[idx] = (int(ox + dx * scale * reach),
                        int(oy + dy * scale * reach))
    if thumb_index_gap is not None:
        ix, iy = pts[8]
        pts[4] = (int(ix + thumb_index_gap), int(iy))
    return pts


class FakeClock:
    """Jam yang dikendalikan test. Tidak ada `sleep` di mana pun."""

    def __init__(self, start=0.0):
        self.now = float(start)

    def advance(self, seconds):
        self.now += float(seconds)
        return self.now


@pytest.fixture
def clock():
    return FakeClock()
```

`tests/test_gestures.py`:

```python
"""Test helper landmark murni: tanpa kamera, tanpa MediaPipe."""

from app import gestures as g
from tests.conftest import make_hand


def test_dist_is_euclidean():
    assert g.dist((0, 0), (3, 4)) == 5.0


def test_hand_points_scales_normalized_landmarks_to_pixels():
    class LM:
        def __init__(self, x, y):
            self.x, self.y = x, y

    pts = g.hand_points([LM(0.5, 0.25), LM(1.0, 1.0)], 640, 480)
    assert pts == [(320, 120), (640, 480)]


def test_palm_scale_is_wrist_to_middle_mcp_distance():
    pts = make_hand(origin=(0, 0), scale=100.0)
    assert g.palm_scale(pts) == g.dist(pts[g.WRIST], pts[g.MIDDLE_MCP])


def test_palm_scale_never_returns_zero():
    # semua landmark menumpuk di satu titik -> pembagi tidak boleh nol
    assert g.palm_scale([(5, 5)] * 21) > 0.0


def test_fingers_extended_reports_open_fingers():
    pts = make_hand(extended=("index", "middle"))
    thumb, index, middle, ring, pinky = g.fingers_extended(pts)
    assert index and middle
    assert not ring and not pinky


def test_fingers_extended_on_open_palm():
    pts = make_hand(extended=("thumb", "index", "middle", "ring", "pinky"))
    assert g.fingers_extended(pts) == [True] * 5


def test_fingers_extended_on_fist():
    pts = make_hand(extended=())
    assert g.fingers_extended(pts)[1:] == [False] * 4
```

- [ ] **Step 4: Jalankan test, pastikan gagal**

Run: `python -m pytest tests/test_gestures.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.gestures'`

- [ ] **Step 5: Tulis `app/gestures.py` (bagian helper saja)**

Salin **apa adanya** dari `git show ee8ce71:foto_kita_blurrr.py`:
- konstanta indeks landmark, baris 168–175
- `HAND_CONNECTIONS`, baris 178–185
- `_dist`, baris 188–189 → **ganti nama jadi `dist`** (jadi API publik; ini satu-satunya perubahan)
- `hand_points`, baris 192–194
- `palm_scale`, baris 197–199
- `fingers_extended`, baris 202–223

Awali file dengan:

```python
"""Landmark tangan -> label gestur.

Modul ini sengaja murni: tidak mengimpor apa pun untuk menggambar dan tidak
pernah membaca jam sendiri. Semua fungsi yang bergantung waktu menerima `now`
sebagai argumen, sehingga seluruh isinya bisa di-test tanpa kamera.
"""

import math
```

- [ ] **Step 6: Jalankan test, pastikan lulus**

Run: `python -m pytest tests/test_gestures.py -q`
Expected: PASS, 7 test

- [ ] **Step 7: Commit**

```bash
git add app tests requirements.txt
git commit -m "Ekstrak helper landmark ke paket app/ + fondasi test

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Classifier gestur, smoother, dan debouncer

**Files:**
- Modify: `app/gestures.py`
- Modify: `tests/test_gestures.py`

**Interfaces:**
- Consumes: `app.gestures.{dist, palm_scale, fingers_extended, WRIST, ...}` dari Task 1
- Produces:
  - `classify_hand(pts) -> str | None` — mengembalikan `"OK"`, `"ILY"`, `"PEACE"`, atau `None`
  - `detect_two_hand_heart(hands_pts, w: int) -> bool`
  - `detect_kicaw(hands_pts, w: int, h: int, mouth=None) -> bool` — `mouth` adalah `(mx, my, r)` atau `None`
  - `HandSmoother(alpha=SMOOTH_ALPHA, match_dist=SMOOTH_MATCH_DIST).update(hands) -> list[list[tuple[int, int]]]`
  - `GestureDebouncer(hold_frames=HOLD_FRAMES).update(current: str | None) -> str | None`

> Catatan: `"OK"` masih dikenali dari bentuk jari di task ini. Task 6 memindahkannya ke pinch-ditahan. Ini disengaja — Task 1–5 tidak mengubah perilaku sama sekali.

- [ ] **Step 1: Tulis test yang gagal**

Tambahkan ke `tests/test_gestures.py`:

```python
def test_classify_peace():
    assert g.classify_hand(make_hand(extended=("index", "middle"))) == "PEACE"


def test_classify_ily():
    assert g.classify_hand(make_hand(extended=("thumb", "index", "pinky"))) == "ILY"


def test_classify_ok_needs_thumb_and_index_touching():
    pts = make_hand(extended=("middle", "ring", "pinky"), thumb_index_gap=8)
    assert g.classify_hand(pts) == "OK"


def test_classify_returns_none_for_fist():
    assert g.classify_hand(make_hand(extended=())) is None


def test_classify_returns_none_for_open_palm():
    # telapak terbuka penuh bukan gestur apa pun -> tidak boleh salah kenal
    assert g.classify_hand(make_hand(extended=("thumb", "index", "middle",
                                               "ring", "pinky"))) is None


def test_heart_needs_two_hands():
    one = [make_hand(extended=("index",))]
    assert g.detect_two_hand_heart(one, 1280) is False


def test_heart_detected_when_tips_meet():
    left = make_hand(extended=("thumb", "index"), origin=(600, 400))
    right = make_hand(extended=("thumb", "index"), origin=(680, 400))
    # paksa ujung telunjuk bertemu di atas, ujung jempol berdekatan di bawah
    left[g.INDEX_TIP] = (630, 300)
    right[g.INDEX_TIP] = (650, 300)
    left[g.THUMB_TIP] = (600, 400)
    right[g.THUMB_TIP] = (680, 400)
    assert g.detect_two_hand_heart([left, right], 1280) is True


def test_heart_rejected_when_index_tips_below_thumbs():
    left = make_hand(origin=(600, 400))
    right = make_hand(origin=(680, 400))
    left[g.INDEX_TIP] = (630, 500)
    right[g.INDEX_TIP] = (650, 500)
    left[g.THUMB_TIP] = (600, 300)
    right[g.THUMB_TIP] = (680, 300)
    assert g.detect_two_hand_heart([left, right], 1280) is False


def test_kicaw_detected_for_either_hand_role():
    mouth_hand = make_hand(extended=(), origin=(640, 200), scale=60.0)
    open_hand = make_hand(extended=("index", "middle", "ring", "pinky"),
                          origin=(300, 500))
    mouth = (640, 200, 120.0)
    assert g.detect_kicaw([mouth_hand, open_hand], 1280, 720, mouth) is True
    # urutan tangan dibalik: hasilnya harus sama
    assert g.detect_kicaw([open_hand, mouth_hand], 1280, 720, mouth) is True


def test_kicaw_rejected_when_second_hand_is_closed():
    mouth_hand = make_hand(extended=(), origin=(640, 200), scale=60.0)
    fist = make_hand(extended=(), origin=(300, 500))
    assert g.detect_kicaw([mouth_hand, fist], 1280, 720, (640, 200, 120.0)) is False


def test_debouncer_requires_stable_frames():
    d = g.GestureDebouncer(hold_frames=3)
    assert d.update("PEACE") is None
    assert d.update("PEACE") is None
    assert d.update("PEACE") == "PEACE"


def test_debouncer_resets_on_a_deviating_frame():
    d = g.GestureDebouncer(hold_frames=3)
    d.update("PEACE")
    d.update("PEACE")
    d.update("OK")           # menyimpang -> hitungan ulang dari nol
    assert d.update("PEACE") is None


def test_debouncer_keeps_last_active_until_new_gesture_is_stable():
    d = g.GestureDebouncer(hold_frames=2)
    d.update("PEACE")
    assert d.update("PEACE") == "PEACE"
    # satu frame kosong belum cukup untuk mematikan
    assert d.update(None) == "PEACE"
    assert d.update(None) is None


def test_smoother_matches_hands_by_wrist_when_order_swaps():
    a = make_hand(origin=(200, 300))
    b = make_hand(origin=(900, 300))
    s = g.HandSmoother(alpha=0.5, match_dist=200.0)
    s.update([a, b])
    out = s.update([b, a])            # urutan tertukar
    # tangan yang wrist-nya dekat 200 harus tetap dihaluskan ke tangan pertama
    assert abs(out[0][g.WRIST][0] - 900) < 5
    assert abs(out[1][g.WRIST][0] - 200) < 5


def test_smoother_returns_integer_pixel_points():
    s = g.HandSmoother()
    out = s.update([make_hand()])
    assert all(isinstance(v, int) for v in out[0][0])
```

- [ ] **Step 2: Jalankan test, pastikan gagal**

Run: `python -m pytest tests/test_gestures.py -q`
Expected: FAIL — `AttributeError: module 'app.gestures' has no attribute 'classify_hand'`

- [ ] **Step 3: Pindahkan classifier ke `app/gestures.py`**

Salin dari `git show ee8ce71:foto_kita_blurrr.py`, ganti setiap panggilan `_dist(` menjadi `dist(`:
- `classify_hand`, baris 226–253
- `detect_two_hand_heart`, baris 256–266
- `detect_kicaw`, baris 269–301
- `HandSmoother`, baris 304–338 — ubah default `__init__` menjadi
  `def __init__(self, alpha=config.SMOOTH_ALPHA, match_dist=config.SMOOTH_MATCH_DIST):`
  dan tambahkan `from app import config` di bagian impor.

- [ ] **Step 4: Tulis `GestureDebouncer` (kode baru)**

Ini mengekstrak mesin-status debounce yang saat ini tertanam di dalam `main()`
(baris 532–539) menjadi kelas yang bisa di-test:

```python
class GestureDebouncer:
    """Gestur baru dianggap aktif hanya setelah bertahan stabil N frame.

    Bikin deteksi anti-kedip dan mencegah suara ter-spam. `update` menerima
    gestur mentah frame ini (boleh `None`) dan mengembalikan gestur yang
    benar-benar aktif — yang tetap bertahan sampai kandidat baru cukup stabil.
    """

    def __init__(self, hold_frames=config.HOLD_FRAMES):
        self.hold_frames = hold_frames
        self.candidate = None
        self.count = 0
        self.active = None

    def update(self, current):
        if current == self.candidate:
            self.count += 1
        else:
            self.candidate = current
            self.count = 1
        if self.count >= self.hold_frames:
            self.active = self.candidate  # boleh None: gestur dilepas
        return self.active
```

- [ ] **Step 5: Jalankan test, pastikan lulus**

Run: `python -m pytest -q`
Expected: PASS, 22 test

Kalau `test_classify_ok_needs_thumb_and_index_touching` atau
`test_kicaw_detected_for_either_hand_role` gagal, **jangan longgarkan
classifier** — perbaiki posisi landmark di `make_hand` sampai pose sintetisnya
benar-benar menyerupai pose asli. Classifier adalah perilaku yang sedang
dilindungi; fixture-lah yang boleh disesuaikan.

- [ ] **Step 6: Commit**

```bash
git add app/gestures.py tests/test_gestures.py
git commit -m "Ekstrak classifier gestur, smoother, dan debouncer

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Primitif menggambar dan adegan per-gestur

**Files:**
- Create: `app/draw.py`, `app/scenes.py`, `tests/test_draw.py`

**Interfaces:**
- Consumes: `app.gestures.HAND_CONNECTIONS`, `app.config`
- Produces:
  - `app.draw.draw_text(img, text, center, scale, color, thickness=2)`
  - `app.draw.hsv_color(hue: float) -> tuple[int, int, int]`
  - `app.draw.draw_note(img, x, y, color, s=1.0)`
  - `app.draw.draw_heart(img, cx, cy, size, color)`
  - `app.draw.draw_hand_skeleton(img, pts, color)`
  - `app.draw.dim_color(color, f: float) -> tuple[int, int, int]`
  - `app.draw.lerp_color(c1, c2, t: float) -> tuple[int, int, int]`
  - `app.draw.make_vignette_layer(w, h, strength=0.30, inner=0.45) -> np.ndarray` (uint8 BGR)
  - `app.scenes.GestureScenes()` dengan `render(frame, gesture, t, effect_fn=None, prev_effect_fn=None, blend=1.0)`
  - `app.scenes.LABELS` — dict gestur → teks

- [ ] **Step 1: Tulis test yang gagal**

`tests/test_draw.py`:

```python
"""Test primitif menggambar. Yang diuji: tidak crash, dan tidak menulis di
luar batas frame — dua kegagalan yang paling mungkin terjadi di kode OpenCV."""

import numpy as np

from app import draw


def blank(h=120, w=160):
    return np.zeros((h, w, 3), np.uint8)


def test_hsv_color_returns_bgr_triple_in_range():
    c = draw.hsv_color(0.33)
    assert len(c) == 3
    assert all(0 <= v <= 255 for v in c)


def test_hsv_color_wraps_around():
    assert draw.hsv_color(0.0) == draw.hsv_color(1.0)


def test_dim_color_clamps():
    assert draw.dim_color((255, 255, 255), 0.0) == (0, 0, 0)
    assert draw.dim_color((255, 255, 255), 2.0) == (255, 255, 255)


def test_lerp_color_endpoints():
    assert draw.lerp_color((0, 0, 0), (10, 20, 30), 0.0) == (0, 0, 0)
    assert draw.lerp_color((0, 0, 0), (10, 20, 30), 1.0) == (10, 20, 30)


def test_draw_text_marks_the_frame():
    img = blank()
    draw.draw_text(img, "HALO", (80, 60), 1.0, (255, 255, 255))
    assert img.any()


def test_draw_text_far_outside_frame_does_not_raise():
    img = blank()
    draw.draw_text(img, "HALO", (-900, -900), 1.0, (255, 255, 255))
    draw.draw_text(img, "HALO", (9000, 9000), 1.0, (255, 255, 255))


def test_draw_heart_and_note_stay_inside_bounds():
    img = blank()
    draw.draw_heart(img, 80, 60, 30, (180, 105, 255))
    draw.draw_note(img, 20, 100, (0, 220, 255))
    assert img.shape == (120, 160, 3)


def test_draw_hand_skeleton_handles_points_outside_frame():
    img = blank()
    pts = [(-50, -50)] * 21
    draw.draw_hand_skeleton(img, pts, (0, 255, 0))


def test_vignette_layer_is_bright_at_center_and_dark_at_corner():
    layer = draw.make_vignette_layer(200, 100)
    assert layer.dtype == np.uint8
    assert layer.shape == (100, 200, 3)
    assert layer[50, 100, 0] > layer[0, 0, 0]
```

- [ ] **Step 2: Jalankan test, pastikan gagal**

Run: `python -m pytest tests/test_draw.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.draw'`

- [ ] **Step 3: Tulis `app/draw.py`**

Salin dari `git show ee8ce71:foto_kita_blurrr.py`: `draw_text` (344–349),
`hsv_color` (352–356), `draw_note` (359–365), `draw_heart` (368–374),
`draw_hand_skeleton` (377–383). Impor `HAND_CONNECTIONS` dari `app.gestures`.

Tambahkan tiga fungsi baru (dua diadaptasi dari `.reference/portal_app.py`):

```python
def dim_color(color, f):
    """Redupkan warna BGR ke arah hitam sebesar faktor f (0..1)."""
    return tuple(max(0, min(255, int(round(c * f)))) for c in color)


def lerp_color(c1, c2, t):
    """Interpolasi linear antara dua warna BGR."""
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(c1, c2))


def make_vignette_layer(w, h, strength=0.30, inner=0.45):
    """Lapisan vignette radial siap-kali sebagai uint8 BGR (0 = hitam).

    Dihitung sekali lalu dipakai ulang tiap frame lewat cv2.multiply.
    Diadaptasi dari .reference/portal_app.py:587-596.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx / max(w - 1, 1)) * 2.0 - 1.0
    ny = (yy / max(h - 1, 1)) * 2.0 - 1.0
    d = np.sqrt(nx * nx + ny * ny)
    t = np.clip((d - inner) / (1.6 - inner), 0.0, 1.0)
    v = 1.0 - strength * t * t
    return (np.stack([v] * 3, axis=-1) * 255.0).astype(np.uint8)
```

`draw_text` perlu satu penjaga tambahan supaya `test_draw_text_far_outside_frame_does_not_raise`
lulus — `cv2.putText` sendiri aman di luar batas, jadi cukup pastikan koordinat
dikonversi ke `int` seperti kode aslinya. Jangan tambah clamping yang mengubah
posisi teks di dalam frame.

- [ ] **Step 4: Tulis `app/scenes.py`**

Mengekstrak `LABELS` (389–395), state partikel modul-level `_hearts`/`_confetti`
(397–398), dan `render_effect` (401–459) dari file lama. Perubahan yang
disengaja: state partikel dipindah dari global modul ke atribut instance, dan
cabang `PEACE` menerima fungsi filter dari luar alih-alih meng-hardcode
`cv2.GaussianBlur`. Di task ini pemanggil masih mengirim `None`, sehingga
perilaku tetap persis sama (blur bawaan); Task 7 yang mulai mengirim filter.

```python
"""Efek visual untuk tiap gestur.

Semua state partikel dipegang instance, bukan global modul, supaya dua adegan
tidak pernah saling mencemari dan test bisa memulai dari kondisi bersih.
"""

import math
import random

import cv2
import numpy as np

from app import config
from app.draw import draw_heart, draw_note, draw_text, hsv_color

LABELS = {
    "PEACE": "FOTO KITA BLURRR",
    "HEART": "I LOVE YOU",
    "ILY": "GOKILL",
    "OK": "OKE",
    "KICAW": "KICAW MANIA!",
}

BLUR_THUMB = None
_img = cv2.imread(config.IMG_BLUR) if os.path.exists(config.IMG_BLUR) else None
if _img is not None:
    BLUR_THUMB = cv2.resize(_img, (160, 90))
elif not os.path.exists(config.IMG_BLUR):
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
```

Metode `_peace`, `_heart`, `_ily`, `_ok`, `_kicaw` berisi persis isi tiap cabang
`if` dari `render_effect` lama (baris 408–459), dengan dua penyesuaian:
`_hearts` → `self.hearts`, `_confetti` → `self.confetti`. Cabang `_peace`
diawali:

```python
    def _peace(self, frame, cx, cy, w, h, pulse, effect_fn,
               prev_effect_fn, blend):
        if effect_fn is None:
            k = 45
            frame[:] = cv2.GaussianBlur(frame, (k, k), 0)
        else:
            from app.effects import apply_full_frame  # diisi di Task 7
            frame[:] = apply_full_frame(frame, effect_fn, prev_effect_fn, blend)
        draw_text(frame, LABELS["PEACE"], (cx, cy), 1.8 * pulse,
                  (255, 255, 255), 3)
        if BLUR_THUMB is not None:
            bh, bw = BLUR_THUMB.shape[:2]
            frame[h - bh - 15:h - 15, w - bw - 15:w - 15] = BLUR_THUMB
```

Jangan lupa `import os` di bagian atas file.

- [ ] **Step 5: Jalankan test, pastikan lulus**

Run: `python -m pytest -q`
Expected: PASS, 31 test

- [ ] **Step 6: Commit**

```bash
git add app/draw.py app/scenes.py tests/test_draw.py
git commit -m "Ekstrak primitif gambar dan adegan per-gestur

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Audio dan pemuatan model

**Files:**
- Create: `app/audio.py`, `app/models.py`, `tests/test_audio.py`

**Interfaces:**
- Consumes: `app.config`
- Produces:
  - `app.audio.AudioPlayer(paths: dict[str, str])` dengan `.play(key)`, `.stop(key)`, `.toggle_mute() -> bool`, `.close()`, dan properti `.enabled`, `.muted`
  - `app.models.ensure_model(path, url, label, required=True) -> bool`
  - `app.models.make_landmarker(running_mode)`
  - `app.models.make_face_detector(running_mode)` — `None` kalau model wajah tidak ada
  - `app.models.mouth_from_faces(face_result, w, h) -> tuple[float, float, float] | None`

- [ ] **Step 1: Tulis test yang gagal**

`tests/test_audio.py` — menguji jalur degradasi, yang justru paling penting dan
paling jarang teruji. Tidak butuh sound device:

```python
"""AudioPlayer harus jalan terus walau tidak ada sound device, tidak ada
pygame, atau file suaranya hilang. Test ini memakai stub, bukan device asli."""

import pytest

from app.audio import AudioPlayer


class FakeChannel:
    def __init__(self):
        self.busy = True
        self.stopped = False

    def get_busy(self):
        return self.busy

    def stop(self):
        self.stopped = True
        self.busy = False


class FakeSound:
    def __init__(self):
        self.plays = 0
        self.channel = FakeChannel()

    def play(self):
        self.plays += 1
        return self.channel


def test_player_is_disabled_when_mixer_fails(monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("tidak ada audio device")

    p = AudioPlayer({}, mixer_init=boom)
    assert p.enabled is False
    p.play("blur")   # tidak boleh melempar
    p.stop("blur")
    p.close()


def test_missing_sound_file_is_skipped_not_fatal(tmp_path):
    p = AudioPlayer({"blur": str(tmp_path / "tidak-ada.wav")},
                    mixer_init=lambda: None, sound_factory=lambda _p: FakeSound())
    assert p.enabled is True
    assert "blur" not in p.sounds
    p.play("blur")   # tidak boleh melempar


def test_play_does_not_stack_while_still_sounding(tmp_path):
    path = tmp_path / "a.wav"
    path.write_bytes(b"x")
    sound = FakeSound()
    p = AudioPlayer({"blur": str(path)}, mixer_init=lambda: None,
                    sound_factory=lambda _p: sound)
    p.play("blur")
    p.play("blur")
    assert sound.plays == 1


def test_stop_stops_the_active_channel(tmp_path):
    path = tmp_path / "a.wav"
    path.write_bytes(b"x")
    sound = FakeSound()
    p = AudioPlayer({"blur": str(path)}, mixer_init=lambda: None,
                    sound_factory=lambda _p: sound)
    p.play("blur")
    p.stop("blur")
    assert sound.channel.stopped is True


def test_mute_silences_playback_and_stops_what_is_sounding(tmp_path):
    path = tmp_path / "a.wav"
    path.write_bytes(b"x")
    sound = FakeSound()
    p = AudioPlayer({"blur": str(path)}, mixer_init=lambda: None,
                    sound_factory=lambda _p: sound)
    p.play("blur")
    assert p.toggle_mute() is True
    assert sound.channel.stopped is True
    p.play("blur")
    assert sound.plays == 1        # dibisukan -> tidak main lagi
    assert p.toggle_mute() is False
```

- [ ] **Step 2: Jalankan test, pastikan gagal**

Run: `python -m pytest tests/test_audio.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.audio'`

- [ ] **Step 3: Tulis `app/audio.py`**

Mengekstrak blok audio lama (baris 56–95) menjadi kelas. `mixer_init` dan
`sound_factory` bisa disuntik supaya test tidak butuh sound device:

```python
"""Pemutar suara. Setiap kegagalan menurunkan kemampuan, tidak menghentikan
aplikasi: tanpa device, tanpa pygame, atau tanpa file, program tetap jalan
dengan efek visual saja."""

import os


def _default_mixer_init():
    import pygame
    pygame.mixer.init()


def _default_sound_factory(path):
    import pygame
    return pygame.mixer.Sound(path)


class AudioPlayer:
    """Memutar satu suara per kunci, tanpa menumpuk kalau masih berbunyi."""

    def __init__(self, paths, mixer_init=None, sound_factory=None):
        self.enabled = False
        self.muted = False
        self.sounds = {}
        self._channels = {}
        mixer_init = mixer_init or _default_mixer_init
        sound_factory = sound_factory or _default_sound_factory
        try:
            mixer_init()
            self.enabled = True
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] audio dimatikan (mixer gagal init): {exc}")
            return
        for key, path in paths.items():
            if not os.path.exists(path):
                print(f"[warn] sound tidak ditemukan: {path}")
                continue
            try:
                self.sounds[key] = sound_factory(path)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] gagal load sound {path}: {exc}")

    def play(self, key):
        """Mainkan sekali. Tidak mengulang kalau masih berbunyi (anti-spam)."""
        if not self.enabled or self.muted or key not in self.sounds:
            return
        ch = self._channels.get(key)
        if ch is not None and ch.get_busy():
            return
        try:
            self._channels[key] = self.sounds[key].play()
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] gagal play {key}: {exc}")

    def stop(self, key):
        """Hentikan saat gestur dilepas, biar tidak terus berbunyi."""
        ch = self._channels.get(key)
        if ch is not None and ch.get_busy():
            ch.stop()

    def toggle_mute(self):
        self.muted = not self.muted
        if self.muted:
            for key in list(self._channels):
                self.stop(key)
        return self.muted

    def close(self):
        if not self.enabled:
            return
        try:
            import pygame
            pygame.mixer.quit()
        except Exception:  # noqa: BLE001
            pass
```

- [ ] **Step 4: Tulis `app/models.py`**

Mengekstrak `make_landmarker` (110–128), `make_face_detector` (131–147), dan
`mouth_from_faces` (150–164) dari file lama, ditambah auto-download yang
diadaptasi dari `.reference/portal_app.py:145-150`. Perubahan perilaku yang
disengaja: model tangan yang hilang **diunduh otomatis**, bukan langsung
`SystemExit`.

```python
"""Pemuatan model MediaPipe, termasuk unduhan otomatis sekali jalan."""

import os
import urllib.request

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from app import config
from app.gestures import dist


def ensure_model(path, url, label, required=True):
    """Pastikan file model ada; unduh kalau belum. Return True kalau tersedia.

    Kalau `required` dan unduhan gagal, lempar SystemExit dengan perintah
    unduh manual — itu satu-satunya kegagalan yang memang tidak bisa dilanjutkan.
    """
    if os.path.exists(path):
        return True
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[info] mengunduh model {label} (sekali saja)...")
    try:
        urllib.request.urlretrieve(url, path, _report_progress)
        print(f"\n[info] tersimpan di {path}")
        return True
    except Exception as exc:  # noqa: BLE001
        if os.path.exists(path):
            os.remove(path)   # buang file separuh jadi
        msg = f"[warn] gagal mengunduh model {label}: {exc}"
        if not required:
            print(msg)
            return False
        raise SystemExit(
            f"{msg}\nUnduh manual lalu jalankan lagi:\n"
            f"  curl -L -o \"{path}\" {url}"
        )


def _report_progress(block_num, block_size, total_size):
    if total_size <= 0:
        return
    done = min(100, block_num * block_size * 100 // total_size)
    print(f"\r  {done}%", end="", flush=True)
```

Lalu `make_landmarker` memanggil `ensure_model(config.MODEL_PATH,
config.MODEL_URL, "hand_landmarker", required=True)` menggantikan cek
`os.path.exists` + `SystemExit` yang lama, dan `make_face_detector` memanggil
`ensure_model(config.FACE_MODEL_PATH, config.FACE_MODEL_URL,
"blaze_face_short_range", required=False)` — kalau `False`, kembalikan `None`
dan cetak info bahwa Kicaw memakai perkiraan posisi (pesan sama seperti baris
134–140 file lama). `mouth_from_faces` disalin apa adanya dengan `_dist` → `dist`.

- [ ] **Step 5: Jalankan test, pastikan lulus**

Run: `python -m pytest -q`
Expected: PASS, 36 test

- [ ] **Step 6: Commit**

```bash
git add app/audio.py app/models.py tests/test_audio.py
git commit -m "Ekstrak audio dan pemuatan model, tambah unduh model otomatis

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Loop utama — ekstraksi selesai, aplikasi jalan lagi

Ini titik pemeriksaan pertama: setelah task ini aplikasi harus berperilaku
**persis** seperti commit `ee8ce71`, hanya dengan struktur berbeda dan model
yang terunduh otomatis.

**Files:**
- Create: `app/main.py`
- Rewrite: `foto_kita_blurrr.py`

**Interfaces:**
- Consumes: semua yang dihasilkan Task 1–4
- Produces: `app.main.main() -> None`

- [ ] **Step 1: Tulis `app/main.py`**

Diadaptasi dari `main()` lama (baris 465–570). Perbedaan yang disengaja:
mesin debounce inline diganti `GestureDebouncer`, blok audio inline diganti
`AudioPlayer`, dan `SOUND_FOR` dipindah ke sini.

```python
"""Loop kamera, mesin mode, dan penanganan keyboard.

Satu-satunya modul yang menyentuh kamera, jam, dan keyboard. Semua logika yang
bisa di-test tanpa perangkat keras ada di modul lain.
"""

import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision

from app import config
from app.audio import AudioPlayer
from app.draw import draw_hand_skeleton, hsv_color
from app.gestures import (GestureDebouncer, HandSmoother, classify_hand,
                          detect_kicaw, detect_two_hand_heart, hand_points)
from app.models import make_face_detector, make_landmarker, mouth_from_faces
from app.scenes import GestureScenes

WINDOW = "Foto-Kita-Blurrr"
SOUND_FOR = {"PEACE": "blur", "KICAW": "kicaw"}


def open_camera():
    """Buka webcam dengan resolusi yang diminta. Keluar kalau tidak ada."""
    cap = cv2.VideoCapture(config.CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)
    if not cap.isOpened():
        raise SystemExit(
            "Kamera tidak bisa dibuka. Pastikan webcam tersambung dan tidak "
            f"dipakai aplikasi lain, atau ubah CAM_INDEX (sekarang "
            f"{config.CAM_INDEX}) di app/config.py."
        )
    return cap


def main():
    cap = open_camera()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)

    audio = AudioPlayer({"blur": config.SOUND_BLUR, "kicaw": config.SOUND_KICAW})
    scenes = GestureScenes()
    smoother = HandSmoother()
    debouncer = GestureDebouncer()
    face_detector = make_face_detector(vision.RunningMode.VIDEO)

    prev_active = None
    start = time.time()
    last_ts = -1

    with make_landmarker(vision.RunningMode.VIDEO) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)      # mode selfie / cermin
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if w > config.DETECT_WIDTH:
                det_img = cv2.resize(
                    rgb, (config.DETECT_WIDTH,
                          int(h * config.DETECT_WIDTH / w)))
            else:
                det_img = rgb
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=det_img)
            ts = max(last_ts + 1, int((time.time() - start) * 1000))
            last_ts = ts
            result = landmarker.detect_for_video(mp_image, ts)

            mouth = None
            if face_detector is not None:
                face_res = face_detector.detect_for_video(mp_image, ts)
                mouth = mouth_from_faces(face_res, w, h)

            hands_pts = smoother.update(
                [hand_points(lm, w, h) for lm in result.hand_landmarks])

            current = None
            if detect_two_hand_heart(hands_pts, w):
                current = "HEART"
            elif detect_kicaw(hands_pts, w, h, mouth):
                current = "KICAW"
            else:
                for pts in hands_pts:
                    g = classify_hand(pts)
                    if g is not None:
                        current = g
                        break
            active = debouncer.update(current)

            t = time.time()
            for pts in hands_pts:
                draw_hand_skeleton(frame, pts, hsv_color((t * 0.25) % 1.0))
            if active:
                scenes.render(frame, active, t)

            # suara: mulai saat gestur masuk, berhenti saat dilepas/ganti
            if active != prev_active:
                old_key = SOUND_FOR.get(prev_active)
                if old_key:
                    audio.stop(old_key)
                new_key = SOUND_FOR.get(active)
                if new_key:
                    audio.play(new_key)
            prev_active = active

            cv2.imshow(WINDOW, frame)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()
    if face_detector is not None:
        face_detector.close()
    audio.close()
```

- [ ] **Step 2: Tulis ulang `foto_kita_blurrr.py` jadi entry point tipis**

```python
"""Foto-Kita-Blurrr — kamera real-time dengan efek gestur tangan.

Jalankan:  python foto_kita_blurrr.py
Kontrol lengkap ada di README.md.
"""

from app.main import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Jalankan seluruh test**

Run: `python -m pytest -q`
Expected: PASS, 36 test

- [ ] **Step 4: Jalankan aplikasinya — pemeriksaan manual**

Run: `python foto_kita_blurrr.py`

Yang harus terlihat, sama persis seperti sebelum refactor: skeleton tangan
berwarna berputar; ✌️ bikin layar blur + tulisan `FOTO KITA BLURRR` + suara;
🫶 🤟 👌 🐦 memunculkan tulisan dan animasi masing-masing; `Q`/`ESC` keluar.
Kalau folder `models/` dikosongkan lebih dulu, model harus terunduh otomatis
dengan indikator persen.

Kalau ada satu saja perilaku yang berubah, **perbaiki sebelum lanjut** — seluruh
task berikutnya bertumpu pada fondasi ini.

- [ ] **Step 5: Commit**

```bash
git add app/main.py foto_kita_blurrr.py
git commit -m "Pindahkan loop utama ke app/main.py, jadikan entry point tipis

Ekstraksi ke paket app/ selesai. Perilaku tidak berubah dari ee8ce71,
kecuali model MediaPipe kini terunduh otomatis.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: `PinchTapDetector` dan perombakan gestur 👌 OK

Task pertama yang mengubah perilaku. Unit paling padat test di seluruh project.

**Files:**
- Modify: `app/gestures.py`, `app/main.py`
- Create: `tests/test_pinch.py`
- Modify: `tests/test_gestures.py`

**Interfaces:**
- Consumes: `app.config.{PINCH_ENTER, PINCH_EXIT, PINCH_HOLD_S}`
- Produces:
  - `app.gestures.PinchEvent` — `namedtuple("PinchEvent", "tap hold_started holding")`
  - `app.gestures.pinch_distance(pts) -> float` — jarak jempol↔telunjuk **ternormalisasi terhadap `palm_scale`**, sehingga tidak berubah saat tangan mendekat/menjauh dari kamera
  - `app.gestures.PinchTapDetector(hold_s, enter, exit).update(dists: list[float], now: float) -> PinchEvent`
  - `PinchTapDetector.adjust(delta: float)` — geser kedua ambang, menjaga selisih histeresis
  - `classify_hand` **tidak lagi** mengembalikan `"OK"`

- [ ] **Step 1: Tulis test yang gagal**

`tests/test_pinch.py`:

```python
"""Mesin-status pinch. Satu ambang memisahkan dua hasil:

    lepas SEBELUM 0,35 dtk   -> TAP  (ganti efek)
    bertahan DI 0,35 dtk     -> HOLD (gestur OK)

Semua test memakai jam yang di-inject; tidak ada `sleep` di sini.
"""

import pytest

from app import config
from app.gestures import PinchTapDetector

TIGHT = 0.03    # jelas di bawah PINCH_ENTER -> dianggap menyentuh
LOOSE = 0.20    # jelas di atas PINCH_EXIT   -> dianggap terlepas


@pytest.fixture
def det():
    return PinchTapDetector()


def test_quick_tap_fires_once(det, clock):
    assert det.update([TIGHT], clock.now).tap is False   # baru menyentuh
    clock.advance(0.1)
    assert det.update([TIGHT], clock.now).tap is False   # masih ditahan
    clock.advance(0.05)
    assert det.update([LOOSE], clock.now).tap is True    # dilepas -> TAP
    clock.advance(0.05)
    assert det.update([LOOSE], clock.now).tap is False   # tidak berulang


def test_long_hold_never_fires_a_tap(det, clock):
    det.update([TIGHT], clock.now)
    clock.advance(1.0)
    det.update([TIGHT], clock.now)
    assert det.update([LOOSE], clock.now).tap is False


def test_hold_starts_exactly_once(det, clock):
    det.update([TIGHT], clock.now)
    clock.advance(config.PINCH_HOLD_S)
    first = det.update([TIGHT], clock.now)
    assert first.hold_started is True
    clock.advance(0.1)
    second = det.update([TIGHT], clock.now)
    assert second.hold_started is False
    assert second.holding is True        # tetap aktif selama ditahan


def test_holding_is_false_before_the_threshold(det, clock):
    det.update([TIGHT], clock.now)
    clock.advance(config.PINCH_HOLD_S - 0.01)
    ev = det.update([TIGHT], clock.now)
    assert ev.holding is False
    assert ev.hold_started is False


def test_exactly_at_threshold_is_a_hold_not_a_tap(det, clock):
    det.update([TIGHT], clock.now)
    clock.advance(config.PINCH_HOLD_S)
    ev = det.update([LOOSE], clock.now)   # dilepas persis di ambang
    assert ev.tap is False


def test_release_then_pinch_again_fires_again(det, clock):
    det.update([TIGHT], clock.now)
    clock.advance(0.05)
    assert det.update([LOOSE], clock.now).tap is True
    clock.advance(0.05)
    det.update([TIGHT], clock.now)
    clock.advance(0.05)
    assert det.update([LOOSE], clock.now).tap is True


def test_two_hands_pinching_together_fire_only_once(det, clock):
    det.update([TIGHT, TIGHT], clock.now)
    clock.advance(0.05)
    ev = det.update([LOOSE, LOOSE], clock.now)
    assert ev.tap is True
    clock.advance(0.05)
    assert det.update([LOOSE, LOOSE], clock.now).tap is False


def test_second_hand_releasing_late_does_not_fire_twice(det, clock):
    det.update([TIGHT, TIGHT], clock.now)
    clock.advance(0.05)
    assert det.update([TIGHT, LOOSE], clock.now).tap is False  # satu masih menyentuh
    clock.advance(0.05)
    assert det.update([LOOSE, LOOSE], clock.now).tap is True   # baru sekarang


def test_jitter_around_the_threshold_does_not_chatter(det, clock):
    """Histeresis: nilai di antara ENTER dan EXIT tidak mengubah status."""
    between = (config.PINCH_ENTER + config.PINCH_EXIT) / 2
    det.update([TIGHT], clock.now)
    taps = 0
    for _ in range(10):
        clock.advance(0.01)
        if det.update([between], clock.now).tap:
            taps += 1
    assert taps == 0


def test_no_hands_releases_the_pinch(det, clock):
    det.update([TIGHT], clock.now)
    clock.advance(0.05)
    assert det.update([], clock.now).tap is True


def test_adjust_shifts_both_thresholds_and_keeps_the_gap(det):
    gap = det.exit - det.enter
    det.adjust(config.PINCH_STEP)
    assert det.exit - det.enter == pytest.approx(gap)
    assert det.enter > config.PINCH_ENTER


def test_adjust_is_clamped(det):
    for _ in range(200):
        det.adjust(config.PINCH_STEP)
    assert det.enter <= config.PINCH_MAX
    for _ in range(400):
        det.adjust(-config.PINCH_STEP)
    assert det.enter >= config.PINCH_MIN
```

Di `tests/test_gestures.py`, **ganti** `test_classify_ok_needs_thumb_and_index_touching`
dengan:

```python
def test_classify_no_longer_returns_ok_from_finger_shape():
    """👌 OK sekarang dipicu pinch-ditahan (lihat tests/test_pinch.py),
    bukan bentuk jari. classify_hand tidak boleh lagi mengklaimnya."""
    pts = make_hand(extended=("middle", "ring", "pinky"), thumb_index_gap=8)
    assert g.classify_hand(pts) != "OK"


def test_pinch_distance_is_normalized_by_palm_size():
    near = make_hand(extended=("middle", "ring", "pinky"), scale=60.0,
                     thumb_index_gap=6)
    far = make_hand(extended=("middle", "ring", "pinky"), scale=120.0,
                    thumb_index_gap=12)
    assert g.pinch_distance(near) == pytest.approx(g.pinch_distance(far), abs=0.02)
```

Tambahkan `import pytest` di puncak `tests/test_gestures.py`.

- [ ] **Step 2: Jalankan test, pastikan gagal**

Run: `python -m pytest tests/test_pinch.py -q`
Expected: FAIL — `ImportError: cannot import name 'PinchTapDetector'`

- [ ] **Step 3: Tulis `PinchTapDetector` di `app/gestures.py`**

```python
PinchEvent = collections.namedtuple("PinchEvent", "tap hold_started holding")


def pinch_distance(pts):
    """Jarak ujung jempol <-> ujung telunjuk, dinormalkan ke ukuran telapak.

    Dinormalkan supaya ambangnya tetap benar saat tangan mendekat atau menjauh
    dari kamera.
    """
    return dist(pts[THUMB_TIP], pts[INDEX_TIP]) / palm_scale(pts)


class PinchTapDetector:
    """Satu mesin-status untuk kedua tangan sekaligus.

        sentuh lalu lepas SEBELUM hold_s  -> TAP  (ganti efek)
        sentuh dan bertahan DI hold_s     -> HOLD (gestur OK)

    Masing-masing memicu tepat sekali per pinch. Dua ambang jarak berbeda
    (masuk lebih ketat daripada keluar) meredam jitter landmark di sekitar
    ambang. Kedua tangan berbagi satu status, jadi pinch bersamaan tetap
    dihitung satu kali.
    """

    def __init__(self, hold_s=config.PINCH_HOLD_S, enter=config.PINCH_ENTER,
                 exit=config.PINCH_EXIT):
        self.hold_s = hold_s
        self.enter = enter
        self.exit = exit
        self._pinching = False
        self._start = 0.0
        self._held = False

    def adjust(self, delta):
        """Geser sensitivitas; selisih histeresis dipertahankan."""
        gap = self.exit - self.enter
        self.enter = max(config.PINCH_MIN,
                         min(config.PINCH_MAX, self.enter + delta))
        self.exit = self.enter + gap

    def update(self, dists, now):
        """`dists` = jarak pinch ternormalisasi tiap tangan; boleh kosong."""
        d = min(dists) if dists else float("inf")
        tap = False
        hold_started = False
        if self._pinching:
            if d > self.exit:                       # dilepas
                if (now - self._start) < self.hold_s:
                    tap = True
                self._pinching = False
                self._held = False
            elif not self._held and (now - self._start) >= self.hold_s:
                self._held = True                   # jadi HOLD
                hold_started = True
        elif d < self.enter:
            self._pinching = True
            self._start = now
            self._held = False
        return PinchEvent(tap, hold_started, self._held)
```

Tambahkan `import collections` di puncak modul.

- [ ] **Step 4: Buang aturan OK dari `classify_hand`**

Hapus blok `if d_thumb_index < 0.42 * scale and middle and ring and pinky:
return "OK"` beserta komentarnya. Perbarui docstring fungsi supaya menyebut
bahwa OK kini ditangani `PinchTapDetector`.

- [ ] **Step 5: Sambungkan di `app/main.py`**

Tambah impor `PinchTapDetector, pinch_distance` dari `app.gestures`. Buat
`pinch = PinchTapDetector()` di samping `smoother`, lalu di dalam loop —
tepat sebelum blok penentuan `current` — sisipkan:

```python
            # Pinch: TAP mengganti efek, HOLD memunculkan gestur 👌 OK.
            ev = pinch.update([pinch_distance(p) for p in hands_pts], t_now)
```

di mana `t_now = time.time()` diambil satu kali di awal iterasi dan dipakai
ulang sebagai `t` untuk animasi (ganti `t = time.time()` yang sekarang ada di
tengah loop). Lalu ubah penentuan gestur menjadi:

```python
            current = None
            if detect_two_hand_heart(hands_pts, w):
                current = "HEART"
            elif detect_kicaw(hands_pts, w, h, mouth):
                current = "KICAW"
            elif ev.holding:
                current = "OK"
            else:
                for pts in hands_pts:
                    g = classify_hand(pts)
                    if g is not None:
                        current = g
                        break
```

Untuk OK, durasi tahan **menggantikan** debounce — jadi setelah `active =
debouncer.update(current)`, tambahkan:

```python
            # 👌 OK sudah dijaga ambang 0,35 dtk; menumpuk debounce di atasnya
            # membuatnya terasa lamban (~0,55 dtk), jadi OK dilewatkan langsung.
            if current == "OK":
                active = "OK"
                debouncer.force("OK")
```

dan tambahkan ke `GestureDebouncer`:

```python
    def force(self, gesture):
        """Setel gestur aktif langsung, melewati hitungan stabil.

        Dipakai gestur yang punya penjaga waktunya sendiri (👌 OK), supaya
        tidak terkena dua penundaan berturut-turut.
        """
        self.candidate = gesture
        self.count = self.hold_frames
        self.active = gesture
```

`ev.tap` belum dipakai di task ini — Task 7 yang menyambungkannya ke pergantian
filter. Jangan tambahkan `pass`-through kosong; biarkan saja.

- [ ] **Step 6: Jalankan test, pastikan lulus**

Run: `python -m pytest -q`
Expected: PASS, 49 test

- [ ] **Step 7: Pemeriksaan manual**

Run: `python foto_kita_blurrr.py`

Sentuh ujung jempol dan telunjuk lalu **tahan** — `OKE` harus muncul kira-kira
sepertiga detik kemudian, dan bentuk tiga jari lain tidak lagi berpengaruh.
Sentuh **cepat** lalu lepas — belum ada yang terjadi (itu benar; TAP baru
tersambung di Task 7).

- [ ] **Step 8: Commit**

```bash
git add app/gestures.py app/main.py tests/test_pinch.py tests/test_gestures.py
git commit -m "Tambah PinchTapDetector, pindahkan gestur OK ke pinch-ditahan

Sentuh lalu lepas <0,35 dtk = TAP (untuk ganti efek, disambung di task
berikutnya); bertahan >=0,35 dtk = gestur OK. Bentuk jari tidak lagi
menentukan OK.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: Sembilan filter gambar + pinch-tap menggantinya

**Files:**
- Create: `app/effects.py`, `tests/test_effects.py`
- Modify: `app/main.py`, `app/scenes.py`

**Interfaces:**
- Consumes: `app.config.EFFECT_SCALE`, `PinchEvent.tap` dari Task 6
- Produces:
  - `app.effects.EFFECTS: list[Callable[[np.ndarray], np.ndarray]]` — 9 fungsi
  - `app.effects.EFFECT_NAMES: list[str]` — 9 nama, indeks sejajar dengan `EFFECTS`
  - `app.effects.apply_full_frame(frame, fn, prev_fn=None, blend=1.0, scale=EFFECT_SCALE) -> np.ndarray`
  - Fungsi individual: `fx_blur, fx_thermal, fx_edge_mesh, fx_posterize_neon, fx_invert_glitch, fx_sketch, fx_chromatic, fx_pixel_mosaic, fx_duotone`

- [ ] **Step 1: Tulis test yang gagal**

`tests/test_effects.py`:

```python
"""Kontrak yang berlaku untuk SEMUA filter: uint8 masuk, uint8 keluar dengan
shape sama, input tidak pernah dimutasi, dan aman untuk crop sekecil apa pun.
Ini yang membuat filter bisa dipasang di mana saja tanpa kejutan."""

import numpy as np
import pytest

from app import effects


def sample(h=48, w=64):
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


@pytest.mark.parametrize("fn", effects.EFFECTS, ids=effects.EFFECT_NAMES)
def test_effect_preserves_shape_and_dtype(fn):
    img = sample()
    out = fn(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


@pytest.mark.parametrize("fn", effects.EFFECTS, ids=effects.EFFECT_NAMES)
def test_effect_does_not_mutate_its_input(fn):
    img = sample()
    before = img.copy()
    fn(img)
    assert np.array_equal(img, before)


@pytest.mark.parametrize("fn", effects.EFFECTS, ids=effects.EFFECT_NAMES)
@pytest.mark.parametrize("size", [(1, 1), (2, 2), (3, 5)])
def test_effect_survives_tiny_crops(fn, size):
    img = sample(*size)
    out = fn(img)
    assert out.shape == img.shape


def test_effects_and_names_line_up():
    assert len(effects.EFFECTS) == len(effects.EFFECT_NAMES) == 9
    assert effects.EFFECT_NAMES[0] == "Blur"


def test_apply_full_frame_returns_same_shape():
    img = sample(120, 160)
    out = effects.apply_full_frame(img, effects.fx_thermal)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_apply_full_frame_blend_endpoints_match_each_effect():
    img = sample(120, 160)
    only_new = effects.apply_full_frame(img, effects.fx_thermal,
                                        effects.fx_duotone, blend=1.0)
    plain_new = effects.apply_full_frame(img, effects.fx_thermal)
    assert np.array_equal(only_new, plain_new)


def test_apply_full_frame_does_not_mutate_its_input():
    img = sample(120, 160)
    before = img.copy()
    effects.apply_full_frame(img, effects.fx_posterize_neon)
    assert np.array_equal(img, before)
```

- [ ] **Step 2: Jalankan test, pastikan gagal**

Run: `python -m pytest tests/test_effects.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.effects'`

- [ ] **Step 3: Tulis `app/effects.py`**

Delapan filter disalin dari `.reference/portal_app.py`: `fx_thermal` (156–158),
`fx_posterize_neon` (161–169), `_K2` + `fx_edge_mesh` (172–182),
`fx_invert_glitch` (185–192), `fx_grayscale_sketch` (195–200) → **ganti nama
jadi `fx_sketch`**, `fx_chromatic` (203–213), `fx_pixel_mosaic` (216–227),
`_build_duotone_lut` + `fx_duotone` (230–247).

`fx_blur` adalah filter baru yang mempertahankan tampilan ✌️ Peace lama, ditulis
supaya juga aman untuk crop kecil (kernel Gaussian tidak boleh melebihi ukuran
gambar):

```python
def fx_blur(img):
    """Blur bawaan ✌️ Peace. Kernel menyesuaikan ukuran gambar supaya crop
    sekecil 1x1 pun tidak membuat OpenCV melempar."""
    h, w = img.shape[:2]
    k = min(45, max(3, (min(h, w) // 8) * 2 + 1))
    return cv2.GaussianBlur(img, (k, k), 0)
```

Daftar dan namanya:

```python
EFFECTS = [
    fx_blur, fx_thermal, fx_edge_mesh, fx_posterize_neon, fx_invert_glitch,
    fx_sketch, fx_chromatic, fx_pixel_mosaic, fx_duotone,
]
EFFECT_NAMES = [
    "Blur", "Thermal", "Edge Mesh", "Posterize Neon", "Invert Glitch",
    "Sketch", "Chromatic", "Pixel Mosaic", "Duotone",
]
```

`apply_full_frame` — inilah mitigasi kinerja dari §8 spec:

```python
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
```

> Perhatikan: saat `blend >= 1.0` `prev_fn` tidak dipanggil sama sekali —
> itulah yang membuat `test_apply_full_frame_blend_endpoints_match_each_effect`
> lulus persis, bukan sekadar mendekati.

- [ ] **Step 4: Jalankan test, pastikan lulus**

Run: `python -m pytest tests/test_effects.py -q`
Expected: PASS, 49 test (9 filter × 5 kasus + 4 test apply_full_frame)

- [ ] **Step 5: Sambungkan pergantian filter di `app/main.py`**

Tambah state sesudah `debouncer`:

```python
    effect_idx = 0
    prev_effect_idx = None     # filter yang sedang di-crossfade keluar
    crossfade_start = 0.0
    label_from = None          # nama filter yang sedang slide keluar
    label_start = 0.0

    def switch_effect(new_idx, now):
        """Pindah filter dengan crossfade + animasi label."""
        nonlocal effect_idx, prev_effect_idx, crossfade_start
        nonlocal label_from, label_start
        if new_idx == effect_idx:
            return
        prev_effect_idx = effect_idx
        crossfade_start = now
        label_from = EFFECT_NAMES[effect_idx]
        label_start = now
        effect_idx = new_idx
```

Sesudah `ev = pinch.update(...)`:

```python
            if ev.tap:
                switch_effect((effect_idx + 1) % len(EFFECTS), t_now)

            if prev_effect_idx is not None:
                blend = min(1.0, (t_now - crossfade_start) / config.CROSSFADE_S)
                if blend >= 1.0:
                    prev_effect_idx = None
            else:
                blend = 1.0
```

Kirim filter ke adegan:

```python
            if active:
                scenes.render(
                    frame, active, t_now,
                    effect_fn=EFFECTS[effect_idx],
                    prev_effect_fn=(EFFECTS[prev_effect_idx]
                                    if prev_effect_idx is not None else None),
                    blend=blend)
```

Tambah penanganan tombol `1`–`9` dan `m` di blok keyboard:

```python
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif ord("1") <= key <= ord("9"):
                n = key - ord("1")
                if n < len(EFFECTS):
                    switch_effect(n, t_now)
                    print(f"Efek: {EFFECT_NAMES[effect_idx]}")
            elif key == ord("m"):
                print(f"Audio: {'BISU' if audio.toggle_mute() else 'NYALA'}")
            elif key in (ord("+"), ord("=")):
                pinch.adjust(config.PINCH_STEP)
                print(f"Ambang pinch: {pinch.enter:.3f}")
            elif key == ord("-"):
                pinch.adjust(-config.PINCH_STEP)
                print(f"Ambang pinch: {pinch.enter:.3f}")
```

Impor `EFFECTS, EFFECT_NAMES` dari `app.effects`.

Terakhir, di `app/scenes.py` hapus impor lokal `from app.effects import
apply_full_frame` yang ditaruh di dalam `_peace` pada Task 3 dan naikkan ke
puncak modul — impor melingkar tidak terjadi karena `app.effects` tidak
mengimpor `app.scenes`.

- [ ] **Step 6: Jalankan seluruh test**

Run: `python -m pytest -q`
Expected: PASS, 98 test

- [ ] **Step 7: Pemeriksaan manual**

Run: `python foto_kita_blurrr.py`

Tahan ✌️ untuk memunculkan efek Peace, lalu **sentuh cepat jempol+telunjuk** —
filter harus maju satu dengan transisi lembut, dan tekan `1` harus kembali ke
Blur (tampilan asli). Uji semua sembilan lewat `1`–`9`. Amati apakah gerakan
masih terasa lancar; kalau tersendat, turunkan `EFFECT_SCALE` di
`app/config.py`.

- [ ] **Step 8: Commit**

```bash
git add app/effects.py app/main.py app/scenes.py tests/test_effects.py
git commit -m "Tambah sembilan filter gambar, disambungkan ke pinch-tap

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: HUD, FPS, screenshot, fullscreen, vignette

**Files:**
- Create: `app/hud.py`
- Modify: `app/main.py`, `.gitignore` (sudah memuat `shots/`, verifikasi saja)

**Interfaces:**
- Consumes: `app.config` (warna, `HINT_SHOW_S`, `HINT_FADE_S`), `app.draw.make_vignette_layer`
- Produces:
  - `app.hud.Hud()` dengan `.draw(frame, mode, effect_idx, effect_name, label_from, label_p, fps, accent, elapsed)`
  - `app.hud.PIL_AVAILABLE: bool`

- [ ] **Step 1: Tulis `app/hud.py`**

Diadaptasi dari `.reference/portal_app.py:616-763`. Salin apa adanya:
`_FONT_CANDIDATES` (616–625), `_find_font_path` (628–632), `_font` (639–659),
`_text_width` (662–666), `blend_panel` (669–685), `build_fps_pill` (722–735).

`build_effect_panel` (688–719) disalin dengan satu perubahan: nomor efek
diformat `{effect_idx + 1:02d}` tetap, tapi lebar minimum panel dinaikkan dari
`300` ke `320` karena nama filter terpanjang kita (`"Posterize Neon"`) lebih
panjang daripada milik referensi.

`build_hint_panel` (738–751) diganti teksnya:

```python
    txt = ("pinch cepat: ganti efek  |  TAB: mode  |  1-9: efek  |  "
           "f: layar penuh  |  s: simpan  |  q: keluar")
```

Tambahkan badge mode — elemen baru:

```python
def build_mode_badge(mode, accent):
    """Badge kecil kiri-atas yang menandai mode aktif."""
    if not PIL_AVAILABLE:
        return None
    font = _font(12, bold=True)
    W = _text_width(mode, font) + 30
    H = 24
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((1, 1, W - 2, H - 2), radius=12,
                        fill=_pil_rgb(config.PANEL_FILL, 150),
                        outline=_pil_rgb(accent, 120), width=1)
    d.text((15, H // 2), mode, font=font, fill=_pil_rgb(accent, 240),
           anchor="lm")
    return np.array(img)
```

Bungkus semuanya dalam kelas `Hud`. Panel yang isinya jarang berubah di-cache,
supaya PIL tidak menggambar ulang tiap frame:

```python
class Hud:
    """Semua elemen HUD dalam satu tempat.

    Panel hint dan badge mode di-cache karena isinya jarang berubah; panel efek
    dibangun tiap frame karena membawa animasi slide.
    """

    def __init__(self):
        self._hint = build_hint_panel()
        self._fps_pill = None
        self._fps_shown = None
        self._badge = None
        self._badge_key = None

    def _mode_badge(self, mode, accent):
        key = (mode, accent)
        if key != self._badge_key:
            self._badge = build_mode_badge(mode, accent)
            self._badge_key = key
        return self._badge

    def draw(self, frame, mode, effect_idx, effect_name, label_from, label_p,
             fps, accent, elapsed):
        dw, dh = frame.shape[1], frame.shape[0]
        if not PIL_AVAILABLE:
            _hud_fallback(frame, mode, effect_name, fps)
            return

        badge = self._mode_badge(mode, accent)
        blend_panel(frame, badge, 16, 16)

        panel = build_effect_panel(effect_idx, effect_name, label_from,
                                   label_p, accent)
        blend_panel(frame, panel, 16, 48)

        # FPS dibulatkan dulu: pill hanya digambar ulang saat angkanya berubah
        shown = int(round(fps))
        if shown != self._fps_shown:
            self._fps_shown = shown
            self._fps_pill = build_fps_pill(shown)
        if self._fps_pill is not None:
            blend_panel(frame, self._fps_pill,
                        dw - self._fps_pill.shape[1] - 16, 16)

        hint_alpha = 1.0 - max(0.0, (elapsed - config.HINT_SHOW_S)
                               / config.HINT_FADE_S)
        if self._hint is not None and hint_alpha > 0.02:
            blend_panel(frame, self._hint, (dw - self._hint.shape[1]) // 2,
                        dh - self._hint.shape[0] - 14, alpha=hint_alpha)
```

Fallback dipakai kalau `PIL_AVAILABLE` bernilai `False` — adaptasi dari
`.reference/portal_app.py:754-763` dengan nama mode ditambahkan:

```python
def _hud_fallback(frame, mode, effect_name, fps):
    """HUD minimal OpenCV, hanya dipakai kalau Pillow tidak terpasang."""
    w = frame.shape[1]
    cv2.putText(frame, f"[{mode}]  {effect_name}", (16, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, config.ACCENT, 2, cv2.LINE_AA)
    cv2.putText(frame, f"{fps:.0f} FPS", (w - 118, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, config.TEXT_LO, 1, cv2.LINE_AA)
```

Tata letak: badge mode di `(16, 16)`, panel efek di `(16, 48)`, FPS pill di
kanan atas, hint di tengah bawah.

- [ ] **Step 2: Sambungkan di `app/main.py`**

Tambah state: `hud = Hud()`, `fps = 30.0`, akumulator FPS
(`.reference/portal_app.py:1022-1030`), `vignette_enabled = config.VIGNETTE`,
`vignette_layer = None`, `fullscreen = config.FULLSCREEN`, `shot_count = 0`,
`show_hint = True`. Impor `os`, `math`, `Hud`, dan
`app.draw.{make_vignette_layer, lerp_color}`.

Sebelum `cv2.imshow`, terapkan vignette lalu HUD:

```python
            if vignette_enabled:
                if (vignette_layer is None
                        or vignette_layer.shape[:2] != (h, w)):
                    vignette_layer = make_vignette_layer(w, h)
                frame = cv2.multiply(frame, vignette_layer, scale=1.0 / 255.0)

            label_p = 1.0
            if label_from is not None:
                label_p = min(1.0, (t_now - label_start) / config.LABEL_ANIM_S)
                if label_p >= 1.0:
                    label_from = None
            hud.draw(frame, "GESTUR", effect_idx, EFFECT_NAMES[effect_idx],
                     label_from, label_p, fps, accent,
                     elapsed=(t_now - start) if show_hint else 1e9)
```

Nama mode masih di-hardcode `"GESTUR"` di sini karena Mode Portal baru ada di
Task 9; task itu yang menggantinya dengan variabel `mode`.

`accent` adalah `lerp_color(config.ACCENT, config.ACCENT_VIOLET,
0.5 + 0.5 * math.sin((t_now - start) * 1.25))` — denyut warna lambat yang sama
dengan portal, sehingga HUD dan portal terasa satu sistem.

Tambah tombol:

```python
            elif key == ord("s"):
                os.makedirs(config.SHOTS_DIR, exist_ok=True)
                shot_count += 1
                path = os.path.join(config.SHOTS_DIR,
                                    f"foto-kita-blurrr-{shot_count:03d}.png")
                cv2.imwrite(path, frame)
                print(f"Tersimpan: {path}")
            elif key == ord("v"):
                vignette_enabled = not vignette_enabled
                print(f"Vignette: {'NYALA' if vignette_enabled else 'MATI'}")
            elif key == ord("h"):
                show_hint = not show_hint
            elif key == ord("f"):
                fullscreen = not fullscreen
                cv2.setWindowProperty(
                    WINDOW, cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
                print(f"Layar penuh: {'NYALA' if fullscreen else 'MATI'}")
```

Screenshot sengaja diambil **sesudah** HUD digambar, jadi hasil simpanan sama
persis dengan yang terlihat di layar.

- [ ] **Step 3: Jalankan test**

Run: `python -m pytest -q`
Expected: PASS, 98 test (task ini tidak menambah test — `hud.py` murni render;
kontraknya diverifikasi lewat pemeriksaan manual di Step 4)

- [ ] **Step 4: Pemeriksaan manual**

Run: `python foto_kita_blurrr.py`

Badge `GESTUR` dan panel nama efek muncul di kiri atas; FPS pill di kanan atas;
hint di bawah memudar sesudah 5 detik dan kembali dengan `h`. `f` masuk/keluar
layar penuh, `v` mematikan vignette, `s` menyimpan PNG ke `shots/` yang isinya
identik dengan layar. Uninstall Pillow sementara
(`pip uninstall -y Pillow && python foto_kita_blurrr.py`) untuk memastikan
fallback OpenCV jalan, lalu pasang lagi (`pip install Pillow`).

- [ ] **Step 5: Commit**

```bash
git add app/hud.py app/main.py
git commit -m "Tambah HUD ber-desain, FPS, screenshot, layar penuh, vignette

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: Mode Portal

**Files:**
- Create: `app/portal.py`, `tests/test_portal.py`
- Modify: `app/main.py`

**Interfaces:**
- Consumes: `app.config` (feather, glow, partikel), `app.draw.dim_color`
- Produces:
  - `app.portal.build_box(tip_a, tip_b, min_size=20) -> np.ndarray` — `(4, 2) float32`, urutan TL, TR, BR, BL
  - `app.portal.QuadSmoother(alpha=0.35).update(quad) -> np.ndarray`, `.reset()`
  - `app.portal.render_portal(frame, effect_fn, quad, prev_effect_fn=None, blend=1.0, alpha=1.0)` — mengubah `frame` di tempat
  - `app.portal.draw_glow(frame, quad, color, intensity)`
  - `app.portal.draw_corner_accents(frame, quad, color, t, alpha=1.0, length=16)`
  - `app.portal.ParticleField(max_particles=MAX_PARTICLES)` dengan `.spawn(quad, count)`, `.update(dt)`, `.draw(frame, color)`

- [ ] **Step 1: Tulis test yang gagal**

`tests/test_portal.py`:

```python
"""Geometri portal dan keamanan komposit. Yang paling mungkin salah di kode
seperti ini adalah menulis di luar batas frame saat portal separuh keluar
layar — jadi itu yang diuji."""

import numpy as np
import pytest

from app import effects, portal


def blank(h=120, w=160):
    return np.zeros((h, w, 3), np.uint8)


def test_build_box_orders_corners_tl_tr_br_bl():
    quad = portal.build_box((30, 90), (110, 20))
    assert quad.shape == (4, 2)
    assert quad.dtype == np.float32
    tl, tr, br, bl = quad
    assert tuple(tl) == (30.0, 20.0)
    assert tuple(tr) == (110.0, 20.0)
    assert tuple(br) == (110.0, 90.0)
    assert tuple(bl) == (30.0, 90.0)


def test_build_box_is_order_independent():
    a = portal.build_box((30, 90), (110, 20))
    b = portal.build_box((110, 20), (30, 90))
    assert np.array_equal(a, b)


def test_build_box_applies_a_minimum_size_floor():
    """Dua ujung telunjuk sejajar tidak boleh membuat kotak jadi seiris garis."""
    quad = portal.build_box((100, 60), (140, 60), min_size=20)
    height = quad[2][1] - quad[0][1]
    assert height >= 20


def test_quad_smoother_moves_toward_the_new_quad():
    s = portal.QuadSmoother(alpha=0.5)
    first = portal.build_box((0, 0), (100, 100))
    s.update(first)
    out = s.update(portal.build_box((0, 0), (200, 200)))
    assert 100 < out[2][0] < 200


def test_quad_smoother_reset_forgets_history():
    s = portal.QuadSmoother(alpha=0.5)
    s.update(portal.build_box((0, 0), (100, 100)))
    s.reset()
    target = portal.build_box((0, 0), (200, 200))
    assert np.array_equal(s.update(target), target)


def test_render_portal_changes_pixels_inside_the_box():
    img = blank()
    img[:] = 200
    quad = portal.build_box((40, 30), (120, 90))
    portal.render_portal(img, effects.fx_invert_glitch, quad)
    assert img[60, 80, 0] != 200


def test_render_portal_leaves_pixels_far_outside_untouched():
    img = blank()
    img[:] = 200
    portal.render_portal(img, effects.fx_invert_glitch,
                         portal.build_box((40, 30), (120, 90)))
    assert img[2, 2, 0] == 200


@pytest.mark.parametrize("a,b", [
    ((-80, -60), (40, 30)),      # menjulur keluar kiri-atas
    ((120, 90), (400, 400)),     # menjulur keluar kanan-bawah
    ((-500, -500), (-400, -400)),  # sepenuhnya di luar layar
])
def test_render_portal_never_writes_out_of_bounds(a, b):
    img = blank()
    portal.render_portal(img, effects.fx_thermal, portal.build_box(a, b))
    assert img.shape == (120, 160, 3)


@pytest.mark.parametrize("a,b", [((-80, -60), (40, 30)), ((120, 90), (400, 400))])
def test_glow_and_accents_survive_a_partly_offscreen_portal(a, b):
    img = blank()
    quad = portal.build_box(a, b)
    portal.draw_glow(img, quad, (255, 170, 64), 0.8)
    portal.draw_corner_accents(img, quad, (255, 170, 64), 1.0)


def test_particle_field_is_capped():
    f = portal.ParticleField(max_particles=10)
    quad = portal.build_box((10, 10), (100, 100))
    for _ in range(50):
        f.spawn(quad, 5)
    assert len(f.particles) <= 10


def test_particles_expire():
    f = portal.ParticleField()
    f.spawn(portal.build_box((10, 10), (100, 100)), 5)
    f.update(10.0)
    assert f.particles == []
```

- [ ] **Step 2: Jalankan test, pastikan gagal**

Run: `python -m pytest tests/test_portal.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'app.portal'`

- [ ] **Step 3: Tulis `app/portal.py`**

Diadaptasi dari `.reference/portal_app.py`. Salin dengan penyesuaian berikut:

- `QuadSmoother` (283–302) — apa adanya.
- `build_quad_from_index_fingers` (341–375) → **`build_box(tip_a, tip_b, min_size=20)`**.
  Pipeline kita sudah punya titik pixel yang dihaluskan, jadi fungsi ini menerima
  dua tuple `(x, y)` langsung, bukan landmark + `w`/`h`. Buang parameter
  `hand_a, hand_b, w, h`; ambil `x_min, x_max = sorted(...)` dari kedua tuple.
  Angka lantai `20` yang di-hardcode jadi parameter `min_size`.
- `render_portal_into_quad` (402–461) → **`render_portal`**, apa adanya.
  Fungsi ini sudah menjepit `x_min/x_max/y_min/y_max` ke ukuran frame dan
  keluar lebih awal kalau wilayahnya di bawah 5 px — itulah yang membuat test
  di luar batas lulus. Jangan hilangkan penjepitan itu.
- `draw_portal_glow` (464–505) → **`draw_glow`**, apa adanya.
- `draw_corner_accents` (508–528) — apa adanya, ganti `_dim` jadi
  `app.draw.dim_color`.
- `Particle` (534–545), `spawn_particles` (547–560), `update_particles`
  (562–574), `draw_particles` (577–581) → dibungkus jadi kelas
  **`ParticleField`** supaya state-nya tidak jadi variabel global:

```python
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
```

`FEATHER_PX`, `FEATHER_SIGMA`, `GLOW_PAD`, `GLOW_SIGMA` diambil dari
`app.config`, bukan didefinisikan ulang di modul ini.

- [ ] **Step 4: Jalankan test, pastikan lulus**

Run: `python -m pytest tests/test_portal.py -q`
Expected: PASS, 14 test

- [ ] **Step 5: Sambungkan Mode Portal di `app/main.py`**

Tambah impor: `random`, `INDEX_TIP` dari `app.gestures`, dan
`build_box, draw_corner_accents, draw_glow, ParticleField, QuadSmoother,
render_portal` dari `app.portal`.

Tambah konstanta mode dan state:

```python
MODE_GESTURE = "GESTUR"
MODE_PORTAL = "PORTAL"
```

```python
    mode = MODE_GESTURE
    quad_smoother = QuadSmoother()
    particles = ParticleField()
    portal_alpha = 0.0
    last_quad = None
    last_time = time.time()
```

Ganti blok render menjadi percabangan mode. Di Mode Gestur, semua kode yang
sudah ada dipertahankan. Di Mode Portal, lima gestur dilewati sepenuhnya —
inilah yang menghilangkan salah-deteksi:

```python
            dt = min(t_now - last_time, 0.1)
            last_time = t_now

            if mode == MODE_GESTURE:
                current = None
                if detect_two_hand_heart(hands_pts, w):
                    current = "HEART"
                elif detect_kicaw(hands_pts, w, h, mouth):
                    current = "KICAW"
                elif ev.holding:
                    current = "OK"
                else:
                    for pts in hands_pts:
                        g = classify_hand(pts)
                        if g is not None:
                            current = g
                            break
                active = debouncer.update(current)
                if current == "OK":
                    active = "OK"
                    debouncer.force("OK")
                for pts in hands_pts:
                    draw_hand_skeleton(frame, pts, hsv_color((t_now * 0.25) % 1.0))
                if active:
                    scenes.render(frame, active, t_now,
                                  effect_fn=EFFECTS[effect_idx],
                                  prev_effect_fn=prev_fn, blend=blend)
            else:
                active = None
                quad = None
                if len(hands_pts) == 2:
                    quad = quad_smoother.update(build_box(
                        hands_pts[0][INDEX_TIP], hands_pts[1][INDEX_TIP]))
                else:
                    quad_smoother.reset()

                # fade in/out eksponensial supaya portal muncul dan hilang halus
                target = 1.0 if quad is not None else 0.0
                portal_alpha += (target - portal_alpha) * (
                    1.0 - math.exp(-dt / config.PORTAL_FADE_S))
                if quad is not None:
                    last_quad = quad
                elif portal_alpha < 0.01:
                    last_quad = None

                if last_quad is not None and portal_alpha > 0.005:
                    q = last_quad
                    grow = 0.92 + 0.08 * min(1.0, portal_alpha)
                    c = q.mean(axis=0)
                    q = c + (q - c) * grow      # scale-in, mask saja
                    render_portal(frame, EFFECTS[effect_idx], q,
                                  prev_effect_fn=prev_fn, blend=blend,
                                  alpha=portal_alpha)
                    draw_glow(frame, q, accent,
                              (0.5 + 0.28 * math.sin(t_anim * 2.1)) * portal_alpha)
                    draw_corner_accents(frame, q, accent, t_anim,
                                        alpha=portal_alpha)
                    if portal_alpha > 0.4:
                        particles.spawn(q, 2 if random.random() < 0.7 else 1)
                particles.update(dt)
                particles.draw(frame, accent)
```

`prev_fn`, `blend`, dan `t_anim` dihitung sekali di atas percabangan — keduanya
dipakai oleh dua cabang, jadi jangan digandakan:

```python
            t_anim = t_now - start
            prev_fn = (EFFECTS[prev_effect_idx]
                       if prev_effect_idx is not None else None)
```

(`blend` sudah dihitung di Task 7; `accent` adalah warna denyut dari Task 8.)

Terakhir, ganti nama mode yang di-hardcode di panggilan `hud.draw` menjadi
variabel: `hud.draw(frame, mode, effect_idx, ...)`.

Saat berpindah mode, suara yang sedang berbunyi harus dihentikan dan state
portal dibersihkan:

```python
            elif key == 9:  # TAB
                mode = MODE_PORTAL if mode == MODE_GESTURE else MODE_GESTURE
                for k in SOUND_FOR.values():
                    audio.stop(k)
                prev_active = None
                quad_smoother.reset()
                portal_alpha = 0.0
                last_quad = None
                print(f"Mode: {mode}")
```

Blok suara yang sudah ada tetap berjalan apa adanya: di Mode Portal `active`
selalu `None`, jadi suara otomatis berhenti dan tidak pernah menyala.

- [ ] **Step 6: Jalankan seluruh test**

Run: `python -m pytest -q`
Expected: PASS, 112 test

- [ ] **Step 7: Pemeriksaan manual**

Run: `python foto_kita_blurrr.py`

Tekan `TAB`: badge berubah jadi `PORTAL`. Angkat dua tangan — portal terbentuk
di antara kedua ujung telunjuk, dengan tepi lembut, glow berdenyut cyan↔violet,
bracket sudut, dan partikel. Sentuh-cepat jempol+telunjuk mengganti filter di
dalam portal. Turunkan tangan: portal memudar, tidak menghilang mendadak.
Gerakkan portal sampai separuh keluar layar — tidak boleh crash. Tekan `TAB`
lagi: kembali ke Mode Gestur, kelima gestur berfungsi seperti semula, dan
tidak ada suara yang tertinggal berbunyi.

- [ ] **Step 8: Commit**

```bash
git add app/portal.py app/main.py tests/test_portal.py
git commit -m "Tambah Mode Portal dengan toggle TAB

Portal ter-feather di antara dua ujung telunjuk, rim glow berdenyut,
bracket sudut, dan partikel energi. Diadaptasi dari milan-kb/fancy-fingers.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 10: Lisensi, README, dan rapikan

**Files:**
- Create: `LICENSE`
- Rewrite: `README.md`
- Modify: `PRD.md`

**Interfaces:**
- Consumes: seluruh perilaku dari Task 1–9
- Produces: — (dokumentasi)

- [ ] **Step 1: Tulis `LICENSE`**

Salin isi `.reference/fancy-fingers-LICENSE`, lalu tambahkan baris copyright
milik project ini **di atas** milik referensi, sehingga kedua notice terbawa:

```
MIT License

Copyright (c) 2026 mrafi (Foto-Kita-Blurrr)
Copyright (c) 2024 milan-kb (HandPortal / fancy-fingers)

Sebagian kode — mode Portal, filter gambar, dan HUD — diadaptasi dari
https://github.com/milan-kb/fancy-fingers yang berlisensi MIT.

<sisa teks MIT standar, disalin apa adanya dari .reference/fancy-fingers-LICENSE>
```

- [ ] **Step 2: Tulis ulang `README.md`**

Struktur, mengikuti kualitas README referensi tapi dengan isi project ini:

1. Judul + satu kalimat penjelas + badge (Python 3.9+, MediaPipe, OpenCV, MIT).
2. **Apa ini** — dua paragraf: mainan kamera gestur, dua mode.
3. **Gestur** — tabel lima gestur beserta efeknya, disalin dari tabel §5.1 spec.
   Catat bahwa 👌 OK kini dipicu dengan **menahan** sentuhan jempol+telunjuk
   selama ~0,35 detik, dan sentuhan cepat justru mengganti filter.
4. **Filter** — tabel sembilan filter, nomor sejajar dengan tombol `1`–`9`.
5. **Mode Portal** — penjelasan singkat + cara masuk (`TAB`).
6. **Kontrol** — tabel lengkap dari §5.5 spec.
7. **Menjalankan** — venv, `pip install -r requirements.txt`,
   `python foto_kita_blurrr.py`. Tekankan bahwa model **terunduh otomatis**;
   hapus seluruh instruksi `curl` manual dari README lama karena sudah tidak
   berlaku.
8. **Struktur project** — pohon dari bagian Peta File di rencana ini.
9. **Konfigurasi** — tabel parameter penting dari `app/config.py`:
   `CAM_INDEX`, `FRAME_W`/`FRAME_H`, `DETECT_WIDTH`, `EFFECT_SCALE`,
   `PINCH_HOLD_S`, `PINCH_ENTER`/`PINCH_EXIT`, `HOLD_FRAMES`, `VIGNETTE`,
   `FULLSCREEN`.
10. **Test** — `python -m pytest -q`, dan catatan bahwa seluruh test berjalan
    tanpa webcam.
11. **Catatan** — audio butuh file WAV di `assets/Sond/` (perintah ffmpeg dari
    README lama tetap dipertahankan, masih berlaku); tanpa sound device program
    tetap jalan; model wajah opsional membuat Kicaw akurat.
12. **Lisensi & kredit** — MIT, dengan tautan ke `milan-kb/fancy-fingers` dan
    penyebutan bagian mana yang diadaptasi.

- [ ] **Step 3: Perbarui `PRD.md`**

Tambahkan di bagian bawah, tanpa mengubah teks aslinya (itu catatan niat awal):

```markdown
---

## Perkembangan

PRD di atas adalah ide awal. Project ini kemudian digabungkan dengan
[milan-kb/fancy-fingers](https://github.com/milan-kb/fancy-fingers) —
menambah sembilan filter gambar, Mode Portal, dan HUD.

Desain penggabungannya:
`docs/superpowers/specs/2026-08-09-foto-kita-blurrr-handportal-merge-design.md`
```

- [ ] **Step 4: Verifikasi akhir**

```bash
python -m pytest -q
python foto_kita_blurrr.py
```

Telusuri tabel Kontrol di README satu per satu dan pastikan setiap baris benar-
benar berfungsi. README yang berbohong lebih buruk daripada tidak ada README.

- [ ] **Step 5: Commit**

```bash
git add LICENSE README.md PRD.md
git commit -m "Tambah lisensi MIT dengan atribusi, tulis ulang README

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Kaitan dengan Spec

| Bagian spec | Task |
|-------------|------|
| §4 Arsitektur, batas antar-modul | 1–5 |
| §5.1 Mode Gestur | 5, 7 |
| §5.2 Mode Portal | 9 |
| §5.3 Pinch: satu mesin-status, dua hasil | 6 |
| §5.4 Sembilan filter + crossfade | 7 |
| §5.5 Kontrol | 7, 8, 9 |
| §5.6 HUD | 8 |
| §6 Penanganan kesalahan | 4 (model, audio), 5 (kamera), 8 (fallback PIL) |
| §7 Strategi test | 1, 2, 3, 4, 6, 7, 9 |
| §8 Kinerja | 7 (`apply_full_frame`), 1 (`EFFECT_SCALE`, `DETECT_WIDTH`) |
| §10 Lisensi dan atribusi | 10 |

