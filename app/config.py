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
DETECT_WIDTH = 768         # detail fingertip lebih baik, masih lebih ringan dari 1280
EFFECT_SCALE = 0.5         # filter layar-penuh dihitung di separuh resolusi

# MediaPipe Tasks. Nilai sedikit di atas bawaan menahan false positive tanpa
# membuat tangan yang sedang bergerak cepat terlalu mudah hilang.
HAND_DETECTION_CONFIDENCE = 0.58
HAND_PRESENCE_CONFIDENCE = 0.52
HAND_TRACKING_CONFIDENCE = 0.52
HANDEDNESS_MIN_CONFIDENCE = 0.65

# ---------------------------------------------------------------- gestur
HOLD_FRAMES = 5            # frame stabil sebelum sebuah gestur dianggap aktif
SMOOTH_ALPHA = 0.6         # kompatibilitas untuk pemanggil yang memilih EMA tetap
SMOOTH_ALPHA_MIN = 0.30    # tangan hampir diam: redam jitter lebih kuat
SMOOTH_ALPHA_MAX = 0.78    # gerakan cepat: kurangi rasa tertinggal
SMOOTH_MOTION_LOW = 0.03   # gerak wrist / ukuran telapak
SMOOTH_MOTION_HIGH = 0.35
SMOOTH_MATCH_DIST = 140.0  # jarak maksimum pencocokan tangan antar-frame
HAND_MISSING_GRACE = 2     # tahan landmark 2 frame saat detector sesaat dropout

# ---------------------------------------------------------------- pinch
# Jarak ternormalisasi ujung jempol <-> ujung telunjuk. Dua ambang berbeda
# (histeresis): masuk lebih ketat daripada keluar, supaya jitter landmark di
# sekitar ambang tidak memicu tap beruntun.
PINCH_ENTER = 0.055
PINCH_EXIT = 0.075
PINCH_HOLD_S = 0.35        # <0,35 dtk = TAP (ganti efek); >=0,35 dtk = HOLD (OK)
PINCH_STEP = 0.005         # langkah penyetelan sensitivitas lewat tombol +/-
PINCH_MIN, PINCH_MAX = 0.015, 0.20

# ---------------------------------------------------------------- transisi
CROSSFADE_S = 0.22         # lama crossfade antar-filter
LABEL_ANIM_S = 0.30        # lama animasi slide label nama efek
PORTAL_FADE_S = 0.30       # tetapan waktu fade in/out portal

# ---------------------------------------------------------------- HUD
HINT_SHOW_S = 5.0          # hint kontrol tampil selama ini...
HINT_FADE_S = 1.5          # ...lalu memudar selama ini

# ---------------------------------------------------------------- portal
FEATHER_PX = 8             # lebar pita blending di tepi portal
FEATHER_SIGMA = 2.0
GLOW_PAD = 26              # seberapa jauh glow melebar dari tepi kotak
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
