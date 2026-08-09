"""
Foto-Kita-Blurrr  ✨
====================
Aplikasi kamera real-time yang mendeteksi gestur tangan dan memberi efek lucu.

Gestur yang dikenali:
    ✌️  Peace        -> layar nge-blur + judul "FOTO KITA BLURRR" + sound
    🫶  Finger Heart -> tulisan + animasi  "I LOVE YOU"
    🤟  ILY / Rock   -> tulisan + animasi  "GOKILL"
    👌  OK           -> tulisan + animasi  "OKE"
    🐦  Kicaw        -> satu tangan tutup mulut + tangan lain menjulur ke depan
                        (jari lurus) -> memutar sound "Kicaw Mania"

Tekan  Q  atau  ESC  untuk keluar.

Cara jalanin:  python foto_kita_blurrr.py
"""

import math
import os
import random
import time

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
except ImportError:  # pragma: no cover
    raise SystemExit(
        "MediaPipe belum terpasang. Jalankan dulu:\n"
        "    pip install -r requirements.txt"
    )

# ----------------------------------------------------------------------------
# Lokasi asset
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GESTUR_DIR = os.path.join(BASE_DIR, "assets", "gestur")
SOUND_DIR = os.path.join(BASE_DIR, "assets", "Sond")

# File audio dipakai dalam format WAV (pygame tidak andal memutar mp3/m4a).
# WAV dibuat dari mp3 asli; lihat README jika perlu regenerasi.
SOUND_BLUR = os.path.join(SOUND_DIR, "foto kita blur.wav")
SOUND_KICAW = os.path.join(SOUND_DIR, "Kicaw Mania.wav")
IMG_BLUR = os.path.join(GESTUR_DIR, "Fotokitablurr.jpg")
MODEL_PATH = os.path.join(BASE_DIR, "models", "hand_landmarker.task")
# Model deteksi wajah (opsional) untuk membuat gestur Kicaw "tutup mulut" akurat.
FACE_MODEL_PATH = os.path.join(BASE_DIR, "models", "blaze_face_short_range.tflite")

# ----------------------------------------------------------------------------
# Audio (pygame mixer)
# ----------------------------------------------------------------------------
AUDIO_ENABLED = False
SOUNDS = {}
try:
    import pygame

    pygame.mixer.init()
    AUDIO_ENABLED = True
    for key, path in (("blur", SOUND_BLUR), ("kicaw", SOUND_KICAW)):
        if os.path.exists(path):
            try:
                SOUNDS[key] = pygame.mixer.Sound(path)
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] gagal load sound {path}: {exc}")
        else:
            print(f"[warn] sound tidak ditemukan: {path}")
except Exception as exc:  # noqa: BLE001
    print(f"[warn] audio dimatikan (mixer gagal init): {exc}")


_channels = {}  # channel pygame per-sound, untuk cegah tumpang-tindih


def play_sound(key):
    """Mainkan sound sekali. Tidak mengulang kalau masih berbunyi (anti-spam)."""
    if not AUDIO_ENABLED or key not in SOUNDS:
        return
    ch = _channels.get(key)
    if ch is not None and ch.get_busy():
        return  # masih main, jangan ditumpuk
    try:
        _channels[key] = SOUNDS[key].play()
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] gagal play {key}: {exc}")


def stop_sound(key):
    """Hentikan sound saat gestur dilepas, biar tidak terus berbunyi."""
    ch = _channels.get(key)
    if ch is not None and ch.get_busy():
        ch.stop()


# ----------------------------------------------------------------------------
# Gambar overlay
# ----------------------------------------------------------------------------
BLUR_THUMB = None
if os.path.exists(IMG_BLUR):
    _img = cv2.imread(IMG_BLUR)
    if _img is not None:
        BLUR_THUMB = cv2.resize(_img, (160, 90))

# ----------------------------------------------------------------------------
# MediaPipe Hands (Tasks API / HandLandmarker)
# ----------------------------------------------------------------------------
def make_landmarker(running_mode):
    """Buat HandLandmarker. running_mode: vision.RunningMode.VIDEO / IMAGE."""
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(
            f"Model tidak ditemukan: {MODEL_PATH}\n"
            "Download dengan:\n"
            "  curl -L -o models/hand_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task"
        )
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=running_mode,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return vision.HandLandmarker.create_from_options(options)


def make_face_detector(running_mode):
    """Buat FaceDetector (opsional). Return None jika model tidak ada."""
    if not os.path.exists(FACE_MODEL_PATH):
        print(
            "[info] model wajah tidak ada -> Kicaw pakai perkiraan posisi.\n"
            "       (opsional) download untuk deteksi mulut akurat:\n"
            "  curl -L -o models/blaze_face_short_range.tflite "
            "https://storage.googleapis.com/mediapipe-models/face_detector/"
            "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
        )
        return None
    options = vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
        running_mode=running_mode,
        min_detection_confidence=0.5,
    )
    return vision.FaceDetector.create_from_options(options)


def mouth_from_faces(face_result, w, h):
    """Ambil titik mulut + radius dari hasil FaceDetector -> (mx, my, r) / None.

    Keypoint BlazeFace: 0=mata kanan,1=mata kiri,2=hidung,3=mulut,
    4=telinga kanan,5=telinga kiri (ternormalisasi 0..1)."""
    dets = getattr(face_result, "detections", None) if face_result else None
    if not dets:
        return None
    det = max(dets, key=lambda d: d.bounding_box.width * d.bounding_box.height)
    kps = det.keypoints
    if len(kps) < 6:
        return None
    mx, my = kps[3].x * w, kps[3].y * h
    face_w = _dist((kps[4].x * w, kps[4].y * h), (kps[5].x * w, kps[5].y * h))
    return (mx, my, max(face_w * 1.1, 40.0))


# Landmark index
WRIST = 0
THUMB_TIP, THUMB_IP, THUMB_MCP = 4, 3, 2
INDEX_TIP, INDEX_PIP, INDEX_MCP = 8, 6, 5
MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP = 12, 10, 9
RING_TIP, RING_PIP = 16, 14
PINKY_TIP, PINKY_PIP, PINKY_MCP = 20, 18, 17

TIP_IDS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]

# Koneksi antar-landmark untuk menggambar "tulang" tangan (skeleton).
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # jempol
    (0, 5), (5, 6), (6, 7), (7, 8),        # telunjuk
    (5, 9), (9, 10), (10, 11), (11, 12),   # tengah
    (9, 13), (13, 14), (14, 15), (15, 16),  # manis
    (13, 17), (17, 18), (18, 19), (19, 20),  # kelingking
    (0, 17),                                # pangkal telapak
]


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def hand_points(landmark_list, w, h):
    """Ubah daftar landmark ternormalisasi (Tasks API) menjadi titik pixel."""
    return [(int(lm.x * w), int(lm.y * h)) for lm in landmark_list]


def palm_scale(pts):
    """Ukuran telapak (wrist -> middle MCP) sebagai skala referensi."""
    return max(_dist(pts[WRIST], pts[MIDDLE_MCP]), 1e-3)


def fingers_extended(pts):
    """
    Tentukan jari mana yang terbuka. Memakai jarak tip-ke-wrist vs pip-ke-wrist
    agar tidak bergantung orientasi tangan.
    Return list bool: [thumb, index, middle, ring, pinky]
    """
    wrist = pts[WRIST]
    extended = []

    # Thumb: bandingkan jarak tip & ip terhadap pangkal jari kelingking.
    ref = pts[PINKY_MCP]
    extended.append(_dist(pts[THUMB_TIP], ref) > _dist(pts[THUMB_IP], ref))

    # Empat jari lain: tip lebih jauh dari wrist dibanding pip => terbuka.
    for tip, pip in (
        (INDEX_TIP, INDEX_PIP),
        (MIDDLE_TIP, MIDDLE_PIP),
        (RING_TIP, RING_PIP),
        (PINKY_TIP, PINKY_PIP),
    ):
        extended.append(_dist(pts[tip], wrist) > _dist(pts[pip], wrist) * 1.05)
    return extended


def classify_hand(pts):
    """
    Klasifikasi gestur satu tangan -> nama gestur atau None.
    Aturan diperketat supaya tidak gampang salah deteksi (false positive).
    """
    thumb, index, middle, ring, pinky = fingers_extended(pts)
    scale = palm_scale(pts)
    d_thumb_index = _dist(pts[THUMB_TIP], pts[INDEX_TIP])

    # 👌 OK: ujung jempol & telunjuk menyatu rapat (lingkaran kecil),
    # SEMUA tiga jari lain (tengah, manis, kelingking) terbuka jelas.
    if d_thumb_index < 0.42 * scale and middle and ring and pinky:
        return "OK"

    # 🤟 ILY: jempol + telunjuk + kelingking terbuka; tengah & manis tertutup.
    # Jempol harus benar-benar melebar (jauh dari telunjuk).
    if (thumb and index and pinky and not middle and not ring
            and d_thumb_index > 0.6 * scale):
        return "ILY"

    # ✌️ Peace: telunjuk & tengah terbuka membentuk huruf V yang jelas;
    # manis & kelingking tertutup. Jempol tidak melebar (bukan ILY).
    if index and middle and not ring and not pinky:
        v_gap = _dist(pts[INDEX_TIP], pts[MIDDLE_TIP])
        if v_gap > 0.35 * scale:
            return "PEACE"

    return None


def detect_two_hand_heart(hands_pts, w):
    """🫶 Heart: dua tangan, ujung telunjuk hampir bersentuhan & ujung jempol
    berdekatan (membentuk hati). Diperketat agar dua tangan peace tidak ikut."""
    if len(hands_pts) != 2:
        return False
    a, b = hands_pts
    index_close = _dist(a[INDEX_TIP], b[INDEX_TIP]) < 0.12 * w
    thumb_close = _dist(a[THUMB_TIP], b[THUMB_TIP]) < 0.18 * w
    # ujung telunjuk (atas hati) lebih tinggi dari ujung jempol (bawah hati)
    index_top = (a[INDEX_TIP][1] + b[INDEX_TIP][1]) < (a[THUMB_TIP][1] + b[THUMB_TIP][1])
    return index_close and thumb_close and index_top


def detect_kicaw(hands_pts, w, h, mouth=None):
    """🐦 Kicaw: satu tangan menutup mulut, tangan satunya menjulur ke depan
    dengan jari-jari lurus terbuka.

    Jika `mouth` (mx, my, r) tersedia dari deteksi wajah, "tutup mulut" dicek
    akurat lewat jarak tangan ke titik mulut. Tanpa wajah, dipakai perkiraan
    posisi tangan di area atas-tengah frame.
    """
    if len(hands_pts) != 2:
        return False
    # uji kedua kemungkinan peran (tangan A=mulut/B=depan, lalu sebaliknya)
    for mhand, fwd in (hands_pts, hands_pts[::-1]):
        mcx = sum(p[0] for p in mhand) / len(mhand)
        mcy = sum(p[1] for p in mhand) / len(mhand)

        _, idx, mid, ring, pinky = fingers_extended(fwd)
        open_hand = (idx + mid + ring + pinky) >= 3  # jari lurus terbuka

        fcx = sum(p[0] for p in fwd) / len(fwd)
        fcy = sum(p[1] for p in fwd) / len(fwd)

        if mouth is not None:
            mx, my, r = mouth
            near_mouth = _dist((mcx, mcy), (mx, my)) < r
            fwd_clear = _dist((fcx, fcy), (mx, my)) > r  # tangan depan menjauh
            roles_ok = near_mouth and fwd_clear
        else:
            near_face = mcy < 0.5 * h and 0.20 * w < mcx < 0.80 * w
            roles_ok = near_face and fcy > mcy  # depan tak lebih tinggi dari mulut

        if roles_ok and open_hand:
            return True
    return False


class HandSmoother:
    """Haluskan koordinat landmark antar-frame (exponential moving average).

    Tangan dicocokkan ke frame sebelumnya berdasarkan posisi pergelangan
    terdekat, jadi smoothing tetap benar walau urutan tangan berubah.
    alpha besar = lebih responsif, alpha kecil = lebih halus.
    """

    def __init__(self, alpha=0.6, match_dist=140.0):
        self.alpha = alpha
        self.match_dist = match_dist
        self.prev = []  # list of list[(x,y) float]

    def update(self, hands):
        out = []
        used = [False] * len(self.prev)
        for pts in hands:
            wrist = pts[WRIST]
            best_d, bi = None, -1
            for i, pp in enumerate(self.prev):
                if used[i]:
                    continue
                d = _dist(wrist, pp[WRIST])
                if best_d is None or d < best_d:
                    best_d, bi = d, i
            if bi >= 0 and best_d < self.match_dist:
                used[bi] = True
                a = self.alpha
                sm = [(a * nx + (1 - a) * ox, a * ny + (1 - a) * oy)
                      for (nx, ny), (ox, oy) in zip(pts, self.prev[bi])]
            else:
                sm = [(float(x), float(y)) for x, y in pts]
            out.append(sm)
        self.prev = out
        return [[(int(round(x)), int(round(y))) for x, y in pts] for pts in out]


# ----------------------------------------------------------------------------
# Util gambar
# ----------------------------------------------------------------------------
def draw_text(img, text, center, scale, color, thickness=2, font=cv2.FONT_HERSHEY_DUPLEX):
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = int(center[0] - tw / 2)
    y = int(center[1] + th / 2)
    cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), thickness + 4, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def hsv_color(hue):
    """hue 0..1 -> warna BGR cerah."""
    c = np.uint8([[[int(hue * 179) % 180, 255, 255]]])
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


# ----------------------------------------------------------------------------
# Efek per-gestur
# ----------------------------------------------------------------------------
LABELS = {
    "PEACE": "FOTO KITA BLURRR",
    "HEART": "I LOVE YOU",
    "ILY": "GOKILL",
    "OK": "OKE",
    "KICAW": "KICAW MANIA!",
}

_hearts = []    # partikel hati untuk efek HEART
_confetti = []  # konfeti warna-warni untuk efek KICAW


def render_effect(frame, gesture, t):
    """Gambar efek besar di tengah layar sesuai gestur aktif."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    pulse = 1.0 + 0.12 * math.sin(t * 6.0)
    hue = (t * 0.25) % 1.0

    if gesture == "PEACE":
        # blur seluruh layar lalu tampilkan judul (teks putih bersih)
        k = 45
        frame[:] = cv2.GaussianBlur(frame, (k, k), 0)
        draw_text(frame, LABELS["PEACE"], (cx, cy), 1.8 * pulse, (255, 255, 255), 3)
        if BLUR_THUMB is not None:
            bh, bw = BLUR_THUMB.shape[:2]
            frame[h - bh - 15:h - 15, w - bw - 15:w - 15] = BLUR_THUMB

    elif gesture == "HEART":
        # partikel hati naik ke atas
        if random.random() < 0.4:
            _hearts.append([random.randint(60, w - 60), h - 40,
                            random.uniform(2.5, 5.5), random.randint(14, 30)])
        for hp in _hearts:
            hp[1] -= hp[2]
        _hearts[:] = [hp for hp in _hearts if hp[1] > -40]
        for x, y, _, s in _hearts:
            draw_heart(frame, int(x), int(y), s, (180, 105, 255))
        draw_text(frame, LABELS["HEART"], (cx, cy), 1.7 * pulse, (180, 105, 255), 3)
        draw_heart(frame, cx, cy + 70, 40 * pulse, (180, 105, 255))

    elif gesture == "ILY":
        draw_text(frame, LABELS["ILY"], (cx, cy), 2.0 * pulse, hsv_color(hue), 3)
        draw_text(frame, "\\m/  ROCK ON  \\m/", (cx, cy + 70), 0.9, (0, 255, 255), 2)

    elif gesture == "OK":
        draw_text(frame, LABELS["OK"], (cx, cy), 2.2 * pulse, (0, 230, 0), 3)
        r = int(60 * pulse)
        cv2.circle(frame, (cx, cy + 90), r, (0, 230, 0), 4, cv2.LINE_AA)

    elif gesture == "KICAW":
        # konfeti warna-warni jatuh dari atas
        if random.random() < 0.9:
            _confetti.append([
                random.randint(0, w), -10,
                random.uniform(3.0, 7.0), random.uniform(-1.5, 1.5),
                hsv_color(random.random()), random.randint(6, 13),
            ])
        for c in _confetti:
            c[0] += c[3]
            c[1] += c[2]
        _confetti[:] = [c for c in _confetti if c[1] < h + 10]
        for x, y, _, _, col, s in _confetti:
            cv2.rectangle(frame, (int(x), int(y)), (int(x + s), int(y + s)), col, -1)
        # not balok beterbangan mengelilingi tulisan
        for i in range(7):
            nx = int(cx + math.sin(t * 3.0 + i) * (160 + i * 22))
            ny = int(cy + math.cos(t * 2.0 + i * 1.3) * 70 - 30)
            draw_note(frame, nx, ny, hsv_color((t * 0.5 + i * 0.14) % 1.0), 1.3)
        draw_text(frame, LABELS["KICAW"], (cx, cy), 1.9 * pulse, (0, 220, 255), 4)
        draw_text(frame, "~ cuit cuit cuit ~", (cx, cy + 70), 1.1, hsv_color(hue), 2)


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise SystemExit("Kamera tidak bisa dibuka. Pastikan webcam tersambung.")

    win = "Foto-Kita-Blurrr"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # State machine debounce: gestur baru dianggap "aktif" hanya kalau
    # bertahan stabil selama HOLD_FRAMES frame berturut-turut. Ini bikin
    # deteksi jauh lebih akurat (anti-kedip) dan sound tidak spam.
    HOLD_FRAMES = 6
    SOUND_FOR = {"PEACE": "blur", "KICAW": "kicaw"}
    candidate = None      # kandidat gestur yang sedang dihitung
    candidate_count = 0
    active = None         # gestur yang benar-benar aktif (sudah stabil)
    prev_active = None    # untuk deteksi rising/falling-edge (sound)
    start = time.time()
    last_ts = -1          # timestamp ms terakhir (harus selalu naik)
    smoother = HandSmoother()  # haluskan landmark antar-frame (anti-getar)
    DETECT_WIDTH = 640    # deteksi di frame diperkecil -> lebih responsif

    # FaceDetector opsional: bikin "tutup mulut" akurat. None jika model absen.
    face_detector = make_face_detector(vision.RunningMode.VIDEO)

    with make_landmarker(vision.RunningMode.VIDEO) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mode selfie / cermin
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Deteksi di citra diperkecil (landmark ternormalisasi -> tetap
            # presisi saat dipetakan balik ke resolusi penuh).
            if w > DETECT_WIDTH:
                det_img = cv2.resize(rgb, (DETECT_WIDTH, int(h * DETECT_WIDTH / w)))
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

            hands_raw = [hand_points(lm, w, h) for lm in result.hand_landmarks]
            hands_pts = smoother.update(hands_raw)
            per_hand_gesture = [classify_hand(p) for p in hands_pts]

            # Tentukan gestur frame ini (pose dua tangan diprioritaskan).
            current = None
            if detect_two_hand_heart(hands_pts, w):
                current = "HEART"
            elif detect_kicaw(hands_pts, w, h, mouth):
                current = "KICAW"
            else:
                for g in per_hand_gesture:
                    if g is not None:
                        current = g
                        break

            # Debounce: gestur harus stabil HOLD_FRAMES frame baru jadi 'active'.
            if current == candidate:
                candidate_count += 1
            else:
                candidate = current
                candidate_count = 1
            if candidate_count >= HOLD_FRAMES:
                active = candidate  # bisa juga None (gestur dilepas)

            # Gambar tulang (skeleton) di tiap tangan.
            t = time.time()
            for pts in hands_pts:
                draw_hand_skeleton(frame, pts, hsv_color((t * 0.25) % 1.0))

            # Efek visual untuk gestur aktif.
            if active:
                render_effect(frame, active, t)

            # Sound: mulai saat gestur masuk, hentikan saat gestur dilepas/ganti.
            if active != prev_active:
                old_key = SOUND_FOR.get(prev_active)
                if old_key:
                    stop_sound(old_key)
                new_key = SOUND_FOR.get(active)
                if new_key:
                    play_sound(new_key)
            prev_active = active

            cv2.imshow(win, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q / ESC
                break

    cap.release()
    cv2.destroyAllWindows()
    if face_detector is not None:
        face_detector.close()
    if AUDIO_ENABLED:
        pygame.mixer.quit()


if __name__ == "__main__":
    main()
