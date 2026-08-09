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
    last_ts = -1  # timestamp ms terakhir (harus selalu naik)

    with make_landmarker(vision.RunningMode.VIDEO) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mode selfie / cermin
            h, w = frame.shape[:2]
            t_now = time.time()

            # Deteksi di citra diperkecil (landmark ternormalisasi -> tetap
            # presisi saat dipetakan balik ke resolusi penuh).
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if w > config.DETECT_WIDTH:
                det_img = cv2.resize(
                    rgb, (config.DETECT_WIDTH,
                          int(h * config.DETECT_WIDTH / w)))
            else:
                det_img = rgb
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=det_img)
            ts = max(last_ts + 1, int((t_now - start) * 1000))
            last_ts = ts
            result = landmarker.detect_for_video(mp_image, ts)

            mouth = None
            if face_detector is not None:
                face_res = face_detector.detect_for_video(mp_image, ts)
                mouth = mouth_from_faces(face_res, w, h)

            hands_pts = smoother.update(
                [hand_points(lm, w, h) for lm in result.hand_landmarks])

            # Tentukan gestur frame ini (pose dua tangan diprioritaskan).
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

            for pts in hands_pts:
                draw_hand_skeleton(frame, pts, hsv_color(t_now * 0.25))
            if active:
                scenes.render(frame, active, t_now)

            # Suara: mulai saat gestur masuk, berhenti saat dilepas/ganti.
            if active != prev_active:
                old_key = SOUND_FOR.get(prev_active)
                if old_key:
                    audio.stop(old_key)
                new_key = SOUND_FOR.get(active)
                if new_key:
                    audio.play(new_key)
            prev_active = active

            cv2.imshow(WINDOW, frame)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):  # q / ESC
                break

    cap.release()
    cv2.destroyAllWindows()
    if face_detector is not None:
        face_detector.close()
    audio.close()
