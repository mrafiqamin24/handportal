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
from app.effects import EFFECT_NAMES, EFFECTS
from app.gestures import (GestureDebouncer, HandSmoother, PinchTapDetector,
                          classify_hand, detect_kicaw, detect_two_hand_heart,
                          hand_points, pinch_distance)
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
    pinch = PinchTapDetector()
    face_detector = make_face_detector(vision.RunningMode.VIDEO)

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

            # Pinch: TAP mengganti efek, HOLD memunculkan gestur 👌 OK.
            ev = pinch.update([pinch_distance(p) for p in hands_pts], t_now)
            if ev.tap:
                switch_effect((effect_idx + 1) % len(EFFECTS), t_now)

            if prev_effect_idx is not None:
                blend = min(1.0, (t_now - crossfade_start) / config.CROSSFADE_S)
                if blend >= 1.0:
                    prev_effect_idx = None
            else:
                blend = 1.0

            # Tentukan gestur frame ini (pose dua tangan diprioritaskan).
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

            # 👌 OK sudah dijaga ambang 0,35 dtk; menumpuk debounce di atasnya
            # membuatnya terasa lamban (~0,55 dtk), jadi OK dilewatkan langsung.
            if current == "OK":
                active = "OK"
                debouncer.force("OK")

            for pts in hands_pts:
                draw_hand_skeleton(frame, pts, hsv_color(t_now * 0.25))
            if active:
                scenes.render(
                    frame, active, t_now,
                    effect_fn=EFFECTS[effect_idx],
                    prev_effect_fn=(EFFECTS[prev_effect_idx]
                                    if prev_effect_idx is not None else None),
                    blend=blend)

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
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q / ESC
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

    cap.release()
    cv2.destroyAllWindows()
    if face_detector is not None:
        face_detector.close()
    audio.close()
