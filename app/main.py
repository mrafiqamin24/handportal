"""Loop kamera, mesin mode, dan penanganan keyboard.

Satu-satunya modul yang menyentuh kamera, jam, dan keyboard. Semua logika yang
bisa di-test tanpa perangkat keras ada di modul lain.
"""

import math
import os
import random
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision

from app import config
from app.audio import AudioPlayer
from app.draw import (draw_hand_skeleton, hsv_color, lerp_color,
                      make_vignette_layer)
from app.effects import EFFECT_NAMES, EFFECTS
from app.gestures import (INDEX_TIP, GestureDebouncer, HandSmoother,
                          PinchTapDetector, classify_hand, detect_kicaw,
                          detect_two_hand_heart, hand_points, pinch_distance)
from app.hud import Hud
from app.models import make_face_detector, make_landmarker, mouth_from_faces
from app.portal import (ParticleField, QuadSmoother, build_box,
                        draw_corner_accents, draw_glow, render_portal)
from app.scenes import GestureScenes

WINDOW = "Foto-Kita-Blurrr"
SOUND_FOR = {"PEACE": "blur", "KICAW": "kicaw"}

MODE_GESTURE = "GESTUR"
MODE_PORTAL = "PORTAL"


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
    if config.FULLSCREEN:
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)

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

    mode = MODE_GESTURE
    quad_smoother = QuadSmoother()
    particles = ParticleField()
    portal_alpha = 0.0
    last_quad = None

    hud = Hud()
    fps = 30.0
    fps_accum = 0.0
    fps_frames = 0
    vignette_enabled = config.VIGNETTE
    vignette_layer = None
    fullscreen = config.FULLSCREEN
    shot_count = 0
    show_hint = True
    hint_start = 0.0  # diisi `start` di bawah; ditekan ulang oleh tombol `h`

    prev_active = None
    start = time.time()
    hint_start = start
    t_prev = start
    last_ts = -1  # timestamp ms terakhir (harus selalu naik)

    with make_landmarker(vision.RunningMode.VIDEO) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mode selfie / cermin
            h, w = frame.shape[:2]
            t_now = time.time()
            # dijepit: jeda panjang (window di-drag, laptop bangun tidur)
            # tidak boleh melompatkan animasi
            dt = min(max(1e-6, t_now - t_prev), 0.1)
            t_prev = t_now

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

            # Dipakai kedua mode, jadi dihitung sekali di sini.
            t_anim = t_now - start
            prev_fn = (EFFECTS[prev_effect_idx]
                       if prev_effect_idx is not None else None)
            # Denyut warna lambat yang sama dipakai HUD dan portal, supaya
            # keduanya terasa satu sistem.
            accent = lerp_color(config.ACCENT, config.ACCENT_VIOLET,
                                0.5 + 0.5 * math.sin(t_anim * 1.25))

            if mode == MODE_GESTURE:
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

                # 👌 OK sudah dijaga ambang 0,35 dtk; menumpuk debounce di
                # atasnya membuatnya lamban (~0,55 dtk), jadi OK dilewatkan.
                if current == "OK":
                    active = "OK"
                    debouncer.force("OK")

                for pts in hands_pts:
                    draw_hand_skeleton(frame, pts, hsv_color(t_now * 0.25))
                if active:
                    scenes.render(frame, active, t_now,
                                  effect_fn=EFFECTS[effect_idx],
                                  prev_effect_fn=prev_fn, blend=blend)
            else:
                # Mode Portal: kelima gestur dilewati sepenuhnya — inilah yang
                # menghilangkan salah-deteksi antara portal dan gestur.
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
                              (0.5 + 0.28 * math.sin(t_anim * 2.1))
                              * portal_alpha)
                    draw_corner_accents(frame, q, accent, t_anim,
                                        alpha=portal_alpha)
                    if portal_alpha > 0.4:
                        particles.spawn(q, 2 if random.random() < 0.7 else 1)
                particles.update(dt)
                particles.draw(frame, accent)

            # Suara: mulai saat gestur masuk, berhenti saat dilepas/ganti.
            if active != prev_active:
                old_key = SOUND_FOR.get(prev_active)
                if old_key:
                    audio.stop(old_key)
                new_key = SOUND_FOR.get(active)
                if new_key:
                    audio.play(new_key)
            prev_active = active

            if vignette_enabled:
                if (vignette_layer is None
                        or vignette_layer.shape[:2] != (h, w)):
                    vignette_layer = make_vignette_layer(w, h)
                frame = cv2.multiply(frame, vignette_layer, scale=1.0 / 255.0)

            fps_accum += dt
            fps_frames += 1
            if fps_accum >= 0.25:
                fps = fps_frames / fps_accum
                fps_frames = 0
                fps_accum = 0.0

            label_p = 1.0
            if label_from is not None:
                label_p = min(1.0, (t_now - label_start) / config.LABEL_ANIM_S)
                if label_p >= 1.0:
                    label_from = None
            hud.draw(frame, mode, effect_idx, EFFECT_NAMES[effect_idx],
                     label_from, label_p, fps, accent,
                     elapsed=(t_now - hint_start) if show_hint else 1e9)

            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q / ESC
                break
            elif key == 9:  # TAB
                mode = MODE_PORTAL if mode == MODE_GESTURE else MODE_GESTURE
                for k in SOUND_FOR.values():
                    audio.stop(k)
                prev_active = None
                quad_smoother.reset()
                portal_alpha = 0.0
                last_quad = None
                print(f"Mode: {mode}")
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
                # timer di-reset supaya hint benar-benar muncul lagi
                show_hint = not show_hint
                hint_start = t_now
            elif key == ord("f"):
                fullscreen = not fullscreen
                cv2.setWindowProperty(
                    WINDOW, cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
                print(f"Layar penuh: {'NYALA' if fullscreen else 'MATI'}")

    cap.release()
    cv2.destroyAllWindows()
    if face_detector is not None:
        face_detector.close()
    audio.close()
