"""Loop kamera, mesin mode, dan penanganan keyboard.

Satu-satunya modul yang menyentuh kamera, jam, dan keyboard. Semua logika yang
bisa di-test tanpa perangkat keras ada di modul lain.
"""

import collections
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
from app.effects import EFFECT_NAMES, EFFECTS, FilterTransition
from app.gestures import (DoublePinchDetector, GestureDebouncer, HandSmoother,
                          PinchTapDetector, classify_hand, detect_kicaw,
                          detect_two_hand_heart, hand_points,
                          handedness_labels, is_ok_pose, pinch_distance)
from app.hud import Hud
from app.models import make_face_detector, make_landmarker, mouth_from_faces
from app.portal import (ParticleField, PortalPoseDetector, QuadSmoother,
                        draw_corner_accents, draw_glow, portal_pose_report,
                        render_portal, update_portal_alpha)
from app.scenes import GestureScenes

WINDOW = "Foto-Kita-Blurrr"
SOUND_FOR = {"PEACE": "blur", "KICAW": "kicaw"}

MODE_GESTURE = "GESTUR"
MODE_PORTAL = "PORTAL"


def _make_capture(index):
    """Buat VideoCapture dengan backend native yang sesuai platform."""
    if os.name == "nt":
        return cv2.VideoCapture(index, cv2.CAP_DSHOW)
    return cv2.VideoCapture(index)


def open_camera(index=None, required=True, capture_factory=None):
    """Buka satu webcam. Return None saat probe opsional gagal."""
    index = config.CAM_INDEX if index is None else int(index)
    factory = capture_factory or _make_capture
    cap = factory(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        cap.release()
        if not required:
            return None
        raise SystemExit(
            "Kamera tidak bisa dibuka. Pastikan webcam tersambung dan tidak "
            f"dipakai aplikasi lain, atau ubah CAM_INDEX (sekarang "
            f"{index}) di app/config.py."
        )
    return cap


def switch_camera(cap, current_index, open_fn=open_camera, scan_max=None):
    """Cari kamera berikutnya tanpa mematikan kamera aktif lebih dahulu.

    Kamera lama baru dilepas setelah kandidat berhasil dibuka. Jika semua
    indeks gagal, aplikasi tetap memakai kamera lama dan tidak blank.
    """
    scan_max = config.CAMERA_SCAN_MAX if scan_max is None else int(scan_max)
    for step in range(1, scan_max + 2):
        index = (current_index + step) % (scan_max + 1)
        if index == current_index:
            continue
        candidate = open_fn(index, required=False)
        if candidate is not None:
            cap.release()
            return candidate, index, True
    return cap, current_index, False


def main():
    cap = open_camera()
    camera_index = config.CAM_INDEX
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    if config.FULLSCREEN:
        cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)

    audio = AudioPlayer({"blur": config.SOUND_BLUR, "kicaw": config.SOUND_KICAW})
    scenes = GestureScenes()
    smoother = HandSmoother()
    debouncer = GestureDebouncer()
    pinch = PinchTapDetector()
    double_pinch = DoublePinchDetector()
    face_detector = make_face_detector(vision.RunningMode.VIDEO)

    effect_idx = 0
    effect_transition = FilterTransition(effect_idx)
    label_from = None          # nama filter yang sedang slide keluar
    label_start = 0.0

    def switch_effect(new_idx, now):
        """Pindah filter dengan crossfade + animasi label."""
        nonlocal effect_idx
        nonlocal label_from, label_start
        source_idx = effect_transition.switch(new_idx, now)
        if source_idx is None:
            return
        label_from = EFFECT_NAMES[source_idx]
        label_start = now
        effect_idx = effect_transition.current_idx

    mode = MODE_GESTURE
    quad_smoother = QuadSmoother()
    portal_pose = PortalPoseDetector()
    particles = ParticleField()
    portal_alpha = 0.0
    last_quad = None
    cached_mouth = None
    last_face_detect_at = float("-inf")
    last_mouth_seen_at = float("-inf")

    hud = Hud()
    pending_actions = collections.deque()
    frame_width = config.FRAME_W

    def on_mouse(event, x, y, _flags, _userdata):
        action = hud.pointer(x, y, frame_width)
        if event == cv2.EVENT_LBUTTONUP and action is not None:
            pending_actions.append(action)

    cv2.setMouseCallback(WINDOW, on_mouse)
    fps = 30.0
    fps_accum = 0.0
    fps_frames = 0
    vignette_enabled = config.VIGNETTE
    vignette_layer = None
    fullscreen = config.FULLSCREEN
    shot_count = 0
    show_hint = True
    hint_start = 0.0  # diisi `start` di bawah; ditekan ulang oleh tombol `h`
    diagnose = False   # tombol `d`: sebutkan syarat portal mana yang gagal
    diag_text = None
    diag_at = 0.0

    prev_active = None
    start = time.time()
    hint_start = start
    t_prev = start
    last_ts = -1  # timestamp ms terakhir (harus selalu naik)

    def change_mode(new_mode, now):
        nonlocal mode, prev_active, portal_alpha, last_quad
        nonlocal cached_mouth, last_face_detect_at, last_mouth_seen_at
        if new_mode == mode:
            return
        mode = new_mode
        for sound_key in SOUND_FOR.values():
            audio.stop(sound_key)
        prev_active = None
        debouncer.force(None)
        pinch.reset()
        double_pinch.reset()
        portal_pose.reset()
        quad_smoother.reset()
        portal_alpha = 0.0
        last_quad = None
        particles.particles.clear()
        cached_mouth = None
        last_face_detect_at = float("-inf")
        last_mouth_seen_at = float("-inf")
        hud.notify(f"Mode {mode.title()} aktif", now)
        print(f"Mode: {mode}")

    def change_camera(now):
        nonlocal cap, camera_index
        nonlocal cached_mouth, last_face_detect_at, last_mouth_seen_at
        cap, next_index, changed = switch_camera(cap, camera_index)
        if changed:
            camera_index = next_index
            smoother.reset()
            pinch.reset()
            double_pinch.reset()
            portal_pose.reset()
            quad_smoother.reset()
            cached_mouth = None
            last_face_detect_at = float("-inf")
            last_mouth_seen_at = float("-inf")
            hud.notify(f"Kamera {camera_index + 1} aktif", now)
            print(f"Kamera: {camera_index + 1} (index {camera_index})")
        else:
            hud.notify("Tidak ada kamera lain", now, error=True)
            print("Kamera lain tidak ditemukan; kamera aktif dipertahankan.")

    with make_landmarker(vision.RunningMode.VIDEO) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mode selfie / cermin
            h, w = frame.shape[:2]
            frame_width = w
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
            face_due = (t_now - last_face_detect_at
                        >= config.FACE_DETECT_INTERVAL_S)
            if (mode == MODE_GESTURE and face_detector is not None
                    and face_due):
                face_res = face_detector.detect_for_video(mp_image, ts)
                last_face_detect_at = t_now
                detected_mouth = mouth_from_faces(face_res, w, h)
                if detected_mouth is not None:
                    cached_mouth = detected_mouth
                    last_mouth_seen_at = t_now
            if (mode == MODE_GESTURE and cached_mouth is not None
                    and t_now - last_mouth_seen_at
                    <= config.FACE_MOUTH_CACHE_S):
                mouth = cached_mouth

            raw_hands = [hand_points(lm, w, h)
                         for lm in result.hand_landmarks]
            raw_labels = handedness_labels(result)
            hands_pts = smoother.update(raw_hands, raw_labels)
            # Trek bridge hanya untuk menjaga gambar skeleton tidak berkedip.
            # Klasifikasi gestur selalu memakai observasi frame terbaru.
            fresh_hands = [pts for pts, missing in
                           zip(hands_pts, smoother.missing_counts)
                           if missing == 0]

            prev_effect_idx, blend = effect_transition.state(t_now)

            # Dipakai kedua mode, jadi dihitung sekali di sini.
            t_anim = t_now - start
            prev_fn = (EFFECTS[prev_effect_idx]
                       if prev_effect_idx is not None else None)
            # Denyut warna lambat yang sama dipakai HUD dan portal, supaya
            # keduanya terasa satu sistem.
            accent = lerp_color(config.ACCENT, config.ACCENT_VIOLET,
                                0.5 + 0.5 * math.sin(t_anim * 1.25))

            if mode == MODE_GESTURE:
                # Pinch satu tangan hanya untuk 👌 OK. Efek portal tidak pernah
                # dipakai/diganti di sini; Peace punya blur latarnya sendiri.
                gesture_dists = [pinch_distance(p) for p in fresh_hands
                                 if is_ok_pose(p, pinch.exit)]
                ev = pinch.update(gesture_dists, t_now)
                double_pinch.reset()
                # Tentukan gestur frame ini (pose dua tangan diprioritaskan).
                current = None
                if detect_two_hand_heart(fresh_hands, w):
                    current = "HEART"
                elif detect_kicaw(fresh_hands, w, h, mouth):
                    current = "KICAW"
                elif ev.holding:
                    current = "OK"
                else:
                    for pts in fresh_hands:
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

                # Revisi pengguna: hanya ✌️ Peace yang mengaburkan kamera.
                # Blur dilakukan sebelum skeleton/text agar overlay tetap tajam.
                frame = scenes.apply_background(frame, active)
                for pts in hands_pts:
                    draw_hand_skeleton(frame, pts, hsv_color(t_now * 0.25))
                if active:
                    scenes.render(frame, active, t_now)
            else:
                # Mode Portal: kelima gestur dilewati sepenuhnya — inilah yang
                # menghilangkan salah-deteksi antara portal dan gestur.
                active = None
                # Double-pinch = thumb+index kedua tangan bertemu bersamaan.
                # Hanya gesture empat-jari ini yang mengganti filter.
                # Portal langsung memakai landmark MediaPipe terbaru. Jalur
                # lama menghaluskan tangan lalu quad sekali lagi (double
                # smoothing), penyebab utama portal terasa tertinggal.
                fresh_dists = [pinch_distance(p) for p in raw_hands]
                double_ev = double_pinch.update(fresh_dists)
                if double_ev.triggered:
                    switch_effect((effect_idx + 1) % len(EFFECTS), t_now)
                    hud.notify(f"Filter: {EFFECT_NAMES[effect_idx]}", t_now)

                if diagnose:
                    # Diulang hanya saat alasannya berubah (atau tiap detik),
                    # supaya konsol tidak tenggelam 30 baris per detik.
                    reason = portal_pose_report(raw_hands)
                    text = reason or "pose diterima"
                    if text != diag_text or t_now - diag_at > 1.0:
                        diag_text, diag_at = text, t_now
                        hud.notify(f"Portal: {text}", t_now,
                                   error=reason is not None)
                        print(f"[portal] {text}")

                raw_quad = portal_pose.update(
                    raw_hands, raw_labels, now=t_now,
                    hold=double_ev.pinching)
                quad = (quad_smoother.update(raw_quad, t_now)
                        if raw_quad is not None else None)
                if raw_quad is None and not portal_pose.active:
                    quad_smoother.reset()

                portal_alpha = update_portal_alpha(
                    portal_alpha, quad is not None, dt)
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

            if vignette_enabled and mode == MODE_PORTAL:
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
                     elapsed=(t_now - hint_start) if show_hint else 1e9,
                     camera_index=camera_index, now=t_now)

            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            while pending_actions:
                action = pending_actions.popleft()
                if action.startswith("mode:"):
                    change_mode(action.split(":", 1)[1], t_now)
                elif action == "camera:next":
                    change_camera(t_now)
            if key in (ord("q"), 27):  # q / ESC
                break
            elif key == 9:  # TAB
                change_mode(
                    MODE_PORTAL if mode == MODE_GESTURE else MODE_GESTURE,
                    t_now)
            elif key == ord("c"):
                change_camera(t_now)
            elif ord("1") <= key <= ord("9"):
                n = key - ord("1")
                if mode != MODE_PORTAL:
                    hud.notify("Filter hanya di Mode Portal", t_now, error=True)
                elif n < len(EFFECTS):
                    switch_effect(n, t_now)
                    hud.notify(f"Filter: {EFFECT_NAMES[effect_idx]}", t_now)
                    print(f"Efek: {EFFECT_NAMES[effect_idx]}")
            elif key in (ord("["), ord("]")):
                # Tombol angka hanya menjangkau sembilan filter pertama; ini
                # yang membuat keempat filter terakhir bisa diraih dari papan
                # ketik, bukan cuma lewat double-pinch berulang.
                if mode != MODE_PORTAL:
                    hud.notify("Filter hanya di Mode Portal", t_now, error=True)
                else:
                    step = 1 if key == ord("]") else -1
                    switch_effect((effect_idx + step) % len(EFFECTS), t_now)
                    hud.notify(f"Filter: {EFFECT_NAMES[effect_idx]}", t_now)
                    print(f"Efek: {EFFECT_NAMES[effect_idx]}")
            elif key == ord("m"):
                print(f"Audio: {'BISU' if audio.toggle_mute() else 'NYALA'}")
            elif key in (ord("+"), ord("=")):
                pinch.adjust(config.PINCH_STEP)
                double_pinch.adjust(config.PINCH_STEP)
                print(f"Ambang pinch: {pinch.enter:.3f}")
            elif key == ord("-"):
                pinch.adjust(-config.PINCH_STEP)
                double_pinch.adjust(-config.PINCH_STEP)
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
            elif key == ord("d"):
                diagnose = not diagnose
                diag_text = None
                if diagnose and mode != MODE_PORTAL:
                    hud.notify("Diagnosa portal: pindah ke Mode Portal", t_now,
                               error=True)
                else:
                    hud.notify(
                        f"Diagnosa portal: {'NYALA' if diagnose else 'MATI'}",
                        t_now)
                print(f"Diagnosa portal: {'NYALA' if diagnose else 'MATI'}")
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
