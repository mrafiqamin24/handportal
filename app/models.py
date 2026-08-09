"""Pemuatan model MediaPipe, termasuk unduhan otomatis sekali jalan."""

import os
import urllib.request

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from app import config
from app.gestures import dist


def _report_progress(block_num, block_size, total_size):
    if total_size <= 0:
        return
    done = min(100, block_num * block_size * 100 // total_size)
    print(f"\r  {done}%", end="", flush=True)


def ensure_model(path, url, label, required=True):
    """Pastikan file model ada; unduh kalau belum. Return True kalau tersedia.

    Kalau `required` dan unduhan gagal, lempar SystemExit dengan perintah unduh
    manual — itu satu-satunya kegagalan yang memang tidak bisa dilanjutkan.
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
            os.remove(path)  # buang file separuh jadi
        msg = f"[warn] gagal mengunduh model {label}: {exc}"
        if not required:
            print(msg)
            return False
        raise SystemExit(
            f"{msg}\nUnduh manual lalu jalankan lagi:\n"
            f'  curl -L -o "{path}" {url}'
        )


def make_landmarker(running_mode):
    """Buat HandLandmarker. running_mode: vision.RunningMode.VIDEO / IMAGE."""
    ensure_model(config.MODEL_PATH, config.MODEL_URL, "hand_landmarker",
                 required=True)
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=config.MODEL_PATH),
        running_mode=running_mode,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return vision.HandLandmarker.create_from_options(options)


def make_face_detector(running_mode):
    """Buat FaceDetector (opsional). Return None kalau model tidak tersedia.

    Tanpa model ini gestur Kicaw tetap jalan, hanya memakai perkiraan posisi
    tangan alih-alih jarak ke titik mulut yang sebenarnya.
    """
    ok = ensure_model(config.FACE_MODEL_PATH, config.FACE_MODEL_URL,
                      "blaze_face_short_range", required=False)
    if not ok:
        print("[info] model wajah tidak tersedia -> Kicaw pakai perkiraan posisi.")
        return None
    options = vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=config.FACE_MODEL_PATH),
        running_mode=running_mode,
        min_detection_confidence=0.5,
    )
    return vision.FaceDetector.create_from_options(options)


def mouth_from_faces(face_result, w, h):
    """Ambil titik mulut + radius dari hasil FaceDetector -> (mx, my, r) / None.

    Keypoint BlazeFace: 0=mata kanan, 1=mata kiri, 2=hidung, 3=mulut,
    4=telinga kanan, 5=telinga kiri (ternormalisasi 0..1).
    """
    dets = getattr(face_result, "detections", None) if face_result else None
    if not dets:
        return None
    det = max(dets, key=lambda d: d.bounding_box.width * d.bounding_box.height)
    kps = det.keypoints
    if len(kps) < 6:
        return None
    mx, my = kps[3].x * w, kps[3].y * h
    face_w = dist((kps[4].x * w, kps[4].y * h), (kps[5].x * w, kps[5].y * h))
    return (mx, my, max(face_w * 1.1, 40.0))
