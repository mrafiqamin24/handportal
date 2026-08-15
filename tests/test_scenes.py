"""Kontrak overlay gestur dan blur latar khusus Peace."""

import numpy as np

from app.scenes import GestureScenes


def test_peace_overlay_leaves_uncovered_pixels_untouched():
    y, x = np.mgrid[0:240, 0:320]
    frame = np.stack((x % 256, y % 256, (x + y) % 256), axis=-1).astype(np.uint8)
    before = frame.copy()
    GestureScenes().render(frame, "PEACE", 0.0)
    # Titik ini jauh dari judul tengah dan thumbnail kanan-bawah.
    assert np.array_equal(frame[10, 10], before[10, 10])


def test_peace_background_restores_blur_without_affecting_other_gestures():
    frame = np.zeros((80, 120, 3), np.uint8)
    frame[:, 60:] = 255
    peace = GestureScenes.apply_background(frame, "PEACE")
    heart = GestureScenes.apply_background(frame, "HEART")
    assert not np.array_equal(peace, frame)
    assert heart is frame
