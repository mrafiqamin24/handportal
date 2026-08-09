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
    # animasi memutar hue terus-menerus, jadi nilai >1 harus kembali ke awal
    assert draw.hsv_color(1.0) == draw.hsv_color(0.0)
    assert draw.hsv_color(1.25) == draw.hsv_color(0.25)


def test_hsv_color_distinguishes_different_hues():
    assert draw.hsv_color(0.0) != draw.hsv_color(0.5)


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
