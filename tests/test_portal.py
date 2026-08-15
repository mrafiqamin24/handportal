"""Geometri portal dan keamanan komposit. Yang paling mungkin salah di kode
seperti ini adalah menulis di luar batas frame saat portal separuh keluar
layar — jadi itu yang diuji."""

import numpy as np
import pytest

from app import effects, portal
from tests.conftest import make_hand


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


def test_build_fingertip_quad_uses_thumb_and_index_of_both_hands():
    left = make_hand(extended=("thumb", "index"), origin=(300, 350))
    right = make_hand(extended=("thumb", "index"), origin=(900, 350))
    quad = portal.build_fingertip_quad([left, right], ["Left", "Right"])
    assert quad is not None
    assert quad.shape == (4, 2)
    assert portal.quad_envelope_area(quad) > 0
    expected = {left[4], left[8], right[4], right[8]}
    assert {tuple(map(int, point)) for point in quad} == expected


def test_fingertip_quad_rejects_one_hand_but_tolerates_handedness_jitter():
    hand = make_hand(extended=("thumb", "index"))
    other = make_hand(extended=("thumb", "index"), origin=(800, 400))
    assert portal.build_fingertip_quad([hand]) is None
    # Geometri dua trek sudah cukup; label Left/Right kadang bergetar.
    assert portal.build_fingertip_quad(
        [hand, other], ["Left", "Left"]) is not None


def test_fingertip_quad_rejects_collapsed_or_non_l_pose():
    left = make_hand(extended=("index",), origin=(300, 350),
                     thumb_index_gap=3)
    right = make_hand(extended=("index",), origin=(900, 350),
                      thumb_index_gap=3)
    assert portal.build_fingertip_quad([left, right]) is None


def test_portal_pose_appears_after_two_valid_frames():
    left = make_hand(extended=("thumb", "index"), origin=(300, 350))
    right = make_hand(extended=("thumb", "index"), origin=(900, 350))
    detector = portal.PortalPoseDetector(acquire_frames=2)
    assert detector.update([left, right], now=0.0) is None
    assert detector.update([left, right], now=1 / 30) is not None


def test_active_portal_tolerates_pose_classifier_jitter():
    left = make_hand(extended=("thumb", "index"), origin=(300, 350))
    right = make_hand(extended=("thumb", "index"), origin=(900, 350))
    detector = portal.PortalPoseDetector(acquire_frames=1)
    assert detector.update([left, right], now=0.0) is not None
    # Rusak sudut sendi telunjuk tetapi pertahankan empat fingertip yang valid.
    left[6] = left[5]
    right[6] = right[5]
    assert portal.build_fingertip_quad([left, right]) is None
    assert detector.update([left, right], now=0.05) is not None


def test_portal_bridges_short_dropout_then_releases_by_elapsed_time():
    left = make_hand(extended=("thumb", "index"), origin=(300, 350))
    right = make_hand(extended=("thumb", "index"), origin=(900, 350))
    detector = portal.PortalPoseDetector(acquire_frames=1, lost_grace_s=0.45)
    assert detector.update([left, right], now=1.0) is not None
    assert detector.update([], now=1.30) is not None
    assert detector.update([], now=1.46) is None


def test_portal_is_held_while_double_pinch_collapses_the_quad():
    left = make_hand(extended=("thumb", "index"), origin=(300, 350))
    right = make_hand(extended=("thumb", "index"), origin=(900, 350))
    detector = portal.PortalPoseDetector(acquire_frames=1, lost_grace_s=0.1)
    assert detector.update([left, right], now=0.0) is not None
    # `hold` boleh melewati grace karena bentuk sengaja runtuh saat pinch.
    assert detector.update([], now=1.0, hold=True) is not None
    assert detector.update([], now=1.05) is not None
    assert detector.update([], now=1.11) is None


def test_portal_alpha_materializes_fast_and_fades_cleanly():
    alpha = portal.update_portal_alpha(0.0, True, 1 / 30)
    assert alpha > 0.35
    alpha = portal.update_portal_alpha(alpha, True, 1 / 30)
    assert alpha > 0.60
    faded = portal.update_portal_alpha(alpha, False, 1 / 30)
    assert 0.0 < faded < alpha


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


def test_quad_smoother_one_euro_reduces_small_jitter_without_freezing():
    s = portal.QuadSmoother()
    first = portal.build_box((0, 0), (100, 100))
    s.update(first, now=0.0)
    jittered = portal.build_box((2, 1), (102, 101))
    out = s.update(jittered, now=1 / 30)
    assert 0 < out[0][0] < 2
    assert 100 < out[2][0] < 102


def test_quad_smoother_follows_large_motion_quickly():
    s = portal.QuadSmoother()
    first = portal.build_box((0, 0), (100, 100))
    s.update(first, now=0.0)
    moved = first + np.array([100, 0], np.float32)
    out = s.update(moved, now=1 / 30)
    assert out[0][0] > 75


def test_fingertip_quad_preserves_crossed_twist_topology():
    left = make_hand(extended=("thumb", "index"), origin=(300, 350))
    right = make_hand(extended=("thumb", "index"), origin=(900, 350))
    # Putar tangan kanan: index di bawah, thumb di atas. Urutan semantik harus
    # menghasilkan polygon silang (shoelace kecil) tanpa ditata ulang ke hull.
    right[8] = (850, 410)
    right[4] = (930, 250)
    quad = portal.build_tracking_quad([right, left])
    assert quad is not None
    assert tuple(quad[0]) == left[8]
    assert tuple(quad[1]) == right[8]
    assert tuple(quad[2]) == right[4]
    assert portal.quad_envelope_area(quad) > portal.polygon_area(quad)


def test_render_portal_supports_a_crossed_bow_tie_shape():
    img = blank()
    img[:] = 200
    bow_tie = np.array([[30, 25], [130, 95], [130, 25], [30, 95]],
                       np.float32)
    portal.render_portal(img, effects.fx_invert_glitch, bow_tie)
    assert np.any(img != 200)


def test_render_portal_changes_pixels_inside_the_box():
    img = blank()
    img[:] = 200
    quad = portal.build_box((40, 30), (120, 90))
    portal.render_portal(img, effects.fx_invert_glitch, quad)
    assert img[60, 80, 0] != 200


def test_render_portal_scaled_effect_preserves_frame_contract():
    img = blank(121, 163)
    before_shape = img.shape
    quad = portal.build_box((17, 13), (151, 109))
    portal.render_portal(img, effects.fx_comic_ink, quad, effect_scale=0.50)
    assert img.shape == before_shape
    assert img.dtype == np.uint8


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
