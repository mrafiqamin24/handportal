"""Geometri portal dan keamanan komposit. Yang paling mungkin salah di kode
seperti ini adalah menulis di luar batas frame saat portal separuh keluar
layar — jadi itu yang diuji."""

import numpy as np
import pytest

from app import effects, portal


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


def test_render_portal_changes_pixels_inside_the_box():
    img = blank()
    img[:] = 200
    quad = portal.build_box((40, 30), (120, 90))
    portal.render_portal(img, effects.fx_invert_glitch, quad)
    assert img[60, 80, 0] != 200


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
