"""Kontrak yang berlaku untuk SEMUA filter: uint8 masuk, uint8 keluar dengan
shape sama, input tidak pernah dimutasi, dan aman untuk crop sekecil apa pun.
Ini yang membuat filter bisa dipasang di mana saja tanpa kejutan."""

import numpy as np
import pytest

from app import effects


def sample(h=48, w=64):
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


@pytest.mark.parametrize("fn", effects.EFFECTS, ids=effects.EFFECT_NAMES)
def test_effect_preserves_shape_and_dtype(fn):
    img = sample()
    out = fn(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


@pytest.mark.parametrize("fn", effects.EFFECTS, ids=effects.EFFECT_NAMES)
def test_effect_does_not_mutate_its_input(fn):
    img = sample()
    before = img.copy()
    fn(img)
    assert np.array_equal(img, before)


@pytest.mark.parametrize("fn", effects.EFFECTS, ids=effects.EFFECT_NAMES)
@pytest.mark.parametrize("size", [(1, 1), (2, 2), (3, 5)])
def test_effect_survives_tiny_crops(fn, size):
    img = sample(*size)
    out = fn(img)
    assert out.shape == img.shape


def test_effects_and_names_line_up():
    assert len(effects.EFFECTS) == len(effects.EFFECT_NAMES) == 13
    assert effects.EFFECT_NAMES[0] == "Blur"


def test_apply_full_frame_returns_same_shape():
    img = sample(120, 160)
    out = effects.apply_full_frame(img, effects.fx_thermal)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_apply_full_frame_blend_endpoints_match_each_effect():
    img = sample(120, 160)
    only_new = effects.apply_full_frame(img, effects.fx_thermal,
                                        effects.fx_duotone, blend=1.0)
    plain_new = effects.apply_full_frame(img, effects.fx_thermal)
    assert np.array_equal(only_new, plain_new)


def test_apply_full_frame_does_not_mutate_its_input():
    img = sample(120, 160)
    before = img.copy()
    effects.apply_full_frame(img, effects.fx_posterize_neon)
    assert np.array_equal(img, before)


def test_filter_transition_eases_and_finishes():
    t = effects.FilterTransition(current_idx=0, duration=1.0)
    assert t.switch(1, 10.0) == 0
    previous, blend = t.state(10.25)
    assert previous == 0
    assert blend == pytest.approx(0.15625)
    assert t.state(11.0) == (None, 1.0)


def test_filter_transition_retargets_from_dominant_visible_effect():
    t = effects.FilterTransition(current_idx=0, duration=1.0)
    t.switch(1, 0.0)
    # Baru 20%: efek lama masih dominan, jadi retarget berangkat dari 0.
    assert t.switch(2, 0.2) == 0
    assert t.current_idx == 2
    assert t.previous_idx == 0
    # Setelah target dominan, retarget berikutnya berangkat dari target itu.
    t.state(0.9)
    assert t.switch(3, 0.9) == 2
