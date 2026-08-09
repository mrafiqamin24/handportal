"""Mesin-status pinch. Satu ambang memisahkan dua hasil:

    lepas SEBELUM 0,35 dtk   -> TAP  (ganti efek)
    bertahan DI 0,35 dtk     -> HOLD (gestur OK)

Semua test memakai jam yang di-inject; tidak ada `sleep` di sini.
"""

import pytest

from app import config
from app.gestures import PinchTapDetector

TIGHT = 0.03    # jelas di bawah PINCH_ENTER -> dianggap menyentuh
LOOSE = 0.20    # jelas di atas PINCH_EXIT   -> dianggap terlepas


@pytest.fixture
def det():
    return PinchTapDetector()


def test_quick_tap_fires_once(det, clock):
    assert det.update([TIGHT], clock.now).tap is False   # baru menyentuh
    clock.advance(0.1)
    assert det.update([TIGHT], clock.now).tap is False   # masih ditahan
    clock.advance(0.05)
    assert det.update([LOOSE], clock.now).tap is True    # dilepas -> TAP
    clock.advance(0.05)
    assert det.update([LOOSE], clock.now).tap is False   # tidak berulang


def test_long_hold_never_fires_a_tap(det, clock):
    det.update([TIGHT], clock.now)
    clock.advance(1.0)
    det.update([TIGHT], clock.now)
    assert det.update([LOOSE], clock.now).tap is False


def test_hold_starts_exactly_once(det, clock):
    det.update([TIGHT], clock.now)
    clock.advance(config.PINCH_HOLD_S)
    first = det.update([TIGHT], clock.now)
    assert first.hold_started is True
    clock.advance(0.1)
    second = det.update([TIGHT], clock.now)
    assert second.hold_started is False
    assert second.holding is True        # tetap aktif selama ditahan


def test_holding_is_false_before_the_threshold(det, clock):
    det.update([TIGHT], clock.now)
    clock.advance(config.PINCH_HOLD_S - 0.01)
    ev = det.update([TIGHT], clock.now)
    assert ev.holding is False
    assert ev.hold_started is False


def test_exactly_at_threshold_is_a_hold_not_a_tap(det, clock):
    det.update([TIGHT], clock.now)
    clock.advance(config.PINCH_HOLD_S)
    ev = det.update([LOOSE], clock.now)   # dilepas persis di ambang
    assert ev.tap is False


def test_release_then_pinch_again_fires_again(det, clock):
    det.update([TIGHT], clock.now)
    clock.advance(0.05)
    assert det.update([LOOSE], clock.now).tap is True
    clock.advance(0.05)
    det.update([TIGHT], clock.now)
    clock.advance(0.05)
    assert det.update([LOOSE], clock.now).tap is True


def test_two_hands_pinching_together_fire_only_once(det, clock):
    det.update([TIGHT, TIGHT], clock.now)
    clock.advance(0.05)
    ev = det.update([LOOSE, LOOSE], clock.now)
    assert ev.tap is True
    clock.advance(0.05)
    assert det.update([LOOSE, LOOSE], clock.now).tap is False


def test_second_hand_releasing_late_does_not_fire_twice(det, clock):
    det.update([TIGHT, TIGHT], clock.now)
    clock.advance(0.05)
    assert det.update([TIGHT, LOOSE], clock.now).tap is False  # satu masih menyentuh
    clock.advance(0.05)
    assert det.update([LOOSE, LOOSE], clock.now).tap is True   # baru sekarang


def test_jitter_around_the_threshold_does_not_chatter(det, clock):
    """Histeresis: nilai di antara ENTER dan EXIT tidak mengubah status."""
    between = (config.PINCH_ENTER + config.PINCH_EXIT) / 2
    det.update([TIGHT], clock.now)
    taps = 0
    for _ in range(10):
        clock.advance(0.01)
        if det.update([between], clock.now).tap:
            taps += 1
    assert taps == 0


def test_no_hands_releases_the_pinch(det, clock):
    det.update([TIGHT], clock.now)
    clock.advance(0.05)
    assert det.update([], clock.now).tap is True


def test_adjust_shifts_both_thresholds_and_keeps_the_gap(det):
    gap = det.exit - det.enter
    det.adjust(config.PINCH_STEP)
    assert det.exit - det.enter == pytest.approx(gap)
    assert det.enter > config.PINCH_ENTER


def test_adjust_is_clamped(det):
    for _ in range(200):
        det.adjust(config.PINCH_STEP)
    assert det.enter <= config.PINCH_MAX
    for _ in range(400):
        det.adjust(-config.PINCH_STEP)
    assert det.enter >= config.PINCH_MIN
