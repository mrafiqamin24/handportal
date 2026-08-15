"""Mesin-status pinch. Satu ambang memisahkan dua hasil:

    lepas SEBELUM 0,35 dtk   -> TAP  (event tersedia, bukan ganti filter)
    bertahan DI 0,35 dtk     -> HOLD (gestur OK)

Semua test memakai jam yang di-inject; tidak ada `sleep` di sini.
"""

import pytest

from app import config
from app.gestures import DoublePinchDetector, PinchTapDetector

TIGHT = 0.20    # jelas di bawah PINCH_ENTER -> dianggap menyentuh
LOOSE = 1.20    # jelas di atas PINCH_EXIT   -> dianggap terlepas


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


def test_no_hands_releases_the_pinch_after_the_grace_frames(det, clock):
    """Tangan yang benar-benar hilang tetap melepas pinch — hanya tidak di
    frame pertama, karena beberapa frame kosong dulu dimaafkan."""
    det.update([TIGHT], clock.now)
    clock.advance(0.05)
    for _ in range(det.missing_grace):
        assert det.update([], clock.now).tap is False
    assert det.update([], clock.now).tap is True


def test_short_dropout_does_not_restart_the_hold_timer(clock):
    """Regresi: pose 👌 yang gagal dikenali satu frame dulu me-reset hitungan
    0,35 dtk, sehingga gestur OK nyaris mustahil dipicu."""
    det = PinchTapDetector(hold_s=0.30, missing_grace=3)
    det.update([TIGHT], clock.now)
    clock.advance(0.15)
    det.update([], clock.now)          # MediaPipe berkedip sekali
    clock.advance(0.16)
    assert det.update([TIGHT], clock.now).hold_started is True


def test_dropout_grace_still_matures_into_a_hold(clock):
    """Frame kosong tidak menahan hold: kalau ambang waktu lewat saat dropout,
    OK tetap menyala begitu grace-nya masih berlaku."""
    det = PinchTapDetector(hold_s=0.20, missing_grace=3)
    det.update([TIGHT], clock.now)
    clock.advance(0.25)
    assert det.update([], clock.now).holding is True


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


def test_single_pinch_reset_clears_an_in_progress_hold(det, clock):
    det.update([TIGHT], clock.now)
    det.reset()
    clock.advance(config.PINCH_HOLD_S + 0.1)
    assert det.update([LOOSE], clock.now).holding is False


def test_double_pinch_requires_both_hands_for_stable_frames():
    det = DoublePinchDetector(stable_frames=3, release_frames=2)
    assert det.update([TIGHT]).triggered is False
    assert det.update([TIGHT, LOOSE]).triggered is False
    assert det.update([TIGHT, TIGHT]).triggered is False
    assert det.update([TIGHT, TIGHT]).triggered is False
    assert det.update([TIGHT, TIGHT]).triggered is True


def test_double_pinch_fires_once_until_both_hands_release():
    det = DoublePinchDetector(stable_frames=2, release_frames=2)
    det.update([TIGHT, TIGHT])
    assert det.update([TIGHT, TIGHT]).triggered is True
    assert det.update([TIGHT, TIGHT]).triggered is False
    # Satu tangan terbuka atau detector dropout belum boleh re-arm.
    det.update([LOOSE, TIGHT])
    det.update([])
    assert det.update([TIGHT, TIGHT]).triggered is False
    det.update([LOOSE, LOOSE])
    det.update([LOOSE, LOOSE])
    det.update([TIGHT, TIGHT])
    assert det.update([TIGHT, TIGHT]).triggered is True


def test_double_pinch_reset_allows_a_fresh_trigger():
    det = DoublePinchDetector(stable_frames=1)
    assert det.update([TIGHT, TIGHT]).triggered is True
    det.reset()
    assert det.update([TIGHT, TIGHT]).triggered is True
