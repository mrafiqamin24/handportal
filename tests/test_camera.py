"""Pergantian kamera diuji dengan capture palsu, tanpa menyentuh webcam."""

from app.main import open_camera, switch_camera


class FakeCapture:
    def __init__(self, opened=True):
        self.opened = opened
        self.released = False
        self.props = []

    def set(self, prop, value):
        self.props.append((prop, value))

    def isOpened(self):
        return self.opened

    def release(self):
        self.released = True


def test_optional_camera_probe_returns_none_and_releases_failed_capture():
    made = []

    def factory(_index):
        cap = FakeCapture(opened=False)
        made.append(cap)
        return cap

    assert open_camera(2, required=False, capture_factory=factory) is None
    assert made[0].released is True


def test_switch_camera_keeps_active_capture_when_no_candidate_opens():
    active = FakeCapture()

    def fail(_index, required=False):
        assert required is False
        return None

    cap, index, changed = switch_camera(active, 0, open_fn=fail, scan_max=2)
    assert cap is active
    assert index == 0
    assert changed is False
    assert active.released is False


def test_switch_camera_releases_old_only_after_candidate_opens():
    active = FakeCapture()
    candidate = FakeCapture()

    def probe(index, required=False):
        return candidate if index == 2 else None

    cap, index, changed = switch_camera(active, 0, open_fn=probe, scan_max=3)
    assert (cap, index, changed) == (candidate, 2, True)
    assert active.released is True
