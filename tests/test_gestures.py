"""Test helper landmark murni: tanpa kamera, tanpa MediaPipe."""

from app import gestures as g
from tests.conftest import make_hand


def test_dist_is_euclidean():
    assert g.dist((0, 0), (3, 4)) == 5.0


def test_hand_points_scales_normalized_landmarks_to_pixels():
    class LM:
        def __init__(self, x, y):
            self.x, self.y = x, y

    pts = g.hand_points([LM(0.5, 0.25), LM(1.0, 1.0)], 640, 480)
    assert pts == [(320, 120), (640, 480)]


def test_palm_scale_is_wrist_to_middle_mcp_distance():
    pts = make_hand(origin=(0, 0), scale=100.0)
    assert g.palm_scale(pts) == g.dist(pts[g.WRIST], pts[g.MIDDLE_MCP])


def test_palm_scale_never_returns_zero():
    # semua landmark menumpuk di satu titik -> pembagi tidak boleh nol
    assert g.palm_scale([(5, 5)] * 21) > 0.0


def test_fingers_extended_reports_open_fingers():
    pts = make_hand(extended=("index", "middle"))
    thumb, index, middle, ring, pinky = g.fingers_extended(pts)
    assert index and middle
    assert not ring and not pinky


def test_fingers_extended_on_open_palm():
    pts = make_hand(extended=("thumb", "index", "middle", "ring", "pinky"))
    assert g.fingers_extended(pts) == [True] * 5


def test_fingers_extended_on_fist():
    pts = make_hand(extended=())
    assert g.fingers_extended(pts)[1:] == [False] * 4
