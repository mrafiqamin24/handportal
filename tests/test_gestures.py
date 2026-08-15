"""Test helper landmark murni: tanpa kamera, tanpa MediaPipe."""

import pytest

from app import gestures as g
from tests.conftest import make_hand


def test_dist_is_euclidean():
    assert g.dist((0, 0), (3, 4)) == 5.0


def test_joint_angle_recognizes_straight_and_bent_joints():
    assert g.joint_angle((0, 0), (1, 0), (2, 0)) == pytest.approx(180)
    assert g.joint_angle((0, 0), (1, 0), (1, 1)) == pytest.approx(90)


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


def test_classify_peace():
    assert g.classify_hand(make_hand(extended=("index", "middle"))) == "PEACE"


def test_peace_rejects_an_open_thumb():
    pts = make_hand(extended=("thumb", "index", "middle"))
    assert g.classify_hand(pts) is None


def test_classify_ily():
    assert g.classify_hand(make_hand(extended=("thumb", "index", "pinky"))) == "ILY"


def test_ily_rejects_middle_finger_that_is_not_folded():
    pts = make_hand(extended=("thumb", "index", "middle", "pinky"))
    assert g.classify_hand(pts) is None


def test_classify_no_longer_returns_ok_from_finger_shape():
    """👌 OK sekarang dipicu pinch-ditahan (lihat tests/test_pinch.py),
    bukan bentuk jari. classify_hand tidak boleh lagi mengklaimnya."""
    pts = make_hand(extended=("middle", "ring", "pinky"), thumb_index_gap=8)
    assert g.classify_hand(pts) != "OK"


def test_pinch_distance_is_normalized_by_palm_size():
    near = make_hand(extended=("middle", "ring", "pinky"), scale=60.0,
                     thumb_index_gap=6)
    far = make_hand(extended=("middle", "ring", "pinky"), scale=120.0,
                    thumb_index_gap=12)
    assert g.pinch_distance(near) == pytest.approx(g.pinch_distance(far), abs=0.02)


def test_classify_returns_none_for_fist():
    assert g.classify_hand(make_hand(extended=())) is None


def test_classify_returns_none_for_open_palm():
    # telapak terbuka penuh bukan gestur apa pun -> tidak boleh salah kenal
    assert g.classify_hand(make_hand(extended=("thumb", "index", "middle",
                                               "ring", "pinky"))) is None


def test_heart_needs_two_hands():
    one = [make_hand(extended=("index",))]
    assert g.detect_two_hand_heart(one, 1280) is False


def test_heart_detected_when_tips_meet():
    left = make_hand(extended=("thumb", "index"), origin=(600, 400))
    right = make_hand(extended=("thumb", "index"), origin=(680, 400))
    # paksa ujung telunjuk bertemu di atas, ujung jempol berdekatan di bawah
    left[g.INDEX_TIP] = (630, 300)
    right[g.INDEX_TIP] = (650, 300)
    left[g.THUMB_TIP] = (600, 400)
    right[g.THUMB_TIP] = (680, 400)
    assert g.detect_two_hand_heart([left, right], 1280) is True


def test_heart_rejected_when_index_tips_below_thumbs():
    left = make_hand(origin=(600, 400))
    right = make_hand(origin=(680, 400))
    left[g.INDEX_TIP] = (630, 500)
    right[g.INDEX_TIP] = (650, 500)
    left[g.THUMB_TIP] = (600, 300)
    right[g.THUMB_TIP] = (680, 300)
    assert g.detect_two_hand_heart([left, right], 1280) is False


def test_kicaw_detected_for_either_hand_role():
    mouth_hand = make_hand(extended=(), origin=(640, 200), scale=60.0)
    open_hand = make_hand(extended=("index", "middle", "ring", "pinky"),
                          origin=(300, 500))
    mouth = (640, 200, 120.0)
    assert g.detect_kicaw([mouth_hand, open_hand], 1280, 720, mouth) is True
    # urutan tangan dibalik: hasilnya harus sama
    assert g.detect_kicaw([open_hand, mouth_hand], 1280, 720, mouth) is True


def test_kicaw_rejected_when_second_hand_is_closed():
    mouth_hand = make_hand(extended=(), origin=(640, 200), scale=60.0)
    fist = make_hand(extended=(), origin=(300, 500))
    assert g.detect_kicaw([mouth_hand, fist], 1280, 720, (640, 200, 120.0)) is False


def test_kicaw_requires_all_four_forward_fingers_open():
    mouth_hand = make_hand(extended=(), origin=(640, 200), scale=60.0)
    three_fingers = make_hand(extended=("index", "middle", "ring"),
                              origin=(300, 500))
    assert g.detect_kicaw(
        [mouth_hand, three_fingers], 1280, 720, (640, 200, 120.0)) is False


def test_debouncer_requires_stable_frames():
    d = g.GestureDebouncer(hold_frames=3)
    assert d.update("PEACE") is None
    assert d.update("PEACE") is None
    assert d.update("PEACE") == "PEACE"


def test_debouncer_resets_on_a_deviating_frame():
    d = g.GestureDebouncer(hold_frames=3)
    d.update("PEACE")
    d.update("PEACE")
    d.update("OK")           # menyimpang -> hitungan ulang dari nol
    assert d.update("PEACE") is None


def test_debouncer_keeps_last_active_until_new_gesture_is_stable():
    d = g.GestureDebouncer(hold_frames=2)
    d.update("PEACE")
    assert d.update("PEACE") == "PEACE"
    # satu frame kosong belum cukup untuk mematikan
    assert d.update(None) == "PEACE"
    assert d.update(None) is None


def test_smoother_matches_hands_by_wrist_when_order_swaps():
    a = make_hand(origin=(200, 300))
    b = make_hand(origin=(900, 300))
    s = g.HandSmoother(alpha=0.5, match_dist=200.0)
    s.update([a, b])
    out = s.update([b, a])            # urutan tertukar
    # tangan yang wrist-nya dekat 200 harus tetap dihaluskan ke tangan pertama
    assert abs(out[0][g.WRIST][0] - 900) < 5
    assert abs(out[1][g.WRIST][0] - 200) < 5


def test_smoother_returns_integer_pixel_points():
    s = g.HandSmoother()
    out = s.update([make_hand()])
    assert all(isinstance(v, int) for v in out[0][0])
