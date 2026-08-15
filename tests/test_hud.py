"""Hit target kontrol HUD murni; tidak membutuhkan window OpenCV."""

from app.hud import CAMERA_BUTTON_W, Hud


def test_mode_switch_has_two_distinct_click_targets():
    hud = Hud()
    assert hud.hit_test(30, 25, 1280) == "mode:GESTUR"
    assert hud.hit_test(200, 25, 1280) == "mode:PORTAL"


def test_camera_button_is_anchored_to_right_edge():
    hud = Hud()
    x = 1280 - CAMERA_BUTTON_W
    assert hud.hit_test(x, 25, 1280) == "camera:next"
    assert hud.hit_test(640, 200, 1280) is None
