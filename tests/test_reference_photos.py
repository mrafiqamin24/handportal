"""Regresi terhadap foto gestur asli di `assets/gestur/`.

Landmark ketiga foto sudah dibekukan ke `fixtures/reference_landmarks.json`
oleh MediaPipe, jadi test di sini tetap murni: tanpa kamera, tanpa MediaPipe,
tanpa membuka file gambar. Fungsinya menjaga agar penyetelan ambang gestur
tidak pernah lagi diam-diam mematikan pose yang benar-benar dipakai pengguna.

Kalau salah satu test di sini gagal, artinya aplikasi berhenti mengenali gestur
yang jelas-jelas terlihat di foto — bukan sekadar angka test yang perlu diubah.
"""

import json
import os

import pytest

from app import gestures as g

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures",
                       "reference_landmarks.json")


@pytest.fixture(scope="module")
def reference():
    with open(FIXTURE, encoding="utf-8") as f:
        return json.load(f)


def photo(reference, name):
    data = reference["photos"][name]
    hands = [[tuple(p) for p in hand] for hand in data["hands"]]
    mouth = tuple(data["mouth"]) if data["mouth"] else None
    return hands, data["labels"], mouth


def test_fixture_matches_the_three_reference_photos(reference):
    assert set(reference["photos"]) == {
        "Fotokitablurr.jpg", "Love.jpg", "Kicaw.jpg"}


# --------------------------------------------------------------- ✌️ Peace

def test_peace_photo_is_classified_as_peace(reference):
    hands, _, _ = photo(reference, "Fotokitablurr.jpg")
    assert len(hands) == 2
    assert [g.classify_hand(h) for h in hands] == ["PEACE", "PEACE"]


def test_peace_photo_is_not_mistaken_for_a_heart(reference):
    """Dua tangan ✌️ adalah kandidat salah-deteksi paling berbahaya untuk
    🫶 Heart, karena keduanya sama-sama dua tangan terangkat."""
    hands, _, _ = photo(reference, "Fotokitablurr.jpg")
    assert g.detect_two_hand_heart(hands) is False


def test_peace_photo_is_not_mistaken_for_kicaw(reference):
    hands, _, mouth = photo(reference, "Fotokitablurr.jpg")
    assert g.detect_kicaw(hands, 1280, 720, mouth) is False


# --------------------------------------------------------------- 🫶 Heart

def test_heart_photo_is_detected(reference):
    """Regresi utama: aturan lama menuntut jempol 0,55 telapak di bawah
    telunjuk dan bentang 0,90 per tangan. Pada foto ini angkanya 0,15 dan
    0,40 — jadi hati asli tidak pernah menyala sama sekali."""
    hands, _, _ = photo(reference, "Love.jpg")
    assert len(hands) == 2
    assert g.detect_two_hand_heart(hands) is True


def test_heart_photo_has_no_single_hand_gesture_competing(reference):
    """Heart menang lebih dulu di `main`, tapi tidak boleh ada gestur satu
    tangan yang ikut menyala di pose yang sama."""
    hands, _, _ = photo(reference, "Love.jpg")
    assert [g.classify_hand(h) for h in hands] == [None, None]


def test_heart_photo_is_not_a_pinch(reference):
    """Ujung jempol dan telunjuk tetap terpisah pada pose hati, jadi 👌 OK
    tidak boleh ikut terpicu."""
    hands, _, _ = photo(reference, "Love.jpg")
    assert not any(g.is_ok_pose(h) for h in hands)


# --------------------------------------------------------------- 🐦 Kicaw

def test_kicaw_photo_only_exposes_one_hand_to_mediapipe(reference):
    """Batasan yang diketahui dan disengaja.

    Pada `Kicaw.jpg` tangan yang menutup mulut menempel di wajah, dan
    MediaPipe tidak menemukannya pada confidence 0,20-0,58, resolusi 768
    maupun 1280, gambar asli maupun dibalik, num_hands 2 maupun 4, serta crop
    ketat di sekitar mulut. Yang terlihat hanya telapak yang menjulur.

    Kicaw sengaja tetap menuntut dua tangan supaya telapak terbuka biasa tidak
    memicunya. Konsekuensinya foto ini memang tidak terdeteksi; test ini ada
    agar batasan itu tercatat, bukan terlupakan.
    """
    hands, _, _ = photo(reference, "Kicaw.jpg")
    assert len(hands) == 1
    assert g.fingers_extended(hands[0]) == [True] * 5


def test_kicaw_needs_the_second_hand(reference):
    hands, _, mouth = photo(reference, "Kicaw.jpg")
    assert g.detect_kicaw(hands, 1280, 720, mouth) is False


def test_lone_open_palm_triggers_no_gesture_at_all(reference):
    """Telapak terbuka penuh sengaja tidak diklaim gestur apa pun."""
    hands, _, _ = photo(reference, "Kicaw.jpg")
    assert g.classify_hand(hands[0]) is None
    assert g.is_ok_pose(hands[0]) is False


# --------------------------------------------------------------- portal

@pytest.mark.parametrize("name", ["Fotokitablurr.jpg", "Love.jpg",
                                  "Kicaw.jpg"])
def test_no_reference_gesture_accidentally_opens_a_portal(reference, name):
    """Ketiga pose gestur tidak boleh lolos validasi pose portal."""
    from app.portal import build_fingertip_quad

    hands, labels, _ = photo(reference, name)
    assert build_fingertip_quad(hands, labels) is None
