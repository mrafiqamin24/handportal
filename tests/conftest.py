"""Perkakas test: landmark tangan sintetis + jam yang di-inject.

Tidak ada test di repo ini yang menyentuh webcam atau `time.sleep`.
"""

import pytest

# Indeks landmark MediaPipe, diurutkan per jari (4 sendi tiap jari).
# 0 = pergelangan; 1..4 jempol; 5..8 telunjuk; 9..12 tengah;
# 13..16 manis; 17..20 kelingking.
FINGER_SLOTS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}

# Arah tiap jari saat terbuka penuh, relatif terhadap pergelangan di (0, 0)
# dengan sumbu y menghadap ke bawah (konvensi pixel OpenCV): jari mengarah
# ke ATAS, jadi y negatif. Jempol melebar ke samping kiri.
FINGER_DIRS = {
    "thumb": (-0.85, -0.45),
    "index": (-0.22, -1.0),
    "middle": (0.0, -1.05),
    "ring": (0.22, -1.0),
    "pinky": (0.42, -0.85),
}


def make_hand(extended=("index", "middle"), origin=(320, 400), scale=100.0,
              thumb_index_gap=None):
    """Bangun 21 titik pixel untuk satu tangan.

    `extended` — nama jari yang lurus; sisanya digambar menekuk (ujung jari
    ditarik balik mendekat ke pergelangan sehingga `fingers_extended` melihat
    tip lebih dekat ke wrist daripada pip).

    `thumb_index_gap` — kalau diisi, ujung jempol dipindah supaya jaraknya ke
    ujung telunjuk persis sekian pixel. Dipakai untuk menguji pinch/OK.
    """
    ox, oy = origin
    pts = [(0, 0)] * 21
    pts[0] = (int(ox), int(oy))
    for name, slots in FINGER_SLOTS.items():
        dx, dy = FINGER_DIRS[name]
        straight = name in extended
        for i, idx in enumerate(slots, start=1):
            # sendi ke-i: 0,25 / 0,5 / 0,75 / 1,0 dari panjang jari
            reach = i / 4.0
            if not straight and i >= 3:
                # dua sendi terakhir melipat balik ke telapak
                reach = 0.5 - (i - 2) * 0.18
            pts[idx] = (int(ox + dx * scale * reach),
                        int(oy + dy * scale * reach))
    if thumb_index_gap is not None:
        ix, iy = pts[8]
        pts[4] = (int(ix + thumb_index_gap), int(iy))
    return pts


class FakeClock:
    """Jam yang dikendalikan test. Tidak ada `sleep` di mana pun."""

    def __init__(self, start=0.0):
        self.now = float(start)

    def advance(self, seconds):
        self.now += float(seconds)
        return self.now


@pytest.fixture
def clock():
    return FakeClock()
