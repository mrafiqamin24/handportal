"""Landmark tangan -> label gestur.

Modul ini sengaja murni: tidak mengimpor apa pun untuk menggambar dan tidak
pernah membaca jam sendiri. Semua fungsi yang bergantung waktu menerima `now`
sebagai argumen, sehingga seluruh isinya bisa di-test tanpa kamera.
"""

import math

# Landmark index
WRIST = 0
THUMB_TIP, THUMB_IP, THUMB_MCP = 4, 3, 2
INDEX_TIP, INDEX_PIP, INDEX_MCP = 8, 6, 5
MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP = 12, 10, 9
RING_TIP, RING_PIP = 16, 14
PINKY_TIP, PINKY_PIP, PINKY_MCP = 20, 18, 17

TIP_IDS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]

# Koneksi antar-landmark untuk menggambar "tulang" tangan (skeleton).
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # jempol
    (0, 5), (5, 6), (6, 7), (7, 8),          # telunjuk
    (5, 9), (9, 10), (10, 11), (11, 12),     # tengah
    (9, 13), (13, 14), (14, 15), (15, 16),   # manis
    (13, 17), (17, 18), (18, 19), (19, 20),  # kelingking
    (0, 17),                                 # pangkal telapak
]


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def hand_points(landmark_list, w, h):
    """Ubah daftar landmark ternormalisasi (Tasks API) menjadi titik pixel."""
    return [(int(lm.x * w), int(lm.y * h)) for lm in landmark_list]


def palm_scale(pts):
    """Ukuran telapak (wrist -> middle MCP) sebagai skala referensi."""
    return max(dist(pts[WRIST], pts[MIDDLE_MCP]), 1e-3)


def fingers_extended(pts):
    """
    Tentukan jari mana yang terbuka. Memakai jarak tip-ke-wrist vs pip-ke-wrist
    agar tidak bergantung orientasi tangan.
    Return list bool: [thumb, index, middle, ring, pinky]
    """
    wrist = pts[WRIST]
    extended = []

    # Thumb: bandingkan jarak tip & ip terhadap pangkal jari kelingking.
    ref = pts[PINKY_MCP]
    extended.append(dist(pts[THUMB_TIP], ref) > dist(pts[THUMB_IP], ref))

    # Empat jari lain: tip lebih jauh dari wrist dibanding pip => terbuka.
    for tip, pip in (
        (INDEX_TIP, INDEX_PIP),
        (MIDDLE_TIP, MIDDLE_PIP),
        (RING_TIP, RING_PIP),
        (PINKY_TIP, PINKY_PIP),
    ):
        extended.append(dist(pts[tip], wrist) > dist(pts[pip], wrist) * 1.05)
    return extended
