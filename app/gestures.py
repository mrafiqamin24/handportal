"""Landmark tangan -> label gestur.

Modul ini sengaja murni: tidak mengimpor apa pun untuk menggambar dan tidak
pernah membaca jam sendiri. Semua fungsi yang bergantung waktu menerima `now`
sebagai argumen, sehingga seluruh isinya bisa di-test tanpa kamera.
"""

import collections
import math

from app import config

# Landmark index
WRIST = 0
THUMB_TIP, THUMB_IP, THUMB_MCP = 4, 3, 2
INDEX_TIP, INDEX_DIP, INDEX_PIP, INDEX_MCP = 8, 7, 6, 5
MIDDLE_TIP, MIDDLE_DIP, MIDDLE_PIP, MIDDLE_MCP = 12, 11, 10, 9
RING_TIP, RING_DIP, RING_PIP, RING_MCP = 16, 15, 14, 13
PINKY_TIP, PINKY_DIP, PINKY_PIP, PINKY_MCP = 20, 19, 18, 17

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


def joint_angle(a, b, c):
    """Sudut ABC dalam derajat, aman untuk landmark yang berimpit."""
    bax, bay = a[0] - b[0], a[1] - b[1]
    bcx, bcy = c[0] - b[0], c[1] - b[1]
    denom = math.hypot(bax, bay) * math.hypot(bcx, bcy)
    if denom < 1e-6:
        return 0.0
    cosine = max(-1.0, min(1.0, (bax * bcx + bay * bcy) / denom))
    return math.degrees(math.acos(cosine))


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
    # Rasio jalur sendi menolak jempol yang terlipat tetapi kebetulan menjauh.
    ref = pts[PINKY_MCP]
    thumb_path = (dist(pts[THUMB_MCP], pts[THUMB_IP])
                  + dist(pts[THUMB_IP], pts[THUMB_TIP]))
    thumb_direct = dist(pts[THUMB_MCP], pts[THUMB_TIP])
    extended.append(
        dist(pts[THUMB_TIP], ref) > dist(pts[THUMB_IP], ref) * 1.03
        and thumb_direct / max(thumb_path, 1e-3) > 0.72
        and joint_angle(pts[THUMB_MCP], pts[THUMB_IP], pts[THUMB_TIP]) > 135
    )

    # Empat jari lain harus sekaligus menjauh dari wrist DAN cukup lurus.
    # Syarat kelurusan berbasis rasio panjang chord/path, jadi tetap bekerja
    # saat tangan diputar dan lebih sulit tertipu jari yang sedang menekuk.
    for tip, dip, pip, mcp in (
        (INDEX_TIP, INDEX_DIP, INDEX_PIP, INDEX_MCP),
        (MIDDLE_TIP, MIDDLE_DIP, MIDDLE_PIP, MIDDLE_MCP),
        (RING_TIP, RING_DIP, RING_PIP, RING_MCP),
        (PINKY_TIP, PINKY_DIP, PINKY_PIP, PINKY_MCP),
    ):
        path = (dist(pts[mcp], pts[pip]) + dist(pts[pip], pts[dip])
                + dist(pts[dip], pts[tip]))
        straightness = dist(pts[mcp], pts[tip]) / max(path, 1e-3)
        reaches_out = dist(pts[tip], wrist) > dist(pts[pip], wrist) * 1.05
        pip_angle = joint_angle(pts[mcp], pts[pip], pts[dip])
        dip_angle = joint_angle(pts[pip], pts[dip], pts[tip])
        extended.append(reaches_out and straightness > 0.72
                        and pip_angle > 145 and dip_angle > 145)
    return extended


def fingers_folded(pts):
    """Jari yang benar-benar menekuk, bukan sekadar gagal dianggap lurus.

    Margin terpisah dari :func:`fingers_extended` membuat pose setengah jadi
    berada di zona netral.  Ini penting agar noise satu sendi tidak langsung
    mengubah pose ambigu menjadi PEACE atau ILY.
    """
    thumb_extended = fingers_extended(pts)[0]
    folded = [not thumb_extended]
    for tip, dip, pip, mcp in (
        (INDEX_TIP, INDEX_DIP, INDEX_PIP, INDEX_MCP),
        (MIDDLE_TIP, MIDDLE_DIP, MIDDLE_PIP, MIDDLE_MCP),
        (RING_TIP, RING_DIP, RING_PIP, RING_MCP),
        (PINKY_TIP, PINKY_DIP, PINKY_PIP, PINKY_MCP),
    ):
        path = (dist(pts[mcp], pts[pip]) + dist(pts[pip], pts[dip])
                + dist(pts[dip], pts[tip]))
        straightness = dist(pts[mcp], pts[tip]) / max(path, 1e-3)
        pip_angle = joint_angle(pts[mcp], pts[pip], pts[dip])
        dip_angle = joint_angle(pts[pip], pts[dip], pts[tip])
        folded.append(straightness < 0.62
                      and min(pip_angle, dip_angle) < 130)
    return folded


def classify_hand(pts):
    """
    Klasifikasi gestur satu tangan -> nama gestur atau None.
    Aturan diperketat supaya tidak gampang salah deteksi (false positive).

    👌 OK tidak ada di sini: gestur itu dipicu pinch yang ditahan, ditangani
    `PinchTapDetector`. Bentuk jari tidak lagi menentukannya.
    """
    thumb, index, middle, ring, pinky = fingers_extended(pts)
    _, index_folded, middle_folded, ring_folded, pinky_folded = (
        fingers_folded(pts))
    scale = palm_scale(pts)
    d_thumb_index = dist(pts[THUMB_TIP], pts[INDEX_TIP])

    # 🤟 ILY: jempol + telunjuk + kelingking terbuka; tengah & manis tertutup.
    # Jempol harus benar-benar melebar (jauh dari telunjuk).
    if (thumb and index and pinky and middle_folded and ring_folded
            and d_thumb_index > 0.75 * scale
            and dist(pts[INDEX_TIP], pts[PINKY_TIP]) > 0.70 * scale):
        return "ILY"

    # ✌️ Peace: telunjuk & tengah terbuka membentuk huruf V yang jelas;
    # manis & kelingking tertutup. Jempol tidak melebar (bukan ILY).
    # Dua ukuran V dipakai sekaligus: lebar mutlak (relatif telapak) menolak
    # dua jari yang rapat, dan rasio terhadap jarak pangkalnya menolak tangan
    # yang sekadar jauh dari kamera.
    if (not thumb and index and middle and ring_folded and pinky_folded
            and not index_folded):
        tip_span = dist(pts[INDEX_TIP], pts[MIDDLE_TIP])
        mcp_span = max(dist(pts[INDEX_MCP], pts[MIDDLE_MCP]), 1e-3)
        if tip_span > 0.42 * scale and tip_span > 1.35 * mcp_span:
            return "PEACE"

    return None


PinchEvent = collections.namedtuple("PinchEvent", "tap hold_started holding")


def pinch_distance(pts):
    """Jarak ujung jempol <-> ujung telunjuk, dinormalkan ke ukuran telapak.

    Dinormalkan supaya ambangnya tetap benar saat tangan mendekat atau menjauh
    dari kamera.
    """
    return dist(pts[THUMB_TIP], pts[INDEX_TIP]) / palm_scale(pts)


class PinchTapDetector:
    """Satu mesin-status untuk kedua tangan sekaligus.

        sentuh lalu lepas SEBELUM hold_s  -> TAP  (ganti efek)
        sentuh dan bertahan DI hold_s     -> HOLD (gestur OK)

    Masing-masing memicu tepat sekali per pinch. Dua ambang jarak berbeda
    (masuk lebih ketat daripada keluar) meredam jitter landmark di sekitar
    ambang. Kedua tangan berbagi satu status, jadi pinch bersamaan tetap
    dihitung satu kali.
    """

    def __init__(self, hold_s=config.PINCH_HOLD_S, enter=config.PINCH_ENTER,
                 exit=config.PINCH_EXIT):
        self.hold_s = hold_s
        self.enter = enter
        self.exit = exit
        self._pinching = False
        self._start = 0.0
        self._held = False

    def adjust(self, delta):
        """Geser sensitivitas; selisih histeresis dipertahankan."""
        gap = self.exit - self.enter
        self.enter = max(config.PINCH_MIN,
                         min(config.PINCH_MAX, self.enter + delta))
        self.exit = self.enter + gap

    def update(self, dists, now):
        """`dists` = jarak pinch ternormalisasi tiap tangan; boleh kosong."""
        d = min(dists) if dists else float("inf")
        tap = False
        hold_started = False
        if self._pinching:
            if d > self.exit:                       # dilepas
                if (now - self._start) < self.hold_s:
                    tap = True
                self._pinching = False
                self._held = False
            elif not self._held and (now - self._start) >= self.hold_s:
                self._held = True                   # jadi HOLD
                hold_started = True
        elif d < self.enter:
            self._pinching = True
            self._start = now
            self._held = False
        return PinchEvent(tap, hold_started, self._held)


def detect_two_hand_heart(hands_pts, w):
    """🫶 Heart: dua tangan, ujung telunjuk hampir bersentuhan & ujung jempol
    berdekatan (membentuk hati). Diperketat agar dua tangan peace tidak ikut."""
    if len(hands_pts) != 2:
        return False
    a, b = hands_pts
    index_close = dist(a[INDEX_TIP], b[INDEX_TIP]) < 0.12 * w
    thumb_close = dist(a[THUMB_TIP], b[THUMB_TIP]) < 0.18 * w
    # ujung telunjuk (atas hati) lebih tinggi dari ujung jempol (bawah hati)
    index_top = (a[INDEX_TIP][1] + b[INDEX_TIP][1]) < (a[THUMB_TIP][1] + b[THUMB_TIP][1])
    return index_close and thumb_close and index_top


def detect_kicaw(hands_pts, w, h, mouth=None):
    """🐦 Kicaw: satu tangan menutup mulut, tangan satunya menjulur ke depan
    dengan jari-jari lurus terbuka.

    Jika `mouth` (mx, my, r) tersedia dari deteksi wajah, "tutup mulut" dicek
    akurat lewat jarak tangan ke titik mulut. Tanpa wajah, dipakai perkiraan
    posisi tangan di area atas-tengah frame.
    """
    if len(hands_pts) != 2:
        return False
    # uji kedua kemungkinan peran (tangan A=mulut/B=depan, lalu sebaliknya)
    for mhand, fwd in (hands_pts, hands_pts[::-1]):
        mcx = sum(p[0] for p in mhand) / len(mhand)
        mcy = sum(p[1] for p in mhand) / len(mhand)

        _, idx, mid, ring, pinky = fingers_extended(fwd)
        open_hand = idx and mid and ring and pinky

        fcx = sum(p[0] for p in fwd) / len(fwd)
        fcy = sum(p[1] for p in fwd) / len(fwd)

        if mouth is not None:
            mx, my, r = mouth
            near_mouth = dist((mcx, mcy), (mx, my)) < r
            fwd_clear = dist((fcx, fcy), (mx, my)) > 1.25 * r
            roles_ok = near_mouth and fwd_clear
        else:
            near_face = mcy < 0.5 * h and 0.20 * w < mcx < 0.80 * w
            separated = dist((fcx, fcy), (mcx, mcy)) > 0.75 * (
                palm_scale(mhand) + palm_scale(fwd))
            roles_ok = near_face and fcy > mcy and separated

        if roles_ok and open_hand:
            return True
    return False


class HandSmoother:
    """Haluskan koordinat landmark antar-frame (exponential moving average).

    Tangan dicocokkan ke frame sebelumnya berdasarkan posisi pergelangan
    terdekat, jadi smoothing tetap benar walau urutan tangan berubah.
    alpha besar = lebih responsif, alpha kecil = lebih halus.
    """

    def __init__(self, alpha=config.SMOOTH_ALPHA,
                 match_dist=config.SMOOTH_MATCH_DIST):
        self.alpha = alpha
        self.match_dist = match_dist
        self.prev = []  # list of list[(x,y) float]

    def update(self, hands):
        out = []
        used = [False] * len(self.prev)
        for pts in hands:
            wrist = pts[WRIST]
            best_d, bi = None, -1
            for i, pp in enumerate(self.prev):
                if used[i]:
                    continue
                d = dist(wrist, pp[WRIST])
                if best_d is None or d < best_d:
                    best_d, bi = d, i
            if bi >= 0 and best_d < self.match_dist:
                used[bi] = True
                a = self.alpha
                sm = [(a * nx + (1 - a) * ox, a * ny + (1 - a) * oy)
                      for (nx, ny), (ox, oy) in zip(pts, self.prev[bi])]
            else:
                sm = [(float(x), float(y)) for x, y in pts]
            out.append(sm)
        self.prev = out
        return [[(int(round(x)), int(round(y))) for x, y in pts] for pts in out]


class GestureDebouncer:
    """Gestur baru dianggap aktif hanya setelah bertahan stabil N frame.

    Bikin deteksi anti-kedip dan mencegah suara ter-spam. `update` menerima
    gestur mentah frame ini (boleh `None`) dan mengembalikan gestur yang
    benar-benar aktif — yang tetap bertahan sampai kandidat baru cukup stabil.
    """

    def __init__(self, hold_frames=config.HOLD_FRAMES):
        self.hold_frames = hold_frames
        self.candidate = None
        self.count = 0
        self.active = None

    def update(self, current):
        if current == self.candidate:
            self.count += 1
        else:
            self.candidate = current
            self.count = 1
        if self.count >= self.hold_frames:
            self.active = self.candidate  # boleh None: gestur dilepas
        return self.active

    def force(self, gesture):
        """Setel gestur aktif langsung, melewati hitungan stabil.

        Dipakai gestur yang punya penjaga waktunya sendiri (👌 OK), supaya
        tidak terkena dua penundaan berturut-turut.
        """
        self.candidate = gesture
        self.count = self.hold_frames
        self.active = gesture
