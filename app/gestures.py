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


def handedness_labels(result):
    """Ambil label Left/Right dari hasil MediaPipe, sejajar dengan landmarks.

    Tasks API mengembalikan daftar kategori per tangan. Fungsi defensif ini
    menjaga loop kamera tetap aman pada hasil kosong atau versi API yang
    memakai atribut nama kategori berbeda.
    """
    labels = []
    for categories in getattr(result, "handedness", None) or []:
        category = categories[0] if categories else None
        name = (getattr(category, "category_name", None)
                or getattr(category, "display_name", None)) if category else None
        score = getattr(category, "score", None) if category else None
        labels.append(name if name and (score is None or score >=
                                        config.HANDEDNESS_MIN_CONFIDENCE)
                      else None)
    return labels


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
DoublePinchEvent = collections.namedtuple("DoublePinchEvent", "triggered pinching")


def pinch_distance(pts):
    """Jarak ujung jempol <-> ujung telunjuk, dinormalkan ke ukuran telapak.

    Dinormalkan supaya ambangnya tetap benar saat tangan mendekat atau menjauh
    dari kamera.
    """
    return dist(pts[THUMB_TIP], pts[INDEX_TIP]) / palm_scale(pts)


def is_ok_pose(pts, max_distance=config.PINCH_EXIT):
    """True hanya untuk bentuk 👌, bukan genggaman dengan ujung jari berimpit.

    Sentuhan jempol-telunjuk tetap menjadi pemicu temporal, tetapi tiga jari
    lainnya wajib terbuka dan telunjuk tidak boleh lurus penuh.
    """
    _, index, middle, ring, pinky = fingers_extended(pts)
    return (not index and middle and ring and pinky
            and pinch_distance(pts) <= max_distance)


class PinchTapDetector:
    """Satu mesin-status untuk kedua tangan sekaligus.

        sentuh lalu lepas SEBELUM hold_s  -> TAP  (tidak dipakai Mode Gestur)
        sentuh dan bertahan DI hold_s     -> HOLD (gestur OK)

    Masing-masing memicu tepat sekali per pinch. Dua ambang jarak berbeda
    (masuk lebih ketat daripada keluar) meredam jitter landmark di sekitar
    ambang. Kedua tangan berbagi satu status, jadi pinch bersamaan tetap
    dihitung satu kali.
    """

    def __init__(self, hold_s=config.PINCH_HOLD_S, enter=config.PINCH_ENTER,
                 exit=config.PINCH_EXIT,
                 missing_grace=config.PINCH_MISSING_GRACE):
        self.hold_s = hold_s
        self.enter = enter
        self.exit = exit
        self.missing_grace = missing_grace
        self._pinching = False
        self._start = 0.0
        self._held = False
        self._missing = 0
        self._last_d = float("inf")

    def adjust(self, delta):
        """Geser sensitivitas; selisih histeresis dipertahankan."""
        gap = self.exit - self.enter
        self.enter = max(config.PINCH_MIN,
                         min(config.PINCH_MAX, self.enter + delta))
        self.exit = self.enter + gap

    def reset(self):
        self._pinching = False
        self._start = 0.0
        self._held = False
        self._missing = 0
        self._last_d = float("inf")

    def update(self, dists, now):
        """`dists` = jarak pinch ternormalisasi tiap tangan; boleh kosong.

        Daftar kosong berarti tak satu pun tangan sedang berpose 👌 pada frame
        ini. Itu sering hanya kedipan MediaPipe, bukan jari yang benar-benar
        dilepas, jadi jarak terakhir dipertahankan selama `missing_grace` frame
        agar hitungan tahan 0,35 detik tidak selalu mulai dari nol.
        """
        if dists:
            self._last_d = min(dists)
            self._missing = 0
        elif self._pinching and self._missing < self.missing_grace:
            self._missing += 1
        else:
            self._last_d = float("inf")
        d = self._last_d
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


class DoublePinchDetector:
    """Gesture empat-jari untuk mengganti filter di Mode Portal.

    Kedua pasangan thumb-index harus sama-sama masuk ambang selama beberapa
    frame. Setelah memicu, detector baru siap lagi ketika kedua pasangan sudah
    benar-benar terbuka. Ini mencegah satu tangan, jitter, atau dropout detector
    mengganti filter berulang kali.
    """

    def __init__(self, enter=config.PINCH_ENTER, exit=config.PINCH_EXIT,
                 stable_frames=config.DOUBLE_PINCH_STABLE_FRAMES,
                 release_frames=config.DOUBLE_PINCH_RELEASE_FRAMES):
        self.enter = enter
        self.exit = exit
        self.stable_frames = stable_frames
        self.release_frames = release_frames
        self._armed = True
        self._close_count = 0
        self._release_count = 0

    def adjust(self, delta):
        gap = self.exit - self.enter
        self.enter = max(config.PINCH_MIN,
                         min(config.PINCH_MAX, self.enter + delta))
        self.exit = self.enter + gap

    def reset(self):
        self._armed = True
        self._close_count = 0
        self._release_count = 0

    def update(self, dists):
        exactly_two = len(dists) == 2
        both_closed = exactly_two and max(dists) < self.enter
        both_open = exactly_two and min(dists) > self.exit
        triggered = False

        if self._armed:
            self._close_count = self._close_count + 1 if both_closed else 0
            if self._close_count >= self.stable_frames:
                self._armed = False
                self._close_count = 0
                triggered = True
        else:
            # Kehilangan tangan tidak langsung dianggap release: tunggu dua
            # tangan terlihat dan terbuka agar dropout tak memicu ulang.
            self._release_count = self._release_count + 1 if both_open else 0
            if self._release_count >= self.release_frames:
                self._armed = True
                self._release_count = 0

        return DoublePinchEvent(triggered, both_closed)


def detect_two_hand_heart(hands_pts, w=None):
    """🫶 Heart: dua tangan, ujung telunjuk bertemu & ujung jempol berdekatan.

    Ambang di sini diukur dari foto hati asli di `assets/gestur/Love.jpg`, bukan
    dikira-kira. Pada pose itu kedua tangan dipegang mendatar, jadi jempol hanya
    sedikit di bawah telunjuk (0,15 telapak) dan bentang jempol-telunjuk tiap
    tangan cuma sekitar 0,4 telapak. Aturan lama menuntut 0,55 dan 0,90, dan
    itulah yang membuat hati asli tidak pernah terdeteksi.

    Pemisah sesungguhnya adalah `index_gap`: 0,06 telapak pada hati versus 2,34
    pada dua tangan ✌️ Peace — beda 40 kali lipat. Syarat lain tinggal menolak
    pose yang runtuh (empat ujung menggumpal) dan tangan terbalik.
    """
    if len(hands_pts) != 2:
        return False
    a, b = hands_pts
    scale = (palm_scale(a) + palm_scale(b)) * 0.5
    index_gap = dist(a[INDEX_TIP], b[INDEX_TIP])
    thumb_gap = dist(a[THUMB_TIP], b[THUMB_TIP])
    index_mid_y = (a[INDEX_TIP][1] + b[INDEX_TIP][1]) * 0.5
    thumb_mid_y = (a[THUMB_TIP][1] + b[THUMB_TIP][1]) * 0.5

    # Kedua pasangan ujung jari bertemu membentuk dua sudut hati.
    tips_meet = index_gap < 0.60 * scale and thumb_gap < 1.30 * scale
    # Tiap tangan tetap membuka: menolak dua tangan yang sekadar mengepal atau
    # sama-sama mencubit sehingga keempat ujungnya menggumpal jadi satu titik.
    hands_open = (dist(a[INDEX_TIP], a[THUMB_TIP]) > 0.30 * scale
                  and dist(b[INDEX_TIP], b[THUMB_TIP]) > 0.30 * scale)
    # Telunjuk membentuk lekuk atas hati, jadi tidak boleh jatuh di bawah jempol.
    index_on_top = (thumb_mid_y - index_mid_y) > -0.35 * scale
    wrists_apart = dist(a[WRIST], b[WRIST]) > 1.50 * scale
    return tips_meet and hands_open and index_on_top and wrists_apart


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

    def __init__(self, alpha=None, match_dist=config.SMOOTH_MATCH_DIST,
                 missing_grace=config.HAND_MISSING_GRACE):
        self.alpha = alpha
        self.match_dist = match_dist
        self.missing_grace = missing_grace
        self.prev = []  # list of list[(x,y) float]
        self.prev_labels = []
        self.prev_missing = []

    def _alpha_for(self, pts, previous, landmark=WRIST):
        if self.alpha is not None:
            return self.alpha
        # Setiap sendi bergerak dengan kecepatannya sendiri. Memakai wrist saja
        # membuat ujung jari tertinggal ketika pengguna membentuk gestur tanpa
        # menggeser telapak; landmark cepat kini mendapat alpha lebih besar.
        motion = dist(pts[landmark], previous[landmark]) / palm_scale(pts)
        span = max(config.SMOOTH_MOTION_HIGH - config.SMOOTH_MOTION_LOW, 1e-6)
        p = max(0.0, min(1.0, (motion - config.SMOOTH_MOTION_LOW) / span))
        return (config.SMOOTH_ALPHA_MIN
                + (config.SMOOTH_ALPHA_MAX - config.SMOOTH_ALPHA_MIN) * p)

    def reset(self):
        """Lupakan semua trek, misalnya setelah sumber kamera berubah."""
        self.prev = []
        self.prev_labels = []
        self.prev_missing = []

    @property
    def labels(self):
        """Label handedness yang sejajar dengan hasil `update` terakhir."""
        return list(self.prev_labels)

    @property
    def missing_counts(self):
        """0 berarti landmark terlihat pada frame terbaru, >0 adalah bridge."""
        return list(self.prev_missing)

    def update(self, hands, labels=None):
        labels = list(labels or [])
        if len(labels) < len(hands):
            labels.extend([None] * (len(hands) - len(labels)))
        out = []
        out_labels = []
        out_missing = []
        used = [False] * len(self.prev)
        for pts, label in zip(hands, labels):
            wrist = pts[WRIST]
            best_d, bi = None, -1
            for i, pp in enumerate(self.prev):
                if used[i]:
                    continue
                prev_label = self.prev_labels[i]
                if label and prev_label and label != prev_label:
                    continue
                d = dist(wrist, pp[WRIST])
                if best_d is None or d < best_d:
                    best_d, bi = d, i
            if bi >= 0 and best_d < self.match_dist:
                used[bi] = True
                sm = []
                for landmark, ((nx, ny), (ox, oy)) in enumerate(
                        zip(pts, self.prev[bi])):
                    a = self._alpha_for(pts, self.prev[bi], landmark)
                    sm.append((a * nx + (1 - a) * ox,
                               a * ny + (1 - a) * oy))
            else:
                sm = [(float(x), float(y)) for x, y in pts]
            out.append(sm)
            out_labels.append(label)
            out_missing.append(0)

        # Dropout satu-dua frame umum saat tangan bergerak cepat. Menahan trek
        # sebentar mencegah skeleton/portal berkedip dan pinch terlepas palsu.
        for i, previous in enumerate(self.prev):
            if len(out) >= 2:
                break
            if used[i]:
                continue
            missing = self.prev_missing[i] + 1
            if missing <= self.missing_grace:
                out.append(previous)
                out_labels.append(self.prev_labels[i])
                out_missing.append(missing)

        self.prev = out
        self.prev_labels = out_labels
        self.prev_missing = out_missing
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
