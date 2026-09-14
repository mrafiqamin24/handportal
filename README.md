# 📸 HandPortal

Mainan kamera real-time: gestur tangan memicu tulisan, animasi, dan suara —
plus jendela portal empat-jari yang dibentuk oleh dua telunjuk dan dua jempol.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks%20API-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## Apa ini

Kamera menyala, tanganmu jadi alat kontrol. Lima gestur dikenali dan masing-masing
punya adegannya sendiri: judul besar, hati melayang, konfeti,
not balok. Semua deteksi berjalan lokal di laptopmu — tidak ada apa pun yang dikirim
ke mana pun.

Ada **dua mode** yang bergantian lewat switch HUD yang bisa diklik atau tombol
`TAB`. **Mode Gestur** adalah aplikasi
aslinya: lima gestur, skeleton tangan berwarna, dan suara. Efek portal tidak
pernah masuk ke mode ini; hanya gestur ✌️ Peace yang memakai blur bawaannya.
**Mode Portal** mematikan pengenalan gestur adegan dan memakai telunjuk + jempol
kedua tangan sebagai empat sudut portal. Filter hanya bekerja di dalam portal.
Pemisahan ini mencegah pose portal salah dikenali sebagai gestur biasa.

## ✋ Gestur

| Gestur | Nama | Efek |
|--------|------|------|
| ✌️ | Peace | Blur kamera + judul `FOTO KITA BLURRR` + suara |
| 🫶 | Finger Heart (dua tangan) | Tulisan **I LOVE YOU** + hati melayang |
| 🤟 | ILY / Rock | Tulisan **GOKILL** |
| 👌 | OK | Tulisan **OKE** + lingkaran berdenyut |
| 🐦 | Kicaw — satu tangan menutup mulut, tangan satunya menjulur dengan jari lurus | Suara `Kicaw Mania` + konfeti + not balok |

> **Catatan jujur soal 🐦 Kicaw.** Kalau tangan yang menutup mulut menempel rapat
> ke wajah, MediaPipe sering tidak menemukannya sama sekali — sudah diuji pada
> confidence 0,20–0,58, resolusi deteksi 768 dan 1280, gambar asli maupun
> dibalik, `num_hands` 2 maupun 4, serta crop ketat di sekitar mulut; hasilnya
> tetap satu tangan. Kicaw sengaja **tetap menuntut dua tangan** supaya telapak
> terbuka biasa tidak ikut memicunya. Agar terdeteksi, tahan tangan mulut sedikit
> renggang dari wajah sehingga siluetnya masih terlihat kamera.

**👌 OK dipicu dengan menahan sentuhan jempol + telunjuk satu tangan selama
~0,35 detik**, bukan dari bentuk jari.

```
Mode Gestur: satu thumb-index ditahan 0,35 dtk → gestur 👌 OK
Mode Portal: thumb-index KEDUA tangan dirapatkan bersama → filter berikutnya
```

Kalau ambangnya terasa terlalu sensitif atau terlalu susah, setel dengan `+` / `-`
saat aplikasi berjalan.

## 🎨 Filter

Sembilan filter pertama bisa dilompati langsung dengan tombol `1`–`9`. Seluruh
13 filter bisa diputar berurutan dengan `]` / `[` atau dengan double-pinch —
itulah satu-satunya cara mencapai empat filter terakhir. Filter-filter ini
**hanya berlaku pada isi portal**; Mode Gestur hanya memiliki blur bawaan
khusus ✌️ Peace.

| # | Nama | Tampilan |
|---|------|----------|
| 1 | Blur | Blur Gaussian — tampilan Peace yang asli, ini bawaannya |
| 2 | Thermal | Luminance dipetakan ke colormap panas |
| 3 | Edge Mesh | Hanya garis tepi putih di atas dasar gelap |
| 4 | Posterize Neon | Warna dikuantisasi, saturasi didorong |
| 5 | Invert Glitch | Warna dibalik + kanal digeser |
| 6 | Sketch | Sketsa pensil hitam-putih |
| 7 | Chromatic | Aberasi kromatik, warna terpisah di tepi |
| 8 | Pixel Mosaic | Kotak-kotak besar dengan garis nat |
| 9 | Duotone | Gradasi dua warna: indigo → cyan |
| 10 | Night Vision | Kontras hijau dengan scanline lembut |
| 11 | VHS | Tint analog, scanline, dan kanal warna bergeser |
| 12 | Comic Ink | Warna komik terkuantisasi dengan kontur tinta |
| 13 | Emboss Chrome | Relief luminance bernuansa metalik |

Pergantian filter selalu melalui crossfade singkat, tidak pernah berganti mendadak.

## 🌀 Mode Portal

Klik `PORTAL` pada switch kiri-atas atau tekan `TAB`. Hadapkan dua telapak ke kamera
dan bentuk huruf **L** dengan telunjuk + jempol pada masing-masing tangan. Empat
ujung jari itu menjadi empat sudut portal, mengikuti bentuk quadrilateral seperti
foto TikTok referensi. Pose dikunci setelah hanya dua frame valid (sekitar 67 ms
pada 30 FPS). Setelah aktif, validasi menjadi lebih toleran dan bentuk terakhir
ditahan 0,45 detik saat landmark terganggu, sehingga portal tidak berkedip.

Setiap sudut mempertahankan identitas jarinya. Putar salah satu tangan sehingga
telunjuk dan jempolnya bertukar posisi atas/bawah untuk memelintir portal menjadi
bentuk silang/X seperti contoh. Convex hull hanya dipakai untuk memvalidasi luas,
bukan mengurutkan ulang sudut. Portal membaca landmark MediaPipe terbaru langsung,
lalu memakai satu One-Euro filter: halus saat diam dan cepat mengikuti gerakan.

Isi portal **tidak pernah di-warp perspektif** — wilayahnya di-crop, difilter apa
adanya, lalu ditempel kembali di posisi yang sama dan di-mask. Itu sebabnya wajah di
dalam portal tidak pernah tampak gepeng.

### Kalau portal tidak mau muncul

Tekan `d`. HUD dan konsol akan menyebutkan syarat pertama yang belum terpenuhi
beserta angkanya — misalnya `bentang jempol-telunjuk tangan-2 kurang lebar: 0.94
< 1.2` atau `dua tangan terlalu rapat: 1.31 < 1.5`. Angka itulah yang dipakai
untuk menyetel `PORTAL_*` di [`app/config.py`](app/config.py), sehingga
penyetelannya berdasar ukuran, bukan tebakan. Diagnosa memeriksa syarat yang
sama persis dengan validator sungguhan, dan kesetaraan keduanya dijaga test.

Turunkan tangan dan portal memudar, tidak hilang mendadak. Untuk mengganti filter,
rapatkan **kedua pasangan** jempol–telunjuk secara bersamaan. Tahan sebentar sampai
HUD menampilkan nama filter, lalu buka kembali kedua tangan sebelum mengulang.

## ⌨️ Kontrol

| Tombol | Fungsi |
|--------|--------|
| klik `GESTUR` / `PORTAL` atau `TAB` | Pilih Mode Gestur ↔ Mode Portal |
| klik `KAMERA` atau `c` | Pindah ke kamera berikutnya; kamera aktif dipertahankan jika kandidat gagal |
| `1`–`9` | Lompat ke sembilan filter pertama, hanya di Mode Portal |
| `]` / `[` | Filter berikutnya / sebelumnya — menjangkau seluruh 13 filter |
| double-pinch | Filter berikutnya: thumb-index kiri dan kanan dirapatkan bersama |
| `+` / `-` | Setel sensitivitas pinch |
| `m` | Bisukan / nyalakan suara |
| `d` | Diagnosa portal: sebutkan syarat mana yang belum terpenuhi |
| `v` | Vignette nyala / mati |
| `h` | Tampilkan / sembunyikan hint kontrol |
| `s` | Simpan screenshot ke `shots/` (persis seperti yang terlihat) |
| `f` | Layar penuh |
| `q` / `ESC` | Keluar |

## 🚀 Cara pasang

Butuh **Python 3.9–3.12** (batas dukungan MediaPipe) dan sebuah webcam.

1. Ambil kodenya:

   ```bash
   git clone https://github.com/mrafiqamin24/handportal.git
   cd handportal
   ```

2. (Disarankan) buat virtual environment:

   ```powershell
   # Windows (PowerShell)
   python -m venv venv
   venv\Scripts\Activate.ps1
   ```

   ```bash
   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Pasang dependensi:

   ```bash
   pip install -r requirements.txt
   ```

4. Siapkan suara (opsional). pygame tidak andal memutar mp3, jadi aplikasi memakai
   file WAV yang tidak ikut di repo. Buat sekali dengan [ffmpeg](https://ffmpeg.org/):

   ```bash
   ffmpeg -y -i "assets/Sond/foto kita blur.mp3" -ar 44100 -ac 2 "assets/Sond/foto kita blur.wav"
   ffmpeg -y -i "assets/Sond/Kicaw Mania.mp3" -ar 44100 -ac 2 "assets/Sond/Kicaw Mania.wav"
   ```

   Tanpa langkah ini aplikasi tetap jalan, hanya tanpa suara.

5. Jalankan:

   ```bash
   python handportal.py
   ```

**Model MediaPipe terunduh otomatis** saat pertama dijalankan, lengkap dengan
indikator persen — tidak ada langkah `curl` manual. Kalau unduhan gagal, aplikasi
mencetak perintah unduh manualnya untukmu.

## 📂 Struktur project

```
handportal/
├── handportal.py          # entry point tipis
├── app/
│   ├── config.py          # semua angka yang bisa disetel
│   ├── gestures.py        # landmark -> label gestur, pinch, smoothing (murni)
│   ├── effects.py         # sembilan filter gambar (murni)
│   ├── draw.py            # primitif gambar: teks, hati, not, skeleton, vignette
│   ├── scenes.py          # adegan per-gestur
│   ├── portal.py          # geometri portal, mask ter-feather, glow, partikel
│   ├── hud.py             # panel HUD (Pillow, dengan fallback OpenCV)
│   ├── audio.py           # pemutar suara yang selalu menurunkan kemampuan
│   ├── models.py          # pemuatan + unduhan otomatis model
│   └── main.py            # loop kamera, mesin mode, keyboard
├── tests/                 # 199 test, tidak satu pun butuh webcam
├── assets/
│   ├── gestur/            # foto contoh gestur: tidak disertakan (lihat Test)
│   └── Sond/              # file suara
├── models/                # terisi sendiri saat pertama jalan
└── shots/                 # hasil screenshot
```

Hanya `app/main.py` yang menyentuh kamera, jam, dan keyboard. Sisanya bisa di-test
tanpa perangkat keras apa pun.

## ⚙️ Konfigurasi

Semuanya ada di [`app/config.py`](app/config.py).

| Parameter | Bawaan | Guna |
|-----------|--------|------|
| `CAM_INDEX` | `0` | Ganti kalau webcam-mu bukan yang pertama |
| `CAMERA_SCAN_MAX` | `5` | Indeks tertinggi yang dicoba oleh switch kamera |
| `FRAME_W` / `FRAME_H` | `1280` / `720` | Resolusi kamera yang diminta |
| `DETECT_WIDTH` | `768` | Resolusi input deteksi; detail fingertip lebih akurat |
| `EFFECT_SCALE` | `0.5` | Filter layar-penuh dihitung di separuh resolusi |
| `PORTAL_EFFECT_SCALE` | `0.5` | Resolusi kerja efek portal untuk mencegah frame drop |
| `FACE_DETECT_INTERVAL_S` | `0.10` | Detector wajah sekunder berjalan 10 Hz; tangan tetap setiap frame |
| `HAND_*_CONFIDENCE` | `0.52–0.58` | Ambang deteksi, presence, dan tracking MediaPipe |
| `HANDEDNESS_MIN_CONFIDENCE` | `0.65` | Label kiri/kanan di bawah ini diabaikan |
| `PINCH_HOLD_S` | `0.35` | Lama pinch satu tangan untuk gestur OK |
| `PINCH_ENTER` / `PINCH_EXIT` | `0.50` / `0.72` | Ambang pinch realistis + histeresis anti-getar |
| `PINCH_MISSING_GRACE` | `3` | Frame tanpa pose 👌 yang dimaafkan sebelum pinch dianggap lepas |
| `DOUBLE_PINCH_*_FRAMES` | `3 / 3` | Stabilitas masuk dan release pergantian filter |
| `PORTAL_ACQUIRE_FRAMES` | `2` | Konfirmasi cepat sebelum portal muncul |
| `PORTAL_LOST_GRACE_S` | `0.45` | Waktu menahan bentuk terakhir saat landmark hilang |
| `PORTAL_FADE_IN/OUT_S` | `0.07 / 0.18` | Masuk cepat, keluar tetap halus |
| `HOLD_FRAMES` | `5` | Frame stabil sebelum sebuah gestur dianggap aktif |
| `SMOOTH_ALPHA_MIN/MAX` | `0.30 / 0.78` | Smoothing adaptif: halus saat diam, responsif saat bergerak |
| `HAND_MISSING_GRACE` | `2` | Frame dropout yang dijembatani agar tangan tidak berkedip |
| `VIGNETTE` | `True` | Vignette saat mulai |
| `FULLSCREEN` | `False` | Mulai dalam layar penuh |

## 🧪 Test

```powershell
python -m pytest -q
```

199 test, semuanya berjalan **tanpa webcam**: landmark dibuat sintetis, jam
di-inject, audio pakai stub. Tidak ada `sleep` di seluruh suite.

Termasuk di dalamnya `tests/test_reference_photos.py`, yang memakai landmark
asli tiga foto gestur — sudah dibekukan oleh MediaPipe ke
`tests/fixtures/reference_landmarks.json`, jadi test tetap murni dan cepat.
Foto aslinya sengaja tidak disertakan di repo; test tidak membutuhkannya.
Test itu adalah penjaga agar penyetelan ambang tidak pernah lagi diam-diam
mematikan gestur yang jelas-jelas terlihat di foto.

## 🛠️ Catatan

- Butuh webcam aktif. Kalau tidak terbuka, aplikasi memberi pesan yang jelas
  beserta cara mengganti `CAM_INDEX`.
- Suara memakai file **WAV** di `assets/Sond/` karena pygame tidak andal memutar
  mp3; cara membuatnya ada di langkah 4 [Cara pasang](#-cara-pasang).
- Thumbnail kecil di pojok layar saat ✌️ Peace diambil dari
  `assets/gestur/Fotokitablurr.jpg`. Berkas itu tidak disertakan; taruh foto 16:9
  milikmu dengan nama tersebut kalau ingin thumbnail tampil. Tanpa berkas itu
  aplikasi tetap jalan.

- Tanpa sound device — atau tanpa pygame sama sekali — program tetap jalan dengan
  efek visual saja.
- Model wajah (`blaze_face_short_range`) bersifat opsional dan membuat gestur Kicaw
  akurat: "menutup mulut" dicek lewat jarak ke titik mulut sebenarnya. Tanpa model
  itu Kicaw tetap jalan memakai perkiraan posisi tangan.
- Tanpa Pillow, HUD turun ke versi OpenCV sederhana. Aplikasi tetap berfungsi penuh.
- MediaPipe mendukung Python 3.9–3.12.

## 🔬 Dasar teknis peningkatan

- Konfigurasi confidence, tracking mode, handedness, dan timestamp mengikuti
  [panduan resmi Hand Landmarker Python](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python)
  serta [implementasi Tasks API](https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/python/vision/hand_landmarker.py).
- Identitas `Left`/`Right` dipakai saat mencocokkan trek agar tangan tidak saling
  tertukar ketika menyilang. Validasi jari menggabungkan jarak, rasio panjang, dan
  sudut sendi PIP/DIP untuk menolak jari tertekuk.
- Ambang 🫶 Heart **diukur dari foto asli**, bukan dikira-kira. Pada hati
  dua-tangan yang dipegang mendatar, jempol cuma 0,15 telapak di bawah telunjuk
  dan bentang jempol–telunjuk tiap tangan sekitar 0,40 telapak. Aturan lama
  menuntut 0,55 dan 0,90, jadi hati asli tidak pernah menyala. Pemisah yang
  sebenarnya adalah jarak antar-ujung telunjuk: 0,06 telapak pada hati versus
  2,34 pada dua tangan ✌️ Peace — beda 40 kali lipat, jauh lebih tegas daripada
  syarat bentuk yang dipakai sebelumnya.
- `PinchTapDetector` memaafkan beberapa frame tanpa pose 👌 (`PINCH_MISSING_GRACE`).
  Sebelumnya satu frame yang gagal diklasifikasi me-reset hitungan 0,35 detik dari
  nol, sehingga OK menuntut belasan frame sempurna berturut-turut.
- Portal memerlukan dua tangan, dua telunjuk lurus, dua bentang thumb-index yang
  cukup lebar, jarak tangan minimum, hull validasi empat titik, luas minimum, dan
  stabilitas beberapa frame. Identitas fingertip dipertahankan agar twist/X
  tidak diurutkan ulang menjadi hull. Sudut portal kemudian dihaluskan dengan
  [One-Euro Filter](https://gery.casiez.net/1euro/).
- Normalisasi landmark dan history/stabilization mengacu pada pola yang digunakan
  [kinivi/hand-gesture-recognition-mediapipe](https://github.com/kinivi/hand-gesture-recognition-mediapipe),
  sedangkan portal, pinch state machine, dan crossfade berangkat dari
  [milan-kb/fancy-fingers](https://github.com/milan-kb/fancy-fingers).
- Switch kamera memakai `VideoCapture` baru sebagai kandidat dan baru melepas
  kamera aktif setelah kandidat berhasil dibuka, sehingga kegagalan pindah kamera
  tidak membuat layar kosong.

## 📄 Lisensi & kredit

MIT — lihat [LICENSE](LICENSE).

Mode Portal, sembilan filter gambar, dan HUD diadaptasi dari
[milan-kb/fancy-fingers](https://github.com/milan-kb/fancy-fingers) (MIT).
Terima kasih. 🙏
