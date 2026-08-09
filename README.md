# 📸 Foto-Kita-Blurrr

Mainan kamera real-time: gestur tangan memicu efek, tulisan, animasi, dan suara —
plus jendela portal yang bisa kamu tarik pakai dua telunjuk.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks%20API-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## Apa ini

Kamera menyala, tanganmu jadi alat kontrol. Lima gestur dikenali dan masing-masing
punya adegannya sendiri: layar nge-blur dengan judul besar, hati melayang, konfeti,
not balok. Semua deteksi berjalan lokal di laptopmu — tidak ada apa pun yang dikirim
ke mana pun.

Ada **dua mode** yang bergantian lewat tombol `TAB`. **Mode Gestur** adalah aplikasi
aslinya: lima gestur, skeleton tangan berwarna, suara. **Mode Portal** mematikan
seluruh pengenalan gestur dan menggantinya dengan satu jendela persegi yang kamu
tarik di antara ujung telunjuk kiri dan kanan — isinya kena filter, tepinya lembut,
sudutnya berkerlip. Dipisah jadi dua mode justru supaya keduanya tidak pernah salah
kenal satu sama lain.

## ✋ Gestur

| Gestur | Nama | Efek |
|--------|------|------|
| ✌️ | Peace | **Filter aktif** dipasang ke seluruh layar + judul `FOTO KITA BLURRR` + suara |
| 🫶 | Finger Heart (dua tangan) | Tulisan **I LOVE YOU** + hati melayang |
| 🤟 | ILY / Rock | Tulisan **GOKILL** |
| 👌 | OK | Tulisan **OKE** + lingkaran berdenyut |
| 🐦 | Kicaw — satu tangan menutup mulut, tangan satunya menjulur dengan jari lurus | Suara `Kicaw Mania` + konfeti + not balok |

**👌 OK sekarang dipicu dengan menahan sentuhan jempol + telunjuk selama ~0,35 detik**,
bukan dari bentuk jari. Satu sentuhan memisahkan dua perintah lewat durasinya:

```
ujung jempol menyentuh ujung telunjuk
  ├─ dilepas SEBELUM 0,35 dtk   → ganti filter berikutnya
  └─ masih menyentuh DI 0,35 dtk → gestur 👌 OK muncul
```

Jadi sentuh-cepat = ganti filter, sentuh-tahan = OK. Kalau ambangnya terasa
terlalu sensitif atau terlalu susah, setel dengan `+` / `-` saat aplikasi jalan.

## 🎨 Filter

Nomornya sejajar dengan tombol `1`–`9`. Filter berlaku untuk ✌️ Peace (layar penuh)
maupun isi portal.

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

Pergantian filter selalu melalui crossfade singkat, tidak pernah berganti mendadak.

## 🌀 Mode Portal

Tekan `TAB`. Badge kiri-atas berubah jadi `PORTAL`. Angkat kedua tangan: sebuah
jendela terbentuk di antara kedua **ujung telunjuk** — satu telunjuk jadi sudut atas,
satunya sudut bawah diagonal, persis seperti menarik kotak seleksi. Isinya diproses
filter aktif, tepinya di-feather beberapa pixel, dikelilingi rim glow yang berdenyut
cyan ↔ violet dengan bracket sudut dan partikel energi.

Isi portal **tidak pernah di-warp perspektif** — wilayahnya di-crop, difilter apa
adanya, lalu ditempel kembali di posisi yang sama dan di-mask. Itu sebabnya wajah di
dalam portal tidak pernah tampak gepeng.

Turunkan tangan dan portal memudar, tidak hilang mendadak. Sentuh-cepat jempol +
telunjuk tetap berfungsi untuk mengganti filter di dalam portal.

## ⌨️ Kontrol

| Tombol | Fungsi |
|--------|--------|
| `TAB` | Ganti Mode Gestur ↔ Mode Portal |
| `1`–`9` | Lompat langsung ke filter tertentu |
| pinch cepat | Filter berikutnya (jempol + telunjuk, lepas < 0,35 dtk) |
| `+` / `-` | Setel sensitivitas pinch |
| `m` | Bisukan / nyalakan suara |
| `v` | Vignette nyala / mati |
| `h` | Tampilkan / sembunyikan hint kontrol |
| `s` | Simpan screenshot ke `shots/` (persis seperti yang terlihat) |
| `f` | Layar penuh |
| `q` / `ESC` | Keluar |

## 🚀 Menjalankan

1. (Disarankan) buat virtual environment:

   ```powershell
   python -m venv venv
   venv\Scripts\Activate.ps1
   ```

2. Pasang dependensi:

   ```powershell
   pip install -r requirements.txt
   ```

3. Jalankan:

   ```powershell
   python foto_kita_blurrr.py
   ```

**Model MediaPipe terunduh otomatis** saat pertama dijalankan, lengkap dengan
indikator persen — tidak ada langkah `curl` manual. Kalau unduhan gagal, aplikasi
mencetak perintah unduh manualnya untukmu.

## 📂 Struktur project

```
Foto-Kita- Blurrr/
├── foto_kita_blurrr.py    # entry point tipis
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
├── tests/                 # 113 test, tidak satu pun butuh webcam
├── assets/
│   ├── gestur/            # contoh foto gestur
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
| `FRAME_W` / `FRAME_H` | `1280` / `720` | Resolusi kamera yang diminta |
| `DETECT_WIDTH` | `640` | Deteksi di frame diperkecil — turunkan kalau berat |
| `EFFECT_SCALE` | `0.5` | Filter layar-penuh dihitung di separuh resolusi |
| `PINCH_HOLD_S` | `0.35` | Batas TAP ↔ HOLD |
| `PINCH_ENTER` / `PINCH_EXIT` | `0.055` / `0.075` | Ambang pinch (histeresis anti-getar) |
| `HOLD_FRAMES` | `6` | Frame stabil sebelum sebuah gestur dianggap aktif |
| `VIGNETTE` | `True` | Vignette saat mulai |
| `FULLSCREEN` | `False` | Mulai dalam layar penuh |

## 🧪 Test

```powershell
python -m pytest -q
```

113 test, semuanya berjalan **tanpa webcam**: landmark dibuat sintetis, jam
di-inject, audio pakai stub. Tidak ada `sleep` di seluruh suite.

## 🛠️ Catatan

- Butuh webcam aktif. Kalau tidak terbuka, aplikasi memberi pesan yang jelas
  beserta cara mengganti `CAM_INDEX`.
- Suara memakai file **WAV** di `assets/Sond/` karena pygame tidak andal memutar
  mp3. WAV dihasilkan dari mp3 asli dengan ffmpeg:

  ```powershell
  ffmpeg -y -i "assets/Sond/foto kita blur.mp3" -ar 44100 -ac 2 "assets/Sond/foto kita blur.wav"
  ffmpeg -y -i "assets/Sond/Kicaw Mania.mp3"   -ar 44100 -ac 2 "assets/Sond/Kicaw Mania.wav"
  ```

- Tanpa sound device — atau tanpa pygame sama sekali — program tetap jalan dengan
  efek visual saja.
- Model wajah (`blaze_face_short_range`) bersifat opsional dan membuat gestur Kicaw
  akurat: "menutup mulut" dicek lewat jarak ke titik mulut sebenarnya. Tanpa model
  itu Kicaw tetap jalan memakai perkiraan posisi tangan.
- Tanpa Pillow, HUD turun ke versi OpenCV sederhana. Aplikasi tetap berfungsi penuh.
- MediaPipe mendukung Python 3.9–3.12.

## 📄 Lisensi & kredit

MIT — lihat [LICENSE](LICENSE).

Mode Portal, sembilan filter gambar, dan HUD diadaptasi dari
[milan-kb/fancy-fingers](https://github.com/milan-kb/fancy-fingers) (MIT).
Terima kasih. 🙏
