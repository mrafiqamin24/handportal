# Foto-Kita-Blurrr × HandPortal — Desain Penggabungan

**Tanggal:** 2026-08-09
**Status:** Disetujui, siap masuk tahap perencanaan implementasi

---

## 1. Latar belakang

`Foto-Kita-Blurrr` adalah mainan kamera real-time: MediaPipe mendeteksi gestur
tangan, OpenCV menggambar efek dan pygame memutar suara. Kode saat ini satu file
`foto_kita_blurrr.py` (575 baris) dengan lima gestur — ✌️ Peace, 🫶 Heart,
🤟 ILY, 👌 OK, 🐦 Kicaw.

Repo referensi [`milan-kb/fancy-fingers`](https://github.com/milan-kb/fancy-fingers)
(HandPortal, `portal_app.py`, 1124 baris) adalah mainan sejenis dengan tumpukan
teknologi identik, tapi jauh lebih matang: model diunduh otomatis, delapan filter
gambar, "portal" ber-mask feather di antara dua ujung telunjuk, HUD ber-desain
memakai PIL, rim glow, partikel, vignette, fullscreen, screenshot, lisensi MIT,
dan README lengkap.

Tujuan pekerjaan ini: menyerap HandPortal ke dalam Foto-Kita-Blurrr secara penuh
— kualitas rekayasanya, filter gambarnya, **dan** fitur portalnya sebagai mode
kedua — tanpa merusak lima gestur yang sudah ada.

## 2. Ruang lingkup

**Termasuk:**

- Refactor dari satu file menjadi paket `app/` bermodul.
- Sembilan filter gambar (Blur bawaan + delapan dari HandPortal).
- Mode Portal sebagai mode kedua, dipilih lewat tombol `TAB`.
- Pinch (sentuh ujung jempol + telunjuk) sebagai kontrol utama ganti efek.
- Auto-download model MediaPipe, HUD, FPS, screenshot, fullscreen, vignette.
- Test pytest untuk logika murni, tanpa webcam.
- `git init`, `.gitignore`, `LICENSE` (MIT + atribusi), README ditulis ulang.

**Tidak termasuk:**

- Gestur baru selain yang sudah ada.
- Perekaman video, versi web/mobile, CI.
- Efek berbasis model tambahan (segmentasi, pose, dsb).

## 3. Keputusan desain

| # | Keputusan | Alasan |
|---|-----------|--------|
| 1 | Dua mode terpisah, di-toggle tombol `TAB` | Portal dan gestur sama-sama memakai dua tangan; memisahkannya menghilangkan salah-deteksi sepenuhnya |
| 2 | Kode dipecah jadi paket `app/` | ~1.500 baris dalam satu file terlalu berat dibaca dan mustahil di-test per bagian |
| 3 | Pinch dibedakan dari 👌 OK lewat **durasi** | Bentuk tangan keduanya identik; durasi adalah pembeda yang paling tidak menuntut ketelitian pengguna |
| 4 | ✌️ Peace memakai filter aktif, bukan blur mati | Menjadikan sembilan filter berguna di Mode Gestur tanpa menambah gestur baru |
| 5 | Test hanya untuk logika murni | Deteksi gestur dan filter gambar bisa diuji tanpa kamera; menguji render/kamera butuh perkakas yang tidak sepadan untuk project ini |

## 4. Arsitektur

```
Foto-Kita- Blurrr/
├── foto_kita_blurrr.py     # entry point tipis -> app.main:main
├── app/
│   ├── __init__.py
│   ├── config.py           # konstanta tunable + path aset
│   ├── models.py           # auto-download model, factory HandLandmarker/FaceDetector
│   ├── audio.py            # AudioPlayer (pygame, degrade dengan anggun)
│   ├── gestures.py         # helper landmark, classifier, HandSmoother, GestureDebouncer, PinchTapDetector
│   ├── effects.py          # sembilan filter gambar
│   ├── draw.py             # primitif gambar: teks beroutline, hati, not balok, skeleton, vignette
│   ├── scenes.py           # efek visual per-gestur
│   ├── portal.py           # box, komposit feather, rim glow, corner bracket, partikel energi
│   ├── hud.py              # panel PIL + fallback OpenCV
│   └── main.py             # loop kamera, mesin mode, keyboard
├── tests/
│   ├── conftest.py         # pembuat landmark sintetis + jam yang di-inject
│   ├── test_gestures.py
│   ├── test_pinch.py
│   ├── test_effects.py
│   └── test_portal.py
├── models/                 # di-gitignore, diunduh otomatis
├── assets/                 # gestur/ dan Sond/ (tidak berubah)
├── docs/superpowers/specs/
├── requirements.txt  LICENSE  .gitignore  README.md  PRD.md
```

### Batas antar-modul

Batas ini yang membuat mayoritas kode bisa di-test tanpa kamera:

- **`effects.py`** — murni `ndarray → ndarray`. Tidak tahu apa pun soal tangan,
  gestur, atau mode. Setiap fungsi mengembalikan array baru dan tidak pernah
  memutasi input.
- **`gestures.py`** — murni landmark → label. Tidak mengimpor apa pun untuk
  menggambar. Semua yang bergantung waktu menerima `now` sebagai argumen, tidak
  memanggil `time.time()` sendiri.
- **`draw.py`** — primitif menggambar. Tidak tahu soal gestur.
- **`scenes.py`** dan **`portal.py`** — lapisan yang merangkai ketiganya.
- **`main.py`** — satu-satunya tempat yang menyentuh kamera, keyboard, dan jam.

### Alur data per frame

```
Webcam
  └─ flip horizontal (mode cermin)
      └─ resize ke DETECT_WIDTH → HandLandmarker (+ FaceDetector opsional)
          ├─ landmark → HandSmoother (EMA, dicocokkan lewat pergelangan)
          │   ├─ PinchTapDetector  → TAP (efek berikutnya) / HOLD (gestur OK)
          │   └─ classifier gestur → GestureDebouncer → gestur aktif
          └─ Mode Gestur  → scenes.render(gestur aktif, filter aktif) + AudioPlayer
             Mode Portal  → portal.render(kotak antar-telunjuk, filter aktif)
                └─ HUD (badge mode, label efek, FPS, hint) → vignette → tampil
```

## 5. Perilaku

### 5.1 Mode Gestur (default)

Lima gestur lama tetap berlaku, dengan debounce stabil-6-frame (`HOLD_FRAMES = 6`,
nilai yang dipakai sekarang).

| Gestur | Efek |
|--------|------|
| ✌️ Peace | **Filter aktif diterapkan ke seluruh layar** + judul `FOTO KITA BLURRR` + thumbnail + suara `foto kita blur` |
| 🫶 Heart | Tulisan `I LOVE YOU` + partikel hati melayang |
| 🤟 ILY | Tulisan `GOKILL` + teks warna berputar |
| 👌 OK | Tulisan `OKE` + lingkaran berdenyut — dipicu oleh **pinch ditahan**, lihat §5.3 |
| 🐦 Kicaw | Suara `Kicaw Mania` + konfeti + not balok |

Perubahan dari perilaku sekarang: ✌️ Peace tidak lagi terkunci pada Gaussian blur.
Blur menjadi filter nomor 1 (tetap default, jadi tampilan awal identik dengan
sekarang), dan delapan filter lain bisa dipilih.

### 5.2 Mode Portal (`TAB`)

Portal ter-feather direntangkan di antara dua ujung telunjuk — tiap ujung jari
menjadi satu sudut diagonal dari kotak sejajar-sumbu. Isi kotak diganti versi
terfilter dari dirinya sendiri, dengan rim glow berdenyut cyan ↔ violet, corner
bracket ala AR, dan partikel energi dari tiap sudut.

Aturan render yang diwarisi dari HandPortal dan **wajib dipertahankan**: isi
portal tidak pernah di-*perspective warp*. Wilayah kotak di-crop, difilter lurus
tanpa distorsi, ditempel balik di posisi yang sama, lalu di-mask ke bentuk
portal. Kemiringan apa pun di masa depan ditangani lewat masking saja.

Lima gestur lama non-aktif di mode ini. Pinch tetap aktif untuk ganti efek.

### 5.3 Pinch: satu mesin-status, dua hasil

Ambang tunggal `PINCH_HOLD_S = 0.35` detik memisahkan dua hasil tanpa celah
maupun tumpang-tindih:

```
ujung jempol menyentuh ujung telunjuk
  ├─ dilepas SEBELUM 0,35 dtk        → TAP  → filter berikutnya
  └─ masih menyentuh TEPAT di 0,35 dtk → HOLD → gestur 👌 OK menjadi aktif
```

Aturan turunan:

- **👌 OK tidak lagi dikenali dari bentuk jari.** Syarat lama (jari tengah,
  manis, dan kelingking harus lurus) dibuang dari `classify_hand`. Dua penjaga
  yang berbeda justru membuat OK sulit keluar.
- **Untuk gestur OK, durasi tahan menggantikan debounce N-frame.** OK menyala
  tepat di 0,35 dtk. Menumpuk debounce di atasnya membuat total ~0,55 dtk dan
  terasa lamban. Gestur lain tetap memakai debounce seperti sekarang.
- TAP dan HOLD masing-masing memicu **tepat satu kali** per pinch (edge-trigger).
  Menahan lebih lama tidak memicu berulang; harus dilepas dulu untuk memicu lagi.
- Pinch di **dua tangan sekaligus** dihitung satu kali, bukan dua.
- Pinch berlaku di **kedua mode**, jadi interaksinya seragam. Di Mode Portal
  HOLD tidak melakukan apa-apa, karena gestur 👌 OK memang tidak aktif di sana.
- Ambang jarak pinch memakai histeresis (masuk lebih ketat daripada keluar) agar
  jitter landmark di sekitar ambang tidak memicu tap beruntun.

### 5.4 Filter

| # | Nama | Asal |
|---|------|------|
| 1 | Blur | Perilaku Peace yang sudah ada (default) |
| 2 | Thermal | HandPortal |
| 3 | Edge Mesh | HandPortal |
| 4 | Posterize Neon | HandPortal |
| 5 | Invert Glitch | HandPortal |
| 6 | Sketch | HandPortal |
| 7 | Chromatic | HandPortal |
| 8 | Pixel Mosaic | HandPortal |
| 9 | Duotone | HandPortal |

Berganti filter memicu crossfade 0,22 dtk dan label nama efek yang slide-in di
HUD, sehingga penyebab perubahan selalu terlihat.

### 5.5 Kontrol

| Tombol / gestur | Aksi |
|-----------------|------|
| **Pinch cepat** | Filter berikutnya |
| `TAB` | Ganti Mode Gestur ↔ Mode Portal |
| `1`–`9` | Lompat langsung ke filter tertentu (pintasan, bukan cara utama) |
| `f` | Fullscreen |
| `s` | Screenshot ke `shots/` |
| `v` | Vignette |
| `d` | Downscale deteksi |
| `m` | Bisukan audio |
| `+` / `-` | Sensitivitas pinch |
| `h` | Tampil/sembunyikan hint kontrol |
| `q` / `ESC` | Keluar |

### 5.6 HUD

Dirender dengan PIL (panel bersudut membulat, tipografi TTF), dengan fallback
OpenCV kalau Pillow tidak tersedia:

- Badge mode di kiri atas (`GESTUR` / `PORTAL`).
- Panel nama efek dengan animasi slide/fade saat berganti.
- FPS pill kecil di kanan atas.
- Hint kontrol yang memudar setelah 5 detik.

## 6. Penanganan kesalahan

Prinsipnya: setiap kegagalan menurunkan kemampuan aplikasi, bukan menghentikannya
— kecuali kamera, yang memang tidak ada gunanya dilanjutkan tanpanya.

| Kondisi | Perilaku |
|---------|----------|
| `hand_landmarker.task` tidak ada | Diunduh otomatis (~7 MB) dengan indikator progres; jika gagal, pesan jelas + perintah unduh manual, lalu keluar |
| `blaze_face_short_range.tflite` tidak ada | Tetap jalan; Kicaw memakai perkiraan posisi tangan (perilaku sekarang) |
| Kamera tidak bisa dibuka | Pesan jelas, keluar dengan kode ≠ 0 |
| Audio device tidak ada / pygame gagal | Jalan tanpa suara, satu peringatan di konsol |
| File suara / gambar aset hilang | Efek terkait dilewati, satu peringatan, tidak crash |
| Pillow tidak terpasang | HUD fallback OpenCV |

## 7. Strategi test

pytest, tanpa webcam. `tests/conftest.py` menyediakan pembuat 21 landmark
sintetis untuk pose peace, ok, ily, kepalan, dan telapak terbuka, plus jam palsu
yang di-inject agar test durasi tidak memakai `sleep`.

**`test_gestures.py`**
- Tiap pose sintetis menghasilkan label yang benar.
- Pose ambigu / setengah jadi menghasilkan `None`.
- 🫶 Heart terdeteksi dari dua tangan; satu tangan tidak pernah memicunya.
- 🐦 Kicaw terdeteksi untuk kedua pembagian peran tangan (mana pun yang menutup mulut).
- `GestureDebouncer` butuh 6 frame stabil sebelum mengaktifkan; satu frame menyimpang me-reset hitungan.
- `HandSmoother` mencocokkan tangan lewat pergelangan meski urutan tangan tertukar antar-frame.

**`test_pinch.py`** — unit paling padat test:
- Tap memicu tepat satu kali per pinch.
- Tahan lama tidak pernah memicu tap.
- Tahan memicu HOLD tepat satu kali, bukan tiap frame.
- Lepas lalu pinch lagi memicu lagi.
- Pinch dua tangan bersamaan menghasilkan satu pemicu, bukan dua.
- Jitter jarak di sekitar ambang tidak menghasilkan tap beruntun (histeresis).
- Tepat di batas 0,35 dtk hasilnya HOLD, bukan TAP.

**`test_effects.py`**
- Tiap `fx_*` menjaga shape dan dtype (`uint8`).
- Tiap `fx_*` tidak memutasi array input.
- Tiap `fx_*` aman untuk crop sangat kecil (1×1, 2×2).

**`test_portal.py`**
- Kotak dibangun benar dari dua titik, termasuk lantai ukuran minimum saat kedua
  ujung jari nyaris sejajar.
- Komposit tidak pernah menulis di luar batas frame saat portal sebagian keluar layar.

## 8. Kinerja

Filter layar-penuh pada 1280×720 setiap frame — terutama Posterize Neon yang
bolak-balik konversi HSV — berisiko menurunkan FPS. Mitigasi: filter dihitung
pada frame separuh resolusi lalu di-upscale. Untuk efek stilisasi selisih
kualitasnya nyaris tak terlihat, dan FPS pill di HUD membuat dampaknya langsung
terukur. Deteksi tangan sudah berjalan di frame yang diperkecil (`DETECT_WIDTH`)
dan itu dipertahankan.

## 9. Urutan implementasi

Pendekatannya *extract-then-extend*: setiap langkah menghasilkan aplikasi yang
masih bisa dijalankan.

1. `git init`, `.gitignore`, `LICENSE`, commit awal dari kondisi sekarang.
2. Ekstrak kode yang ada ke paket `app/` — perilaku tidak berubah sama sekali.
3. Tambahkan test untuk logika yang sudah diekstrak; jadikan hijau.
4. `PinchTapDetector` + perombakan gestur OK (§5.3), beserta test-nya.
5. `effects.py` + sembilan filter + crossfade; sambungkan ke ✌️ Peace.
6. Auto-download model, HUD, FPS, screenshot, fullscreen, vignette.
7. `portal.py` + Mode Portal + toggle `TAB`.
8. Tulis ulang README; rapikan `requirements.txt`.

## 10. Lisensi dan atribusi

`fancy-fingers` berlisensi MIT, jadi menyalin dan mengadaptasi kodenya sah selama
notice-nya dibawa serta. Project ini akan memakai `LICENSE` MIT yang memuat
copyright notice asli dari `milan-kb/fancy-fingers` di samping milik pemilik
project, dan README akan menyebut repo tersebut sebagai sumber mode Portal serta
filter gambarnya.
