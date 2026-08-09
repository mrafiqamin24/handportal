# 📸 Foto-Kita-Blurrr

Project main-main pakai Python: kamera mendeteksi **gestur tangan real-time**,
lalu kasih efek lucu di layar lengkap dengan **frame yang menempel di tangan**,
tulisan, animasi, dan suara. 🎉

## ✋ Gestur yang dikenali

| Gestur | Nama | Efek |
|--------|------|------|
| ✌️ | Peace | Layar **nge-blur** + judul `FOTO KITA BLURRR` + suara `foto kita blur.mp3` |
| 🫶 | Finger Heart (dua tangan) | Tulisan + animasi **I LOVE YOU** (hati melayang) |
| 🤟 | ILY / Rock | Tulisan + animasi **GOKILL** |
| 👌 | OK | Tulisan + animasi **OKE** |
| 🐦 | Kicaw — **satu tangan menutup mulut**, tangan satunya **menjulur ke depan dengan jari lurus** | Memutar suara `Kicaw Mania` + konfeti meriah |

> Contoh foto gestur ada di `assets/gestur/`. Khusus Kicaw: jika model wajah
> tersedia, "menutup mulut" dicek **akurat** lewat jarak tangan ke titik mulut;
> tanpa model wajah, dipakai perkiraan posisi tangan di **area atas-tengah**.
> Tangan satunya harus **telapak terbuka** (jari lurus) dan menjauh dari mulut.

> Tampilan dibuat **clean**: tidak ada teks instruksi di layar, dan tangan
> digambar sebagai **tulang/skeleton jari** (bukan bingkai kotak). Deteksi
> dijalankan di frame yang diperkecil + **smoothing** landmark, jadi lebih
> responsif dan tidak getar.

## 📂 Struktur

```
Foto-Kita- Blurrr/
├── foto_kita_blurrr.py      # aplikasi utama
├── requirements.txt          # dependensi
├── PRD.md                    # deskripsi project
├── README.md
├── models/                   # model MediaPipe (hand_landmarker.task,
│                             #   blaze_face_short_range.tflite opsional)
└── assets/
    ├── gestur/               # contoh foto gestur (Fotokitablurr, Love, Kicaw)
    └── Sond/                 # file suara (.mp3 asli + .wav hasil konversi)
```

## 🚀 Cara menjalankan

1. (Disarankan) buat virtual environment:

   ```powershell
   python -m venv venv
   venv\Scripts\Activate.ps1
   ```

2. Pasang dependensi:

   ```powershell
   pip install -r requirements.txt
   ```

3. Download model tangan MediaPipe (sekali saja) ke folder `models/`:

   ```powershell
   curl -L -o models/hand_landmarker.task `
     https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
   ```

   **(Opsional, disarankan)** model wajah agar gestur **Kicaw** ("tutup mulut")
   akurat. Tanpa ini Kicaw tetap jalan, hanya memakai perkiraan posisi tangan:

   ```powershell
   curl -L -o models/blaze_face_short_range.tflite `
     https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite
   ```

4. Jalankan:

   ```powershell
   python foto_kita_blurrr.py
   ```

5. Arahkan tangan ke kamera dan coba gestur di atas.
   Tekan **`Q`** atau **`ESC`** untuk keluar.

## 🛠️ Catatan

- Butuh **webcam** yang aktif. Kalau pakai kamera lain, ubah index di
  `cv2.VideoCapture(0, ...)` pada `foto_kita_blurrr.py`.
- Suara butuh `pygame`. Kalau audio tidak ada device, program tetap jalan
  (efek visual saja) dan menampilkan peringatan.
- File suara dipakai dalam format **WAV** (`assets/Sond/*.wav`) karena pygame
  tidak andal memutar mp3/m4a. WAV dihasilkan dari mp3 asli memakai ffmpeg:

  ```powershell
  ffmpeg -y -i "assets/Sond/foto kita blur.mp3" -ar 44100 -ac 2 "assets/Sond/foto kita blur.wav"
  ffmpeg -y -i "assets/Sond/Kicaw Mania.mp3"   -ar 44100 -ac 2 "assets/Sond/Kicaw Mania.wav"
  ```

- Aplikasi memakai **MediaPipe Tasks API** (`HandLandmarker`), jadi butuh file
  model `models/hand_landmarker.task` (lihat langkah 3).
- MediaPipe mendukung Python 3.8–3.12. Disarankan Python 3.10/3.11/3.12.
- Gestur **🫶 Heart** dideteksi dengan **dua tangan** membentuk hati
  (ujung telunjuk bertemu di atas, ujung jempol bertemu di bawah).
