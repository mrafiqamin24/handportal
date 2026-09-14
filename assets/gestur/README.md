# Foto gestur (opsional)

Folder ini sengaja kosong di repo. Foto contoh gestur aslinya tidak disertakan karena
berisi wajah pemilik proyek.

Satu-satunya berkas yang dipakai aplikasi adalah `Fotokitablurr.jpg`: thumbnail kecil
yang ditempel di pojok kanan bawah layar saat gestur ✌️ Peace aktif (diperkecil ke
160×90). Taruh foto 16:9 milikmu dengan nama itu kalau ingin thumbnail tampil. Tanpa
berkas itu aplikasi tetap jalan dan hanya mencetak peringatan `gambar tidak ditemukan`.

Berkas `*.jpg` di folder ini diabaikan git (lihat `.gitignore`), jadi fotomu tidak akan
ikut ter-commit.

Test tidak membutuhkan foto apa pun: landmark ketiga pose referensi sudah dibekukan di
`tests/fixtures/reference_landmarks.json`.
