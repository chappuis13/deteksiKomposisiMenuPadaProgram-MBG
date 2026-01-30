# 🍱 Nutri-Scan: Sistem Monitoring Program Makan Bergizi Gratis (MBG)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![YOLOv8](https://img.shields.io/badge/AI-YOLOv8-green)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)

**Nutri-Scan** adalah aplikasi berbasis Artificial Intelligence (Computer Vision) yang dirancang untuk membantu pengawasan Program Makan Bergizi Gratis di sekolah. Aplikasi ini mendeteksi komponen menu, menghitung estimasi kalori, dan memberikan analisis kelengkapan gizi secara otomatis melalui foto.

---

## 🌟 Fitur Unggulan

1.  **AI Detection (51 Kelas):** Mendeteksi Nasi, Lauk (Ayam/Ikan/Telur), Sayur, Buah, dan Susu dengan presisi.
2.  **Smart Filtering:** Algoritma cerdas untuk mengatasi tumpukan makanan (misal: Ayam di atas Nasi).
3.  **HD Upscaling (PCD):** Fitur *Image Processing* untuk memperjelas foto buram/gelap sebelum dideteksi AI.
4.  **Auto-Calculation:** Menghitung Total Kalori & Estimasi Makronutrisi (Karbo/Protein/Lemak).
5.  **PDF Reporting:** Cetak laporan resmi hasil analisis sekali klik.
6.  **Database History:** Menyimpan riwayat scan harian secara otomatis.

---

## 🛠️ Teknologi yang Digunakan

* **Deep Learning:** YOLOv8 (Ultralytics) - Trained on 12k+ Images.
* **Web Framework:** Streamlit.
* **Image Processing:** OpenCV (CLAHE, Bilateral Filter).
* **Visualization:** Matplotlib.
* **Database:** SQLite.

---

## 📸 Cara Penggunaan

1.  Buka aplikasi (Link Deployment).
2.  **Upload Foto** nampan makanan atau gunakan **Kamera Langsung**.
3.  (Opsional) Centang **"Mode HD"** jika foto kurang jelas.
4.  Klik **"Mulai Analisis"**.
5.  Lihat hasil deteksi, grafik gizi, dan rekomendasi sistem.
6.  Download **Laporan PDF** untuk arsip.

---

## 💻 Instalasi Lokal

Jika ingin menjalankan di laptop sendiri:

```bash
git clone [https://github.com/username-anda/NutriScan-MBG.git](https://github.com/username-anda/NutriScan-MBG.git)
cd NutriScan-MBG
pip install -r requirements.txt
streamlit run app.py