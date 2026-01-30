# 🍱 Sistem Deteksi Komposisi Menu pada Program Makan Bergizi Gratis (MBG)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://deteksikomposisimenupadaprogram-mbg.streamlit.app/)
![Python Version](https://img.shields.io/badge/Python-3.9%20%7C%203.10-blue)
![YOLOv8](https://img.shields.io/badge/AI-YOLOv8s-green)
![Status](https://img.shields.io/badge/Status-Deployed-success)

> **Link Aplikasi Live:** [https://deteksikomposisimenupadaprogram-mbg.streamlit.app/](https://deteksikomposisimenupadaprogram-mbg.streamlit.app/)

## 📖 Tentang Proyek

Proyek ini adalah implementasi **Computer Vision** dan **Deep Learning** untuk mendukung pengawasan (monitoring) **Program Makan Bergizi Gratis (MBG)**. Sistem ini dirancang untuk mendeteksi komponen makanan dalam nampan/kotak makan secara otomatis, menghitung estimasi kalori, serta memverifikasi kelengkapan gizi (Karbohidrat, Protein, Sayur, Buah) sesuai standar kesehatan.

Sistem dibangun menggunakan algoritma **YOLOv8 (You Only Look Once)** yang telah dilatih ulang (fine-tuned) dengan dataset spesifik menu katering sekolah Indonesia.

---

## 🌟 Fitur Utama

### 1. Deteksi Multi-Kelas (51 Jenis Makanan)
Sistem mampu mengenali 51 item spesifik yang umum ditemukan pada menu makan siang sekolah, mulai dari nasi, berbagai olahan ayam/ikan, hingga variasi sayuran dan buah potong.

### 2. Smart Filtering & Handling Occlusion
Menggunakan algoritma *post-processing* khusus untuk mengatasi masalah tumpukan makanan (misal: Ayam menutupi Nasi). Sistem memprioritaskan deteksi objek krusial (Nasi/Lauk) dibanding wadah/tray.

### 3. HD Upscaling (PCD Preprocessing)
Terintegrasi dengan modul **Pengolahan Citra Digital (PCD)**:
* **CLAHE:** *Contrast Limited Adaptive Histogram Equalization* untuk memperjelas tekstur nasi putih pada nampan stainless.
* **Sharpening:** Mempertegas tepi objek agar deteksi AI lebih akurat pada kondisi cahaya minim.

### 4. Analisis Gizi & Makronutrisi
Secara otomatis mengonversi hasil deteksi menjadi data kuantitatif:
* Total Kalori (kkal).
* Estimasi Gramasi Makronutrisi (Karbohidrat, Protein, Lemak).
* Visualisasi grafik Pie Chart dan Bar Chart.

### 5. Pelaporan Otomatis (PDF)
Menghasilkan laporan resmi dalam format PDF yang berisi bukti visual (foto terdeteksi), rincian menu, analisis grafik, dan rekomendasi perbaikan gizi.

### 6. Riwayat Scan (Database)
Menyimpan data hasil pemindaian harian ke dalam database SQLite untuk keperluan rekapitulasi dan audit.

---

## 🛠️ Teknologi yang Digunakan

* **Core AI:** [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (Model: `yolov8s.pt`)
* **Frontend Framework:** [Streamlit](https://streamlit.io/)
* **Image Processing:** OpenCV (`cv2`) - Headless Version
* **Data Manipulation:** Pandas & NumPy
* **Visualization:** Matplotlib
* **Report Generation:** FPDF
* **Database:** SQLite3

---

## 🧠 Dataset & Model Training

Model AI dilatih menggunakan dataset yang dikurasi khusus untuk konteks MBG:

* **Jumlah Kelas:** 51 Kelas (Lihat rincian di bawah).
* **Jumlah Data:** 12.000+ Citra (Training/Validation/Test).
* **Preprocessing:** Resize 640x640, Auto-Orientation.
* **Augmentation:** Flip, Rotation, Exposure, Blur (untuk simulasi kamera HP).
* **Base Model:** YOLOv8s (Small) - Dipilih karena keseimbangan antara kecepatan dan akurasi.

<details>
<summary><b>🔻 Klik untuk melihat Daftar 51 Kelas Deteksi</b></summary>

| Kategori | Item Menu |
| :--- | :--- |
| **Karbohidrat** | Nasi Putih, Nasi Kuning, Nasi Goreng, Mie, Kentang Goreng, Burger |
| **Protein Hewani** | Ayam (Goreng/Bakar/Kecap/Krispi/Suwir), Katsu, Nugget, Olahan Sapi, Bakso, Lele Goreng, Telur (Rebus/Goreng/Kecap) |
| **Protein Nabati** | Tahu (Goreng/Bacem), Tempe (Goreng/Bacem/Orek) |
| **Sayuran** | Sayur Sop, Bayam, Tumis Sayur, Pakcoy, Kangkung, Cah Sayur, Selada, Timun, Tomat, Kemangi |
| **Buah-buahan** | Pisang, Jeruk, Apel, Semangka, Melon, Pepaya, Salak, Anggur, Kelengkeng, Buah Naga, Strawberry |
| **Pelengkap** | Acar, Keripik Tempe, Susu, Tray/Wadah |

</details>

---

## 📸 Tangkapan Layar (Screenshot)

| Tampilan Awal & Upload | Hasil Deteksi AI |
| :---: | :---: |
| ![UI Upload](https://via.placeholder.com/400x200?text=UI+Upload+Foto) | ![Hasil AI](https://via.placeholder.com/400x200?text=Deteksi+YOLOv8) |

| Analisis Grafik | Laporan PDF |
| :---: | :---: |
| ![Grafik Gizi](https://via.placeholder.com/400x200?text=Pie+Chart+Gizi) | ![PDF Report](https://via.placeholder.com/400x200?text=Contoh+Laporan+PDF) |

---

## 💻 Cara Instalasi Lokal (Localhost)

Jika Anda ingin menjalankan aplikasi ini di laptop sendiri:

1.  **Clone Repository**
    ```bash
    git clone [https://github.com/USERNAME_ANDA/NAMA_REPO_ANDA.git](https://github.com/USERNAME_ANDA/NAMA_REPO_ANDA.git)
    cd NAMA_REPO_ANDA
    ```

2.  **Buat Virtual Environment (Disarankan)**
    ```bash
    python -m venv venv
    # Windows:
    venv\Scripts\activate
    # Mac/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Jalankan Aplikasi**
    ```bash
    streamlit run app.py
    ```

---

## 📂 Struktur Direktori

```text
📦 ROOT PROJECT
 ┣ 📜 app.py              # File Utama (Frontend & Logic)
 ┣ 📜 database.py         # Modul Manajemen Database (SQLite)
 ┣ 📜 best.pt             # Model YOLOv8 Hasil Training (Weights)
 ┣ 📜 requirements.txt    # Daftar Pustaka Python
 ┣ 📜 packages.txt        # Dependencies Linux (untuk Streamlit Cloud)
 ┗ 📜 README.md           # Dokumentasi Proyek
```

---

## 👨‍💻 Developer

<img src="https://via.placeholder.com/150" align="right" width="120" alt="Foto Profil">

**Chairil Soetiesna**
* **Peran:** Peneliti & Pengembang (Developer)
* **Instansi:** Mahasiswa Teknik Informatika - Universitas Nusa Putra
* **Email:** chairilcs13@gmail.com
* **Fokus:** Computer Vision, Deep Learning, & AI Deployment

> *"Proyek ini diajukan untuk memenuhi Tugas Akhir Mata Kuliah Citra Digital dengan judul: "Implementasi Deep Learning Menggunakan Algoritma YOLOv8 untuk Deteksi Komposisi Menu pada Program Makan Bergizi Gratis".*

---
