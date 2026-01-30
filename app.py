import database
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
import tempfile
import os
import io
from datetime import datetime

# 1. KONFIGURASI & SETUP
st.set_page_config(
    page_title="Sistem Deteksi MBG",
    page_icon="🍱",
    layout="wide",
    initial_sidebar_state="expanded"
)

database.init_db()
st.markdown("""
    <style>
    .main {background-color: #f4f6f9;}
    h1 {color: #1F618D; font-family: 'Helvetica', sans-serif;}
    .stButton>button {width: 100%; border-radius: 8px; height: 45px; font-weight: bold; box-shadow: 0px 4px 6px rgba(0,0,0,0.1);}
    </style>
    """, unsafe_allow_html=True)

# Sidebar Info
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2921/2921226.png", width=80)
    st.title("Panel Kontrol")
    st.info("""
    **Tentang Proyek:**
    Implementasi Deep Learning YOLOv8 untuk Deteksi Komposisi Menu Program Makan Bergizi Gratis.
    """)

    conf_threshold = st.slider("Akurasi (Confidence)", 0.05, 1.0, 0.15, help="Geser ke kiri jika objek tidak muncul.")
    nms_threshold = st.slider("IoU (Tumpukan)", 0.1, 1.0, 0.45)
    st.divider()
    st.info("ℹ️ **Nutri-Scan Final**\nDatabase: 51 Kelas MBG\nFitur: PDF, HD Upscale, Smart Filter.")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except:
    st.error("⚠️ File 'best.pt' tidak ditemukan! Pastikan file ada di folder proyek.")
    st.stop()

# --- DATABASE GIZI LENGKAP (51 KELAS DATASET) ---
# Format Macros: (Karbohidrat, Protein, Lemak) dalam gram
database_gizi = {
    # === KARBOHIDRAT (NASI & MIE) ===
    "nasi putih":     {"label": "Nasi Putih",         "kalori": 175, "tipe": "Karbohidrat", "macros": (40, 4, 0)},
    "nasi kuning":    {"label": "Nasi Kuning",        "kalori": 200, "tipe": "Karbohidrat", "macros": (42, 4, 4)},
    "nasi goreng":    {"label": "Nasi Goreng",        "kalori": 250, "tipe": "Karbohidrat", "macros": (45, 6, 8)},
    "mie":            {"label": "Mie Goreng/Rebus",   "kalori": 220, "tipe": "Karbohidrat", "macros": (35, 5, 8)},
    "kentang goreng": {"label": "Kentang Goreng",     "kalori": 150, "tipe": "Karbohidrat", "macros": (20, 2, 7)},
    "burger":         {"label": "Burger",             "kalori": 300, "tipe": "Karbohidrat/Protein", "macros": (30, 15, 12)},

    # === PROTEIN HEWANI (AYAM) ===
    "ayam goreng":    {"label": "Ayam Goreng",        "kalori": 200, "tipe": "Protein Hewani", "macros": (2, 22, 12)},
    "ayam bakar":     {"label": "Ayam Bakar",         "kalori": 180, "tipe": "Protein Hewani", "macros": (4, 24, 8)},
    "ayam kecap":     {"label": "Ayam Kecap",         "kalori": 210, "tipe": "Protein Hewani", "macros": (8, 22, 10)},
    "ayam krispi":    {"label": "Ayam Krispi/Fried Chicken", "kalori": 280, "tipe": "Protein Hewani", "macros": (15, 20, 18)},
    "ayam suwir":     {"label": "Ayam Suwir",         "kalori": 150, "tipe": "Protein Hewani", "macros": (2, 25, 5)},
    "katsu":          {"label": "Chicken Katsu",      "kalori": 250, "tipe": "Protein Hewani", "macros": (15, 18, 14)},
    "nugget":         {"label": "Nugget Ayam",        "kalori": 180, "tipe": "Protein Hewani", "macros": (12, 10, 10)},

    # === PROTEIN HEWANI (DAGING, IKAN, BAKSO) ===
    "olahan daging sapi": {"label": "Olahan Daging Sapi", "kalori": 220, "tipe": "Protein Hewani", "macros": (5, 22, 12)},
    "bakso":          {"label": "Bakso",              "kalori": 190, "tipe": "Protein Hewani", "macros": (10, 12, 10)},
    "lele goreng":    {"label": "Lele Goreng",        "kalori": 160, "tipe": "Protein Hewani", "macros": (3, 15, 10)},

    # === PROTEIN HEWANI (TELUR) ===
    "telur rebus":    {"label": "Telur Rebus",        "kalori": 75,  "tipe": "Protein Hewani", "macros": (0, 7, 5)},
    "telur goreng":   {"label": "Telur Goreng/Dadar", "kalori": 110, "tipe": "Protein Hewani", "macros": (1, 7, 9)},
    "telur kecap":    {"label": "Telur Kecap",        "kalori": 130, "tipe": "Protein Hewani", "macros": (5, 7, 9)},

    # === PROTEIN NABATI (TAHU & TEMPE) ===
    "tahu":           {"label": "Tahu",               "kalori": 70,  "tipe": "Protein Nabati", "macros": (2, 8, 4)},
    "tahu goreng":    {"label": "Tahu Goreng",        "kalori": 85,  "tipe": "Protein Nabati", "macros": (3, 8, 6)},
    "tahu bacem":     {"label": "Tahu Bacem",         "kalori": 100, "tipe": "Protein Nabati", "macros": (8, 8, 5)},
    "tempe goreng":   {"label": "Tempe Goreng",       "kalori": 100, "tipe": "Protein Nabati", "macros": (7, 9, 6)},
    "tempe bacem":    {"label": "Tempe Bacem",        "kalori": 120, "tipe": "Protein Nabati", "macros": (12, 9, 6)},
    "tempe orek":     {"label": "Tempe Orek",         "kalori": 130, "tipe": "Protein Nabati", "macros": (15, 8, 6)},
    "keripik tempe":  {"label": "Keripik Tempe",      "kalori": 150, "tipe": "Pelengkap", "macros": (18, 5, 8)},

    # === SAYURAN (SERAT & VITAMIN) ===
    "sayur sop":      {"label": "Sayur Sop",          "kalori": 60,  "tipe": "Serat & Vitamin", "macros": (8, 2, 2)},
    "sayur bayam":    {"label": "Sayur Bayam",        "kalori": 40,  "tipe": "Serat & Vitamin", "macros": (4, 2, 0)},
    "sayur tumis":    {"label": "Tumis Sayur",        "kalori": 70,  "tipe": "Serat & Vitamin", "macros": (6, 2, 4)},
    "sayur pakcoy":   {"label": "Tumis Pakcoy",       "kalori": 50,  "tipe": "Serat & Vitamin", "macros": (5, 2, 3)},
    "cah kangkung":   {"label": "Cah Kangkung",       "kalori": 60,  "tipe": "Serat & Vitamin", "macros": (5, 3, 4)},
    "cah sayur":      {"label": "Cah Sayuran",        "kalori": 60,  "tipe": "Serat & Vitamin", "macros": (5, 2, 4)},
    "selada air":     {"label": "Selada Air",         "kalori": 20,  "tipe": "Serat & Vitamin", "macros": (2, 1, 0)},
    "kemangi":        {"label": "Daun Kemangi",       "kalori": 5,   "tipe": "Serat & Vitamin", "macros": (1, 0, 0)},
    "timun":          {"label": "Potongan Timun",     "kalori": 10,  "tipe": "Serat & Vitamin", "macros": (2, 0, 0)},
    "tomat":          {"label": "Potongan Tomat",     "kalori": 15,  "tipe": "Serat & Vitamin", "macros": (3, 1, 0)},

    # === BUAH-BUAHAN (VITAMIN) ===
    "buah pisang":    {"label": "Pisang",             "kalori": 90,  "tipe": "Vitamin", "macros": (23, 1, 0)},
    "buah jeruk":     {"label": "Jeruk",              "kalori": 45,  "tipe": "Vitamin", "macros": (11, 1, 0)},
    "buah apel":      {"label": "Apel",               "kalori": 52,  "tipe": "Vitamin", "macros": (14, 0, 0)},
    "buah semangka":  {"label": "Semangka",           "kalori": 30,  "tipe": "Vitamin", "macros": (8, 1, 0)},
    "buah melon":     {"label": "Melon",              "kalori": 34,  "tipe": "Vitamin", "macros": (9, 1, 0)},
    "buah pepaya":    {"label": "Pepaya",             "kalori": 43,  "tipe": "Vitamin", "macros": (11, 0, 0)},
    "buah salak":     {"label": "Salak",              "kalori": 70,  "tipe": "Vitamin", "macros": (17, 1, 0)},
    "buah anggur":    {"label": "Anggur",             "kalori": 67,  "tipe": "Vitamin", "macros": (17, 0, 0)},
    "buah kelengkeng":{"label": "Kelengkeng",         "kalori": 60,  "tipe": "Vitamin", "macros": (15, 1, 0)},
    "buah naga":      {"label": "Buah Naga",          "kalori": 50,  "tipe": "Vitamin", "macros": (13, 1, 0)},
    "buah strawberry":{"label": "Strawberry",         "kalori": 32,  "tipe": "Vitamin", "macros": (8, 1, 0)},

    # === PELENGKAP & MINUMAN ===
    "acar biasa":     {"label": "Acar Timun/Wortel",  "kalori": 30,  "tipe": "Pelengkap", "macros": (5, 0, 0)},
    "acar mayo":      {"label": "Salad/Acar Mayo",    "kalori": 80,  "tipe": "Pelengkap", "macros": (4, 1, 7)},
    "susu":           {"label": "Susu UHT/Kotak",     "kalori": 120, "tipe": "Minuman Bergizi", "macros": (10, 6, 6)},
    
    # === LAINNYA (NON-MAKANAN) ===
    "tray mbg":       {"label": "Nampan MBG",         "kalori": 0,   "tipe": "Wadah", "macros": (0, 0, 0)},
}

# 3. KELAS PDF GENERATOR
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'LAPORAN DETEKSI KOMPOSISI MENU MBG', 0, 1, 'C')
        self.set_font('Arial', 'I', 10)
        self.cell(0, 5, 'Implementasi Deep Learning YOLOv8 pada Program Makan Bergizi Gratis', 0, 1, 'C')
        self.ln(10)
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Halaman {self.page_no()} - Sistem Deteksi Komposisi Menu', 0, 0, 'C')

# 4. FUNGSI PENGOLAHAN CITRA (PCD) & FILTER
def process_image(pil_image, upscale=False):
    img_array = np.array(pil_image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    img_yuv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
    img_enhanced = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    
    # 2. Bilateral Filter
    img_smooth = cv2.bilateralFilter(img_enhanced, 9, 75, 75)
    
    if upscale:
        # Super Resolution Simulation
        height, width = img_smooth.shape[:2]
        img_upscaled = cv2.resize(img_smooth, (width * 2, height * 2), interpolation=cv2.INTER_LANCZOS4)
        gaussian = cv2.GaussianBlur(img_upscaled, (9, 9), 10.0)
        img_final = cv2.addWeighted(img_upscaled, 1.5, gaussian, -0.5, 0, img_upscaled)
    else:
        img_final = img_smooth
    return img_enhanced, img_final

def smart_filter_boxes(boxes, class_names):
    if len(boxes) == 0: return []
    detections = []
    
    # Prioritas: Barang Spesifik > Barang Umum
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu().numpy())
        cls_id = int(box.cls[0].cpu().numpy())
        label = class_names[cls_id]
        
        priority = 2
        if label == "tray mbg": priority = 1
        if "nasi" in label: priority = 3
        
        detections.append({"box": [x1, y1, x2, y2], "conf": conf, "label": label, "priority": priority, "keep": True})

    # Cek Tumpukan
    for i in range(len(detections)):
        for j in range(len(detections)):
            if i == j: continue
            boxA = detections[i]["box"]; boxB = detections[j]["box"]
            
            xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
            
            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
            minArea = min(boxAArea, boxBArea)
            overlap = interArea / minArea if minArea > 0 else 0
            
            if overlap > 0.5:
                if detections[i]["priority"] < detections[j]["priority"]: detections[i]["keep"] = False
                elif detections[i]["priority"] > detections[j]["priority"]: detections[j]["keep"] = False
                elif detections[i]["priority"] == detections[j]["priority"]:
                    if detections[i]["conf"] < detections[j]["conf"]: detections[i]["keep"] = False
                    else: detections[j]["keep"] = False

    return [d for d in detections if d["keep"]]

# 5. USER INTERFACE (UI) UTAMA
st.title("Implementasi YOLOv8 untuk Deteksi Komposisi Menu MBG")
st.markdown("Sistem monitoring otomatis untuk Program Makan Bergizi Gratis.")
st.divider()

tab1, tab2 = st.tabs(["📁 Upload File", "📸 Kamera Langsung"])
source_image = None
if tab1:
    uploaded_file = tab1.file_uploader("Upload foto", type=["jpg", "png", "jpeg"])
    if uploaded_file: source_image = Image.open(uploaded_file)
if tab2:
    camera_file = tab2.camera_input("Ambil foto")
    if camera_file: source_image = Image.open(camera_file)

if source_image:
    col1, col2 = st.columns([1, 1])

    # === KOLOM KIRI: PCD & DOWNLOAD GAMBAR ===
    with col1:
        st.subheader("1. Pra-pemrosesan Citra (PCD)")
        st.image(source_image, caption="Citra Asli", use_container_width=True)

        use_upscale = st.checkbox("✨ Aktifkan Mode HD (Contrast Enhancement & Sharpening)", value=False)
        _, img_ready = process_image(source_image, upscale=use_upscale)
        
        with st.expander("👁️ Lihat Hasil Preprocessing", expanded=True):
            st.image(img_ready, caption=f"Hasil Olah Citra ({'HD Mode' if use_upscale else 'Standard'})", use_container_width=True)
            st.info("Citra ini telah melalui proses Histogram Equalization untuk memperjelas fitur nasi dan lauk.")

        # Download Gambar HD
        im_pil = Image.fromarray(img_ready)
        buf = io.BytesIO()
        im_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button(label="⬇️ Download Citra Hasil PCD", data=byte_im, file_name="NutriScan_Enhanced.png", mime="image/png")

    # === KOLOM KANAN: AI & LAPORAN ===
    with col2:
        st.subheader("2. Hasil Deteksi Komposisi")
        
        if st.button("🚀 ANALISIS KOMPOSISI MENU", type="primary", use_container_width=True):
            with st.spinner('Sedang melakukan segmentasi objek dan perhitungan gizi...'):
                
                img_for_ai = Image.fromarray(img_ready)
                
                # PREDIKSI
                results = model.predict(img_for_ai, conf=conf_threshold, iou=nms_threshold, agnostic_nms=False, max_det=50)
                
                # FILTERING
                filtered_detections = []
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    label = model.names[cls_id]
                    
                    filtered_detections.append({
                        "box": [x1, y1, x2, y2], 
                        "conf": conf, 
                        "label": label, 
                        "priority": 1, 
                        "keep": True
                    })

                # VISUALISASI
                img_res = np.array(img_ready)
                total_kalori = 0
                total_karbo, total_protein, total_lemak = 0, 0, 0
                komposisi_tipe = {"Karbohidrat": 0, "Protein": 0, "Sayuran": 0, "Buah-buahan": 0, "Lainnya": 0}
                found_types = set()
                item_details = []

                if len(filtered_detections) > 0:
                    for item in filtered_detections:
                        x1, y1, x2, y2 = map(int, item["box"])
                        label = item["label"]
                        
                        # Gambar Kotak
                        cv2.rectangle(img_res, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img_res, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        # Ambil Data Gizi
                        default_info = {"label": label, "tipe": "Lainnya", "kalori": 0, "macros": (0,0,0)}
                        info = database_gizi.get(label, default_info)
                        
                        total_kalori += info['kalori']
                        c, p, f = info.get('macros', (0,0,0))
                        total_karbo += c; total_protein += p; total_lemak += f
                        
                        item_details.append([info['label'], info['tipe'], f"{info['kalori']} kkal"])
                        
                        # Klasifikasi
                        tipe = info['tipe']
                        if "Karbohidrat" in tipe: komposisi_tipe["Karbohidrat"] += 1; found_types.add("Karbo")
                        elif "Protein" in tipe: komposisi_tipe["Protein"] += 1; found_types.add("Protein")
                        elif "Serat" in tipe or "Sayur" in label: komposisi_tipe["Sayuran"] += 1; found_types.add("Sayur")
                        elif "Vitamin" in tipe or "Buah" in label: komposisi_tipe["Buah-buahan"] += 1; found_types.add("Buah")
                        else: komposisi_tipe["Lainnya"] += 1

                    st.image(img_res, caption="Hasil Deteksi AI", use_container_width=True)
                    st.success(f"✅ Ditemukan {len(filtered_detections)} item.")

                    # Dashboard Grafik
                    st.write("### 📊 Statistik Gizi")
                    tg1, tg2, tg3 = st.tabs(["Proporsi", "Kalori", "Makro"])
                    
                    with tg1:
                        fig1, ax1 = plt.subplots(figsize=(4, 3))
                        d_pie = {k: v for k, v in komposisi_tipe.items() if v > 0}
                        if d_pie: ax1.pie(d_pie.values(), labels=d_pie.keys(), autopct='%1.1f%%', startangle=90)
                        st.pyplot(fig1)
                    with tg2:
                        fig2, ax2 = plt.subplots(figsize=(5, 3))
                        names = [x[0] for x in item_details]; cals = [int(x[2].split()[0]) for x in item_details]
                        ax2.barh(names, cals, color='skyblue')
                        st.pyplot(fig2)
                    with tg3:
                        fig3, ax3 = plt.subplots(figsize=(4, 3))
                        ax3.bar(['Karbo', 'Protein', 'Lemak'], [total_karbo, total_protein, total_lemak], color=['#F4D03F', '#E74C3C', '#5DADE2'])
                        st.pyplot(fig3)

                    # Logika Rekomendasi (Standar MBG)
                    rekomendasi = []
                    if total_kalori < 400: rekomendasi.append("- Total Kalori di bawah standar makan siang MBG (Min. 400 kkal).")
                    if "Sayur" not in found_types: rekomendasi.append("- Komponen Sayuran tidak ditemukan. Menu tidak seimbang.")
                    if "Protein" not in found_types: rekomendasi.append("- Komponen Lauk/Protein tidak ditemukan.")
                    if "Buah" not in found_types: rekomendasi.append("- Komponen Buah tidak ditemukan.")
                    
                    if not rekomendasi: 
                        status_text = "✅ Menu MEMENUHI Standar Gizi Program MBG (4 Sehat 5 Sempurna)."
                    else:
                        status_text = "⚠️ Menu BELUM MEMENUHI Standar Lengkap."
                        
                    st.info(status_text)

                    # --- SAVE DATABASE ---
                    status_db = "Seimbang" if not rekomendasi else "Kurang Lengkap"
                    database.simpan_scan(total_kalori, len(filtered_detections), status_db, item_details)

                    # Generate PDF
                    with tempfile.TemporaryDirectory() as tmpdirname:
                        img_path = os.path.join(tmpdirname, "detected.jpg")
                        cv2.imwrite(img_path, cv2.cvtColor(img_res, cv2.COLOR_RGB2BGR))
                        chart1_path = os.path.join(tmpdirname, "chart1.png"); fig1.savefig(chart1_path)
                        chart3_path = os.path.join(tmpdirname, "chart3.png"); fig3.savefig(chart3_path)
                        
                        pdf = PDF()
                        pdf.add_page()
                        pdf.set_font("Arial", size=12)
                        pdf.cell(0, 10, f"Tanggal: {datetime.now().strftime('%d-%m-%Y')}", ln=True)
                        pdf.cell(0, 10, f"Total Kalori: {total_kalori} kkal", ln=True)
                        pdf.ln(5)
                        
                        pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, "1. HASIL DETEKSI VISUAL", ln=True)
                        pdf.image(img_path, x=10, w=100); pdf.ln(5)
                        
                        pdf.cell(0, 10, "2. RINCIAN KOMPOSISI", ln=True)
                        pdf.set_font("Arial", 'B', 10)
                        pdf.cell(70, 10, "Item Menu", 1); pdf.cell(70, 10, "Kategori", 1); pdf.cell(40, 10, "Kalori", 1); pdf.ln()
                        pdf.set_font("Arial", size=10)
                        for item in item_details:
                            pdf.cell(70, 10, item[0], 1); pdf.cell(70, 10, item[1], 1); pdf.cell(40, 10, item[2], 1); pdf.ln()
                        
                        pdf.ln(10)
                        pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, "3. GRAFIK KOMPOSISI", ln=True)
                        y_pos = pdf.get_y()
                        pdf.image(chart1_path, x=10, y=y_pos, w=80); pdf.image(chart3_path, x=100, y=y_pos, w=80)
                        pdf.ln(65)
                        
                        pdf.cell(0, 10, "4. REKOMENDASI", ln=True)
                        pdf.set_font("Arial", size=11)
                        if not rekomendasi: pdf.multi_cell(0, 7, "- Tidak ada catatan. Menu sudah sesuai standar.")
                        for rek in rekomendasi: pdf.multi_cell(0, 7, rek)
                        
                        pdf_bytes = pdf.output(dest='S').encode('latin-1')
                        st.divider()
                        st.download_button("📄 DOWNLOAD LAPORAN HASIL (PDF)", data=pdf_bytes, file_name="Laporan_Deteksi_MBG.pdf", mime="application/pdf", type="primary")

                else:
                    st.error("❌ Objek tidak terdeteksi. Silakan atur pencahayaan atau geser slider Threshold.")

# --- BOTTOM : RIWAYAT SCAN ---
st.divider()
st.subheader("📜 Riwayat Scan Hari Ini")

# database.py
df_history = database.ambil_riwayat_hari_ini()

if not df_history.empty:
    st.dataframe(df_history, use_container_width=True, hide_index=True)
else:
    st.info("Belum ada data scan hari ini.")