import sqlite3
import pandas as pd
from datetime import datetime

DB_NAME = "riwayat_mbg.db"

def init_db():
    """Membuat tabel jika belum ada"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS riwayat (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            waktu TEXT,
            total_kalori INTEGER,
            jumlah_item INTEGER,
            status_gizi TEXT,
            detail_menu TEXT
        )
    ''')
    conn.commit()
    conn.close()

def simpan_scan(total_kalori, jumlah_item, status_gizi, item_list):
    """Menyimpan hasil scan baru"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    waktu_sekarang = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    menu_str = ", ".join([x[0] for x in item_list])
    c.execute('''
        INSERT INTO riwayat (waktu, total_kalori, jumlah_item, status_gizi, detail_menu)
        VALUES (?, ?, ?, ?, ?)
    ''', (waktu_sekarang, total_kalori, jumlah_item, status_gizi, menu_str))
    
    conn.commit()
    conn.close()

def ambil_riwayat_hari_ini():
    """Mengambil data scan HANYA hari ini"""
    conn = sqlite3.connect(DB_NAME)
    hari_ini = datetime.now().strftime("%Y-%m-%d")
    query = f"SELECT waktu, detail_menu, total_kalori, status_gizi FROM riwayat WHERE waktu LIKE '{hari_ini}%' ORDER BY waktu DESC"
    df = pd.read_sql_query(query, conn)
    conn.close()
    df.columns = ["Jam Scan", "Menu Terdeteksi", "Kalori (kkal)", "Status"]
    return df