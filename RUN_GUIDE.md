# Panduan Menjalankan Aplikasi (Run Guide)

Dokumen ini berisi panduan singkat dan cepat untuk menjalankan dashboard Streamlit BI Alumni DSI Unand.

---

## 🚀 Menjalankan Harian (Jika Database Sudah Terisi)
Jika Anda sudah pernah memasukkan data ke database sebelumnya, Anda hanya perlu menjalankan dua perintah ini untuk membuka dashboard:

1. **Aktifkan Virtual Environment:**
   * **PowerShell (VS Code default):**
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   * **Command Prompt (CMD):**
     ```cmd
     .\.venv\Scripts\activate.bat
     ```

2. **Jalankan Dashboard:**
   ```bash
   streamlit run app.py
   ```
   *(Atau jika perintah di atas tidak ditemukan: `python -m streamlit run app.py`)*

---

## 📥 Impor Data Baru (Hanya Jika Data di CSV Berubah / Database Kosong)
Jika Anda baru pertama kali setup atau ada pembaruan data pada berkas CSV di `data/riwayatalumniDSI_clean_final.csv`, jalankan perintah berikut untuk memproses NLP dan memasukkannya ke PostgreSQL:

1. **Pastikan database `bi_alumni_dsi` sudah dibuat di PostgreSQL.**
2. **Aktifkan virtual environment (seperti langkah di atas).**
3. **Jalankan script ETL:**
   ```bash
   python scripts\load_star_schema.py --db-url "postgresql://postgres:password@127.0.0.1:5432/bi_alumni_dsi"
   ```
   *(Sesuaikan `username`, `password`, `host`, dan `port` jika konfigurasi PostgreSQL Anda berbeda)*.
