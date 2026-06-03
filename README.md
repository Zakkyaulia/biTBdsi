# Dashboard BI Alumni DSI Unand

Web dashboard Streamlit untuk data mart alumni DSI Unand dengan alur BI:

CSV hasil preprocessing -> NLP SBERT + cosine similarity -> PostgreSQL star schema -> dashboard Streamlit.

## Menjalankan

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Setup Database

Gunakan PostgreSQL dan buat database:

```sql
CREATE DATABASE bi_alumni_dsi;
```

Set connection string bila user/password berbeda dari default:

```powershell
$env:BI_DATABASE_URL="postgresql://postgres:postgres@localhost:5432/bi_alumni_dsi"
```

Load CSV bersih, hitung NLP SBERT + cosine similarity, lalu masukkan ke star schema:

```powershell
.\.venv\Scripts\python.exe scripts\load_star_schema.py
```

Jalankan dashboard:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Dataset utama untuk proses ETL disimpan di `data/riwayatalumniDSI_clean_final.csv`.

## Isi Dashboard

- **Akademik**: KPI alumni, tren lulusan, distribusi IPK, lama studi, dan scatter IPK vs lama studi.
- **Dosen**: analisis bridge many-to-many untuk pembimbing dan penguji.
- **NLP Tugas Akhir**: membaca atribut NLP dari `dim_tugas_akhir`, yaitu judul preprocessed, judul final, skor SBERT cosine similarity tertinggi, kategori keunikan, judul termirip, dan koordinat PCA.
- **Star Schema**: sample tabel dimensi, fact, dan bridge sesuai rancangan data mart.
- **Kualitas Data**: missing value, validasi tanggal, validasi IPK, dan preview data.
