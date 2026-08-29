import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import psycopg

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import app


def init_schema(conn):
    schema_file = ROOT_DIR / "sql" / "01_schema.sql"
    if schema_file.exists():
        print(f"Executing schema from {schema_file.name}...")
        sql_script = schema_file.read_text(encoding="utf-8")
        with conn.cursor() as cur:
            cur.execute(sql_script)
        conn.commit()
        print("Schema DDL executed successfully.")


def load_star_schema_batch(db_url: str):
    data_file = ROOT_DIR / "data" / "riwayatalumniDSI_clean_final.csv"
    if not data_file.exists():
        print(f"Error: CSV file not found at {data_file}")
        sys.exit(1)

    print(f"Reading CSV data from {data_file.name}...")
    df = pd.read_csv(data_file)
    column_mapping = {
        "NO": "no",
        "Nama": "nama",
        "NIM": "nim",
        "Pembimbing 1": "pembimbing_1",
        "Pembimbing 2": "pembimbing_2",
        "Judul Tugas Akhir": "judul_ta",
        "Dosen Penguji 1": "penguji_1",
        "Dosen Penguji 2": "penguji_2",
        "Dosen Penguji 3": "penguji_3",
        "Periode Wisuda": "periode_wisuda",
        "Tahun Wisuda": "tahun_wisuda",
        "Tanggal Lulus": "tanggal_lulus",
        "IPK": "ipk",
        "Lama Studi": "lama_studi",
    }
    df = df.rename(columns=column_mapping)
    records = df.to_dict("records")

    print("Building dimension and fact tables in memory...")

    # Data structures
    mahasiswa_map = {}  # nim -> key
    waktu_map = {}      # tanggal_lulus -> key
    periode_map = {}    # (tahun_wisuda, periode_label) -> key
    ipk_map = {}        # ipk_val -> key
    lama_studi_map = {} # months_val -> key
    dosen_map = {}      # normalized_name -> key

    dim_mahasiswa_rows = []
    dim_waktu_rows = []
    dim_periode_rows = []
    dim_ipk_rows = []
    dim_lama_studi_rows = []
    dim_dosen_rows = []
    dim_ta_rows = []
    fact_kelulusan_rows = []
    bridge_peran_rows = []

    mahasiswa_counter = 1
    waktu_counter = 1
    periode_counter = 1
    ipk_counter = 1
    lama_studi_counter = 1
    dosen_counter = 1
    ta_counter = 1
    kelulusan_counter = 1
    peran_counter = 1

    # 1. Prepare dim_tugas_akhir for SBERT computation first
    ta_records = []
    for idx, r in enumerate(records):
        t_key = idx + 1
        j_ta = str(r["judul_ta"])
        j_clean = app.clean_title(j_ta)
        j_final = app.final_title(j_ta)
        ta_records.append({
            "ta_key": t_key,
            "judul_tugas_akhir": j_ta,
            "judul_preprocessed": j_clean,
            "judul_final": j_final
        })

    print("Computing NLP SBERT embeddings & similarity metrics in batch...")
    title_df = pd.DataFrame(ta_records)
    metrics_df = app.compute_title_similarity_metrics(title_df)
    metrics_map = {int(row["ta_key"]): row for row in metrics_df.to_dict("records")}

    # Build rows for each record
    for idx, r in enumerate(records):
        # 1. dim_mahasiswa
        nim_str = str(r["nim"])
        if nim_str not in mahasiswa_map:
            m_key = mahasiswa_counter
            mahasiswa_counter += 1
            mahasiswa_map[nim_str] = m_key
            dim_mahasiswa_rows.append((m_key, nim_str, str(r["nama"])))
        else:
            m_key = mahasiswa_map[nim_str]

        # 2. dim_waktu
        parsed_dt = app.parse_tanggal_lulus(r["tanggal_lulus"])
        if pd.isna(parsed_dt):
            t_date = pd.Timestamp(year=int(r.get("tahun_wisuda", 2000)), month=1, day=1).date()
        else:
            t_date = parsed_dt.date()

        if t_date not in waktu_map:
            w_key = waktu_counter
            waktu_counter += 1
            waktu_map[t_date] = w_key
            dim_waktu_rows.append((w_key, t_date, t_date.day, t_date.month, t_date.year))
        else:
            w_key = waktu_map[t_date]

        # 3. dim_periode_wisuda
        p_label = str(r["periode_wisuda"])
        p_tahun = int(r["tahun_wisuda"])
        p_tuple = (p_tahun, p_label)
        if p_tuple not in periode_map:
            p_key = periode_counter
            periode_counter += 1
            periode_map[p_tuple] = p_key
            p_num = app.PERIODE_ORDER.get(p_label, 99)
            dim_periode_rows.append((p_key, p_tahun, p_num, p_label))
        else:
            p_key = periode_map[p_tuple]

        # 4. dim_ipk
        ipk_val = float(r["ipk"])
        if ipk_val not in ipk_map:
            i_key = ipk_counter
            ipk_counter += 1
            ipk_map[ipk_val] = i_key
            r_ipk = app.rentang_ipk(ipk_val)
            pred_ipk = app.predikat_ipk(ipk_val)
            dim_ipk_rows.append((i_key, ipk_val, r_ipk, pred_ipk))
        else:
            i_key = ipk_map[ipk_val]

        # 5. dim_lama_studi
        months_val = int(r["lama_studi"])
        if months_val not in lama_studi_map:
            l_key = lama_studi_counter
            lama_studi_counter += 1
            lama_studi_map[months_val] = l_key
            k_studi = app.kategori_lama_studi(months_val, 54)
            f_tepat = months_val <= 54
            dim_lama_studi_rows.append((l_key, months_val, k_studi, f_tepat))
        else:
            l_key = lama_studi_map[months_val]

        # 6. dim_tugas_akhir
        ta_k = idx + 1
        m_row = metrics_map[ta_k]
        dim_ta_rows.append((
            ta_k,
            str(r["judul_ta"]),
            app.clean_title(r["judul_ta"]),
            app.final_title(r["judul_ta"]),
            round(float(m_row["skor_kemiripan_tertinggi"]), 6) if pd.notna(m_row["skor_kemiripan_tertinggi"]) else None,
            str(m_row["kategori_keunikan"]),
            int(m_row["ta_key_termirip"]) if pd.notna(m_row["ta_key_termirip"]) else None,
            round(float(m_row["pca_x"]), 8) if pd.notna(m_row["pca_x"]) else None,
            round(float(m_row["pca_y"]), 8) if pd.notna(m_row["pca_y"]) else None,
        ))

        # 7. fact_kelulusan
        k_key = kelulusan_counter
        kelulusan_counter += 1
        f_tepat_bool = months_val <= 54
        fact_kelulusan_rows.append((
            k_key, m_key, w_key, p_key, i_key, l_key, ta_k, ipk_val, months_val, f_tepat_bool, 1
        ))

        # 8. Lecturers bridge
        role_columns = [
            ("pembimbing_1", "Pembimbing", 1),
            ("pembimbing_2", "Pembimbing", 2),
            ("penguji_1", "Penguji", 1),
            ("penguji_2", "Penguji", 2),
            ("penguji_3", "Penguji", 3),
        ]
        for key_name, jenis_peran, urutan_peran in role_columns:
            val = r.get(key_name)
            normalized = app.normalize_person_name(val)
            if normalized is None:
                continue
            if normalized not in dosen_map:
                d_key = dosen_counter
                dosen_counter += 1
                dosen_map[normalized] = d_key
                dim_dosen_rows.append((d_key, normalized, str(val).strip()))
            else:
                d_key = dosen_map[normalized]

            pr_key = peran_counter
            peran_counter += 1
            bridge_peran_rows.append((pr_key, k_key, d_key, jenis_peran, urutan_peran, 1))

    print(f"Connecting to database and performing batch inserts...")
    with psycopg.connect(db_url) as conn:
        init_schema(conn)

        tables_to_truncate = [
            "bridge_peran_dosen",
            "fact_kelulusan",
            "dim_tugas_akhir",
            "dim_dosen",
            "dim_mahasiswa",
            "dim_waktu",
            "dim_periode_wisuda",
            "dim_ipk",
            "dim_lama_studi",
        ]
        with conn.cursor() as cur:
            for table in tables_to_truncate:
                cur.execute(f"TRUNCATE TABLE {table} CASCADE;")
            print("Truncated old star schema tables.")

            print("Batch inserting dim_mahasiswa...")
            cur.executemany("INSERT INTO dim_mahasiswa VALUES (%s, %s, %s)", dim_mahasiswa_rows)

            print("Batch inserting dim_waktu...")
            cur.executemany("INSERT INTO dim_waktu VALUES (%s, %s, %s, %s, %s)", dim_waktu_rows)

            print("Batch inserting dim_periode_wisuda...")
            cur.executemany("INSERT INTO dim_periode_wisuda VALUES (%s, %s, %s, %s)", dim_periode_rows)

            print("Batch inserting dim_ipk...")
            cur.executemany("INSERT INTO dim_ipk VALUES (%s, %s, %s, %s)", dim_ipk_rows)

            print("Batch inserting dim_lama_studi...")
            cur.executemany("INSERT INTO dim_lama_studi VALUES (%s, %s, %s, %s)", dim_lama_studi_rows)

            print("Batch inserting dim_dosen...")
            cur.executemany("INSERT INTO dim_dosen VALUES (%s, %s, %s)", dim_dosen_rows)

            print("Batch inserting dim_tugas_akhir...")
            cur.executemany(
                "INSERT INTO dim_tugas_akhir VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                dim_ta_rows
            )

            print("Batch inserting fact_kelulusan...")
            cur.executemany(
                "INSERT INTO fact_kelulusan VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                fact_kelulusan_rows
            )

            print("Batch inserting bridge_peran_dosen...")
            cur.executemany(
                "INSERT INTO bridge_peran_dosen VALUES (%s, %s, %s, %s, %s, %s)",
                bridge_peran_rows
            )

        conn.commit()
    print(f"SUCCESS! Star schema batch migration to Neon PostgreSQL complete ({len(records)} records).")


def main():
    parser = argparse.ArgumentParser(description="Load star schema into PostgreSQL database.")
    parser.add_argument("--db-url", type=str, help="PostgreSQL connection URI", default=None)
    args = parser.parse_args()

    db_url = args.db_url or os.getenv("BI_DATABASE_URL") or app.DEFAULT_DB_URL
    os.environ["BI_DATABASE_URL"] = db_url

    print(f"Target DB Host: {db_url.split('@')[-1] if '@' in db_url else db_url}")
    load_star_schema_batch(db_url)


if __name__ == "__main__":
    main()
