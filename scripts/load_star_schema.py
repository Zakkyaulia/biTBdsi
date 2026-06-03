from __future__ import annotations

import argparse
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT_DIR / "data" / "riwayatalumniDSI_clean_final.csv"
DEFAULT_DB_URL = "postgresql://postgres:postgres@localhost:5432/bi_alumni_dsi"
SCHEMA_SQL = ROOT_DIR / "sql" / "01_schema.sql"
SBERT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

PERIODE_ORDER = {
    "WISUDA I": 1,
    "WISUDA II": 2,
    "WISUDA III": 3,
    "WISUDA IV": 4,
    "WISUDA V": 5,
    "WISUDA VI": 6,
}

MONTH_MAP = {
    "januari": "January",
    "februari": "February",
    "maret": "March",
    "april": "April",
    "mei": "May",
    "juni": "June",
    "juli": "July",
    "agustus": "August",
    "september": "September",
    "oktober": "October",
    "nopember": "November",
    "november": "November",
    "desember": "December",
}

STOPWORDS_UNAND = {
    "dan",
    "pada",
    "di",
    "untuk",
    "dengan",
    "ke",
    "dari",
    "yang",
    "sistem",
    "informasi",
    "pembangunan",
    "berbasis",
    "menggunakan",
    "padang",
    "kota",
    "andalas",
    "universitas",
    "studi",
    "kasus",
    "sumatera",
    "barat",
    "melalui",
}

LECTURER_ALIASES = {
    "dwi welly sukma nirad": "dwi welly sukma n",
    "fajril kabar": "fajril akbar",
    "hafid yoza p": "hafid yoza putra",
    "hasdi puta": "hasdi putra",
    "rahmatika pratama": "rahmatika pratama s",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load CSV alumni DSI into PostgreSQL star schema.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="Path CSV hasil preprocessing.")
    parser.add_argument("--db-url", default=os.getenv("BI_DATABASE_URL", DEFAULT_DB_URL), help="PostgreSQL connection URL.")
    parser.add_argument("--target-months", type=int, default=48, help="Batas tepat waktu dalam bulan.")
    parser.add_argument("--skip-schema", action="store_true", help="Jangan eksekusi sql/01_schema.sql sebelum load.")
    return parser.parse_args()


def parse_tanggal_lulus(value: object) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT
    text = str(value).strip()
    lowered = text.lower()
    for indo, english in MONTH_MAP.items():
        lowered = re.sub(rf"\b{indo}\b", english, lowered, flags=re.IGNORECASE)
    parsed = pd.to_datetime(lowered, dayfirst=True, errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(text, dayfirst=True, errors="coerce")
    return parsed


def clean_title(text: object) -> str:
    if pd.isna(text):
        return ""
    normalized = unicodedata.normalize("NFKD", str(text).lower())
    normalized = re.sub(r"[^a-z\s]", " ", normalized)
    return " ".join(normalized.split())


def final_title(text: object) -> str:
    tokens = clean_title(text).split()
    return " ".join(token for token in tokens if token not in STOPWORDS_UNAND)


def normalize_person_name(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    if text in {"", "-", "nan", "none"}:
        return None
    return LECTURER_ALIASES.get(text, text)


def predikat_ipk(ipk: float) -> str:
    if ipk >= 3.51:
        return "Dengan Pujian"
    if ipk >= 2.76:
        return "Sangat Memuaskan"
    if ipk >= 2.0:
        return "Memuaskan"
    return "Perlu Review"


def rentang_ipk(ipk: float) -> str:
    if ipk >= 3.75:
        return "3.75 - 4.00"
    if ipk >= 3.50:
        return "3.50 - 3.74"
    if ipk >= 3.25:
        return "3.25 - 3.49"
    if ipk >= 3.00:
        return "3.00 - 3.24"
    return "< 3.00"


def kategori_lama_studi(months: int, target_months: int) -> str:
    if months <= target_months:
        return "Tepat Waktu"
    if months <= target_months + 6:
        return "Melewati Target <= 6 Bulan"
    if months <= 72:
        return "Terlambat"
    return "Sangat Terlambat"


def kategori_keunikan(score: float) -> str:
    if score >= 0.85:
        return "Tidak Unik"
    if score >= 0.70:
        return "Agak Unik / Perlu Review"
    return "Unik"


def load_clean_csv(path: Path, target_months: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [column.strip() for column in df.columns]
    df = df.reset_index(drop=True)
    df["kelulusan_key"] = np.arange(1, len(df) + 1)
    df["mahasiswa_key"] = np.arange(1, len(df) + 1)
    df["ta_key"] = np.arange(1, len(df) + 1)
    df["Tanggal Parsed"] = df["Tanggal Lulus"].apply(parse_tanggal_lulus)
    df["IPK"] = pd.to_numeric(df["IPK"], errors="coerce")
    df["Lama Studi"] = pd.to_numeric(df["Lama Studi"], errors="coerce").astype(int)
    df["Tahun Wisuda"] = pd.to_numeric(df["Tahun Wisuda"], errors="coerce").astype(int)
    df["Periode Num"] = df["Periode Wisuda"].map(PERIODE_ORDER).fillna(99).astype(int)
    df["Judul Clean"] = df["Judul Tugas Akhir"].apply(clean_title)
    df["Judul Final"] = df["Judul Tugas Akhir"].apply(final_title)
    df["Predikat IPK"] = df["IPK"].apply(predikat_ipk)
    df["Rentang IPK"] = df["IPK"].apply(rentang_ipk)
    df["flag_tepat_waktu"] = df["Lama Studi"].le(target_months)
    df["kategori_lama_studi"] = df["Lama Studi"].apply(lambda value: kategori_lama_studi(int(value), target_months))
    return df


def enrich_nlp(df: pd.DataFrame) -> pd.DataFrame:
    from sentence_transformers import SentenceTransformer

    titles = df["Judul Final"].replace("", np.nan).fillna(df["Judul Clean"]).tolist()
    model = SentenceTransformer(SBERT_MODEL)
    embeddings = model.encode(titles, show_progress_bar=True, normalize_embeddings=True)

    similarity = cosine_similarity(embeddings)
    similarity_without_self = similarity.copy()
    np.fill_diagonal(similarity_without_self, -1)
    max_scores = similarity_without_self.max(axis=1)
    nearest_idx = similarity_without_self.argmax(axis=1)

    if len(df) >= 2:
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(embeddings)
    else:
        coords = np.zeros((len(df), 2))

    enriched = df.copy()
    enriched["skor_kemiripan_tertinggi"] = max_scores
    enriched["ta_key_termirip"] = nearest_idx + 1
    enriched["kategori_keunikan"] = [kategori_keunikan(float(score)) for score in max_scores]
    enriched["pca_x"] = coords[:, 0]
    enriched["pca_y"] = coords[:, 1]
    return enriched


def unique_key(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    return pd.factorize(df[columns].astype(str).agg("|".join, axis=1))[0] + 1


def build_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    working = df.copy()
    working["waktu_key"] = pd.factorize(working["Tanggal Parsed"].astype(str))[0] + 1
    working["periode_key"] = unique_key(working, ["Tahun Wisuda", "Periode Wisuda"])
    working["ipk_key"] = unique_key(working, ["IPK", "Rentang IPK", "Predikat IPK"])
    working["lama_studi_key"] = pd.factorize(working["Lama Studi"].astype(str))[0] + 1

    dim_mahasiswa = working[["mahasiswa_key", "NIM", "Nama"]].rename(columns={"NIM": "nim", "Nama": "nama"})
    dim_mahasiswa["nim"] = dim_mahasiswa["nim"].apply(lambda value: None if pd.isna(value) else str(value).replace(".0", ""))

    dim_waktu = (
        working[["waktu_key", "Tanggal Parsed"]]
        .drop_duplicates("waktu_key")
        .assign(
            tanggal_lulus=lambda value: value["Tanggal Parsed"].dt.date,
            hari=lambda value: value["Tanggal Parsed"].dt.day,
            bulan=lambda value: value["Tanggal Parsed"].dt.month,
            tahun=lambda value: value["Tanggal Parsed"].dt.year,
        )
        [["waktu_key", "tanggal_lulus", "hari", "bulan", "tahun"]]
    )

    dim_periode_wisuda = (
        working[["periode_key", "Tahun Wisuda", "Periode Num", "Periode Wisuda"]]
        .drop_duplicates("periode_key")
        .rename(
            columns={
                "Tahun Wisuda": "tahun_wisuda",
                "Periode Num": "periode_num",
                "Periode Wisuda": "periode_label",
            }
        )
        .sort_values(["tahun_wisuda", "periode_num"])
    )

    dim_ipk = (
        working[["ipk_key", "IPK", "Rentang IPK", "Predikat IPK"]]
        .drop_duplicates("ipk_key")
        .rename(columns={"IPK": "ipk_numeric", "Rentang IPK": "rentang_ipk", "Predikat IPK": "predikat_ipk"})
    )

    dim_lama_studi = (
        working[["lama_studi_key", "Lama Studi", "kategori_lama_studi", "flag_tepat_waktu"]]
        .drop_duplicates("lama_studi_key")
        .rename(columns={"Lama Studi": "lama_studi_bulan"})
    )

    dim_tugas_akhir = working[
        [
            "ta_key",
            "Judul Tugas Akhir",
            "Judul Clean",
            "Judul Final",
            "skor_kemiripan_tertinggi",
            "kategori_keunikan",
            "ta_key_termirip",
            "pca_x",
            "pca_y",
        ]
    ].rename(
        columns={
            "Judul Tugas Akhir": "judul_tugas_akhir",
            "Judul Clean": "judul_preprocessed",
            "Judul Final": "judul_final",
        }
    )

    role_columns = [
        ("Pembimbing 1", "Pembimbing", 1),
        ("Pembimbing 2", "Pembimbing", 2),
        ("Dosen Penguji 1", "Penguji", 1),
        ("Dosen Penguji 2", "Penguji", 2),
        ("Dosen Penguji 3", "Penguji", 3),
    ]
    bridge_rows: list[dict[str, Any]] = []
    for _, row in working.iterrows():
        for column, jenis_peran, urutan_peran in role_columns:
            normalized = normalize_person_name(row[column])
            if normalized is None:
                continue
            bridge_rows.append(
                {
                    "kelulusan_key": int(row["kelulusan_key"]),
                    "nama_dosen_normalized": normalized,
                    "nama_asal": str(row[column]).strip(),
                    "jenis_peran": jenis_peran,
                    "urutan_peran": urutan_peran,
                    "role_count": 1,
                }
            )

    bridge_source = pd.DataFrame(bridge_rows)
    dim_dosen = (
        bridge_source[["nama_dosen_normalized", "nama_asal"]]
        .drop_duplicates("nama_dosen_normalized")
        .sort_values("nama_dosen_normalized")
        .reset_index(drop=True)
    )
    dim_dosen.insert(0, "dosen_key", np.arange(1, len(dim_dosen) + 1))

    bridge_peran_dosen = bridge_source.merge(dim_dosen[["dosen_key", "nama_dosen_normalized"]], on="nama_dosen_normalized")
    bridge_peran_dosen = bridge_peran_dosen[
        ["kelulusan_key", "dosen_key", "jenis_peran", "urutan_peran", "role_count"]
    ].reset_index(drop=True)
    bridge_peran_dosen.insert(0, "peran_dosen_key", np.arange(1, len(bridge_peran_dosen) + 1))

    fact_kelulusan = working[
        [
            "kelulusan_key",
            "mahasiswa_key",
            "waktu_key",
            "periode_key",
            "ipk_key",
            "lama_studi_key",
            "ta_key",
            "IPK",
            "Lama Studi",
            "flag_tepat_waktu",
        ]
    ].rename(columns={"IPK": "ipk", "Lama Studi": "lama_studi_bulan"})
    fact_kelulusan["jumlah_record"] = 1

    return {
        "dim_mahasiswa": dim_mahasiswa,
        "dim_waktu": dim_waktu,
        "dim_periode_wisuda": dim_periode_wisuda,
        "dim_ipk": dim_ipk,
        "dim_lama_studi": dim_lama_studi,
        "dim_tugas_akhir": dim_tugas_akhir,
        "dim_dosen": dim_dosen,
        "fact_kelulusan": fact_kelulusan,
        "bridge_peran_dosen": bridge_peran_dosen,
    }


def database_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    return value


def records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {column: database_value(value) for column, value in row.items()}
        for row in df.to_dict("records")
    ]


def insert_table(conn: psycopg.Connection, table: str, df: pd.DataFrame) -> None:
    columns = list(df.columns)
    placeholders = ", ".join(f"%({column})s" for column in columns)
    column_sql = ", ".join(columns)
    sql = f"INSERT INTO {table} ({column_sql}) VALUES ({placeholders})"
    with conn.cursor() as cur:
        cur.executemany(sql, records(df))


def load_database(db_url: str, tables: dict[str, pd.DataFrame], execute_schema: bool) -> None:
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            if execute_schema:
                cur.execute(SCHEMA_SQL.read_text(encoding="utf-8"))
            cur.execute(
                """
                TRUNCATE TABLE
                    bridge_peran_dosen,
                    fact_kelulusan,
                    dim_tugas_akhir,
                    dim_dosen,
                    dim_lama_studi,
                    dim_ipk,
                    dim_periode_wisuda,
                    dim_waktu,
                    dim_mahasiswa
                RESTART IDENTITY CASCADE
                """
            )

        for table in [
            "dim_mahasiswa",
            "dim_waktu",
            "dim_periode_wisuda",
            "dim_ipk",
            "dim_lama_studi",
        ]:
            insert_table(conn, table, tables[table])

        # ta_key_termirip uses a deferrable self-reference, so it is checked at commit.
        insert_table(conn, "dim_tugas_akhir", tables["dim_tugas_akhir"])
        insert_table(conn, "dim_dosen", tables["dim_dosen"])
        insert_table(conn, "fact_kelulusan", tables["fact_kelulusan"])
        insert_table(conn, "bridge_peran_dosen", tables["bridge_peran_dosen"])
        conn.commit()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV tidak ditemukan: {csv_path}")

    print(f"Reading CSV: {csv_path}")
    df = load_clean_csv(csv_path, args.target_months)
    print(f"Rows: {len(df)}")
    print("Computing SBERT embedding and cosine similarity...")
    df = enrich_nlp(df)
    tables = build_tables(df)
    print("Loading PostgreSQL star schema...")
    load_database(args.db_url, tables, execute_schema=not args.skip_schema)
    print("Done.")
    for table_name, table_df in tables.items():
        print(f"{table_name}: {len(table_df)} rows")


if __name__ == "__main__":
    main()
