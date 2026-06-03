from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "riwayatalumniDSI_clean_final.csv"
FALLBACK_DOWNLOADS_PATH = Path.home() / "Downloads" / "riwayatalumniDSI_clean_final (8).csv"
DEFAULT_DB_URL = "postgresql://postgres:password@127.0.0.1:5432/bi_alumni_dsi"

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


@dataclass(frozen=True)
class StarSchema:
    fact_kelulusan: pd.DataFrame
    dim_mahasiswa: pd.DataFrame
    dim_waktu: pd.DataFrame
    dim_periode_wisuda: pd.DataFrame
    dim_ipk: pd.DataFrame
    dim_lama_studi: pd.DataFrame
    dim_tugas_akhir: pd.DataFrame
    dim_dosen: pd.DataFrame
    bridge_peran_dosen: pd.DataFrame


st.set_page_config(
    page_title="Dashboard BI Alumni DSI Unand",
    page_icon="BI",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --surface: #f7f8f4;
            --panel: #eef2ea;
            --ink: #25302a;
            --muted: #67736b;
            --accent: #246b55;
            --accent-soft: #dce9df;
            --line: #d8ded5;
            --warn: #b56c23;
            --danger: #ad3f43;
        }

        .stApp {
            background: var(--surface);
            color: var(--ink);
        }

        [data-testid="stSidebar"] {
            background: var(--panel);
            border-right: 1px solid var(--line);
        }

        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2.5rem;
            max-width: 1420px;
        }

        h1, h2, h3 {
            letter-spacing: 0;
            color: var(--ink);
        }

        h1 {
            font-size: 2rem;
            font-weight: 740;
            margin-bottom: .2rem;
        }

        h2 {
            font-size: 1.35rem;
            font-weight: 700;
            margin-top: 1rem;
        }

        h3 {
            font-size: 1.05rem;
            font-weight: 680;
        }

        [data-testid="stMetric"] {
            background: #fbfcf8;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: .85rem 1rem;
            box-shadow: 0 1px 2px rgba(37, 48, 42, .04);
        }

        [data-testid="stMetricLabel"] {
            color: var(--muted);
        }

        [data-testid="stMetricValue"] {
            color: var(--ink);
            font-size: 1.55rem;
            font-weight: 760;
        }

        div[data-testid="stTabs"] button {
            min-height: 42px;
        }

        div[data-testid="stTabs"] button[aria-selected="true"] {
            border-bottom-color: var(--accent);
            color: var(--accent);
        }

        .section-note {
            color: var(--muted);
            font-size: .93rem;
            line-height: 1.5;
            margin: -.25rem 0 .75rem 0;
        }

        .schema-box {
            background: #fbfcf8;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1rem;
            min-height: 160px;
        }

        .schema-title {
            color: var(--accent);
            font-weight: 760;
            margin-bottom: .45rem;
        }

        .schema-field {
            color: var(--ink);
            font-size: .9rem;
            line-height: 1.55;
        }

        .status-chip {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: .18rem .55rem;
            font-size: .82rem;
            background: var(--accent-soft);
            color: var(--accent);
            border: 1px solid #c8d9cc;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def normalize_person_name(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    if text in {"", "-", "nan", "none"}:
        return None
    return LECTURER_ALIASES.get(text, text)


def clean_title(text: object) -> str:
    if pd.isna(text):
        return ""
    normalized = unicodedata.normalize("NFKD", str(text).lower())
    normalized = re.sub(r"[^a-z\s]", " ", normalized)
    return " ".join(normalized.split())


def final_title(text: object) -> str:
    tokens = clean_title(text).split()
    filtered = [token for token in tokens if token not in STOPWORDS_UNAND]
    return " ".join(filtered)


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


def kategori_lama_studi(months: int, target_months: int = 48) -> str:
    if months <= target_months:
        return "Tepat Waktu"
    if months <= target_months + 6:
        return "Melewati Target <= 6 Bulan"
    if months <= 72:
        return "Terlambat"
    return "Sangat Terlambat"


def similarity_category(score: float) -> str:
    if score >= 0.85:
        return "Sangat Mirip / Redundan"
    if score >= 0.65:
        return "Cukup Mirip / Perlu Review"
    return "Relatif Unik"


def uniqueness_category(score: float) -> str:
    if score >= 0.85:
        return "Tidak Unik"
    if score >= 0.70:
        return "Agak Unik / Perlu Review"
    return "Unik"


def get_data_path() -> Path:
    if DATA_PATH.exists():
        return DATA_PATH
    return FALLBACK_DOWNLOADS_PATH


def get_database_url() -> str:
    return os.getenv("BI_DATABASE_URL", DEFAULT_DB_URL)


def fetch_dataframe(query: str) -> pd.DataFrame:
    with psycopg.connect(get_database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
            columns = [description.name for description in cur.description]
    return pd.DataFrame(rows, columns=columns)


@st.cache_data(show_spinner=False, ttl=60)
def load_dashboard_data_from_db() -> pd.DataFrame:
    db_df = fetch_dataframe("SELECT * FROM vw_dashboard_kelulusan ORDER BY kelulusan_key")
    if db_df.empty:
        return pd.DataFrame()

    df = pd.DataFrame(
        {
            "NO": db_df["kelulusan_key"],
            "Nama": db_df["nama"],
            "NIM": db_df["nim"],
            "Judul Tugas Akhir": db_df["judul_tugas_akhir"],
            "Periode Wisuda": db_df["periode_label"],
            "Tahun Wisuda": db_df["tahun_wisuda"],
            "Tanggal Parsed": pd.to_datetime(db_df["tanggal_lulus"], errors="coerce"),
            "IPK": pd.to_numeric(db_df["ipk"], errors="coerce"),
            "Lama Studi": pd.to_numeric(db_df["lama_studi_bulan"], errors="coerce"),
            "Periode Num": pd.to_numeric(db_df["periode_num"], errors="coerce").fillna(99).astype(int),
            "Judul Clean": db_df["judul_preprocessed"],
            "Judul Final": db_df["judul_final"],
            "Predikat IPK": db_df["predikat_ipk"],
            "Rentang IPK": db_df["rentang_ipk"],
            "kategori_keunikan": db_df["kategori_keunikan"],
            "skor_kemiripan_tertinggi": pd.to_numeric(db_df["skor_kemiripan_tertinggi"], errors="coerce"),
            "ta_key": db_df["ta_key"],
            "ta_key_termirip": db_df["ta_key_termirip"],
            "judul_termirip": db_df["judul_termirip"],
            "pemilik_judul_termirip": db_df["pemilik_judul_termirip"],
            "pca_x": pd.to_numeric(db_df["pca_x"], errors="coerce"),
            "pca_y": pd.to_numeric(db_df["pca_y"], errors="coerce"),
        }
    )
    df["Tanggal Lulus"] = df["Tanggal Parsed"].dt.strftime("%d %B %Y")
    df["Periode Sort"] = df["Tahun Wisuda"].astype(str) + "-" + df["Periode Num"].astype(str).str.zfill(2)
    df["kategori_kemiripan"] = df["skor_kemiripan_tertinggi"].apply(similarity_category)
    return df


@st.cache_data(show_spinner=False, ttl=60)
def load_roles_data_from_db() -> pd.DataFrame:
    role_df = fetch_dataframe("SELECT * FROM vw_dashboard_peran_dosen ORDER BY peran_dosen_key")
    if role_df.empty:
        return pd.DataFrame()
    role_df = role_df.rename(
        columns={
            "tahun_wisuda": "Tahun Wisuda",
            "periode_label": "Periode Wisuda",
            "periode_num": "Periode Num",
            "ipk": "IPK",
        }
    )
    role_df["Tahun Wisuda"] = pd.to_numeric(role_df["Tahun Wisuda"], errors="coerce")
    role_df["Periode Num"] = pd.to_numeric(role_df["Periode Num"], errors="coerce").fillna(99).astype(int)
    role_df["IPK"] = pd.to_numeric(role_df["IPK"], errors="coerce")
    return role_df


@st.cache_data(show_spinner=False, ttl=60)
def load_star_schema_from_db() -> StarSchema:
    return StarSchema(
        fact_kelulusan=fetch_dataframe("SELECT * FROM fact_kelulusan ORDER BY kelulusan_key"),
        dim_mahasiswa=fetch_dataframe("SELECT * FROM dim_mahasiswa ORDER BY mahasiswa_key"),
        dim_waktu=fetch_dataframe("SELECT * FROM dim_waktu ORDER BY waktu_key"),
        dim_periode_wisuda=fetch_dataframe("SELECT * FROM dim_periode_wisuda ORDER BY tahun_wisuda, periode_num"),
        dim_ipk=fetch_dataframe("SELECT * FROM dim_ipk ORDER BY ipk_key"),
        dim_lama_studi=fetch_dataframe("SELECT * FROM dim_lama_studi ORDER BY lama_studi_key"),
        dim_tugas_akhir=fetch_dataframe("SELECT * FROM dim_tugas_akhir ORDER BY ta_key"),
        dim_dosen=fetch_dataframe("SELECT * FROM dim_dosen ORDER BY dosen_key"),
        bridge_peran_dosen=fetch_dataframe("SELECT * FROM bridge_peran_dosen ORDER BY peran_dosen_key"),
    )


def render_database_setup(error: Exception) -> None:
    st.title("Dashboard BI Alumni DSI Unand")
    st.error("Dashboard belum bisa membaca PostgreSQL star schema.")
    st.markdown(
        f"""
        App ini sekarang mengikuti alur BI final: CSV bersih dan hasil NLP SBERT dimuat dulu ke PostgreSQL star schema,
        lalu dashboard membaca dari database.

        **Koneksi yang dicoba**
        `{get_database_url()}`

        **Error**
        `{error}`

        **Langkah setup**
        ```powershell
        # 1. Buat database PostgreSQL bernama bi_alumni_dsi
        # 2. Set connection string bila user/password berbeda
        $env:BI_DATABASE_URL="postgresql://postgres:postgres@localhost:5432/bi_alumni_dsi"

        # 3. Load CSV + NLP SBERT + star schema
        .\\.venv\\Scripts\\python.exe scripts\\load_star_schema.py

        # 4. Jalankan dashboard
        .\\.venv\\Scripts\\python.exe -m streamlit run app.py
        ```
        """
    )


@st.cache_data(show_spinner=False)
def load_raw_data() -> pd.DataFrame:
    path = get_data_path()
    df = pd.read_csv(path)
    df.columns = [col.strip() for col in df.columns]
    df["Tanggal Parsed"] = df["Tanggal Lulus"].apply(parse_tanggal_lulus)
    df["IPK"] = pd.to_numeric(df["IPK"], errors="coerce")
    df["Lama Studi"] = pd.to_numeric(df["Lama Studi"], errors="coerce").astype("Int64")
    df["Tahun Wisuda"] = pd.to_numeric(df["Tahun Wisuda"], errors="coerce").astype("Int64")
    df["Periode Num"] = df["Periode Wisuda"].map(PERIODE_ORDER).fillna(99).astype(int)
    df["Periode Sort"] = df["Tahun Wisuda"].astype(str) + "-" + df["Periode Num"].astype(str).str.zfill(2)
    df["Judul Clean"] = df["Judul Tugas Akhir"].apply(clean_title)
    df["Judul Final"] = df["Judul Tugas Akhir"].apply(final_title)
    df["Predikat IPK"] = df["IPK"].apply(predikat_ipk)
    df["Rentang IPK"] = df["IPK"].apply(rentang_ipk)
    return df


@st.cache_data(show_spinner=False)
def build_star_schema(df: pd.DataFrame, target_months: int) -> StarSchema:
    working = df.reset_index(drop=True).copy()
    working["kelulusan_key"] = np.arange(1, len(working) + 1)
    working["mahasiswa_key"] = np.arange(1, len(working) + 1)
    working["waktu_key"] = pd.factorize(working["Tanggal Parsed"].astype(str))[0] + 1
    working["periode_key"] = pd.factorize(working[["Tahun Wisuda", "Periode Wisuda"]].astype(str).agg("|".join, axis=1))[0] + 1
    working["ipk_key"] = pd.factorize(working["Rentang IPK"].astype(str) + "|" + working["Predikat IPK"].astype(str))[0] + 1
    working["lama_studi_key"] = pd.factorize(working["Lama Studi"].astype(str))[0] + 1
    working["ta_key"] = np.arange(1, len(working) + 1)
    working["flag_tepat_waktu"] = working["Lama Studi"].le(target_months)
    working["kategori_lama_studi"] = working["Lama Studi"].apply(lambda x: kategori_lama_studi(int(x), target_months))

    dim_mahasiswa = working[["mahasiswa_key", "NIM", "Nama"]].rename(columns={"NIM": "nim", "Nama": "nama"})

    dim_waktu = (
        working[["waktu_key", "Tanggal Parsed"]]
        .drop_duplicates("waktu_key")
        .assign(
            tanggal_lulus=lambda x: x["Tanggal Parsed"].dt.date,
            hari=lambda x: x["Tanggal Parsed"].dt.day,
            bulan=lambda x: x["Tanggal Parsed"].dt.month,
            tahun=lambda x: x["Tanggal Parsed"].dt.year,
        )
        [["waktu_key", "tanggal_lulus", "hari", "bulan", "tahun"]]
        .sort_values("waktu_key")
    )

    dim_periode_wisuda = (
        working[["periode_key", "Tahun Wisuda", "Periode Wisuda", "Periode Num"]]
        .drop_duplicates("periode_key")
        .rename(
            columns={
                "Tahun Wisuda": "tahun_wisuda",
                "Periode Wisuda": "periode_label",
                "Periode Num": "periode_num",
            }
        )
        .sort_values(["tahun_wisuda", "periode_num"])
    )

    dim_ipk = (
        working[["ipk_key", "IPK", "Rentang IPK", "Predikat IPK"]]
        .drop_duplicates("ipk_key")
        .rename(columns={"IPK": "ipk_numeric", "Rentang IPK": "rentang_ipk", "Predikat IPK": "predikat_ipk"})
        .sort_values("ipk_key")
    )

    dim_lama_studi = (
        working[["lama_studi_key", "Lama Studi", "kategori_lama_studi", "flag_tepat_waktu"]]
        .drop_duplicates("lama_studi_key")
        .rename(columns={"Lama Studi": "lama_studi_bulan"})
        .sort_values("lama_studi_key")
    )

    dim_tugas_akhir = working[
        ["ta_key", "Judul Tugas Akhir", "Judul Clean", "Judul Final"]
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
    bridge_rows: list[dict[str, object]] = []
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

    bridge = pd.DataFrame(bridge_rows)
    if bridge.empty:
        dim_dosen = pd.DataFrame(columns=["dosen_key", "nama_dosen_normalized", "nama_asal"])
        bridge_peran_dosen = pd.DataFrame(
            columns=["peran_dosen_key", "kelulusan_key", "dosen_key", "jenis_peran", "urutan_peran", "role_count"]
        )
    else:
        dim_dosen = (
            bridge[["nama_dosen_normalized", "nama_asal"]]
            .drop_duplicates("nama_dosen_normalized")
            .sort_values("nama_dosen_normalized")
            .reset_index(drop=True)
        )
        dim_dosen.insert(0, "dosen_key", np.arange(1, len(dim_dosen) + 1))
        bridge_peran_dosen = bridge.merge(dim_dosen[["dosen_key", "nama_dosen_normalized"]], on="nama_dosen_normalized")
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

    return StarSchema(
        fact_kelulusan=fact_kelulusan,
        dim_mahasiswa=dim_mahasiswa,
        dim_waktu=dim_waktu,
        dim_periode_wisuda=dim_periode_wisuda,
        dim_ipk=dim_ipk,
        dim_lama_studi=dim_lama_studi,
        dim_tugas_akhir=dim_tugas_akhir,
        dim_dosen=dim_dosen,
        bridge_peran_dosen=bridge_peran_dosen,
    )


def filter_dataframe(df: pd.DataFrame, years: tuple[int, int], periode: list[str], ipk_range: tuple[float, float]) -> pd.DataFrame:
    filtered = df[
        df["Tahun Wisuda"].between(years[0], years[1])
        & df["Periode Wisuda"].isin(periode)
        & df["IPK"].between(ipk_range[0], ipk_range[1])
    ].copy()
    return filtered


def filter_roles_dataframe(role_df: pd.DataFrame, years: tuple[int, int], periode: list[str], ipk_range: tuple[float, float]) -> pd.DataFrame:
    if role_df.empty:
        return role_df
    return role_df[
        role_df["Tahun Wisuda"].between(years[0], years[1])
        & role_df["Periode Wisuda"].isin(periode)
        & role_df["IPK"].between(ipk_range[0], ipk_range[1])
    ].copy()


def top_keywords(titles: Iterable[str], n: int = 20) -> pd.DataFrame:
    counter: dict[str, int] = {}
    for title in titles:
        for token in str(title).split():
            if len(token) <= 2:
                continue
            counter[token] = counter.get(token, 0) + 1
    return pd.DataFrame(sorted(counter.items(), key=lambda item: item[1], reverse=True)[:n], columns=["Kata", "Frekuensi"])


def plot_empty(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    fig.update_layout(height=280, xaxis_visible=False, yaxis_visible=False, margin=dict(l=20, r=20, t=20, b=20))
    return fig


def academic_dashboard(df: pd.DataFrame, target_months: int) -> None:
    total = len(df)
    avg_ipk = df["IPK"].mean()
    avg_study = df["Lama Studi"].mean()
    on_time = df["Lama Studi"].le(target_months).mean() * 100 if total else 0
    latest_year = int(df["Tahun Wisuda"].max()) if total else "-"

    kpi_cols = st.columns(5)
    kpi_cols[0].metric("Total Alumni", f"{total:,}".replace(",", "."))
    kpi_cols[1].metric("Rata-rata IPK", f"{avg_ipk:.2f}" if total else "-")
    kpi_cols[2].metric("Rata-rata Lama Studi", f"{avg_study:.1f} bulan" if total else "-")
    kpi_cols[3].metric("Tepat Waktu", f"{on_time:.1f}%")
    kpi_cols[4].metric("Tahun Terbaru", latest_year)

    if df.empty:
        st.plotly_chart(plot_empty("Tidak ada data pada filter ini."), use_container_width=True)
        return

    yearly = (
        df.groupby("Tahun Wisuda", as_index=False)
        .agg(jumlah_lulusan=("NO", "count"), rata_ipk=("IPK", "mean"), rata_lama_studi=("Lama Studi", "mean"))
        .sort_values("Tahun Wisuda")
    )

    period = (
        df.groupby(["Tahun Wisuda", "Periode Wisuda", "Periode Num"], as_index=False)
        .agg(jumlah_lulusan=("NO", "count"), rata_ipk=("IPK", "mean"))
        .sort_values(["Tahun Wisuda", "Periode Num"])
    )
    period["label"] = period["Tahun Wisuda"].astype(str) + " " + period["Periode Wisuda"].str.replace("WISUDA ", "W")

    col_left, col_right = st.columns([1.35, 1])
    with col_left:
        fig = px.line(
            yearly,
            x="Tahun Wisuda",
            y="jumlah_lulusan",
            markers=True,
            title="Tren Jumlah Lulusan per Tahun",
            labels={"jumlah_lulusan": "Jumlah Lulusan", "Tahun Wisuda": "Tahun Wisuda"},
        )
        fig.update_traces(line_color="#246b55", marker_size=8)
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        fig = px.bar(
            df["Rentang IPK"].value_counts().rename_axis("Rentang IPK").reset_index(name="Jumlah"),
            x="Rentang IPK",
            y="Jumlah",
            title="Distribusi Rentang IPK",
            color="Rentang IPK",
            color_discrete_sequence=["#6f8f72", "#246b55", "#5b7894", "#a88349", "#ad6b6e"],
        )
        fig.update_layout(height=360, showlegend=False, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)

    col_left, col_right = st.columns([1.1, 1])
    with col_left:
        fig = px.bar(
            period.tail(40),
            x="label",
            y="jumlah_lulusan",
            color="rata_ipk",
            title="Lulusan per Periode Wisuda",
            labels={"label": "Periode", "jumlah_lulusan": "Jumlah", "rata_ipk": "Rata-rata IPK"},
            color_continuous_scale=["#e6e2cc", "#5f8b6b", "#245c56"],
        )
        fig.update_layout(height=390, margin=dict(l=10, r=10, t=55, b=95), xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        study = df.assign(Kategori=df["Lama Studi"].apply(lambda x: kategori_lama_studi(int(x), target_months)))
        study_counts = study["Kategori"].value_counts().rename_axis("Kategori").reset_index(name="Jumlah")
        fig = px.pie(
            study_counts,
            names="Kategori",
            values="Jumlah",
            title="Komposisi Lama Studi",
            hole=.48,
            color_discrete_sequence=["#246b55", "#b7a05a", "#c47c43", "#ad3f43"],
        )
        fig.update_layout(height=390, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(
        df,
        x="Lama Studi",
        y="IPK",
        color="Tahun Wisuda",
        custom_data=["Nama", "NIM", "Periode Wisuda", "Tahun Wisuda", "Judul Tugas Akhir"],
        title="Sebaran IPK dan Lama Studi Alumni",
        labels={"Lama Studi": "Lama Studi (bulan)", "IPK": "IPK"},
        color_continuous_scale=["#b8c5a3", "#4f7f70", "#203b5b"],
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "NIM: %{customdata[1]}<br>"
            "Periode: %{customdata[2]} %{customdata[3]}<br>"
            "Lama Studi: %{x} bulan<br>"
            "IPK: %{y:.2f}<br>"
            "Judul: %{customdata[4]}<extra></extra>"
        )
    )
    fig.add_vline(x=target_months, line_dash="dash", line_color="#ad3f43", annotation_text=f"Target {target_months} bulan")
    fig.update_layout(height=460, margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Audit Data Alumni"):
        query = st.text_input("Cari nama/NIM", placeholder="Contoh: Fitrah Annisa Sari")
        audit_df = df.copy()
        if query.strip():
            pattern = query.strip().lower()
            audit_df = audit_df[
                audit_df["Nama"].astype(str).str.lower().str.contains(pattern, na=False)
                | audit_df["NIM"].astype(str).str.lower().str.contains(pattern, na=False)
            ]
        st.dataframe(
            audit_df[
                [
                    "NO",
                    "Nama",
                    "NIM",
                    "Periode Wisuda",
                    "Tahun Wisuda",
                    "Tanggal Lulus",
                    "IPK",
                    "Lama Studi",
                    "Judul Tugas Akhir",
                ]
            ].head(50),
            use_container_width=True,
            hide_index=True,
        )


def lecturer_dashboard(df: pd.DataFrame, schema: StarSchema) -> None:
    st.markdown('<p class="section-note">Analisis dosen memakai bridge table agar satu alumni dapat terhubung ke beberapa pembimbing dan penguji.</p>', unsafe_allow_html=True)
    if schema.bridge_peran_dosen.empty:
        st.info("Tidak ada data peran dosen pada filter ini.")
        return

    bridge = schema.bridge_peran_dosen.merge(schema.dim_dosen, on="dosen_key", how="left")
    total_roles = len(bridge)
    unique_lecturers = bridge["dosen_key"].nunique()
    advisor_roles = bridge["jenis_peran"].eq("Pembimbing").sum()
    examiner_roles = bridge["jenis_peran"].eq("Penguji").sum()

    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Total Relasi Dosen", f"{total_roles:,}".replace(",", "."))
    kpi_cols[1].metric("Dosen Unik", unique_lecturers)
    kpi_cols[2].metric("Peran Pembimbing", advisor_roles)
    kpi_cols[3].metric("Peran Penguji", examiner_roles)

    top_all = (
        bridge.groupby(["dosen_key", "nama_dosen_normalized"], as_index=False)
        .agg(jumlah_peran=("role_count", "sum"))
        .sort_values("jumlah_peran", ascending=False)
        .head(15)
    )
    top_role = (
        bridge.groupby(["nama_dosen_normalized", "jenis_peran"], as_index=False)
        .agg(jumlah=("role_count", "sum"))
        .sort_values("jumlah", ascending=False)
    )

    col_left, col_right = st.columns([1.2, 1])
    with col_left:
        fig = px.bar(
            top_all.sort_values("jumlah_peran"),
            x="jumlah_peran",
            y="nama_dosen_normalized",
            orientation="h",
            title="Top Dosen Berdasarkan Total Peran",
            labels={"jumlah_peran": "Jumlah Peran", "nama_dosen_normalized": "Dosen"},
            color_discrete_sequence=["#246b55"],
        )
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        fig = px.bar(
            top_role.head(24),
            x="jumlah",
            y="nama_dosen_normalized",
            color="jenis_peran",
            orientation="h",
            title="Komposisi Peran Pembimbing dan Penguji",
            labels={"jumlah": "Jumlah", "nama_dosen_normalized": "Dosen"},
            color_discrete_map={"Pembimbing": "#246b55", "Penguji": "#a88349"},
        )
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=55, b=10), yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        top_role.rename(
            columns={
                "nama_dosen_normalized": "Dosen",
                "jenis_peran": "Jenis Peran",
                "jumlah": "Jumlah Relasi",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def lecturer_dashboard_from_roles(role_df: pd.DataFrame) -> None:
    st.markdown(
        '<p class="section-note">Analisis dosen memakai vw_dashboard_peran_dosen yang berasal dari bridge_peran_dosen pada PostgreSQL star schema.</p>',
        unsafe_allow_html=True,
    )
    if role_df.empty:
        st.info("Tidak ada data peran dosen pada filter ini.")
        return

    total_roles = int(role_df["role_count"].sum())
    total_alumni = role_df["kelulusan_key"].nunique()
    unique_lecturers = role_df["dosen_key"].nunique()
    advisor_roles = int(role_df.loc[role_df["jenis_peran"].eq("Pembimbing"), "role_count"].sum())
    examiner_roles = int(role_df.loc[role_df["jenis_peran"].eq("Penguji"), "role_count"].sum())

    kpi_cols = st.columns(5)
    kpi_cols[0].metric("Total Relasi Peran", f"{total_roles:,}".replace(",", "."))
    kpi_cols[1].metric("Alumni Terlibat", f"{total_alumni:,}".replace(",", "."))
    kpi_cols[2].metric("Dosen Unik", unique_lecturers)
    kpi_cols[3].metric("Peran Pembimbing", advisor_roles)
    kpi_cols[4].metric("Peran Penguji", examiner_roles)

    top_all = (
        role_df.groupby(["dosen_key", "nama_dosen_normalized"], as_index=False)
        .agg(jumlah_peran=("role_count", "sum"), jumlah_alumni=("kelulusan_key", "nunique"))
        .sort_values("jumlah_peran", ascending=False)
        .head(15)
    )
    top_role = (
        role_df.groupby(["dosen_key", "nama_dosen_normalized", "jenis_peran"], as_index=False)
        .agg(jumlah=("role_count", "sum"))
        .sort_values("jumlah", ascending=False)
    )
    top_role_chart = top_role[top_role["dosen_key"].isin(top_all["dosen_key"])].merge(
        top_all[["dosen_key", "jumlah_peran"]], on="dosen_key", how="left"
    )

    col_left, col_right = st.columns([1.2, 1])
    with col_left:
        fig = px.bar(
            top_role_chart.sort_values("jumlah_peran"),
            x="jumlah",
            y="nama_dosen_normalized",
            color="jenis_peran",
            orientation="h",
            title="Top Dosen Berdasarkan Total Relasi Peran",
            labels={"jumlah": "Jumlah Peran", "nama_dosen_normalized": "Dosen", "jenis_peran": "Jenis Peran"},
            color_discrete_map={"Pembimbing": "#246b55", "Penguji": "#a88349"},
        )
        fig.update_layout(
            height=520,
            margin=dict(l=10, r=10, t=55, b=10),
            barmode="stack",
            yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        fig = px.bar(
            top_all.sort_values("jumlah_peran"),
            x=["jumlah_peran", "jumlah_alumni"],
            y="nama_dosen_normalized",
            orientation="h",
            title="Perbandingan Total Peran vs Alumni Unik",
            labels={"value": "Jumlah", "nama_dosen_normalized": "Dosen", "variable": "Metrik"},
            color_discrete_sequence=["#246b55", "#5b7894"],
        )
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=55, b=10), yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    table = (
        top_role.pivot_table(
            index=["dosen_key", "nama_dosen_normalized"],
            columns="jenis_peran",
            values="jumlah",
            fill_value=0,
            aggfunc="sum",
        )
        .reset_index()
        .merge(top_all[["dosen_key", "jumlah_peran", "jumlah_alumni"]], on="dosen_key", how="right")
        .sort_values("jumlah_peran", ascending=False)
    )
    for column in ["Pembimbing", "Penguji"]:
        if column not in table.columns:
            table[column] = 0

    st.dataframe(
        table[
            ["nama_dosen_normalized", "jumlah_peran", "jumlah_alumni", "Pembimbing", "Penguji"]
        ].rename(
            columns={
                "nama_dosen_normalized": "Dosen",
                "jumlah_peran": "Total Relasi Peran",
                "jumlah_alumni": "Alumni Unik",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def nlp_dashboard(df: pd.DataFrame) -> None:
    st.markdown(
        '<p class="section-note">Tab ini membaca atribut NLP dari dim_tugas_akhir: judul hasil preprocessing, skor SBERT + cosine similarity tertinggi, kategori keunikan, judul termirip, dan koordinat PCA.</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<span class="status-chip">NLP source: dim_tugas_akhir, SBERT + cosine similarity</span>', unsafe_allow_html=True)

    if df.empty or df["skor_kemiripan_tertinggi"].isna().all():
        st.info("Tidak ada atribut NLP pada dim_tugas_akhir untuk filter ini.")
        return

    df_nlp = df.copy()
    avg_similarity = df_nlp["skor_kemiripan_tertinggi"].mean()
    uniqueness_index = (1 - avg_similarity) * 100
    redundant_count = df_nlp["skor_kemiripan_tertinggi"].ge(0.85).sum()
    review_count = df_nlp["skor_kemiripan_tertinggi"].between(0.70, 0.8499).sum()

    kpi_cols = st.columns(4)
    kpi_cols[0].metric("Rata-rata Similarity Tertinggi", f"{avg_similarity:.3f}")
    kpi_cols[1].metric("Indeks Keunikan", f"{uniqueness_index:.1f}/100")
    kpi_cols[2].metric("Tidak Unik", int(redundant_count))
    kpi_cols[3].metric("Perlu Review", int(review_count))

    heatmap_size = st.slider(
        "Jumlah pasangan dalam heatmap NLP",
        min_value=8,
        max_value=min(30, len(df_nlp)),
        value=min(18, len(df_nlp)),
        help="Diurutkan dari judul dengan skor SBERT cosine similarity tertinggi.",
    )
    heatmap_df = df_nlp.sort_values("skor_kemiripan_tertinggi", ascending=False).head(heatmap_size).copy()
    heatmap_df["Judul"] = heatmap_df["Nama"].astype(str) + " | " + heatmap_df["Judul Tugas Akhir"].astype(str).str.slice(0, 54)
    heatmap_df["Judul Termirip"] = (
        heatmap_df["pemilik_judul_termirip"].fillna("-").astype(str)
        + " | "
        + heatmap_df["judul_termirip"].fillna("-").astype(str).str.slice(0, 54)
    )
    heatmap_matrix = heatmap_df.pivot_table(
        index="Judul",
        columns="Judul Termirip",
        values="skor_kemiripan_tertinggi",
        aggfunc="max",
    )
    fig = px.imshow(
        heatmap_matrix,
        zmin=0,
        zmax=1,
        color_continuous_scale=["#f3f1e7", "#c9d9c8", "#6b9a7b", "#ad3f43"],
        title="Heatmap NLP dari dim_tugas_akhir: Judul dan Judul Termirip",
        labels={"x": "Judul Termirip", "y": "Judul TA", "color": "Skor"},
        aspect="auto",
    )
    fig.update_layout(height=620, margin=dict(l=10, r=10, t=60, b=145), xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    col_left, col_right = st.columns([1.2, 1])
    with col_left:
        fig = px.histogram(
            df_nlp,
            x="skor_kemiripan_tertinggi",
            nbins=30,
            color="kategori_keunikan",
            title="Distribusi Skor SBERT Cosine Similarity Tertinggi",
            labels={"skor_kemiripan_tertinggi": "Skor Cosine Similarity", "count": "Jumlah Judul"},
            color_discrete_map={
                "Unik": "#246b55",
                "Agak Unik / Perlu Review": "#b56c23",
                "Tidak Unik": "#ad3f43",
            },
        )
        fig.add_vline(x=0.70, line_dash="dash", line_color="#b56c23", annotation_text="Review 0.70")
        fig.add_vline(x=0.85, line_dash="dash", line_color="#ad3f43", annotation_text="Tidak unik 0.85")
        fig.update_layout(height=410, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        fig = go.Figure()
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=uniqueness_index,
                title={"text": "Indeks Keunikan Dataset TA"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#246b55"},
                    "steps": [
                        {"range": [0, 40], "color": "#efd2cf"},
                        {"range": [40, 70], "color": "#eee0b6"},
                        {"range": [70, 100], "color": "#d8e6d5"},
                    ],
                    "threshold": {"line": {"color": "#ad3f43", "width": 4}, "value": uniqueness_index},
                },
            )
        )
        fig.update_layout(height=410, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)

    col_left, col_right = st.columns([1.1, 1])
    with col_left:
        scatter_df = df_nlp.dropna(subset=["pca_x", "pca_y"]).copy()
        if scatter_df.empty:
            st.plotly_chart(plot_empty("Koordinat PCA belum tersedia di dim_tugas_akhir."), use_container_width=True)
        else:
            fig = px.scatter(
                scatter_df,
                x="pca_x",
                y="pca_y",
                color="kategori_keunikan",
                hover_data=["Nama", "Judul Tugas Akhir", "skor_kemiripan_tertinggi", "judul_termirip"],
                title="Peta Kedekatan Judul TA dari Koordinat PCA SBERT",
                color_discrete_map={
                    "Unik": "#246b55",
                    "Agak Unik / Perlu Review": "#b56c23",
                    "Tidak Unik": "#ad3f43",
                },
            )
            fig.update_layout(height=460, margin=dict(l=10, r=10, t=55, b=10))
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        keywords = top_keywords(df_nlp["Judul Final"], 18)
        fig = px.bar(
            keywords.sort_values("Frekuensi"),
            x="Frekuensi",
            y="Kata",
            orientation="h",
            title="Kata Kunci Dominan Setelah Stopword Removal",
            color_discrete_sequence=["#5b7894"],
        )
        fig.update_layout(height=460, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Judul Paling Mirip Berdasarkan dim_tugas_akhir")
    top_pairs = df_nlp.sort_values("skor_kemiripan_tertinggi", ascending=False)[
        [
            "Nama",
            "Judul Tugas Akhir",
            "skor_kemiripan_tertinggi",
            "pemilik_judul_termirip",
            "judul_termirip",
            "kategori_keunikan",
        ]
    ].head(25)
    st.dataframe(top_pairs.style.format({"skor_kemiripan_tertinggi": "{:.4f}"}), use_container_width=True, hide_index=True)


def star_schema_dashboard(schema: StarSchema) -> None:
    st.markdown(
        '<p class="section-note">Model dimensional mengikuti grain: satu baris fact mewakili satu mahasiswa yang lulus pada satu periode wisuda.</p>',
        unsafe_allow_html=True,
    )
    table_counts = pd.DataFrame(
        [
            ("fact_kelulusan", len(schema.fact_kelulusan)),
            ("dim_mahasiswa", len(schema.dim_mahasiswa)),
            ("dim_waktu", len(schema.dim_waktu)),
            ("dim_periode_wisuda", len(schema.dim_periode_wisuda)),
            ("dim_ipk", len(schema.dim_ipk)),
            ("dim_lama_studi", len(schema.dim_lama_studi)),
            ("dim_tugas_akhir", len(schema.dim_tugas_akhir)),
            ("dim_dosen", len(schema.dim_dosen)),
            ("bridge_peran_dosen", len(schema.bridge_peran_dosen)),
        ],
        columns=["Tabel", "Jumlah Baris"],
    )
    cols = st.columns(3)
    cols[0].metric("Fact Kelulusan", len(schema.fact_kelulusan))
    cols[1].metric("Dimensi", 7)
    cols[2].metric("Bridge Peran Dosen", len(schema.bridge_peran_dosen))

    schema_cols = st.columns(3)
    boxes = [
        ("fact_kelulusan", ["PK kelulusan_key", "FK mahasiswa_key", "FK waktu_key", "FK periode_key", "FK ipk_key", "FK lama_studi_key", "FK ta_key", "Measure: ipk, lama_studi_bulan, jumlah_record"]),
        ("dim_mahasiswa", ["PK mahasiswa_key", "nim", "nama"]),
        ("dim_waktu", ["PK waktu_key", "tanggal_lulus", "hari", "bulan", "tahun"]),
        ("dim_periode_wisuda", ["PK periode_key", "tahun_wisuda", "periode_num", "periode_label"]),
        ("dim_ipk", ["PK ipk_key", "ipk_numeric", "rentang_ipk", "predikat_ipk"]),
        ("dim_lama_studi", ["PK lama_studi_key", "lama_studi_bulan", "kategori_lama_studi", "flag_tepat_waktu"]),
        ("dim_tugas_akhir", ["PK ta_key", "judul_tugas_akhir", "judul_preprocessed", "judul_final", "skor_kemiripan_tertinggi", "kategori_keunikan", "ta_key_termirip", "pca_x, pca_y"]),
        ("dim_dosen", ["PK dosen_key", "nama_dosen_normalized", "nama_asal"]),
        ("bridge_peran_dosen", ["PK peran_dosen_key", "FK kelulusan_key", "FK dosen_key", "jenis_peran", "urutan_peran"]),
    ]
    for index, (title, fields) in enumerate(boxes):
        with schema_cols[index % 3]:
            st.markdown(
                f"""
                <div class="schema-box">
                    <div class="schema-title">{title}</div>
                    <div class="schema-field">{'<br>'.join(fields)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.subheader("Ukuran Tabel Star Schema")
    st.dataframe(table_counts, use_container_width=True, hide_index=True)

    sample_choice = st.selectbox(
        "Lihat sample tabel",
        [
            "fact_kelulusan",
            "dim_mahasiswa",
            "dim_waktu",
            "dim_periode_wisuda",
            "dim_ipk",
            "dim_lama_studi",
            "dim_tugas_akhir",
            "dim_dosen",
            "bridge_peran_dosen",
        ],
    )
    sample_df = getattr(schema, sample_choice)
    st.dataframe(sample_df.head(50), use_container_width=True, hide_index=True)


def data_quality_dashboard(df: pd.DataFrame, schema: StarSchema, source_label: str) -> None:
    st.markdown(
        f'<p class="section-note">Sumber aktif: {source_label}. Validasi ini membaca view dan tabel PostgreSQL star schema.</p>',
        unsafe_allow_html=True,
    )
    missing = df.isna().sum().rename_axis("Kolom").reset_index(name="Jumlah Kosong")
    duplicate_nim = df["NIM"].duplicated(keep=False).sum()
    invalid_dates = df["Tanggal Parsed"].isna().sum()
    invalid_ipk = (~df["IPK"].between(0, 4)).sum()

    cols = st.columns(4)
    cols[0].metric("Record Dashboard View", len(df))
    cols[1].metric("NIM Duplikat", int(duplicate_nim))
    cols[2].metric("Tanggal Gagal Parse", int(invalid_dates))
    cols[3].metric("IPK di Luar 0-4", int(invalid_ipk))

    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.subheader("Missing Value")
        st.dataframe(missing, use_container_width=True, hide_index=True)
    with col_right:
        role_summary = (
            schema.bridge_peran_dosen.groupby(["jenis_peran", "urutan_peran"], as_index=False)
            .agg(jumlah_relasi=("role_count", "sum"))
            .sort_values(["jenis_peran", "urutan_peran"])
        )
        st.subheader("Relasi Bridge Peran Dosen")
        st.dataframe(role_summary, use_container_width=True, hide_index=True)

    st.subheader("Preview Data Alumni")
    preview_cols = [
        "Nama",
        "NIM",
        "Periode Wisuda",
        "Tahun Wisuda",
        "Tanggal Lulus",
        "IPK",
        "Lama Studi",
        "Judul Tugas Akhir",
    ]
    st.dataframe(df[preview_cols].head(150), use_container_width=True, hide_index=True)


def render_header(df: pd.DataFrame) -> None:
    min_year = int(df["Tahun Wisuda"].min())
    max_year = int(df["Tahun Wisuda"].max())
    st.title("Dashboard BI Alumni DSI Unand")
    st.markdown(
        f"""
        <p class="section-note">
        Data mart akademik alumni Departemen Sistem Informasi Universitas Andalas, rentang wisuda {min_year}-{max_year},
        dengan tambahan analisis NLP similarity judul tugas akhir.
        </p>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_css()

    st.sidebar.header("Database")
    if st.sidebar.button("Refresh data dari PostgreSQL"):
        st.cache_data.clear()
        st.rerun()

    try:
        df = load_dashboard_data_from_db()
        role_df = load_roles_data_from_db()
        schema = load_star_schema_from_db()
    except Exception as exc:
        render_database_setup(exc)
        return

    if df.empty:
        render_database_setup(Exception("vw_dashboard_kelulusan kosong. Jalankan scripts/load_star_schema.py untuk memuat data."))
        return

    render_header(df)

    st.sidebar.header("Filter Dashboard")
    min_year = int(df["Tahun Wisuda"].min())
    max_year = int(df["Tahun Wisuda"].max())
    years = st.sidebar.slider("Tahun Wisuda", min_year, max_year, (min_year, max_year))
    periode_options = sorted(df["Periode Wisuda"].dropna().unique().tolist(), key=lambda x: PERIODE_ORDER.get(x, 99))
    selected_periode = st.sidebar.multiselect("Periode Wisuda", periode_options, default=periode_options)
    ipk_min = float(np.floor(df["IPK"].min() * 10) / 10)
    ipk_max = float(np.ceil(df["IPK"].max() * 10) / 10)
    ipk_range = st.sidebar.slider("Rentang IPK", ipk_min, ipk_max, (ipk_min, ipk_max), step=0.01)
    target_months = st.sidebar.slider("Target Tepat Waktu (bulan)", 42, 60, 48, step=1)
    st.sidebar.caption("Target default 48 bulan mengikuti desain star schema terbaru.")

    filtered = filter_dataframe(df, years, selected_periode, ipk_range)
    filtered_roles = filter_roles_dataframe(role_df, years, selected_periode, ipk_range)

    tabs = st.tabs(["Akademik", "Dosen", "NLP Tugas Akhir", "Star Schema", "Kualitas Data"])
    with tabs[0]:
        academic_dashboard(filtered, target_months)
    with tabs[1]:
        lecturer_dashboard_from_roles(filtered_roles)
    with tabs[2]:
        nlp_dashboard(filtered)
    with tabs[3]:
        star_schema_dashboard(schema)
    with tabs[4]:
        data_quality_dashboard(df, schema, get_database_url())


if __name__ == "__main__":
    main()
