from __future__ import annotations

import base64
import datetime
import html
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
import streamlit.components.v1 as components
from dotenv import load_dotenv

load_dotenv()


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

PERIODE_FILTER_OPTIONS = [
    ("WISUDA I", "I", "", "#1f644e"),
    ("WISUDA II", "II", "", "#1f644e"),
    ("WISUDA III", "III", "", "#1f644e"),
    ("WISUDA IV", "IV", "", "#1f644e"),
    ("WISUDA V", "V", "", "#1f644e"),
    ("WISUDA VI", "VI", "", "#1f644e"),
]

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
    page_title="Lensa DSI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_base64_image(image_path: Path) -> str:
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');

        :root {
            /* Palette */
            --surface: #fcfbfa; /* Warm ivory/cream background */
            --panel: #d6deda; /* Soft light sage green for sidebar */
            --ink: #1c2826; /* Deep slate / charcoal */
            --muted: #5e6b66; /* Elegant slate gray */
            --accent: #1f644e; /* Dark premium forest/emerald green */
            --accent-soft: #e2ede7; /* Soft light sage */
            --line: #b4c5be; /* Subtle warm border line matching sage */
            --warn: #b87333; /* Copper / amber */
            --danger: #c23b3f; /* Deep crimson */
            
            /* Sidebar-specific colors */
            --sidebar-text: #1c2826;
            --sidebar-muted: #5e6b66;
            --sidebar-min-width: 300px;
            --sidebar-max-width: 420px;
        }

        /* Typography & Globals */
        html, body, .stApp {
            font-family: 'Inter', sans-serif;
            background-color: var(--surface) !important;
            color: var(--ink) !important;
        }

        /* Hide default Streamlit sidebar headers, collapse buttons, etc. if needed */
        [data-testid="stSidebarCollapseButton"] *,
        [data-testid="stSidebarCollapseButton"] button *,
        [data-testid="stHeader"] [data-testid="stSidebarCollapseButton"] *,
        [data-testid="stHeader"] button[data-testid="stBaseButton-headerNoPadding"] *,
        [data-testid="collapsedControl"] button *,
        [data-testid="collapsedSidebar"] button *,
        button[title="Collapse sidebar"] *,
        button[title="Expand sidebar"] * {
            display: none !important;
        }

        /* Styling the collapse button (inside the open sidebar) */
        [data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebar"] button[data-testid="stBaseButton-headerNoPadding"],
        button[title="Collapse sidebar"] {
            background-color: #c9d5cf !important;
            border: 1px solid #b4c5be !important;
            border-radius: 6px !important;
            opacity: 1 !important;
            width: 32px !important;
            height: 32px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            position: relative !important;
            transition: all 0.2s ease !important;
            visibility: visible !important;
        }
        [data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"]:hover,
        [data-testid="stSidebar"] button[data-testid="stBaseButton-headerNoPadding"]:hover,
        button[title="Collapse sidebar"]:hover {
            background-color: #b8c8c0 !important;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05) !important;
        }

        /* Styling the expand button when collapsed */
        [data-testid="stHeader"] [data-testid="stSidebarCollapseButton"],
        [data-testid="stHeader"] button[data-testid="stBaseButton-headerNoPadding"],
        [data-testid="collapsedControl"],
        [data-testid="collapsedControl"] button,
        [data-testid="collapsedSidebar"] button,
        button[title="Expand sidebar"] {
            background-color: var(--accent) !important;
            border: none !important;
            border-radius: 0 8px 8px 0 !important;
            box-shadow: 2px 0 8px rgba(0, 0, 0, 0.15) !important;
            width: 24px !important;
            height: 64px !important;
            opacity: 1 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            margin: 0 !important;
            padding: 0 !important;
            position: fixed !important;
            left: 0 !important;
            top: 60px !important;
            z-index: 999999 !important;
            transition: all 0.2s ease !important;
            visibility: visible !important;
        }
        [data-testid="stHeader"] [data-testid="stSidebarCollapseButton"]:hover,
        [data-testid="stHeader"] button[data-testid="stBaseButton-headerNoPadding"]:hover,
        [data-testid="collapsedControl"]:hover,
        [data-testid="collapsedControl"] button:hover,
        [data-testid="collapsedSidebar"] button:hover,
        button[title="Expand sidebar"]:hover {
            background-color: #1b5e43 !important;
            width: 30px !important;
        }

        /* Draw custom SVG double-arrow-left on the collapse button */
        [data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"]::before,
        [data-testid="stSidebar"] button[data-testid="stBaseButton-headerNoPadding"]::before,
        button[title="Collapse sidebar"]::before {
            content: "" !important;
            position: absolute !important;
            top: 50% !important;
            left: 50% !important;
            transform: translate(-50%, -50%) !important;
            width: 16px !important;
            height: 16px !important;
            background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23114b36' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'><polyline points='11 17 6 12 11 7'/><polyline points='18 17 13 12 18 7'/></svg>") !important;
            background-size: contain !important;
            background-repeat: no-repeat !important;
            background-position: center !important;
            display: block !important;
        }

        /* Draw custom SVG chevron-right on the expand button */
        [data-testid="stHeader"] [data-testid="stSidebarCollapseButton"]::before,
        [data-testid="stHeader"] button[data-testid="stBaseButton-headerNoPadding"]::before,
        [data-testid="collapsedControl"]::before,
        [data-testid="collapsedControl"] button::before,
        [data-testid="collapsedSidebar"] button::before,
        button[title="Expand sidebar"]::before {
            content: "" !important;
            position: absolute !important;
            top: 50% !important;
            left: 50% !important;
            transform: translate(-50%, -50%) !important;
            width: 14px !important;
            height: 14px !important;
            background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23ffffff' stroke-width='3' stroke-linecap='round' stroke-linejoin='round'><polyline points='9 18 15 12 9 6'/></svg>") !important;
            background-size: contain !important;
            background-repeat: no-repeat !important;
            background-position: center !important;
            display: block !important;
        }

        .main {
            background-color: var(--surface) !important;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: var(--panel) !important;
            border-right: 1px solid #b4c5be !important;
            min-width: min(var(--sidebar-min-width), calc(100vw - 48px)) !important;
            max-width: min(var(--sidebar-max-width), calc(100vw - 48px)) !important;
            flex-shrink: 0 !important;
            overflow-x: hidden !important;
        }
        [data-testid="stSidebar"] > div:first-child,
        [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
            min-width: min(var(--sidebar-min-width), calc(100vw - 48px)) !important;
            max-width: min(var(--sidebar-max-width), calc(100vw - 48px)) !important;
            overflow-x: hidden !important;
        }
        [data-testid="stSidebar"] * {
            box-sizing: border-box !important;
        }
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"],
        [data-testid="stSidebar"] [data-testid="stHorizontalBlock"],
        [data-testid="stSidebar"] [data-testid="column"],
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] [data-testid="stButton"],
        [data-testid="stSidebar"] [data-testid="stExpander"],
        [data-testid="stSidebar"] details,
        [data-testid="stSidebar"] summary {
            min-width: 0 !important;
            max-width: 100% !important;
        }
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] summary {
            overflow-wrap: break-word !important;
        }

        /* Typography & Layout for Sidebar Custom Brand elements */
        .sidebar-brand-container {
            display: flex !important;
            align-items: center !important;
            gap: 12px !important;
            margin-bottom: 6px !important;
            margin-right: 48px !important; /* Prevent overlap with collapse control */
            font-family: 'Outfit', sans-serif !important;
            min-width: 0 !important;
            overflow: hidden !important;
        }
        .logo-box {
            background: linear-gradient(135deg, #114b36, #0b3626) !important;
            border-radius: 12px !important;
            padding: 8px !important;
            width: 44px !important;
            height: 44px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            box-shadow: 0 4px 10px rgba(17, 75, 54, 0.15) !important;
        }
        .logo-cap {
            width: 24px !important;
            height: 24px !important;
        }
        .brand-logo-img {
            height: 38px !important;
            width: auto !important;
            max-width: 48px !important;
            object-fit: contain !important;
            flex: 0 0 auto !important;
        }
        .brand-text {
            display: flex !important;
            flex-direction: column !important;
            min-width: 0 !important;
        }
        .brand-title {
            font-size: 1.35rem !important;
            font-weight: 800 !important;
            color: #1c2826 !important;
            letter-spacing: 0.5px !important;
            line-height: 1.1 !important;
            max-width: 100% !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            white-space: nowrap !important;
        }
        .brand-subtitle {
            font-size: 0.8rem !important;
            color: #5e6b66 !important;
            font-weight: 500 !important;
            max-width: 100% !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            white-space: nowrap !important;
        }
        .brand-org {
            font-size: 0.85rem !important;
            color: #5e6b66 !important;
            margin-left: 2px !important;
            margin-bottom: 16px !important;
            font-family: 'Inter', sans-serif !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            white-space: nowrap !important;
        }
        .brand-divider {
            margin-top: -8px !important;
            margin-bottom: 24px !important;
            border-bottom: 1px solid #b4c5be !important;
        }

        /* Style the Data Source container in the sidebar specifically */
        [data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #ffffff !important;
            border: 1px solid #b4c5be !important; /* Sage-colored border */
            border-radius: 16px !important;
            padding: 16px !important;
            box-shadow: 0 2px 8px rgba(0,0,0,0.02) !important;
            margin-bottom: 16px !important;
            min-width: 0 !important;
            max-width: 100% !important;
            overflow: hidden !important;
        }
        .card-header {
            display: flex !important;
            align-items: center !important;
            gap: 8px !important;
            margin-bottom: 12px !important;
            min-width: 0 !important;
        }
        .db-icon {
            stroke: #114b36 !important;
        }
        .card-title {
            font-family: 'Outfit', sans-serif !important;
            font-weight: 700 !important;
            font-size: 1.05rem !important;
            color: #1c2826 !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            white-space: nowrap !important;
        }
        .card-row {
            display: flex !important;
            justify-content: space-between !important;
            align-items: center !important;
            gap: 10px !important;
            margin-bottom: 8px !important;
            min-width: 0 !important;
        }
        .card-row-stack {
            display: flex !important;
            flex-direction: column !important;
            gap: 4px !important;
            margin-bottom: 8px !important;
        }
        .row-label {
            font-size: 0.88rem !important;
            color: #5e6b66 !important;
            font-weight: 500 !important;
            min-width: 0 !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            white-space: nowrap !important;
        }
        .status-badge {
            display: inline-flex !important;
            align-items: center !important;
            gap: 6px !important;
            background-color: #e8f5e9 !important;
            color: #0f975a !important;
            padding: 4px 12px !important;
            border-radius: 20px !important;
            font-size: 0.8rem !important;
            font-weight: 600 !important;
            flex: 0 1 auto !important;
            min-width: 0 !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            white-space: nowrap !important;
        }
        .status-badge .dot {
            width: 8px !important;
            height: 8px !important;
            background-color: #0f975a !important;
            border-radius: 50% !important;
            display: inline-block !important;
        }
        .update-time {
            font-size: 0.95rem !important;
            font-weight: 700 !important;
            color: #1c2826 !important;
            max-width: 100% !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            white-space: nowrap !important;
        }

        /* Style the Sync Data button inside the card container specifically */
        [data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] button {
            background-color: #114b36 !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 24px !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            padding: 0.6rem 1.2rem !important;
            width: 100% !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 8px !important;
            box-shadow: 0 4px 6px rgba(17, 75, 54, 0.1) !important;
            transition: all 0.2s ease !important;
            margin: 12px 0 0 0 !important; /* No negative margins! */
            min-width: 0 !important;
            max-width: 100% !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            white-space: nowrap !important;
        }
        [data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] button:hover {
            background-color: #0b3626 !important;
            box-shadow: 0 6px 12px rgba(17, 75, 54, 0.25) !important;
            transform: translateY(-1px) !important;
        }
        [data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] button:active {
            transform: translateY(1px) !important;
        }

        /* Custom Sidebar filter title */
        .filter-title {
            font-family: 'Outfit', sans-serif !important;
            font-weight: 700 !important;
            font-size: 1.1rem !important;
            color: #1c2826 !important;
            margin-top: 10px !important;
            margin-bottom: 12px !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            white-space: nowrap !important;
        }

        /* Custom Expander Dropdowns in Sidebar */
        [data-testid="stSidebar"] [data-testid="stExpander"] {
            border: 1px solid #b4c5be !important; /* Sage border */
            border-radius: 12px !important;
            background-color: #ffffff !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.01) !important;
            margin-bottom: 10px !important;
            overflow: hidden !important;
            max-width: 100% !important;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] details {
            border: none !important;
            background-color: transparent !important;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] summary {
            font-family: 'Inter', sans-serif !important;
            font-weight: 600 !important;
            color: #1c2826 !important;
            background-color: transparent !important;
            min-width: 0 !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            white-space: nowrap !important;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
            color: var(--accent) !important;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
            background-color: transparent !important;
            border-top: 1px solid #eef1ed !important;
        }
        .expander-subtitle {
            font-size: 0.8rem !important;
            color: #5e6b66 !important;
            font-weight: 600 !important;
            margin-bottom: 6px !important;
        }
        .expander-note {
            font-size: 0.78rem !important;
            color: #5e6b66 !important;
            margin-top: 8px !important;
            max-width: 100% !important;
            overflow-wrap: anywhere !important;
        }

        /* Text and Inputs in Sidebar */
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] label {
            color: var(--sidebar-text) !important;
            font-family: 'Inter', sans-serif !important;
        }
        [data-testid="stSidebar"] .stSlider span {
            color: var(--sidebar-muted) !important;
        }

        /* Multiselect Tags in Sidebar */
        div[data-baseweb="tag"], 
        span[data-baseweb="tag"],
        .stMultiSelect div[data-baseweb="tag"],
        [data-testid="stSidebar"] .stMultiSelect span,
        [data-testid="stSidebar"] div[data-baseweb="tag"] {
            background-color: #f1f5f9 !important;
            color: #1c2826 !important;
            border: 1px solid #b4c5be !important;
            border-radius: 6px !important;
            min-width: 0 !important;
            max-width: 100% !important;
        }
        div[data-baseweb="tag"] span,
        [data-testid="stSidebar"] div[data-baseweb="tag"] span {
            color: #1c2826 !important;
            font-weight: 500 !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            white-space: nowrap !important;
        }
        
        /* Sliders in Sidebar */
        [data-testid="stSidebar"] div[data-testid="stSlider"] > div > div {
            background-color: #b4c5be !important;
        }
        [data-testid="stSidebar"] div[data-testid="stSlider"] div[data-testid="stThumbValue"] + div > div {
            background-color: var(--accent) !important;
        }
        [data-testid="stSidebar"] div[data-testid="stSlider"] [role="slider"] {
            background-color: var(--accent) !important;
            border: 2px solid #ffffff !important;
        }

        /* 1. Semua Periode button styling */
        div[data-testid="stSidebar"] button[key="btn_semua_periode"] {
            background-color: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 20px !important;
            color: #64748b !important;
            font-weight: 600 !important;
            text-align: left !important;
            padding: 8px 16px !important;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
            margin-bottom: 12px !important;
            min-width: 0 !important;
            max-width: 100% !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            white-space: nowrap !important;
        }

        /* 2. Circle timeline container and labels */
        .circle-label {
            text-align: center !important;
            font-size: 0.75rem !important;
            font-weight: 600 !important;
            color: #64748b !important;
            margin-top: 4px !important;
            margin-bottom: 12px !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            white-space: nowrap !important;
        }

        [data-testid="stSidebar"] details div[data-testid="stHorizontalBlock"] {
            gap: 4px !important;
        }

        [data-testid="stSidebar"] details div[data-testid="stHorizontalBlock"] div[data-testid="stColumn"],
        [data-testid="stSidebar"] details div[data-testid="stHorizontalBlock"] div[data-testid="column"] {
            min-width: 0 !important;
        }

        /* Checkbox square button standard dimensions */
        div[data-testid="stSidebar"] details div[data-testid="stHorizontalBlock"] div[data-testid="column"] button,
        div[data-testid="stSidebar"] details div[data-testid="stHorizontalBlock"] div[data-testid="stColumn"] button,
        [data-testid="stSidebar"] details div[data-testid="stHorizontalBlock"] div[data-testid="column"] button,
        [data-testid="stSidebar"] details div[data-testid="stHorizontalBlock"] div[data-testid="stColumn"] button {
            border-radius: 8px !important;
            width: 100% !important;
            height: auto !important;
            min-width: 0 !important;
            max-width: 44px !important;
            min-height: 36px !important;
            max-height: 44px !important;
            aspect-ratio: 1 / 1 !important;
            padding: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-family: 'Outfit', sans-serif !important;
            font-weight: 700 !important;
            font-size: 1rem !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
            transition: all 0.2s ease !important;
            background-color: #ffffff !important;
            color: #475569 !important;
            border: 1px solid #cbd5e1 !important;
        }

        /* Prevent Period Wisuda Roman numerals text from wrapping */
        div[data-testid="stSidebar"] details div[data-testid="stHorizontalBlock"] div[data-testid="column"] button p,
        div[data-testid="stSidebar"] details div[data-testid="stHorizontalBlock"] div[data-testid="stColumn"] button p,
        [data-testid="stSidebar"] details div[data-testid="stHorizontalBlock"] div[data-testid="column"] button p,
        [data-testid="stSidebar"] details div[data-testid="stHorizontalBlock"] div[data-testid="stColumn"] button p {
            white-space: nowrap !important;
            margin: 0 !important;
            padding: 0 !important;
            width: auto !important;
            max-width: none !important;
            overflow: visible !important;
            text-overflow: clip !important;
        }

        [data-testid="stSidebar"] button,
        [data-testid="stSidebar"] button p,
        [data-testid="stSidebar"] button span {
            min-width: 0 !important;
            max-width: 100% !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
            white-space: nowrap !important;
        }

        [data-testid="stSidebar"][aria-expanded="false"],
        [data-testid="stSidebar"][aria-hidden="true"] {
            min-width: 0 !important;
            max-width: 0 !important;
            border-right: 0 !important;
            overflow: hidden !important;
        }

        @media (max-width: 640px) {
            :root {
                --sidebar-min-width: 248px;
                --sidebar-max-width: 88vw;
            }
        }

        /* Style all main primary action buttons green */
        .main [data-testid="stButton"] button,
        .main div[data-testid="stFormSubmitButton"] button {
            background-color: var(--accent) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            padding: 8px 16px !important;
            box-shadow: 0 4px 6px rgba(17, 75, 54, 0.15) !important;
            transition: all 0.2s ease !important;
            width: 100% !important;
        }

        .main [data-testid="stButton"] button:hover,
        .main div[data-testid="stFormSubmitButton"] button:hover {
            background-color: #0b3626 !important;
            box-shadow: 0 6px 12px rgba(17, 75, 54, 0.25) !important;
        }

        /* Override: Hapus Semua Preview button (Column 1 of horizontal action block) */
        .main div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-of-type(1) div[data-testid="stButton"] button {
            background-color: #fee2e2 !important;
            color: #dc2626 !important;
            border: 1px solid #fca5a5 !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            padding: 8px 16px !important;
            box-shadow: none !important;
            transition: all 0.2s ease !important;
        }
        .main div[data-testid="stHorizontalBlock"] div[data-testid="column"]:nth-of-type(1) div[data-testid="stButton"] button:hover {
            background-color: #fca5a5 !important;
            color: #b91c1c !important;
            box-shadow: none !important;
        }

        /* Main Page Layout Details */
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1420px;
        }

        [data-testid="stHeader"] {
            background-color: rgba(252, 251, 250, 0.8) !important;
            backdrop-filter: blur(8px) !important;
        }

        /* Titles and Headings */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Outfit', sans-serif !important;
            color: var(--ink) !important;
        }

        h1 {
            font-size: 2.3rem !important;
            font-weight: 800 !important;
            background: linear-gradient(135deg, #114b36, #0b3626);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.4rem !important;
        }

        h2 {
            font-size: 1.45rem !important;
            font-weight: 700 !important;
            margin-top: 1.5rem !important;
            border-bottom: 2px solid #eef1ed;
            padding-bottom: 0.3rem !important;
        }

        /* Premium card borders for st.container(border=True) */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #ffffff !important;
            border: 1px solid #eef1ed !important;
            border-radius: 16px !important;
            padding: 20px !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02) !important;
            transition: all 0.2s ease-in-out !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:hover {
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.04) !important;
            border-color: #e2ede7 !important;
        }
         /* Tabs Selection Bar */
        div[data-testid="stTabs"] {
            border-bottom: 2px solid #eef1ed !important;
            margin-bottom: 1.5rem !important;
        }
        div[data-testid="stTabs"] button {
            background-color: transparent !important;
            border: none !important;
            padding: 0.75rem 1.5rem !important;
            margin-right: 0.5rem !important;
            transition: all 0.2s ease !important;
        }
        div[data-testid="stTabs"] button p {
            font-size: 1rem !important;
            font-weight: 600 !important;
            color: var(--muted) !important;
            transition: color 0.2s ease !important;
        }
        div[data-testid="stTabs"] button:hover p {
            color: var(--accent) !important;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            border-bottom: 3px solid var(--accent) !important;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] p {
            color: var(--accent) !important;
            font-weight: 700 !important;
        }
        
        /* Overrides to prevent default Streamlit red/orange highlights on hover/active/focus */
        div[data-testid="stTabs"] button[data-baseweb="tab"] {
            border-bottom: 3px solid transparent !important;
        }
        div[data-testid="stTabs"] button[data-baseweb="tab"]:hover {
            border-bottom: 3px solid var(--accent-soft) !important;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            border-bottom: 3px solid var(--accent) !important;
        }
        div[data-testid="stTabs"] button[data-baseweb="tab"]:focus,
        div[data-testid="stTabs"] button[data-baseweb="tab"]:active,
        div[data-testid="stTabs"] button[data-baseweb="tab"] * {
            outline: none !important;
            box-shadow: none !important;
        }
        div[data-baseweb="tab-highlight"] {
            background-color: var(--accent) !important;
        }

        /* Custom HTML KPI Cards styling */
        .kpi-row {
            display: flex !important;
            gap: 16px !important;
            width: 100% !important;
            margin-bottom: 24px !important;
        }
        
        .kpi-card-custom {
            flex: 1 !important;
            background-color: #ffffff !important;
            border: 1px solid #eef1ed !important;
            border-radius: 16px !important;
            padding: 16px 20px !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02), 0 2px 4px -1px rgba(0, 0, 0, 0.01) !important;
            transition: all 0.2s ease-in-out !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: space-between !important;
            min-height: 125px !important;
            position: relative !important;
        }
        
        .kpi-card-custom:hover {
            transform: translateY(-3px) !important;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.04), 0 4px 6px -2px rgba(0, 0, 0, 0.01) !important;
            border-color: #e2ede7 !important;
        }
        
        .kpi-card-header {
            display: flex !important;
            justify-content: space-between !important;
            align-items: flex-start !important;
            width: 100% !important;
        }
        
        .kpi-card-title {
            color: #64748b !important;
            font-size: 0.88rem !important;
            font-weight: 600 !important;
            font-family: 'Inter', sans-serif !important;
        }
        
        .kpi-card-icon {
            width: 36px !important;
            height: 36px !important;
            border-radius: 10px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        
        /* Icon backgrounds and stroke colors */
        .icon-alumni { background-color: #e8f5e9 !important; color: #114b36 !important; }
        .icon-ipk { background-color: #e6f6f4 !important; color: #0d9488 !important; }
        .icon-studi { background-color: #fffbeb !important; color: #d97706 !important; }
        .icon-tepat { background-color: #fef2f2 !important; color: #dc2626 !important; }
        .icon-tahun { background-color: #f0f9ff !important; color: #0284c7 !important; }
        
        .kpi-card-value {
            color: #1e293b !important;
            font-size: 1.9rem !important;
            font-weight: 800 !important;
            font-family: 'Outfit', sans-serif !important;
            margin-top: 8px !important;
            margin-bottom: 4px !important;
            line-height: 1.1 !important;
        }
        
        .kpi-card-subtext {
            color: #94a3b8 !important;
            font-size: 0.8rem !important;
            font-weight: 500 !important;
            font-family: 'Inter', sans-serif !important;
        }
        
        .trend-up {
            color: #0f975a !important;
            font-weight: 600 !important;
        }
        .trend-down {
            color: #dc2626 !important;
            font-weight: 600 !important;
        }
        .trend-neutral {
            color: #94a3b8 !important;
            font-weight: 600;
        }

        .section-note {
            color: var(--muted) !important;
            font-size: 0.95rem !important;
            line-height: 1.6 !important;
            margin-top: -0.5rem !important;
            margin-bottom: 1.25rem !important;
        }

        .insight-panel-title {
            color: var(--ink) !important;
            font-family: 'Outfit', sans-serif !important;
            font-size: 1.05rem !important;
            font-weight: 800 !important;
            margin: 0 0 0.85rem 0 !important;
        }
        .insight-list {
            display: grid !important;
            gap: 0.72rem !important;
            margin: 0 !important;
            padding: 0 !important;
            list-style: none !important;
        }
        .insight-item {
            color: var(--ink) !important;
            font-size: 0.9rem !important;
            line-height: 1.48 !important;
            padding-left: 1rem !important;
            position: relative !important;
        }
        .insight-item::before {
            content: "" !important;
            position: absolute !important;
            left: 0 !important;
            top: 0.58rem !important;
            width: 6px !important;
            height: 6px !important;
            border-radius: 999px !important;
            background-color: var(--accent) !important;
        }

        .audit-summary-grid {
            display: grid !important;
            grid-template-columns: repeat(4, minmax(0, 1fr)) !important;
            gap: 10px !important;
            margin: 6px 0 14px 0 !important;
        }
        .audit-summary-card {
            background-color: #ffffff !important;
            border: 1px solid #eef1ed !important;
            border-radius: 10px !important;
            padding: 12px 14px !important;
        }
        .audit-summary-label {
            color: var(--muted) !important;
            font-size: 0.78rem !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.02em !important;
            margin-bottom: 6px !important;
        }
        .audit-summary-value {
            color: var(--ink) !important;
            font-family: 'Outfit', sans-serif !important;
            font-size: 1.35rem !important;
            font-weight: 800 !important;
            line-height: 1 !important;
        }
        .audit-toolbar-note {
            color: var(--muted) !important;
            font-size: 0.86rem !important;
            margin: -4px 0 10px 0 !important;
        }

        @media (max-width: 900px) {
            .audit-summary-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
            }
        }

        .schema-box {
            background-color: #ffffff !important;
            border: 1px solid #eef1ed !important;
            border-radius: 12px !important;
            padding: 1.25rem !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02) !important;
            min-height: 180px !important;
            transition: all 0.2s ease !important;
        }
        .schema-box:hover {
            box-shadow: 0 8px 12px -3px rgba(0, 0, 0, 0.03) !important;
            border-color: var(--accent-soft) !important;
        }

        .schema-title {
            color: var(--accent) !important;
            font-family: 'Outfit', sans-serif !important;
            font-weight: 700 !important;
            font-size: 1.12rem !important;
            margin-bottom: 0.5rem !important;
        }

        .schema-field {
            color: var(--ink) !important;
            font-size: 0.88rem !important;
            line-height: 1.55 !important;
        }

        .status-chip {
            display: inline-flex !important;
            align-items: center !important;
            border-radius: 6px !important;
            padding: 0.25rem 0.75rem !important;
            font-size: 0.82rem !important;
            font-weight: 600 !important;
            background-color: var(--accent-soft) !important;
            color: var(--accent) !important;
            border: 1px solid #c8d9cc !important;
            margin-bottom: 1rem !important;
        }

        .main div[data-testid="stFormSubmitButton"] button {
            background-color: var(--accent) !important;
            color: #ffffff !important;
            border: 1px solid var(--accent) !important;
            border-radius: 8px !important;
            width: 100% !important;
            min-height: 44px !important;
            font-weight: 700 !important;
            font-size: 1rem !important;
            box-shadow: 0 6px 14px rgba(17, 75, 54, 0.18) !important;
            cursor: pointer !important;
            transition: background-color 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease, transform 0.18s ease !important;
        }
        .main div[data-testid="stFormSubmitButton"] button:hover {
            background-color: #0b3626 !important;
            border-color: #0b3626 !important;
            box-shadow: 0 8px 18px rgba(17, 75, 54, 0.26) !important;
            transform: translateY(-1px) !important;
        }
        .main div[data-testid="stFormSubmitButton"] button:active {
            transform: translateY(0) !important;
            box-shadow: 0 4px 10px rgba(17, 75, 54, 0.16) !important;
        }
        .main div[data-testid="stFormSubmitButton"] button:disabled,
        .main div[data-testid="stFormSubmitButton"] button[disabled] {
            background-color: #c8d9cc !important;
            color: #5e6b66 !important;
            border-color: #c8d9cc !important;
            box-shadow: none !important;
            cursor: not-allowed !important;
            transform: none !important;
            opacity: 0.75 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_sidebar_resize_guard() -> None:
    components.html(
        """
        <script>
        (() => {
            const MIN_WIDTH = 300;
            const MAX_WIDTH = 420;

            const getSidebar = () => parent.document.querySelector('[data-testid="stSidebar"]');

            const clampSidebar = () => {
                const sidebar = getSidebar();
                if (!sidebar || sidebar.getAttribute('aria-expanded') === 'false') {
                    return;
                }

                const currentWidth = parseFloat(sidebar.style.width || sidebar.getBoundingClientRect().width);
                if (!Number.isFinite(currentWidth)) {
                    return;
                }

                const clampedWidth = Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, currentWidth));
                if (Math.abs(currentWidth - clampedWidth) > 0.5) {
                    sidebar.style.width = `${clampedWidth}px`;
                }
            };

            const install = () => {
                const sidebar = getSidebar();
                if (!sidebar) {
                    parent.requestAnimationFrame(install);
                    return;
                }

                clampSidebar();
                new parent.ResizeObserver(clampSidebar).observe(sidebar);
                parent.window.addEventListener('resize', clampSidebar, { passive: true });
                parent.document.addEventListener('mouseup', clampSidebar, true);
                parent.document.addEventListener('touchend', clampSidebar, true);
            };

            install();
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def inject_manual_form_input_guard() -> None:
    components.html(
        """
        <script>
        (() => {
            const FIELD_RULES = [
                {
                    labels: ["NIM *"],
                    inputMode: "numeric",
                    maxLength: 10,
                    filter: (value) => value.replace(/\\D/g, "").slice(0, 10),
                },
                {
                    labels: ["IPK Lulus *"],
                    inputMode: "decimal",
                    maxLength: 4,
                    filter: (value, previous) => {
                        let next = value.replace(/[^0-9.,]/g, "");
                        const firstSeparator = next.search(/[.,]/);
                        if (firstSeparator !== -1) {
                            const head = next.slice(0, firstSeparator + 1);
                            const tail = next.slice(firstSeparator + 1).replace(/[.,]/g, "");
                            next = head + tail;
                        }
                        next = next.replace(/^([0-9])([0-9])/, "$1");
                        next = next.replace(/^([0-9][.,][0-9]{0,2}).*$/, "$1");
                        if (/^[5-9]/.test(next)) {
                            return previous || "";
                        }
                        const numeric = Number(next.replace(",", "."));
                        if (Number.isFinite(numeric) && numeric > 4) {
                            return previous || "4";
                        }
                        return next;
                    },
                },
                {
                    labels: ["Lama Studi (Bulan) *"],
                    inputMode: "numeric",
                    maxLength: 3,
                    filter: (value) => value.replace(/\\D/g, "").slice(0, 3),
                },
                {
                    labels: ["Tahun Wisuda *"],
                    inputMode: "numeric",
                    maxLength: 4,
                    filter: (value) => value.replace(/\\D/g, "").slice(0, 4),
                },
                {
                    labels: ["Nama Lengkap *"],
                    inputMode: "text",
                    maxLength: 120,
                    filter: (value) => value.replace(/[^\\p{L}\\s.',-]/gu, "").replace(/\\s{2,}/g, " ").slice(0, 120),
                },
            ];

            const setNativeValue = (input, value) => {
                if (input.value === value) {
                    return;
                }
                const prototype = input instanceof parent.HTMLTextAreaElement
                    ? parent.HTMLTextAreaElement.prototype
                    : parent.HTMLInputElement.prototype;
                const setter = Object.getOwnPropertyDescriptor(prototype, "value")?.set;
                setter.call(input, value);
                input.dispatchEvent(new parent.Event("input", { bubbles: true }));
            };

            const labelMatches = (input, labels) => {
                const aria = input.getAttribute("aria-label") || "";
                return labels.some((label) => aria === label || aria.endsWith(` ${label}`));
            };

            const isVisible = (element) => {
                if (!element || element.disabled || element.getAttribute("aria-disabled") === "true") {
                    return false;
                }
                const rect = element.getBoundingClientRect();
                const style = parent.getComputedStyle(element);
                return rect.width > 0 && rect.height > 0 && style.visibility !== "hidden" && style.display !== "none";
            };

            const isManualField = (element) => {
                const aria = element.getAttribute("aria-label") || "";
                const manualFormVisible = parent.document.body.innerText.includes("Formulir Input Manual Alumni");
                if (manualFormVisible && aria === "Select a date.") {
                    return true;
                }
                const manualLabels = [
                    "NIM *",
                    "Nama Lengkap *",
                    "Judul Tugas Akhir *",
                    "Tanggal Lulus *",
                    "IPK Lulus *",
                    "Lama Studi (Bulan) *",
                    "Periode Wisuda *",
                    "Tahun Wisuda *",
                    "Dosen Pembimbing 1",
                    "Dosen Pembimbing 2",
                    "Dosen Penguji 1",
                    "Dosen Penguji 2",
                    "Dosen Penguji 3",
                ];
                return manualLabels.some((label) => aria === label || aria.includes(label));
            };

            const getManualFields = () => {
                const fields = [...parent.document.querySelectorAll("input, textarea, [role='combobox']")]
                    .filter((element) => isVisible(element) && isManualField(element));
                return fields.filter((field, index, list) => list.indexOf(field) === index);
            };

            const focusNextManualField = (current) => {
                const fields = getManualFields();
                const currentIndex = fields.indexOf(current);
                const nextField = fields.slice(currentIndex + 1).find(isVisible);
                if (nextField) {
                    nextField.focus();
                    if (typeof nextField.select === "function" && nextField.tagName !== "TEXTAREA") {
                        nextField.select();
                    }
                }
            };

            const bindEnterNavigation = (input) => {
                if (input.dataset.manualEnterNav === "1") {
                    return;
                }
                input.dataset.manualEnterNav = "1";
                input.addEventListener("keydown", (event) => {
                    if (event.key !== "Enter" || event.shiftKey || event.ctrlKey || event.altKey || event.metaKey) {
                        return;
                    }
                    if (input.tagName === "TEXTAREA") {
                        return;
                    }
                    event.preventDefault();
                    event.stopPropagation();
                    focusNextManualField(input);
                }, true);
            };

            const bindInput = (input, rule) => {
                if (input.dataset.manualGuard === "1") {
                    return;
                }
                input.dataset.manualGuard = "1";
                input.setAttribute("inputmode", rule.inputMode);
                input.setAttribute("autocomplete", "off");
                if (rule.maxLength) {
                    input.setAttribute("maxlength", String(rule.maxLength));
                }

                const applyFilter = () => {
                    const previous = input.dataset.lastValidValue || "";
                    const filtered = rule.filter(input.value, previous);
                    setNativeValue(input, filtered);
                    input.dataset.lastValidValue = filtered;
                };

                input.addEventListener("beforeinput", (event) => {
                    if (!event.data) {
                        return;
                    }
                    const start = input.selectionStart ?? input.value.length;
                    const end = input.selectionEnd ?? start;
                    const proposed = input.value.slice(0, start) + event.data + input.value.slice(end);
                    const previous = input.dataset.lastValidValue || input.value;
                    if (rule.filter(proposed, previous) !== proposed) {
                        event.preventDefault();
                    }
                });
                input.addEventListener("input", applyFilter);
                input.addEventListener("paste", () => parent.requestAnimationFrame(applyFilter));
                applyFilter();
            };

            const install = () => {
                const inputs = [...parent.document.querySelectorAll("input, textarea, [role='combobox']")];
                for (const input of inputs) {
                    if (isManualField(input)) {
                        bindEnterNavigation(input);
                    }
                    const rule = FIELD_RULES.find((candidate) => labelMatches(input, candidate.labels));
                    if (rule) {
                        bindInput(input, rule);
                    }
                }
            };

            install();
            new parent.MutationObserver(install).observe(parent.document.body, { childList: true, subtree: true });
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def style_plotly_fig(fig) -> None:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="'Inter', sans-serif", color="#1e293b"),
        title=dict(font=dict(family="'Outfit', sans-serif", size=15, color="#1e293b")),
        margin=dict(t=55, b=30, l=10, r=10),
    )
    fig.update_xaxes(
        showgrid=True, 
        gridcolor="#e2e8f0", 
        gridwidth=1,
        griddash="dot",
        showline=False,
        zeroline=False,
        title_font=dict(family="'Inter', sans-serif", size=11, color="#64748b"),
        tickfont=dict(family="'Inter', sans-serif", size=9, color="#64748b"),
    )
    fig.update_yaxes(
        showgrid=True, 
        gridcolor="#e2e8f0", 
        gridwidth=1,
        griddash="dot",
        showline=False,
        zeroline=False,
        title_font=dict(family="'Inter', sans-serif", size=11, color="#64748b"),
        tickfont=dict(family="'Inter', sans-serif", size=9, color="#64748b"),
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


def display_person_name(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    return " ".join(part.capitalize() for part in text.split())


def get_lecturer_options(dim_dosen: pd.DataFrame) -> list[str]:
    if dim_dosen.empty:
        return []
    source_col = "nama_asal" if "nama_asal" in dim_dosen.columns else "nama_dosen_normalized"
    options = (
        dim_dosen[source_col]
        .dropna()
        .astype(str)
        .str.strip()
        .replace("", np.nan)
        .dropna()
        .drop_duplicates()
        .sort_values(key=lambda s: s.str.lower())
        .tolist()
    )
    display_options: list[str] = []
    seen: set[str] = set()
    for option in options:
        display_value = display_person_name(option)
        lookup = display_value.lower()
        if display_value and lookup not in seen:
            display_options.append(display_value)
            seen.add(lookup)
    return display_options


def parse_ipk_input(value: str) -> tuple[float | None, str | None]:
    normalized = str(value).strip().replace(",", ".")
    if not re.fullmatch(r"\d(?:[.,]\d{1,2})?", str(value).strip()):
        return None, "IPK hanya boleh angka desimal dengan titik atau koma, contoh 3,50 atau 3.50."
    ipk_value = float(normalized)
    if not 0 <= ipk_value <= 4:
        return None, "IPK harus berada pada rentang 0 sampai 4."
    return round(ipk_value, 2), None


def validate_manual_record(nim: str, nama: str, judul: str, ipk_text: str) -> tuple[float | None, list[str]]:
    errors: list[str] = []
    nim_text = str(nim).strip()
    if not re.fullmatch(r"\d{10}", nim_text):
        errors.append("NIM harus tepat 10 angka.")

    nama_clean = " ".join(nama.split())
    if not nama_clean:
        errors.append("Nama Lengkap wajib diisi.")
    elif not re.fullmatch(r"[A-Za-z .,'-]+", nama_clean):
        errors.append("Nama Lengkap hanya boleh berisi huruf dan tanda baca nama umum.")
    elif re.search(r"\d", nama_clean):
        errors.append("Nama Lengkap tidak boleh mengandung angka.")

    if not judul.strip():
        errors.append("Judul Tugas Akhir wajib diisi.")

    ipk_value, ipk_error = parse_ipk_input(ipk_text)
    if ipk_error:
        errors.append(ipk_error)

    return ipk_value, errors


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


def kategori_lama_studi(months: int, target_months: int = 54) -> str:
    if months <= target_months:
        return "Tepat Waktu"
    return "Terlambat"


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
    if "urutan_peran" in role_df.columns:
        role_df["urutan_peran"] = pd.to_numeric(role_df["urutan_peran"], errors="coerce")
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
    st.title("Lensa DSI")
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


def safe_text(value: object) -> str:
    return html.escape(str(value), quote=True)


def format_change(current: float, previous: float, unit: str = "") -> str:
    if pd.isna(current) or pd.isna(previous):
        return "belum dapat dibandingkan"
    delta = current - previous
    if abs(delta) < 0.005:
        return f"stabil di {current:.2f}{unit}"
    direction = "naik" if delta > 0 else "turun"
    return f"{direction} {abs(delta):.2f}{unit} menjadi {current:.2f}{unit}"


def render_dynamic_panels(insights: list[str], recommendations: list[str]) -> None:
    def render_items(items: list[str]) -> str:
        return "".join(f'<li class="insight-item">{safe_text(item)}</li>' for item in items[:5])

    col_insight, col_reco = st.columns(2)
    with col_insight:
        with st.container(border=True):
            st.markdown(
                f"""
                <div class="insight-panel-title">Insight Otomatis</div>
                <ul class="insight-list">{render_items(insights)}</ul>
                """,
                unsafe_allow_html=True,
            )
    with col_reco:
        with st.container(border=True):
            st.markdown(
                f"""
                <div class="insight-panel-title">Rekomendasi Tindakan</div>
                <ul class="insight-list">{render_items(recommendations)}</ul>
                """,
                unsafe_allow_html=True,
            )


def build_academic_insights(df: pd.DataFrame, target_months: int, target_ipk: float) -> tuple[list[str], list[str]]:
    if df.empty:
        return (
            ["Tidak ada data alumni pada kombinasi filter saat ini."],
            ["Periksa kembali kombinasi filter agar analisis akademik dapat ditampilkan."],
        )

    total = len(df)
    avg_ipk = df["IPK"].mean()
    avg_study = df["Lama Studi"].mean()
    on_time_rate = df["Lama Studi"].le(target_months).mean() * 100
    insights = [
        f"Filter aktif menampilkan {total} alumni dengan rata-rata IPK {avg_ipk:.2f} dan rata-rata lama studi {avg_study:.1f} bulan.",
        f"Sebanyak {on_time_rate:.1f}% alumni berada pada atau di bawah target {target_months} bulan.",
    ]

    yearly = df.groupby("Tahun Wisuda", as_index=False).agg(jumlah=("NO", "count"), rata_ipk=("IPK", "mean"), rata_studi=("Lama Studi", "mean")).sort_values("Tahun Wisuda")
    if len(yearly) >= 2:
        latest = yearly.iloc[-1]
        previous = yearly.iloc[-2]
        count_delta = int(latest["jumlah"] - previous["jumlah"])
        count_direction = "naik" if count_delta > 0 else "turun" if count_delta < 0 else "stabil"
        insights.append(
            f"Jumlah lulusan tahun {int(latest['Tahun Wisuda'])} {count_direction} {abs(count_delta)} orang dibanding tahun {int(previous['Tahun Wisuda'])}."
        )
        insights.append(f"Rata-rata IPK tahun terbaru {format_change(float(latest['rata_ipk']), float(previous['rata_ipk']))} dibanding tahun sebelumnya.")

    period_perf = (
        df.groupby("Periode Wisuda", as_index=False)
        .agg(jumlah=("NO", "count"), rata_ipk=("IPK", "mean"), rata_studi=("Lama Studi", "mean"))
        .sort_values(["rata_ipk", "jumlah"], ascending=[False, False])
    )
    if not period_perf.empty:
        best_period = period_perf.iloc[0]
        insights.append(
            f"{best_period['Periode Wisuda']} menjadi periode dengan performa IPK terbaik, rata-rata {best_period['rata_ipk']:.2f} dari {int(best_period['jumlah'])} alumni."
        )

    recommendations = []
    if avg_study > target_months:
        recommendations.append(f"Prioritaskan evaluasi proses akademik karena rata-rata lama studi masih melewati target {target_months} bulan.")
    else:
        recommendations.append(f"Pertahankan pola pembinaan akademik karena rata-rata lama studi sudah berada di bawah target {target_months} bulan.")
    if avg_ipk < target_ipk:
        recommendations.append(f"Lakukan monitoring kelompok IPK rendah karena rata-rata IPK masih di bawah target {target_ipk:.2f}.")
    else:
        recommendations.append(f"Gunakan kelompok dengan IPK di atas target {target_ipk:.2f} sebagai referensi praktik akademik yang baik.")
    late_periods = (
        df.groupby("Periode Wisuda", as_index=False)
        .agg(rata_studi=("Lama Studi", "mean"), jumlah=("NO", "count"))
        .sort_values("rata_studi", ascending=False)
    )
    if not late_periods.empty:
        period = late_periods.iloc[0]
        recommendations.append(f"Tinjau alumni pada {period['Periode Wisuda']} karena memiliki rata-rata lama studi tertinggi, yaitu {period['rata_studi']:.1f} bulan.")
    recommendations.append("Gunakan hasil scatter IPK dan lama studi untuk mengidentifikasi alumni ber-IPK baik tetapi masa studi panjang sebagai bahan telaah proses bimbingan.")

    return insights[:5], recommendations[:5]


def build_lecturer_insights(role_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    if role_df.empty:
        return (
            ["Tidak ada relasi dosen pada kombinasi filter saat ini."],
            ["Periksa kembali kombinasi filter agar analisis beban dosen dapat ditampilkan."],
        )

    total_roles = int(role_df["role_count"].sum())
    total_alumni = role_df["kelulusan_key"].nunique()
    unique_lecturers = role_df["dosen_key"].nunique()
    advisor_roles = int(role_df.loc[role_df["jenis_peran"].eq("Pembimbing"), "role_count"].sum())
    examiner_roles = int(role_df.loc[role_df["jenis_peran"].eq("Penguji"), "role_count"].sum())
    top_all = (
        role_df.groupby(["dosen_key", "nama_dosen_normalized"], as_index=False)
        .agg(jumlah_peran=("role_count", "sum"), jumlah_alumni=("kelulusan_key", "nunique"))
        .sort_values("jumlah_peran", ascending=False)
    )
    top_lecturer = top_all.iloc[0]
    concentration = (top_all.head(5)["jumlah_peran"].sum() / total_roles * 100) if total_roles else 0
    role_balance = "pembimbing" if advisor_roles >= examiner_roles else "penguji"
    advisor_positions = (
        role_df[role_df["jenis_peran"].eq("Pembimbing")]
        .groupby("urutan_peran", as_index=False)
        .agg(jumlah=("role_count", "sum"))
        .sort_values("urutan_peran")
    )
    advisor_1 = int(advisor_positions.loc[advisor_positions["urutan_peran"].eq(1), "jumlah"].sum())
    advisor_2 = int(advisor_positions.loc[advisor_positions["urutan_peran"].eq(2), "jumlah"].sum())

    insights = [
        f"Terdapat {total_roles} relasi peran dari {unique_lecturers} dosen untuk {total_alumni} alumni pada filter aktif.",
        f"{top_lecturer['nama_dosen_normalized']} memiliki relasi terbanyak dengan {int(top_lecturer['jumlah_peran'])} peran pada {int(top_lecturer['jumlah_alumni'])} alumni.",
        f"Lima dosen teratas menangani {concentration:.1f}% dari total relasi peran.",
        f"Komposisi peran lebih dominan sebagai {role_balance}: {advisor_roles} pembimbing dan {examiner_roles} penguji.",
        f"Peran pembimbing terdiri dari {advisor_1} relasi Pembimbing 1 dan {advisor_2} relasi Pembimbing 2 pada filter aktif.",
    ]

    recommendations = [
        f"Evaluasi distribusi beban {top_lecturer['nama_dosen_normalized']} agar kualitas bimbingan dan pengujian tetap terjaga.",
        "Gunakan konsentrasi lima dosen teratas sebagai dasar pemerataan penugasan pada periode berikutnya.",
    ]
    if advisor_roles > examiner_roles * 1.25:
        recommendations.append("Seimbangkan kembali peran penguji agar evaluasi sidang tidak terlalu terkonsentrasi pada kelompok tertentu.")
    elif examiner_roles > advisor_roles * 1.25:
        recommendations.append("Seimbangkan kembali peran pembimbing agar pendampingan tugas akhir tidak tertinggal dari aktivitas pengujian.")
    else:
        recommendations.append("Pertahankan keseimbangan peran pembimbing dan penguji karena komposisinya relatif proporsional.")
    if advisor_1 > advisor_2 * 1.5 and advisor_2 > 0:
        recommendations.append("Tinjau distribusi Pembimbing 2 karena relasinya jauh lebih sedikit dibanding Pembimbing 1.")
    elif advisor_2 > advisor_1 * 1.5 and advisor_1 > 0:
        recommendations.append("Tinjau distribusi Pembimbing 1 karena relasinya jauh lebih sedikit dibanding Pembimbing 2.")
    else:
        recommendations.append("Pertahankan komposisi Pembimbing 1 dan Pembimbing 2 karena distribusinya relatif seimbang.")
    recommendations.append("Gunakan chart pembimbing untuk mengidentifikasi dosen yang terlalu sering berada pada posisi pembimbing tertentu.")

    return insights[:5], recommendations[:5]


def build_nlp_insights(df_nlp: pd.DataFrame, target_months: int) -> tuple[list[str], list[str]]:
    if df_nlp.empty:
        return (
            [f"Tidak ada data tugas akhir pada filter aktif dengan lama studi hingga {target_months} bulan."],
            ["Longgarkan filter lama studi atau filter lain untuk melihat pola kemiripan tugas akhir."],
        )

    scored_df = df_nlp.dropna(subset=["skor_kemiripan_tertinggi"]).copy()
    if scored_df.empty:
        return (
            [f"Data tugas akhir tersedia, tetapi skor kemiripan belum tersedia pada filter lama studi hingga {target_months} bulan."],
            ["Lengkapi atau sinkronkan atribut similarity agar insight kemiripan dapat dihitung untuk filter aktif."],
        )

    avg_similarity = scored_df["skor_kemiripan_tertinggi"].mean()
    uniqueness_index = (1 - avg_similarity) * 100
    redundant_count = int(scored_df["skor_kemiripan_tertinggi"].ge(0.85).sum())
    review_count = int(scored_df["skor_kemiripan_tertinggi"].between(0.70, 0.8499).sum())
    category_counts = scored_df["kategori_keunikan"].value_counts()
    dominant_category = category_counts.index[0] if not category_counts.empty else "-"
    keywords = top_keywords(df_nlp["Judul Final"], 5)

    insights = [
        f"Filter aktif menampilkan {len(df_nlp)} judul tugas akhir dengan indeks keunikan {uniqueness_index:.1f}/100.",
        f"Rata-rata similarity tertinggi berada di {avg_similarity * 100:.1f}%, sehingga kedekatan judul berada pada kategori {dominant_category}.",
        f"Terdapat {redundant_count} judul tidak unik dan {review_count} judul perlu review pada ambang similarity yang digunakan.",
        f"Analisis ini sudah dibatasi pada alumni dengan lama studi hingga {target_months} bulan.",
    ]
    if not keywords.empty:
        top_keyword = keywords.iloc[0]
        insights.append(f"Kata kunci paling dominan adalah '{top_keyword['Kata']}' dengan {int(top_keyword['Frekuensi'])} kemunculan.")

    recommendations = []
    if redundant_count:
        recommendations.append("Lakukan review substansi pada judul berkategori tidak unik sebelum periode pengajuan tugas akhir berikutnya.")
    else:
        recommendations.append("Pertahankan proses kurasi judul karena tidak ada judul yang melewati ambang tidak unik pada filter aktif.")
    if review_count:
        recommendations.append("Buat daftar pemantauan untuk judul pada kategori perlu review agar kemiripan konseptual tidak meningkat menjadi redundan.")
    if not keywords.empty:
        recommendations.append(f"Evaluasi kejenuhan topik berbasis kata kunci '{keywords.iloc[0]['Kata']}' dan dorong variasi tema riset baru.")
    recommendations.append("Gunakan peta PCA untuk mengidentifikasi klaster judul yang rapat sebagai bahan diskusi topik prioritas departemen.")
    recommendations.append(f"Bandingkan hasil ini dengan target lama studi lain untuk melihat apakah kemiripan topik berkaitan dengan alumni yang selesai lebih cepat.")

    return insights[:5], recommendations[:5]


def academic_dashboard(df: pd.DataFrame, target_months: int, target_ipk: float) -> None:
    total = len(df)
    avg_ipk = df["IPK"].mean()
    avg_study = df["Lama Studi"].mean()
    on_time = df["Lama Studi"].le(target_months).mean() * 100 if total else 0
    latest_year = int(df["Tahun Wisuda"].max()) if total else "-"

    # Trend calculation for Total Alumni
    trend_html = "dari tahun sebelumnya"
    if total > 0 and "Tahun Wisuda" in df.columns:
        years_sorted = sorted(df["Tahun Wisuda"].dropna().unique())
        if len(years_sorted) >= 2:
            last_yr = years_sorted[-1]
            prev_yr = years_sorted[-2]
            count_last = df[df["Tahun Wisuda"] == last_yr]["NO"].count()
            count_prev = df[df["Tahun Wisuda"] == prev_yr]["NO"].count()
            if count_prev > 0:
                pct_change = ((count_last - count_prev) / count_prev) * 100
                if pct_change > 0:
                    trend_html = f'<span class="trend-up">↑ {pct_change:.0f}%</span> dari tahun sebelumnya'
                elif pct_change < 0:
                    trend_html = f'<span class="trend-down">↓ {abs(pct_change):.0f}%</span> dari tahun sebelumnya'
                else:
                    trend_html = '<span class="trend-neutral">0%</span> dari tahun sebelumnya'
    
    # Subtext for Rata-rata IPK
    ipk_cat = predikat_ipk(avg_ipk) if total else "-"
    
    # Subtext for Tahun Terbaru
    latest_year_graduates = 0
    if total:
        latest_year_graduates = df[df["Tahun Wisuda"] == latest_year]["NO"].count()
        latest_year_subtext = f"{latest_year_graduates} lulusan (s.d Juni)" if latest_year == 2026 else f"{latest_year_graduates} lulusan"
    else:
        latest_year_subtext = "-"

    # Render KPI Cards in columns
    kpi_cols = st.columns(5)
    
    with kpi_cols[0]:
        st.markdown(
            f"""
            <div class="kpi-card-custom">
                <div class="kpi-card-header">
                    <span class="kpi-card-title">Total Alumni</span>
                    <div class="kpi-card-icon icon-alumni">
                        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>
                            <circle cx="9" cy="7" r="4"></circle>
                            <path d="M23 21v-2a4 4 0 0 0-3-3.87"></path>
                            <path d="M16 3.13a4 4 0 0 1 0 7.75"></path>
                        </svg>
                    </div>
                </div>
                <div>
                    <div class="kpi-card-value">{total}</div>
                    <div class="kpi-card-subtext">{trend_html}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with kpi_cols[1]:
        st.markdown(
            f"""
            <div class="kpi-card-custom">
                <div class="kpi-card-header">
                    <span class="kpi-card-title">Rata-rata IPK</span>
                    <div class="kpi-card-icon icon-ipk">
                        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                            <polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline>
                            <polyline points="17 6 23 6 23 12"></polyline>
                        </svg>
                    </div>
                </div>
                <div>
                    <div class="kpi-card-value">{f"{avg_ipk:.2f}" if total else "-"}</div>
                    <div class="kpi-card-subtext">Kategori: {ipk_cat}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with kpi_cols[2]:
        st.markdown(
            f"""
            <div class="kpi-card-custom">
                <div class="kpi-card-header">
                    <span class="kpi-card-title">Rata-rata Lama Studi</span>
                    <div class="kpi-card-icon icon-studi">
                        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                            <circle cx="12" cy="12" r="10"></circle>
                            <polyline points="12 6 12 12 16 14"></polyline>
                        </svg>
                    </div>
                </div>
                <div>
                    <div class="kpi-card-value">{f"{avg_study:.1f}" if total else "-"}</div>
                    <div class="kpi-card-subtext">bulan</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with kpi_cols[3]:
        st.markdown(
            f"""
            <div class="kpi-card-custom">
                <div class="kpi-card-header">
                    <span class="kpi-card-title">Tepat Waktu</span>
                    <div class="kpi-card-icon icon-tepat">
                        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                            <circle cx="12" cy="12" r="10"></circle>
                            <circle cx="12" cy="12" r="6"></circle>
                            <circle cx="12" cy="12" r="2"></circle>
                        </svg>
                    </div>
                </div>
                <div>
                    <div class="kpi-card-value">{f"{on_time:.1f}%" if total else "-"}</div>
                    <div class="kpi-card-subtext">≤ {target_months} bulan</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with kpi_cols[4]:
        st.markdown(
            f"""
            <div class="kpi-card-custom">
                <div class="kpi-card-header">
                    <span class="kpi-card-title">Tahun Terbaru</span>
                    <div class="kpi-card-icon icon-tahun">
                        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                            <rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect>
                            <line x1="16" y1="2" x2="16" y2="6"></line>
                            <line x1="8" y1="2" x2="8" y2="6"></line>
                            <line x1="3" y1="10" x2="21" y2="10"></line>
                        </svg>
                    </div>
                </div>
                <div>
                    <div class="kpi-card-value">{latest_year}</div>
                    <div class="kpi-card-subtext">{latest_year_subtext}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
    insights, recommendations = build_academic_insights(df, target_months, target_ipk)
    render_dynamic_panels(insights, recommendations)
    st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)

    if df.empty:
        fig_empty = plot_empty("Tidak ada data pada filter ini.")
        style_plotly_fig(fig_empty)
        st.plotly_chart(fig_empty, use_container_width=True, config={'displayModeBar': False})
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
        with st.container(border=True):
            fig = px.area(
                yearly,
                x="Tahun Wisuda",
                y="jumlah_lulusan",
                title="Trend Jumlah Lulusan per Tahun",
                labels={"jumlah_lulusan": "Jumlah Lulusan", "Tahun Wisuda": "Tahun Wisuda"},
            )
            # Make the spline line smooth and dashed/dotted, with area fill
            fig.update_traces(
                line=dict(color="#114b36", width=3, shape="spline", dash="dash"),
                fill="tozeroy",
                fillcolor="rgba(17, 75, 54, 0.05)"
            )
            # Add small markers on top
            fig.add_scatter(
                x=yearly["Tahun Wisuda"],
                y=yearly["jumlah_lulusan"],
                mode="markers",
                marker=dict(color="#114b36", size=7, symbol="circle"),
                showlegend=False
            )
            style_plotly_fig(fig)
            fig.update_layout(height=360)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with col_right:
        with st.container(border=True):
            # Sort the Rentang IPK in order from highest to lowest
            ipk_counts = df["Rentang IPK"].value_counts()
            order = ["3.75 - 4.00", "3.50 - 3.74", "3.25 - 3.49", "3.00 - 3.24", "< 3.00"]
            valid_order = [o for o in order if o in ipk_counts.index]
            ipk_counts_df = ipk_counts.reindex(valid_order).reset_index(name="Jumlah")
            
            fig = px.bar(
                ipk_counts_df,
                x="Rentang IPK",
                y="Jumlah",
                title="Distribusi Rentang IPK",
                color="Rentang IPK",
                color_discrete_map={
                    "3.75 - 4.00": "#0b3626", # Dark Green
                    "3.50 - 3.74": "#114b36", # Forest Green
                    "3.25 - 3.49": "#20c997", # Teal Green
                    "3.00 - 3.24": "#86efac", # Soft Green
                    "< 3.00": "#dc2626",      # Crimson Red
                }
            )
            try:
                fig.update_layout(barcornerradius=8)
            except Exception:
                pass
            style_plotly_fig(fig)
            fig.update_layout(height=360, showlegend=False)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    col_left, col_right = st.columns([1.1, 1])
    with col_left:
        with st.container(border=True):
            fig = px.bar(
                period.tail(40),
                x="label",
                y="jumlah_lulusan",
                color="rata_ipk",
                title="Lulusan per Periode Wisuda",
                labels={"label": "Periode", "jumlah_lulusan": "Jumlah", "rata_ipk": "Rata-rata IPK"},
                color_continuous_scale=["#e8f5e9", "#4f826d", "#114b36"],
            )
            style_plotly_fig(fig)
            fig.update_layout(height=390, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with col_right:
        with st.container(border=True):
            study = df.assign(Kategori=df["Lama Studi"].apply(lambda x: kategori_lama_studi(int(x), target_months)))
            study_counts = study["Kategori"].value_counts().rename_axis("Kategori").reset_index(name="Jumlah")
            fig = px.pie(
                study_counts,
                names="Kategori",
                values="Jumlah",
                title="Komposisi Lama Studi",
                hole=.48,
                color="Kategori",
                color_discrete_map={
                    "Tepat Waktu": "#114b36",
                    "Terlambat": "#c23b3f",
                }
            )
            style_plotly_fig(fig)
            fig.update_layout(height=390)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with st.container(border=True):
        fig = px.scatter(
            df,
            x="Lama Studi",
            y="IPK",
            color="Tahun Wisuda",
            custom_data=["Nama", "NIM", "Periode Wisuda", "Tahun Wisuda", "Judul Tugas Akhir"],
            title="Sebaran IPK dan Lama Studi Alumni",
            labels={"Lama Studi": "Lama Studi (bulan)", "IPK": "IPK"},
            color_continuous_scale=["#b8c5a3", "#4f7f70", "#114b36"],
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
        fig.add_vline(x=target_months, line_dash="dash", line_color="#c23b3f", annotation_text=f"Target {target_months} bulan")
        fig.add_hline(y=target_ipk, line_dash="dash", line_color="#c23b3f", annotation_text=f"Target IPK {target_ipk:.2f}")
        style_plotly_fig(fig)
        fig.update_layout(height=460)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with st.expander("Audit Data Alumni"):
        st.markdown(
            '<div class="audit-toolbar-note">Telusuri data akademik alumni berdasarkan nama atau NIM, lalu cek atribut kelulusan utama dalam satu tabel ringkas.</div>',
            unsafe_allow_html=True,
        )
        query = st.text_input("Cari nama/NIM", placeholder="Contoh: Fitrah Annisa Sari atau 2011521001")
        audit_df = df.copy()
        if query.strip():
            pattern = query.strip().lower()
            audit_df = audit_df[
                audit_df["Nama"].astype(str).str.lower().str.contains(pattern, na=False)
                | audit_df["NIM"].astype(str).str.lower().str.contains(pattern, na=False)
            ]
        avg_ipk_audit = audit_df["IPK"].mean() if not audit_df.empty else 0
        avg_study_audit = audit_df["Lama Studi"].mean() if not audit_df.empty else 0
        latest_year_audit = int(audit_df["Tahun Wisuda"].max()) if not audit_df.empty else "-"
        st.markdown(
            f"""
            <div class="audit-summary-grid">
                <div class="audit-summary-card">
                    <div class="audit-summary-label">Record</div>
                    <div class="audit-summary-value">{len(audit_df):,}</div>
                </div>
                <div class="audit-summary-card">
                    <div class="audit-summary-label">Rata IPK</div>
                    <div class="audit-summary-value">{avg_ipk_audit:.2f}</div>
                </div>
                <div class="audit-summary-card">
                    <div class="audit-summary-label">Rata Studi</div>
                    <div class="audit-summary-value">{avg_study_audit:.1f}</div>
                </div>
                <div class="audit-summary-card">
                    <div class="audit-summary-label">Tahun Terbaru</div>
                    <div class="audit-summary-value">{latest_year_audit}</div>
                </div>
            </div>
            """.replace(",", "."),
            unsafe_allow_html=True,
        )
        audit_preview = audit_df[
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
        ].head(50).rename(
            columns={
                "NO": "No",
                "Tahun Wisuda": "Tahun",
                "Tanggal Lulus": "Tanggal Lulus",
                "Lama Studi": "Lama Studi (Bulan)",
            }
        )
        st.dataframe(
            audit_preview.style.format({"IPK": "{:.2f}", "Lama Studi (Bulan)": "{:.0f}"}),
            use_container_width=True,
            hide_index=True,
        )



def lecturer_dashboard(df: pd.DataFrame, schema: StarSchema) -> None:
    st.markdown('<p class="section-note">Analisis dosen memakai bridge table agar satu alumni dapat terhubung ke beberapa pembimbing dan penguji.</p>', unsafe_allow_html=True)
    if schema.bridge_peran_dosen.empty:
        st.info("Tidak ada data peran dosen pada filter ini.")
        return

    bridge = schema.bridge_peran_dosen.merge(schema.dim_dosen, on="dosen_key", how="left")
    unique_lecturers = bridge["dosen_key"].nunique()
    advisor_roles = bridge["jenis_peran"].eq("Pembimbing").sum()
    examiner_roles = bridge["jenis_peran"].eq("Penguji").sum()

    kpi_cols = st.columns(3)
    kpi_cols[0].metric("Dosen", unique_lecturers)
    kpi_cols[1].metric("Peran Pembimbing", advisor_roles)
    kpi_cols[2].metric("Peran Penguji", examiner_roles)

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
            color_discrete_sequence=["#1f644e"],
        )
        style_plotly_fig(fig)
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with col_right:
        fig = px.bar(
            top_role.head(24),
            x="jumlah",
            y="nama_dosen_normalized",
            color="jenis_peran",
            orientation="h",
            title="Komposisi Peran Pembimbing dan Penguji",
            labels={"jumlah": "Jumlah", "nama_dosen_normalized": "Dosen"},
            color_discrete_map={"Pembimbing": "#1f644e", "Penguji": "#b87333"},
        )
        style_plotly_fig(fig)
        fig.update_layout(height=520, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

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
    if role_df.empty:
        insights, recommendations = build_lecturer_insights(role_df)
        render_dynamic_panels(insights, recommendations)
        st.info("Tidak ada data peran dosen pada filter ini.")
        return

    total_roles = int(role_df["role_count"].sum())
    total_alumni = role_df["kelulusan_key"].nunique()
    unique_lecturers = role_df["dosen_key"].nunique()
    advisor_roles = int(role_df.loc[role_df["jenis_peran"].eq("Pembimbing"), "role_count"].sum())
    examiner_roles = int(role_df.loc[role_df["jenis_peran"].eq("Penguji"), "role_count"].sum())

    # Render custom HTML KPI Cards
    kpi_cols = st.columns(4)
    
    with kpi_cols[0]:
        st.markdown(
            f"""
            <div class="kpi-card-custom">
                <div class="kpi-card-header">
                    <span class="kpi-card-title">Alumni Terlibat</span>
                    <div class="kpi-card-icon icon-ipk">
                        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>
                            <circle cx="9" cy="7" r="4"></circle>
                            <path d="M23 21v-2a4 4 0 0 0-3-3.87"></path>
                            <path d="M16 3.13a4 4 0 0 1 0 7.75"></path>
                        </svg>
                    </div>
                </div>
                <div>
                    <div class="kpi-card-value">{total_alumni:,}</div>
                    <div class="kpi-card-subtext">Alumni Unik</div>
                </div>
            </div>
            """.replace(",", "."),
            unsafe_allow_html=True
        )

    with kpi_cols[1]:
        st.markdown(
            f"""
            <div class="kpi-card-custom">
                <div class="kpi-card-header">
                    <span class="kpi-card-title">Dosen</span>
                    <div class="kpi-card-icon icon-tahun">
                        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M2 22s1-4 8-4 8 4 8 4"></path>
                            <circle cx="10" cy="8" r="5"></circle>
                            <path d="M17 11h6"></path>
                        </svg>
                    </div>
                </div>
                <div>
                    <div class="kpi-card-value">{unique_lecturers}</div>
                    <div class="kpi-card-subtext">Dosen Terdaftar</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with kpi_cols[2]:
        st.markdown(
            f"""
            <div class="kpi-card-custom">
                <div class="kpi-card-header">
                    <span class="kpi-card-title">Peran Pembimbing</span>
                    <div class="kpi-card-icon icon-studi">
                        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                            <circle cx="12" cy="12" r="10"></circle>
                            <polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"></polygon>
                        </svg>
                    </div>
                </div>
                <div>
                    <div class="kpi-card-value">{advisor_roles}</div>
                    <div class="kpi-card-subtext">Pembimbing TA</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with kpi_cols[3]:
        st.markdown(
            f"""
            <div class="kpi-card-custom">
                <div class="kpi-card-header">
                    <span class="kpi-card-title">Peran Penguji</span>
                    <div class="kpi-card-icon icon-tepat">
                        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
                            <polyline points="22 4 12 14.01 9 11.01"></polyline>
                        </svg>
                    </div>
                </div>
                <div>
                    <div class="kpi-card-value">{examiner_roles}</div>
                    <div class="kpi-card-subtext">Penguji Sidang</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
    insights, recommendations = build_lecturer_insights(role_df)
    render_dynamic_panels(insights, recommendations)
    st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)

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
        with st.container(border=True):
            fig = px.bar(
                top_role_chart.sort_values("jumlah_peran"),
                x="jumlah",
                y="nama_dosen_normalized",
                color="jenis_peran",
                orientation="h",
                title="Top Dosen Berdasarkan Total Relasi Peran",
                labels={"jumlah": "Jumlah Peran", "nama_dosen_normalized": "Dosen", "jenis_peran": "Jenis Peran"},
                color_discrete_map={"Pembimbing": "#114b36", "Penguji": "#d4a373"},
            )
            try:
                fig.update_layout(barcornerradius=6)
            except Exception:
                pass
            style_plotly_fig(fig)
            fig.update_layout(
                height=520,
                margin=dict(l=10, r=10, t=55, b=10),
                barmode="stack",
                yaxis={"categoryorder": "total ascending"},
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with col_right:
        with st.container(border=True):
            advisor_position = (
                role_df[
                    role_df["jenis_peran"].eq("Pembimbing")
                    & role_df["urutan_peran"].isin([1, 2])
                    & role_df["dosen_key"].isin(top_all["dosen_key"])
                ]
                .assign(Posisi=lambda data: "Pembimbing " + data["urutan_peran"].astype(int).astype(str))
                .groupby(["dosen_key", "nama_dosen_normalized", "Posisi"], as_index=False)
                .agg(jumlah=("role_count", "sum"))
                .merge(top_all[["dosen_key", "jumlah_peran"]], on="dosen_key", how="left")
            )
            fig = px.bar(
                advisor_position.sort_values("jumlah_peran"),
                x="jumlah",
                y="nama_dosen_normalized",
                color="Posisi",
                orientation="h",
                title="Distribusi Dosen Pembimbing 1 dan 2",
                labels={
                    "jumlah": "Jumlah Relasi Pembimbing",
                    "nama_dosen_normalized": "Dosen",
                    "Posisi": "Posisi Pembimbing",
                },
                color_discrete_map={"Pembimbing 1": "#114b36", "Pembimbing 2": "#5b7894"},
            )
            try:
                fig.update_layout(barcornerradius=6)
            except Exception:
                pass
            style_plotly_fig(fig)
            fig.update_layout(
                height=520,
                margin=dict(l=10, r=10, t=55, b=10),
                barmode="stack",
                yaxis={"categoryorder": "total ascending"},
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


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


def nlp_dashboard(df: pd.DataFrame, target_months: int) -> None:
    if df.empty:
        insights, recommendations = build_nlp_insights(df, target_months)
        render_dynamic_panels(insights, recommendations)
        st.info("Tidak ada atribut NLP pada dim_tugas_akhir untuk filter ini.")
        return

    df_nlp = df[df["Lama Studi"].le(target_months)].copy()
    if df_nlp.empty or df_nlp["skor_kemiripan_tertinggi"].isna().all():
        insights, recommendations = build_nlp_insights(df_nlp, target_months)
        render_dynamic_panels(insights, recommendations)
        st.info("Tidak ada atribut NLP pada dim_tugas_akhir untuk filter lama studi ini.")
        return

    avg_similarity = df_nlp["skor_kemiripan_tertinggi"].mean()
    uniqueness_index = (1 - avg_similarity) * 100
    redundant_count = df_nlp["skor_kemiripan_tertinggi"].ge(0.85).sum()
    review_count = df_nlp["skor_kemiripan_tertinggi"].between(0.70, 0.8499).sum()

    # Render custom HTML KPI Cards
    kpi_cols = st.columns(4)
    
    with kpi_cols[0]:
        st.markdown(
            f"""
            <div class="kpi-card-custom">
                <div class="kpi-card-header">
                    <span class="kpi-card-title">Rata-rata Similarity Tertinggi</span>
                    <div class="kpi-card-icon icon-alumni">
                        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                        </svg>
                    </div>
                </div>
                <div>
                    <div class="kpi-card-value">{avg_similarity * 100:.1f}%</div>
                    <div class="kpi-card-subtext">Skor Cosine SBERT</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with kpi_cols[1]:
        st.markdown(
            f"""
            <div class="kpi-card-custom">
                <div class="kpi-card-header">
                    <span class="kpi-card-title">Indeks Keunikan</span>
                    <div class="kpi-card-icon icon-ipk">
                        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M12 2L2 7l10 5 10-5-10-5z"></path>
                            <path d="M2 17l10 5 10-5"></path>
                            <path d="M2 12l10 5 10-5"></path>
                        </svg>
                    </div>
                </div>
                <div>
                    <div class="kpi-card-value">{uniqueness_index:.1f}/100</div>
                    <div class="kpi-card-subtext">Keunikan Dataset</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with kpi_cols[2]:
        st.markdown(
            f"""
            <div class="kpi-card-custom">
                <div class="kpi-card-header">
                    <span class="kpi-card-title">Tidak Unik</span>
                    <div class="kpi-card-icon icon-tepat">
                        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                            <circle cx="12" cy="12" r="10"></circle>
                            <line x1="12" y1="8" x2="12" y2="12"></line>
                            <line x1="12" y1="16" x2="12.01" y2="16"></line>
                        </svg>
                    </div>
                </div>
                <div>
                    <div class="kpi-card-value">{int(redundant_count)}</div>
                    <div class="kpi-card-subtext">Skor similarity &ge; 0.85</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with kpi_cols[3]:
        st.markdown(
            f"""
            <div class="kpi-card-custom">
                <div class="kpi-card-header">
                    <span class="kpi-card-title">Perlu Review</span>
                    <div class="kpi-card-icon icon-studi">
                        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
                            <circle cx="12" cy="12" r="3"></circle>
                        </svg>
                    </div>
                </div>
                <div>
                    <div class="kpi-card-value">{int(review_count)}</div>
                    <div class="kpi-card-subtext">Skor similarity 0.70 - 0.84</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)
    insights, recommendations = build_nlp_insights(df_nlp, target_months)
    render_dynamic_panels(insights, recommendations)
    st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)



    col_left, col_right = st.columns([1.2, 1])
    with col_left:
        with st.container(border=True):
            fig = px.histogram(
                df_nlp,
                x="skor_kemiripan_tertinggi",
                nbins=30,
                color="kategori_keunikan",
                title="Distribusi Skor SBERT Cosine Similarity Tertinggi",
                labels={"skor_kemiripan_tertinggi": "Skor Cosine Similarity", "count": "Jumlah Judul"},
                color_discrete_map={
                    "Unik": "#114b36",
                    "Agak Unik / Perlu Review": "#b87333",
                    "Tidak Unik": "#c23b3f",
                },
            )
            fig.add_vline(x=0.70, line_dash="dash", line_color="#b87333", annotation_text="Review 0.70")
            fig.add_vline(x=0.85, line_dash="dash", line_color="#c23b3f", annotation_text="Tidak unik 0.85")
            style_plotly_fig(fig)
            fig.update_layout(height=410)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with col_right:
        with st.container(border=True):
            fig = go.Figure()
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=uniqueness_index,
                    title={"text": "Indeks Keunikan Dataset TA"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#114b36"},
                        "steps": [
                            {"range": [0, 40], "color": "#efd2cf"},
                            {"range": [40, 70], "color": "#eee0b6"},
                            {"range": [70, 100], "color": "#d8e6d5"},
                        ],
                        "threshold": {"line": {"color": "#c23b3f", "width": 4}, "value": uniqueness_index},
                    },
                )
            )
            style_plotly_fig(fig)
            fig.update_layout(height=410)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    col_left, col_right = st.columns([1.1, 1])
    with col_left:
        with st.container(border=True):
            scatter_df = df_nlp.dropna(subset=["pca_x", "pca_y"]).copy()
            if scatter_df.empty:
                fig_empty = plot_empty("Koordinat PCA belum tersedia di dim_tugas_akhir.")
                style_plotly_fig(fig_empty)
                st.plotly_chart(fig_empty, use_container_width=True, config={'displayModeBar': False})
            else:
                fig = px.scatter(
                    scatter_df,
                    x="pca_x",
                    y="pca_y",
                    color="kategori_keunikan",
                    hover_data=["Nama", "Judul Tugas Akhir", "skor_kemiripan_tertinggi", "judul_termirip"],
                    title="Peta Kedekatan Judul TA dari Koordinat PCA SBERT",
                    color_discrete_map={
                        "Unik": "#114b36",
                        "Agak Unik / Perlu Review": "#b87333",
                        "Tidak Unik": "#c23b3f",
                    },
                )
                style_plotly_fig(fig)
                fig.update_layout(height=460)
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with col_right:
        with st.container(border=True):
            keywords = top_keywords(df_nlp["Judul Final"], 18)
            fig = px.bar(
                keywords.sort_values("Frekuensi"),
                x="Frekuensi",
                y="Kata",
                orientation="h",
                title="Kata Kunci Dominan Setelah Stopword Removal",
                color_discrete_sequence=["#5b7894"],
            )
            try:
                fig.update_layout(barcornerradius=4)
            except Exception:
                pass
            style_plotly_fig(fig)
            fig.update_layout(height=460)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

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



def get_next_key(cur, table, key_column):
    cur.execute(f"SELECT COALESCE(MAX({key_column}), 0) + 1 FROM {table}")
    return cur.fetchone()[0]


def save_records_to_db(records_list: list[dict]) -> tuple[bool, str]:
    db_url = get_database_url()
    try:
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                for r in records_list:
                    # 1. Check or insert dim_mahasiswa
                    cur.execute("SELECT mahasiswa_key FROM dim_mahasiswa WHERE nim = %s", (str(r["nim"]),))
                    row = cur.fetchone()
                    if row:
                        mahasiswa_key = row[0]
                    else:
                        mahasiswa_key = get_next_key(cur, "dim_mahasiswa", "mahasiswa_key")
                        cur.execute(
                            "INSERT INTO dim_mahasiswa (mahasiswa_key, nim, nama) VALUES (%s, %s, %s)",
                            (mahasiswa_key, str(r["nim"]), r["nama"])
                        )
                    
                    # 2. Check or insert dim_waktu
                    tanggal_lulus = pd.to_datetime(r["tanggal_lulus"]).date()
                    cur.execute("SELECT waktu_key FROM dim_waktu WHERE tanggal_lulus = %s", (tanggal_lulus,))
                    row = cur.fetchone()
                    if row:
                        waktu_key = row[0]
                    else:
                        waktu_key = get_next_key(cur, "dim_waktu", "waktu_key")
                        cur.execute(
                            "INSERT INTO dim_waktu (waktu_key, tanggal_lulus, hari, bulan, tahun) VALUES (%s, %s, %s, %s, %s)",
                            (waktu_key, tanggal_lulus, tanggal_lulus.day, tanggal_lulus.month, tanggal_lulus.year)
                        )
                        
                    # 3. Check or insert dim_periode_wisuda
                    cur.execute(
                        "SELECT periode_key FROM dim_periode_wisuda WHERE tahun_wisuda = %s AND periode_label = %s",
                        (int(r["tahun_wisuda"]), r["periode_wisuda"])
                    )
                    row = cur.fetchone()
                    if row:
                        periode_key = row[0]
                    else:
                        periode_key = get_next_key(cur, "dim_periode_wisuda", "periode_key")
                        p_num = PERIODE_ORDER.get(r["periode_wisuda"], 99)
                        cur.execute(
                            "INSERT INTO dim_periode_wisuda (periode_key, tahun_wisuda, periode_num, periode_label) VALUES (%s, %s, %s, %s)",
                            (periode_key, int(r["tahun_wisuda"]), p_num, r["periode_wisuda"])
                        )
                        
                    # 4. Check or insert dim_ipk
                    ipk_val = float(r["ipk"])
                    r_ipk = rentang_ipk(ipk_val)
                    p_ipk = predikat_ipk(ipk_val)
                    cur.execute(
                        "SELECT ipk_key FROM dim_ipk WHERE ipk_numeric = %s",
                        (ipk_val,)
                    )
                    row = cur.fetchone()
                    if row:
                        ipk_key = row[0]
                    else:
                        ipk_key = get_next_key(cur, "dim_ipk", "ipk_key")
                        cur.execute(
                            "INSERT INTO dim_ipk (ipk_key, ipk_numeric, rentang_ipk, predikat_ipk) VALUES (%s, %s, %s, %s)",
                            (ipk_key, ipk_val, r_ipk, p_ipk)
                        )
                        
                    # 5. Check or insert dim_lama_studi
                    months_val = int(r["lama_studi"])
                    k_studi = kategori_lama_studi(months_val, 54)
                    f_tepat = months_val <= 54
                    cur.execute(
                        "SELECT lama_studi_key FROM dim_lama_studi WHERE lama_studi_bulan = %s",
                        (months_val,)
                    )
                    row = cur.fetchone()
                    if row:
                        lama_studi_key = row[0]
                    else:
                        lama_studi_key = get_next_key(cur, "dim_lama_studi", "lama_studi_key")
                        cur.execute(
                            "INSERT INTO dim_lama_studi (lama_studi_key, lama_studi_bulan, kategori_lama_studi, flag_tepat_waktu) VALUES (%s, %s, %s, %s)",
                            (lama_studi_key, months_val, k_studi, f_tepat)
                        )
                        
                    # 6. Insert dim_tugas_akhir
                    ta_key = get_next_key(cur, "dim_tugas_akhir", "ta_key")
                    j_clean = clean_title(r["judul_ta"])
                    j_final = final_title(r["judul_ta"])
                    cur.execute(
                        "INSERT INTO dim_tugas_akhir (ta_key, judul_tugas_akhir, judul_preprocessed, judul_final, skor_kemiripan_tertinggi, kategori_keunikan, ta_key_termirip, pca_x, pca_y) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                        (ta_key, r["judul_ta"], j_clean, j_final, None, "Belum Dihitung", None, None, None)
                    )
                    
                    # 7. Insert fact_kelulusan
                    kelulusan_key = get_next_key(cur, "fact_kelulusan", "kelulusan_key")
                    cur.execute(
                        "INSERT INTO fact_kelulusan (kelulusan_key, mahasiswa_key, waktu_key, periode_key, ipk_key, lama_studi_key, ta_key, ipk, lama_studi_bulan, flag_tepat_waktu, jumlah_record) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                        (kelulusan_key, mahasiswa_key, waktu_key, periode_key, ipk_key, lama_studi_key, ta_key, ipk_val, months_val, f_tepat, 1)
                    )
                    
                    # 8. Check and insert Lecturers in bridge table
                    role_columns = [
                        ("pembimbing_1", "Pembimbing", 1),
                        ("pembimbing_2", "Pembimbing", 2),
                        ("penguji_1", "Penguji", 1),
                        ("penguji_2", "Penguji", 2),
                        ("penguji_3", "Penguji", 3),
                    ]
                    for key_name, jenis_peran, urutan_peran in role_columns:
                        val = r.get(key_name)
                        normalized = normalize_person_name(val)
                        if normalized is None:
                            continue
                        
                        # Get or insert dim_dosen
                        cur.execute("SELECT dosen_key FROM dim_dosen WHERE nama_dosen_normalized = %s", (normalized,))
                        row = cur.fetchone()
                        if row:
                            dosen_key = row[0]
                        else:
                            dosen_key = get_next_key(cur, "dim_dosen", "dosen_key")
                            cur.execute(
                                "INSERT INTO dim_dosen (dosen_key, nama_dosen_normalized, nama_asal) VALUES (%s, %s, %s)",
                                (dosen_key, normalized, str(val).strip())
                            )
                            
                        # Insert bridge_peran_dosen
                        peran_key = get_next_key(cur, "bridge_peran_dosen", "peran_dosen_key")
                        cur.execute(
                            "INSERT INTO bridge_peran_dosen (peran_dosen_key, kelulusan_key, dosen_key, jenis_peran, urutan_peran, role_count) VALUES (%s, %s, %s, %s, %s, %s)",
                            (peran_key, kelulusan_key, dosen_key, jenis_peran, urutan_peran, 1)
                        )
            conn.commit()
        return True, "Data berhasil disimpan ke database!"
    except Exception as e:
        return False, str(e)


def render_header(df: pd.DataFrame) -> None:
    min_year = int(df["Tahun Wisuda"].min())
    max_year = int(df["Tahun Wisuda"].max())
    st.title("Lensa DSI")
    st.markdown(
        f"""
        <div style="margin-top: -0.8rem; margin-bottom: 1.5rem;">
            <div style="font-family: 'Outfit', sans-serif; font-size: 1.25rem; font-weight: 600; color: #1f644e; line-height: 1.2;">
                Learning, Evaluation & Analytics System
            </div>
            <div style="font-size: 0.95rem; color: #5e6b66; margin-top: 0.25rem;">
                Departemen Sistem Informasi Universitas Andalas — Data Mart Akademik Alumni (Wisuda {min_year}-{max_year})
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_css()
    inject_sidebar_resize_guard()
    inject_manual_form_input_guard()

    # Render Logo and Brand in Sidebar (Compact, Premium Front-end Layout)
    logo_path = Path("img/unandlogo.png")
    logo_html = ""
    if logo_path.exists():
        try:
            logo_base64 = get_base64_image(logo_path)
            logo_html = f'<img src="data:image/png;base64,{logo_base64}" class="brand-logo-img" />'
        except Exception:
            logo_html = """
            <div class="logo-box">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="logo-cap">
                    <path d="M22 10v6M2 10l10-5 10 5-10 5z"/>
                    <path d="M6 12v5c0 2 2 3 6 3s6-1 6-3v-5"/>
                </svg>
            </div>
            """
    else:
        logo_html = """
        <div class="logo-box">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="logo-cap">
                <path d="M22 10v6M2 10l10-5 10 5-10 5z"/>
                <path d="M6 12v5c0 2 2 3 6 3s6-1 6-3v-5"/>
            </svg>
        </div>
        """

    st.sidebar.markdown(
        f"""
        <div class="sidebar-brand-container">
            {logo_html}
            <div class="brand-text">
                <div class="brand-title">LENSA DSI</div>
                <div class="brand-subtitle">Analytics Dashboard</div>
            </div>
        </div>
        <div class="brand-org">Universitas Andalas</div>
        <div class="brand-divider"></div>
        """,
        unsafe_allow_html=True
    )

    try:
        # Load data first to get years and database status
        df = load_dashboard_data_from_db()
        role_df = load_roles_data_from_db()
        schema = load_star_schema_from_db()
    except Exception as exc:
        render_database_setup(exc)
        return

    if df.empty:
        render_database_setup(Exception("vw_dashboard_kelulusan kosong. Jalankan scripts/load_star_schema.py untuk memuat data."))
        return

    # Render Data Source Card inside a container to avoid overlapping issues
    now = datetime.datetime.now()
    months = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Ags", "Sep", "Okt", "Nov", "Des"]
    current_time_str = f"{now.day:02d} {months[now.month-1]} {now.year} - {now.strftime('%H:%M')}"
    
    with st.sidebar.container(border=True):
        st.markdown(
            f"""
            <div class="card-header" style="margin-bottom: 12px;">
                <svg class="db-icon" xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                    <ellipse cx="12" cy="5" rx="9" ry="3"></ellipse>
                    <path d="M3 5V19A9 3 0 0 0 21 19V5"></path>
                    <path d="M3 12A9 3 0 0 0 21 12"></path>
                </svg>
                <span class="card-title">Data Source</span>
            </div>
            <div class="card-row" style="margin-bottom: 8px;">
                <span class="row-label">Status:</span>
                <span class="status-badge"><span class="dot"></span>Connected</span>
            </div>
            <div class="card-row-stack" style="margin-bottom: 8px;">
                <div class="row-label">Last Update:</div>
                <div class="update-time">{current_time_str}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("🔄 Sync Data", key="sync_data_btn", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    render_header(df)

    if "page" not in st.session_state:
        st.session_state.page = "Dashboard"

    style_lines = []

    st.sidebar.markdown(
        '<div style="font-family: \'Outfit\', sans-serif; font-size: 0.85rem; font-weight: 700; color: #5e6b66; margin-bottom: 8px; margin-top: 10px;">NAVIGASI</div>',
        unsafe_allow_html=True
    )
    col_nav1, col_nav2 = st.sidebar.columns(2)
    with col_nav1:
        if st.button("📊 Dashboard", key="btn_nav_dash", use_container_width=True):
            st.session_state.page = "Dashboard"
            st.rerun()
    with col_nav2:
        if st.button("👤 Tendik", key="btn_nav_tendik", use_container_width=True):
            st.session_state.page = "Tendik"
            st.rerun()

    # Dynamic CSS for sidebar navigation buttons active state
    nav_css = []
    if st.session_state.page == "Dashboard":
        nav_css.append("""
        [data-testid="stSidebar"] .st-key-btn_nav_dash button {
            background-color: var(--accent) !important;
            color: #ffffff !important;
            border-color: var(--accent) !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 10px rgba(17, 75, 54, 0.16) !important;
        }
        [data-testid="stSidebar"] .st-key-btn_nav_tendik button {
            background-color: #ffffff !important;
            color: #475569 !important;
            border: 1px solid #cbd5e1 !important;
        }
        [data-testid="stSidebar"] .st-key-btn_nav_tendik button:hover {
            border-color: var(--accent) !important;
            color: var(--accent) !important;
            background-color: var(--accent-soft) !important;
        }
        """)
    else:
        nav_css.append("""
        [data-testid="stSidebar"] .st-key-btn_nav_tendik button {
            background-color: var(--accent) !important;
            color: #ffffff !important;
            border-color: var(--accent) !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 10px rgba(17, 75, 54, 0.16) !important;
        }
        [data-testid="stSidebar"] .st-key-btn_nav_dash button {
            background-color: #ffffff !important;
            color: #475569 !important;
            border: 1px solid #cbd5e1 !important;
        }
        [data-testid="stSidebar"] .st-key-btn_nav_dash button:hover {
            border-color: var(--accent) !important;
            color: var(--accent) !important;
            background-color: var(--accent-soft) !important;
        }
        """)
    st.sidebar.markdown('<div class="brand-divider" style="margin-top: 10px; margin-bottom: 15px;"></div>', unsafe_allow_html=True)

    if st.session_state.page == "Dashboard":
        # Render Filters
        st.sidebar.markdown('<div class="filter-title">Filter Dashboard</div>', unsafe_allow_html=True)
        
        min_year = int(df["Tahun Wisuda"].min())
        max_year = int(df["Tahun Wisuda"].max())
        
        with st.sidebar.expander("Tahun Wisuda", expanded=True):
            years = st.slider("Pilih Tahun Wisuda", min_year, max_year, (min_year, max_year), label_visibility="collapsed")
            
        periode_options = [item[0] for item in PERIODE_FILTER_OPTIONS]
        with st.sidebar.expander("Periode Wisuda", expanded=True):
            if "selected_periods" not in st.session_state:
                st.session_state.selected_periods = set(df["Periode Wisuda"].dropna().unique().tolist())
            st.session_state.selected_periods = set(st.session_state.selected_periods).intersection(set(periode_options))
                
            # Checkbox Squares timeline row
            for row_start in range(0, len(PERIODE_FILTER_OPTIONS), 3):
                cols = st.columns(3)
                for offset, (p_name, roman_label, month_label, _) in enumerate(PERIODE_FILTER_OPTIONS[row_start:row_start + 3]):
                    idx = row_start + offset
                    is_sel = p_name in st.session_state.selected_periods
                    with cols[offset]:
                        if st.button(roman_label, key=f"circle_btn_{idx}", use_container_width=True):
                             if is_sel:
                                 st.session_state.selected_periods.remove(p_name)
                             else:
                                 st.session_state.selected_periods.add(p_name)
                             st.rerun()
                        st.markdown(f'<div class="circle-label" style="text-align: center; margin-top: 4px; font-weight: 600; font-size: 0.75rem; color: #64748b;">{month_label}</div>', unsafe_allow_html=True)
                    
            # Generate dynamic CSS for current state
            for idx, (p_name, _, _, _) in enumerate(PERIODE_FILTER_OPTIONS):
                is_sel = p_name in st.session_state.selected_periods
                
                # Checkbox square styles
                if is_sel:
                    style_lines.append(f"""
                    [data-testid="stSidebar"] [data-testid="stSidebarContent"] details div[data-testid="stHorizontalBlock"] div[data-testid="stColumn"] .st-key-circle_btn_{idx} div[data-testid="stButton"] button[data-testid="stBaseButton-secondary"] {{
                        background-color: var(--accent) !important;
                        color: #ffffff !important;
                        border: 2px solid var(--accent) !important;
                        box-shadow: 0 5px 12px rgba(17, 75, 54, 0.16) !important;
                    }}
                    """)
                else:
                    style_lines.append(f"""
                    [data-testid="stSidebar"] [data-testid="stSidebarContent"] details div[data-testid="stHorizontalBlock"] div[data-testid="stColumn"] .st-key-circle_btn_{idx} div[data-testid="stButton"] button[data-testid="stBaseButton-secondary"] {{
                        background-color: #ffffff !important;
                        color: #475569 !important;
                        border: 1px solid #cbd5e1 !important;
                        box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
                    }}
                    """)
                    
            selected_periode = list(st.session_state.selected_periods)
            
        ipk_min = float(np.floor(df["IPK"].min() * 10) / 10)
        ipk_max = float(np.ceil(df["IPK"].max() * 10) / 10)
        with st.sidebar.expander("Rentang IPK", expanded=False):
            ipk_range = st.slider("Pilih Rentang IPK", ipk_min, ipk_max, (ipk_min, ipk_max), step=0.01, label_visibility="collapsed")
            
        with st.sidebar.expander("Lama Studi (Bulan)", expanded=False):
            st.markdown('<div class="expander-subtitle">Target Tepat Waktu:</div>', unsafe_allow_html=True)
            target_months = st.slider("Target Tepat Waktu (bulan)", 42, 54, 54, step=1, label_visibility="collapsed")
            st.markdown(f'<div class="expander-note">Target aktif: <b>{target_months} bulan</b>.</div>', unsafe_allow_html=True)

        filtered = filter_dataframe(df, years, selected_periode, ipk_range)
        filtered_roles = filter_roles_dataframe(role_df, years, selected_periode, ipk_range)

        tabs = st.tabs(["Akademik", "Dosen", "Kemiripan Tugas Akhir"])
        with tabs[0]:
            academic_dashboard(filtered, target_months, ipk_range[0])
        with tabs[1]:
            lecturer_dashboard_from_roles(filtered_roles)
        with tabs[2]:
            nlp_dashboard(filtered, target_months)

    else:  # Tendik page
        tab_excel, tab_manual = st.tabs(["📁 Unggah via Excel/CSV", "✏️ Input Manual Satu-Satu"])
        
        if "tendik_preview_records" not in st.session_state:
            st.session_state.tendik_preview_records = []
            
        with tab_excel:
            st.markdown("### Unggah Data Berkas Excel / CSV")
            st.markdown(
                """
                Pastikan berkas Excel (.xlsx, .xls) atau CSV (.csv) yang diunggah memiliki kolom dengan format berikut:
                `NIM`, `Nama`, `Judul Tugas Akhir`, `Tanggal Lulus` (YYYY-MM-DD), `IPK`, `Lama Studi` (bulan), 
                `Periode Wisuda` (WISUDA I s.d V), `Tahun Wisuda`, `Pembimbing 1`, `Pembimbing 2`, `Dosen Penguji 1`, `Dosen Penguji 2`, `Dosen Penguji 3`.
                """
            )
            excel_file = st.file_uploader("Pilih berkas Excel atau CSV", type=["xlsx", "xls", "csv"])
            if excel_file is not None:
                try:
                    if excel_file.name.endswith(".csv"):
                        uploaded_df = pd.read_csv(excel_file)
                    else:
                        uploaded_df = pd.read_excel(excel_file)
                        
                    # Normalize columns and map them
                    def normalize_header(h: str) -> str:
                        h = str(h).strip().lower()
                        h = re.sub(r'[\s_\-\(\)]+', '', h)
                        return h

                    mapped_cols = {}
                    for col in uploaded_df.columns:
                        norm = normalize_header(col)
                        if "nim" in norm:
                            mapped_cols["nim"] = col
                        elif "nama" in norm:
                            mapped_cols["nama"] = col
                        elif "judul" in norm:
                            mapped_cols["judul_ta"] = col
                        elif "tanggal" in norm or "tgl" in norm:
                            mapped_cols["tanggal_lulus"] = col
                        elif "ipk" in norm:
                            mapped_cols["ipk"] = col
                        elif "lama" in norm:
                            mapped_cols["lama_studi"] = col
                        elif "periode" in norm:
                            mapped_cols["periode_wisuda"] = col
                        elif "tahun" in norm:
                            mapped_cols["tahun_wisuda"] = col
                        elif "pembimbing1" in norm or ("pembimbing" in norm and "1" in norm):
                            mapped_cols["pembimbing_1"] = col
                        elif "pembimbing2" in norm or ("pembimbing" in norm and "2" in norm):
                            mapped_cols["pembimbing_2"] = col
                        elif "penguji1" in norm or ("penguji" in norm and "1" in norm):
                            mapped_cols["penguji_1"] = col
                        elif "penguji2" in norm or ("penguji" in norm and "2" in norm):
                            mapped_cols["penguji_2"] = col
                        elif "penguji3" in norm or ("penguji" in norm and "3" in norm):
                            mapped_cols["penguji_3"] = col

                    required = ["nim", "nama", "judul_ta", "tanggal_lulus", "ipk", "lama_studi", "periode_wisuda", "tahun_wisuda"]
                    missing = [r for r in required if r not in mapped_cols]
                    
                    if missing:
                        st.error(f"Berkas kekurangan atau tidak mengenali kolom wajib: {', '.join(missing)}")
                    else:
                        parsed_records = []
                        error_rows = []
                        num_rows = len(uploaded_df)
                        
                        for r_idx in range(num_rows):
                            try:
                                # Parse NIM
                                raw_nim = uploaded_df[mapped_cols["nim"]].iloc[r_idx]
                                if pd.isna(raw_nim):
                                    raise ValueError("NIM kosong")
                                nim_str = str(raw_nim).strip()
                                if nim_str.endswith(".0"):
                                    nim_str = nim_str[:-2]
                                
                                # Parse Nama
                                raw_nama = uploaded_df[mapped_cols["nama"]].iloc[r_idx]
                                if pd.isna(raw_nama):
                                    raise ValueError("Nama kosong")
                                nama_str = str(raw_nama).strip()
                                
                                # Parse Judul
                                raw_judul = uploaded_df[mapped_cols["judul_ta"]].iloc[r_idx]
                                if pd.isna(raw_judul):
                                    raise ValueError("Judul TA kosong")
                                judul_str = str(raw_judul).strip()
                                
                                # Parse Tanggal Lulus
                                raw_tgl = uploaded_df[mapped_cols["tanggal_lulus"]].iloc[r_idx]
                                if pd.isna(raw_tgl):
                                    raise ValueError("Tanggal lulus kosong")
                                tgl_parsed = pd.to_datetime(raw_tgl)
                                tgl_str = tgl_parsed.strftime("%Y-%m-%d")
                                
                                # Parse IPK
                                raw_ipk = uploaded_df[mapped_cols["ipk"]].iloc[r_idx]
                                if pd.isna(raw_ipk):
                                    raise ValueError("IPK kosong")
                                ipk_val = float(raw_ipk)
                                if not (0.0 <= ipk_val <= 4.0):
                                    raise ValueError(f"IPK {ipk_val} di luar rentang 0-4")
                                    
                                # Parse Lama Studi
                                raw_studi = uploaded_df[mapped_cols["lama_studi"]].iloc[r_idx]
                                if pd.isna(raw_studi):
                                    raise ValueError("Lama studi kosong")
                                studi_val = int(float(raw_studi))
                                
                                # Parse Periode Wisuda
                                raw_periode = uploaded_df[mapped_cols["periode_wisuda"]].iloc[r_idx]
                                if pd.isna(raw_periode):
                                    raise ValueError("Periode wisuda kosong")
                                periode_str = str(raw_periode).strip().upper()
                                # Normalize Roman numerals
                                if not periode_str.startswith("WISUDA "):
                                    norm_p = periode_str.replace("WISUDA", "").strip()
                                    if norm_p in ["1", "I"]:
                                        periode_str = "WISUDA I"
                                    elif norm_p in ["2", "II"]:
                                        periode_str = "WISUDA II"
                                    elif norm_p in ["3", "III"]:
                                        periode_str = "WISUDA III"
                                    elif norm_p in ["4", "IV"]:
                                        periode_str = "WISUDA IV"
                                    elif norm_p in ["5", "V"]:
                                        periode_str = "WISUDA V"
                                    else:
                                        periode_str = f"WISUDA {norm_p}"
                                
                                # Parse Tahun Wisuda
                                raw_tahun = uploaded_df[mapped_cols["tahun_wisuda"]].iloc[r_idx]
                                if pd.isna(raw_tahun):
                                    raise ValueError("Tahun wisuda kosong")
                                tahun_val = int(float(raw_tahun))
                                
                                # Optional fields
                                p1 = uploaded_df[mapped_cols["pembimbing_1"]].iloc[r_idx] if "pembimbing_1" in mapped_cols else None
                                p2 = uploaded_df[mapped_cols["pembimbing_2"]].iloc[r_idx] if "pembimbing_2" in mapped_cols else None
                                u1 = uploaded_df[mapped_cols["penguji_1"]].iloc[r_idx] if "penguji_1" in mapped_cols else None
                                u2 = uploaded_df[mapped_cols["penguji_2"]].iloc[r_idx] if "penguji_2" in mapped_cols else None
                                u3 = uploaded_df[mapped_cols["penguji_3"]].iloc[r_idx] if "penguji_3" in mapped_cols else None
                                
                                rec = {
                                    "nim": nim_str,
                                    "nama": nama_str,
                                    "judul_ta": judul_str,
                                    "tanggal_lulus": tgl_str,
                                    "ipk": ipk_val,
                                    "lama_studi": studi_val,
                                    "periode_wisuda": periode_str,
                                    "tahun_wisuda": tahun_val,
                                    "pembimbing_1": str(p1).strip() if pd.notna(p1) and str(p1).strip() != "" else None,
                                    "pembimbing_2": str(p2).strip() if pd.notna(p2) and str(p2).strip() != "" else None,
                                    "penguji_1": str(u1).strip() if pd.notna(u1) and str(u1).strip() != "" else None,
                                    "penguji_2": str(u2).strip() if pd.notna(u2) and str(u2).strip() != "" else None,
                                    "penguji_3": str(u3).strip() if pd.notna(u3) and str(u3).strip() != "" else None,
                                }
                                parsed_records.append(rec)
                            except Exception as row_err:
                                error_rows.append(f"Baris {r_idx + 2}: {row_err}")

                        if error_rows:
                            with st.expander(f"⚠️ {len(error_rows)} baris dengan format tidak sesuai dilewati", expanded=False):
                                st.write("\n".join(error_rows))
                                
                        if parsed_records:
                            if st.button("Tambahkan Data Berkas ke Preview", key="btn_add_excel_to_preview", use_container_width=True):
                                st.session_state.tendik_preview_records.extend(parsed_records)
                                st.success(f"{len(parsed_records)} data berhasil ditambahkan ke preview!")
                                st.rerun()
                except Exception as e:
                    st.error(f"Gagal membaca berkas: {e}")
                    
        with tab_manual:
            st.markdown("### Formulir Input Manual Alumni")
            lecturer_options = get_lecturer_options(schema.dim_dosen)
            lecturer_select_options = [""] + lecturer_options
            with st.form("form_manual_input", clear_on_submit=True):
                col_form1, col_form2 = st.columns(2)
                with col_form1:
                    m_nim = st.text_input("NIM *", value="2011521001", placeholder="Contoh: 2011521001")
                    m_nama = st.text_input("Nama Lengkap *", placeholder="Contoh: Fitrah Annisa Sari")
                    m_judul = st.text_area("Judul Tugas Akhir *", placeholder="Tuliskan judul tugas akhir lengkap...")
                    m_tgl = st.date_input("Tanggal Lulus *", value=datetime.date.today())
                with col_form2:
                    m_ipk = st.text_input("IPK Lulus *", value="3,50", placeholder="Contoh: 3,50")
                    m_studi = st.text_input("Lama Studi (Bulan) *", value="48", placeholder="Contoh: 48")
                    m_periode = st.selectbox("Periode Wisuda *", [item[0] for item in PERIODE_FILTER_OPTIONS])
                    m_tahun = st.text_input("Tahun Wisuda *", value=str(datetime.date.today().year), placeholder="Contoh: 2026")
                    
                st.markdown("**Peran Dosen (Opsional)**")
                col_dosen1, col_dosen2 = st.columns(2)
                with col_dosen1:
                    m_pem1 = st.selectbox("Dosen Pembimbing 1", lecturer_select_options, key="manual_pembimbing_1")
                    m_pem2 = st.selectbox("Dosen Pembimbing 2", lecturer_select_options, key="manual_pembimbing_2")
                with col_dosen2:
                    m_peng1 = st.selectbox("Dosen Penguji 1", lecturer_select_options, key="manual_penguji_1")
                    m_peng2 = st.selectbox("Dosen Penguji 2", lecturer_select_options, key="manual_penguji_2")
                    m_peng3 = st.selectbox("Dosen Penguji 3", lecturer_select_options, key="manual_penguji_3")
                    
                submit_manual = st.form_submit_button("Tambahkan ke Preview", use_container_width=True)
                if submit_manual:
                    ipk_value, validation_errors = validate_manual_record(m_nim, m_nama, m_judul, m_ipk)
                    if not m_studi.strip():
                        validation_errors.append("Lama Studi wajib diisi angka.")
                    if not re.fullmatch(r"\d{4}", m_tahun.strip()):
                        validation_errors.append("Tahun Wisuda harus 4 angka.")
                    if validation_errors:
                        st.error("Periksa kembali input berikut:\n- " + "\n- ".join(validation_errors))
                    else:
                        new_rec = {
                            "nim": m_nim.strip(),
                            "nama": " ".join(m_nama.split()),
                            "judul_ta": m_judul.strip(),
                            "tanggal_lulus": str(m_tgl),
                            "ipk": float(ipk_value),
                            "lama_studi": int(m_studi.strip()),
                            "periode_wisuda": m_periode.upper(),
                            "tahun_wisuda": int(m_tahun.strip()),
                            "pembimbing_1": m_pem1 if m_pem1 else None,
                            "pembimbing_2": m_pem2 if m_pem2 else None,
                            "penguji_1": m_peng1 if m_peng1 else None,
                            "penguji_2": m_peng2 if m_peng2 else None,
                            "penguji_3": m_peng3 if m_peng3 else None,
                        }
                        st.session_state.tendik_preview_records.append(new_rec)
                        st.toast("Data ditambahkan ke preview!", icon="✏️")
                        st.success("Data berhasil ditambahkan ke preview.")

        # Preview Section
        st.markdown("---")
        st.markdown("### Preview Data Baru")
        
        if st.session_state.tendik_preview_records:
            preview_df = pd.DataFrame(st.session_state.tendik_preview_records)
            st.dataframe(preview_df, use_container_width=True)
            
            col_act1, col_act2 = st.columns([1, 4])
            with col_act1:
                if st.button("🗑️ Hapus Semua Preview", key="btn_clear_preview", use_container_width=True):
                    st.session_state.tendik_preview_records = []
                    st.rerun()
            with col_act2:
                if st.button("💾 Simpan Permanen ke Database PostgreSQL", key="btn_save_to_db", use_container_width=True):
                    with st.spinner("Menyimpan ke PostgreSQL..."):
                        success, message = save_records_to_db(st.session_state.tendik_preview_records)
                        if success:
                            st.success(message)
                            st.session_state.tendik_preview_records = []
                            st.cache_data.clear()
                        else:
                            st.error(f"Gagal menyimpan ke database: {message}")
        else:
            st.info("Belum ada data baru di preview. Silakan masukkan data manual atau unggah Excel di atas.")    # Render all dynamic CSS at the root level of the page
    all_dynamic_css = []
    if 'nav_css' in locals() and nav_css:
        all_dynamic_css.extend(nav_css)
    if 'style_lines' in locals() and style_lines:
        all_dynamic_css.extend(style_lines)
        
    if all_dynamic_css:
        st.markdown(f"<style>{''.join(all_dynamic_css)}</style>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
