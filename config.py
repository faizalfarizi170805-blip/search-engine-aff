"""Semua pengaturan aplikasi + pembaca korpus (dipakai app.py, dev_panel.py, tools.py)."""
import json
from dataclasses import fields
from pathlib import Path

import pandas as pd

from ir.engine import SearchConfig

BASE = Path(__file__).resolve().parent

APP_TITLE = "ASEAN Football Championship Search Engine"
TOP_N_OPTIONS = (5, 10, 20, 60)   # pilihan "Tampilkan: Top N"
TOP_N_DEFAULT = 5

# Korpus: pakai file pertama yang ada.
_CORPUS_CANDIDATES = ["Corpus_PialaAFF_Preprocessed.xlsx"]
CORPUS_PATH = next((BASE / n for n in _CORPUS_CANDIDATES if (BASE / n).exists()), BASE / _CORPUS_CANDIDATES[-1])

_QRELS_CANDIDATES = ["qrels_AFF.csv"]
QUERIES_PATH = BASE / "queries.json"          # daftar query + information need (untuk qrels format q_id,doc_id,relevance)
QRELS_TEST_PATH = BASE / "qrels_test.csv"     # opsional: data uji terpisah (tidak dipakai untuk tuning)
QRELS_PATH = next((BASE / n for n in _QRELS_CANDIDATES if (BASE / n).exists()), BASE / _QRELS_CANDIDATES[-1])

# Nama kolom di spreadsheet. None = ditebak otomatis dari nama kolom.
TITLE_COL = "title"     
TEXT_COL = "text_clean"   
LINK_COL = "url"      
DATE_COL = "date"      
SOURCE_COL = "media"    

TITLE_BOOST = 2          # judul dihitung 2x lebih penting dari isi

# Parameter pencarian. Ditentukan SEBELUM UI dijalankan:
#   python tools.py tune  -> menguji kombinasi terhadap qrels dan menulis tuned_params.json
# Kalau tuned_params.json belum ada, dipakai nilai bawaan di bawah ini.
TUNED_PATH = BASE / "tuned_params.json"

_DEFAULTS = dict(
    use_wordnet=True, use_domain=True,
    w_wordnet=0.5, w_domain=0.6, max_per_term=3, use_hypernyms=False,
)
if TUNED_PATH.exists():
    _valid = {f.name for f in fields(SearchConfig)}
    _DEFAULTS.update({k: v for k, v in json.loads(TUNED_PATH.read_text(encoding="utf-8")).get("params", {}).items() if k in _valid})

# Hanya dua pilihan: Baseline (SearchConfig()) atau Query Expansion = WordNet + Domain (SEARCH).
_DEFAULTS.update(use_wordnet=True, use_domain=True)

# Kalau WordNet belum terpasang di laptop, strategi itu otomatis dilewati.
SEARCH = SearchConfig(**_DEFAULTS)


# ------------------------------------------------------------------ korpus
def guess(cols, keys, required=True):
    for k in keys:
        for c in cols:
            if k in str(c).lower():
                return c
    return cols[0] if required else None


def read_corpus(path=None) -> pd.DataFrame:
    path = str(path or CORPUS_PATH)
    return pd.read_csv(path) if path.endswith(".csv") else pd.read_excel(path)


def detect_columns(df: pd.DataFrame) -> dict:
    cols = list(df.columns)
    return {
        "title": TITLE_COL or guess(cols, ["judul", "title", "headline"]),
        "text": TEXT_COL or guess(cols, ["isi", "konten", "content", "teks", "text", "berita"]),
        "link": LINK_COL or guess(cols, ["link", "url", "tautan"], required=False),
        "date": DATE_COL or guess(cols, ["tanggal", "date", "waktu"], required=False),
        "source": SOURCE_COL or guess(cols, ["media", "sumber", "source"], required=False),
    }


def doc_ids(df: pd.DataFrame) -> list[str]:
    """ID dokumen per baris: kolom 'doc_id' kalau ada, kalau tidak D001, D002, ... sesuai urutan baris."""
    for c in df.columns:
        if str(c).strip().lower() in ("doc_id", "docid", "id_dokumen", "id"):
            return [str(x).strip() for x in df[c]]
    return [f"D{i + 1:03d}" for i in range(len(df))]
