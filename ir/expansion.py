"""Query Expansion.

Sumber istilah tambahan:
  1. WordNet  : sinonim (synset) Bahasa Indonesia via Open Multilingual WordNet (lang='ind')
  2. Domain   : kamus AFF buatan sendiri (data/KamusAFFSendiri.json)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .preprocess import STOPWORDS, stem_word, tokenize

DOMAIN_PATH = Path(__file__).resolve().parent.parent / "data" / "KamusAFFSendiri.json"
WN_TABLE_PATH = DOMAIN_PATH.parent / "wordnet_ind.json"   # tabel sinonim hasil `python tools.py warmup`


@dataclass
class ExpansionTerm:
    term: str      # bentuk stem (yang masuk ke vektor)
    source: str    # asli | wordnet | domain
    weight: float
    origin: str    # kata query asal
    surface: str = ""  # bentuk sebelum di-stem (untuk ditampilkan)


# ---------------------------------------------------------------- WordNet
_WN = None
_WN_TRIED = False


def get_wordnet():
    """Return objek wordnet NLTK, atau None kalau datanya belum diunduh."""
    global _WN, _WN_TRIED
    if _WN_TRIED:
        return _WN
    _WN_TRIED = True
    try:
        from nltk.corpus import wordnet as wn
        wn.synsets("makan", lang="ind")  # memicu load; LookupError bila data belum ada
        _WN = wn
    except Exception:
        _WN = None
    return _WN


# Tabel sinonim ringan (kata -> daftar sinonim). Dibaca dari JSON dalam hitungan milidetik,
# jadi NLTK + WordNet (import & load-nya 10-60 dtk) tidak perlu dimuat saat aplikasi jalan.
_WN_TABLE = None
_WN_TABLE_TRIED = False


def _wn_table():
    global _WN_TABLE, _WN_TABLE_TRIED
    if not _WN_TABLE_TRIED:
        _WN_TABLE_TRIED = True
        try:
            _WN_TABLE = json.loads(WN_TABLE_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _WN_TABLE = None
    return _WN_TABLE


def wordnet_ready() -> bool:
    """True kalau sinonim bisa dipakai: tabel JSON ada (cepat) atau NLTK WordNet terpasang (lambat)."""
    return _wn_table() is not None or get_wordnet() is not None


def wordnet_terms(raw_token: str, max_terms: int = 4, use_hypernyms: bool = False) -> list[str]:
    table = _wn_table()
    if table is not None and not use_hypernyms:
        return table.get(raw_token, [])[: max_terms * 3]
    return _wordnet_terms_nltk(raw_token, max_terms, use_hypernyms)


def build_wordnet_table(path: Path = WN_TABLE_PATH, per_key: int = 10) -> int:
    """Sekali jalan (lewat tools.py warmup): ekspor semua sinonim Indonesia WordNet ke JSON."""
    wn = get_wordnet()
    if wn is None:
        raise RuntimeError("WordNet belum terpasang. Jalankan dulu: python tools.py setup")
    keys = set()
    for syn in wn.all_synsets():
        for lem in syn.lemma_names("ind"):
            lem = lem.lower().strip()
            if lem and "_" not in lem:
                keys.add(lem)
    table = {}
    for k in sorted(keys):
        terms = _wordnet_terms_nltk(k, per_key, False)
        if terms:
            table[k] = terms
    path.write_text(json.dumps(table, ensure_ascii=False), encoding="utf-8")
    return len(table)


def _wordnet_terms_nltk(raw_token: str, max_terms: int = 4, use_hypernyms: bool = False) -> list[str]:
    """Ambil lemma Indonesia dari semua synset yang memuat kata tsb (urut synset = urut sense)."""
    wn = get_wordnet()
    if wn is None:
        return []
    out: list[str] = []

    def add(lemmas):
        for lem in lemmas:
            lem = lem.replace("_", " ").lower().strip()
            if not lem or lem == raw_token or lem in out:
                continue
            out.append(lem)

    for syn in wn.synsets(raw_token, lang="ind"):
        add(syn.lemma_names("ind"))
        if use_hypernyms:
            for hyp in syn.hypernyms():
                add(hyp.lemma_names("ind"))
        if len(out) >= max_terms * 3:  # cukup kandidat, nanti dipotong setelah filter vocab
            break
    return out


# ---------------------------------------------------------------- Kamus Sendiri (AFF)
def load_domain_groups(path: Path = DOMAIN_PATH) -> dict[str, list[str]]:
    """Return mapping: stem -> daftar istilah terkait (surface form)."""
    if not path.exists():
        return {}
    groups = json.loads(path.read_text(encoding="utf-8")).get("groups", [])
    index: dict[str, list[str]] = {}
    for g in groups:
        stems = {}
        for phrase in g:
            for tok in tokenize(phrase):
                if tok in STOPWORDS:
                    continue
                stems[stem_word(tok)] = tok
        for s in stems:
            index.setdefault(s, [])
            for other_stem, other_surface in stems.items():
                if other_stem != s and other_surface not in index[s]:
                    index[s].append(other_surface)
    return index
