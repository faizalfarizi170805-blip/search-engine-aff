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


def wordnet_terms(raw_token: str, max_terms: int = 4, use_hypernyms: bool = False) -> list[str]:
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
