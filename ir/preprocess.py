import json
import re
import threading
from pathlib import Path

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

STOPWORDS = set(StopWordRemoverFactory().get_stop_words()) | {
    "jadi", "kata", "mengatakan", "ujar", "tutur", "kompas", "detik", "com", "wib", "foto",
}

# ---------------------------------------------------------------- stemming + cache di disk
# Sastrawi itu lambat: membuat stemmer ~1-3 dtk, dan tiap kata unik ~ms. Dulu semua artikel di-stem
# ulang SETIAP aplikasi dijalankan. Sekarang hasil stem disimpan di data/stem_cache.json,
# jadi run berikutnya cukup baca file (milidetik) dan Sastrawi tidak perlu dimuat sama sekali.
CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "stem_cache.json"

_stemmer = None
_lock = threading.Lock()
_cache: dict[str, str] = {}
_dirty = False


def _get_stemmer():
    """Stemmer dibuat malas (lazy): hanya kalau ada kata yang belum ada di cache."""
    global _stemmer
    if _stemmer is None:
        with _lock:
            if _stemmer is None:
                from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
                _stemmer = StemmerFactory().create_stemmer()
    return _stemmer


def _load_cache():
    try:
        _cache.update(json.loads(CACHE_PATH.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        pass


_load_cache()


def stem_word(word: str) -> str:
    global _dirty
    s = _cache.get(word)
    if s is None:
        s = _get_stemmer().stem(word)
        _cache[word] = s
        _dirty = True
    return s


def save_stem_cache() -> None:
    """Tulis cache ke disk kalau ada kata baru. Aman kalau folder read-only (diabaikan)."""
    global _dirty
    if not _dirty:
        return
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = CACHE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(_cache, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        tmp.replace(CACHE_PATH)
        _dirty = False
    except OSError:
        pass


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def content_tokens(text: str) -> list[str]:
    """Token mentah (belum di-stem) tanpa stopword. Dipakai untuk lookup WordNet."""
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]


def preprocess_tokens(text: str) -> list[str]:
    stems = (stem_word(t) for t in content_tokens(text))
    return [s for s in stems if s and s not in STOPWORDS]
