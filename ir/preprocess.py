import re
from functools import lru_cache

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

_stemmer = StemmerFactory().create_stemmer()
STOPWORDS = set(StopWordRemoverFactory().get_stop_words()) | {
    "jadi", "kata", "mengatakan", "ujar", "tutur", "kompas", "detik", "com", "wib", "foto",
}


@lru_cache(maxsize=None)
def stem_word(word: str) -> str:
    return _stemmer.stem(word)


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def content_tokens(text: str) -> list[str]:
    """Token mentah (belum di-stem) tanpa stopword. Dipakai untuk lookup WordNet."""
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]


def preprocess_tokens(text: str) -> list[str]:
    stems = (stem_word(t) for t in content_tokens(text))
    return [s for s in stems if s and s not in STOPWORDS]
