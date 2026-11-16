from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from .expansion import (ExpansionTerm, get_wordnet, load_domain_groups, wordnet_terms)
from .preprocess import STOPWORDS, content_tokens, preprocess_tokens, stem_word


@dataclass
class SearchConfig:
    use_wordnet: bool = False
    use_domain: bool = False
    w_wordnet: float = 0.5
    w_domain: float = 0.6
    max_per_term: int = 3       # maks. tambahan per kata query per sumber
    use_hypernyms: bool = False

    def label(self) -> str:
        parts = [n for n, f in (("WordNet", self.use_wordnet), ("Domain", self.use_domain)) if f]
        return "Baseline (tanpa QE)" if not parts else "QE: " + " + ".join(parts)


@dataclass
class QueryAnalysis:
    raw_tokens: list[str]
    stems: list[str]
    terms: list[ExpansionTerm]
    weights: dict[str, float]
    wordnet_available: bool
    skipped_oov: int = 0  # istilah ekspansi yang tidak ada di korpus (dibuang)


class SearchEngine:
    def __init__(self, df: pd.DataFrame, title_col: str, text_col: str, title_boost: int = 2):
        self.df = df.reset_index(drop=True)
        self.title_col, self.text_col = title_col, text_col
        titles = self.df[title_col].fillna("").astype(str)
        bodies = self.df[text_col].fillna("").astype(str)
        docs = (titles + " ") * title_boost + bodies
        self.doc_tokens = [preprocess_tokens(d) for d in docs]

        # analyzer=callable: input sudah berupa list token hasil Sastrawi
        self.vec = TfidfVectorizer(analyzer=lambda toks: toks, sublinear_tf=True)
        self.matrix = self.vec.fit_transform(self.doc_tokens).tocsr()
        self.vocab: dict[str, int] = self.vec.vocabulary_
        self.inv_vocab = {j: t for t, j in self.vocab.items()}
        self.df_counts = np.asarray((self.matrix > 0).sum(axis=0)).ravel()
        self.domain_index = load_domain_groups()

    # ------------------------------------------------------------ vektor query
    def _qvec(self, weights: dict[str, float]) -> csr_matrix:
        idx, val = [], []
        for t, w in weights.items():
            j = self.vocab.get(t)
            if j is not None:
                idx.append(j)
                val.append(w * self.vec.idf_[j])
        q = csr_matrix((val, ([0] * len(idx), idx)), shape=(1, len(self.vocab)))
        return normalize(q)

    def _scores(self, weights: dict[str, float]) -> np.ndarray:
        return cosine_similarity(self._qvec(weights), self.matrix).ravel()

    # ------------------------------------------------------------ expansion
    def analyze_query(self, query: str, cfg: SearchConfig) -> QueryAnalysis:
        raw = content_tokens(query)
        stems = [stem_word(t) for t in raw]
        terms: list[ExpansionTerm] = []
        seen_orig = set()
        for r, s in zip(raw, stems):
            if s not in seen_orig:
                seen_orig.add(s)
                terms.append(ExpansionTerm(s, "asli", 1.0, s, r))

        skipped = 0

        def push(stem, source, weight, origin, surface):
            nonlocal skipped
            if not stem or stem in STOPWORDS or stem in seen_orig:
                return
            if stem not in self.vocab:       # tidak ada di korpus -> percuma
                skipped += 1
                return
            if any(t.term == stem and t.source == source for t in terms):
                return
            terms.append(ExpansionTerm(stem, source, weight, origin, surface))

        wn_ok = get_wordnet() is not None
        for r, s in zip(raw, stems):
            if cfg.use_domain:
                for surf in self.domain_index.get(s, [])[: cfg.max_per_term * 2]:
                    push(stem_word(surf), "domain", cfg.w_domain, s, surf)
            if cfg.use_wordnet and wn_ok:
                added = 0
                for surf in wordnet_terms(r, cfg.max_per_term, cfg.use_hypernyms):
                    for tok in surf.split():
                        st = stem_word(tok)
                        before = len(terms)
                        push(st, "wordnet", cfg.w_wordnet, s, surf)
                        added += len(terms) - before
                    if added >= cfg.max_per_term:
                        break

        weights = self._merge(terms)
        return QueryAnalysis(raw, stems, terms, weights, wn_ok, skipped)

    @staticmethod
    def _merge(terms: list[ExpansionTerm]) -> dict[str, float]:
        w: dict[str, float] = {}
        for t in terms:
            w[t.term] = max(w.get(t.term, 0.0), t.weight)
        return w

    # ------------------------------------------------------------ search
    def search(self, query: str, cfg: SearchConfig, top_n: int = 10, min_score: float = 0.0):
        qa = self.analyze_query(query, cfg)
        scores = self._scores(qa.weights)
        order = [i for i in np.argsort(-scores) if scores[i] > max(min_score, 0)][:top_n]
        res = self.df.iloc[order].copy()
        res["skor"] = scores[order]
        res["_idx"] = order
        return res, qa, scores

    def explain(self, doc_idx: int, qa: QueryAnalysis, top: int = 6) -> list[tuple[str, float, str]]:
        """Kontribusi tiap term ke skor cosine dokumen: (term, kontribusi, sumber)."""
        q = self._qvec(qa.weights)
        contrib = q.multiply(self.matrix[doc_idx]).tocoo()
        src = {}
        for t in qa.terms:
            src.setdefault(t.term, t.source)
        items = [(self.inv_vocab[c], float(v), src.get(self.inv_vocab[c], "?"))
                 for c, v in zip(contrib.col, contrib.data)]
        return sorted(items, key=lambda x: -x[1])[:top]
