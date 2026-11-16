"""Evaluasi: Precision@k, Recall@k, AP/MAP. (nDCG sengaja tidak dipakai: qrels bersifat biner.)

Dua format qrels didukung:
  1. Panjang (disarankan, standar TREC):  qrels.csv  = q_id,doc_id,relevance
                                          queries.json = [{"q_id","query","scenario","information_need"}, ...]
  2. Lama (satu baris per query):         qrels.csv  = query,relevan   (relevan = nomor baris 1-based, pisah '|')
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .engine import SearchConfig, SearchEngine

# qrels = list[(teks_query, {indeks_dokumen_0based})]; meta = {teks_query: {"q_id", "scenario"}}


def parse_qrels(df: pd.DataFrame) -> list[tuple[str, set[int]]]:
    """Format lama: kolom query | relevan (nomor dokumen 1-based dipisah '|' atau ',')."""
    out = []
    for _, r in df.iterrows():
        q = str(r["query"]).strip()
        rel = {int(x) - 1 for x in str(r["relevan"]).replace(",", "|").split("|") if x.strip().isdigit()}
        if q and rel:
            out.append((q, rel))
    return out


def load_qrels(path, doc_ids: list[str], queries_path=None):
    """Baca qrels format panjang ATAU lama. Mengembalikan (qrels, meta, masalah).

    doc_ids = daftar ID dokumen sesuai urutan baris korpus (mis. ['D001', ...]).
    masalah = daftar peringatan yang perlu dibaca user (doc_id tidak dikenal, query tanpa qrels, dsb.).
    """
    path = Path(path)
    df = pd.read_csv(path, dtype=str)
    cols = {c.strip().lower() for c in df.columns}
    problems: list[str] = []

    if {"q_id", "doc_id"} <= cols:
        df.columns = [c.strip().lower() for c in df.columns]
        qp = Path(queries_path) if queries_path else path.with_name("queries.json")
        if not qp.exists():
            raise FileNotFoundError(f"Format q_id,doc_id,relevance butuh {qp.name} (daftar query) di folder yang sama.")
        queries = {q["q_id"].strip(): q for q in json.loads(qp.read_text(encoding="utf-8"))}
        pos = {d: i for i, d in enumerate(doc_ids)}
        if "relevance" in df.columns:
            df = df[pd.to_numeric(df["relevance"], errors="coerce").fillna(0) > 0]
        rel_by_q: dict[str, set[int]] = {}
        for qid, did in zip(df["q_id"].str.strip(), df["doc_id"].str.strip()):
            if did not in pos:
                problems.append(f"{qid}: doc_id '{did}' tidak ada di korpus (diabaikan)")
                continue
            rel_by_q.setdefault(qid, set()).add(pos[did])
        qrels, meta = [], {}
        for qid, rel in rel_by_q.items():
            if qid not in queries:
                problems.append(f"{qid}: ada di qrels tapi tidak ada di {qp.name} (diabaikan)")
                continue
            text = queries[qid]["query"].strip()
            qrels.append((text, rel))
            meta[text] = {"q_id": qid, "scenario": queries[qid].get("scenario", "")}
        return qrels, meta, problems

    if {"query", "relevan"} <= cols:
        return parse_qrels(df), {}, problems
    raise ValueError("Kolom qrels tidak dikenali. Pakai 'q_id,doc_id,relevance' atau 'query,relevan'.")


def precision_at_k(ranked, rel, k):
    return sum(1 for d in ranked[:k] if d in rel) / k


def recall_at_k(ranked, rel, k):
    return sum(1 for d in ranked[:k] if d in rel) / len(rel)


def average_precision(ranked, rel):
    hits, s = 0, 0.0
    for i, d in enumerate(ranked, 1):
        if d in rel:
            hits += 1
            s += hits / i
    return s / len(rel)


def evaluate(engine: SearchEngine, qrels, cfg: SearchConfig, k: int = 5, meta: dict | None = None):
    rows = []
    for q, rel in qrels:
        res, _, _ = engine.search(q, cfg, top_n=len(engine.df))
        ranked = list(res["_idx"])
        row = {"query": q}
        if meta and q in meta:
            row = {"id": meta[q]["q_id"], "kelompok": meta[q]["scenario"], **row}
        row.update({f"P@{k}": precision_at_k(ranked, rel, k), f"R@{k}": recall_at_k(ranked, rel, k),
                    "AP": average_precision(ranked, rel), "jml relevan": len(rel)})
        rows.append(row)
    return pd.DataFrame(rows)


_NON_METRIC = {"query", "id", "kelompok", "jml relevan"}


def mean_metrics(df: pd.DataFrame) -> pd.Series:
    return df.drop(columns=[c for c in _NON_METRIC if c in df.columns]).mean()


def compare_modes(engine, qrels, configs: dict[str, SearchConfig], k: int = 5) -> pd.DataFrame:
    out = []
    for name, cfg in configs.items():
        m = mean_metrics(evaluate(engine, qrels, cfg, k))
        out.append({"Mode": name, **m.to_dict(), "MAP": m["AP"]})
    return pd.DataFrame(out).drop(columns="AP").set_index("Mode")