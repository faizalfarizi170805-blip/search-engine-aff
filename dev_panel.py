"""Panel 'nerd stats' - hanya tampil kalau Mode developer dinyalakan."""
import numpy as np
import pandas as pd
import streamlit as st

import config as C
from ir.engine import SearchConfig
from ir.evaluation import compare_modes, evaluate, load_qrels
from ir.expansion import get_wordnet

SRC_BADGE = {"asli": "🟦 asli", "wordnet": "🟩 WordNet", "domain": "🟧 kamus AFF (Sendiri)"}


def render(engine, cfg: SearchConfig, query: str):
    st.divider()
    st.subheader("🛠️ Detail teknis")
    t_query, t_eval, t_corpus = st.tabs(["Analisis Query", "Evaluasi", "Korpus"])

    with t_query:
        if not query.strip():
            st.caption("Ketik query di kolom pencarian untuk melihat detail ekspansinya.")
        else:
            qa = engine.analyze_query(query, cfg)
            c1, c2, c3 = st.columns(3)
            c1.metric("Token query (stem)", len(set(qa.stems)))
            c2.metric("Istilah ekspansi", len([t for t in qa.terms if t.source != "asli"]))
            c3.metric("Dibuang (tidak ada di korpus)", qa.skipped_oov)
            if cfg.use_wordnet and not qa.wordnet_available:
                st.warning("WordNet tidak termuat.")
            st.dataframe(pd.DataFrame([{"istilah": t.surface or t.term, "stem": t.term, "sumber": SRC_BADGE[t.source],
                                        "asal kata": t.origin, "bobot": round(t.weight, 3)} for t in qa.terms]),
                         hide_index=True, width="stretch")
            st.caption("Query akhir = vektor berbobot (asli 1.0, ekspansi lebih kecil) × IDF, dinormalisasi L2, "
                       "lalu dibandingkan dengan cosine similarity.")

    with t_eval:
        K = 5
        if not C.QRELS_PATH.exists():
            st.caption(f"Belum ada {C.QRELS_PATH.name} di folder project.")
        else:
            ids = C.doc_ids(engine.df)
            try:
                qrels, meta, problems = load_qrels(C.QRELS_PATH, ids, C.QUERIES_PATH)
            except (ValueError, FileNotFoundError) as e:
                qrels, meta, problems = [], {}, [str(e)]
            for p_ in problems:
                st.warning(p_)
            if not qrels:
                st.error("qrels kosong/tidak valid.")
            else:
                wn_ok = get_wordnet() is not None
                modes = {
                    "Baseline": SearchConfig(),
                    "+ Domain saja": SearchConfig(use_domain=True, w_domain=cfg.w_domain, max_per_term=cfg.max_per_term),
                }
                if wn_ok:
                    modes["+ WordNet saja"] = SearchConfig(use_wordnet=True, w_wordnet=cfg.w_wordnet,
                                                           max_per_term=cfg.max_per_term, use_hypernyms=cfg.use_hypernyms)
                modes["Query Expansion (WordNet + Domain)"] = cfg
                st.markdown("**Data tuning** (dipakai memilih parameter; cenderung optimis)")
                comp = compare_modes(engine, qrels, modes, K)
                st.caption(f"{len(qrels)} query, k = {K}")
                st.dataframe(comp.round(3), width="stretch")
                st.bar_chart(comp[["MAP"]])

                if C.QRELS_TEST_PATH.exists():
                    tq, tmeta, tprob = load_qrels(C.QRELS_TEST_PATH, ids, C.QUERIES_PATH)
                    for p_ in tprob:
                        st.warning(p_)
                    if tq:
                        st.markdown("**Data uji terpisah** (tidak dipakai memilih parameter; ini angka yang jujur)")
                        tcomp = compare_modes(engine, tq, modes, K)
                        st.caption(f"{len(tq)} query, k = {K}")
                        st.dataframe(tcomp.round(3), width="stretch")
                else:
                    st.caption("Opsional: qrels_test.csv (query terpisah) memberi hasil yang tidak ikut dituning.")

                st.subheader("Per query (Query Expansion vs baseline)")
                per = evaluate(engine, qrels, cfg, K, meta)
                base_per = evaluate(engine, qrels, SearchConfig(), K, meta)
                per["AP baseline"] = base_per["AP"]
                per["selisih AP"] = per["AP"] - per["AP baseline"]
                st.dataframe(per.round(3), hide_index=True)
                if "kelompok" in per.columns and per["kelompok"].astype(bool).any():
                    st.subheader("Per kelompok query")
                    st.dataframe(per.groupby("kelompok")[["AP baseline", "AP"]].mean().round(3).rename(
                        columns={"AP": "AP (Query Expansion)"}))

    with t_corpus:
        lens = [len(t) for t in engine.doc_tokens]
        c1, c2, c3 = st.columns(3)
        c1.metric("Jumlah dokumen", len(engine.df))
        c2.metric("Ukuran vocabulary", len(engine.vocab))
        c3.metric("Rata-rata token/dok", f"{np.mean(lens):.0f}")
        order = np.argsort(-engine.df_counts)[:20]
        st.subheader("20 term dengan document frequency tertinggi")
        st.dataframe(pd.DataFrame({"term": [engine.inv_vocab[j] for j in order], "df": engine.df_counts[order]}),
                     hide_index=True)
        st.subheader("Data")
        st.dataframe(engine.df.astype(str), width="stretch")