"""Alat bantu di luar UI (dijalankan dari terminal). Dua perintah:

    python tools.py setup               # sekali saja: unduh WordNet + Open Multilingual WordNet (Bahasa Indonesia)
    python tools.py tune [--dry-run]    # cari parameter Query Expansion terbaik, simpan ke tuned_params.json
                         [--qrels file.csv] [--k 5] [--out file.json]

tune: tiap kombinasi (bobot + jumlah istilah tambahan) dievaluasi dengan MAP pada qrels. Kalau MAP sama,
dipilih yang lebih sederhana. Kalau ada qrels_test.csv, hasil akhir juga diuji di sana
(query yang TIDAK dipakai memilih parameter).
"""
import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import config as C


# ------------------------------------------------------------------ setup
def cmd_setup(_args):
    import nltk

    for pkg in ("wordnet", "omw-1.4"):
        print(pkg, "->", nltk.download(pkg))

    from nltk.corpus import wordnet as wn
    for w in ("menang", "juara", "pemain"):
        print(w, [s.lemma_names("ind") for s in wn.synsets(w, lang="ind")[:3]])


# ------------------------------------------------------------------ tune
WEIGHTS = (0.3, 0.5, 0.7)
MAX_PER_TERM = (2, 3, 5)


def candidates(wn_ok: bool):
    """Hanya dua pilihan: baseline, atau Query Expansion = WordNet + Domain (yang dicari: bobot & jumlah istilah)."""
    from ir.engine import SearchConfig

    yield (), SearchConfig()
    names = ("wordnet", "domain") if wn_ok else ("domain",)
    for ww in (WEIGHTS if wn_ok else (0.5,)):
        for wd in WEIGHTS:
            for mpt in MAX_PER_TERM:
                yield names, SearchConfig(use_wordnet=wn_ok, use_domain=True, w_wordnet=ww, w_domain=wd, max_per_term=mpt)


def cmd_tune(args):
    import pandas as pd

    from ir.engine import SearchConfig, SearchEngine
    from ir.evaluation import evaluate, load_qrels, mean_metrics
    from ir.expansion import get_wordnet

    df = C.read_corpus()
    ids = C.doc_ids(df)
    qrels, meta, problems = load_qrels(args.qrels, ids, C.QUERIES_PATH)
    for p_ in problems:
        print("PERINGATAN:", p_)
    if not qrels:
        sys.exit("Qrels kosong/tidak valid. Format: q_id,doc_id,relevance (+ queries.json) atau query,relevan.")
    if len(qrels) < 8:
        print(f"PERINGATAN: hanya {len(qrels)} query uji. Hasil tuning mudah overfit; disarankan minimal 8-10 query.\n")

    cols = C.detect_columns(df)
    engine = SearchEngine(df, cols["title"], cols["text"], C.TITLE_BOOST)
    wn_ok = get_wordnet() is not None
    print(f"Korpus: {len(df)} artikel | qrels: {len(qrels)} query | WordNet: {'ya' if wn_ok else 'TIDAK (dilewati)'}")

    rows, t0 = [], time.time()
    combos = list(candidates(wn_ok))
    print(f"Menguji {len(combos)} konfigurasi (baseline + WordNet+Domain)...")
    for i, (subset, cfg) in enumerate(combos, 1):
        m = mean_metrics(evaluate(engine, qrels, cfg, args.k, meta))
        rows.append({"strategi": "+".join(subset) or "baseline", "n_strategi": len(subset), "w_wn": cfg.w_wordnet if cfg.use_wordnet else None,
                     "w_dom": cfg.w_domain if cfg.use_domain else None, "maks": cfg.max_per_term, "MAP": m["AP"],
                     f"P@{args.k}": m[f"P@{args.k}"], f"R@{args.k}": m[f"R@{args.k}"], "cfg": cfg})
        if i % 100 == 0:
            print(f"  {i}/{len(combos)}  ({time.time()-t0:.0f}s)")
    res = pd.DataFrame(rows)
    res["_r"] = res["MAP"].round(6)
    res = res.sort_values(["_r", f"P@{args.k}", "n_strategi"], ascending=[False, False, True]).reset_index(drop=True)

    base = res[res["strategi"] == "baseline"].iloc[0]
    best = res[res["n_strategi"] >= 1].iloc[0]   # QE adalah fitur andalan: pilih terbaik di antara yang memakai QE
    print("\n10 terbaik:")
    print(res.drop(columns=["cfg", "_r", "n_strategi"]).head(10).round(3).to_string(index=False))
    print(f"\nBaseline (tanpa QE): MAP {base['MAP']:.3f}")
    print(f"Terbaik dengan QE  : MAP {best['MAP']:.3f}  ({best['strategi']})")
    gain = best["MAP"] - base["MAP"]
    if gain < 0.01:
        print("CATATAN: QE tidak meningkatkan MAP secara berarti (< 0.01) dibanding baseline pada qrels ini. Itu tetap hasil yang sah untuk dilaporkan dan dibahas di laporan.")

    test_info = {}
    if C.QRELS_TEST_PATH.exists() and C.QRELS_TEST_PATH != Path(args.qrels):
        tq, tmeta, tprob = load_qrels(C.QRELS_TEST_PATH, ids, C.QUERIES_PATH)
        if tq:
            mb = mean_metrics(evaluate(engine, tq, SearchConfig(), args.k, tmeta))
            mt = mean_metrics(evaluate(engine, tq, best["cfg"], args.k, tmeta))
            print(f"\nDATA UJI TERPISAH ({len(tq)} query, tidak dipakai untuk memilih parameter):")
            print(f"  Baseline : MAP {mb['AP']:.3f}  P@{args.k} {mb[f'P@{args.k}']:.3f}  R@{args.k} {mb[f'R@{args.k}']:.3f}")
            print(f"  Terbaik  : MAP {mt['AP']:.3f}  P@{args.k} {mt[f'P@{args.k}']:.3f}  R@{args.k} {mt[f'R@{args.k}']:.3f}")
            print("  (Angka inilah yang jujur dilaporkan; angka tuning di atas cenderung terlalu optimis.)")
            test_info = {"MAP_test": round(float(mt["AP"]), 4), "MAP_test_baseline": round(float(mb["AP"]), 4), "qrels_test": len(tq)}

    params = {k: v for k, v in asdict(best["cfg"]).items()}
    if args.dry_run:
        print("\n(dry-run) tidak menyimpan. Parameter terbaik:", json.dumps(params))
        return
    out = {"params": params, "qrels": len(qrels), "k": args.k, "MAP": round(float(best["MAP"]), 4),
           "MAP_baseline": round(float(base["MAP"]), 4), **test_info}
    open(args.out, "w", encoding="utf-8").write(json.dumps(out, indent=2))
    print(f"\nDisimpan ke {args.out}. Jalankan UI: streamlit run app.py")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Alat bantu AFF search: setup, tune.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("setup", help="unduh WordNet + OMW (sekali saja)").set_defaults(func=cmd_setup)

    t = sub.add_parser("tune", help="cari parameter QE terbaik -> tuned_params.json")
    t.add_argument("--qrels", default=str(C.QRELS_PATH))
    t.add_argument("--k", type=int, default=5)
    t.add_argument("--dry-run", action="store_true")
    t.add_argument("--out", default=str(C.TUNED_PATH))
    t.set_defaults(func=cmd_tune)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
