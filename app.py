import html
import os
import re
from datetime import datetime

import pandas as pd
import streamlit as st

import config as C
import dev_panel
from ir.engine import SearchConfig, SearchEngine
from ir.preprocess import STOPWORDS, stem_word

st.set_page_config(page_title=C.APP_TITLE, page_icon="📰", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {max-width: 820px; padding-top: 2rem;}
.top h2 {margin: 0; font-size: 1.5rem;}
.top p {margin: .15rem 0 0 0; opacity: .6; font-size: .9rem;}
.expbox {border: 1px solid rgba(59,130,246,.4); background: rgba(59,130,246,.12); border-radius: 10px;
         padding: .7rem .9rem; margin: .8rem 0 1rem 0;}
.expbox .t {color: #60a5fa; font-size: .85rem; font-weight: 600; margin-bottom: .45rem;}
.chip {display: inline-block; border-radius: 999px; padding: .15rem .7rem; margin: 0 .3rem .3rem 0; font-size: .85rem;}
.chip.o {border: 1px solid rgba(128,128,128,.5); background: rgba(128,128,128,.15);}
.chip.e {background: #93c5fd; color: #0b2a4a; font-weight: 500;}
.rtop {display: flex; justify-content: space-between; align-items: flex-start; gap: .8rem;}
.rtitle {font-weight: 700; font-size: 1.05rem; line-height: 1.35;}
.rtitle a {color: inherit; text-decoration: none;}
.rtitle a:hover {text-decoration: underline;}
.badge {flex: none; border-radius: 8px; padding: .1rem .6rem; font-size: .85rem; font-weight: 600;}
.badge.g {background: rgba(34,197,94,.15); color: #22c55e;}
.badge.o {background: rgba(249,115,22,.18); color: #f97316;}
.badge.x {background: rgba(128,128,128,.18); color: #9ca3af;}
.snip {margin: .35rem 0 .45rem 0; opacity: .9;}
.small {font-size: .8rem; opacity: .6;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ data (semuanya di belakang layar)
@st.cache_data(show_spinner=False)
def load_corpus(path: str, mtime: float):
    return C.read_corpus(path)


@st.cache_resource(show_spinner="Menyiapkan mesin pencari...")
def get_engine(path: str, mtime: float, title_col: str, text_col: str, boost: int):
    return SearchEngine(load_corpus(path, mtime), title_col, text_col, boost)


path = str(C.CORPUS_PATH)
if not os.path.exists(path):
    st.error(f"File korpus tidak ditemukan: {path}")
    st.stop()
mtime = os.path.getmtime(path)
_c = C.detect_columns(load_corpus(path, mtime))
TITLE, TEXT, LINK, DATE, SOURCE = _c["title"], _c["text"], _c["link"], _c["date"], _c["source"]

engine = get_engine(path, mtime, TITLE, TEXT, C.TITLE_BOOST)

# Mode developer tersembunyi di sidebar (statistik, evaluasi, detail teknis)
dev = st.sidebar.toggle("Nerd Stats", False, help="Tampilkan evaluasi, analisis query, dan statistik korpus.")


# ------------------------------------------------------------------ helper tampilan
def snippet(text, stems, n=2, max_chars=240):
    sents = re.split(r"(?<=[.!?])\s+", str(text))
    score = [sum(stem_word(w.lower()) in stems for w in re.findall(r"[A-Za-z0-9]+", s)) for s in sents]
    i = max(range(len(sents)), key=score.__getitem__) if sents else 0
    out = " ".join(sents[i:i + n])
    return out if len(out) <= max_chars else out[:max_chars].rsplit(" ", 1)[0] + "…"


def fmt_date(v):
    """Tanggal di korpus campur: teks dd-mm-yyyy / dd/mm/yyyy dan tanggal Excel."""
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return None
    if isinstance(v, str):
        for f in ("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(v.strip(), f).strftime("%d %b %Y")
            except ValueError:
                pass
        return v
    try:
        return pd.Timestamp(v).strftime("%d %b %Y")
    except Exception:
        return str(v)


def badge(score, top):
    """Label relatif terhadap hasil teratas (cosine artikel panjang memang kecil). Cosine asli ada di tooltip."""
    rel = score / top if top else 0
    cls, label = ("g", "Sangat cocok") if rel >= 0.75 else ("o", "Cocok") if rel >= 0.5 else ("x", "Cukup")
    return f'<span class="badge {cls}" title="cosine = {score:.3f}">{label}</span>'


def matched_words(idx, qa, n=6):
    """Kata query yang cocok dengan dokumen: (asli, ekspansi), memakai bentuk kata sebelum di-stem."""
    surface = {}
    for t in qa.terms:
        surface.setdefault(t.term, (t.surface or t.term, t.source))
    orig, exp = [], []
    for term, _, src in engine.explain(idx, qa, top=n):
        word, source = surface.get(term, (term, src))
        (orig if source == "asli" else exp).append(word)
    return orig, exp


# ------------------------------------------------------------------ halaman utama
st.markdown(f'<div class="top"><h2>📰 {html.escape(C.APP_TITLE)}</h2>'
            f'<p>Korpus {len(engine.df)} artikel · TF-IDF + Cosine Similarity</p></div>', unsafe_allow_html=True)
st.divider()

c_q, c_b = st.columns([6, 1], vertical_alignment="bottom")
query = c_q.text_input("Cari", placeholder="mis. perjalanan Vietnam menjadi juara",
                       label_visibility="collapsed", key="q").strip()
c_b.button("🔍 Cari", width="stretch")

o1, o2, o3 = st.columns([2.2, 1.1, 3], vertical_alignment="center")
use_qe = o1.checkbox("Query Expansion", value=True)
o2.markdown("Tampilkan:")
labels = [f"Top {n}" for n in C.TOP_N_OPTIONS]
if hasattr(st, "segmented_control"):
    pick = o3.segmented_control("Tampilkan", labels, default=f"Top {C.TOP_N_DEFAULT}", label_visibility="collapsed")
else:
    pick = o3.radio("Tampilkan", labels, horizontal=True, label_visibility="collapsed")
top_n = int((pick or f"Top {C.TOP_N_DEFAULT}").split()[1])

cfg = C.SEARCH if use_qe else SearchConfig()   # dua pilihan saja: baseline, atau WordNet + Domain

if query:
    res, qa, _ = engine.search(query, cfg, top_n=top_n)
    stems = {t.term for t in qa.terms}

    expansions = []
    for t in qa.terms:
        w = t.surface or t.term
        if t.source != "asli" and w not in expansions:
            expansions.append(w)
    if use_qe and expansions:
        chips = "".join(f'<span class="chip o">{html.escape(t.surface or t.term)}</span>'
                        for t in qa.terms if t.source == "asli" and t.term not in STOPWORDS)
        chips += "".join(f'<span class="chip e">+ {html.escape(w)}</span>' for w in expansions[:10])
        st.markdown(f'<div class="expbox"><div class="t">✦ Query setelah diperluas</div>{chips}</div>',
                    unsafe_allow_html=True)

    if res.empty:
        st.info("Tidak ada artikel yang cocok. Coba kata kunci lain yang lebih umum.")
    else:
        st.caption(f"{len(res)} hasil, diurutkan dari skor tertinggi")
        top_score = float(res["skor"].iloc[0])
        for _, r in res.iterrows():
            title = html.escape(str(r[TITLE]))
            if LINK and isinstance(r[LINK], str) and r[LINK].startswith("http"):
                title = f'<a href="{html.escape(r[LINK])}" target="_blank">{title}</a>'
            orig, exp = matched_words(r["_idx"], qa)
            words = ", ".join(orig)
            if exp:
                words += (" • " if orig else "") + ", ".join(exp) + " (ekspansi)"
            meta = " · ".join(m for m in [str(r[SOURCE]) if SOURCE and pd.notna(r[SOURCE]) else None,
                                          fmt_date(r[DATE]) if DATE else None] if m)
            small = " · ".join(x for x in [f"No. {int(r['_idx']) + 1}", html.escape(meta) if meta else "", f"Kata cocok: {html.escape(words)}" if words else "",
                                          f"cosine = {float(r['skor']):.3f}".replace(".", ",") if dev else ""] if x)
            with st.container(border=True):
                st.markdown(
                    f'<div class="rtop"><div class="rtitle">{title}</div>{badge(float(r["skor"]), top_score)}</div>'
                    f'<div class="snip">{html.escape(snippet(r[TEXT], stems))}</div>'
                    f'<div class="small">{small}</div>', unsafe_allow_html=True)

if dev:
    dev_panel.render(engine, cfg, query)