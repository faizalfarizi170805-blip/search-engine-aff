import base64
import html
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

import config as C
import dev_panel
from ir.engine import SearchConfig, SearchEngine
from ir.preprocess import STOPWORDS, stem_word

# ── background image ───────────────────────────────────────────────────────
_IMG_PATH = Path(__file__).parent / "assets" / "players.png"
_IMG_B64 = ""
if _IMG_PATH.exists():
    with open(_IMG_PATH, "rb") as _fh:
        _IMG_B64 = base64.b64encode(_fh.read()).decode()

st.set_page_config(page_title=C.APP_TITLE, page_icon="⚽", layout="centered", initial_sidebar_state="collapsed")

# ── Background image CSS (f-string to embed base64) ──────────────────────
if _IMG_B64:
    st.markdown(f"""<style>
    [data-testid="stAppViewContainer"]::after {{
        content: "";
        position: fixed;
        right: -20px;
        bottom: -10px;
        width: 530px;
        height: 630px;
        background: url("data:image/png;base64,{_IMG_B64}") no-repeat right bottom / contain;
        opacity: 0.09;
        pointer-events: none;
        z-index: 0;
        animation: floatImg 8s ease-in-out infinite;
    }}
    @keyframes floatImg {{
        0%, 100% {{ transform: translateY(0px) rotate(0deg); }}
        50%       {{ transform: translateY(-22px) rotate(1.5deg); }}
    }}
    </style>""", unsafe_allow_html=True)

# ── Main CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── RESET ── */
#MainMenu, footer {visibility: hidden;}

/* ── BACKGROUND: bright white ── */
[data-testid="stAppViewContainer"] {
    background: #ffffff;
    min-height: 100vh;
}

/* header bar: frosted white — keeps sidebar toggle visible */
[data-testid="stHeader"] {
    background: rgba(255,255,255,.92) !important;
    backdrop-filter: blur(14px);
    border-bottom: 1.5px solid #fee2e2;
}

/* sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #fff5f5 0%, #fff 100%);
    border-right: 1.5px solid #fecaca;
}

.block-container {
    max-width: 860px;
    padding-top: 1rem;
    padding-bottom: 5rem;
    position: relative;
    z-index: 1;
}

/* ── HERO HEADER ── */
.top {
    text-align: center;
    padding: 2.4rem 0 .9rem 0;
}
.top .logo-row {
    font-size: 4rem;
    display: inline-block;
    margin-bottom: .3rem;
    animation: soccerBounce 2.5s ease-in-out infinite;
    filter: drop-shadow(0 4px 14px rgba(220,38,38,.3));
}
@keyframes soccerBounce {
    0%, 100% { transform: translateY(0) rotate(-8deg); }
    40%       { transform: translateY(-18px) rotate(7deg); }
    70%       { transform: translateY(-6px) rotate(-3deg); }
}
.top h2 {
    margin: 0;
    font-size: 2.4rem;
    font-weight: 900;
    letter-spacing: .3px;
    background: linear-gradient(135deg, #dc2626 0%, #f97316 55%, #fbbf24 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
}
.top p {
    margin: .55rem 0 0 0;
    color: #94a3b8;
    font-size: .84rem;
    letter-spacing: .4px;
}

/* ── STATS STRIP ── */
.stats-strip {
    display: flex;
    justify-content: center;
    gap: 2.5rem;
    margin: 1.1rem 0 1.5rem 0;
    padding: .9rem 2.5rem;
    background: linear-gradient(135deg, #fff5f5 0%, #fff7ed 100%);
    border: 1.5px solid #fecaca;
    border-radius: 20px;
    box-shadow: 0 2px 18px rgba(220,38,38,.07);
}
.stat-item { text-align: center; }
.stat-num  { font-size: 1.35rem; font-weight: 900; color: #dc2626; }
.stat-lbl  { font-size: .68rem; color: #94a3b8; text-transform: uppercase; letter-spacing: .7px; margin-top: .1rem; }

/* ── DIVIDER ── */
hr { border-color: #fee2e2 !important; }

/* ── SEARCH WRAP ── */
.search-wrap {
    background: linear-gradient(135deg, #fff5f5 0%, #fffbeb 100%);
    border: 2px solid #fecaca;
    border-radius: 24px;
    padding: 1.3rem 1.5rem 1.1rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 28px rgba(220,38,38,.08);
    transition: box-shadow .3s ease;
}
.search-wrap:hover { box-shadow: 0 10px 40px rgba(220,38,38,.14); }

/* ── SEARCH INPUT ── */
[data-testid="stTextInput"] input {
    background: #fff !important;
    border: 2px solid #fecaca !important;
    border-radius: 999px !important;
    color: #0f172a !important;
    font-size: 1.05rem !important;
    padding: .65rem 1.3rem !important;
    box-shadow: 0 2px 10px rgba(220,38,38,.05) !important;
    transition: all .25s ease !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #dc2626 !important;
    box-shadow: 0 0 0 4px rgba(220,38,38,.1), 0 2px 10px rgba(220,38,38,.08) !important;
}
[data-testid="stTextInput"] input::placeholder { color: #cbd5e1 !important; }

/* ── PRIMARY BUTTON = tombol cari ── */
button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 999px !important;
    font-weight: 800 !important;
    font-size: .95rem !important;
    padding: .65rem 1.3rem !important;
    box-shadow: 0 4px 18px rgba(220,38,38,.4) !important;
    transition: all .2s ease !important;
    letter-spacing: .3px;
}
button[data-testid="stBaseButton-primary"]:hover {
    transform: translateY(-2px) scale(1.04) !important;
    box-shadow: 0 8px 28px rgba(220,38,38,.55) !important;
}

/* ── SECONDARY BUTTONS = expansion chip pills ── */
button[data-testid="stBaseButton-secondary"] {
    background: #fef3c7 !important;
    color: #b45309 !important;
    border: 1.5px solid #fde68a !important;
    border-radius: 999px !important;
    font-weight: 700 !important;
    font-size: .82rem !important;
    padding: .22rem .9rem !important;
    transition: all .2s ease !important;
    white-space: nowrap;
    animation: chipPop .35s cubic-bezier(.34,1.56,.64,1) both;
}
button[data-testid="stBaseButton-secondary"]:hover {
    background: #fde68a !important;
    transform: translateY(-2px) scale(1.07) !important;
    box-shadow: 0 4px 14px rgba(217,119,6,.22) !important;
}
@keyframes chipPop {
    from { opacity: 0; transform: scale(.75) translateY(6px); }
    to   { opacity: 1; transform: scale(1) translateY(0); }
}

/* ── OPTIONS ── */
[data-testid="stCheckbox"] span { color: #475569 !important; font-size: .9rem !important; }

/* ── EXPANSION BOX ── */
.expbox {
    border: 1.5px solid #fde68a;
    background: linear-gradient(135deg, #fffbeb 0%, #fff7ed 100%);
    border-radius: 18px;
    padding: .9rem 1.2rem .5rem 1.2rem;
    margin: .9rem 0 .5rem 0;
    animation: slideDown .4s cubic-bezier(.34,1.56,.64,1) both;
}
.expbox .t {
    color: #d97706;
    font-size: .78rem;
    font-weight: 800;
    margin-bottom: .4rem;
    text-transform: uppercase;
    letter-spacing: .9px;
}
@keyframes slideDown {
    from { opacity: 0; transform: translateY(-14px) scale(.97); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}

/* ── STATIC CHIPS ── */
.chip { display: inline-block; border-radius: 999px; padding: .2rem .8rem; margin: 0 .3rem .3rem 0; font-size: .82rem; }
.chip.o { border: 1.5px solid #e2e8f0; background: #f8fafc; color: #475569; }
.chip.e { background: #fef3c7; color: #b45309; border: 1.5px solid #fde68a; font-weight: 700; }

/* ── RESULT CARDS ── */
[data-testid="stVerticalBlock"] [data-testid="stContainer"] > div {
    background: #fff !important;
    border: 1.5px solid #f1f5f9 !important;
    border-radius: 20px !important;
    box-shadow: 0 2px 20px rgba(0,0,0,.055) !important;
    transition: all .25s ease !important;
    animation: fadeSlideUp .4s ease both;
}
[data-testid="stVerticalBlock"] [data-testid="stContainer"] > div:nth-child(1) { animation-delay: .00s; }
[data-testid="stVerticalBlock"] [data-testid="stContainer"] > div:nth-child(2) { animation-delay: .07s; }
[data-testid="stVerticalBlock"] [data-testid="stContainer"] > div:nth-child(3) { animation-delay: .14s; }
[data-testid="stVerticalBlock"] [data-testid="stContainer"] > div:nth-child(4) { animation-delay: .21s; }
[data-testid="stVerticalBlock"] [data-testid="stContainer"] > div:nth-child(5) { animation-delay: .28s; }
[data-testid="stVerticalBlock"] [data-testid="stContainer"] > div:nth-child(6) { animation-delay: .35s; }
[data-testid="stVerticalBlock"] [data-testid="stContainer"] > div:hover {
    transform: translateY(-5px) !important;
    border-color: #fecaca !important;
    box-shadow: 0 12px 40px rgba(220,38,38,.12) !important;
}
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── RESULT CONTENT ── */
.rtop   { display: flex; justify-content: space-between; align-items: flex-start; gap: .8rem; }
.rtitle { font-weight: 700; font-size: 1.05rem; line-height: 1.45; color: #0f172a; }
.rtitle a { color: #dc2626; text-decoration: none; }
.rtitle a:hover { color: #991b1b; text-decoration: underline; }
.badge { flex: none; border-radius: 999px; padding: .18rem .8rem; font-size: .77rem; font-weight: 800;
         letter-spacing: .3px; text-transform: uppercase; }
.badge.g { background: #dcfce7; color: #16a34a; border: 1.5px solid #bbf7d0; }
.badge.o { background: #fef3c7; color: #d97706; border: 1.5px solid #fde68a; }
.badge.x { background: #f1f5f9; color: #94a3b8; border: 1.5px solid #e2e8f0; }
.snip   { margin: .4rem 0 .5rem 0; color: #64748b; font-size: .91rem; line-height: 1.65; }
.small  { font-size: .76rem; color: #94a3b8; }

/* ── CAPTION / ALERT ── */
[data-testid="stCaptionContainer"] p { color: #94a3b8 !important; font-size: .82rem !important; }
[data-testid="stAlert"] {
    background: #fff5f5 !important;
    border: 1.5px solid #fecaca !important;
    border-radius: 10px !important;
}
[data-testid="stAlert"] p {color: #fca5a5 !important;}
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
st.markdown(f'''
<div class="top">
  <div class="logo-row">⚽</div>
  <h2>{html.escape(C.APP_TITLE)}</h2>
  <p>Mesin Pencari Berita &nbsp;·&nbsp; TF-IDF + Cosine Similarity + Query Expansion</p>
</div>
<div class="stats-strip">
  <div class="stat-item"><div class="stat-num">{len(engine.df)}</div><div class="stat-lbl">Artikel</div></div>
  <div class="stat-item"><div class="stat-num">TF-IDF</div><div class="stat-lbl">Model</div></div>
  <div class="stat-item"><div class="stat-num">WordNet</div><div class="stat-lbl">Ekspansi</div></div>
</div>
''', unsafe_allow_html=True)
st.divider()

# Callback untuk menambahkan kata rekomendasi ke query pencarian saat chip diklik
def append_search_query(term_to_add):
    current = st.session_state.get("q", "").strip()
    if current:
        words = current.split()
        if term_to_add.lower() not in [w.lower() for w in words]:
            st.session_state["q"] = f"{current} {term_to_add}"
    else:
        st.session_state["q"] = term_to_add

st.markdown('<div class="search-wrap">', unsafe_allow_html=True)
c_q, c_b = st.columns([6, 1], vertical_alignment="bottom")
query = c_q.text_input("Cari", placeholder="🔎  cari berita bola... mis. Vietnam juara",
                       label_visibility="collapsed", key="q").strip()
c_b.button("Cari ⚡", type="primary", width="stretch")
st.markdown('</div>', unsafe_allow_html=True)

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
        # Tampilkan kata original query sebagai static chips
        orig_chips = "".join(
            f'<span class="chip o">{html.escape(t.surface or t.term)}</span>'
            for t in qa.terms if t.source == "asli" and t.term not in STOPWORDS
        )
        st.markdown(
            f'<div class="expbox"><div class="t">✦ Query diperluas · klik saran kata buat tambahkan ke pencarian ↓</div>{orig_chips}</div>',
            unsafe_allow_html=True
        )
        # Expansion terms → clickable secondary buttons (styled as chips via CSS)
        n_exp = min(len(expansions), 6)
        exp_cols = st.columns(n_exp)
        for i, w in enumerate(expansions[:n_exp]):
            exp_cols[i].button(
                f"+ {w}",
                key=f"exp_{i}_{w}",
                use_container_width=True,
                on_click=append_search_query,
                args=(w,)
            )

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