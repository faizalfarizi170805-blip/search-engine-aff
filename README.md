# AFF News Search Engine

Search engine berita AFF: **TF-IDF + Cosine Similarity** dengan **Query Expansion**
(WordNet synset Bahasa Indonesia + kamus AFF buatan sendiri).

## Jalankan
```bash
pip install -r requirements.txt
python tools.py setup          # sekali: unduh WordNet + OMW Bahasa Indonesia
python tools.py warmup         # opsional, sekali: cache stemming + tabel sinonim (startup lebih cepat)
streamlit run app.py
```

## Struktur
- `app.py`              UI Streamlit (halaman pencarian)
- `dev_panel.py`        panel "Nerd Stats" (toggle di sidebar): analisis query, evaluasi, statistik korpus
- `config.py`           semua pengaturan (path file, nama kolom, parameter) + pembaca korpus
- `tools.py`            alat terminal: `setup` (unduh WordNet), `warmup` (siapkan cache), `tune` (cari parameter QE)
- `ir/preprocess.py`    tokenisasi, stopword & stemming Sastrawi
- `ir/expansion.py`     WordNet (lang='ind') + kamus AFF sendiri (`data/KamusAFFSendiri.json`)
- `ir/engine.py`        TF-IDF, query berbobot, explain skor
- `ir/evaluation.py`    P@k, R@k, MAP; bandingkan Baseline vs QE

## Evaluasi
1. Tulis hasil penilaian di `qrels_AFF.csv`: kolom `query`, `relevan` (nomor baris dokumen, mulai 1, pisah `|`).
   (Alternatif format TREC `q_id,doc_id,relevance` juga didukung; format ini memakai `queries.json` sebagai daftar query.)
2. `python tools.py tune` -> parameter terbaik disimpan ke `tuned_params.json` (otomatis dipakai UI).
3. Restart `streamlit run app.py` agar parameter baru terbaca, lalu lihat hasilnya di Nerd Stats -> tab Evaluasi.
