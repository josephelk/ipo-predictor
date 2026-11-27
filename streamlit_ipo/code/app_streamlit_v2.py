"""IPO Stock Performance Predictor, versi 2: kerangka aplikasi dan menu kiri.

Menjalankan (dari folder proyek):  streamlit run code/app_streamlit_v2.py

Versi 2 memakai alur code/preprocessing_model.ipynb: klasifikasi biner (outperform vs IHSG) dengan lima
model (Logistic Regression, Random Forest, XGBoost, AdaBoost, CatBoost), tanpa tuning. Ditambah model regresi harga
(Random Forest, dilatih oleh code/latih_regresi.py) yang memperkirakan harga setelah 3 atau 6 bulan.
Artefak model ada di output/v2/.

File ini hanya memasang tema dan menu kiri. Isi tiap menu ada di modul terpisah:
  Beranda, Panduan Penggunaan, Cara Membaca Analisis, Informasi Tambahan -> halaman_info.py
  Prediksi -> halaman_prediksi.py        Indikator Teknikal -> halaman_teknikal.py
Bagian bersama (konstanta, pemuat model dan data) ada di umum.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

st.set_page_config(page_title="IPO Stock Performance Predictor", page_icon="📈", layout="wide",
                   initial_sidebar_state="auto")

import umum  # noqa: E402
from halaman_info import beranda, cara_membaca, info_tambahan, panduan  # noqa: E402
from halaman_prediksi import halaman as prediksi  # noqa: E402
from halaman_teknikal import indikator_teknikal  # noqa: E402
from tampilan import CSS  # noqa: E402

st.markdown(CSS, unsafe_allow_html=True)

# Urutan menu: Beranda, Panduan, Prediksi, Cara Membaca, Indikator, Info.
HALAMAN = {
    "beranda": st.Page(beranda, title="Beranda", icon=":material/home:", default=True),
    "panduan": st.Page(panduan, title="Panduan Penggunaan", icon=":material/menu_book:", url_path="panduan"),
    "prediksi": st.Page(prediksi, title="Prediksi", icon=":material/query_stats:", url_path="prediksi"),
    "cara_membaca": st.Page(cara_membaca, title="Cara Membaca Analisis", icon=":material/insights:", url_path="cara-membaca"),
    "teknikal": st.Page(indikator_teknikal, title="Indikator Teknikal", icon=":material/show_chart:", url_path="indikator-teknikal"),
    "info": st.Page(info_tambahan, title="Informasi Tambahan", icon=":material/info:", url_path="informasi-tambahan"),
}
umum.HALAMAN.clear()
umum.HALAMAN.update(HALAMAN)

pg = st.navigation(list(HALAMAN.values()), position="sidebar")
st.sidebar.caption("Keluaran penelitian akademik, bukan rekomendasi investasi.")
pg.run()
