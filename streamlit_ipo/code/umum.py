"""Bagian bersama aplikasi v2: jalur berkas, konstanta, pemuat model/data, dan pembantu tampilan.

Dipakai oleh app_streamlit_v2.py (kerangka dan menu) dan semua halaman_*.py.
Isi bagian konstanta dan pemuat dipindahkan apa adanya dari app_streamlit_v2.py versi sebelumnya.
"""
from __future__ import annotations

import html
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))
import prospectus_extractor as pe  # noqa: E402

ART_DIR = ROOT / "output" / "artifacts"
V2_DIR = ROOT / "output" / "v2"
DATA_DIR = ROOT / "data"
FILE_ASLI = ART_DIR / "dataset_final.parquet"

MAKS_MB = 50
HORIZON = {"3 Bulan": "90", "6 Bulan": "180"}
SKENARIO = {
    "num_tfidf": "Terstruktur + teks faktor risiko (baku)",
    "num": "Terstruktur saja (tanpa teks)",
    "tfidf": "Teks faktor risiko saja",
}
SKENARIO_PENDEK = {"num_tfidf": "Terstruktur + teks", "num": "Terstruktur", "tfidf": "Teks"}
SKENARIO_TEKS = ("tfidf", "num_tfidf")                  # memakai teks faktor risiko
SKENARIO_NUM = ("num", "num_tfidf")                     # memakai fitur terstruktur
CADANGAN_TANPA_TEKS = {"num_tfidf": "num"}              # bila prospektus tidak diunggah
# Variabel yang baru diketahui setelah masa penawaran ditutup; tidak dipakai skenario pra-penawaran.
PASCA_PENAWARAN: list[str] = []   # Oversubscribed Ratio dan Proceeds of IPO sudah dibuang dari fitur v2
PETA_RISIKO = {"A": "risk_utama", "B": "risk_kegiatan_usaha",
               "C": "risk_umum", "D": "risk_investasi_saham"}
NAMA_FITUR = {
    "Debt (Liability) To Equity": "Debt to Equity (DER)", "ROA": "Return on Assets (ROA)",
    "Number of Shares Offered": "Jumlah saham ditawarkan", "Share offered": "Porsi saham ditawarkan",
    "IPO Offer Size (represented in Rupiah)": "Nilai penawaran (Rp)",
    "Oversubscribed Ratio": "Oversubscription ratio", "Owner Retention": "Ownership retention",
    "Proceeds of IPO": "Orientasi penggunaan dana", "Promised Dividend": "Rasio dividen dijanjikan",
    "Overall UW Rep": "Reputasi penjamin emisi", "UW Tally": "Jumlah penjamin emisi",
    "Market Activity": "Aktivitas pasar IPO (bulan listing)", "Offer Price": "Harga penawaran",
    "uop_pct_capex": "Dana untuk belanja modal (%)", "uop_pct_bayar_utang": "Dana untuk bayar utang (%)",
    "uop_pct_modal_kerja": "Dana untuk modal kerja (%)", "uop_pct_akuisisi": "Dana untuk akuisisi (%)",
    "uop_pct_ekspansi": "Dana untuk ekspansi (%)", "Is Shareholders Lock": "Ada lock-up pemegang saham",
}
SEGMEN_RISIKO = {"risk_utama": "risiko utama", "risk_kegiatan_usaha": "risiko usaha",
                 "risk_umum": "risiko umum", "risk_investasi_saham": "risiko investasi"}
FITUR_FORM = [   # (kolom dataset, label, keterangan); Overall UW Rep dan UW Tally datang dari isian penjamin emisi
    ("Offer Price", "Offer Price (Rp)", "Harga penawaran per lembar saham"),
    ("Number of Shares Offered", "Jumlah saham ditawarkan", "Lembar saham yang dilepas ke publik"),
    ("Share offered", "Porsi saham ditawarkan", "Rasio terhadap total saham (0,1 = 10%)"),
    ("IPO Offer Size (represented in Rupiah)", "Nilai penawaran (Rp)", "Jumlah saham x harga penawaran"),
    ("Owner Retention", "Ownership retention", "Rasio kepemilikan lama setelah IPO (0,9 = 90%)"),
    ("Debt (Liability) To Equity", "Debt to Equity (DER)", "Total liabilitas / total ekuitas (kali)"),
    ("ROA", "Return on Assets (ROA)", "Laba bersih / total aset sebagai rasio (0,03 = 3%)"),
    ("Promised Dividend", "Rasio dividen dijanjikan", "Rasio dividen terhadap laba yang dijanjikan"),
    ("Market Activity", "Aktivitas pasar IPO", "Jumlah IPO pada bulan pencatatan yang sama"),
    ("uop_pct_capex", "Dana untuk belanja modal (%)", "Persentase dana IPO, 0-100"),
    ("uop_pct_bayar_utang", "Dana untuk bayar utang (%)", "Persentase dana IPO, 0-100"),
    ("uop_pct_modal_kerja", "Dana untuk modal kerja (%)", "Persentase dana IPO, 0-100"),
    ("uop_pct_akuisisi", "Dana untuk akuisisi (%)", "Persentase dana IPO, 0-100"),
    ("uop_pct_ekspansi", "Dana untuk ekspansi (%)", "Persentase dana IPO, 0-100"),
]
NAMA_HORIZON = {"1": "H+1", "30": "H+30", "90": "H+90", "180": "H+180"}

# Diisi oleh app_streamlit_v2.py saat navigasi dibuat; dipakai tombol pindah halaman.
HALAMAN: dict = {}

MONO_CSS = """<style>
h3,.app-header,[data-testid="stWidgetLabel"] p,.stButton>button,.stDownloadButton>button,input{
  font-family:'JetBrains Mono',monospace!important}
</style>"""


def header(judul: str = "IPO Stock Performance Predictor", mono: bool = False) -> None:
    """Pita judul halaman (di dalam kartu isi)."""
    if mono:
        st.markdown(MONO_CSS, unsafe_allow_html=True)
    st.markdown(f'<div class="app-header">{html.escape(judul)}</div>', unsafe_allow_html=True)


def pindah(nama: str) -> None:
    """Pindah ke halaman lain di menu kiri (dipanggil dari tombol)."""
    st.switch_page(HALAMAN[nama])


# ════════════════════════════════════════════════════════════════════
# Sumber model (dimuat malas: landing dan form tidak perlu model besar)
# ════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def muat_ringan():
    trf = pickle.load(open(V2_DIR / "fitted_transformers_v2.pkl", "rb"))
    return {
        "trf": trf,
        "ringkas": pd.read_csv(V2_DIR / "summary_clf_v2.csv"),
        "kalender": pe.baca_kalender(DATA_DIR / "kalender_ipo.csv"),
        "skor_uw": pe.baca_skor_uw(DATA_DIR / "skor_underwriter.csv"),
    }


@st.cache_resource(show_spinner="Memuat model...")
def muat_model():
    art = pickle.load(open(V2_DIR / "models_v2.pkl", "rb"))
    return art["clf"], art["kolom"]


@st.cache_resource(show_spinner=False)
def muat_regresi() -> dict | None:
    """Model regresi harga (output/v2/models_reg_v2.pkl, dibuat oleh code/latih_regresi.py); None bila belum ada."""
    f = V2_DIR / "models_reg_v2.pkl"
    if not f.exists():
        return None
    return pickle.load(open(f, "rb"))


@st.cache_data(show_spinner=False)
def ringkas_regresi() -> pd.DataFrame | None:
    f = V2_DIR / "summary_reg_v2.csv"
    return pd.read_csv(f) if f.exists() else None


def urut_model(horizon: str, skenario: str) -> list[str]:
    r = muat_ringan()["ringkas"]
    r = r[(r["Feature Set"] == skenario) & (r["Horizon"] == f"H+{horizon}")]
    return r.sort_values("CV AUC", ascending=False)["Model"].tolist()


def kinerja_model(horizon: str, skenario: str, nama: str) -> pd.Series:
    r = muat_ringan()["ringkas"]
    return r[(r["Feature Set"] == skenario) & (r["Horizon"] == f"H+{horizon}") & (r["Model"] == nama)].iloc[0]


def nama_fitur_ramah(kol: str) -> str:
    import re
    if kol in NAMA_FITUR:
        return NAMA_FITUR[kol]
    if kol.startswith("Sector_"):
        return "Sektor: " + kol[7:]
    if kol.startswith("Listing Board_"):
        return "Papan: " + kol[14:]
    m = re.match(r"tfidf_(risk_\w+?)__(.+)$", kol)
    if m:
        return f"Teks {SEGMEN_RISIKO.get(m.group(1), m.group(1))}: {m.group(2)}"
    return kol


def importance(model, kolom: list[str]) -> pd.Series | None:
    imp = getattr(model, "feature_importances_", None)
    if imp is None and hasattr(model, "coef_"):
        imp = np.abs(np.ravel(model.coef_))
    if imp is None or len(imp) != len(kolom):
        return None
    imp = np.asarray(imp, dtype=float)
    total = imp.sum()
    return pd.Series(imp / total if total > 0 else imp, index=kolom).sort_values(ascending=False)


# ════════════════════════════════════════════════════════════════════
# Data penelitian dan evaluasi ulang model pada data uji
# ════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def _baca_asli(mtime: float, kolom: tuple | None) -> pd.DataFrame:
    return pd.read_parquet(FILE_ASLI, columns=list(kolom) if kolom else None).reset_index(drop=True)


def data_asli(kolom: list[str] | None = None) -> pd.DataFrame:
    """dataset_final.parquet (data yang dipakai melatih model), bukan salinan yang diedit di halaman Data."""
    return _baca_asli(FILE_ASLI.stat().st_mtime, tuple(kolom) if kolom else None)


def bagi_data(label_90: pd.Series) -> tuple[list[int], list[int]]:
    """Pembagian latih/uji 80:20 yang sama dengan notebook: acak (seed 42), stratifikasi pada label_90."""
    from sklearn.model_selection import train_test_split
    itr, ite = train_test_split(label_90.index, test_size=0.2, random_state=42, stratify=label_90)
    return sorted(itr), sorted(ite)


def fitur_terstruktur(d: pd.DataFrame) -> pd.DataFrame:
    """Fitur skenario 'Terstruktur' untuk banyak emiten sekaligus (transformer sama dengan saat prediksi)."""
    trf = muat_ringan()["trf"]
    num = d[trf["numeric_cols"]].apply(pd.to_numeric, errors="coerce")
    num = pd.DataFrame(trf["scaler"].transform(trf["imputer"].transform(num)),
                       columns=trf["numeric_cols"], index=d.index)
    kat = pd.DataFrame(trf["ohe"].transform(d[trf["categorical_cols"]]), columns=trf["ohe_cols"], index=d.index)
    biner = d[trf["binary_cols"]].astype(float)
    return pd.concat([num, kat, biner], axis=1)


@st.cache_resource(show_spinner="Menyiapkan evaluasi model pada data uji...")
def evaluasi_uji() -> dict:
    """Probabilitas skenario Terstruktur untuk seluruh emiten, plus pembagian latih/uji.

    Pembagian dan prediksi direproduksi dari dataset_final.parquet. Hasilnya dianggap sah hanya bila akurasi dan
    ROC-AUC yang dihitung ulang sama dengan summary_clf_v2.csv untuk SEMUA model skenario ini ('cocok'). Skenario
    berbasis teks tidak dipakai di sini karena tidak bisa direproduksi persis dari berkas data yang ada.
    """
    from sklearn.metrics import roc_auc_score
    df = data_asli()
    MODEL, KOLOM = muat_model()
    itr, ite = bagi_data(df["label_90"])
    X = fitur_terstruktur(df)[KOLOM["num"]]
    prob, cocok = {}, True
    for (h, sk, nama), obj in MODEL.items():
        if sk != "num":
            continue
        p = obj["model"].predict_proba(X)[:, 1]
        prob[(h, nama)] = p
        y = df.loc[ite, f"label_{h}"].to_numpy()
        r = kinerja_model(h, "num", nama)
        acc, auc = float(((p[ite] >= 0.5) == y).mean()), float(roc_auc_score(y, p[ite]))
        cocok &= abs(acc - r["Accuracy"]) < 1e-6 and abs(auc - r["ROC-AUC"]) < 1e-6
    meta = muat_ringan()["trf"].get("meta", {})
    cocok &= (len(itr), len(ite)) == (meta.get("n_train"), meta.get("n_test"))
    return {"prob": prob, "train": itr, "test": ite, "cocok": bool(cocok)}


def ringkasan_penelitian() -> dict:
    """Angka kunci untuk Beranda dan halaman penjelasan (semua dihitung dari berkas, tidak diketik manual)."""
    df = data_asli(["Ticker Code", "Listing Date", "label_1", "label_30", "label_90", "label_180"])
    R = muat_ringan()
    s, meta = R["ringkas"], R["trf"].get("meta", {})
    terbaik = {}
    for h in ("90", "180"):
        r = s[s["Horizon"] == f"H+{h}"].sort_values("CV AUC", ascending=False).iloc[0]
        terbaik[h] = {"model": r["Model"], "skenario": r["Feature Set"], "cv_auc": float(r["CV AUC"]),
                      "auc_uji": float(r["ROC-AUC"]), "akurasi": float(r["Accuracy"])}
    return {"n": len(df), "n_latih": meta.get("n_train"), "n_uji": meta.get("n_test"),
            "tahun_awal": int(df["Listing Date"].dt.year.min()), "tahun_akhir": int(df["Listing Date"].dt.year.max()),
            "base": {h: float(df[f"label_{h}"].mean()) for h in ("1", "30", "90", "180")}, "terbaik": terbaik}


def persen(x: float, desimal: int = 0) -> str:
    """Format persen gaya Indonesia (koma desimal)."""
    return f"{x * 100:.{desimal}f}%".replace(".", ",")


def angka(x: float, desimal: int = 2) -> str:
    return f"{x:.{desimal}f}".replace(".", ",")
