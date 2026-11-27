"""Menu Prediksi: unggah prospektus, isi/koreksi data, jalankan model, lalu baca hasilnya.

Logika (ekstraksi, pembentukan fitur, hitung, laporan, form, hasil) dipindahkan apa adanya dari
app_streamlit_v2.py versi sebelumnya. Perubahan hanya: judul halaman, draf isian yang disimpan
(_draft) agar form tidak hilang saat pindah menu, dan fungsi halaman() sebagai pintu masuk menu.
"""
from __future__ import annotations

import html
import re
import tempfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from umum import (CADANGAN_TANPA_TEKS, FILE_ASLI, FITUR_FORM, HORIZON, MAKS_MB, NAMA_FITUR, PASCA_PENAWARAN,
                  PETA_RISIKO, SEGMEN_RISIKO, SKENARIO, SKENARIO_NUM, SKENARIO_PENDEK, SKENARIO_TEKS, angka as fmt_angka,
                  header, kinerja_model, muat_model, muat_regresi, muat_ringan, persen, pe, ringkas_regresi,
                  urut_model)


@st.cache_data(show_spinner=False, max_entries=8)
def ekstrak(berkas_bytes: bytes, nama_berkas: str) -> dict:
    R = muat_ringan()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(berkas_bytes)
        tmp = Path(f.name)
    try:
        halaman = pe.baca_pdf(tmp)
    finally:
        tmp.unlink(missing_ok=True)

    pen = pe.ekstrak_penawaran(halaman)
    keu = pe.ekstrak_keuangan(halaman)
    uw = pe.ekstrak_penjamin(halaman)
    uop = pe.ekstrak_uop(halaman, dana_kotor=pen["nilai_penawaran"])
    tgl = pe.tanggal_pencatatan(halaman)
    risiko = pe.ekstrak_risiko(halaman)
    hasil = {
        "Number of Shares Offered": pen["jumlah_saham"], "Share offered": pen["porsi_saham"],
        "Offer Price": pen["harga_penawaran"], "IPO Offer Size (represented in Rupiah)": pen["nilai_penawaran"],
        "Owner Retention": pen["owner_retention"], "Debt (Liability) To Equity": keu["der"], "ROA": keu["roa"],
        "Promised Dividend": pe.ekstrak_dividen(halaman)["dividen_rasio"],
        "Is Shareholders Lock": pe.ekstrak_lockup(halaman)["lockup"],
        "UW Tally": uw["uw_total"],
        "Overall UW Rep": pe.overall_uw_rep(uw["uw_nama"], uw["uw_porsi"], R["skor_uw"]),
        **{f"uop_pct_{k}": uop[f"uop_pct_{k}"] for k in pe.KATEGORI_UOP},
    }
    hasil["Market Activity"] = pe.market_activity(tgl, R["kalender"]) if tgl else None
    return {"fitur": hasil, "uw": {"nama": uw["uw_nama"], "porsi": uw["uw_porsi"]},
            "info": {"jumlah_halaman": len(halaman), "nama_berkas": nama_berkas,
                     "risiko": {s: risiko[f"risk_{s}"] for s in "ABCD"},
                     "risk_ditemukan": risiko["risk_ditemukan"]}}


def bersihkan_teks(teks) -> str:
    if not isinstance(teks, str):
        return ""
    teks = re.sub(r"[^a-zA-Z\s]", " ", teks.lower())
    return re.sub(r"\s+", " ", teks).strip()


def fitur_numerik(nilai: dict) -> pd.DataFrame:
    R = muat_ringan()
    trf = R["trf"]
    d = pd.DataFrame([nilai])
    for kol in trf["numeric_cols"]:
        d[kol] = pd.to_numeric(d[kol], errors="coerce")
    d[trf["numeric_cols"]] = trf["imputer"].transform(d[trf["numeric_cols"]])   # median data latih
    num = pd.DataFrame(trf["scaler"].transform(d[trf["numeric_cols"]]), columns=trf["numeric_cols"])
    kat = pd.DataFrame(trf["ohe"].transform(d[trf["categorical_cols"]]), columns=trf["ohe_cols"])
    biner = d[trf["binary_cols"]].astype(float).reset_index(drop=True)
    return pd.concat([num, kat, biner], axis=1)


def fitur_teks(risiko: dict) -> pd.DataFrame:
    trf = muat_ringan()["trf"]
    blok = []
    for seg, awalan in PETA_RISIKO.items():
        vec = trf["tfidf_vectorizers"][f"{awalan}_clean"]
        nilai = vec.transform([bersihkan_teks(risiko.get(seg))]).toarray()
        blok.append(pd.DataFrame(nilai, columns=[f"tfidf_{awalan}__{t}" for t in vec.get_feature_names_out()]))
    return pd.concat(blok, axis=1)


def baris_fitur(nilai: dict, risiko: dict, skenario: str, kolom: dict) -> pd.DataFrame:
    bagian = []
    if skenario in SKENARIO_NUM:
        bagian.append(fitur_numerik(nilai))
    if skenario in SKENARIO_TEKS:
        bagian.append(fitur_teks(risiko))
    return pd.concat(bagian, axis=1)[kolom[skenario]]


def hitung(form: dict, pdf: tuple | None) -> dict:
    R = muat_ringan()
    MODEL, KOLOM = muat_model()
    trf = R["trf"]
    skenario, h = form["skenario"], HORIZON[form["jangka"]]

    fitur = {k: None for k in trf["numeric_cols"]}
    kunci = {k: None for k in trf["numeric_cols"]}       # asal tiap nilai
    lock, info = 0.0, None
    if pdf is not None:
        ek = ekstrak(*pdf)
        info = ek["info"]
        for k, v in ek["fitur"].items():
            if k in fitur and v is not None:
                fitur[k], kunci[k] = v, "prospektus"
        lock = float(bool(ek["fitur"].get("Is Shareholders Lock")))
    if "lock" in form:
        lock = float(form["lock"])
    for k, v in form["angka"].items():
        if v is None:
            continue
        if kunci.get(k) == "prospektus" and fitur[k] is not None and np.isclose(float(v), float(fitur[k])):
            continue                                       # tidak diubah pengguna: tetap bersumber prospektus
        fitur[k], kunci[k] = v, "diisi pengguna"            # isian pengguna menimpa ekstraksi
    fitur["Sector"], fitur["Listing Board"], fitur["Is Shareholders Lock"] = form["sektor"], form["papan"], lock

    risiko = info["risiko"] if info else {}
    catatan = None
    if skenario in SKENARIO_TEKS and not (info and info["risk_ditemukan"]):
        if skenario in CADANGAN_TANPA_TEKS:        # tanpa prospektus: pakai skenario yang sama tanpa teks
            skenario = CADANGAN_TANPA_TEKS[skenario]
            catatan = ("Prospektus tidak diunggah (atau bab Faktor Risiko tidak ditemukan), sehingga prediksi memakai "
                       f"skenario \"{SKENARIO[skenario]}\". Unggah prospektus agar teks faktor risiko ikut dipakai.")
        else:
            return {"galat": "Skenario berbasis teks memerlukan prospektus yang memuat bab Faktor Risiko. "
                             "Unggah prospektus atau pilih skenario lain di Pengaturan lanjutan."}

    dipakai = list(trf["numeric_cols"])
    terisi = [k for k in dipakai if fitur[k] is not None]

    X = baris_fitur(fitur, risiko, skenario, KOLOM)
    nama = form["model"] or urut_model(h, skenario)[0]
    model = MODEL[(h, skenario, nama)]["model"]
    p = float(model.predict_proba(X)[0, 1])
    return {"p": p, "model": nama, "skenario": skenario, "h": h, "jangka": form["jangka"],
            "X": X, "fitur": fitur, "asal": kunci, "terisi": len(terisi), "total": len(dipakai),
            "kinerja": kinerja_model(h, skenario, nama), "info": info, "catatan": catatan,
            "harga": prediksi_harga(fitur, risiko, h, KOLOM)}


REG_MODEL = "Random Forest"   # model regresi terbaik pada data uji (lihat output/v2/summary_reg_v2.csv)


def prediksi_harga(fitur: dict, risiko: dict, h: str, kolom: dict) -> dict | None:
    """Prediksi harga setelah h hari = harga penawaran x exp(perubahan log yang diprediksi).

    Rentang 80% memakai sebaran selisih (aktual - prediksi) dari validasi silang pada data latih.
    Bila harga penawaran tidak terisi, hanya perubahan persen yang bisa ditampilkan (harga Rupiah tidak).
    """
    art = muat_regresi()
    if art is None or (h, REG_MODEL) not in art["reg"]:
        return None
    reg = art["reg"][(h, REG_MODEL)]
    x = baris_fitur(fitur, risiko, "num", kolom)[art["kolom"]]
    log_ubah = float(reg["model"].predict(x)[0])
    ubah = {"tengah": float(np.exp(log_ubah) - 1), "bawah": float(np.exp(log_ubah + reg["q10"]) - 1),
            "atas": float(np.exp(log_ubah + reg["q90"]) - 1)}
    offer = fitur.get("Offer Price")
    harga = None if offer is None else {k: float(offer) * (1 + v) for k, v in ubah.items()}
    ringkas = ringkas_regresi()
    metrik = None
    if ringkas is not None:
        r = ringkas[ringkas["Horizon"] == f"H+{h}"].set_index("Model")
        if REG_MODEL in r.index and "Baseline (median latih)" in r.index:
            metrik = {"mae_model": float(np.exp(r.loc[REG_MODEL, "MAE log"]) - 1),
                      "mae_dasar": float(np.exp(r.loc["Baseline (median latih)", "MAE log"]) - 1)}
    return {"offer": offer, "ubah": ubah, "harga": harga, "model": REG_MODEL, "metrik": metrik}


def laporan_html(h: dict) -> str:
    ok = h["p"] >= 0.5
    baris = "".join(f"<tr><td>{html.escape(NAMA_FITUR.get(k, k))}</td><td>{'' if v is None else html.escape(f'{v:,.4g}')}</td>"
                    f"<td>{html.escape(h['asal'].get(k) or 'median data latih')}</td></tr>"
                    for k, v in h["fitur"].items() if k in NAMA_FITUR)
    ph, hg = h.get("harga"), ""
    if ph:
        rp = lambda v: f"Rp{v:,.0f}".replace(",", ".")
        pct = lambda v: f"{v:+.0%}"
        if ph["harga"]:
            hg = (f"<tr><td>Harga penawaran</td><td>{rp(ph['offer'])}</td></tr>"
                  f"<tr><td>Perkiraan harga setelah {h['h']} hari</td><td>{rp(ph['harga']['tengah'])} ({pct(ph['ubah']['tengah'])})</td></tr>"
                  f"<tr><td>Rentang 80%</td><td>{rp(ph['harga']['bawah'])} sampai {rp(ph['harga']['atas'])}</td></tr>")
        else:
            hg = (f"<tr><td>Perkiraan perubahan harga setelah {h['h']} hari</td><td>{pct(ph['ubah']['tengah'])} "
                  f"(rentang 80%: {pct(ph['ubah']['bawah'])} sampai {pct(ph['ubah']['atas'])})</td></tr>")
    return f"""<!doctype html><html lang="id"><meta charset="utf-8"><title>Laporan prediksi IPO</title>
<style>body{{font-family:Arial,sans-serif;max-width:760px;margin:2rem auto;color:#222}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:6px 10px;text-align:left}}
.b{{font-size:1.6rem;font-weight:700;color:{'#2e6b2e' if ok else '#9a3a30'}}}</style>
<h1>Laporan Prediksi Kinerja Saham IPO</h1><p>Dibuat {date.today():%d-%m-%Y}</p>
<p class="b">{'OUTPERFORM' if ok else 'UNDERPERFORM'} &mdash; peluang mengungguli IHSG {h['p']:.1%}</p>
<p>Horizon: {h['h']} hari ({html.escape(h['jangka'])}) &middot; Model: {html.escape(h['model'])} &middot;
Skenario: {html.escape(SKENARIO[h['skenario']])}<br>
Kinerja model pada penelitian: akurasi uji {h['kinerja']['Accuracy']:.0%}, ROC-AUC validasi silang {h['kinerja']['CV AUC']:.2f}.<br>
Fitur terisi nilai nyata: {h['terisi']} dari {h['total']} (sisanya median data latih).</p>
<h2>Prediksi harga (regresi)</h2><table>{hg or '<tr><td>tidak tersedia</td></tr>'}</table>
<p><i>Perkiraan harga sangat tidak pasti; gunakan rentangnya, bukan satu angka.</i></p>
<h2>Nilai fitur</h2><table><tr><th>Variabel</th><th>Nilai</th><th>Sumber</th></tr>{baris}</table>
<p><i>Keluaran penelitian akademik, bukan rekomendasi investasi.</i></p></html>"""


KUNCI_ISIAN = ["der", "harga", "retensi", "roa"]


def cocokkan_uw(nama: list[str], tabel: dict) -> list[str]:
    """Samakan nama penjamin hasil ekstraksi dengan nama pada tabel skor (mis. tanpa 'Indonesia')."""
    import difflib
    hasil = []
    for n in nama:
        n = n.strip().lower()
        cocok = n if n in tabel else (difflib.get_close_matches(n, list(tabel), n=1, cutoff=0.8) or [n])[0]
        if cocok not in hasil:
            hasil.append(cocok)
    return hasil


def format_nilai(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    v = float(v)
    return f"{v:,.0f}" if abs(v) >= 1e5 else f"{v:.4g}"


def tampil_risiko(teks: dict, kunci: str) -> None:
    """Isi teks tiap segmen Faktor Risiko hasil ekstraksi (A-D), dapat dibuka dan dibaca."""
    st.markdown("**Isi faktor risiko yang diekstraksi**")
    ada = False
    isi = {sg: (str(teks.get(sg)).strip() if teks.get(sg) else "") for sg in "ABCD"}
    label = [f"{SEGMEN_RISIKO.get(PETA_RISIKO.get(sg), sg).capitalize()} ({len(t.split()):,} kata)" if t
             else f"{SEGMEN_RISIKO.get(PETA_RISIKO.get(sg), sg).capitalize()} (kosong)" for sg, t in isi.items()]
    for tab, (sg, t) in zip(st.tabs(label), isi.items()):      # tab (bukan expander) agar boleh di dalam expander
        with tab:
            if t:
                ada = True
                st.text_area("Teks", t, height=280, disabled=True, label_visibility="collapsed",
                             key=f"risk_{kunci}_{sg}")
            else:
                st.caption("Segmen ini tidak ditemukan pada prospektus (sebagian prospektus tidak memuat keempat kategori).")
    if ada:
        st.caption("Teks ini (setelah dibersihkan: huruf kecil, tanpa angka/tanda baca, tanpa stopword) diubah menjadi "
                   "fitur TF-IDF untuk model.")


def tampil_terbaca(ekdata: dict) -> None:
    """Tabel semua variabel yang berhasil dibaca dari prospektus, termasuk yang tidak masuk kolom form."""
    baris = []
    for kol, nilai in ekdata["fitur"].items():
        if kol == "Is Shareholders Lock":
            baris.append({"Variabel": NAMA_FITUR.get(kol, kol), "Nilai": "Ya" if nilai else "Tidak ada",
                          "Status": "terbaca" if nilai is not None else "tidak terbaca"})
            continue
        baris.append({"Variabel": NAMA_FITUR.get(kol, kol), "Nilai": format_nilai(nilai),
                      "Status": "terbaca" if nilai is not None else "tidak terbaca"})
    for kol in PASCA_PENAWARAN:
        baris.append({"Variabel": NAMA_FITUR.get(kol, kol), "Nilai": "-", "Status": "tidak ada di prospektus"})
    for s, kata in ekdata["risiko"].items():
        baris.append({"Variabel": f"Faktor risiko - {SEGMEN_RISIKO.get(PETA_RISIKO.get(s), s)}",
                      "Nilai": f"{kata:,} kata" if kata else "-", "Status": "terbaca" if kata else "tidak terbaca"})
    tabel = pd.DataFrame(baris)
    n_ok = int((tabel["Status"] == "terbaca").sum())
    st.markdown(f"**Data yang terbaca dari prospektus: {n_ok} dari {len(tabel)} variabel**")
    st.dataframe(tabel, hide_index=True, width="stretch", height=min(40 + 35 * len(tabel), 520))
    st.caption("Variabel yang tidak terbaca diisi median data latih saat prediksi. Kolom di form dapat dikoreksi; "
               "isian Anda menimpa hasil ekstraksi.")
    if ekdata.get("teks"):
        tampil_risiko(ekdata["teks"], ekdata.get("kunci", "x"))


def skala_uw(R) -> tuple[float, float]:
    """Rentang skor penjamin pada skor_underwriter.csv (untuk konversi skala 0-10 <-> 0-1)."""
    skor = list(R["skor_uw"].values()) if isinstance(R["skor_uw"], dict) else []
    return (min(skor), max(skor)) if skor else (0.0, 10.0)


def isian_dari_pdf(fitur: dict, lo: float, hi: float) -> dict:
    """Hasil ekstraksi -> nilai kolom form (dalam satuan yang tampil di form)."""
    peta = {"der": ("Debt (Liability) To Equity", 1), "harga": ("Offer Price", 1),
            "retensi": ("Owner Retention", 100), "roa": ("ROA", 100)}
    isi = {k: round(float(fitur[kol]) * f, 4) for k, (kol, f) in peta.items() if fitur.get(kol) is not None}
    return isi


def layar_form():
    header("Prediksi", mono=True)
    R = muat_ringan()
    trf = R["trf"]
    # Isian terakhir (draf) ikut dibaca supaya form tidak kosong lagi setelah pengguna membuka menu lain.
    simpan = {**st.session_state.get("form", {}), **st.session_state.get("_draft", {})}

    # Nilai penjamin hasil ekstraksi dipasang SEBELUM widget dibuat (Streamlit tidak mengizinkan
    # mengubah state widget setelah widget itu tampil pada run yang sama).
    pend_uw = st.session_state.pop("_prefill_uw", None)
    if pend_uw is not None:
        st.session_state["in_uw_nama"] = pend_uw["nama"]
        st.session_state["uw_porsi_map"] = pend_uw["porsi"]
    elif "in_uw_nama" not in st.session_state:
        st.session_state["in_uw_nama"] = simpan.get("uw_nama", [])
        st.session_state["uw_porsi_map"] = simpan.get("uw_porsi", {})

    # ---- 1. Upload prospektus (dulu); data lain terisi darinya ----
    st.markdown("### 1. Upload prospektus")
    pdf_lama = st.session_state.get("pdf")
    berkas = st.file_uploader("Prospektus (PDF)", type="pdf",
                              help="Setelah diunggah, seluruh data di bawah terisi otomatis dari prospektus. "
                                   f"Periksa dan koreksi bila perlu. Maks. {MAKS_MB}MB.")
    st.markdown(f'<div class="cap">Format: PDF, maks. {MAKS_MB}MB. Tanpa prospektus, data di bawah dapat diisi manual.</div>',
                unsafe_allow_html=True)

    if berkas is not None:
        pid_baru = (berkas.name, berkas.size)
        if st.session_state.get("pdf_id") != pid_baru:       # file baru -> ekstrak dan isi form
            if berkas.size > MAKS_MB * 1024 * 1024:
                st.error(f"Ukuran file melebihi {MAKS_MB}MB.")
            else:
                try:
                    with st.spinner("Membaca prospektus (bisa 1-2 menit untuk dokumen tebal)..."):
                        ek = ekstrak(berkas.getvalue(), berkas.name)
                except Exception as e:                       # PDF rusak / format tak dikenali
                    st.error(f"Prospektus tidak dapat dibaca ({type(e).__name__}). "
                             "Isi data di bawah secara manual.")
                else:
                    nama_uw = cocokkan_uw(ek["uw"]["nama"], R["skor_uw"])
                    porsi_uw = {}
                    for n_asli, p in zip(ek["uw"]["nama"], ek["uw"]["porsi"]):
                        porsi_uw.setdefault(cocokkan_uw([n_asli], R["skor_uw"])[0], p)
                    st.session_state.update({
                        "pdf": (berkas.getvalue(), berkas.name), "pdf_id": pid_baru,
                        "_prefill_uw": {"nama": nama_uw, "porsi": porsi_uw},
                        "_ekdata": {"fitur": ek["fitur"], "teks": ek["info"]["risiko"], "kunci": berkas.name,
                                    "risiko": {s: len(str(t).split()) if t else 0
                                               for s, t in ek["info"]["risiko"].items()}},
                        "_ekinfo": {"nama": berkas.name, "halaman": ek["info"]["jumlah_halaman"]}})
                    st.rerun()
    pdf_lama = st.session_state.get("pdf")
    info_ek = st.session_state.get("_ekinfo")
    ekdata = st.session_state.get("_ekdata") if pdf_lama else None
    pid = st.session_state.get("pdf_id") if pdf_lama else None
    if pdf_lama and info_ek:
        n_ok = sum(ekdata["fitur"].get(k) is not None for k, _, _ in FITUR_FORM) if ekdata else 0
        st.success(f"Terbaca {info_ek['halaman']} halaman dari **{html.escape(info_ek['nama'])}**: "
                   f"{n_ok} dari {len(FITUR_FORM)} baris data fundamental terisi otomatis. "
                   "Periksa dan koreksi nilai di tabel bila ada yang keliru.")
        if berkas is None and st.button("Hapus file prospektus", key="hapus_pdf"):
            for k in ("pdf", "pdf_id", "_ekinfo", "_ekdata"):
                st.session_state.pop(k, None)
            st.rerun()

    # ---- 2. Data fundamental: semua kolom, dapat diubah ----
    st.markdown("### 2. Data fundamental perusahaan")
    st.caption("Semua variabel fitur terstruktur. Klik sel pada kolom Nilai untuk mengubah; "
               "nilai yang Anda ubah menimpa hasil ekstraksi prospektus.")
    fund_simpan = simpan.get("fund") if simpan.get("pdf_id") == pid else None
    baris = []
    for kol, label, ket in FITUR_FORM:
        nilai_ek = ekdata["fitur"].get(kol) if ekdata else None
        nilai = fund_simpan.get(kol) if fund_simpan and kol in fund_simpan else nilai_ek
        status = ("terbaca" if nilai_ek is not None else "tidak terbaca") if ekdata else "isi manual"
        baris.append({"Variabel": label, "Nilai": None if nilai is None else float(nilai),
                      "Status": status, "Keterangan": ket})
    ed = st.data_editor(
        pd.DataFrame(baris), hide_index=True, width="stretch", num_rows="fixed",
        height=40 + 35 * len(baris), key=f"fund_{pid}", disabled=["Variabel", "Status", "Keterangan"],
        column_config={"Nilai": st.column_config.NumberColumn(format="%.6g")})
    fund = {kol: (None if pd.isna(v) else float(v)) for (kol, _, _), v in zip(FITUR_FORM, ed["Nilai"])}
    lock = st.checkbox("Ada lock-up pemegang saham",
                       value=(bool(simpan["lock"]) if "lock" in simpan and simpan.get("pdf_id") == pid
                              else bool(ekdata["fitur"].get("Is Shareholders Lock")) if ekdata else False),
                       key=f"lock_{pid}")

    # ---- Penjamin emisi: skor reputasi dihitung otomatis dari nama dan porsi ----
    tabel_uw = R["skor_uw"]
    uw_nama = st.multiselect(
        "Penjamin emisi (underwriter)", sorted(tabel_uw, key=lambda n: -tabel_uw[n]), key="in_uw_nama",
        accept_new_options=True, placeholder="Pilih atau ketik nama penjamin...",
        help="Boleh lebih dari satu. Skor reputasi dihitung otomatis dari tabel data/skor_underwriter.csv. "
             "Penjamin yang tidak ada di tabel diberi skor median.")
    uw_skala, uw_porsi_skrg = None, {}
    if uw_nama:
        peta_porsi = st.session_state.get("uw_porsi_map", {})
        sama = round(100 / len(uw_nama), 2)
        lengkap = all(peta_porsi.get(n) is not None for n in uw_nama)   # porsi terbaca untuk semua penjamin
        awal = pd.DataFrame({
            "Penjamin": uw_nama,
            "Skor": [tabel_uw.get(n.strip().lower()) for n in uw_nama],
            "Porsi (%)": [peta_porsi[n] for n in uw_nama] if lengkap else [sama] * len(uw_nama)})
        ed_uw = st.data_editor(
            awal, hide_index=True, width="stretch", num_rows="fixed", key=f"uw_porsi_{abs(hash(tuple(uw_nama)))}",
            column_config={"Penjamin": st.column_config.TextColumn(disabled=True),
                           "Skor": st.column_config.NumberColumn(disabled=True, format="%.2f",
                                                                 help="Kosong = tidak ada di tabel (dipakai median)"),
                           "Porsi (%)": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, format="%.2f")})
        porsi = [None if pd.isna(p) else float(p) for p in ed_uw["Porsi (%)"]]
        uw_porsi_skrg = dict(zip(ed_uw["Penjamin"], porsi))
        uw_skala = pe.overall_uw_rep(list(ed_uw["Penjamin"]), porsi, tabel_uw)
        lo_t, hi_t = skala_uw(R)
        if uw_skala is not None:
            st.markdown(f"Skor reputasi gabungan: **{lo_t + uw_skala * (hi_t - lo_t):.2f}** "
                        f"(skala {lo_t:.1f}-{hi_t:.1f}) · {len(uw_nama)} penjamin")
        st.caption("Lebih dari satu penjamin: skor gabungan = rata-rata skor yang ditimbang porsi penjaminan. "
                   "Jika semua porsi 0, dipakai rata-rata biasa.")

    if ekdata and ekdata.get("teks"):
        with st.expander("Isi faktor risiko yang diekstraksi dari prospektus"):
            tampil_risiko(ekdata["teks"], ekdata.get("kunci", "x"))

    sektor_opsi = list(trf["ohe"].categories_[1])
    sektor = st.selectbox("Sektor industri", sektor_opsi, index=sektor_opsi.index(simpan["sektor"])
                          if simpan.get("sektor") in sektor_opsi else None, placeholder="Pilih sektor...")
    jangka = st.selectbox("Jangka Waktu", list(HORIZON), index=list(HORIZON).index(simpan.get("jangka", "6 Bulan")))

    with st.expander("Pengaturan lanjutan"):
        papan_opsi = list(trf["ohe"].categories_[0])
        papan = st.selectbox("Papan pencatatan", papan_opsi,
                             index=papan_opsi.index(simpan["papan"]) if simpan.get("papan") in papan_opsi
                             else min(2, len(papan_opsi) - 1))
        skenario = st.selectbox("Fitur yang dipakai", list(SKENARIO), format_func=lambda s: SKENARIO[s],
                                index=list(SKENARIO).index(simpan.get("skenario", "num_tfidf")),
                                help="Baku: fitur terstruktur dari prospektus + teks faktor risiko (TF-IDF). Pada data "
                                     "penelitian ini, tambahan teks belum terbukti memperbaiki ROC-AUC. Tanpa "
                                     "prospektus, teks tidak tersedia dan model memakai fitur terstruktur saja.")
        h = HORIZON[jangka]
        opsi_model = ["Terbaik (otomatis)"] + urut_model(h, skenario)
        pilih = st.selectbox("Model", opsi_model,
                             index=opsi_model.index(simpan["model_pilih"]) if simpan.get("model_pilih") in opsi_model else 0)

    st.session_state["_draft"] = {"pdf_id": pid, "fund": fund, "lock": lock, "uw_nama": uw_nama,
                                 "uw_porsi": uw_porsi_skrg, "sektor": sektor, "jangka": jangka, "papan": papan,
                                 "skenario": skenario, "model_pilih": pilih}
    st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
    if st.button("Jalankan prediksi →", key="jalan", width="stretch"):
        pdf = (berkas.getvalue(), berkas.name) if berkas is not None else pdf_lama
        terisi = sum(v is not None for v in fund.values()) + (uw_skala is not None)
        if sektor is None:
            st.error("Pilih sektor industri terlebih dahulu.")
        elif pdf is None and terisi < 3:
            st.error("Isi minimal 3 baris data fundamental atau upload prospektus.")
        elif pdf is not None and len(pdf[0]) > MAKS_MB * 1024 * 1024:
            st.error(f"Ukuran file melebihi {MAKS_MB}MB.")
        else:
            angka = {**{k: v for k, v in fund.items() if v is not None},
                     "UW Tally": len(uw_nama) if uw_nama else None, "Overall UW Rep": uw_skala}
            form = {"angka": angka, "fund": fund, "pdf_id": pid, "lock": lock, "sektor": sektor, "jangka": jangka,
                    "papan": papan, "uw_nama": uw_nama, "uw_porsi": uw_porsi_skrg, "skenario": skenario,
                    "model": None if pilih.startswith("Terbaik") else pilih}
            st.session_state["form"] = {**form}
            st.session_state["pdf"] = pdf
            with st.spinner("Membaca prospektus dan menghitung prediksi..." if pdf else "Menghitung prediksi..."):
                st.session_state["hasil"] = hitung(form, pdf)
            st.session_state["halaman"] = "hasil"
            st.rerun()


@st.cache_data(show_spinner=False)
def data_historis(mtime: float) -> pd.DataFrame:
    return pd.read_parquet(FILE_ASLI, columns=["Ticker Code", "Sector", "log_excess_return_90", "log_excess_return_180",
                                                "label_90", "label_180"])


def grafik_historis(hasil: dict) -> None:
    """Sebaran return relatif terhadap IHSG pada IPO historis, dibandingkan dengan peluang prediksi model."""
    import altair as alt
    h, p = hasil["h"], hasil["p"]
    if not FILE_ASLI.exists():
        return
    d = data_historis(FILE_ASLI.stat().st_mtime).copy()
    d["rel"] = (np.exp(d[f"log_excess_return_{h}"]) - 1) * 100          # % lebih tinggi/rendah dari IHSG
    d["Hasil"] = np.where(d["rel"] > 0, "Outperform", "Underperform")
    base = float(d[f"label_{h}"].mean())
    sektor = st.session_state.get("form", {}).get("sektor")
    ds = d[d["Sector"] == sektor] if sektor else d.iloc[0:0]

    st.markdown(f'<div class="fi-title">Konteks historis: {len(d)} IPO BEI (H+{h})</div>', unsafe_allow_html=True)
    k1, k2, k3 = st.columns(3)
    k1.metric(f"Mengungguli IHSG (semua IPO)", f"{base:.0%}")
    if len(ds) >= 5:
        k2.metric(f"Sektor {sektor} (n = {len(ds)})", f"{float(ds[f'label_{h}'].mean()):.0%}")
    else:
        k2.metric("Sektor yang sama", "n < 5")
    k3.metric("Prediksi model untuk emiten ini", f"{p:.0%}")

    batas = (-100, 300)
    dd = d[(d["rel"] >= batas[0]) & (d["rel"] <= batas[1])]
    grafik = (alt.Chart(dd).mark_bar().encode(
        x=alt.X("rel:Q", bin=alt.Bin(step=20, extent=list(batas)), title="Return relatif terhadap IHSG (%)"),
        y=alt.Y("count()", title="Jumlah emiten"),
        color=alt.Color("Hasil:N", scale=alt.Scale(domain=["Underperform", "Outperform"],
                                                    range=["#e9a39b", "#b6dcae"]), legend=alt.Legend(orient="top")),
        tooltip=["count()"]).properties(height=240))
    garis = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="white", strokeDash=[4, 4]).encode(x="x:Q")
    st.altair_chart(grafik + garis, width="stretch")
    luar = len(d) - len(dd)
    st.caption(f"Garis putus-putus = kinerja sama dengan IHSG. {luar} emiten di luar rentang {batas[0]}% sampai "
               f"{batas[1]}% tidak ditampilkan. Ini gambaran hasil IPO historis, bukan perkiraan harga emiten ini; "
               "peluang model di atas adalah probabilitas mengungguli IHSG. Perkiraan harga ada di bagian bawah.")
    interpretasi_historis(d, h, p, sektor, ds)


def interpretasi_historis(d: pd.DataFrame, h: str, p: float, sektor: str | None, ds: pd.DataFrame) -> None:
    """Penjelasan untuk pembaca: arti grafik, dan insight yang bisa diambil (angka dihitung dari data)."""
    base = float(d[f"label_{h}"].mean())
    bulan = int(h) // 30
    rel = d["rel"]
    median_rel = float(rel.median())
    besar_naik = float((rel > 100).mean())     # unggul lebih dari 100% di atas IHSG
    besar_turun = float((rel < -50).mean())    # tertinggal lebih dari 50% di bawah IHSG
    selisih = p - base

    if abs(selisih) < 0.05:
        posisi = (f"Prediksi model untuk emiten ini ({persen(p)}) **dekat dengan rata-rata IPO** ({persen(base)}). "
                  "Artinya, dari data prospektusnya model tidak melihat tanda yang jelas lebih baik maupun lebih buruk "
                  "dibanding IPO pada umumnya.")
    elif selisih > 0:
        posisi = (f"Prediksi model untuk emiten ini ({persen(p)}) **di atas rata-rata IPO** ({persen(base)}), sekitar "
                  f"{fmt_angka(p / base, 1)} kali lipat. Artinya, karakteristik prospektus emiten ini lebih mirip IPO "
                  "yang dulu berhasil mengungguli IHSG daripada yang tertinggal.")
    else:
        posisi = (f"Prediksi model untuk emiten ini ({persen(p)}) **di bawah rata-rata IPO** ({persen(base)}). "
                  "Artinya, karakteristik prospektus emiten ini lebih mirip IPO yang dulu tertinggal dari IHSG.")

    if base < 0.5:
        mayoritas = (f"**Kebanyakan IPO tertinggal dari IHSG.** Hanya {persen(base)} dari {len(d)} IPO yang mengungguli IHSG "
                     f"dalam {bulan} bulan. Tanpa analisis tambahan, peluang awal sebuah IPO justru di bawah separuh.")
    else:
        mayoritas = (f"**Sebagian besar IPO mengungguli IHSG.** {persen(base)} dari {len(d)} IPO berhasil unggul dalam "
                     f"{bulan} bulan, tetapi sisanya tetap tertinggal.")

    sektor_txt = ""
    if sektor and len(ds) >= 5:
        bs = float(ds[f"label_{h}"].mean())
        sektor_txt = (f"- **Pembanding sektor.** Pada sektor {sektor} ({len(ds)} emiten), {persen(bs)} IPO mengungguli IHSG. "
                      + ("Jumlah emitennya sedikit, jadi angka ini mudah berubah dan jangan dianggap pasti. " if len(ds) < 20 else "")
                      + "\n")

    with st.container(border=True):
        st.markdown(
            "**Cara membaca grafik ini**\n\n"
            f"Grafik menunjukkan hasil {len(d)} IPO di BEI, diukur {bulan} bulan setelah listing: seberapa jauh kenaikan "
            "harga sahamnya lebih tinggi (kanan) atau lebih rendah (kiri) dibanding IHSG. Batang **merah** adalah IPO yang "
            "tertinggal dari IHSG, batang **hijau** yang mengungguli. Semakin tinggi batangnya, semakin banyak emiten "
            "dengan hasil seperti itu. Garis putus-putus adalah titik impas: kinerjanya sama dengan IHSG.\n\n"
            "**Insight yang bisa diambil**\n\n"
            f"- {mayoritas}\n"
            f"- **Hasilnya sangat beragam.** Selisih terhadap IHSG ada di kisaran dari tertinggal tajam sampai unggul jauh. "
            f"Sekitar {persen(besar_naik)} IPO unggul lebih dari 100% di atas IHSG, sedangkan {persen(besar_turun)} "
            f"tertinggal lebih dari 50% di bawahnya. Nilai tengahnya {median_rel:+.0f}% terhadap IHSG.\n"
            + sektor_txt +
            f"- **Posisi emiten ini.** {posisi}\n\n"
            "**Yang perlu diingat**\n\n"
            f"Peluang model bukan jaminan dan bukan besarnya keuntungan: {persen(p)} berarti model menaksir peluang unggul, "
            f"bukan perkiraan imbal hasil {persen(p)}. Peluang tinggi tetap bisa berakhir tertinggal, dan sebaliknya. "
            "Grafik ini gambaran masa lalu, bukan rekomendasi investasi.")


def rupiah(v: float) -> str:
    return f"Rp{v:,.0f}".replace(",", ".")


def bagian_harga(hasil: dict) -> None:
    """Prediksi harga (regresi) pada horizon yang dipilih pengguna, disajikan sebagai perkiraan dengan rentang."""
    ph = hasil.get("harga")
    bulan, h = int(hasil["h"]) // 30, hasil["h"]
    st.markdown(f'<div class="fi-title">Prediksi harga setelah {bulan} bulan (H+{h})</div>', unsafe_allow_html=True)
    if not ph:
        st.info("Model regresi harga tidak tersedia. Jalankan python code/latih_regresi.py untuk membuatnya.")
        return
    u = ph["ubah"]
    arah = "naik" if u["tengah"] >= 0 else "turun"
    if ph["harga"]:
        hg = ph["harga"]
        k1, k2, k3 = st.columns(3)
        k1.metric("Harga penawaran (IPO)", rupiah(ph["offer"]))
        k2.metric(f"Perkiraan harga {bulan} bulan lagi", rupiah(hg["tengah"]), f"{u['tengah']:+.0%}".replace(".", ","))
        k3.metric("Rentang perkiraan (80%)", f"{rupiah(hg['bawah'])} - {rupiah(hg['atas'])}")
        ringkas = (f"Model memperkirakan harga saham sekitar **{rupiah(hg['tengah'])}** setelah {bulan} bulan, atau "
                   f"**{arah} {abs(u['tengah']):.0%}** dari harga penawaran {rupiah(ph['offer'])}. Rentang yang lebih "
                   f"realistis adalah **{rupiah(hg['bawah'])} sampai {rupiah(hg['atas'])}**.")
    else:
        k1, k2 = st.columns(2)
        k1.metric(f"Perkiraan perubahan harga {bulan} bulan lagi", f"{u['tengah']:+.0%}".replace(".", ","))
        k2.metric("Rentang perkiraan (80%)", f"{u['bawah']:+.0%} sampai {u['atas']:+.0%}".replace(".", ","))
        ringkas = (f"Model memperkirakan harga **{arah} sekitar {abs(u['tengah']):.0%}** dari harga penawaran dalam {bulan} bulan "
                   f"(rentang {u['bawah']:+.0%} sampai {u['atas']:+.0%}). Isi **Offer Price** di form untuk melihat "
                   "perkiraan dalam Rupiah.")
        st.warning("Harga penawaran belum terisi, sehingga perkiraan hanya ditampilkan sebagai persentase perubahan.")
    with st.container(border=True):
        st.markdown("**Cara membaca prediksi harga**\n\n" + ringkas + "\n\n"
                    "Rentang 80% berarti: pada data validasi, sekitar 8 dari 10 hasil aktual jatuh di dalam selisih sebesar "
                    "itu dari angka tengah. Sisanya, sekitar 2 dari 10, bisa lebih tinggi atau lebih rendah lagi.")
        m = ph.get("metrik")
        if m:
            st.markdown("**Seberapa bisa dipercaya?** Terbatas. Pada data uji, tebakan model rata-rata meleset sekitar "
                        f"**±{m['mae_model']:.0%}**, sedangkan menebak tanpa model (memakai nilai tengah IPO historis) meleset "
                        f"±{m['mae_dasar']:.0%}. Model hanya sedikit lebih baik daripada tebakan sederhana, jadi pakailah "
                        "**rentangnya sebagai gambaran, bukan satu angka sebagai target harga**.")
        st.caption("Prediksi harga memakai model terpisah dari peluang outperform di atas (Random Forest regresi, fitur "
                   "terstruktur). Keduanya bisa terlihat tidak searah, misalnya harga diperkirakan naik tetapi peluang "
                   "mengungguli IHSG rendah, karena IHSG juga bisa naik. Bukan rekomendasi investasi.")


def layar_hasil():
    header("Hasil prediksi", mono=True)
    hasil = st.session_state.get("hasil")
    if not hasil or "galat" in hasil:
        st.error(hasil["galat"] if hasil else "Belum ada hasil prediksi.")
        if st.button("← Kembali ke form", width="stretch"):
            st.session_state["halaman"] = "form"
            st.rerun()
        return

    ok = hasil["p"] >= 0.5
    bulan = int(hasil["h"]) // 30
    st.markdown(f"""
<div class="res-grid">
  <div class="res-main {'ok' if ok else 'no'}"><div class="lbl">PREDIKSI</div>
    <div class="big">{'OUTPERFORM' if ok else 'UNDERPERFORM'}</div>
    <div class="sub">Saham diprediksi {'mengungguli' if ok else 'tertinggal dari'} IHSG dalam {bulan} bulan</div></div>
  <div class="res-stat"><div class="lbl">Probabilitas</div><div class="val {'g' if ok else 'r'}">{hasil['p']:.1%}</div></div>
  <div class="res-stat"><div class="lbl">Model</div><div class="val">{html.escape(hasil['model'])}</div></div>
  <div class="res-stat"><div class="lbl">Skenario</div><div class="val">{SKENARIO_PENDEK[hasil['skenario']]}</div></div>
  <div class="res-stat"><div class="lbl">Horizon</div><div class="val">{hasil['h']} hari</div></div>
</div>""", unsafe_allow_html=True)

    if hasil.get("catatan"):
        st.info(hasil["catatan"])
    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
    grafik_historis(hasil)
    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
    bagian_harga(hasil)
    st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
    b1, b2 = st.columns(2)
    with b1:
        if st.button("↻ Prediksi ulang", key="ulang", width="stretch"):
            st.session_state["halaman"] = "form"
            st.rerun()
    with b2:
        st.download_button("⤓ Download laporan", data=laporan_html(hasil).encode("utf-8"),
                           file_name="laporan_prediksi_ipo.html", mime="text/html", width="stretch")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    if hasil["terisi"] < hasil["total"] / 2:
        st.warning(f"Hanya {hasil['terisi']} dari {hasil['total']} fitur model terisi nilai nyata; sisanya memakai "
                   "median data latih. Upload prospektus agar prediksi lebih informatif.")
    with st.expander("Detail model dan keterbatasan"):
        k = hasil["kinerja"]
        if hasil["skenario"] in SKENARIO_TEKS and hasil.get("info"):
            kata = {SEGMEN_RISIKO.get(PETA_RISIKO.get(sg), sg): len(str(t).split()) if t else 0
                    for sg, t in hasil["info"]["risiko"].items()}
            st.markdown("- Teks faktor risiko yang dipakai (jumlah kata per segmen): "
                        + ", ".join(f"{n} **{c:,}**" for n, c in kata.items()) + ".")
            with st.expander("Lihat isi teks faktor risiko yang dipakai"):
                tampil_risiko(hasil["info"]["risiko"], "hasil")
        st.markdown(
            f"- Fitur terisi nilai nyata: **{hasil['terisi']} dari {hasil['total']}** (sisanya median data latih).\n"
            f"- Kinerja model pada data uji (±50 emiten): ROC-AUC **{k['ROC-AUC']:.2f}**, akurasi "
            f"**{k['Accuracy']:.0%}**, F1 **{k['F1-Score']:.2f}**; ROC-AUC validasi silang **{k['CV AUC']:.2f}** "
            "(0,5 = tebakan acak).\n"
            "- Target: apakah saham mengungguli IHSG sejak harga penawaran, bukan kenaikan harga absolut.\n"
            "- Uji data hanya ±50 emiten sehingga ketidakpastiannya besar. Keluaran ini hasil penelitian "
            "akademik, **bukan rekomendasi investasi**.")
        df_f = pd.DataFrame([{"Variabel": NAMA_FITUR.get(k_, k_), "Nilai": v,
                              "Sumber": hasil["asal"].get(k_) or "median data latih"}
                             for k_, v in hasil["fitur"].items() if k_ in NAMA_FITUR and k_ in hasil["asal"]])
        st.dataframe(df_f.astype({"Nilai": "string"}), hide_index=True, width="stretch")


def halaman() -> None:
    """Pintu masuk menu Prediksi. Tahap terakhir (form atau hasil) diingat selama sesi."""
    {"hasil": layar_hasil}.get(st.session_state.get("halaman"), layar_form)()
