"""Halaman penjelasan: Beranda, Panduan Penggunaan, Cara Membaca Analisis, Informasi Tambahan.

Semua angka (jumlah emiten, model terbaik, base rate, contoh kasus) dihitung dari berkas di folder output/ dan data/,
bukan diketik manual, sehingga ikut berubah bila model atau data diperbarui.
"""
from __future__ import annotations

import html

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from umum import (DATA_DIR, FITUR_FORM, HORIZON, MAKS_MB, NAMA_HORIZON, SKENARIO_PENDEK, angka, data_asli,
                  evaluasi_uji, header, kinerja_model, muat_ringan, persen, pindah, ringkasan_penelitian, urut_model)

SVG = {
    "doc": '<svg viewBox="0 0 24 24"><path d="M7 3h7l5 5v13H7z M14 3v5h5 M9.5 13h6 M9.5 17h6"/></svg>',
    "net": '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="2.2"/><circle cx="5" cy="6" r="1.8"/><circle cx="19" cy="6" r="1.8"/><circle cx="5" cy="18" r="1.8"/><circle cx="19" cy="18" r="1.8"/><path d="M6.5 7l4 3.5 M17.5 7l-4 3.5 M6.5 17l4-3.5 M17.5 17l-4-3.5"/></svg>',
    "bar": '<svg viewBox="0 0 24 24"><rect x="4" y="4" width="16" height="16" rx="2"/><path d="M8 16v-4 M12 16V8 M16 16v-6"/></svg>',
}


def _kpi(label: str, nilai: str, ket: str) -> str:
    return (f'<div class="kpi"><div class="k-lbl">{html.escape(label)}</div><div class="k-val">{html.escape(nilai)}</div>'
            f'<div class="k-ket">{html.escape(ket)}</div></div>')


# ════════════════════════════════════════════════════════════════════
# 1. Beranda
# ════════════════════════════════════════════════════════════════════
def beranda() -> None:
    header("Beranda")
    rk = ringkasan_penelitian()
    st.markdown('<div class="hero"><h1>Prediksi kinerja saham IPO berbasis<br>prospektus</h1>'
                '<p>Unggah prospektus atau isi datanya, lalu lihat peluang saham tersebut mengungguli IHSG '
                'dalam 3 atau 6 bulan setelah listing. Hasilnya keluaran penelitian akademik, bukan rekomendasi investasi.</p></div>',
                unsafe_allow_html=True)
    _, tengah, _ = st.columns([1, 1.6, 1])
    with tengah:
        a, b = st.columns(2)
        if a.button("Mulai prediksi →", key="b_mulai", width="stretch"):
            pindah("prediksi")
        if b.button("Baca panduan", key="b_panduan", width="stretch"):
            pindah("panduan")
    st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)

    t90, t180 = rk["terbaik"]["90"], rk["terbaik"]["180"]
    st.markdown('<div class="kpi-grid">'
                + _kpi("Emiten dalam data", f"{rk['n']}", f"IPO BEI {rk['tahun_awal']} sampai {rk['tahun_akhir']}")
                + _kpi("Data uji", f"{rk['n_uji']}", f"{rk['n_latih']} emiten untuk melatih model")
                + _kpi("Model terbaik, 3 bulan", t90["model"],
                       f"ROC-AUC uji {angka(t90['auc_uji'])} · {SKENARIO_PENDEK[t90['skenario']]}")
                + _kpi("Model terbaik, 6 bulan", t180["model"],
                       f"ROC-AUC uji {angka(t180['auc_uji'])} · {SKENARIO_PENDEK[t180['skenario']]}")
                + '</div>', unsafe_allow_html=True)
    st.caption("Model terbaik dipilih dari ROC-AUC validasi silang tertinggi pada data latih. ROC-AUC uji dihitung pada "
               f"{rk['n_uji']} emiten sehingga ketidakpastiannya besar; 0,5 setara tebakan acak.")

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
    k1, k2, k3 = st.columns(3)
    for kol, ikon, judul, isi in [
        (k1, "doc", "Analisis prospektus", "Ekstraksi otomatis variabel penawaran, keuangan, dan penjamin dari PDF."),
        (k2, "net", "Model ML hybrid", "Data terstruktur dan teks faktor risiko, dibandingkan lewat validasi silang."),
        (k3, "bar", "Hasil prediksi", "Probabilitas outperform terhadap IHSG dan fitur yang paling berpengaruh."),
    ]:
        kol.markdown(f'<div class="feat">{SVG[ikon]}<b>{judul}</b><span>{isi}</span></div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# 2. Panduan Penggunaan
# ════════════════════════════════════════════════════════════════════
LANGKAH = [
    ("Unggah prospektus", f"Di menu Prediksi, unggah PDF prospektus (maksimal {MAKS_MB} MB). Tanpa PDF, data tetap bisa diisi "
                          "manual, minimal 3 baris data fundamental."),
    ("Tunggu pembacaan", "Prospektus tebal membutuhkan sekitar 1 sampai 3 menit. Setelah selesai, tabel data fundamental terisi otomatis."),
    ("Cek dan koreksi tabel data", "Klik sel pada kolom Nilai untuk mengubah. Nilai yang Anda ubah menimpa hasil ekstraksi; "
                                    "baris berstatus tidak terbaca boleh diisi sendiri."),
    ("Isi penjamin emisi", "Pilih atau ketik nama penjamin dan cek porsinya. Skor reputasi dihitung otomatis dari tabel skor underwriter."),
    ("Pilih sektor dan jangka waktu", "Sektor wajib dipilih manual karena tidak tercantum di prospektus. Jangka waktu: 3 bulan (H+90) atau 6 bulan (H+180)."),
    ("Jalankan prediksi", "Klik Jalankan prediksi, lalu baca hasilnya dengan bantuan menu Cara Membaca Analisis."),
]

FAQ = [
    ("Bagaimana kalau datanya tidak terbaca?",
     "Baris berstatus tidak terbaca bisa diisi manual di tabel. Baris yang tetap kosong diisi median data latih, dan jumlah fitur "
     "yang terisi nilai nyata ditampilkan di hasil. Jika kurang dari separuh fitur terisi, aplikasi memberi peringatan karena "
     "prediksinya kurang informatif."),
    ("PDF saya gagal dibaca. Apa yang harus dilakukan?",
     "Aplikasi menampilkan pesan galat dan Anda tetap bisa mengisi data secara manual. PDF yang berupa gambar hasil pindai tidak "
     "memiliki teks yang dapat dibaca, jadi gunakan PDF asli dari BEI bila ada."),
    ("Kenapa sektor harus dipilih sendiri?",
     "Sektor industri tidak tercantum di prospektus sehingga tidak bisa diekstraksi otomatis."),
    ("Kenapa hasil memakai skenario Terstruktur saja, padahal saya memilih yang memakai teks?",
     "Skenario berbasis teks butuh bab Faktor Risiko dari prospektus. Untuk skenario baku (Terstruktur + teks), bila prospektus "
     "tidak diunggah atau bab tersebut tidak ditemukan, aplikasi otomatis beralih ke Terstruktur saja dan memberi catatan di "
     "halaman hasil. Skenario Teks saja tidak bisa dijalankan tanpa prospektus dan menampilkan pesan galat."),
    ("Apakah PDF saya disimpan?",
     "Aplikasi hanya menyimpan PDF sementara selama dibaca lalu menghapusnya; isi dan hasilnya hanya ada di sesi browser Anda."),
    ("Apakah ini rekomendasi investasi?",
     "Bukan. Ini keluaran penelitian akademik dengan data uji kecil; lihat keterbatasannya di menu Cara Membaca Analisis."),
]


def panduan() -> None:
    header("Panduan Penggunaan")
    st.markdown("### Langkah memakai aplikasi")
    isi = "".join(f'<div class="langkah"><div class="no">{i}</div><div><b>{html.escape(j)}</b><br>'
                  f'<span>{html.escape(t)}</span></div></div>' for i, (j, t) in enumerate(LANGKAH, 1))
    st.markdown(isi, unsafe_allow_html=True)
    if st.button("Buka menu Prediksi →", key="p_buka"):
        pindah("prediksi")

    st.markdown("### Format dan waktu tunggu")
    st.markdown(f"- **Format:** PDF, maksimal {MAKS_MB} MB.\n"
                "- **Waktu:** PDF tebal sekitar 1 sampai 3 menit; PDF yang sudah pernah dibaca dalam sesi yang sama lebih cepat karena disimpan sementara.\n"
                "- **Tanpa PDF:** isi tabel secara manual dan prediksi langsung dihitung dalam hitungan detik.")

    st.markdown("### Arti tiap kolom pada form")
    baris = [{"Kolom": label, "Arti": ket} for _, label, ket in FITUR_FORM]
    baris += [{"Kolom": "Penjamin emisi", "Arti": "Nama dan porsi penjamin; menghasilkan skor reputasi gabungan dan jumlah penjamin."},
              {"Kolom": "Ada lock-up pemegang saham", "Arti": "Centang bila pemegang saham lama dilarang menjual sahamnya selama periode tertentu."},
              {"Kolom": "Sektor industri", "Arti": "Dipilih manual dari daftar sektor yang dikenal model."},
              {"Kolom": "Jangka Waktu", "Arti": "3 Bulan = H+90 hari bursa, 6 Bulan = H+180 hari bursa sejak listing."}]
    st.dataframe(pd.DataFrame(baris), hide_index=True, width="stretch", height=40 + 35 * len(baris))
    st.caption("Istilah teknis lain ada di menu Informasi Tambahan, bagian Kamus istilah.")

    st.markdown("### Pertanyaan yang sering muncul")
    for tanya, jawab in FAQ:
        with st.expander(tanya):
            st.write(jawab)


# ════════════════════════════════════════════════════════════════════
# 3. Cara Membaca Analisis
# ════════════════════════════════════════════════════════════════════
def _contoh_kasus(h: str = "180") -> tuple[pd.DataFrame, str] | None:
    """Dua emiten nyata dari data uji: tebakan yakin yang benar dan tebakan yakin yang salah."""
    ev = evaluasi_uji()
    if not ev["cocok"]:
        return None
    nama = urut_model(h, "num")[0]
    df = data_asli(["Ticker Code", "Listing Date", f"label_{h}", f"log_excess_return_{h}"])
    t = df.loc[ev["test"]].copy()
    t["p"] = ev["prob"][(h, nama)][ev["test"]]
    t["tebak"] = (t["p"] >= 0.5).astype(int)
    t["yakin"] = (t["p"] - 0.5).abs()
    t["rel"] = (np.exp(t[f"log_excess_return_{h}"]) - 1) * 100
    benar = t[t["tebak"] == t[f"label_{h}"]].sort_values("yakin", ascending=False).head(1)
    salah = t[t["tebak"] != t[f"label_{h}"]].sort_values("yakin", ascending=False).head(1)
    if benar.empty or salah.empty:
        return None
    out = pd.concat([benar.assign(Hasil="Tepat"), salah.assign(Hasil="Meleset")])
    return pd.DataFrame({
        "Hasil tebakan": out["Hasil"], "Emiten": out["Ticker Code"], "Listing": out["Listing Date"].dt.strftime("%d-%m-%Y"),
        "Peluang outperform menurut model": out["p"].map(lambda x: persen(x)),
        "Kenyataan": np.where(out[f"label_{h}"] == 1, "Outperform", "Underperform"),
        "Return relatif IHSG": out["rel"].map(lambda x: f"{x:+.0f}%".replace(".", ","))}), nama


def cara_membaca() -> None:
    header("Cara Membaca Analisis")
    rk = ringkasan_penelitian()
    b90, b180 = rk["base"]["90"], rk["base"]["180"]
    t90, t180 = rk["terbaik"]["90"], rk["terbaik"]["180"]

    st.markdown("### 1. Arti probabilitas")
    st.markdown(f"Angka probabilitas adalah **peluang menurut model bahwa saham mengungguli IHSG** pada horizon yang dipilih. "
                f"Misalnya 88% berarti model menaksir peluang 88%, **bukan** kepastian dan bukan perkiraan keuntungan 88%.")
    st.markdown(f"Bacalah bersama **base rate**, yaitu porsi IPO historis yang memang mengungguli IHSG: "
                f"**{persen(b90)}** pada 3 bulan dan **{persen(b180)}** pada 6 bulan di {rk['n']} emiten. Probabilitas yang jauh di atas "
                "base rate berarti model melihat tanda yang lebih baik dari IPO rata-rata; probabilitas di dekat base rate berarti "
                "model tidak punya banyak yang dikatakan.")
    st.info("Kalibrasi probabilitas (apakah 80% benar-benar terjadi pada sekitar 80% kasus) tidak dievaluasi pada ringkasan model "
            "yang dipakai aplikasi ini. Anggap probabilitas sebagai urutan peringkat, bukan angka yang presisi.")

    st.markdown("### 2. ROC-AUC")
    st.markdown("ROC-AUC mengukur seberapa baik model memberi skor lebih tinggi kepada emiten yang benar-benar mengungguli IHSG "
                "dibanding yang tidak. **0,5 = tebakan acak, 1,0 = sempurna.** "
                f"Model terbaik pada penelitian ini mencapai {angka(t90['auc_uji'])} (3 bulan) dan {angka(t180['auc_uji'])} (6 bulan) "
                f"pada data uji, tetapi data uji hanya {rk['n_uji']} emiten, sehingga selisih kecil antar model tidak bisa dianggap bermakna.")

    st.markdown("### 3. Grafik historis di halaman hasil")
    st.markdown("- Batang menunjukkan sebaran return relatif terhadap IHSG pada IPO historis: **hijau** untuk yang mengungguli, **merah** untuk yang tertinggal.\n"
                "- Garis putus-putus adalah titik impas terhadap IHSG (sama dengan pasar).\n"
                "- Tiga angka di atasnya: porsi outperform seluruh IPO, porsi pada sektor yang sama (bila ada minimal 5 emiten), dan prediksi model untuk emiten Anda.\n"
                "- Di bawah grafik ada penjelasan cara membaca dan insight yang bisa diambil, dihitung dari data historis.\n"
                "- Grafik ini gambaran hasil IPO sebelumnya, **bukan perkiraan harga** emiten Anda.")

    st.markdown("### 4. Prediksi harga")
    st.markdown("Di bawah konteks historis, aplikasi juga memperkirakan **harga saham setelah 3 atau 6 bulan**, sesuai jangka waktu "
                "yang Anda pilih. Perkiraan ini berasal dari model regresi (Random Forest) yang memprediksi perubahan harga dari "
                "harga penawaran, lalu diubah menjadi harga dalam Rupiah.")
    st.markdown("- **Baca rentangnya, bukan satu angka.** Rentang 80% berarti sekitar 8 dari 10 hasil historis jatuh di dalam selisih sebesar itu.\n"
                "- **Akurasinya terbatas.** Pada data uji, model regresi hanya sedikit lebih baik daripada menebak memakai nilai tengah IPO historis.\n"
                "- **Terpisah dari peluang outperform.** Harga bisa diperkirakan naik walau peluang mengungguli IHSG rendah, karena IHSG juga bisa naik.\n"
                "- Tanpa **Offer Price** yang terisi, perkiraan hanya tampil sebagai persentase perubahan.")

    st.markdown("### 5. Contoh kasus dari data uji")
    contoh = _contoh_kasus("180")
    if contoh is None:
        st.caption("Contoh kasus tidak ditampilkan karena evaluasi ulang pada data uji tidak cocok dengan tabel metrik "
                   "(data atau model kemungkinan berubah).")
    else:
        tabel, nama = contoh
        st.caption(f"Dua emiten dari data uji (6 bulan, skenario Terstruktur, model {nama}): satu tebakan yakin yang tepat dan "
                   "satu tebakan yakin yang meleset. Model bisa salah, termasuk saat ia terlihat yakin.")
        st.dataframe(tabel, hide_index=True, width="stretch")

    st.markdown("### 6. Keterbatasan")
    st.markdown(f"- **Data kecil:** {rk['n']} emiten, hanya {rk['n_uji']} untuk uji. Selisih antar model kecil dan belum diuji signifikansinya pada data ini.\n"
                "- **Dua horizon:** model aplikasi ini hanya dilatih untuk 3 bulan (H+90) dan 6 bulan (H+180).\n"
                "- **Teks belum terbukti membantu:** penambahan teks faktor risiko belum terbukti memperbaiki ROC-AUC dibanding fitur terstruktur saja.\n"
                "- **Model belum di-tuning:** semua model memakai parameter bawaan.\n"
                "- **Ekstraksi PDF bisa keliru:** terutama DER dan ROA, yang dapat berbeda karena periode laporan keuangan; periksa nilainya sebelum prediksi.\n"
                "- **Target relatif:** peluang outperform diukur terhadap IHSG sejak harga penawaran, bukan kenaikan harga absolut.\n"
                "- **Prediksi harga sangat tidak pasti:** model regresi hanya sedikit lebih baik daripada tebakan sederhana pada data uji; gunakan rentangnya.\n"
                "- **Bukan rekomendasi investasi.**")


# ════════════════════════════════════════════════════════════════════
# 4. Informasi Tambahan
# ════════════════════════════════════════════════════════════════════
KAMUS = [
    ("DER (Debt to Equity Ratio)", "Total liabilitas dibagi total ekuitas. Makin tinggi, makin besar porsi pendanaan dari utang."),
    ("ROA (Return on Assets)", "Laba bersih dibagi total aset; ukuran seberapa efisien aset menghasilkan laba. Dalam aplikasi dinyatakan sebagai rasio (0,03 = 3%)."),
    ("Owner retention", "Porsi kepemilikan pemegang saham lama setelah IPO (0,9 = 90%)."),
    ("Share offered", "Porsi saham yang dilepas ke publik terhadap total saham setelah IPO."),
    ("Harga penawaran (Offer Price)", "Harga per lembar saham saat IPO; titik awal penghitungan return pada penelitian ini."),
    ("Underwriter (penjamin emisi)", "Perusahaan sekuritas yang menjamin dan memasarkan penawaran saham. Reputasinya dipakai sebagai fitur."),
    ("Skor reputasi underwriter", "Skor numerik tiap penjamin dari tabel data/skor_underwriter.csv; bila ada beberapa penjamin, dirata-ratakan menurut porsi."),
    ("Jumlah penjamin (UW Tally)", "Banyaknya penjamin emisi dalam satu IPO."),
    ("Market Activity", "Jumlah IPO pada bulan pencatatan yang sama; gambaran keramaian pasar IPO saat itu."),
    ("Lock-up pemegang saham", "Larangan menjual saham bagi pemegang saham tertentu selama periode setelah IPO."),
    ("Use of proceeds (uop_pct_*)", "Rencana penggunaan dana IPO dalam persen: belanja modal, bayar utang, modal kerja, akuisisi, ekspansi."),
    ("Faktor risiko", "Bab prospektus yang memuat risiko usaha, risiko investasi dan saham, risiko umum, dan risiko utama; sumber fitur teks."),
    ("TF-IDF", "Cara mengubah teks menjadi angka: kata yang sering muncul di satu dokumen tetapi jarang di dokumen lain mendapat bobot tinggi."),
    ("IHSG", "Indeks Harga Saham Gabungan, patokan pergerakan pasar saham Indonesia."),
    ("Outperform / Underperform", "Outperform: return saham sejak harga penawaran melebihi return IHSG pada periode yang sama. Underperform: sebaliknya."),
    ("Log excess return", "Selisih log return saham dengan log return IHSG; positif berarti mengungguli IHSG. Dasar label outperform."),
    ("Horizon (H+90, H+180)", "Hari bursa ke-90 dan ke-180 sejak listing, tempat harga penutupan diukur."),
    ("Base rate", "Porsi IPO historis yang mengungguli IHSG; patokan pembanding untuk probabilitas model."),
    ("ROC-AUC", "Kemampuan model mengurutkan emiten yang outperform di atas yang underperform. 0,5 acak, 1 sempurna."),
    ("CV AUC", "ROC-AUC dari validasi silang pada data latih; dipakai untuk memilih model terbaik."),
    ("Papan pencatatan", "Papan BEI tempat saham dicatatkan: Utama, Pengembangan, Akselerasi, atau Ekonomi Baru."),
]

REFERENSI = [
    "Dhanendra, N. W. D. A., Pramudita, A., Halim, K., & Arifin, Y. (2025). Predicting Potential Returns of Indonesian IPOs on Their Listing Day Using Machine Learning Techniques. Bina Nusantara University.",
    "Meidiaswati, H., Kusumawati, N. D., Akbar, R. F., & Sasikirono, N. (2026). Risk Management Strategies in Prospectus Disclosures and Initial Stock Returns in Indonesia. Quality Access to Success, 27(164).",
    "Sharpe, J., & Decker, K. (2022). Prospectus Language and IPO Performance. Proceedings of the Fourth Workshop on Financial Technology and Natural Language Processing (FinNLP). https://doi.org/10.18653/V1/2022.FINNLP-1.21",
    "Tsui, C. S. K., & Li, K. C. K. (2021). IPO Underpricing and Prospectus Readability: A Machine Learning Approach. Journal of Applied Business and Economics, 23(7). https://doi.org/10.33423/jabe.v23i7.4867",
    "Ni, S. (2023). Predicting IPO Performance from Prospectus Sentiment. BCP Business & Management, 38, 3063-3075. https://doi.org/10.54691/bcpbm.v38i.4237",
    "Supsermpol, P., Huynh, V. N., Thajchayapong, S., Suppakitjarak, N., & Chiadamrong, N. (2025). Predicting post-IPO financial performance: a hybrid approach using logistic regression and decision trees. Journal of Asian Business and Economic Studies, 52-65.",
    "Alahmadi, M. F., & Yilmaz, M. T. (2025). Prediction of IPO performance from prospectus using multinomial logistic regression, a machine learning model. Data Science in Finance and Economics, 5(1), 105-135. https://doi.org/10.3934/dsfe.2025006",
]


def _tab_kamus() -> None:
    cari = st.text_input("Cari istilah", placeholder="mis. DER, base rate, lock-up", key="kamus_cari")
    q = cari.strip().lower()
    hasil = [(i, a) for i, a in KAMUS if not q or q in i.lower() or q in a.lower()]
    if not hasil:
        st.info("Tidak ada istilah yang cocok. Coba kata kunci lain.")
    for istilah, arti in hasil:
        st.markdown(f'<div class="istilah"><b>{html.escape(istilah)}</b><span>{html.escape(arti)}</span></div>', unsafe_allow_html=True)


def _tab_kalender() -> None:
    kal = pd.read_csv(DATA_DIR / "kalender_ipo.csv")
    kal["tanggal_listing"] = pd.to_datetime(kal["tanggal_listing"])
    kal["bulan"] = kal["tanggal_listing"].dt.to_period("M").dt.to_timestamp()
    st.caption("Jumlah IPO per bulan pencatatan. Angka inilah yang menjadi fitur Market Activity (jumlah IPO pada bulan listing yang sama).")
    per = kal.groupby("bulan").size().reset_index(name="Jumlah IPO")
    st.altair_chart(alt.Chart(per).mark_bar(color="#8a2c22").encode(
        x=alt.X("bulan:T", title="Bulan listing"), y=alt.Y("Jumlah IPO:Q"),
        tooltip=[alt.Tooltip("bulan:T", format="%b %Y"), "Jumlah IPO"]).properties(height=220), width="stretch")
    tahun = sorted(kal["tanggal_listing"].dt.year.unique())
    pilih = st.multiselect("Tahun", tahun, default=tahun, key="kal_tahun")
    t = kal[kal["tanggal_listing"].dt.year.isin(pilih)].sort_values("tanggal_listing", ascending=False)
    tampil = pd.DataFrame({"Kode": t["ticker"], "Tanggal listing": t["tanggal_listing"].dt.strftime("%d-%m-%Y"),
                           "Nilai penawaran (Rp miliar)": (t["nilai_penawaran"] / 1e9).round(1), "Penjamin emisi": t["penjamin"]})
    st.dataframe(tampil, hide_index=True, width="stretch", height=380)


def _tab_skor_uw() -> None:
    skor = pd.read_csv(DATA_DIR / "skor_underwriter.csv").sort_values("skor", ascending=False).reset_index(drop=True)
    st.caption("Skor reputasi tiap penjamin emisi yang dipakai sebagai fitur. Skor lebih tinggi menandakan reputasi yang lebih kuat "
               "menurut tabel acuan penelitian; penjamin yang tidak ada di tabel diberi skor median.")
    n = st.slider("Tampilkan penjamin teratas", 5, len(skor), min(15, len(skor)), key="uw_n")
    atas = skor.head(n)
    st.altair_chart(alt.Chart(atas).mark_bar(color="#6ea8ff").encode(
        y=alt.Y("nama:N", sort="-x", title=None), x=alt.X("skor:Q", title="Skor reputasi"),
        tooltip=["nama", alt.Tooltip("skor:Q", format=".2f")]).properties(height=22 * n + 30), width="stretch")
    tampil = skor.rename(columns={"kode": "Kode", "nama": "Penjamin emisi", "skor": "Skor"})
    tampil.insert(0, "Peringkat", range(1, len(tampil) + 1))
    with st.expander("Tabel lengkap"):
        st.dataframe(tampil, hide_index=True, width="stretch", height=360)


def _tab_metodologi() -> None:
    rk = ringkasan_penelitian()
    st.markdown("#### Definisi target")
    st.markdown("Target adalah apakah saham mengungguli IHSG sejak harga penawaran (excess return terhadap pasar), diukur pada hari bursa ke-90 dan ke-180.")
    st.latex(r"\text{LER}_h=\ln\!\left(\frac{P_h}{P_0}\right)-\ln\!\left(\frac{\mathrm{IHSG}_h}{\mathrm{IHSG}_0}\right)")
    st.markdown("**Outperform** bila LER positif, **underperform** bila nol atau negatif. $P_0$ adalah harga penawaran.")
    st.markdown("#### Pembagian data")
    st.markdown(f"- {rk['n']} emiten dibagi acak menjadi **{rk['n_latih']} latih dan {rk['n_uji']} uji** (80:20), distratifikasi pada label 3 bulan, dengan benih acak tetap.\n"
                "- Penskala, encoder kategori, dan TF-IDF hanya dilatih pada data latih, lalu diterapkan ke data uji, supaya data uji tidak bocor ke proses pelatihan.\n"
                "- Satu pembagian dipakai untuk semua horizon dan semua model, sehingga hasilnya sebanding.")
    st.markdown("#### Model dan skenario")
    st.markdown("- Lima model: Logistic Regression, Random Forest, XGBoost, AdaBoost, CatBoost (parameter bawaan, tanpa tuning).\n"
                "- Tiga skenario fitur: Terstruktur, Teks (TF-IDF faktor risiko), dan gabungan keduanya.\n"
                "- Model terbaik untuk tiap horizon dan skenario dipilih dari ROC-AUC validasi silang pada data latih.\n"
                "- **Prediksi harga:** model regresi (Random Forest, fitur terstruktur) memprediksi ln(harga setelah H hari / harga penawaran), "
                "dilatih pada pembagian data yang sama. Rentang 80% diambil dari sebaran selisih (aktual - prediksi) pada validasi silang data latih.")


def _tab_sumber() -> None:
    st.markdown("#### Sumber data")
    st.markdown("- **Prospektus IPO:** dokumen PDF dari Bursa Efek Indonesia (idx.co.id); variabel penawaran, keuangan, penjamin emisi, penggunaan dana, dan teks faktor risiko diekstraksi dari sini.\n"
                "- **Harga saham:** harga historis hari bursa ke-90 dan ke-180 dari Google Finance, hanya untuk membentuk target.\n"
                "- **Skor underwriter dan kalender IPO:** berkas data/skor_underwriter.csv dan data/kalender_ipo.csv.\n"
                "- **Indikator Teknikal:** harga harian diambil dari Yahoo Finance saat Anda memintanya; data ini tidak dipakai untuk melatih model.")
    st.markdown("#### Disclaimer")
    st.warning("Aplikasi ini keluaran penelitian akademik. Hasilnya bukan rekomendasi investasi, tidak menjamin hasil di masa depan, "
               "dan diuji hanya pada puluhan emiten. Keputusan investasi menjadi tanggung jawab pengguna.")
    st.markdown("#### Referensi")
    for r in REFERENSI:
        st.markdown(f"- {r}")
    st.caption("Daftar lengkap ada pada laporan skripsi.")


def info_tambahan() -> None:
    header("Informasi Tambahan")
    t1, t2, t3, t4, t5 = st.tabs(["Kamus istilah", "Kalender IPO", "Skor underwriter", "Metodologi singkat", "Sumber dan referensi"])
    with t1:
        _tab_kamus()
    with t2:
        _tab_kalender()
    with t3:
        _tab_skor_uw()
    with t4:
        _tab_metodologi()
    with t5:
        _tab_sumber()
