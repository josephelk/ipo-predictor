"""Menu Indikator Teknikal: informasi pasar untuk saham yang sudah tercatat.

Harga harian diambil dari Yahoo Finance (kode saham + ".JK", IHSG = ^JKSE) saat pengguna menekan tombol, atau dari CSV
yang diunggah. Data ini hanya untuk informasi dan TIDAK dipakai melatih atau menjalankan model prediksi.
Bagian "Prediksi vs realisasi" memakai data penelitian (tanpa internet).
"""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from umum import (NAMA_HORIZON, angka, data_asli, evaluasi_uji, header, persen, urut_model)

SIMBOL_IHSG = "^JKSE"
RENTANG = {"Sejak listing": None, "1 tahun": 365, "6 bulan": 183, "3 bulan": 92}
WARNA = {"Close": "#f2f2f2", "MA20": "#6ea8ff", "MA50": "#f0b429", "MA200": "#e9a39b"}
CATATAN = ("Indikator teknikal hanya merangkum pergerakan harga yang sudah terjadi. Ia bukan prediksi, tidak dipakai oleh model "
           "di menu Prediksi, dan bukan rekomendasi investasi.")


# ════════════════════════════════════════════════════════════════════
# Data harga
# ════════════════════════════════════════════════════════════════════
def rapikan(d: pd.DataFrame) -> pd.DataFrame:
    """Samakan bentuk data harga: indeks tanggal (tanpa zona waktu), kolom Close dan Volume, urut naik."""
    d = d.copy()
    if not isinstance(d.index, pd.DatetimeIndex):
        d.index = pd.to_datetime(d.index)
    if d.index.tz is not None:
        d.index = d.index.tz_localize(None)
    d.index = d.index.normalize()
    d.index.name = "Date"
    d = d.rename(columns={c: str(c).strip().title() for c in d.columns})
    if "Close" not in d.columns:
        raise ValueError("kolom Close tidak ditemukan")
    d["Close"] = pd.to_numeric(d["Close"], errors="coerce")
    d["Volume"] = pd.to_numeric(d["Volume"], errors="coerce") if "Volume" in d.columns else np.nan
    d = d[~d.index.duplicated(keep="last")].sort_index()
    return d.dropna(subset=["Close"])[["Close", "Volume"]]


@st.cache_data(ttl=3600, show_spinner=False, max_entries=32)
def unduh_harga(simbol: str, mulai: str, akhir: str) -> pd.DataFrame:
    """Harga harian dari Yahoo Finance (disesuaikan split dan dividen). Memunculkan ValueError bila gagal."""
    try:
        import yfinance as yf
    except ImportError as e:
        raise ValueError("library yfinance belum terpasang (jalankan: pip install yfinance)") from e
    try:
        d = yf.Ticker(simbol).history(start=mulai, end=akhir, auto_adjust=True)
    except Exception as e:                                   # jaringan, pemblokiran, kode tidak dikenal
        raise ValueError(f"{type(e).__name__}: {e}") from e
    if d is None or d.empty:
        raise ValueError(f"tidak ada data untuk {simbol} pada periode ini (koneksi gagal atau kode tidak tersedia di Yahoo Finance)")
    return rapikan(d)


def baca_csv(berkas) -> pd.DataFrame:
    """CSV bergaya unduhan Yahoo Finance: kolom Date (atau Tanggal), Close, dan opsional Volume."""
    d = pd.read_csv(berkas)
    d.columns = [str(c).strip() for c in d.columns]
    kecil = {c.lower().rstrip("."): c for c in d.columns}
    if "close" in kecil and "adj close" in kecil:            # unduhan Yahoo memuat keduanya; pakai Close
        d = d.drop(columns=[kecil["adj close"]])
    pemetaan = {"tanggal": "Date", "date": "Date", "time": "Date", "penutupan": "Close", "close": "Close",
                "adj close": "Close", "volume": "Volume", "vol": "Volume"}
    d = d.rename(columns={c: pemetaan.get(c.lower().rstrip("."), c) for c in d.columns})
    if "Date" not in d.columns:
        raise ValueError("kolom tanggal (Date atau Tanggal) tidak ditemukan")
    d["Date"] = pd.to_datetime(d["Date"], errors="coerce", utc=True).dt.tz_localize(None)
    return rapikan(d.dropna(subset=["Date"]).set_index("Date"))


# ════════════════════════════════════════════════════════════════════
# Indikator
# ════════════════════════════════════════════════════════════════════
def _rsi_wilder(harga: np.ndarray, n: int = 14) -> np.ndarray:
    """RSI metode Wilder: rata-rata awal = rata-rata sederhana n perubahan pertama, lalu diperhalus (n-1)/n."""
    hasil = np.full(len(harga), np.nan)
    if len(harga) <= n:
        return hasil
    selisih = np.diff(harga)
    naik, turun = np.maximum(selisih, 0.0), np.maximum(-selisih, 0.0)
    rn, rt = naik[:n].mean(), turun[:n].mean()
    for i in range(n, len(harga)):
        if i > n:
            rn = (rn * (n - 1) + naik[i - 1]) / n
            rt = (rt * (n - 1) + turun[i - 1]) / n
        hasil[i] = 50.0 if (rt == 0 and rn == 0) else (100.0 if rt == 0 else 100 - 100 / (1 + rn / rt))
    return hasil


def hitung_indikator(d: pd.DataFrame) -> pd.DataFrame:
    """MA20/50/200 (rata-rata sederhana), RSI 14 (Wilder), dan MACD (12, 26, 9) dari harga penutupan."""
    c, o = d["Close"], d.copy()
    for n in (20, 50, 200):
        o[f"MA{n}"] = c.rolling(n).mean()
    o["RSI"] = _rsi_wilder(c.to_numpy(dtype=float), 14)
    o["MACD"] = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    o["Signal"] = o["MACD"].ewm(span=9, adjust=False).mean()
    o["Histogram"] = o["MACD"] - o["Signal"]
    return o


def tanggal_hari_bursa(d: pd.DataFrame, listing: pd.Timestamp, n: int):
    """Tanggal hari bursa ke-n sejak listing (hari listing = 0), atau None bila datanya belum sepanjang itu."""
    sesudah = d.index[d.index >= listing]
    return sesudah[n] if len(sesudah) > n else None


# ════════════════════════════════════════════════════════════════════
# Grafik
# ════════════════════════════════════════════════════════════════════
def _penanda(d: pd.DataFrame, listing: pd.Timestamp, batas_kiri: pd.Timestamp):
    isi = []
    for h in ("90", "180"):
        t = tanggal_hari_bursa(d, listing, int(h))
        if t is not None and t >= batas_kiri:
            isi.append({"Date": t, "Label": NAMA_HORIZON[h]})
    if not isi:
        return None
    pd_ = pd.DataFrame(isi)
    garis = alt.Chart(pd_).mark_rule(color="#9b9b9b", strokeDash=[4, 4]).encode(x="Date:T")
    teks = alt.Chart(pd_).mark_text(align="left", dx=4, color="#9b9b9b", fontSize=11).encode(
        x="Date:T", y=alt.value(10), text="Label:N")
    return garis + teks


def _grafik_harga(d: pd.DataFrame, penanda) -> alt.Chart:
    m = d.reset_index().melt("Date", ["Close", "MA20", "MA50", "MA200"], var_name="Seri", value_name="Harga").dropna()
    g = alt.Chart(m).mark_line().encode(
        x=alt.X("Date:T", title=None), y=alt.Y("Harga:Q", scale=alt.Scale(zero=False), title="Harga (Rp)"),
        color=alt.Color("Seri:N", scale=alt.Scale(domain=list(WARNA), range=list(WARNA.values())),
                        legend=alt.Legend(orient="top", title=None)),
        strokeWidth=alt.condition(alt.datum.Seri == "Close", alt.value(2.2), alt.value(1.2)),
        tooltip=[alt.Tooltip("Date:T", format="%d-%m-%Y"), "Seri", alt.Tooltip("Harga:Q", format=",.0f")])
    return (g + penanda if penanda is not None else g).properties(height=300)


def _grafik_ihsg(e: pd.DataFrame, i: pd.DataFrame) -> tuple[alt.Chart, float, float] | None:
    gab = pd.concat([e["Close"].rename("Emiten"), i["Close"].rename("IHSG")], axis=1).dropna()
    if len(gab) < 2:
        return None
    rel = (gab / gab.iloc[0] * 100).reset_index().melt("Date", var_name="Seri", value_name="Indeks")
    g = alt.Chart(rel).mark_line().encode(
        x=alt.X("Date:T", title=None), y=alt.Y("Indeks:Q", scale=alt.Scale(zero=False), title="Indeks (awal rentang = 100)"),
        color=alt.Color("Seri:N", scale=alt.Scale(domain=["Emiten", "IHSG"], range=["#f2f2f2", "#6ea8ff"]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("Date:T", format="%d-%m-%Y"), "Seri", alt.Tooltip("Indeks:Q", format=".1f")]).properties(height=300)
    garis100 = alt.Chart(pd.DataFrame({"y": [100]})).mark_rule(color="#555", strokeDash=[3, 3]).encode(y="y:Q")
    akhir = gab.iloc[-1] / gab.iloc[0] * 100 - 100
    return g + garis100, float(akhir["Emiten"]), float(akhir["IHSG"])


def _grafik_rsi(d: pd.DataFrame) -> alt.Chart:
    r = d.reset_index()[["Date", "RSI"]].dropna()
    g = alt.Chart(r).mark_line(color="#b6dcae").encode(
        x=alt.X("Date:T", title=None), y=alt.Y("RSI:Q", scale=alt.Scale(domain=[0, 100]), title="RSI (14)"),
        tooltip=[alt.Tooltip("Date:T", format="%d-%m-%Y"), alt.Tooltip("RSI:Q", format=".1f")])
    batas = alt.Chart(pd.DataFrame({"y": [30, 70]})).mark_rule(color="#9b9b9b", strokeDash=[4, 4]).encode(y="y:Q")
    return (g + batas).properties(height=240)


def _grafik_macd(d: pd.DataFrame) -> alt.Chart:
    r = d.reset_index()[["Date", "MACD", "Signal", "Histogram"]].dropna()
    hist = alt.Chart(r).mark_bar(opacity=0.55).encode(
        x=alt.X("Date:T", title=None), y=alt.Y("Histogram:Q", title="MACD"),
        color=alt.condition(alt.datum.Histogram >= 0, alt.value("#b6dcae"), alt.value("#e9a39b")),
        tooltip=[alt.Tooltip("Date:T", format="%d-%m-%Y"), alt.Tooltip("Histogram:Q", format=".2f")])
    gar = r.melt("Date", ["MACD", "Signal"], var_name="Garis", value_name="Nilai")
    lines = alt.Chart(gar).mark_line().encode(
        x="Date:T", y="Nilai:Q", color=alt.Color("Garis:N", scale=alt.Scale(domain=["MACD", "Signal"], range=["#6ea8ff", "#f0b429"]),
                                                  legend=alt.Legend(orient="top", title=None)))
    return (hist + lines).properties(height=260)


def _grafik_volume(d: pd.DataFrame) -> alt.Chart:
    v = d.reset_index()[["Date", "Volume"]].dropna()
    return alt.Chart(v).mark_bar(color="#6ea8ff", opacity=0.8).encode(
        x=alt.X("Date:T", title=None), y=alt.Y("Volume:Q", title="Volume (lembar)"),
        tooltip=[alt.Tooltip("Date:T", format="%d-%m-%Y"), alt.Tooltip("Volume:Q", format=",.0f")]).properties(height=220)


# ════════════════════════════════════════════════════════════════════
# Halaman
# ════════════════════════════════════════════════════════════════════
def _sumber_data(kode: str, listing: pd.Timestamp) -> None:
    """Tombol ambil data + jalur cadangan unggah CSV; hasil disimpan di session_state['_tek']."""
    with st.expander("Sumber data harga"):
        mode = st.radio("Sumber", ["Yahoo Finance (otomatis)", "Unggah CSV"], horizontal=True, key="tek_sumber",
                        help="Yahoo Finance memerlukan koneksi internet. Bila gagal, unggah CSV hasil unduhan harga historis.")
        csv_e = csv_i = None
        if mode == "Unggah CSV":
            st.caption("Format seperti unduhan Yahoo Finance: kolom Date (atau Tanggal), Close, dan opsional Volume.")
            csv_e = st.file_uploader(f"CSV harga {kode}", type="csv", key="tek_csv_e")
            csv_i = st.file_uploader("CSV IHSG (opsional, untuk grafik pembanding)", type="csv", key="tek_csv_i")
    if not st.button("Ambil data harga", key="tek_ambil"):
        return
    try:
        if mode == "Unggah CSV":
            if csv_e is None:
                st.error("Unggah CSV harga emiten terlebih dahulu.")
                return
            e = baca_csv(csv_e)
            i = baca_csv(csv_i) if csv_i is not None else None
            sumber = "CSV unggahan"
        else:
            mulai = (listing - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
            akhir = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            with st.spinner("Mengambil data dari Yahoo Finance..."):
                e = unduh_harga(f"{kode}.JK", mulai, akhir)
                try:
                    i = unduh_harga(SIMBOL_IHSG, mulai, akhir)
                except ValueError:
                    i = None
            sumber = "Yahoo Finance"
    except ValueError as err:
        st.error("Data harga tidak bisa diambil. Periksa koneksi internet lalu coba lagi, atau pilih Unggah CSV pada "
                 "Sumber data harga.")
        st.caption(f"Detail teknis: {str(err).rstrip('.')}")
        return
    st.session_state["_tek"] = {"kode": kode, "emiten": e, "ihsg": i, "sumber": sumber}


def _tampil_grafik(kode: str, listing: pd.Timestamp) -> None:
    s = st.session_state.get("_tek")
    if not s or s["kode"] != kode:
        st.info("Pilih emiten, lalu tekan Ambil data harga untuk menampilkan grafik dan indikator.")
        return
    rentang = st.radio("Rentang tampilan", list(RENTANG), horizontal=True, key="tek_rentang")
    e = hitung_indikator(s["emiten"])
    akhir = e.index.max()
    kiri = e.index.min() if RENTANG[rentang] is None else max(e.index.min(), akhir - pd.Timedelta(days=RENTANG[rentang]))
    tampil = e[e.index >= kiri]
    if len(tampil) < 2:
        st.warning("Data pada rentang ini terlalu sedikit untuk digambar.")
        return

    ret = tampil["Close"].iloc[-1] / tampil["Close"].iloc[0] - 1
    rsi = tampil["RSI"].dropna()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Harga terakhir", f"Rp{tampil['Close'].iloc[-1]:,.0f}".replace(",", "."), help=f"Per {akhir:%d-%m-%Y}")
    k2.metric(f"Return {rentang.lower()}", persen(ret, 1))
    k3.metric("RSI (14) terakhir", angka(float(rsi.iloc[-1]), 1) if len(rsi) else "-")
    k4.metric("Hari bursa dalam data", f"{len(e):,}".replace(",", "."))
    st.caption(f"Sumber: {s['sumber']}. Harga disesuaikan untuk split dan dividen oleh penyedia data, sehingga bisa berbeda dari "
               "harga nominal di prospektus.")
    if len(e) < 200:
        st.caption("Data kurang dari 200 hari bursa sehingga MA200 belum bisa dihitung.")

    penanda = _penanda(e, listing, kiri)
    t1, t2, t3, t4, t5 = st.tabs(["Harga dan MA", "Vs IHSG", "RSI", "MACD", "Volume"])
    with t1:
        st.altair_chart(_grafik_harga(tampil, penanda), width="stretch")
        st.caption("MA20/50/200 adalah rata-rata harga penutupan 20, 50, dan 200 hari bursa terakhir; garis putus-putus menandai perkiraan "
                   "hari bursa ke-90 dan ke-180 sejak listing, titik ukur target model.")
    with t2:
        if s["ihsg"] is None:
            st.info("Data IHSG tidak tersedia, jadi grafik pembanding tidak dapat ditampilkan. Unggah CSV IHSG pada Sumber data harga.")
        else:
            hasil = _grafik_ihsg(tampil, s["ihsg"][s["ihsg"].index >= kiri])
            if hasil is None:
                st.info("Tanggal data emiten dan IHSG tidak beririsan pada rentang ini.")
            else:
                grafik, r_e, r_i = hasil
                st.altair_chart(grafik, width="stretch")
                st.caption(f"Keduanya diindeks 100 pada awal rentang. Pada akhir rentang emiten {r_e:+.0f}% dan IHSG {r_i:+.0f}% "
                           f"(selisih {r_e - r_i:+.0f} poin).")
    with t3:
        st.altair_chart(_grafik_rsi(tampil), width="stretch")
        st.caption("RSI (Relative Strength Index, 14 hari) bergerak 0 sampai 100. Secara konvensi, di atas 70 disebut jenuh beli dan "
                   "di bawah 30 jenuh jual; keduanya penanda kondisi harga, bukan sinyal pasti.")
    with t4:
        st.altair_chart(_grafik_macd(tampil), width="stretch")
        st.caption("MACD adalah selisih rata-rata bergerak eksponensial 12 dan 26 hari; Signal adalah rata-rata eksponensial 9 hari dari MACD. "
                   "Batang menunjukkan selisih keduanya: hijau bila MACD di atas Signal, merah bila di bawahnya.")
    with t5:
        if tampil["Volume"].notna().any():
            st.altair_chart(_grafik_volume(tampil), width="stretch")
            st.caption("Volume = jumlah lembar saham yang diperdagangkan per hari bursa.")
        else:
            st.info("Data volume tidak tersedia pada sumber ini.")
    st.caption(CATATAN)


def _prediksi_vs_realisasi(kode: str) -> None:
    st.markdown("### Prediksi vs realisasi")
    ev = evaluasi_uji()
    if not ev["cocok"]:
        st.info("Perbandingan tidak ditampilkan: evaluasi ulang model pada data penelitian tidak cocok dengan tabel metrik.")
        return
    df = data_asli(["Ticker Code", "label_90", "label_180", "log_excess_return_90", "log_excess_return_180"])
    idx = int(df.index[df["Ticker Code"] == kode][0])
    uji = idx in set(ev["test"])
    baris = []
    for h in ("90", "180"):
        nama = urut_model(h, "num")[0]
        p = float(ev["prob"][(h, nama)][idx])
        nyata = int(df.loc[idx, f"label_{h}"])
        rel = (np.exp(df.loc[idx, f"log_excess_return_{h}"]) - 1) * 100
        baris.append({"Horizon": f"{NAMA_HORIZON[h]} ({int(h) // 30} bulan)", "Model": nama,
                      "Peluang outperform (model)": persen(p), "Tebakan": "Outperform" if p >= 0.5 else "Underperform",
                      "Kenyataan": "Outperform" if nyata else "Underperform",
                      "Return relatif IHSG": f"{rel:+.0f}%".replace(".", ","),
                      "Hasil": "Tepat" if (p >= 0.5) == bool(nyata) else "Meleset"})
    st.dataframe(pd.DataFrame(baris), hide_index=True, width="stretch")
    if uji:
        st.caption("Emiten ini termasuk data uji: model tidak melihatnya saat dilatih, sehingga perbandingan ini adil. "
                   "Satu emiten tetap tidak cukup untuk menilai model.")
    else:
        st.warning("Emiten ini termasuk data latih: model sudah melihatnya saat belajar, sehingga kecocokan di sini tidak "
                   "mencerminkan kemampuan menebak emiten baru. Pilih emiten berstatus data uji untuk perbandingan yang adil.")
    st.caption("Skenario Terstruktur, model terbaik menurut CV AUC. Kenyataan = apakah return saham sejak harga penawaran melampaui IHSG.")


def indikator_teknikal() -> None:
    header("Indikator Teknikal")
    st.caption("Informasi pasar untuk saham yang sudah tercatat: harga dibanding IHSG, moving average, RSI, MACD, dan volume. " + CATATAN)
    df = data_asli(["Ticker Code", "Listing Date", "Sector"])
    ev = evaluasi_uji()
    uji = set(df.loc[ev["test"], "Ticker Code"]) if ev["cocok"] else set()
    kode_list = sorted(df["Ticker Code"])
    kode = st.selectbox("Emiten", kode_list, key="tek_kode",
                        format_func=lambda k: f"{k} (data uji)" if k in uji else k,
                        help="Emiten bertanda data uji tidak dipakai melatih model, jadi cocok untuk membandingkan prediksi dengan kenyataan.")
    baris = df[df["Ticker Code"] == kode].iloc[0]
    listing = pd.Timestamp(baris["Listing Date"])
    st.caption(f"{kode} · sektor {baris['Sector']} · listing {listing:%d-%m-%Y}")
    _sumber_data(kode, listing)
    _tampil_grafik(kode, listing)
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    _prediksi_vs_realisasi(kode)
