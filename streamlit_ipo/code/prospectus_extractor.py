"""Ekstraktor informasi dari prospektus IPO Bursa Efek Indonesia.

Satu modul dipakai bersama oleh:
  - notebook validasi  (code/ekstraksi_prospektus.ipynb)
  - aplikasi Streamlit (menerima unggahan PDF prospektus baru)

Teks tiap PDF dibaca sekali dengan pdfplumber lalu disimpan sebagai cache JSON,
karena membaca satu prospektus (200-400 halaman) memakan 30-60 detik.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pdfplumber


# ════════════════════════════════════════════════════════════════════
# Pembacaan PDF
# ════════════════════════════════════════════════════════════════════
def ticker_dari_nama(nama_file: str) -> str | None:
    """'01_yoii-prospektus-ipo-2025.pdf' -> 'YOII';  '33-MHKI.pdf' -> 'MHKI'."""
    stem = Path(nama_file).stem.lower()
    m = re.search(r"^\d+[_-]([a-z]{4})(?:[-_]|$)", stem)
    return m.group(1).upper() if m else None


def baca_pdf(path, cache_dir=None, ulang: bool = False) -> list[str]:
    """Kembalikan teks per halaman. Jika cache_dir diberikan, hasil disimpan
    dan pemanggilan berikutnya untuk PDF yang sama langsung dibaca dari cache.
    ulang=True memaksa PDF dibaca ulang (mis. setelah cara pembacaan diubah)."""
    path = Path(path)
    cache = Path(cache_dir) / f"{path.stem}.json" if cache_dir else None
    if cache is not None and cache.exists() and not ulang:
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass                        # cache rusak -> baca ulang dari PDF

    with pdfplumber.open(path) as pdf:
        halaman = [_teks_halaman(page) for page in pdf.pages]
    _simpan_cache(cache, halaman)
    return halaman


def _terlihat_menyatu(teks: str) -> bool:
    """Halaman dua kolom yang terbaca menyatu menghasilkan banyak baris sangat
    panjang (kolom kiri + kanan dalam satu baris). Halaman satu kolom biasa
    jarang melebihi ~120 karakter per baris."""
    baris = [b for b in teks.split("\n") if b.strip()]
    return len(baris) >= 10 and sum(len(b) > 135 for b in baris) / len(baris) > 0.15


def _pisah_kolom(page) -> str | None:
    """Cari celah vertikal di tengah halaman yang tidak dilintasi satu kata pun,
    lalu baca kolom kiri dan kanan secara terpisah. None bila tidak ada celah."""
    kata = page.extract_words()
    if len(kata) < 50:
        return None
    x0, x1 = page.bbox[0], page.bbox[2]
    lebar = x1 - x0
    for frac in [i / 100 for i in range(50, 36, -1)] + [i / 100 for i in range(51, 64)]:
        x = x0 + frac * lebar
        if any(w["x0"] < x < w["x1"] for w in kata):
            continue
        kiri = sum(w["x1"] <= x for w in kata)
        if min(kiri, len(kata) - kiri) < 0.2 * len(kata):
            continue
        teks_kiri = page.crop((x0, page.bbox[1], x, page.bbox[3])).extract_text() or ""
        teks_kanan = page.crop((x, page.bbox[1], x1, page.bbox[3])).extract_text() or ""
        return teks_kiri + "\n" + teks_kanan
    return None


def _teks_halaman(page) -> str:
    try:
        # Sebagian PDF (mis. MINE, BOAT) menyimpan teks halaman sebelumnya di
        # luar area halaman (koordinat x negatif); pdfplumber tetap membacanya
        # sehingga dua halaman tercampur. Ambil hanya karakter di dalam halaman.
        page = page.within_bbox(page.bbox)
        teks = page.extract_text() or ""
        if _terlihat_menyatu(teks):
            teks = _pisah_kolom(page) or teks
        return teks
    except Exception:                   # halaman rusak / gambar -> kosong
        return ""


def perbaiki_kolom_cache(path, cache_dir) -> int:
    """Untuk cache lama (dibuat sebelum deteksi kolom): baca ulang hanya halaman
    yang terlihat menyatu. Mengembalikan jumlah halaman yang diperbaiki."""
    path = Path(path)
    cache = Path(cache_dir) / f"{path.stem}.json"
    halaman = json.loads(cache.read_text(encoding="utf-8"))
    target = [i for i, t in enumerate(halaman) if _terlihat_menyatu(t)]
    if not target:
        return 0
    n = 0
    with pdfplumber.open(path) as pdf:
        for i in target:
            baru = _pisah_kolom(pdf.pages[i])
            if baru is not None:
                halaman[i], n = baru, n + 1
    if n:
        _simpan_cache(cache, halaman)
    return n


def _simpan_cache(cache, halaman) -> None:
    if cache is not None:
        # Tulis ke file sementara lalu rename: kalau proses dihentikan di
        # tengah jalan, tidak ada cache setengah jadi yang tertinggal.
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(halaman, ensure_ascii=False), encoding="utf-8")
        tmp.replace(cache)


# ════════════════════════════════════════════════════════════════════
# Utilitas baris
# ════════════════════════════════════════════════════════════════════
_TOC = re.compile(r"(\.{4,}|…{2,}|_{4,})\s*\d{1,4}\s*$")   # "FAKTOR RISIKO ....... 36"
_NOMOR_HALAMAN = re.compile(r"^\s*[-–]?\s*\d{1,4}\s*[-–]?\s*$")
_ROMAWI = r"(?:X{0,3})(?:IX|IV|V?I{0,3})"


def _baris(halaman: list[str]) -> list[tuple[int, str]]:
    """Ratakan teks per halaman menjadi (indeks_halaman, baris) tanpa baris kosong."""
    out = []
    for i, teks in enumerate(halaman):
        for b in (teks or "").split("\n"):
            b = b.strip()
            if b:
                out.append((i, b))
    return out


def _gabung(baris: list[tuple[int, str]]) -> str:
    """Gabungkan baris menjadi teks; nomor halaman yang berdiri sendiri dibuang."""
    return "\n".join(b for _, b in baris if not _NOMOR_HALAMAN.match(b)).strip()


# ════════════════════════════════════════════════════════════════════
# Bab Faktor Risiko
# ════════════════════════════════════════════════════════════════════
# Judul bab: wajib bernomor Romawi (tidak boleh kosong) dan berdiri satu baris
# penuh. Tanpa syarat ini, kalimat seperti "Faktor Risiko dalam Prospektus
# ini ..." atau "FAKTOR RISIKO USAHA ... TELAH DISUSUN" ikut dianggap judul.
# Sampai tiga kata tambahan diizinkan ("VI. FAKTOR RISIKO USAHA PERSEROAN").
_JUDUL_BAB_RISIKO = re.compile(
    rf"^(?:BAB\s+)?(?=[IVX]){_ROMAWI}\s*\.?\s*FAKTOR\s+RISIKO(?:\s+[A-Za-z]+){{0,3}}\s*$", re.I)
# Sebagian PDF (mis. BOAT) menumpuk teks halaman sebelumnya di halaman yang
# sama, sehingga judul bab muncul di UJUNG baris. Bentuk ini hanya diterima
# bila nomor Romawi diikuti titik: "... Jarang Terjadi VI. FAKTOR RISIKO".
_JUDUL_BAB_RISIKO_UJUNG = re.compile(
    rf"\s(?=[IVX]){_ROMAWI}\.\s*FAKTOR\s+RISIKO\s*$")
_JUDUL_BAB_LAIN = re.compile(rf"^(?:BAB\s+)?(?=[IVX]){_ROMAWI}\s*\.\s+[A-Z][A-Z ,&/()-]{{4,}}$")

# Penutup bab: pernyataan manajemen yang selalu ditulis huruf kapital, mis.
# "MANAJEMEN PERSEROAN (DENGAN INI) MENYATAKAN BAHWA (PERSEROAN TELAH
# MENGUNGKAPKAN) SEMUA/SELURUH RISIKO ..." atau "... TELAH DISUSUN BERDASARKAN".
_PENUTUP_RISIKO = re.compile(
    r"MENYATAKAN BAHWA\b.{0,40}\b(SEMUA|SELURUH|FAKTOR|RISIKO|MENGUNGKAPKAN)\b|"
    r"MENYATAKAN BAHWA (SEMUA|SELURUH)\b|"
    r"TELAH DISUSUN (OLEH PERSEROAN )?BERDASARKAN|^FAKTOR RISIKO USAHA DAN RISIKO UMUM", re.I)
# Bila dua kolom terbaca menyatu, pernyataan kapital itu bisa berada di tengah
# baris berhuruf kecil; frasa kapital ini cukup khas untuk dicari langsung.
_PENUTUP_TENGAH = re.compile(r"MANAJEMEN PERSEROAN (DENGAN INI )?MENYATAKAN BAHWA")

# Judul segmen muncul dalam tiga bentuk:
#   berhuruf  "C. RISIKO UMUM"   huruf KAPITAL (bagian ringkasan memakai "a. b. c.")
#   bernomor  "2. Risiko Umum"   hanya kategori utama / umum / investor
#   polos     "RISIKO UMUM"      baris kapital tanpa nomor
_SEG_HURUF = re.compile(r"^([A-E])\.\s*((?i:RISIKO)\b.*)")
_SEG_NOMOR = re.compile(
    r"^\d{1,2}\s*[.)]\s*((?i:RISIKO)\s+(?i:UTAMA|UMUM|BAGI\s+INVESTOR|"
    r"(?:TERKAIT|YANG\s+BERKAITAN\s+DENGAN)\s+(?:INVESTASI|SAHAM|PEMEGANG))\b.*)")
_SEG_POLOS = re.compile(
    r"^RISIKO\s+(UTAMA|UMUM|BAGI\s+INVESTOR|"
    r"(?:TERKAIT|YANG\s+(?:BERHUBUNGAN|BERKAITAN)\s+DENGAN)\s+(?:INVESTASI|SAHAM))\b[A-Z ,]{0,90}$")

# Pemetaan semantik sesuai Tabel 3.3; urutan daftar = prioritas.
_KATEGORI = [("A", re.compile(r"\bUTAMA\b", re.I)),
             ("C", re.compile(r"\bUMUM\b", re.I)),
             ("D", re.compile(r"\b(INVESTOR|INVESTASI|SAHAM|PEMEGANG)\b", re.I)),
             ("B", re.compile(r"\b(USAHA|MATERIAL|KEGIATAN|OPERASIONAL|BISNIS)\b", re.I))]


def _kategori(judul: str) -> str | None:
    for kode, pola in _KATEGORI:
        if pola.search(judul):
            return kode
    return None


def _penutup(baris: str) -> bool:
    return ((_kapital(baris) and bool(_PENUTUP_RISIKO.search(baris)))
            or bool(_PENUTUP_TENGAH.search(baris)))


# Penomoran segmen utama berbeda antar-prospektus: "A." (paling umum),
# "a." (mis. CGAS) atau "1." kapital penuh (mis. PGEO). Gaya ditentukan dari
# judul segmen pertama di dalam bab; judul bergaya lain dianggap sub-bagian.
_SEG_UTAMA = re.compile(r"^([A-Ea-e]|\d{1,2})\s*[.)]\s*((?i:RISIKO)\b.*)")


def _kapital(baris: str) -> bool:
    """Baris berhuruf kapital; toleran terhadap font yang menampilkan 'U'
    sebagai 'u' (mis. COAL: "RISIKO uTAMA")."""
    huruf = [c for c in baris if c.isalpha()]
    return bool(huruf) and sum(c.islower() for c in huruf) <= 0.15 * len(huruf)


def _segmen_utama(baris: str) -> tuple[str, str, str] | None:
    """(gaya, penanda, judul) bila baris adalah judul segmen utama yang
    kategorinya dikenali; gaya: 'A' huruf kapital, 'a' huruf kecil, '1' angka."""
    m = _SEG_UTAMA.match(baris)
    if not m or len(baris) > 120 or not _kategori(m.group(2)):
        return None
    tanda = m.group(1)
    if tanda.isdigit():
        if not _kapital(baris):           # "3. Risiko Terkait Kredit" = sub-butir
            return None
        return "1", tanda, m.group(2)
    return ("A" if tanda.isupper() else "a"), tanda, m.group(2)


def cari_bab_risiko(halaman: list[str]) -> tuple[int, int] | None:
    """Rentang baris (awal, akhir) bab Faktor Risiko dalam hasil _baris().

    Kandidat awal: judul bab bernomor Romawi yang bukan baris daftar isi.
    Bab asli dibedakan dari ringkasan/daftar isi karena memuat judul segmen
    "A. RISIKO ..."; bila ada beberapa kandidat, diambil yang terpanjang.
    """
    bs = _baris(halaman)
    calon = []
    for i, (_, b) in enumerate(bs):
        judul = _JUDUL_BAB_RISIKO.match(b) or _JUDUL_BAB_RISIKO_UJUNG.search(b)
        if not judul or _TOC.search(b):
            continue
        akhir, ada_segmen = len(bs), False
        for j in range(i + 1, len(bs)):
            bj = bs[j][1]
            if _SEG_HURUF.match(bj) or _segmen_utama(bj) or _SEG_POLOS.match(bj):
                ada_segmen = True
            # penutup hanya sah setelah segmen pertama: paragraf pembuka bab
            # kadang juga menyebut "... telah disusun berdasarkan bobot ..."
            if (ada_segmen and _penutup(bj)) or (_JUDUL_BAB_LAIN.match(bj)
                                                 and not _JUDUL_BAB_RISIKO.match(bj)):
                akhir = j
                break
        if ada_segmen:
            calon.append((akhir - i, i, akhir))

    if not calon:
        return None
    _, awal, akhir = max(calon)
    return awal, akhir


def ekstrak_risiko(halaman: list[str]) -> dict:
    """Pecah bab Faktor Risiko menjadi risk_A..risk_D (Tabel 3.3).

    risk_A = risiko utama, risk_B = risiko usaha, risk_C = risiko umum,
    risk_D = risiko bagi investor / terkait saham. Pemetaan berdasarkan MAKNA
    judul, bukan hurufnya: sebagian prospektus hanya punya A-C (dengan C =
    risiko investor), atau menulis risiko umum sebagai sub-bab "2. Risiko
    Umum" / "RISIKO UMUM" di dalam segmen B. risk_E menampung segmen berhuruf
    yang judulnya tidak dapat dikategorikan.
    """
    hasil = {f"risk_{h}": None for h in "ABCDE"}
    hasil.update({"risk_halaman_awal": None, "risk_halaman_akhir": None,
                  "risk_ditemukan": False, "risk_judul": []})
    rentang = cari_bab_risiko(halaman)
    if rentang is None:
        return hasil

    bs = _baris(halaman)
    awal, akhir = rentang
    hasil.update({"risk_halaman_awal": bs[awal][0] + 1,
                  "risk_halaman_akhir": bs[akhir - 1][0] + 1, "risk_ditemukan": True})

    # Judul huruf kapital ("C. RISIKO UMUM") dan angka berhuruf kapital
    # ("1. RISIKO UTAMA ...") selalu dianggap segmen. Huruf kecil ("a. Risiko
    # Utama") hanya dipakai bila bab tidak punya keduanya (mis. CGAS), karena
    # di bab lain huruf kecil adalah sub-butir. Struktur bertingkat seperti
    # COAL (A. Risiko Perseroan > 1. Utama, 2. Usaha; B. Risiko Entitas Anak >
    # 1. Utama, 2. Usaha) membuat penomoran angka diulang setelah huruf baru.
    segs = [(k, _segmen_utama(bs[k][1])) for k in range(awal + 1, akhir)]
    pakai_kecil = not any(g and g[0] in "A1" for _, g in segs)

    ada_huruf = any(_SEG_HURUF.match(bs[k][1]) for k, _ in segs)
    batas, huruf_terakhir, angka_terakhir, induk_umum = [], None, None, False
    for k, seg in segs:
        b = bs[k][1]
        m = _SEG_HURUF.match(b)
        if m:                                           # huruf kapital
            kat = _kategori(m.group(2))
            # huruf harus naik; pengecualian: salah cetak di PDF seperti KOKA
            # ("A. RISIKO BAGI INVESTOR" sesudah "C. RISIKO UMUM") diterima bila
            # kategorinya belum pernah muncul
            if (huruf_terakhir is None or m.group(1) > huruf_terakhir
                    or (kat and kat not in {x[1] for x in batas})):
                huruf_terakhir, angka_terakhir = max(m.group(1), huruf_terakhir or ""), None
                induk_umum = _kategori(m.group(2)) is None     # "A. RISIKO PERSEROAN"
                batas.append((k, _kategori(m.group(2)) or "E", b))
            continue
        if seg and seg[0] == "1" and (not ada_huruf or induk_umum):   # angka, kapital
            if angka_terakhir is None or int(seg[1]) > angka_terakhir:
                angka_terakhir = int(seg[1])
                batas.append((k, _kategori(seg[2]), b))
            continue
        if seg and seg[0] == "a" and pakai_kecil:       # huruf kecil
            if huruf_terakhir is None or seg[1].upper() > huruf_terakhir:
                huruf_terakhir = seg[1].upper()
                batas.append((k, _kategori(seg[2]), b))
            continue
        cocok = (len(b) <= 90 and _SEG_NOMOR.match(b)) or _SEG_POLOS.match(b)
        if cocok and _kategori(b):
            batas.append((k, _kategori(b), b))

    potongan: dict[str, list[str]] = {}
    for n, (k, kode, judul) in enumerate(batas):
        ujung = batas[n + 1][0] if n + 1 < len(batas) else akhir
        potongan.setdefault(kode, []).append(_gabung(bs[k:ujung]))
        hasil["risk_judul"].append(f"{kode}: {judul[:70]}")
    for kode, bagian in potongan.items():
        hasil[f"risk_{kode}"] = "\n".join(bagian)
    return hasil


# ════════════════════════════════════════════════════════════════════
# Rencana Penggunaan Dana (use of proceeds)
# ════════════════════════════════════════════════════════════════════
KATEGORI_UOP = ["capex", "bayar_utang", "modal_kerja", "akuisisi", "ekspansi"]

_JUDUL_UOP = re.compile(
    r"^(?:BAB\s+)?(?:[IVX]\s?){1,5}\.?\s*(?:RENCANA\s+)?PENGG?UNAAN\s+DANA", re.I)   # juga "I I.", salah ketik "PENGUNAAN"
_HALAMAN_TOC = re.compile(r"(\.{4,}|…{2,}|_{4,}|\s)\d{1,3}\s*$")      # "... SAHAM 3"

# Penanda butir. Gaya dipakai untuk menyusun tingkat (1. > a. > (i) ...).
_BUTIR = re.compile(r"^(\d{1,2}[.)]|\d{1,2}(?=\s+(?:Sekitar|Sebesar|Sebanyak|Sisa))|[a-z][.)]|"
                    r"\((?:[a-z]|[ivx]{1,5})\)|[ivx]{1,5}[.)]|[•●▪◦\-–])\s+")

_PERSEN = re.compile(r"(\d{1,3}(?:[.,]\d{1,4})?)\s*%")
_RUPIAH = re.compile(r"Rp\.?\s*(\d{1,3}(?:[.,]\d{3})*(?:,\d+)?|\d+(?:,\d+)?)\s*(triliun|miliar|milyar|juta)?", re.I)
_SISA = re.compile(r"\bsisa(nya)?\b|\bselebihnya\b", re.I)
# Butir rincian biaya emisi (underwriting fee, jasa akuntan, dsb.) bukan alokasi dana.
_BIAYA_EMISI = re.compile(
    r"\bfee\b|biaya\s+jasa|jasa\s+(penjaminan|penyelenggaraan|penjualan)|biaya\s+emisi|"
    r"lembaga\s+penunjang|profesi\s+penunjang|biaya\s+lain[- ]lain|biaya\s+pendaftaran|"
    r"biaya\s+pencatatan|biaya\s+percetakan|biaya\s+(kustodian|KSEI|BEI)|"
    r"^(akuntan\s+publik|konsultan\s+hukum|notaris|kantor\s+jasa\s+penilai|penilai\s+publik|"
    r"biro\s+administrasi|konsultan\s+keuangan)\b", re.I)
# Alokasi tunggal tanpa daftar: "... seluruhnya akan digunakan ... untuk modal kerja".
_TUNGGAL = re.compile(r"seluruh(nya)?\b.{0,200}?\bdigunakan\b.{0,120}?\buntuk\b(.{0,700})", re.I | re.S)

# Kata kunci per kategori; kategori ditentukan oleh kata kunci yang PALING
# AWAL muncul di teks butir (tujuan utama biasanya disebut di kalimat pertama).
_POLA_UOP = {
    "bayar_utang": re.compile(
        r"pelunasan|melunasi|pembayaran\s+(\S+\s+){0,4}(utang|hutang|pinjaman|kredit|fasilitas|"
        r"facilit(y|ies)|obligasi)|"
        r"refinanc|pembiayaan kembali", re.I),
    "akuisisi": re.compile(
        r"akuisisi|pengambilalihan|mengambil\s*alih|(pembelian|membeli)\s+(sebagian\s+|seluruh\s+)?saham", re.I),
    "ekspansi": re.compile(
        r"ekspansi|(pembukaan|membuka|penambahan)\s+(\S+\s+){0,4}(outlet|cabang|toko|gerai|klinik|"
        r"restoran|kantor|jaringan)", re.I),
    # catatan: "pengembangan usaha" sengaja TIDAK dipakai; prospektus menulis
    # "pengembangan usaha dalam bentuk belanja modal / modal kerja" sehingga
    # kategorinya ditentukan oleh bentuk konkret sesudahnya.
    "capex": re.compile(
        r"belanja\s+(barang\s+)?modal|capital\s+expenditure|capex|(pembelian|pengadaan|membeli)\s+(\S+\s+){0,4}"
        r"(mesin|peralatan|alat|kendaraan|kapal|tugboat|tongkang|armada|unit|perangkat|fasilitas|"
        # membeli tanah / bangunan / aset tetap = belanja modal (perlakuan akuntansi
        # PSAK 16). Dataset.xlsx tidak konsisten: mayoritas (TOOL, BATR, DATA, VAST,
        # AYAM) capex, sebagian kecil (HBAT, RELF, KAQI) akuisisi.
        r"tanah|lahan|landbank|aset|bangunan|gedung|gudang|pabrik|properti|ruko)|"
        r"pembangunan|renovasi|konstruksi|perluasan\s+(\S+\s+){0,2}(pabrik|gudang|kantor|fasilitas|"
        r"distribution|sarana|area|lahan)|infrastruktur|"
        r"(pengembangan|pembaharuan|pembaruan|peningkatan)\s+(\S+\s+){0,2}(IT|sistem|teknologi|aplikasi|"
        r"perangkat\s+lunak|software|platform|laboratorium|lab)\b", re.I),
    "modal_kerja": re.compile(
        r"modal\s+kerja|working\s+capital|biaya\s+operasional|kegiatan\s+operasional|"
        r"operational\s+expenditure|opex|(pembelian|pengadaan)\s+(\S+\s+){0,2}"
        r"(persediaan|bahan\s+baku|barang\s+dagang|inventori)", re.I),
}


def _angka(teks: str) -> float:
    """'69,65' -> 69.65; '275.000.000.000' -> 275e9; '1.234,5' -> 1234.5."""
    if "," in teks and "." in teks:
        return float(teks.replace(".", "").replace(",", "."))
    if teks.count(".") >= 1 and re.fullmatch(r"\d{1,3}(\.\d{3})+", teks):
        return float(teks.replace(".", ""))
    return float(teks.replace(",", "."))


def _nilai_rupiah(m) -> float:
    kali = {"triliun": 1e12, "miliar": 1e9, "milyar": 1e9, "juta": 1e6}
    return _angka(m.group(1)) * kali.get((m.group(2) or "").lower(), 1)


def kategori_uop(teks: str) -> str | None:
    """Kategori dari kata kunci paling awal; mula-mula di 500 karakter pertama,
    lalu diperluas bila kalimat pembuka hanya menyebut penyertaan modal dsb."""
    teks = re.sub(r"penyertaan\s+modal|setoran\s+modal|penyetoran\s+modal", "", teks, flags=re.I)
    for n in (500, 1500):
        potong = teks[:n]
        posisi = {k: m.start() for k, p in _POLA_UOP.items() if (m := p.search(potong))}
        if posisi:
            return min(posisi, key=posisi.get)
    return None


def _gaya_butir(penanda: str) -> str:
    p = penanda.rstrip(".)")
    if penanda[0] in "•●▪◦-–":
        return "•"
    if penanda.isdigit():                             # "1 Sekitar ..." tanpa tanda baca
        return "1"
    if p.isdigit():
        return "1" + penanda[-1]
    romawi = bool(re.fullmatch(r"\(?[ivx]+", p))      # "(iv)", "ii." (huruf i/v/x tunggal dianggap Romawi)
    if penanda.startswith("("):
        return "(i)" if romawi else "(a)"
    return ("i" if romawi else "a") + penanda[-1]


def cari_bab_uop(halaman: list[str]) -> tuple[int, int] | None:
    """Rentang baris bab Rencana Penggunaan Dana. Dipilih kandidat judul yang
    rentangnya memuat butir alokasi (persen/rupiah) terbanyak."""
    bs = _baris(halaman)
    terbaik = None
    for i, (_, b) in enumerate(bs):
        if not _JUDUL_UOP.match(b) or _HALAMAN_TOC.search(b):
            continue
        akhir = min(len(bs), i + 1500)
        for j in range(i + 1, akhir):
            if _JUDUL_BAB_LAIN.match(bs[j][1]) and not _JUDUL_UOP.match(bs[j][1]):
                akhir = j
                break
        n = sum(1 for _, x in bs[i:akhir] if _BUTIR.match(x) and (_PERSEN.search(x) or "Rp" in x))
        if n and (terbaik is None or n > terbaik[0]):
            terbaik = (n, i, akhir)
    return (terbaik[1], terbaik[2]) if terbaik else None


def _butir_alokasi(bs, awal, akhir) -> list[dict]:
    """Semua butir di bab beserta tingkat, persen / rupiah, dan teksnya."""
    butir, tumpukan = [], []
    for k in range(awal + 1, akhir):
        b = bs[k][1]
        m = _BUTIR.match(b)
        if not m:
            if butir:
                butir[-1]["teks"] += " " + b
            elif _SISA.search(b) and re.search(r"digunakan|dialokasikan", b):
                pass
            continue
        gaya = _gaya_butir(m.group(1))
        if gaya in tumpukan:
            tumpukan = tumpukan[:tumpukan.index(gaya) + 1]
        else:
            tumpukan.append(gaya)
        butir.append({"tingkat": len(tumpukan) - 1, "teks": b[m.end():], "baris": k})
    butir = [x for x in butir if not _BIAYA_EMISI.search(x["teks"][:150])]
    for x in butir:
        kepala = x["teks"][:250]
        # angka alokasi selalu di awal butir ("Sekitar 48% ..."); persen yang
        # muncul jauh di tengah kalimat bukan alokasi butir ini
        mp, mr = _PERSEN.search(kepala[:80]), _RUPIAH.search(kepala[:80])
        if not mp and not mr and re.match(r"(Sekitar|Sebesar|Sebanyak)?\s*(USD|US\$)", kepala):
            mr = _RUPIAH.search(x["teks"][:350])       # "USD 4,8 juta (ekuivalen Rp75 miliar)"
        x["persen"] = _angka(mp.group(1)) if mp and (not mr or mp.start() < mr.start()) else None
        x["rupiah"] = _nilai_rupiah(mr) if mr and (not mp or mr.start() < mp.start()) else None
        x["sisa"] = x["persen"] is None and x["rupiah"] is None and bool(_SISA.search(kepala))
        teks = x["teks"]
        if x["sisa"]:
            # "Sisa dana setelah dikurangi pembayaran utang ... akan digunakan untuk
            # modal kerja": tujuan ada SESUDAH kata "digunakan"
            m = re.search(r"\b(digunakan|dialokasikan)\b", teks)
            teks = teks[m.end():] if m else teks
        x["kategori"] = kategori_uop(teks)
    return butir


def _anak(butir, i):
    """Indeks butir tepat satu tingkat di bawah butir i."""
    t, hasil = butir[i]["tingkat"], []
    for j in range(i + 1, len(butir)):
        if butir[j]["tingkat"] <= t:
            break
        if butir[j]["tingkat"] == t + 1:
            hasil.append(j)
    return hasil


def _pecah(butir, i, nilai, kunci) -> list[tuple[float, str | None]]:
    """Turun ke sub-butir bila tujuannya berbeda-beda dan angkanya konsisten
    (jumlah sub-butir = induk, atau = 100% bila sub-butir relatif terhadap induk)."""
    semua_anak = _anak(butir, i)
    anak = [j for j in semua_anak if butir[j][kunci] is not None]
    kat = {butir[j]["kategori"] for j in anak} - {None}
    induk_kosong = butir[i]["kategori"] is None
    if anak and (len(kat) >= 2 or induk_kosong):
        jumlah = sum(butir[j][kunci] for j in anak)
        skala = None
        if abs(jumlah - nilai) <= max(0.01 * nilai, 0.5 if kunci == "persen" else 0):
            skala = 1.0
        elif kunci == "persen" and abs(jumlah - 100) <= 0.5:
            skala = nilai / 100
        if skala:
            out = []
            for j in anak:
                for v, k in _pecah(butir, j, butir[j][kunci] * skala, kunci):
                    out.append((v, k or butir[i]["kategori"]))
            return out
    if induk_kosong:
        # "... digunakan untuk penyertaan saham, yaitu:" -> tujuan ada di sub-butir
        keturunan = []
        for j in range(i + 1, len(butir)):
            if butir[j]["tingkat"] <= butir[i]["tingkat"]:
                break
            keturunan.append(butir[j]["teks"])
        return [(nilai, kategori_uop(" ".join(keturunan)))]
    return [(nilai, butir[i]["kategori"])]


def ekstrak_uop(halaman: list[str], dana_kotor: float | None = None) -> dict:
    """Persentase alokasi dana IPO per kategori (uop_pct_*), dijumlah menjadi 100.

    dana_kotor: nilai Penawaran Umum (harga x jumlah saham). Hanya dibutuhkan
    bila alokasi ditulis dalam rupiah dan butir terakhirnya "sisanya ...":
    sisa = dana_kotor - jumlah butir lain (konvensi yang sama dengan
    Dataset.xlsx, yang membagi dengan IPO Offer Size).
    Mengembalikan juga rincian butir tingkat teratas agar mudah diperiksa.
    """
    hasil = {f"uop_pct_{k}": None for k in KATEGORI_UOP}
    hasil.update({"uop_ditemukan": False, "uop_mode": None, "uop_total_asli": None,
                  "uop_rincian": [], "uop_halaman": None})
    rentang = cari_bab_uop(halaman)
    if rentang is None:
        return hasil
    bs = _baris(halaman)
    awal, akhir = rentang
    hasil["uop_halaman"] = bs[awal][0] + 1
    butir = _butir_alokasi(bs, awal, akhir)

    # tingkat teratas = tingkat paling dangkal yang punya angka alokasi
    beralokasi = [x for x in butir if x["persen"] is not None or x["rupiah"] is not None]
    # pengantar = teks sebelum butir pertama; bila berupa satu kalimat
    # "seluruhnya digunakan untuk ..." (bukan pengantar daftar "... sebagai
    # berikut:"), seluruh dana untuk satu tujuan (mis. CBDK, OBAT)
    batas_intro = min([akhir, awal + 15] + [x["baris"] for x in butir])
    intro = " ".join(b for _, b in bs[awal + 1:batas_intro])
    m = _TUNGGAL.search(intro)
    tunggal = bool(m) and not re.search(r"berikut|rincian|:\s*$", intro, re.I)
    if tunggal or not beralokasi:
        if m is None:
            teks = " ".join(b for _, b in bs[awal + 1:min(akhir, awal + 15)])
            m = _TUNGGAL.search(teks)
        if m and kategori_uop(m.group(2)):
            kat = kategori_uop(m.group(2))
            hasil.update({f"uop_pct_{k}": (100.0 if k == kat else 0.0) for k in KATEGORI_UOP})
            hasil.update({"uop_ditemukan": True, "uop_mode": "tunggal", "uop_total_asli": 100.0,
                          "uop_rincian": [(100.0, kat, m.group(0)[:90])]})
        return hasil
    puncak = min(x["tingkat"] for x in beralokasi)
    kandidat = [i for i, x in enumerate(butir) if x["tingkat"] == puncak
                and (x["persen"] is not None or x["rupiah"] is not None or x["sisa"])]
    # satuan mengikuti butir alokasi pertama (tabel pinjaman sesudahnya bisa
    # memuat banyak angka rupiah yang bukan alokasi)
    pertama = next(butir[i] for i in kandidat if not butir[i]["sisa"])
    mode = "persen" if pertama["persen"] is not None else "rupiah"

    # daftar alokasi berakhir pada butir "sisanya" atau saat jumlah mencapai 100%
    atas, jumlah = [], 0.0
    for i in kandidat:
        if butir[i][mode] is None and not butir[i]["sisa"]:
            continue
        atas.append(i)
        if butir[i]["sisa"]:
            break
        if mode == "persen" and butir[i]["persen"] is not None:
            jumlah += butir[i]["persen"]
            if jumlah >= 99.5:
                break

    bagian, sisa = [], []
    for i in atas:
        x = butir[i]
        if x[mode] is not None:
            bagian += _pecah(butir, i, x[mode], mode)
        elif x["sisa"]:
            sisa.append(x["kategori"])
        hasil["uop_rincian"].append((x[mode], x["kategori"], x["teks"][:90]))
    if not bagian:
        return hasil
    total = sum(v for v, _ in bagian)
    if mode == "persen":
        hasil["uop_total_asli"] = round(total, 2)
        if sisa and total < 100:
            bagian += [((100 - total) / len(sisa), k) for k in sisa]
            total = 100.0
    elif sisa:
        if not dana_kotor or dana_kotor <= total:
            return hasil        # sisa dalam rupiah tidak dapat dihitung
        bagian += [((dana_kotor - total) / len(sisa), k) for k in sisa]
        total = dana_kotor
    hasil["uop_mode"] = mode
    pct = {k: 0.0 for k in KATEGORI_UOP}
    for v, k in bagian:
        pct[k or "modal_kerja"] += 100 * v / total
    hasil.update({f"uop_pct_{k}": round(v, 2) for k, v in pct.items()})
    hasil["uop_ditemukan"] = True
    return hasil


# ════════════════════════════════════════════════════════════════════
# Fakta penawaran (sampul prospektus)
# ════════════════════════════════════════════════════════════════════
# Paragraf di bawah judul "PENAWARAN UMUM PERDANA SAHAM" pada sampul memuat
# jumlah saham, porsi terhadap modal setelah IPO, nilai nominal, harga
# penawaran, dan nilai penawaran. Bentuknya sangat seragam antar-prospektus:
#   "Sebanyak 1.880.000.000 (...) saham baru ... yang mewakili 21,08% (...)
#    dari modal ditempatkan dan disetor Perseroan setelah Penawaran Umum
#    Perdana dengan nilai nominal Rp70,- (...) per saham ... dengan Harga
#    Penawaran sebesar Rp188,- (...) per saham. Jumlah seluruh nilai Penawaran
#    Umum Perdana Saham adalah sebesar Rp353.440.000.000,-"
_JUDUL_PENAWARAN = re.compile(r"^PENAWARAN\s+UMUM\s+(PERDANA\s+)?SAHAM\s*$", re.M | re.I)
_N_SAHAM = re.compile(r"\b(?:Sebanyak(?:-banyaknya)?|Sebesar|Sejumlah)\s+([\d.,]+)\s*\(", re.I)
_PORSI = re.compile(r"([\d.,]+)\s*%\s*\([^)]{0,140}\)\s*(?:dari|atas)\s+(?:\w+\s+){0,4}"
                    r"modal\s+(?:\w+\s+){0,3}\w*ditempatkan", re.I)   # "modal S A Sditempatkan"
_WARAN = re.compile(r"\bWaran\s+Seri\b", re.I)
# "menerbitkan sebanyak 85.000.000 (delapan puluh lima juta) Waran Seri I"
_WARAN_JUMLAH = re.compile(
    r"(?:sebanyak|sebesar|menerbitkan)[^.]{0,80}?([\d.]{7,})\s*\([^)]{0,140}\)\s*Waran\s+Seri\s+I", re.I)
_NOMINAL = re.compile(r"nilai\s+nominal\s*Rp\.?\s*([\d.,]+)", re.I)
_HARGA = re.compile(r"Harga\s+Penawaran\s*(?:sebesar\s*)?Rp\.?\s*([\d.,]+)", re.I)
_NILAI_PENAWARAN = re.compile(
    r"[Jj]umlah\s+(?:seluruh\s+)?(?:nilai\s+)?Penawaran\s+Umum[^.]{0,90}?"
    r"(?:sebesar|sebanyak)\s*Rp\.?\s*([\d.,]+)")


def blok_penawaran(halaman: list[str], n_halaman: int = 6) -> str | None:
    """Paragraf penawaran pada sampul (beberapa halaman pertama).

    Bila sampul tidak terbaca (mis. teksnya berupa gambar), paragraf yang sama
    dicari di seluruh dokumen: Bab I "Penawaran Umum" mengulang isinya.
    """
    teks = "\n".join(h or "" for h in halaman[:n_halaman])
    for m in _JUDUL_PENAWARAN.finditer(teks):
        blok = teks[m.end():m.end() + 2000]
        if _N_SAHAM.search(blok[:400]):
            return blok

    for sumber in (teks, "\n".join(h or "" for h in halaman)):
        for m in _HARGA.finditer(sumber):
            blok = sumber[max(0, m.start() - 1500):m.start() + 600]
            n = _N_SAHAM.search(blok)
            if n:
                return blok[n.start():]
    return None


def ekstrak_penawaran(halaman: list[str]) -> dict:
    """Fakta penawaran dari sampul: jumlah saham, porsi, nominal, harga, nilai."""
    hasil = {"jumlah_saham": None, "porsi_saham": None, "nilai_nominal": None,
             "harga_penawaran": None, "nilai_penawaran": None, "jumlah_waran": 0.0,
             "saham_setelah_ipo": None, "owner_retention": None, "penawaran_ditemukan": False}
    blok = blok_penawaran(halaman)
    if blok is None:
        return hasil

    m = _N_SAHAM.search(blok)
    if m:
        hasil["jumlah_saham"] = _angka(m.group(1))
    # porsi: bagian Waran dipotong dulu ("19,77% dari total saham ditempatkan"
    # milik Waran Seri I bukan porsi saham yang ditawarkan). Bila penawaran
    # terdiri atas beberapa bagian (mis. YUPI: saham baru 3% + divestasi 7%),
    # porsinya dijumlahkan, kecuali salah satu angka sudah merupakan totalnya.
    # kalimat penawaran berakhir pada "Harga Penawaran"; persen sesudahnya
    # milik hal lain (program ESA/MESOP, Waran) dan tidak ikut dihitung
    batas = [x.start() for x in (_WARAN.search(blok), _HARGA.search(blok)) if x]
    porsi = [_angka(x.group(1)) for x in _PORSI.finditer(blok[:min(batas)] if batas else blok)]
    if porsi:
        total = sum(porsi)
        besar = max(porsi)
        hasil["porsi_saham"] = (besar if abs(besar - (total - besar)) < 0.05 else total) / 100
    m = _NOMINAL.search(blok)
    if m:
        hasil["nilai_nominal"] = _angka(m.group(1))
    m = _HARGA.search(blok)
    if m:
        hasil["harga_penawaran"] = _angka(m.group(1))
    m = _NILAI_PENAWARAN.search(blok)
    if m:
        hasil["nilai_penawaran"] = _angka(m.group(1))
    elif hasil["jumlah_saham"] and hasil["harga_penawaran"]:
        hasil["nilai_penawaran"] = hasil["jumlah_saham"] * hasil["harga_penawaran"]

    m = _WARAN_JUMLAH.search(blok_penawaran(halaman, 4) or "") or \
        _WARAN_JUMLAH.search("\n".join(h or "" for h in halaman[:4]))
    hasil["jumlah_waran"] = _angka(m.group(1)) if m else 0.0

    # Saham beredar setelah IPO = saham ditawarkan / porsinya.
    if hasil["jumlah_saham"] and hasil["porsi_saham"]:
        setelah = hasil["jumlah_saham"] / hasil["porsi_saham"]
        hasil["saham_setelah_ipo"] = setelah
        # Owner retention = porsi pemilik lama setelah IPO, memperhitungkan
        # dilusi penuh oleh Waran Seri I (konvensi Dataset.xlsx; tanpa Waran
        # nilainya sama dengan 1 - porsi saham).
        hasil["owner_retention"] = (setelah - hasil["jumlah_saham"]) / (setelah + hasil["jumlah_waran"])

    hasil["penawaran_ditemukan"] = bool(hasil["jumlah_saham"] and hasil["harga_penawaran"])
    return hasil


# ════════════════════════════════════════════════════════════════════
# Penjamin emisi, kebijakan dividen, lock-up
# ════════════════════════════════════════════════════════════════════
_BAB_PENJAMIN = re.compile(
    r"^(?:BAB\s+)?[IVX]+\.?\s*PENJAMIN(?:AN)?\s+(?:PELAKSANA\s+)?EMISI\s+EFEK", re.I)
# baris tabel porsi penjaminan: "1. PT Lotus Andalan Sekuritas 375.000.000 44.250.000.000 100,00"
_BARIS_UW = re.compile(
    r"^(?:(\d{1,2})[.)]?\s+)?(PT\s+[A-Z][\w&.\-]*(?:\s+[A-Z][\w&.\-]*){0,4}\s+"
    r"(?:Sekuritas|Securities)(?:\s+\w+){0,2}?)(?:\s*\([^)]{0,30}\))?\s+(?P<sisa>[\d.]{3,}.*)$", re.I)
# porsi penjaminan = angka terakhir pada baris, mis. "... Rp254.148.716.320,- 71,91%"
_PORSI_UW = re.compile(r"(\d{1,3}(?:[.,]\d+)?)\s*%?\s*$")
_LABEL_LEAD = re.compile(r"^Penjamin\s+Pelaksana\s+Emisi\s+Efek\s*:?\s*$", re.I)
_LABEL_ANGGOTA = re.compile(r"^(?:Penjamin\s+Emisi\s+Efek|Anggota\s+Sindikasi)\s*:?\s*$", re.I)

_BAB_DIVIDEN = re.compile(r"^(?:BAB\s+)?[IVX]+\.?\s*KEBIJAKAN\s+DIVIDEN", re.I)
_RASIO_DIVIDEN = re.compile(
    r"(\d{1,3}(?:[.,]\d+)?)\s*%[^.]{0,120}?\b(?:dari|atas|terhadap)\b[^.]{0,60}?"
    r"(?:laba\s+bersih|saldo\s+laba)|"
    # "membayar dividen ... sebanyak/sebesar-besarnya 20%," tanpa menyebut labanya
    r"(?:sebanyak|sebesar)\s*-?\s*(?:banyaknya|besarnya)\s+(\d{1,3}(?:[.,]\d+)?)\s*%", re.I)

_LOCKUP = re.compile(
    r"tidak\s+akan\s+(?:mengalihkan|menjual|melepas)[^.]{0,200}?saham|lock\s*-?\s*up|"
    r"(?:dilarang|tidak\s+dapat)\s+mengalihkan[^.]{0,120}?saham|penguncian\s+saham", re.I)


def _nama_uw(nama: str) -> str:
    nama = re.sub(r"\s+", " ", nama).strip().lower()
    return re.sub(r"\s*(tbk\.?|indonesia)\s*$", "", nama).strip()


def ekstrak_penjamin(halaman: list[str]) -> dict:
    """Jumlah penjamin emisi dari tabel porsi penjaminan.

    Tabel memisahkan "Penjamin Pelaksana Emisi Efek" (lead) dari "Penjamin
    Emisi Efek" biasa, baik lewat baris label maupun lewat penomoran yang
    diulang dari 1. Keduanya dikembalikan karena Dataset.xlsx memakai definisi
    yang tidak konsisten (lihat catatan di laporan).
    """
    hasil = {"uw_lead": None, "uw_total": None, "uw_nama": [], "uw_porsi": []}
    bs = _baris(halaman)
    bab = [k for k, (_, b) in enumerate(bs)
           if _BAB_PENJAMIN.match(b) and not _TOC.search(b) and not re.search(r"\s\d{1,3}$", b)]
    terbaik = ([], [], [])
    for b0 in bab:
        nama, lead, porsi, grup_lead = [], [], [], True
        for k in range(b0, min(len(bs), b0 + 200)):
            b = bs[k][1]
            if _LABEL_LEAD.match(b):
                grup_lead = True
                continue
            if _LABEL_ANGGOTA.match(b) and nama:
                grup_lead = False
                continue
            m = _BARIS_UW.match(b)
            if not m:
                continue
            if m.group(1) and int(m.group(1)) == 1 and nama:   # penomoran diulang
                grup_lead = False
            n = _nama_uw(m.group(2))
            if n not in nama:
                nama.append(n)
                mp = _PORSI_UW.search(m.group("sisa").strip())
                porsi.append(_angka(mp.group(1)) if mp else None)
                if grup_lead:
                    lead.append(n)
        if len(nama) > len(terbaik[0]):
            terbaik = (nama, lead, porsi)
    if terbaik[0]:
        hasil.update({"uw_total": len(terbaik[0]), "uw_lead": len(terbaik[1]),
                      "uw_nama": terbaik[0], "uw_porsi": terbaik[2]})
    return hasil


def ekstrak_dividen(halaman: list[str]) -> dict:
    """Rasio pembayaran dividen yang dijanjikan (0-1) dari bab Kebijakan Dividen,
    mis. "sebanyak-banyaknya 30% (tiga puluh persen) dari laba bersih"."""
    bs = _baris(halaman)
    bab = [k for k, (_, b) in enumerate(bs)
           if _BAB_DIVIDEN.match(b) and not _TOC.search(b) and not re.search(r"\s\d{1,3}$", b)]
    for b0 in bab:
        teks = " ".join(b for _, b in bs[b0:b0 + 120])
        # kalimat harus benar-benar tentang dividen: bab yang sama juga memuat
        # "dana cadangan ... 20% dari jumlah modal disetor"
        for m in _RASIO_DIVIDEN.finditer(teks):
            if re.search(r"dividen", teks[max(0, m.start() - 250):m.start()], re.I):
                return {"dividen_rasio": _angka(m.group(1) or m.group(2)) / 100,
                        "dividen_kalimat": m.group(0)[:150]}
    return {"dividen_rasio": None, "dividen_kalimat": None}


# ════════════════════════════════════════════════════════════════════
# Ikhtisar Data Keuangan Penting: DER dan ROA
# ════════════════════════════════════════════════════════════════════
_BAB_KEUANGAN = re.compile(
    r"^(?:BAB\s+)?[IVX]+\.?\s*IK[HT]{1,2}ISAR\s+(?:DATA\s+)?KEUANGAN", re.I)   # juga salah ketik "IKTHISAR"
# Sebagian prospektus tidak punya bab Ikhtisar tersendiri; tabel yang sama ada di
# bab Analisis dan Pembahasan oleh Manajemen.
_BAB_KEUANGAN_ALT = re.compile(r"^(?:BAB\s+)?[IVX]+\.?\s*ANALISIS\s+DAN\s+PEMBAHASAN", re.I)
# Kolom pertama sesudah label = periode terbaru.
_BARIS_ANGKA = re.compile(r"^(?P<label>[A-Za-z()\/\-,. ]{4,70}?)\s+(?P<sisa>\(?-?\d[\d.,].*)$")
# token angka utuh; kolom yang tercetak menempel ("1.994.539.5571.228.049") dilewati
_ANGKA_TABEL = re.compile(r"^\(?-?\d{1,3}(?:\.\d{3})+(?:,\d+)?\)?$")
_LABEL = {
    "aset": re.compile(r"^(total|jumlah)\s+aset$", re.I),
    "liabilitas": re.compile(r"^(total|jumlah)\s+liabilitas$", re.I),
    "ekuitas": re.compile(r"^(total|jumlah)\s+ekuitas(\s*[-–]?\s*neto)?$", re.I),
    "laba": re.compile(
        r"^(jumlah\s+)?(laba|rugi)\s*(\(rugi\)|\(laba\))?\s*"
        r"(bersih|neto|komprehensif)?\s*(periode\s*/?\s*)?(tahun|periode)?\s*"
        r"(berjalan|berjalan\s+neto)?$", re.I),
}
# Sebagian prospektus mencetak rasionya langsung di bagian "Rasio Keuangan";
# angka ini lebih dapat dipercaya daripada hasil hitungan sendiri.
_LABEL_RASIO = {
    # hanya laba bersih/tahun berjalan; "Laba bruto / Jumlah Aset" bukan ROA
    "roa": re.compile(r"^(laba\s*(\(rugi\)|bersih|neto)?\s*(periode\s*/?\s*)?(tahun|periode)?\s*"
                      r"(berjalan)?\s*(/|terhadap)\s*(jumlah|total)\s+aset|"
                      r"(rasio\s+)?(imbal\s+hasil|return\s+on)\s+(aset|asset)s?|roa)$", re.I),
    "der": re.compile(r"^((jumlah|total)\s+liabilitas\s*/\s*(jumlah|total)\s+ekuitas|"
                      r"(rasio\s+)?liabilitas\s+terhadap\s+ekuitas|"
                      r"debt\s+to\s+equity(\s+ratio)?|der)$", re.I),
}
_ANGKA_RASIO = re.compile(r"^\(?-?\d{1,3}(?:[.,]\d+)?\)?%?$")


def _nilai_tabel(teks: str) -> float:
    negatif = teks.startswith("(") or teks.startswith("-")
    n = _angka(re.sub(r"[()\-]", "", teks))
    return -n if negatif else n


_HEADER_INTERIM = re.compile(
    r"\b\d{1,2}\s+(Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|September|Oktober|November)\b", re.I)
_HEADER_DESEMBER = re.compile(r"\b31\s+Desember\b", re.I)


def ekstrak_keuangan(halaman: list[str], kolom: str = "tahun_buku") -> dict:
    """Total aset, liabilitas, ekuitas, dan laba dari bab Ikhtisar Data Keuangan
    Penting, lalu DER dan ROA.

    kolom="terbaru"    : kolom pertama (bisa periode interim, mis. 8 bulan).
    kolom="tahun_buku" : kolom tahun buku terakhir (31 Desember). Dipakai secara
                         default karena laba interim tidak sebanding antar-emiten.
    """
    hasil = {"total_aset": None, "total_liabilitas": None, "total_ekuitas": None,
             "laba_bersih": None, "der": None, "roa": None, "der_tercetak": None}
    bs = _baris(halaman)

    def _judul(pola):
        return [k for k, (_, b) in enumerate(bs) if pola.match(b) and not _TOC.search(b)
                and not re.search(r"\s\d{1,3}$", b)]

    bab = _judul(_BAB_KEUANGAN)
    # bab Analisis dan Pembahasan jauh lebih panjang, jendela pemindaiannya dilebarkan
    jendela = 250 if bab else 700
    bab = bab or _judul(_BAB_KEUANGAN_ALT)
    for b0 in bab:
        # bila kolom pertama adalah periode interim ("30 September 31 Desember"),
        # tahun buku terakhir ada di kolom kedua
        # Baris header tabel, mis. "30 September 31 Desember" atau
        # "30 Juni 30 Juni 31 Desember": jumlah tanggal interim di depan
        # menentukan letak kolom tahun buku terakhir.
        idx = 0
        if kolom == "tahun_buku":
            for _, b in bs[b0:b0 + 60]:
                if _HEADER_DESEMBER.search(b) and _HEADER_INTERIM.search(b):
                    idx = len(_HEADER_INTERIM.findall(b))
                    break
        nilai = {}
        for k in range(b0, min(len(bs), b0 + jendela)):
            m = _BARIS_ANGKA.match(bs[k][1])
            if not m:
                continue
            label = m.group("label").strip()
            # Label panjang sering terpotong dua baris: "Laba (rugi) periode/tahun"
            # lalu "berjalan  123.456  ...". Gabungkan dengan baris sebelumnya
            # bila baris itu tidak memuat angka.
            labels = [label]
            if k > b0 and not re.search(r"\d", bs[k - 1][1]) and len(bs[k - 1][1]) <= 60:
                labels.append(f"{bs[k - 1][1].strip()} {label}")
            angka = [_nilai_tabel(x) for x in m.group("sisa").split() if _ANGKA_TABEL.match(x)]
            if angka:
                for kunci, pola in _LABEL.items():
                    if kunci not in nilai and any(pola.match(x) for x in labels):
                        nilai[kunci] = angka[idx] if len(angka) > idx else angka[0]
            rasio = [x for x in m.group("sisa").split() if _ANGKA_RASIO.match(x)]
            if rasio:
                for kunci, pola in _LABEL_RASIO.items():
                    if kunci in nilai or not any(pola.match(x) for x in labels):
                        continue
                    teks = rasio[idx] if len(rasio) > idx else rasio[0]
                    v = _nilai_tabel(teks.rstrip("%"))
                    # rasio ROA dicetak dalam persen, kadang tanpa tanda "%"
                    if teks.endswith("%") or (kunci == "roa" and abs(v) > 1.5):
                        v /= 100
                    nilai[kunci] = v
        if "aset" in nilai:
            if "ekuitas" not in nilai and "liabilitas" in nilai:
                nilai["ekuitas"] = nilai["aset"] - nilai["liabilitas"]
            hasil.update({"total_aset": nilai.get("aset"), "total_liabilitas": nilai.get("liabilitas"),
                          "total_ekuitas": nilai.get("ekuitas"), "laba_bersih": nilai.get("laba"),
                          "der_tercetak": nilai.get("der"), "roa": nilai.get("roa")})
            break
    # DER dihitung sendiri; rasio tercetak hanya cadangan karena satuannya
    # tidak seragam (ada yang persen, ada yang kali)
    if hasil["total_liabilitas"] and hasil["total_ekuitas"]:
        hasil["der"] = hasil["total_liabilitas"] / hasil["total_ekuitas"]
    elif hasil.get("der_tercetak") is not None:
        v = hasil["der_tercetak"]
        hasil["der"] = v / 100 if abs(v) > 15 else v
    if hasil["roa"] is None and hasil["laba_bersih"] is not None and hasil["total_aset"]:
        hasil["roa"] = hasil["laba_bersih"] / hasil["total_aset"]
    return hasil


# ════════════════════════════════════════════════════════════════════
# Kalender IPO: Market Activity dan reputasi penjamin emisi
# ════════════════════════════════════════════════════════════════════
# Kedua fitur ini tidak ada di dalam prospektus; keduanya dihitung dari daftar
# IPO lain di bursa. Kalender disimpan sebagai CSV (kolom: ticker,
# tanggal_listing, nilai_penawaran, penjamin) dan perlu diperbarui bila ada
# IPO baru, baik untuk penelitian maupun untuk aplikasi.
_BULAN = {"januari": 1, "februari": 2, "maret": 3, "april": 4, "mei": 5, "juni": 6, "juli": 7,
          "agustus": 8, "september": 9, "oktober": 10, "november": 11, "desember": 12}
# "Tanggal Pencatatan di Bursa Efek Indonesia : 14 April 2025"
_TGL_LISTING = re.compile(
    r"Tanggal\s+Pencatatan[^:\n]{0,60}:?\s*(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", re.I)


def tanggal_pencatatan(halaman: list[str]):
    """Tanggal listing dari jadwal di sampul prospektus (datetime.date)."""
    import datetime
    teks = "\n".join(h or "" for h in halaman[:3])
    for m in _TGL_LISTING.finditer(teks):
        if m.group(2).lower() not in _BULAN:
            continue
        try:
            return datetime.date(int(m.group(3)), _BULAN[m.group(2).lower()], int(m.group(1)))
        except ValueError:                  # angka lain yang kebetulan cocok
            continue
    return None


def baca_kalender(path) -> "pd.DataFrame":
    import pandas as pd
    k = pd.read_csv(path, parse_dates=["tanggal_listing"])
    return k


def market_activity(tanggal, kalender) -> int:
    """Jumlah IPO yang tercatat pada bulan kalender yang sama.

    Definisi ini yang dipakai Dataset.xlsx (korelasi 0,977 terhadap kalender
    dari 222 emiten; selisihnya berasal dari IPO yang tidak ada di sampel).
    """
    import pandas as pd
    t = pd.Timestamp(tanggal)
    bln = kalender["tanggal_listing"].dt.to_period("M")
    return int((bln == t.to_period("M")).sum())


def tabel_reputasi_uw(kalender, sebelum=None) -> dict:
    """Pangsa pasar tiap penjamin emisi berdasarkan nilai IPO yang dijamin
    (ukuran reputasi Megginson-Weiss). Bila `sebelum` diisi, hanya IPO sebelum
    tanggal itu yang dihitung, sehingga tidak ada kebocoran informasi masa depan.

    Catatan: kolom "Overall UW Rep" di Dataset.xlsx TIDAK dapat direproduksi
    dengan ukuran ini (korelasi ~0) maupun dengan tingkat keberhasilan historis
    (korelasi ~0,4); sumbernya tidak diketahui.
    """
    import pandas as pd
    k = kalender
    if sebelum is not None:
        k = k[k["tanggal_listing"] < pd.Timestamp(sebelum)]
    bobot: dict[str, float] = {}
    for _, r in k.iterrows():
        teks = r.get("penjamin")
        teks = "" if teks is None or pd.isna(teks) else str(teks)
        nama = [x.strip().lower() for x in teks.split(";") if x.strip()]
        nilai = float(r.get("nilai_penawaran") or 0)
        if not nama or not nilai:
            continue
        for n in nama:                       # nilai dibagi rata antar-penjamin
            bobot[n] = bobot.get(n, 0.0) + nilai / len(nama)
    total = sum(bobot.values())
    return {n: v / total for n, v in bobot.items()} if total else {}


def reputasi_uw(nama_penjamin: list[str], tabel: dict) -> float | None:
    """Reputasi satu emisi = pangsa pasar tertinggi di antara penjaminnya."""
    skor = [tabel.get(n.strip().lower()) for n in nama_penjamin if n.strip().lower() in tabel]
    return max(skor) if skor else None


# ── Overall UW Rep (metode tim: skor broker x porsi penjaminan) ──────
# Skor tiap broker dihitung di luar prospektus dari 6 metrik (jumlah IPO,
# total alokasi, kinerja tertimbang, konsistensi, volume emisi, dampak
# broker summary), dinormalisasi lalu dijumlah berbobot -> skala 2,5-7,9.
# Tabel skor bersifat tetap, jadi prospektus baru cukup memakai porsi
# penjaminannya: Overall UW Rep = SUM(porsi% x skor) / 100, lalu diskalakan
# 0-1 memakai skor terendah/tertinggi pada tabel.
def baca_skor_uw(path) -> dict:
    """{nama_penjamin: skor} dari data/skor_underwriter.csv."""
    import pandas as pd
    t = pd.read_csv(path)
    return {str(r["nama"]).strip().lower(): float(r["skor"])
            for _, r in t.iterrows() if isinstance(r.get("nama"), str) and r["nama"].strip()}


def overall_uw_rep(nama_penjamin: list[str], porsi: list, tabel: dict,
                   batas: tuple[float, float] | None = None) -> float | None:
    """Rata-rata skor penjamin, ditimbang porsi penjaminan, diskalakan 0-1.

    Penjamin yang tidak ada di tabel (mis. broker baru) diberi skor median.
    """
    if not tabel:
        return None
    median = sorted(tabel.values())[len(tabel) // 2]
    bobot = [(p if p is not None else 0.0) for p in porsi] or []
    nilai = [tabel.get(n.strip().lower(), median) for n in nama_penjamin]
    if not nilai:
        return None
    if sum(bobot) <= 0:                      # porsi tidak terbaca -> rata-rata biasa
        bobot = [1.0] * len(nilai)
    bobot = bobot[:len(nilai)] + [0.0] * max(0, len(nilai) - len(bobot))
    skor = sum(b * v for b, v in zip(bobot, nilai)) / sum(bobot)
    lo, hi = batas or (min(tabel.values()), max(tabel.values()))
    return (skor - lo) / (hi - lo) if hi > lo else None


def ekstrak_lockup(halaman: list[str]) -> dict:
    """1 bila prospektus memuat pernyataan lock-up pemegang saham."""
    for teks in halaman:
        m = _LOCKUP.search(teks or "")
        if m:
            return {"lockup": 1, "lockup_kalimat": re.sub(r"\s+", " ", m.group(0))[:150]}
    return {"lockup": 0, "lockup_kalimat": None}
