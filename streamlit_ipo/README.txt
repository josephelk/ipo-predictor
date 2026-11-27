IPO STOCK PERFORMANCE PREDICTOR (versi 2, menu kiri)
====================================================
Aplikasi Streamlit untuk memprediksi apakah saham IPO akan mengungguli IHSG (klasifikasi biner)
dari data prospektus. Skripsi: Prediksi Kinerja Harga Saham Pasca-IPO berdasarkan Informasi Prospektus.

MENU (sidebar kiri)
  1. Beranda               tujuan aplikasi, angka kunci (jumlah emiten, model terbaik dan ROC-AUC), tombol ke Prediksi
  2. Panduan Penggunaan    langkah pakai, format dan ukuran PDF, waktu tunggu, arti kolom form, FAQ
  3. Prediksi              unggah prospektus, data fundamental yang bisa diedit, penjamin, sektor, horizon, hasil:
                           peluang outperform, konteks historis (+ penjelasan cara membaca dan insight), dan prediksi harga
  4. Cara Membaca Analisis arti probabilitas dan base rate, ROC-AUC, grafik historis, prediksi harga, contoh kasus, keterbatasan
  5. Indikator Teknikal    harga vs IHSG, MA20/50/200, RSI, MACD, volume + prediksi vs realisasi per emiten
  6. Informasi Tambahan    kamus istilah, kalender IPO, skor underwriter, metodologi singkat, sumber, disclaimer, referensi

PERUBAHAN DARI VERSI SEBELUMNYA
  - Menu "Hasil Model" dan "Data" dihapus (halaman_model.py dan halaman_data.py ikut dihapus).
  - Halaman hasil prediksi: "Top 5 fitur paling berpengaruh" dihapus.
  - Konteks historis kini diberi penjelasan cara membaca grafik dan insight (dihitung otomatis dari data).
  - Ditambah prediksi harga (regresi) di bawah konteks historis, mengikuti jangka waktu pilihan pengguna (3 atau 6 bulan).
    Model: Random Forest regresi pada fitur Terstruktur, target ln(harga H hari / harga penawaran). Ditampilkan sebagai
    perkiraan + rentang 80%. Akurasinya terbatas (lihat output/v2/summary_reg_v2.csv: pada data uji hanya sedikit lebih
    baik daripada tebakan median), jadi rentang lebih penting daripada angka tengah.
  - Model regresi dibuat ulang dengan:  python code/latih_regresi.py  (menghasilkan output/v2/models_reg_v2.pkl dan summary_reg_v2.csv)

ISI FOLDER
  code/app_streamlit_v2.py        kerangka aplikasi: tema dan menu kiri (jalankan file ini)
  code/umum.py                    konstanta, pemuat model/data, evaluasi ulang pada data uji (dipakai semua halaman)
  code/halaman_info.py            Beranda, Panduan Penggunaan, Cara Membaca Analisis, Informasi Tambahan
  code/halaman_prediksi.py        menu Prediksi (form, hasil, konteks historis, prediksi harga)
  code/latih_regresi.py           melatih model regresi harga (dijalankan sekali bila model perlu dibuat ulang)
  code/halaman_teknikal.py        menu Indikator Teknikal
  code/tampilan.py                CSS
  code/prospectus_extractor.py    ekstraksi data dari PDF prospektus
  output/v2/                      model terlatih (models_v2.pkl, models_reg_v2.pkl, fitted_transformers_v2.pkl) dan ringkasan metrik
  output/artifacts/dataset_final.parquet   data 249 emiten
  data/kalender_ipo.csv, data/skor_underwriter.csv   kalender IPO dan skor reputasi penjamin emisi
  .streamlit/config.toml          tema tampilan

CARA MENJALANKAN (Windows)
  1. Pasang Python 3.10 (disarankan 3.10 atau 3.11; versi di komputer pembuat: 3.10.11).
  2. Buka terminal di folder ini lalu buat lingkungan virtual:
         python -m venv venv
         venv\Scripts\activate
  3. Pasang library (versi dikunci karena file model .pkl sensitif terhadap versi scikit-learn/xgboost/catboost):
         pip install -r requirements.txt
  4. Jalankan:
         streamlit run code/app_streamlit_v2.py
     (atau klik dua kali jalankan.bat setelah langkah 2-3 selesai)
  5. Buka http://localhost:8501 di browser.

INDIKATOR TEKNIKAL: KEBUTUHAN INTERNET
  Harga harian diambil dari Yahoo Finance (kode saham + ".JK", IHSG = ^JKSE) saat tombol "Ambil data harga" ditekan,
  lewat library yfinance. Menu lain tidak memerlukan internet. Bila Yahoo Finance gagal (koneksi, kode tidak tersedia,
  atau saham sudah delisting), pilih "Unggah CSV" pada bagian Sumber data harga (format seperti unduhan Yahoo Finance:
  kolom Date atau Tanggal, Close, dan opsional Volume). Data harga ini hanya untuk informasi, tidak dipakai model.

CATATAN
  - Hasil adalah keluaran penelitian akademik, BUKAN rekomendasi investasi. Data uji hanya 50 emiten.
  - Sektor industri dipilih manual (tidak tercantum di prospektus).
  - Perbandingan prediksi vs realisasi memakai skenario Terstruktur saja: pembagian latih/uji direproduksi
    dari dataset_final.parquet dan diperiksa cocok dengan summary_clf_v2.csv saat aplikasi dibuka. Skenario berbasis teks
    tidak bisa direproduksi persis dari berkas data yang ada.
  - Kalau muncul error saat memuat model, hampir pasti versi library berbeda: pakai versi di requirements.txt.
