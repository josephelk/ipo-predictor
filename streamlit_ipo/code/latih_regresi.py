"""Melatih model regresi harga pasca-IPO dan menyimpannya ke output/v2/.

Dijalankan sekali (dari folder proyek):  python code/latih_regresi.py
Target: log(harga setelah H hari / harga penawaran), H = 90 dan 180 hari.
Fitur : skenario Terstruktur (sama dengan klasifikasi), pembagian latih/uji sama (80:20, seed 42).
Keluaran: output/v2/models_reg_v2.pkl, output/v2/summary_reg_v2.csv
"""
from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import umum  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score  # noqa: E402
from sklearn.model_selection import KFold, cross_val_predict  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402


def buat_model() -> dict:
    return {
        "Ridge": Ridge(alpha=10.0),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=200, max_depth=2, learning_rate=0.03, subsample=0.8,
                                colsample_bytree=0.8, random_state=42, verbosity=0),
    }


def main() -> None:
    df = umum.data_asli()
    _, kolom = umum.muat_model()
    itr, ite = umum.bagi_data(df["label_90"])
    X = umum.fitur_terstruktur(df)[kolom["num"]]
    model_simpan, baris = {}, []
    for h in ("90", "180"):
        y = np.log(df[f"price_{h}"] / df["Offer Price"])
        ytr, yte = y.loc[itr], y.loc[ite]
        baris.append({"Horizon": f"H+{h}", "Model": "Baseline (median latih)", "MAE log": mean_absolute_error(yte, np.full(len(yte), ytr.median())),
                      "R2 uji": r2_score(yte, np.full(len(yte), ytr.mean())), "R2 CV": np.nan})
        for nama, m in buat_model().items():
            oof = cross_val_predict(m, X.loc[itr], ytr, cv=KFold(5, shuffle=True, random_state=42))
            m.fit(X.loc[itr], ytr)
            p = m.predict(X.loc[ite])
            baris.append({"Horizon": f"H+{h}", "Model": nama, "MAE log": mean_absolute_error(yte, p),
                          "R2 uji": r2_score(yte, p), "R2 CV": r2_score(ytr, oof)})
            # Rentang ketidakpastian: sebaran selisih (aktual - prediksi) dari validasi silang pada data latih.
            sisa = (ytr - oof).to_numpy()
            model_simpan[(h, nama)] = {"model": m, "q10": float(np.quantile(sisa, 0.10)),
                                       "q90": float(np.quantile(sisa, 0.90)), "median_sisa": float(np.median(sisa))}
    ringkas = pd.DataFrame(baris)
    ringkas.to_csv(umum.V2_DIR / "summary_reg_v2.csv", index=False)
    with open(umum.V2_DIR / "models_reg_v2.pkl", "wb") as f:
        pickle.dump({"reg": model_simpan, "kolom": kolom["num"]}, f)
    print(ringkas.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
