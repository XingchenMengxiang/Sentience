# -*- coding: utf-8 -*-
"""
Regime classifier for A-share V-cycle (18~22d window)
- Predict state: Down / Sideways / Recovery
- Universe: 河化股份、卓翼科技、上海电气、全聚德、孩子王、华谊集团、宝信软件、铭普光磁

Requirements:
pip install pandas numpy scikit-learn xgboost
"""

from __future__ import annotations
import os
import glob
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier
from stock_pool import StockPool, fetch_daily_data
from config import CFG, Config


# -----------------------------
# 3) Data loading
# -----------------------------
def load_one_csv(path: str, cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(path)
    # normalize date
    if "trade_date" not in df.columns:
        raise ValueError(f"Missing trade_date in {path}")

    df["trade_date"] = df["trade_date"].astype(str).str.replace("-", "")
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")

    # required columns
    for c in cfg.required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column {c} in {path}")

    df = df.sort_values("trade_date").reset_index(drop=True)
    return df


def load_universe(cfg: Config) -> Dict[str, pd.DataFrame]:
    """
    Accept either:
    - files named like 000953.SZ.csv
    - or Chinese names like 河化股份.csv
    """
    files = glob.glob(os.path.join(cfg.data_dir, "*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {cfg.data_dir}")

    out: Dict[str, pd.DataFrame] = {}
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]
        ts_code = None
        if base in cfg.stocks:
            ts_code = cfg.stocks[base]
        elif base.endswith(".SZ") or base.endswith(".SH"):
            ts_code = base
        else:
            # unknown file naming
            continue

        df = load_one_csv(f, cfg)
        df["ts_code"] = ts_code
        out[ts_code] = df

    # Keep only the specified stocks if configured
    if cfg.stocks:
        wanted = set(cfg.stocks.values())
        out = {k: v for k, v in out.items() if k in wanted}

    if len(out) == 0:
        raise ValueError("Loaded 0 target stocks. Check filenames & mapping.")
    return out


# -----------------------------
# 4) Feature engineering (22-day window)
# -----------------------------
def rolling_slope(y: np.ndarray) -> float:
    # slope of log(price) vs time index
    x = np.arange(len(y))
    y = np.log(np.maximum(y, 1e-12))
    x = x - x.mean()
    y = y - y.mean()
    denom = (x**2).sum()
    if denom == 0:
        return 0.0
    return float((x * y).sum() / denom)


def add_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    d = df.copy()

    # returns
    d["ret1"] = d["close"].pct_change(1)
    d["ret5"] = d["close"].pct_change(5)
    d["ret10"] = d["close"].pct_change(10)
    d["ret22"] = d["close"].pct_change(window)

    # vol stats
    d["vol5"] = d["vol"].rolling(5).mean()
    d["vol22"] = d["vol"].rolling(window).mean()
    d["vol_ratio_5_22"] = d["vol5"] / (d["vol22"] + 1e-12)

    # ATR-like (simple TR)
    prev_close = d["close"].shift(1)
    tr = pd.concat([
        (d["high"] - d["low"]),
        (d["high"] - prev_close).abs(),
        (d["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    d["tr"] = tr
    d["atr14"] = d["tr"].rolling(14).mean()

    # moving averages & distances
    d["ma5"] = d["close"].rolling(5).mean()
    d["ma10"] = d["close"].rolling(10).mean()
    d["ma20"] = d["close"].rolling(20).mean()
    d["dist_ma5"] = (d["close"] - d["ma5"]) / (d["ma5"] + 1e-12)
    d["dist_ma10"] = (d["close"] - d["ma10"]) / (d["ma10"] + 1e-12)
    d["dist_ma20"] = (d["close"] - d["ma20"]) / (d["ma20"] + 1e-12)
    d["ma5_ma20"] = (d["ma5"] - d["ma20"]) / (d["ma20"] + 1e-12)

    # window structure: max, min, drawdown, percentile
    roll_max = d["close"].rolling(window).max()
    roll_min = d["close"].rolling(window).min()
    d["win_max"] = roll_max
    d["win_min"] = roll_min
    d["dd22"] = d["close"] / (roll_max + 1e-12) - 1.0
    d["rebound_from_min"] = d["close"] / (roll_min + 1e-12) - 1.0
    d["range_amp"] = (roll_max - roll_min) / (d["close"].rolling(window).mean() + 1e-12)

    # percentile position in window
    def pct_rank_last(x: np.ndarray) -> float:
        # rank of last element within window
        if len(x) == 0:
            return np.nan
        last = x[-1]
        return float((x <= last).mean())

    d["pct_pos"] = d["close"].rolling(window).apply(pct_rank_last, raw=True)

    # slope & trend strength
    d["slope22"] = d["close"].rolling(window).apply(lambda x: rolling_slope(x), raw=True)
    d["volatility22"] = d["ret1"].rolling(window).std()

    # candle features (optional but useful)
    d["body"] = (d["close"] - d["open"]).abs() / (d["open"] + 1e-12)
    d["upper_shadow"] = (d["high"] - d[["close", "open"]].max(axis=1)) / (d["open"] + 1e-12)
    d["lower_shadow"] = (d[["close", "open"]].min(axis=1) - d["low"]) / (d["open"] + 1e-12)

    return d


# -----------------------------
# 5) Labeling
# -----------------------------
LABELS = {0: "Down", 1: "Sideways", 2: "Recovery"}

def label_online(d: pd.DataFrame, cfg: Config) -> pd.Series:
    """
    Only use past & current info.
    """
    # Rules:
    # Down: slope<0 and (ret22 very negative OR drawdown large)
    # Recovery: ret10 positive and rebound-from-min significant and drawdown improving
    # else Sideways

    ret22 = d["ret22"]
    ret10 = d["ret10"]
    slope = d["slope22"]
    dd22 = d["dd22"]
    rebound = d["rebound_from_min"]

    # drawdown improvement proxy: dd22 - dd22.shift(5)
    dd_improve = dd22 - dd22.shift(5)

    y = pd.Series(index=d.index, dtype="float64")

    cond_down = (slope < 0) & ((ret22 < -cfg.theta_down_r22) | (dd22 < -cfg.theta_dd22))
    cond_rec = (ret10 > cfg.theta_up_r10) & (rebound > cfg.theta_rebound) & (dd_improve > 0)

    y[cond_down] = 0
    y[cond_rec] = 2
    y[~cond_down & ~cond_rec] = 1

    return y.astype("int64")


def label_vconfirm(d: pd.DataFrame, cfg: Config) -> pd.Series:
    """
    Use future horizon to confirm a 'V repair' within 18~22 days.
    This is closer to your 'V-cycle 18-day law' definition.
    """
    H = cfg.horizon
    close = d["close"].values

    y = np.full(len(d), np.nan)

    for t in range(len(d) - H):
        # define recent high within past window (22)
        w0 = max(0, t - cfg.window + 1)
        past = close[w0:t+1]
        if len(past) < cfg.window:
            continue

        idx_high = np.argmax(past)
        p_high = past[idx_high]

        # in the next H days, find low and final price
        future = close[t+1:t+1+H]
        idx_low_f = np.argmin(future)
        p_low = future[idx_low_f]
        p_end = future[-1]

        drop = (p_high - p_low) / (p_high + 1e-12)
        if drop < cfg.v_drop:
            # no strong V event -> likely sideways unless trend down
            # fallback to online logic at time t for label
            continue

        # fast drop constraint: low happens quickly after high
        if idx_low_f + 1 > cfg.v_fastdrop_days:
            continue

        # recover ratio: how much of drop is recovered by end of horizon
        recover_ratio = (p_end - p_low) / (p_high - p_low + 1e-12)

        if recover_ratio >= cfg.v_recover:
            # V recovery confirmed
            y[t] = 2
        else:
            # if it dropped and did NOT recover, it's "Down" regime
            y[t] = 0

    # fill remaining NaNs with online label (acts as a default)
    d_tmp = d.copy()
    online = label_online(d_tmp, cfg).values
    mask = np.isnan(y)
    y[mask] = online[mask]
    return pd.Series(y.astype(int), index=d.index)


def build_samples(universe: Dict[str, pd.DataFrame], cfg: Config) -> pd.DataFrame:
    rows = []
    for code, df in universe.items():
        if len(df) < cfg.min_rows:
            continue

        d = add_features(df, cfg.window)

        if cfg.label_mode == "online":
            y = label_online(d, cfg)
        elif cfg.label_mode == "vconfirm":
            y = label_vconfirm(d, cfg)
        else:
            raise ValueError("label_mode must be 'online' or 'vconfirm'")

        d["y"] = y
        d["ts_code"] = code

        # drop early NaNs caused by rolling
        d = d.dropna(subset=cfg.feature_cols + ["y"]).copy()

        rows.append(d[["trade_date","ts_code","y"] + cfg.feature_cols])

    if not rows:
        raise ValueError("No usable stocks after filtering; check data length/columns.")

    all_df = pd.concat(rows, ignore_index=True)
    all_df = all_df.sort_values(["trade_date","ts_code"]).reset_index(drop=True)
    return all_df


# -----------------------------
# 7) Walk-forward split
# -----------------------------
def time_split(df: pd.DataFrame, test_years: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simple split: last N years as test
    """
    last_date = df["trade_date"].max()
    cutoff = last_date - pd.DateOffset(years=test_years)
    train = df[df["trade_date"] < cutoff].copy()
    test = df[df["trade_date"] >= cutoff].copy()
    return train, test


# -----------------------------
# 8) Train & evaluate
# -----------------------------
def train_xgb(train: pd.DataFrame, test: pd.DataFrame, cfg: Config) -> Pipeline:
    X_train = train[cfg.feature_cols].values
    y_train = train["y"].values.astype(int)
    X_test = test[cfg.feature_cols].values
    y_test = test["y"].values.astype(int)

    # RobustScaler helps with outliers
    params = dict(cfg.xgb_params)
    params.setdefault("random_state", cfg.random_state)
    model = XGBClassifier(**params)

    pipe = Pipeline([
        ("scaler", RobustScaler(
            with_centering=cfg.scaler_with_centering,
            with_scaling=cfg.scaler_with_scaling
        )),
        ("clf", model),
    ])

    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)
    pred = np.argmax(proba, axis=1)

    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(y_test, pred))
    print("\n=== Classification Report ===")
    print(classification_report(y_test, pred, target_names=[LABELS[i] for i in range(3)]))

    # show a few feature importances (after scaling, importances are from XGB)
    clf = pipe.named_steps["clf"]
    importances = clf.feature_importances_
    imp_df = pd.DataFrame({"feature": cfg.feature_cols, "importance": importances}).sort_values("importance", ascending=False)
    print("\n=== Top Feature Importances ===")
    print(imp_df.head(12).to_string(index=False))

    return pipe


def main():
    if CFG.auto_fetch:
        fetch_daily_data(
            data_dir=CFG.data_dir,
            start_date=CFG.fetch_start_date,
            pool=StockPool(stocks=dict(CFG.stocks)),
        )
    universe = load_universe(CFG)
    ds = build_samples(universe, CFG)
    train, test = time_split(ds, test_years=CFG.test_years)

    print(f"Total samples: {len(ds)}, train: {len(train)}, test: {len(test)}")
    print("Label distribution (all):")
    print(ds["y"].value_counts(normalize=True).rename(index=LABELS).sort_index())

    pipe = train_xgb(train, test, CFG)

    # example: latest day prediction per stock
    latest = ds.sort_values("trade_date").groupby("ts_code").tail(1)
    proba = pipe.predict_proba(latest[CFG.feature_cols].values)
    pred = np.argmax(proba, axis=1)

    out = latest[["trade_date","ts_code"]].copy()
    out["pred"] = [LABELS[i] for i in pred]
    out["p_down"] = proba[:,0]
    out["p_sideways"] = proba[:,1]
    out["p_recovery"] = proba[:,2]

    print("\n=== Latest regime prediction per stock ===")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
