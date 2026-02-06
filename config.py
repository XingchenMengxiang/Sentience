# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from stock_pool import DEFAULT_STOCK_POOL


@dataclass
class Config:
    data_dir: str = "./data"
    stocks: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_STOCK_POOL.stocks))
    required_cols: List[str] = field(
        default_factory=lambda: ["open", "high", "low", "close", "vol"]
    )
    feature_cols: List[str] = field(
        default_factory=lambda: [
            "ret1", "ret5", "ret10", "ret22",
            "vol_ratio_5_22",
            "atr14",
            "dist_ma5", "dist_ma10", "dist_ma20",
            "ma5_ma20",
            "dd22", "rebound_from_min",
            "range_amp", "pct_pos",
            "slope22", "volatility22",
            "body", "upper_shadow", "lower_shadow",
        ]
    )
    window: int = 22
    label_mode: str = "online"
    horizon: int = 22
    test_years: int = 1
    min_rows: int = 200
    auto_fetch: bool = False
    fetch_start_date: str = "20200205"

    theta_r22: float = 0.03
    theta_down_r22: float = 0.06
    theta_dd22: float = 0.08
    theta_up_r10: float = 0.04
    theta_rebound: float = 0.06

    v_drop: float = 0.10
    v_recover: float = 0.80
    v_fastdrop_days: int = 7

    random_state: int = 42
    scaler_with_centering: bool = True
    scaler_with_scaling: bool = True
    xgb_params: Dict[str, object] = field(
        default_factory=lambda: {
            "n_estimators": 500,
            "max_depth": 4,
            "learning_rate": 0.03,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "n_jobs": 4,
        }
    )


CFG = Config()
