# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Dict, Optional

import pandas as pd
import tushare as ts


@dataclass(frozen=True)
class StockPool:
    stocks: Dict[str, str]


DEFAULT_STOCK_POOL = StockPool(
    stocks={
        "河化股份": "000953.SZ",
        "卓翼科技": "002369.SZ",
        "上海电气": "601727.SH",
        "全聚德": "002186.SZ",
        "孩子王": "301078.SZ",
        "华谊集团": "600623.SH",
        "宝信软件": "600845.SH",
        # "铭普光磁": "002902.SZ",
    }
)


def fetch_daily_data(
    data_dir: str,
    start_date: str,
    token: Optional[str] = None,
    pool: StockPool = DEFAULT_STOCK_POOL,
) -> None:
    """
    Fetch daily data for the configured stock pool and save to CSV in data_dir.
    """
    token = token or os.getenv("TUSHARE_TOKEN")
    if not token:
        raise ValueError("Missing TUSHARE_TOKEN. Set env var or pass token param.")

    os.makedirs(data_dir, exist_ok=True)
    ts.set_token(token)
    pro = ts.pro_api()

    for code in pool.stocks.values():
        df = pro.daily(ts_code=code, start_date=start_date)
        df.to_csv(os.path.join(data_dir, f"{code}.csv"), index=False)
