import tushare as ts
import pandas as pd
import os

ts.set_token("YOUR_TUSHARE_TOKEN")
pro = ts.pro_api()

codes = [
    "000953.SZ","002369.SZ","601727.SH","002186.SZ",
    "301078.SZ","600623.SH","600845.SH","002902.SZ"
]

os.makedirs("./data", exist_ok=True)

for code in codes:
    df = pro.daily(ts_code=code, start_date="20180101")
    # TuShare 默认字段: trade_date, open, high, low, close, vol, amount
    df.to_csv(f"./data/{code}.csv", index=False)
