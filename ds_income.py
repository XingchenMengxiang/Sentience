from config import CFG
from stock_pool import StockPool, fetch_daily_data

if __name__ == "__main__":
    fetch_daily_data(
        data_dir=CFG.data_dir,
        start_date=CFG.fetch_start_date,
        pool=StockPool(stocks=dict(CFG.stocks)),
    )
