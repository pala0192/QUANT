import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import random
import FinanceDataReader as fdr

class MarketDataLoader:
    def __init__(self, ticker):
        self.ticker = ticker
        
    def fetch_data(self):
        """최근 3년치 데이터를 자동으로 수집합니다."""
        print(f"[{self.ticker}] Data fetch: 3 years")
        
        df = yf.download(self.ticker, period="3y", progress=False)
        if df.empty:
            raise ValueError(f"No data found for {self.ticker}")
        
        # MultiIndex columns 처리 (yfinance 최신 버전 호환성)
        if isinstance(df.columns, pd.MultiIndex):
            # level 1 이 ticker 인 경우 제거
            df.columns = df.columns.droplevel(1)
            
        df.dropna(inplace=True)
        return df
