import pandas as pd
from ta.trend import MACD, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange

class IndicatorEngine:
    @staticmethod
    def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
        """
        데이터프레임에 각종 보조지표를 일괄 계산하여 추가합니다 (ta 라이브러리 활용).
        """
        # 원본 데이터 보호
        df = df.copy()
        
        # MACD
        macd = MACD(close=df['Close'], window_slow=26, window_fast=12, window_sign=9)
        df['MACD_12_26_9'] = macd.macd()
        df['MACDs_12_26_9'] = macd.macd_signal()
        df['MACDh_12_26_9'] = macd.macd_diff()
        
        # EMA
        df['EMA_5'] = EMAIndicator(close=df['Close'], window=5).ema_indicator()
        df['EMA_20'] = EMAIndicator(close=df['Close'], window=20).ema_indicator()
        df['EMA_60'] = EMAIndicator(close=df['Close'], window=60).ema_indicator()
        df['EMA_120'] = EMAIndicator(close=df['Close'], window=120).ema_indicator()
        
        # RSI
        df['RSI_14'] = RSIIndicator(close=df['Close'], window=14).rsi()
        
        # Stochastic
        stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
        df['STOCHk_14_3_3'] = stoch.stoch()
        df['STOCHd_14_3_3'] = stoch.stoch_signal()
        
        # Bollinger Bands
        bb = BollingerBands(close=df['Close'], window=20, window_dev=2)
        df['BBL_20_2.0'] = bb.bollinger_lband()
        df['BBU_20_2.0'] = bb.bollinger_hband()
        
        # ATR (Average True Range)
        atr = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)
        df['ATRr_14'] = atr.average_true_range()
        
        # 결측치가 생긴 초반부 데이터 제거
        df.dropna(inplace=True)
        return df
