import pandas as pd
import numpy as np
from config import Config

# 신호 그룹: 기술(추세/모멘텀) vs 지표(오실레이터)
TREND_FEATURES = ['macd', 'ema20', 'ema60', 'ema120', 'atr']
OSCILLATOR_FEATURES = ['rsi', 'stoch', 'bollinger']
SIGNAL_FEATURES = OSCILLATOR_FEATURES + TREND_FEATURES


def score_col(feature: str) -> str:
    return f'score_{feature}'


class StrategyAnalyzer:
    def __init__(self, config: Config):
        self.config = config
        
    @staticmethod
    def _normalize_rsi(rsi_series):
        # RSI 30 이하 과매도(매수시그널 +1), 70 이상 과매수(매도시그널 -1)
        score = (50 - rsi_series) / 20.0
        return np.clip(score, -1.0, 1.0)

    @staticmethod
    def _normalize_stoch(k_series, d_series):
        # K가 D를 상향 돌파하면 매수, 하향 돌파하면 매도
        diff = k_series - d_series
        return np.clip(diff / 10.0, -1.0, 1.0)

    @staticmethod
    def _normalize_macd(macd_series, signal_series):
        diff = macd_series - signal_series
        return np.clip(diff / (macd_series.abs() + 0.001), -1.0, 1.0)

    @staticmethod
    def _normalize_bollinger(price_series, lower_series, upper_series):
        # 하단 터치 시 +1, 상단 터치 시 -1
        mid = (lower_series + upper_series) / 2
        score = (mid - price_series) / (upper_series - lower_series) * 2
        return np.clip(score, -1.0, 1.0)

    @staticmethod
    def _normalize_ema(price_series, ema_series):
        diff_pct = (price_series - ema_series) / ema_series * 100
        return np.clip(diff_pct / 5.0, -1.0, 1.0)

    @classmethod
    def compute_signal_scores(cls, df: pd.DataFrame) -> pd.DataFrame:
        """신호별 -1~+1 점수 (학습/분석 공통 단일 소스)"""
        scores = pd.DataFrame(index=df.index)
        scores['rsi'] = cls._normalize_rsi(df['RSI_14'])
        scores['stoch'] = cls._normalize_stoch(df['STOCHk_14_3_3'], df['STOCHd_14_3_3'])
        scores['macd'] = cls._normalize_macd(df['MACD_12_26_9'], df['MACDs_12_26_9'])
        scores['bollinger'] = cls._normalize_bollinger(df['Close'], df['BBL_20_2.0'], df['BBU_20_2.0'])
        scores['ema20'] = cls._normalize_ema(df['Close'], df['EMA_20'])
        scores['ema60'] = cls._normalize_ema(df['Close'], df['EMA_60'])
        scores['ema120'] = cls._normalize_ema(df['Close'], df['EMA_120'])

        # [허점 2 수정] ATR 편향 오류 수정: 가격의 20일 변동 방향성을 ATR로 나누어 방향과 변동성을 동시 고려
        price_momentum = df['Close'] - df['Close'].shift(20)
        scores['atr'] = np.clip(price_momentum / (df['ATRr_14'] + 0.001), -1.0, 1.0).fillna(0)
        return scores

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """전체 데이터프레임에 신호별 점수 컬럼(score_*)을 추가합니다. (Vectorized)"""
        df = df.copy()
        scores = self.compute_signal_scores(df)
        for f in SIGNAL_FEATURES:
            df[score_col(f)] = scores[f]
        return df

class ConsultingInterpreter:
    @staticmethod
    def interpret(row, holding_days=20, target_return_pct=5.0):
        """특정 시점의 데이터를 바탕으로 자연어 해석과 기대 수익을 산출합니다."""
        feedback = []
        
        # 1. RSI 분석
        rsi = row['RSI_14']
        if rsi >= 70:
            feedback.append("⚠️ [RSI(14D)] 현재 70 이상으로 과매수 구간입니다. 단기 하락 조정 가능성이 높습니다.")
        elif rsi <= 30:
            feedback.append("✅ [RSI(14D)] 현재 30 이하로 과매도 구간입니다. 단기적인 기술적 반등이 기대됩니다.")
        else:
            feedback.append(f"ℹ️ [RSI(14D)] 현재 {rsi:.1f}로 중립적인 모멘텀입니다.")
            
        # 2. 볼린저 밴드 분석
        price = row['Close']
        bbu = row['BBU_20_2.0']
        bbl = row['BBL_20_2.0']
        if price >= bbu:
            feedback.append("⚠️ [볼린저 밴드(20D, 2.0σ)] 주가가 상단 밴드를 터치/돌파했습니다. 강한 저항 및 매도 압력이 예상됩니다.")
        elif price <= bbl:
            feedback.append("✅ [볼린저 밴드(20D, 2.0σ)] 주가가 하단 밴드를 터치/돌파했습니다. 지지선 역할을 하여 반등할 확률이 높습니다.")
            
        # 3. MACD 분석
        macd = row['MACD_12_26_9']
        signal = row['MACDs_12_26_9']
        if macd > signal and macd > 0:
            feedback.append("📈 [MACD(12D-26D-9D)] MACD가 시그널 선 위에 위치하여 상승 추세가 견고합니다.")
        elif macd < signal and macd < 0:
            feedback.append("📉 [MACD(12D-26D-9D)] MACD가 시그널 선 아래 위치하여 하락 추세가 진행 중입니다.")

        # 4. 목표가 및 손절가 산출 (기간별 고정 목표 수익률 기준)
        target_price = price * (1 + target_return_pct / 100.0)
        stop_loss = price * (1 - target_return_pct / 100.0 * 0.5) # 손익비 2:1 가정
        
        expected_return_pct = (target_price - price) / price * 100
        risk_pct = (price - stop_loss) / price * 100
        
        return {
            'text': "\n".join(feedback),
            'current_price': price,
            'target_price': target_price,
            'stop_loss': stop_loss,
            'expected_return_pct': expected_return_pct,
            'risk_pct': risk_pct
        }


