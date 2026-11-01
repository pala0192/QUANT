"""
options_analyzer.py - 옵션 가격 및 역사적 변동성 기반 확률 계산
- 한국 주식: 역사적 변동성(HV) + log-normal 모델
- 미국 주식: yfinance 옵션 체인 + Black-Scholes
"""
import numpy as np
import pandas as pd
from scipy.stats import norm


class OptionsAnalyzer:

    @staticmethod
    def get_historical_vol_probability(
        df: pd.DataFrame,
        target_return_pct: float,
        holding_days: int
    ) -> dict:
        """
        역사적 변동성 기반 실현 확률 계산 (한국 주식 기본 방법)
        log-normal 분포 가정: P(R >= target)
        """
        try:
            log_returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
            if len(log_returns) < 20:
                return {'probability': 0.5, 'historical_vol': 0.0, 'method': 'fallback'}

            daily_vol = log_returns.std()
            annual_vol = daily_vol * np.sqrt(252)
            T = holding_days / 252

            # 최근 60일 실현 수익률(연환산)
            mu_daily = log_returns.tail(60).mean()
            mu_annual = mu_daily * 252

            target_log_return = np.log(1 + target_return_pct / 100.0)

            # 실세계 확률 (historical drift 사용)
            if annual_vol > 0 and T > 0:
                d_real = (mu_annual * T - target_log_return) / (annual_vol * np.sqrt(T))
                prob_real = float(norm.cdf(d_real))

                # 위험중립 확률 (drift=0 가정, 보수적)
                d_rn = -target_log_return / (annual_vol * np.sqrt(T))
                prob_neutral = float(norm.cdf(d_rn))
            else:
                prob_real = 0.5
                prob_neutral = 0.5

            # 60% 실세계 + 40% 위험중립 혼합
            probability = 0.6 * prob_real + 0.4 * prob_neutral
            probability = float(np.clip(probability, 0.01, 0.99))

            return {
                'probability': probability,
                'historical_vol': round(annual_vol * 100, 2),
                'mu_annual': round(mu_annual * 100, 2),
                'method': 'historical_vol_lognormal',
                'data_source': f'역사적 변동성 기반 (연간 {annual_vol*100:.1f}%)'
            }
        except Exception as e:
            print(f"[OptionsAnalyzer] HV 계산 오류: {e}")
            return {'probability': 0.5, 'historical_vol': 0.0, 'method': 'error'}
