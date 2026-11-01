"""
probability_engine.py - 다층 신호 통합 → 실현가능성% + 매수/매도 수치 (-1~+1)
"""
import numpy as np

from analyzer import SIGNAL_FEATURES, score_col


class ProbabilityEngine:
    """
    입력: 기술(추세) 점수, 지표(오실레이터) 점수, 뉴스 감성 점수, 옵션 확률, 목표 수익률, 홀딩 기간
    출력: feasibility_pct (0~100), buy_sell_score (-1~+1)

    기간별 가중치 (거래일 기준):
      단기(≤30일):  뉴스 30% / 기술 40% / 지표 25% / 옵션  5%
      장기(>30일):  뉴스 25% / 기술 30% / 지표 25% / 옵션 20%  ← 우선 중기 가중치 사용
    옵션 데이터가 없으면(한국 주식) 옵션 비중은 나머지에 비례 배분.

    역사적 변동성(HV) 확률은 매수/매도 점수에 넣지 않고,
    최종 실현가능성에서만 모델 확률과 50:50으로 블렌딩 (이중 반영 방지).
    """

    SHORT_TIME_WEIGHTS = {'news': 0.30, 'technical': 0.40, 'indicator': 0.25, 'options': 0.05}
    MID_TIME_WEIGHTS   = {'news': 0.25, 'technical': 0.30, 'indicator': 0.25, 'options': 0.20}

    @classmethod
    def get_time_weights(cls, holding_days: float) -> dict:
        if holding_days <= 30:
            return cls.SHORT_TIME_WEIGHTS.copy()
        return cls.MID_TIME_WEIGHTS.copy()

    @staticmethod
    def sigmoid(x: float, scale: float = 2.5) -> float:
        return 1.0 / (1.0 + np.exp(-x * scale))

    @staticmethod
    def apply_ticker_weights(latest_row, ticker_weights: dict, features=None) -> float:
        """
        머신러닝 최적화 가중치를 신호 점수(score_* 컬럼, StrategyAnalyzer.generate_signals 산출)에 적용.
        features를 주면 해당 그룹의 신호만 사용하고 그룹 내에서 가중치를 재정규화.
        """
        features = features or SIGNAL_FEATURES
        weights = {f: ticker_weights.get(f, 0.0) for f in features}
        total = sum(weights.values())
        if total <= 0:
            return 0.0

        signal = 0.0
        for f, w in weights.items():
            val = latest_row.get(score_col(f), 0.0)
            val = 0.0 if val is None or np.isnan(val) else float(val)
            signal += val * w
        return float(np.clip(signal / total, -1.0, 1.0))

    @classmethod
    def compute(
        cls,
        technical_score: float,
        news_score: float,
        indicator_score: float,
        options_probability,
        target_return_pct: float,
        holding_days: float,
        hist_vol_probability=None,
    ) -> dict:
        """최종 분석 계산."""
        weights = cls.get_time_weights(holding_days)

        # 옵션 점수 (-1~+1 변환) — 실제 옵션 데이터가 있을 때만
        if options_probability is not None:
            options_score = (options_probability - 0.5) * 2.0
        else:
            options_score = None
            weights['options'] = 0.0

        # 합계가 1이 되도록 정규화 (빠진 옵션 비중은 비례 배분)
        total_w = sum(weights.values()) or 1.0
        weights = {k: v / total_w for k, v in weights.items()}

        # 가중 종합 매수/매도 점수
        buy_sell_score = (
            news_score      * weights['news'] +
            technical_score * weights['technical'] +
            indicator_score * weights['indicator'] +
            (options_score or 0.0) * weights['options']
        )
        buy_sell_score = float(np.clip(buy_sell_score, -1.0, 1.0))

        # 방향 정렬: 목표 수익률이 양수면 매수 방향 점수를 확률로
        if target_return_pct >= 0:
            direction_score = buy_sell_score
        else:
            direction_score = -buy_sell_score

        # sigmoid → 확률 (%)
        base_prob = cls.sigmoid(direction_score) * 100.0

        # 역사적 변동성 확률과 블렌딩
        if hist_vol_probability is not None:
            hv_prob_pct = hist_vol_probability * 100.0
            # 모델 신호 50% + 통계적 확률 50%
            feasibility_pct = 0.5 * base_prob + 0.5 * hv_prob_pct
        else:
            feasibility_pct = base_prob

        feasibility_pct = float(np.clip(feasibility_pct, 1.0, 99.0))

        # 기간 레이블
        term = '단기' if holding_days <= 30 else '장기'
        period_label = f"{term} ({int(holding_days)}거래일 · 목표 {target_return_pct:+g}%)"

        return {
            'feasibility_pct': round(feasibility_pct, 1),
            'buy_sell_score': round(buy_sell_score, 3),
            'model_probability': round(base_prob, 1),
            'weights_used': weights,
            'breakdown': {
                'news': round(news_score, 3),
                'technical': round(technical_score, 3),
                'indicator': round(indicator_score, 3),
                'options': None if options_score is None else round(options_score, 3),
            },
            'holding_period': period_label,
        }
