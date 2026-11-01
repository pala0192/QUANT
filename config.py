"""
config.py - 시장별 + 종목별 + 기간별 가중치 관리
"""
import json
import os

# 고정 홀딩 기간 (레이블 → 거래일 수)
# 일수는 모두 '거래일' 기준 (OptionsAnalyzer의 T=days/252, train.py의 shift(-days)와 일치)
HOLDING_PERIODS = {
    'short': 20,    # 단기: 약 1개월
    'long':  100,   # 장기: 약 5개월
}

# 기간별 고정 목표 수익률 (%)
TARGET_RETURNS = {
    'short': 5.0,
    'long':  13.0,
}

PERIOD_NAMES = {
    'short': '단기',
    'long':  '장기',
}

DEFAULT_PERIOD = 'short'


def period_label_to_days(label: str) -> float:
    """'short' → 20, 'long' → 100 (거래일)"""
    return HOLDING_PERIODS.get(label, HOLDING_PERIODS[DEFAULT_PERIOD])


def target_return_for(label: str) -> float:
    """기간 레이블별 고정 목표 수익률(%)"""
    return TARGET_RETURNS.get(label, TARGET_RETURNS[DEFAULT_PERIOD])


# 기간별 기본 신호 가중치 (장기는 우선 중기 가중치 사용)
_DEFAULT_SIGNAL_WEIGHTS = {
    'short': {'macd': 0.40, 'ema20': 0.30, 'rsi': 0.20, 'bollinger': 0.10},   # 중단기
    'long':  {'macd': 0.35, 'ema20': 0.25, 'ema60': 0.20, 'rsi': 0.20},       # 중기
}


def _default_weights_for(label: str) -> dict:
    return _DEFAULT_SIGNAL_WEIGHTS.get(label, _DEFAULT_SIGNAL_WEIGHTS[DEFAULT_PERIOD]).copy()


class Config:
    TICKER_WEIGHT_FILE = 'ticker_weights.json'

    def __init__(self):
        self.investment_mode = 'AUTO'
        self.initial_capital = 10_000_000

        # 기간(단기/장기)별 독립 가중치 저장소
        self.base_weights = {label: _default_weights_for(label) for label in HOLDING_PERIODS}

        # 종목별 + 기간별 가중치: {ticker: {period_label: {signal: weight}}}
        self._ticker_weights: dict = {}

    # ── 기본 가중치 ──────────────────────────────────────────────────────────
    def get_base_weights(self) -> dict:
        return self.base_weights

    def update_weights(self, period: str, new_weights: dict):
        target = self.get_base_weights()
        if period in target:
            target[period].update(new_weights)

    # ── 종목+기간 가중치 ─────────────────────────────────────────────────────
    def get_ticker_period_weights(self, ticker: str, period_label: str) -> dict:
        """
        종목+기간별 저장된 가중치를 반환.
        없으면 기간별 기본 가중치 반환.
        """
        default = self.base_weights.get(period_label, _default_weights_for(period_label))
        return self._ticker_weights.get(ticker, {}).get(period_label, default).copy()

    def update_ticker_period_weights(self, ticker: str, period_label: str, new_weights: dict):
        """종목+기간 가중치 업데이트 (AI 옵티마이저 호출용)"""
        if ticker not in self._ticker_weights:
            self._ticker_weights[ticker] = {}
        existing = self.get_ticker_period_weights(ticker, period_label)
        existing.update(new_weights)
        self._ticker_weights[ticker][period_label] = existing

    def get_all_ticker_weights(self) -> dict:
        return self._ticker_weights

    # ── 머신러닝 가중치 저장/로드 ─────────────────────────────────────────────────────────────

    def save_ticker_weights(self, filepath=None):
        filepath = filepath or self.TICKER_WEIGHT_FILE
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self._ticker_weights, f, indent=2, ensure_ascii=False)
        print(f"[Config] 종목별 가중치 저장 → {filepath}")

    def load_ticker_weights(self, filepath=None) -> bool:
        filepath = filepath or self.TICKER_WEIGHT_FILE
        if not os.path.exists(filepath):
            return False
        with open(filepath, 'r', encoding='utf-8') as f:
            self._ticker_weights = json.load(f)
        print(f"[Config] 종목별 가중치 로드 ✓ ({len(self._ticker_weights)} 종목)")
        return True
