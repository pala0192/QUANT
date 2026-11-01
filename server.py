"""
server.py - 퀀트 트레이딩 분석 Flask REST API 서버
실행: python server.py
접속: http://localhost:5000
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from config import (HOLDING_PERIODS, TARGET_RETURNS, PERIOD_NAMES, DEFAULT_PERIOD,
                    period_label_to_days, target_return_for)

import os
import json
import traceback
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS

from config import Config
from data_loader import MarketDataLoader
from indicators import IndicatorEngine
from analyzer import StrategyAnalyzer, ConsultingInterpreter, TREND_FEATURES, OSCILLATOR_FEATURES
from news_analyzer import NewsAnalyzer
from options_analyzer import OptionsAnalyzer
from probability_engine import ProbabilityEngine

# ── 앱 초기화 ──────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder='templates')
CORS(app)

# Config 및 가중치 로드
config = Config()
config.load_ticker_weights()  # 종목별+기간별 가중치(머신러닝 결과) 로드

# Gemini API Key 설정
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
news_analyzer = NewsAnalyzer(api_key=GEMINI_API_KEY)

# ── 분석 대상 5개 종목 ─────────────────────────────────────────────────────
KR_WATCHLIST = {
    '373220.KS': 'LG에너지솔루션',   # 배터리
    '034020.KS': '두산에너빌리티',    # 원자력
    '068270.KS': '셀트리온',          # 바이오
    '138040.KS': '메리츠금융지주',    # 증권
    '003230.KS': '삼양식품',          # 식품
}

# 섹터 정보 (UI 표시용)
KR_SECTORS = {
    '373220.KS': '🔋 배터리',
    '034020.KS': '⚛️ 원자력',
    '068270.KS': '🧬 바이오',
    '138040.KS': '📊 증권',
    '003230.KS': '🍜 식품',
}


# ── 헬퍼 함수 ─────────────────────────────────────────────────────────────
def _check_macd_alert(latest_row, df: pd.DataFrame) -> dict:
    """MACD 데드크로스/골든크로스 감지"""
    macd = float(latest_row.get('MACD_12_26_9', 0) or 0)
    signal = float(latest_row.get('MACDs_12_26_9', 0) or 0)
    if len(df) >= 2:
        prev = df.iloc[-2]
        prev_macd = float(prev.get('MACD_12_26_9', 0) or 0)
        prev_signal = float(prev.get('MACDs_12_26_9', 0) or 0)
        death_cross = prev_macd >= prev_signal and macd < signal
        golden_cross = prev_macd <= prev_signal and macd > signal
        if death_cross:
            return {
                'type': 'sell', 'triggered': True,
                'message': '⚠️ MACD 데드크로스 발생! 매도 시그널',
                'macd': round(macd, 4), 'signal': round(signal, 4)
            }
        elif golden_cross:
            return {
                'type': 'buy', 'triggered': True,
                'message': '✅ MACD 골든크로스 발생! 매수 시그널',
                'macd': round(macd, 4), 'signal': round(signal, 4)
            }

    current_state = 'MACD > Signal (상승세)' if macd > signal else 'MACD < Signal (하락세)'
    return {
        'type': 'none', 'triggered': False,
        'message': current_state,
        'macd': round(macd, 4), 'signal': round(signal, 4)
    }


def _safe_float(val):
    """NaN/None 안전 float 변환"""
    if val is None:
        return None
    try:
        v = float(val)
        return None if np.isnan(v) or np.isinf(v) else v
    except Exception:
        return None


def _build_chart_data(df: pd.DataFrame) -> dict:
    """Plotly용 차트 데이터 직렬화"""
    def to_list(series):
        return [_safe_float(v) for v in series]

    return {
        'dates':      [d.strftime('%Y-%m-%d') for d in df.index],
        'open':       to_list(df['Open']),
        'high':       to_list(df['High']),
        'low':        to_list(df['Low']),
        'close':      to_list(df['Close']),
        'volume':     to_list(df['Volume']),
        'ema20':      to_list(df['EMA_20'])    if 'EMA_20'       in df.columns else [],
        'ema60':      to_list(df['EMA_60'])    if 'EMA_60'       in df.columns else [],
        'bbu':        to_list(df['BBU_20_2.0']) if 'BBU_20_2.0'  in df.columns else [],
        'bbl':        to_list(df['BBL_20_2.0']) if 'BBL_20_2.0'  in df.columns else [],
        'macd':       to_list(df['MACD_12_26_9'])  if 'MACD_12_26_9'  in df.columns else [],
        'macd_signal':to_list(df['MACDs_12_26_9']) if 'MACDs_12_26_9' in df.columns else [],
        'macd_hist':  to_list(df['MACDh_12_26_9']) if 'MACDh_12_26_9' in df.columns else [],
        'rsi':        to_list(df['RSI_14'])    if 'RSI_14'       in df.columns else [],
    }


def _load_and_analyze(ticker: str):
    """데이터 로드 및 기술적 분석 공통 로직"""
    loader = MarketDataLoader(ticker)
    raw_df = loader.fetch_data()
    full_df = IndicatorEngine.calculate_all(raw_df)

    cutoff = datetime.now() - timedelta(days=180)
    display_df = full_df[full_df.index >= cutoff]
    if len(display_df) == 0:
        display_df = full_df

    analyzer = StrategyAnalyzer(config)
    analyzed_df = analyzer.generate_signals(display_df)
    return full_df, analyzed_df


# ── 라우트 ─────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/watchlist')
def watchlist():
    """사전 정의된 한국 주요 종목 목록 반환"""
    return jsonify({
        'success': True, 
        'watchlist': KR_WATCHLIST,
        'sectors': KR_SECTORS
    })


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    메인 분석 엔드포인트
    Input (JSON): { ticker, period_label: 'short' | 'long' }
    목표 수익률은 기간별로 고정 (config.TARGET_RETURNS)
    Output: 실현가능성%, 매수매도 점수, 분석 세부내역, 차트 데이터
    """
    try:
        data = request.get_json(force=True)
        ticker            = str(data.get('ticker', '373220.KS')).strip().upper()
        period_label      = str(data.get('period_label', DEFAULT_PERIOD))

        if not ticker:
            return jsonify({'success': False, 'error': '티커를 입력해주세요'}), 400
        if period_label not in HOLDING_PERIODS:
            return jsonify({'success': False, 'error': f"기간은 {list(HOLDING_PERIODS)} 중 하나여야 합니다"}), 400

        holding_days      = period_label_to_days(period_label)     # 거래일
        target_return_pct = target_return_for(period_label)

        print(f"\n[Server] 분석 요청: {ticker} | {PERIOD_NAMES[period_label]}({holding_days}거래일) | 목표수익 {target_return_pct}%")

        stock_name = KR_WATCHLIST.get(ticker, ticker)

        # 1. 데이터 로드 + 기술 분석
        full_df, analyzed_df = _load_and_analyze(ticker)
        latest = analyzed_df.iloc[-1]
        current_price = float(latest['Close'])

        # 2. 뉴스 분석 (Gemini)
        news_result = news_analyzer.get_news_score(ticker, stock_name)
        news_score = news_result.get('score', 0.0)

        # 3. 역사적 변동성 기반 확률
        hist_result = OptionsAnalyzer.get_historical_vol_probability(
            full_df, target_return_pct, holding_days
        )
        hist_prob = hist_result.get('probability', None)

        # (한국 주식은 옵션 데이터가 제공되지 않으므로 역사적 변동성만 사용)
        options_prob = None

        # 종목+기간별 저장된 신호 가중치 조회 (머신러닝 결과 또는 기간별 기본값)
        ticker_weights = config.get_ticker_period_weights(ticker, period_label)

        # 5. 기술(추세) / 지표(오실레이터) 점수 — 학습 가중치를 그룹별로 적용
        tech_score = ProbabilityEngine.apply_ticker_weights(latest, ticker_weights, TREND_FEATURES)
        indicator_score = ProbabilityEngine.apply_ticker_weights(latest, ticker_weights, OSCILLATOR_FEATURES)

        # 6. 확률 종합 계산
        result = ProbabilityEngine.compute(
            technical_score=tech_score,
            news_score=news_score,
            indicator_score=indicator_score,
            options_probability=options_prob,
            target_return_pct=target_return_pct,
            holding_days=holding_days,
            hist_vol_probability=hist_prob,
        )

        # 7. MACD 알림
        macd_alert = _check_macd_alert(latest, analyzed_df)

        # 8. 자연어 해석
        interpretation = ConsultingInterpreter.interpret(
            latest, holding_days=holding_days, target_return_pct=target_return_pct
        )

        # 9. 차트 데이터
        chart_data = _build_chart_data(analyzed_df)

        return jsonify({
            'success': True,
            'ticker': ticker,
            'stock_name': stock_name,
            'market': 'KR',
            'current_price': round(current_price, 2),
            'analysis_date': latest.name.strftime('%Y-%m-%d'),
            # 핵심 결과
            'feasibility_pct': result['feasibility_pct'],
            'buy_sell_score': result['buy_sell_score'],
            'period_label': period_label,
            'holding_days': holding_days,
            'target_return_pct': target_return_pct,
            'holding_period': result['holding_period'],
            'weights_used': result['weights_used'],
            'ticker_weights': ticker_weights,
            'breakdown': result['breakdown'],
            'model_probability': result['model_probability'],
            # 뉴스
            'news': {
                'score': round(news_score, 3),
                'reason': news_result.get('reason', ''),
                'key_topics': news_result.get('key_topics', []),
                'sentiment': news_result.get('sentiment', '중립'),
                'headlines': news_result.get('recent_headlines', []),
                'count': news_result.get('news_count', 0),
            },
            # 옵션/변동성
            'volatility': {
                'hist_vol_pct': hist_result.get('historical_vol', 0),
                'hist_probability': round((hist_prob or 0) * 100, 1),
                'options_method': 'n/a',
                'options_probability': None,
                'data_source': hist_result.get('data_source', ''),
            },
            # 알림
            'macd_alert': macd_alert,
            # 자연어 해석
            'interpretation': {
                'text': interpretation.get('text', ''),
                'target_price': round(interpretation.get('target_price', 0), 2),
                'stop_loss': round(interpretation.get('stop_loss', 0), 2),
                'expected_return_pct': round(interpretation.get('expected_return_pct', 0), 2),
                'risk_pct': round(interpretation.get('risk_pct', 0), 2),
            },
            # 차트
            'chart_data': chart_data,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/macd-alert/<path:ticker>')
def macd_alert_check(ticker):
    """MACD 알림 폴링용 경량 엔드포인트 (5분마다 호출)"""
    try:
        ticker = ticker.strip().upper()
        loader = MarketDataLoader(ticker)
        raw_df = loader.fetch_data()
        full_df = IndicatorEngine.calculate_all(raw_df)
        analyzer = StrategyAnalyzer(config)
        analyzed_df = analyzer.generate_signals(full_df)
        latest = analyzed_df.iloc[-1]
        alert = _check_macd_alert(latest, analyzed_df)
        return jsonify({'success': True, 'ticker': ticker, 'macd_alert': alert})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/periods')
def get_periods():
    """고정 홀딩 기간(단기/장기)과 기간별 목표 수익률 반환"""
    periods = [
        {'label': k, 'name': PERIOD_NAMES[k], 'days': v, 'target_return_pct': TARGET_RETURNS[k]}
        for k, v in HOLDING_PERIODS.items()
    ]
    return jsonify({'success': True, 'periods': periods})


@app.route('/api/ticker-weights/<path:ticker>')
def get_ticker_weights(ticker):
    """특정 종목의 모든 기간별 가중치 반환"""
    ticker = ticker.strip().upper()
    weights = {}
    for label in HOLDING_PERIODS:
        weights[label] = config.get_ticker_period_weights(ticker, label)
    return jsonify({'success': True, 'ticker': ticker, 'weights': weights})


@app.route('/api/ticker-weights/<path:ticker>', methods=['POST'])
def update_ticker_weights(ticker):
    """종목+기간별 가중치 수동 업데이트"""
    ticker = ticker.strip().upper()
    data = request.get_json(force=True)
    period_label = data.get('period_label', DEFAULT_PERIOD)
    new_weights  = data.get('weights', {})
    if period_label not in HOLDING_PERIODS:
        return jsonify({'success': False, 'error': f"기간은 {list(HOLDING_PERIODS)} 중 하나여야 합니다"}), 400
    if not new_weights:
        return jsonify({'success': False, 'error': 'weights 필드가 없습니다'}), 400
    config.update_ticker_period_weights(ticker, period_label, new_weights)
    config.save_ticker_weights()
    return jsonify({'success': True, 'ticker': ticker, 'period_label': period_label,
                    'saved': config.get_ticker_period_weights(ticker, period_label)})


@app.route('/api/quick-score/<path:ticker>')
def quick_score(ticker):
    """차트 없이 최신 매수/매도 스코어만 빠르게 반환"""
    try:
        ticker = ticker.strip().upper()
        _, analyzed_df = _load_and_analyze(ticker)
        latest = analyzed_df.iloc[-1]
        return jsonify({
            'success': True,
            'ticker': ticker,
            'score_short': 0.0,
            'score_mid':   0.0,
            'score_long':  0.0,
            'date': latest.name.strftime('%Y-%m-%d'),
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 55)
    print("  📈 퀀트 트레이딩 분석 서버 시작")
    print("  접속 URL: http://localhost:5000")
    print("  GEMINI_API_KEY:", "✓ 설정됨" if GEMINI_API_KEY else "✗ 미설정 (뉴스 분석 비활성화)")
    for k, v in HOLDING_PERIODS.items():
        print(f"  {PERIOD_NAMES[k]}: {v}거래일 / 목표 수익률 {TARGET_RETURNS[k]}%")
    print("=" * 55)
    app.run(host='0.0.0.0', port=5000, debug=True)
