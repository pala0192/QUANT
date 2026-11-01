import pandas as pd

class BacktestSimulator:
    def __init__(self, initial_capital=10000000):
        self.initial_capital = initial_capital
        
    def run(self, df: pd.DataFrame, weights: dict, threshold=0.3):
        """
        간단한 백테스트 시뮬레이션
        weights: 특정 기간에 대한 지표별 가중치 딕셔너리
        threshold: 매수/매도 시그널 임계값
        """
        from probability_engine import ProbabilityEngine
        capital = self.initial_capital
        position = 0
        history = []
        
        for index, row in df.iterrows():
            price = row['Close']
            score = ProbabilityEngine.apply_ticker_weights(row, weights)
            
            # 매수 시그널
            if score > threshold and capital >= price:
                shares_to_buy = int(capital // price)
                if shares_to_buy > 0:
                    capital -= shares_to_buy * price
                    position += shares_to_buy
                    history.append({'date': index, 'action': 'BUY', 'price': price, 'shares': shares_to_buy})
                    
            # 매도 시그널
            elif score < -threshold and position > 0:
                capital += position * price
                history.append({'date': index, 'action': 'SELL', 'price': price, 'shares': position})
                position = 0
                
        # 최종 자산 가치 계산
        final_value = capital + (position * df.iloc[-1]['Close'])
        return_pct = (final_value - self.initial_capital) / self.initial_capital * 100
        
        return {
            'final_value': final_value,
            'return_pct': return_pct,
            'trade_history': history
        }
