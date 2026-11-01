import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

class ConsultingReport:
    @staticmethod
    def generate_chart(df: pd.DataFrame, ticker: str, stock_name: str = ""):
        # 종목명이 있으면 '삼성전자(005930.KS)' 형식으로, 없으면 티커만 표시
        display_name = f"{stock_name}({ticker})" if stock_name else ticker

        # Create subplots
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            vertical_spacing=0.03,
                            subplot_titles=(f'{display_name} Price & Bollinger Bands', 'MACD', 'RSI'),
                            row_width=[0.2, 0.2, 0.6])

        # Candlestick
        fig.add_trace(go.Candlestick(x=df.index,
                                     open=df['Open'], high=df['High'],
                                     low=df['Low'], close=df['Close'],
                                     name='Price'), row=1, col=1)

        # Bollinger Bands
        if 'BBU_20_2.0' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['BBU_20_2.0'], line=dict(color='gray', width=1), name='Upper Band'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BBL_20_2.0'], line=dict(color='gray', width=1), name='Lower Band'), row=1, col=1)

        # EMAs
        if 'EMA_20' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], line=dict(color='orange', width=1.5), name='EMA 20'), row=1, col=1)
        if 'EMA_60' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_60'], line=dict(color='blue', width=1.5), name='EMA 60'), row=1, col=1)

        # MACD
        if 'MACD_12_26_9' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD_12_26_9'], line=dict(color='blue', width=2), name='MACD(12D-26D)'), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MACDs_12_26_9'], line=dict(color='red', width=2), name='Signal'), row=2, col=1)
            fig.add_trace(go.Bar(x=df.index, y=df['MACDh_12_26_9'], name='Histogram'), row=2, col=1)

        # RSI
        if 'RSI_14' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['RSI_14'], line=dict(color='purple', width=2), name='RSI'), row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

        fig.update_layout(title=f'{display_name} Technical Analysis Report',
                          xaxis_rangeslider_visible=False,
                          height=800)
        
        # Save to HTML file
        output_file = f"{ticker}_report.html"
        fig.write_html(output_file)
        print(f"Chart saved to {output_file}")
