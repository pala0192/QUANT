"""
DMserver.py - 이커머스 고객 이탈 분석 Flask 서버
실행: python DMserver.py
접속: http://localhost:5001/dm/
"""
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

from DMconfig import DM_PORT
from DMdata_fetcher import ChurnDataFetcher
from DMchurn_analyzer import ChurnAnalyzer

app = Flask(__name__, template_folder='templates')
CORS(app)

fetcher = ChurnDataFetcher()
analyzer = ChurnAnalyzer()


@app.route('/dm/')
def index():
    return render_template('DMindex.html')


@app.route('/dm/api/news-preview')
def news_preview():
    try:
        articles = fetcher.fetch_churn_news()
        return jsonify({'success': True, 'count': len(articles), 'articles': articles})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/dm/api/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'CSV 파일이 없습니다.'}), 400

    f = request.files['file']
    if not f.filename.endswith('.csv'):
        return jsonify({'success': False, 'error': 'CSV 파일만 업로드 가능합니다.'}), 400

    file_bytes = f.read()
    news_items = fetcher.fetch_churn_news()
    result = analyzer.analyze(file_bytes, news_items)

    if 'error' in result:
        return jsonify({'success': False, 'error': result['error']}), 500

    return jsonify({'success': True, 'news_count': len(news_items), **result})


if __name__ == '__main__':
    print(f"[DM] 이커머스 고객 이탈 분석 서버 시작 → http://localhost:{DM_PORT}/dm/")
    app.run(debug=True, port=DM_PORT)
