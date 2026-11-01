"""
DMconfig.py - 고객 이탈 분석 앱 설정
"""
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
DM_PORT = 5001

RSS_SOURCES = [
    {
        'url': 'https://news.google.com/rss/search?q=%EC%9D%B4%EC%BB%A4%EB%A8%B8%EC%8A%A4+%EA%B3%A0%EA%B0%9D%EC%9D%B4%ED%83%88&hl=ko&gl=KR&ceid=KR:ko',
        'label': '이커머스 고객이탈 (한국)'
    },
    {
        'url': 'https://news.google.com/rss/search?q=%EC%87%BC%ED%95%91%EB%AA%B0+%EC%9D%B4%ED%83%88+%EC%9B%90%EC%9D%B8&hl=ko&gl=KR&ceid=KR:ko',
        'label': '쇼핑몰 이탈 원인 (한국)'
    },
    {
        'url': 'https://news.google.com/rss/search?q=ecommerce+customer+churn+retention&hl=en&gl=US&ceid=US:en',
        'label': 'ecommerce churn (글로벌)'
    },
]

MAX_NEWS_PER_SOURCE = 5
CACHE_TTL_HOURS = 1
