"""
DMdata_fetcher.py - RSS/뉴스 API로 이커머스 고객 이탈 관련 외부 정보 수집
"""
import feedparser
import hashlib
import re
import requests
from datetime import datetime

from DMconfig import RSS_SOURCES, MAX_NEWS_PER_SOURCE, CACHE_TTL_HOURS


class ChurnDataFetcher:
    def __init__(self):
        self._cache = {}
        self._headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    def _cache_key(self) -> str:
        hour = datetime.now().strftime('%Y%m%d%H')
        return hashlib.md5(f"churn_news_{hour}".encode()).hexdigest()

    def fetch_churn_news(self) -> list[dict]:
        key = self._cache_key()
        if key in self._cache:
            print("[ChurnDataFetcher] 캐시 히트")
            return self._cache[key]

        articles = []
        for source in RSS_SOURCES:
            try:
                resp = requests.get(source['url'], headers=self._headers, timeout=6)
                feed = feedparser.parse(resp.content)
                for entry in feed.entries[:MAX_NEWS_PER_SOURCE]:
                    title = entry.get('title', '').strip()
                    summary = re.sub(r'<[^>]+>', '', entry.get('summary', '')).strip()
                    if title:
                        articles.append({
                            'title': title,
                            'summary': summary[:400],
                            'published': entry.get('published', ''),
                            'source': source['label'],
                        })
                print(f"[ChurnDataFetcher] {source['label']}: {min(MAX_NEWS_PER_SOURCE, len(feed.entries))}건")
            except Exception as e:
                print(f"[ChurnDataFetcher] RSS 오류 ({source['label']}): {e}")

        self._cache[key] = articles
        print(f"[ChurnDataFetcher] 총 {len(articles)}건 수집 완료")
        return articles
