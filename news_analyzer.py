"""
news_analyzer.py - 한국 주식 뉴스 수집 및 Gemini LLM 감성 분석
"""
# pyrefly: ignore [missing-import]
import feedparser
import json
import os
import hashlib
import re
from datetime import datetime

from google import genai
from google.genai import types


DEFAULT_GEMINI_MODEL = 'gemini-3.8-flash'


class NewsAnalyzer:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get('GEMINI_API_KEY', '')
        self.model = os.environ.get('GEMINI_MODEL', DEFAULT_GEMINI_MODEL)
        self.naver_client_id = os.environ.get('NAVER_CLIENT_ID', '')
        self.naver_client_secret = os.environ.get('NAVER_CLIENT_SECRET', '')
        
        self.client = None
        if self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
                print(f"[NewsAnalyzer] Gemini API 연결 성공 (모델: {self.model})")
            except Exception as e:
                print(f"[NewsAnalyzer] Gemini API 연결 실패: {e}")
        else:
            print("[NewsAnalyzer] ⚠️ GEMINI_API_KEY 없음 → 뉴스 감성 분석 비활성화")
            
        if not self.naver_client_id:
            print("[NewsAnalyzer] ⚠️ NAVER_CLIENT_ID 없음 → 네이버 뉴스 API 비활성화 (기존 RSS 방식 사용)")
            
        self._cache = {}

    def fetch_news(self, ticker: str, company_name: str) -> list:
        """네이버 뉴스 검색 API 및 Google/Naver RSS로 한국어 뉴스 수집"""
        import requests
        news_items = []
        
        # 1) 네이버 검색 API (가장 안정적)
        if self.naver_client_id and self.naver_client_secret:
            try:
                url = "https://openapi.naver.com/v1/search/news.json"
                headers = {
                    "X-Naver-Client-Id": self.naver_client_id,
                    "X-Naver-Client-Secret": self.naver_client_secret
                }
                params = {
                    "query": f"{company_name} 주식",
                    "display": 10,
                    "sort": "sim"  # date(최신순) 또는 sim(정확도순)
                }
                resp = requests.get(url, headers=headers, params=params, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    for item in data.get('items', []):
                        # HTML 태그 제거 및 특수문자 변환
                        title = re.sub('<[^<]+?>', '', item.get('title', '')).replace('&quot;', '"').replace('&apos;', "'").replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
                        summary = re.sub('<[^<]+?>', '', item.get('description', '')).replace('&quot;', '"').replace('&apos;', "'").replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
                        if title:
                            news_items.append({
                                'title': title,
                                'summary': summary[:300],
                                'published': item.get('pubDate', ''),
                                'source': 'naver_api'
                            })
                    if news_items:
                        print(f"[NewsAnalyzer] {ticker}({company_name}) 네이버 API 뉴스 {len(news_items)}건 수집")
                        return news_items
                else:
                    print(f"[NewsAnalyzer] 네이버 API 요청 실패: {resp.status_code}")
            except Exception as e:
                print(f"[NewsAnalyzer] 네이버 API 오류: {e}")

        # 2) Fallback: Google News RSS (한국어)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
        }
        urls = [
            f"https://news.google.com/rss/search?q={company_name}+주식&hl=ko&gl=KR&ceid=KR:ko",
            f"https://news.google.com/rss/search?q={company_name}&hl=ko&gl=KR&ceid=KR:ko",
        ]
        for url in urls:
            try:
                resp = requests.get(url, headers=headers, timeout=5)
                if resp.status_code == 200:
                    feed = feedparser.parse(resp.content)
                    for entry in feed.entries[:8]:
                        title = entry.get('title', '')
                        summary = re.sub('<[^<]+?>', '', entry.get('summary', ''))
                        if title:
                            news_items.append({
                                'title': title,
                                'summary': summary[:300],
                                'published': entry.get('published', ''),
                                'source': 'google_news_kr'
                            })
                    if news_items:
                        break
            except Exception as e:
                print(f"[NewsAnalyzer] RSS 수집 오류: {e}")

        # 2) 네이버 금융 뉴스 RSS (코드 기반)
        code = ticker.replace('.KS', '').replace('.KQ', '')
        naver_url = f"https://finance.naver.com/rss/item.naver?code={code}"
        try:
            resp = requests.get(naver_url, headers=headers, timeout=5)
            if resp.status_code == 200:
                # 네이버 RSS는 EUC-KR 인코딩일 수 있으므로 feedparser 내부 처리 유도
                feed = feedparser.parse(resp.content)
                for entry in feed.entries[:5]:
                    title = entry.get('title', '')
                    summary = re.sub('<[^<]+?>', '', entry.get('description', ''))
                    if title:
                        news_items.append({
                            'title': title,
                            'summary': summary[:300],
                            'published': entry.get('published', ''),
                            'source': 'naver_finance'
                        })
        except Exception as e:
            print(f"[NewsAnalyzer] 네이버 RSS 오류: {e}")

        print(f"[NewsAnalyzer] {ticker}({company_name}) 뉴스 {len(news_items)}건 수집")
        return news_items

    def analyze_sentiment(self, news_items: list, ticker: str, company_name: str) -> dict:
        """Gemini로 뉴스 감성 점수 개별 산출 및 평균 계산"""
        # 1시간 단위 캐시
        cache_key = hashlib.md5(
            f"{ticker}_individual_{datetime.now().strftime('%Y%m%d%H')}".encode()
        ).hexdigest()
        if cache_key in self._cache:
            print(f"[NewsAnalyzer] 캐시 히트: {ticker}")
            return self._cache[cache_key]

        if not self.client or not news_items:
            return {'score': 0.0, 'reason': 'Gemini 미연결 또는 뉴스 없음', 'key_topics': []}

        total_score = 0.0
        analyzed_count = 0
        all_key_topics = []
        
        # API 호출 제한 및 테스트를 위해 최신 뉴스 딱 1건만 개별 분석
        target_news = news_items[:1]
        print(f"[NewsAnalyzer] {ticker} - {len(target_news)}건의 뉴스만 분석 테스트를 시작합니다...")
        
        for item in target_news:
            prompt = f"""당신은 주식 투자 전문 애널리스트입니다.
아래는 '{company_name}({ticker})' 종목에 관한 단일 뉴스 기사입니다.
이 뉴스가 단기 주가에 미치는 영향을 분석해주세요.

제목: {item['title']}
내용: {item.get('summary', '')}

반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트 없이 JSON만:
{{
  "score": 0.0,
  "reason": "판단 근거 한 줄 요약",
  "key_topics": ["키워드1", "키워드2"],
  "sentiment": "긍정/부정/중립"
}}

score 기준: -1.0(매우 부정) ~ 0.0(중립) ~ +1.0(매우 긍정)"""

            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt
                )
                text = response.text.strip()
                if '```json' in text:
                    text = text.split('```json')[1].split('```')[0].strip()
                elif '```' in text:
                    text = text.split('```')[1].split('```')[0].strip()
                match = re.search(r'\{.*\}', text, re.DOTALL)
                if match:
                    text = match.group()
                    
                result = json.loads(text)
                score = float(result.get('score', 0.0))
                score = max(-1.0, min(1.0, score))
                
                total_score += score
                analyzed_count += 1
                all_key_topics.extend(result.get('key_topics', []))
                
                print(f"  -> 개별 분석 완료: score={score:.2f} | {item['title'][:20]}...")
            except Exception as e:
                print(f"  -> 개별 분석 실패: {e}")
                
        if analyzed_count == 0:
            return {'score': 0.0, 'reason': '모든 개별 기사 분석 실패', 'key_topics': []}
            
        avg_score = total_score / analyzed_count
        avg_score = max(-1.0, min(1.0, avg_score))
        
        final_sentiment = "긍정" if avg_score >= 0.2 else "부정" if avg_score <= -0.2 else "중립"
        unique_topics = list(set(all_key_topics))[:5]
        
        final_result = {
            'score': avg_score,
            'reason': f"{analyzed_count}건의 개별 뉴스 분석 후 평균 반영",
            'key_topics': unique_topics,
            'sentiment': final_sentiment
        }
        
        self._cache[cache_key] = final_result
        print(f"[NewsAnalyzer] {ticker} 최종 평균 감성 점수: {avg_score:.2f}")
        return final_result

    def get_news_score(self, ticker: str, company_name: str) -> dict:
        """전체 뉴스 분석 파이프라인 실행"""
        news_items = self.fetch_news(ticker, company_name)
        sentiment = self.analyze_sentiment(news_items, ticker, company_name)
        return {
            'score': sentiment.get('score', 0.0),
            'reason': sentiment.get('reason', ''),
            'key_topics': sentiment.get('key_topics', []),
            'sentiment': sentiment.get('sentiment', '중립'),
            'news_count': len(news_items),
            'recent_headlines': [item['title'] for item in news_items[:4]]
        }
