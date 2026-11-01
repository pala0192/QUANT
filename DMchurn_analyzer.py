"""
DMchurn_analyzer.py - CSV 파싱 + Gemini API 이커머스 고객 이탈 분석
"""
import io
import json
import re

import pandas as pd
from google import genai

from DMconfig import GEMINI_API_KEY


class ChurnAnalyzer:
    def __init__(self):
        self.client = None
        if GEMINI_API_KEY:
            try:
                self.client = genai.Client(api_key=GEMINI_API_KEY)
                print("[ChurnAnalyzer] Gemini API 연결 성공")
            except Exception as e:
                print(f"[ChurnAnalyzer] Gemini API 연결 실패: {e}")
        else:
            print("[ChurnAnalyzer] GEMINI_API_KEY 없음")

    def _parse_csv(self, file_bytes: bytes) -> tuple[pd.DataFrame, str]:
        """CSV 바이트를 DataFrame으로 파싱하고 통계 요약 문자열 반환"""
        for enc in ('utf-8-sig', 'utf-8', 'cp949', 'euc-kr'):
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
                break
            except Exception:
                continue
        else:
            raise ValueError("CSV 파일을 읽을 수 없습니다. 인코딩을 확인해주세요.")

        lines = [
            f"- 총 고객 수: {len(df)}명",
            f"- 컬럼: {', '.join(df.columns.tolist())}",
        ]
        for col in df.select_dtypes(include='number').columns[:10]:
            s = df[col].dropna()
            lines.append(
                f"- {col}: 평균={s.mean():.2f}, 최소={s.min():.2f}, 최대={s.max():.2f}, 결측={df[col].isna().sum()}"
            )

        # 이탈 컬럼 자동 탐지
        churn_col = next(
            (c for c in df.columns if any(k in c.lower() for k in ('churn', '이탈', 'churned', 'cancel'))),
            None
        )
        if churn_col:
            rate = df[churn_col].apply(lambda x: 1 if str(x).strip() in ('1', 'True', 'true', 'Y', 'yes') else 0).mean()
            lines.append(f"- 실제 이탈률 ('{churn_col}' 기준): {rate*100:.1f}%")

        return df, '\n'.join(lines)

    def _format_news(self, news_items: list[dict]) -> str:
        if not news_items:
            return "(수집된 외부 기사 없음)"
        parts = []
        for i, item in enumerate(news_items[:15], 1):
            parts.append(f"{i}. [{item['source']}] {item['title']}\n   {item['summary'][:200]}")
        return '\n'.join(parts)

    # 모델 우선순위: 할당량이 남아있는 모델을 순서대로 시도
    _MODELS = ['gemini-2.0-flash-lite', 'gemini-2.0-flash', 'gemini-2.5-flash']

    def _call_gemini(self, csv_summary: str, news_text: str) -> dict:
        prompt = f"""당신은 이커머스/쇼핑몰 고객 이탈 분석 전문가입니다.

## 고객 데이터 요약 (업로드된 CSV):
{csv_summary}

## 최신 업계 이탈 관련 뉴스·트렌드 (외부 RSS 수집):
{news_text}

위 데이터를 종합 분석하여 반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트 없이 JSON만:
{{
  "top_churn_factors": [
    {{"factor": "요인명", "weight": 85, "description": "설명"}}
  ],
  "risk_distribution": {{"high_pct": 30, "medium_pct": 45, "low_pct": 25}},
  "recommendations": ["액션1", "액션2", "액션3", "액션4", "액션5"],
  "overall_risk_score": 62,
  "key_insight": "핵심 인사이트 한 문장"
}}

규칙:
- top_churn_factors: 정확히 5개, weight는 0-100 정수
- risk_distribution: 세 값의 합이 반드시 100
- recommendations: 정확히 5개의 구체적 액션
- overall_risk_score: 0(낮음)-100(높음) 정수
- CSV 데이터와 외부 뉴스 트렌드를 모두 반영할 것"""

        last_err = None
        for model in self._MODELS:
            try:
                print(f"[ChurnAnalyzer] 모델 시도: {model}")
                response = self.client.models.generate_content(
                    model=model,
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
                print(f"[ChurnAnalyzer] {model} 성공")
                return json.loads(text)
            except Exception as e:
                msg = str(e)
                if '429' in msg or 'RESOURCE_EXHAUSTED' in msg:
                    print(f"[ChurnAnalyzer] {model} 할당량 초과, 다음 모델 시도")
                    last_err = e
                    continue
                raise

        raise RuntimeError(f"모든 모델 할당량 초과. 마지막 오류: {last_err}")

    def analyze(self, file_bytes: bytes, news_items: list[dict]) -> dict:
        if not self.client:
            return {'error': 'GEMINI_API_KEY가 설정되지 않았습니다.'}

        try:
            df, csv_summary = self._parse_csv(file_bytes)
        except ValueError as e:
            return {'error': str(e)}

        news_text = self._format_news(news_items)
        print(f"[ChurnAnalyzer] Gemini 분석 시작 (고객 {len(df)}명, 뉴스 {len(news_items)}건)")

        try:
            result = self._call_gemini(csv_summary, news_text)
        except Exception as e:
            return {'error': f'Gemini 분석 오류: {e}'}

        result['csv_summary'] = csv_summary
        result['row_count'] = len(df)
        result['columns'] = df.columns.tolist()
        print("[ChurnAnalyzer] 분석 완료")
        return result
