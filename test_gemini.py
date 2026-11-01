import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

api_key = os.environ.get('GEMINI_API_KEY')
print(f"1. .env 파일 확인 중...")
if not api_key:
    print("❌ 에러: .env 파일에서 GEMINI_API_KEY를 찾을 수 없습니다.")
    exit(1)
else:
    print(f"✅ API 키 로드됨 (시작: {api_key[:5]}...)")

print("\n2. Gemini API 테스트 요청 중...")
try:
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents="안녕하세요. 주식 시장에 대해 짧게 한 문장으로 말해주세요."
    )
    print("✅ 텍스트 응답 성공:", response.text.strip())

    print("\n3. JSON 구조화 응답 테스트 중...")
    prompt = """당신은 주식 애널리스트입니다. 삼성전자 주가 상승이라는 뉴스에 대한 감성 분석 결과를 아래 JSON 형식으로만 답하세요.
{ "score": 0.5, "sentiment": "긍정" }
"""
    response_json = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )
    print("✅ JSON 응답 성공:", response_json.text.strip())

    try:
        data = json.loads(response_json.text.strip())
        print("✅ JSON 파싱 성공!")
    except Exception as e:
        print("❌ JSON 파싱 에러:", e)
        
    print("\n🎉 모든 API 테스트가 정상적으로 완료되었습니다!")

except Exception as e:
    print("\n❌ API 요청 중 오류가 발생했습니다:")
    print(e)
    print("\n[해결 방법 안내]")
    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
        print("➡️ 1일 사용량(1,500회)이 아직 초기화되지 않았거나 초과되었습니다.")
        print("➡️ aistudio.google.com 에 접속해서 '다른 구글 계정'으로 로그인 후 새 API 키를 발급받아 .env 에 덮어쓰세요.")
    elif "400" in str(e) or "API_KEY_INVALID" in str(e):
        print("➡️ API 키가 유효하지 않습니다. 복사하는 과정에서 공백이 들어갔거나 오타가 있는지 확인하세요.")
    else:
        print("➡️ 서버 일시적 장애이거나 다른 문제일 수 있습니다.")
