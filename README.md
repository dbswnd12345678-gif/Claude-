# AI Agent — Claude Opus 4.6

Claude Opus 4.6 기반의 커스텀 AI 에이전트 웹 애플리케이션입니다.

## 기능

| 기능 | 설명 |
|------|------|
| 🔍 **웹 검색** | Claude 내장 `web_search` 도구로 실시간 검색 |
| 📄 **파일 분석 (RAG)** | PDF · CSV · DOCX · 이미지 업로드 후 질문 |
| 📝 **Google Docs** | 문서 읽기 / 내용 추가 |
| 📊 **Google Sheets** | 셀 범위 읽기 / 쓰기 |
| 💬 **스트리밍 채팅** | 토큰 단위 실시간 응답 + 도구 호출 시각화 |

## 빠른 시작

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 환경 변수 설정
cp .env.example .env
# .env 파일을 열어 ANTHROPIC_API_KEY 입력

# 3. 서버 실행
uvicorn main:app --reload --port 8000
```

브라우저에서 `http://localhost:8000` 접속

## Google Workspace 연동 (선택)

1. [Google Cloud Console](https://console.cloud.google.com/)에서 프로젝트 생성
2. **Google Docs API** 및 **Google Sheets API** 활성화
3. 서비스 계정 생성 → JSON 키 파일 다운로드
4. `.env`에 `GOOGLE_SERVICE_ACCOUNT_FILE=/path/to/key.json` 설정
5. 서비스 계정 이메일을 편집할 Google Docs/Sheets 문서에 **공유**

## 프로젝트 구조

```
├── main.py          # FastAPI 애플리케이션 (라우터, 스트리밍 에이전트 루프)
├── tools.py         # 커스텀 도구 정의 및 구현
├── static/
│   └── index.html   # 단일 파일 프론트엔드
├── requirements.txt
└── .env.example
```

## 환경 변수

| 변수 | 필수 | 설명 |
|------|------|------|
| `ANTHROPIC_API_KEY` | ✅ | Anthropic API 키 |
| `GOOGLE_SERVICE_ACCOUNT_FILE` | ❌ | Google 서비스 계정 JSON 파일 경로 |

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `POST` | `/api/session` | 새 세션 생성 |
| `GET` | `/api/session/{id}` | 세션 정보 조회 |
| `DELETE` | `/api/session/{id}` | 세션 초기화 |
| `POST` | `/api/upload/{id}` | 파일 업로드 (multipart) |
| `POST` | `/api/chat/{id}` | 채팅 (SSE 스트리밍) |
| `GET` | `/api/tools` | 사용 가능한 도구 목록 |
