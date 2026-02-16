# 1. 프로젝트 타이틀
**Project Name:** income-tax-data-automation (종합소득세 신고대리 데이터 입력 및 계산 자동화)
**Goal:** 로컬 Docker 환경에서 구동되며, LLM을 활용한 비정형 데이터 정규화 및 Excel 수식 엔진을 활용한 세무 계산 자동화 파이프라인 구축.

---

# 2. 용어의 정의 및 데이터 처리 전략
1) `고객 (Client)`: 종합소득세 신고대리 대상자. Unique ID는 `{No.}_{이름}_{주민등록번호}`로 관리함.
2) `부양가족정보 (Dependents Data)`: 고객이 작성한 비정형 텍스트.
   - **처리 전략:** LLM(Gemma)을 통해 JSON 포맷으로 정규화(Normalization)하여 저장함.
3) `세무 PDF 자료 (Tax PDF: Coordinate-based OCR/Parsing)`: 국세청 등에서 발급된 소득/공제 증명 자료.
   - **문제점:** 텍스트 추출 시 테이블 구조 붕괴로 데이터 오염 발생.
   - **해결책:** `pymupdf`의 `get_text("words")` 또는 `get_textbox(rect)` 기능을 활용하여 **좌표 기반(ROI: Region of Interest) 추출**.
   - **정의:** 각 데이터 항목(예: 급여총액, 결정세액 등)이 위치한 PDF 상의 페이지 번호와 XY 좌표값을 정의한 **'좌표 맵(Coordinate Map)'**을 기준으로 추출함.
4) `업무 Excel 파일 (Calculation Engine)`:
   - **구조:** 복잡한 세무 계산 수식이 미리 정의된 템플릿 파일 (`template.xlsx`).
   - **처리 전략:**
     - **Input:** `openpyxl`을 사용하여 파싱된 데이터를 특정 '입력 셀'에 기입.
     - **Process:** `formulas` 라이브러리를 로드하여 엑셀 수식을 Python 메모리 상에서 계산.
     - **Output:** 계산된 결과값이 나타나는 '결과 셀'의 값을 추출하여 시스템에 저장.
5) `Team 업무 공간`:
   - **Google Sheet:** 진행 상황(Status), 오류 메시지(Error Log), 최종 결과값 공유.
   - **Dropbox:** PDF 파일 소스 및 결과 엑셀 파일 저장소.

---

# 3. 데이터 스키마 (Input/Output 데이터 구조)

**각 필드 별 isNullable이 true에 해당하는 경우 해당 필드는 필수가 아님 (또는 없어도 무방함)**

## 3.1 Input data: 고객별 원천 데이터

### 3.1.1 Google Sheet
|분류|필드명|google sheet column 주소|타입|설명|isNullable|
|---|--------------|----|---|-------------|
|`고객 기본 정보`|`No.`|A열|`str`|고객 접수 순번|false|
|`고객 기본 정보`|`client_name`|C열|`str`|고객 이름|false|
|`고객 기본 정보`|`status`|D열|`str`|해당 고객에 대한 업무진행상황|false|
|`고객 기본 정보`|`phone_number`|E열|`str`|고객의 연락처|true|
|`고객 기본 정보`|`rrn`|F열|`str`|고객의 주민등록번호|false|
|`고객 기본 정보`|`hometax_id`|G열|`str`|고객의 홈택스 ID|true|
|`고객 기본 정보`|`hometax_pw`|H열|`str`|고객의 홈택스 pw|true|
|`고객 기본 정보`|`report_type`|I열|`str`|고객의 종합소득세 신고유형|false|
|`부양가족정보`|`dependent_text`|J열|`str`|고객이 입력한 부양가족정보 (비정형 텍스트. 정규화 대상)|true|

**[중요] `고객 기본 정보` 중 `No.`, `client_name`, `rrn`은 다음의 규칙에 따라 unique key로 설정되어, 각 스키마들을 서로 연결**
`unique key={No.}_{client_name}_{rrn}`

### 3.1.2 Dropbox


---

# 4. 시스템 아키텍처 및 개요

## 4.1 아키텍처 (Frontend - Backend 분리)
대량 데이터(일 1,000건)의 안정적 처리를 위해 UI와 연산 로직을 분리함.

- **Frontend (Streamlit):**
  - 사용자의 작업 실행 요청(Batch Trigger).
  - FastAPI로부터 실시간 진행률(Progress) 및 로그(Log) 폴링(Polling) 및 시각화.
  - 완료된 결과 엑셀 파일 다운로드 링크 제공.
- **Backend (FastAPI):**
  - `BackgroundTasks` 또는 `asyncio`를 활용한 비동기 작업 큐 관리.
  - LLM 통신, PDF 파싱, Excel 수식 계산 등 무거운 연산 수행.
  - Google Sheet 및 Dropbox API 연동 담당.

## 4.2 작업 흐름 (Workflow)
1. **Trigger:** 사용자가 Streamlit에서 '작업 시작' 버튼 클릭.
2. **Fetch:** 백엔드가 Google Sheet에서 `Status="계산요청"`인 고객 리스트 로드.
3. **Async Process (Batch Processing):** 10명 단위로 비동기 병렬 처리.
   - PDF 다운로드 -> Parsing -> LLM 정규화 -> Excel 주입 -> `formulas` 계산 -> 결과 추출.
4. **Update:** 처리 결과(성공/실패/결과값)를 Google Sheet에 업데이트하고 Dropbox에 파일 저장.
5. **Monitor:** Streamlit 화면에 실시간 처리 현황 갱신.

---

# 5. 기능적 요구사항 (Functional Requirements)

## 5.1 Backend (FastAPI) 주요 기능
1. **Batch 작업 제어:** 10명 단위 비동기 처리 및 작업 상태 관리.
2. **좌표 기반 PDF Extractor:**
   - **입력:** PDF 파일 + 좌표 정의 파일(JSON/YAML).
   - **로직:**
     - PDF 페이지별 렌더링 스케일(DPI) 확인.
     - 설정된 Rect(x0, y0, x1, y1) 범위 내의 텍스트만 병합하여 데이터화.
     - 테이블의 경우, 행(Row) 간의 Y축 간격을 계산하여 리스트 구조로 복원.
   - **예외 처리:** PDF 양식이 변경되어 좌표 이탈이 발생할 경우 `ExtractionError` 발생 및 로그 기록.
3. **LLM & Excel Integration:** 정규화된 데이터를 엑셀 템플릿에 주입하고 수식 결과값 추출.

## 5.2 Frontend (Streamlit) 주요 기능
1. **Dashboard:**
   - 전체 진행률(Progress Bar).
   - 실시간 로그 창 (Log Console).
2. **Control:**
   - 작업 시작 / 중지 버튼.
   - 설정(Config) 입력 (예: 처리할 행 범위, Dropbox 경로 등).

---

# 6. 업무 파이프라인 (Step-by-Step flow)

---

# 7. 구현 시 참조 사항 (Implementation Context)

## 7.1. **PDF 좌표 정의 예시 (Codex 참조용):**
AI는 아래와 같은 구조의 설정 파일이 존재한다고 가정하고 파싱 함수를 작성할 것.
```json
{
  "total_income": {"page": 1, "rect": [100, 200, 150, 220]},
  "tax_deduction": {"page": 2, "rect": [300, 450, 400, 470]}
}
```
## 7.3. 좌표 기반 추출 시 추가 고려사항 (Codex 참조용)**
PDF 생성 드라이버(정부24, 홈택스 등)나 폰트에 따라 **좌표가 미세하게 틀어지는 경우**가 있음.
좌표 기반 PDF Extractor 생성 시 다음의 로직을 포함할 것.
    1.  **Anchor Point 설정:** PDF 내의 특정 로고나 '사업소득' 같은 고정 단어의 위치를 먼저 찾고, 그 위치를 기준으로 상대 좌표를 계산할 것.
    2.  **Visual Debugger:** 추출한 영역을 빨간색 박스로 그려서 별도의 PDF로 저장해 주는 '디버깅용 시각화 기능'을 백엔드에 넣어줄 것. (혹시 좌표가 틀렸을 때 어디가 문제인지 확인하기 위함).

---

# 6. 기술 스펙 (Tech Stack)

- **Language:** Python 3.9+
- **Version Control:** uv (or pip/poetry)
- **Containerization:** Docker, Docker Compose
- **Web Framework:**
  - Backend: FastAPI (Async/Await 필수)
  - Frontend: Streamlit
- **Excel Processing:**
  - Read/Write: `openpyxl`
  - Calculation: `formulas` (https://pypi.org/project/formulas/)
- **Integration:**
  - Google Sheet: `gspread`
  - Dropbox: `dropbox` SDK v2
  - PDF: `pymupdf`
  - LLM: `ollama` (Local API)