# 📚 문서 인덱스

> SPY ETF 변동성 예측 시스템 - 문서 안내

---

## 개요

이 문서는 프로젝트의 모든 문서를 체계적으로 정리한 인덱스입니다.

---

## 📁 문서 구조

```
docs/
│
├── 📄 INDEX.md                      # 이 파일 (문서 인덱스)
│
├── 📂 01-overview/                  # 프로젝트 개요
│   └── PROJECT_OVERVIEW.md          # 배경, 목표, 핵심 개념
│
├── 📂 02-architecture/              # 시스템 아키텍처
│   └── SYSTEM_ARCHITECTURE.md       # 구조, 파이프라인, 모듈
│
├── 📂 03-models/                    # 모델 설명
│   └── MODEL_SPECIFICATION.md       # ElasticNet 상세
│
├── 📂 04-validation/                # 검증 방법론
│   └── VALIDATION_METHODOLOGY.md    # Purged CV, 백테스트
│
├── 📂 05-results/                   # 결과 분석
│   └── RESULTS_ANALYSIS.md          # 성능, XAI, 경제적 가치
│
└── 📂 06-usage/                     # 사용 가이드
    └── USER_GUIDE.md                # 설치, 실행, API
```

---

## 📖 문서별 상세

### 1. 프로젝트 개요 (`01-overview/`)

| 문서 | 설명 | 대상 독자 |
|------|------|-----------|
| **PROJECT_OVERVIEW.md** | 프로젝트 배경, 목표, 데이터, 3대 금기사항 | 모든 이해관계자 |

**핵심 내용:**
- 왜 SPY ETF인가?
- 5일 변동성 예측 목표
- 3대 금기사항 (하드코딩, Random, Leakage)
- 데이터 개요 (2015-2024, 31개 특성)

---

### 2. 시스템 아키텍처 (`02-architecture/`)

| 문서 | 설명 | 대상 독자 |
|------|------|-----------|
| **SYSTEM_ARCHITECTURE.md** | 전체 시스템 구조, 데이터 흐름 | 개발자, 연구자 |

**핵심 내용:**
- 시스템 구조 다이어그램
- 데이터 파이프라인
- 모듈별 역할
- 학습/검증 흐름

---

### 3. 모델 설명 (`03-models/`)

| 문서 | 설명 | 대상 독자 |
|------|------|-----------|
| **MODEL_SPECIFICATION.md** | ElasticNet 모델 상세 사양 | 연구자, 데이터 사이언티스트 |

**핵심 내용:**
- ElasticNet 알고리즘
- 하이퍼파라미터 (alpha=0.0005, l1_ratio=0.3)
- 31개 특성 목록
- 계수 분석 및 해석

---

### 4. 검증 방법론 (`04-validation/`)

| 문서 | 설명 | 대상 독자 |
|------|------|-----------|
| **VALIDATION_METHODOLOGY.md** | 검증 체계 및 방법 | 연구자, 감사자 |

**핵심 내용:**
- Purged K-Fold CV
- Walk-Forward Validation
- 데이터 누출 검사
- 경제적 백테스트
- 3대 금기사항 준수 증명

---

### 5. 결과 분석 (`05-results/`)

| 문서 | 설명 | 대상 독자 |
|------|------|-----------|
| **RESULTS_ANALYSIS.md** | 성능, XAI, 경제적 가치 분석 | 모든 이해관계자 |

**핵심 내용:**
- 성능 지표 (R²=0.2218)
- 시각화 (실제 vs 예측)
- SHAP 분석 (특성 중요도)
- 백테스트 결과 (변동성 0.8% 감소)

---

### 6. 사용 가이드 (`06-usage/`)

| 문서 | 설명 | 대상 독자 |
|------|------|-----------|
| **USER_GUIDE.md** | 설치, 실행, API 사용법 | 개발자, 사용자 |

**핵심 내용:**
- 설치 방법
- 빠른 시작
- 대시보드 사용법
- 명령줄 실행
- API 예제 코드
- 문제 해결

---

## 🚀 시작하기

### 처음 방문자

1. `01-overview/PROJECT_OVERVIEW.md` → 프로젝트 이해
2. `06-usage/USER_GUIDE.md` → 설치 및 실행

### 개발자

1. `02-architecture/SYSTEM_ARCHITECTURE.md` → 시스템 구조
2. `03-models/MODEL_SPECIFICATION.md` → 모델 상세
3. `06-usage/USER_GUIDE.md` → API 사용법

### 연구자

1. `01-overview/PROJECT_OVERVIEW.md` → 연구 배경
2. `04-validation/VALIDATION_METHODOLOGY.md` → 검증 방법론
3. `05-results/RESULTS_ANALYSIS.md` → 결과 분석

---

## 📊 핵심 성과 요약

| 지표 | 값 | 설명 |
|------|-----|------|
| **Test R²** | 0.2218 | 테스트 세트 결정계수 |
| **변동성 감소** | -0.80% | Buy & Hold 대비 |
| **특성 수** | 31개 | 변동성/래그/통계 |
| **데이터 기간** | 2015-2024 | 약 10년 |

---

## 🔗 관련 링크

- **README**: [/README.md](/README.md)
- **개발 가이드**: [/CLAUDE.md](/CLAUDE.md)
- **GitHub**: [ai-stock-prediction](https://github.com/15tkdgns/ai-stock-prediction)

---

**문서 작성일**: 2025-12-04
**버전**: 1.0
