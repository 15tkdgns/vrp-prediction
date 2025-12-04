# 문서 인덱스

**최종 업데이트**: 2025-11-04

이 디렉토리는 SPY 변동성 예측 시스템의 모든 문서를 포함합니다.

---

## 📁 문서 구조

### 1. 발표 자료 (`presentation/`)
논문 발표 및 프레젠테이션용 최종 문서

- **FINAL_PRESENTATION.md** (12KB) - 완전한 발표 자료 (12개 섹션, 10개 테이블)
- **FINAL_CHECKLIST.md** (11KB) - 논문 제출 전 최종 검증 체크리스트
- **QUICK_REFERENCE.md** (8KB) - 발표 중 빠른 참조용 (암기 수치, FAQ)
- **PRESENTATION_SUMMARY.md** (7KB) - 발표 요약 (이전 버전)

**용도**: 논문 발표, 학회 프레젠테이션, 최종 검증

---

### 2. 기술 문서 (`technical/`)
시스템 아키텍처 및 기술 상세 문서

- **ARCHITECTURE.md** (20KB) - 시스템 아키텍처 및 설계
- **VALIDATION_METHODOLOGY.md** (14KB) - 검증 방법론 상세 (Purged K-Fold CV)
- **VARIABLES_DOCUMENTATION.md** (16KB) - 변수 정의 및 특성 설명
- **MODEL_PERFORMANCE_REPORT.md** (12KB) - 모델 성능 분석 리포트

**용도**: 개발자 참조, 코드 리뷰, 기술 검증

---

### 3. 프로세스 문서 (`process/`)
프로젝트 진행 과정 및 워크플로우

- **PROJECT_PROCESS_FLOWCHART.md** (10KB) - 프로젝트 플로우차트
- **PROJECT_PROCESS_TREE.md** (6KB) - 프로젝트 프로세스 트리

**용도**: 프로젝트 관리, 진행 상황 추적

---

### 4. 기존 문서 (legacy)

#### Architecture (`architecture/`)
- 시스템 아키텍처 관련 문서

#### Development (`development/`)
- 개발 가이드 및 설정

#### Reports (`reports/`)
- 실험 리포트 및 분석

#### User Guide (`user-guide/`)
- 사용자 가이드

---

## 🎯 목적별 문서 찾기

### 발표 준비 시
1. **FINAL_PRESENTATION.md** - 완전한 발표 자료
2. **QUICK_REFERENCE.md** - 암기용 핵심 수치
3. **FINAL_CHECKLIST.md** - 최종 검증

### 논문 작성 시
1. **VALIDATION_METHODOLOGY.md** - 검증 방법론
2. **MODEL_PERFORMANCE_REPORT.md** - 성능 분석
3. **VARIABLES_DOCUMENTATION.md** - 변수 설명
4. **paper/** 디렉토리 참조

### 코드 이해 시
1. **ARCHITECTURE.md** - 시스템 구조
2. **VARIABLES_DOCUMENTATION.md** - 변수 정의
3. **CLAUDE.md** (루트) - 개발 가이드

### 프로젝트 관리 시
1. **PROJECT_PROCESS_FLOWCHART.md** - 워크플로우
2. **PROJECT_PROCESS_TREE.md** - 프로세스 트리

---

## 📊 핵심 수치 빠른 참조

### 모델 성능
- **R²**: 0.303 (30.3%)
- **HAR 대비**: 1.41배 개선
- **샘플**: 2,460개 (2015-2024 SPY)
- **특성**: 31개 (변동성 중심)

### 데이터 무결성
- ✅ 하드코딩: 없음
- ✅ Random 데이터: 없음
- ✅ 데이터 누출: 없음 (30.3% < 95%)

---

## 🔗 관련 디렉토리

- **paper/** - 논문 관련 파일 (참고문헌, 초록, 서론, 피규어)
- **src/** - 소스 코드
- **data/** - 데이터셋 및 결과
- **archive/** - 격리된 실험
- **dashboard/** - 시각화 대시보드

---

**문의**: `/root/workspace` 프로젝트 리포지토리 참조
