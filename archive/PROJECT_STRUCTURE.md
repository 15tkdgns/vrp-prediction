# SPY 변동성 예측 시스템 - 프로젝트 구조

**최종 정리 날짜**: 2025-11-04
**상태**: ✅ 완전히 정리 완료

---

## 📁 최종 폴더 구조

```
workspace/
│
├── README.md (11KB)              ⭐ 프로젝트 메인 문서
├── CLAUDE.md (8.7KB)             ⭐ 개발 가이드
├── PROJECT_STRUCTURE.md          ⭐ 이 파일 (프로젝트 구조)
│
├── docs/                         📚 문서 (체계적으로 정리됨)
│   ├── INDEX.md                  # 문서 인덱스
│   ├── presentation/             # 발표 자료
│   │   ├── FINAL_PRESENTATION.md       (12KB) ⭐
│   │   ├── FINAL_CHECKLIST.md          (11KB)
│   │   ├── QUICK_REFERENCE.md          (8KB)
│   │   ├── PRESENTATION_SUMMARY.md     (7KB)
│   │   └── README.md
│   ├── technical/                # 기술 문서
│   │   ├── ARCHITECTURE.md             (20KB)
│   │   ├── VALIDATION_METHODOLOGY.md   (14KB)
│   │   ├── VARIABLES_DOCUMENTATION.md  (16KB)
│   │   ├── MODEL_PERFORMANCE_REPORT.md (12KB)
│   │   └── README.md
│   └── process/                  # 프로세스 문서
│       ├── PROJECT_PROCESS_FLOWCHART.md
│       ├── PROJECT_PROCESS_TREE.md
│       └── README.md
│
├── paper/                        📄 논문 관련
│   ├── PAPER_REFERENCES.bib      # 참고문헌 (30+ 개)
│   ├── PAPER_STRUCTURE.md        # 논문 구조
│   ├── PAPER_ABSTRACT.md         # 초록
│   ├── PAPER_INTRODUCTION.md     # 서론
│   ├── PAPER_SUBMISSION_STATUS.md
│   ├── README.md
│   ├── figures/                  # 논문 피규어 (PDF/PNG)
│   │   ├── main_results/
│   │   ├── methodology/
│   │   ├── analysis/
│   │   ├── best_visibility/
│   │   └── correlation/
│   ├── data/                     # 논문 데이터
│   │   ├── model_performance_comparison.csv
│   │   ├── key_findings_summary.csv
│   │   └── economic_backtest_results.csv
│   └── scripts/
│
├── src/                          💻 소스 코드
│   ├── core/                     # 핵심 모듈
│   │   ├── config.py
│   │   ├── logger.py
│   │   └── data_processor.py
│   ├── models/                   # 모델
│   │   ├── correct_target_design.py     ⭐ 메인 모델
│   │   ├── enhanced_volatility_model_v2.py
│   │   └── 기타 모델들...
│   ├── validation/               # 검증
│   │   ├── purged_cross_validation.py
│   │   ├── economic_backtest_validator.py
│   │   └── advanced_leakage_detection.py
│   ├── features/                 # 특성 공학
│   │   └── advanced_feature_engineering.py
│   ├── data/                     # 데이터 처리
│   │   ├── loader.py
│   │   └── leak_free_data_processor.py
│   ├── analysis/                 # 분석
│   │   ├── xai_dashboard_connector.py
│   │   └── advanced_model_metrics_calculator.py
│   ├── utils/                    # 유틸리티
│   │   └── system_orchestrator.py       ⭐ 시스템 통합
│   └── visualization/            # 시각화
│       └── performance_dashboard.py
│
├── data/                         💾 데이터
│   ├── raw/                      # 원시 데이터 및 결과 (37개 JSON)
│   │   ├── model_performance.json            ⭐
│   │   ├── har_vs_ridge_comparison.json
│   │   ├── rv_economic_backtest_results.json
│   │   ├── integrity_validation_report.json
│   │   └── 기타 결과 파일들...
│   ├── training/                 # 학습 데이터 (4개 CSV)
│   │   ├── sp500_2020_2024.csv
│   │   ├── sp500_2020_2024_enhanced.csv
│   │   ├── sp500_leak_free_dataset.csv
│   │   └── sp500_ultra_leak_free.csv
│   ├── validation/               # 검증 데이터
│   │   └── comprehensive_model_validation.json
│   └── xai_analysis/             # XAI 분석 결과
│       └── verified_xai_analysis_*.json
│
├── dashboard/                    📊 대시보드
│   ├── index.html                # 메인 페이지
│   ├── package.json
│   ├── modules/                  # JavaScript 모듈
│   │   ├── DataLoader.js
│   │   ├── VolatilityChart.js
│   │   ├── FeatureImpactChart.js
│   │   └── EconomicValueChart.js
│   └── styles/
│
├── archive/                      🗄️ 격리된 실험 및 오래된 폴더
│   ├── failed_experiments/       # Random 데이터 실험
│   │   ├── README.md
│   │   ├── data_pipelines/       # 3개 파이프라인
│   │   ├── models/               # 6개 모델
│   │   └── 메타데이터...
│   ├── old_folders/              # 중복/불필요 폴더 백업 ✨ NEW
│   │   ├── README.md
│   │   ├── analysis/             # (7개 파일)
│   │   ├── experiments/          # (11개 파일)
│   │   ├── reports/              # (9개 파일)
│   │   ├── results/              # (15개 파일)
│   │   ├── paper_figures/        # (12개 파일, paper/로 병합됨)
│   │   ├── scripts/              # (24개 파일)
│   │   └── models/               # (12개 파일)
│   └── 기타 오래된 데이터들...
│
├── config/                       ⚙️ 설정
│   ├── Dockerfile
│   └── docker-compose.yml
│
└── requirements/                 📦 의존성
    └── base.txt
```

---

## 🎯 핵심 디렉토리 설명

### 1. **docs/** - 문서 (완전히 재구성됨)
**용도**: 발표, 기술 문서, 프로세스 관리

#### presentation/ (발표 자료)
- 논문 발표 및 프레젠테이션용
- 4개 파일 (FINAL_PRESENTATION.md 핵심)

#### technical/ (기술 문서)
- 개발자 참조 및 코드 리뷰용
- 4개 파일 (ARCHITECTURE.md 핵심)

#### process/ (프로세스 문서)
- 프로젝트 관리 및 워크플로우
- 2개 파일

### 2. **paper/** - 논문
**용도**: 논문 작성 및 제출

- 참고문헌 30+ 개 (BibTeX)
- 논문 피규어 (PDF/PNG)
- 논문 데이터 (CSV)

### 3. **src/** - 소스 코드
**용도**: 시스템 구현

#### 핵심 파일
- `models/correct_target_design.py` - 메인 모델 ⭐
- `validation/purged_cross_validation.py` - 검증
- `utils/system_orchestrator.py` - 시스템 통합 ⭐

### 4. **data/** - 데이터
**용도**: 원시 데이터, 결과, 학습 데이터

- `raw/` - 37개 JSON 결과 파일
- `training/` - 4개 CSV 학습 데이터
- `validation/` - 검증 데이터
- `xai_analysis/` - XAI 분석 결과

### 5. **dashboard/** - 대시보드
**용도**: 시각화 및 결과 표시

- 정적 HTML 대시보드
- 3-Tab 인터페이스
- Chart.js 시각화

### 6. **archive/** - 격리
**용도**: 실패한 실험 및 오래된 폴더 보관

#### failed_experiments/ (Random 데이터)
- Random 데이터 생성 파이프라인 3개
- 의존 모델 6개
- 메타데이터

#### old_folders/ (중복 폴더) ✨ 신규
- 7개 폴더 이동 (100+ 파일)
- 필요시 복구 가능

---

## 📊 파일 통계

### 루트 레벨
- **마크다운**: 2개 (README, CLAUDE)
- **정리 완료**: 12개 문서 → docs/ 이동

### 문서 (docs/)
- **발표 자료**: 4개 (32KB)
- **기술 문서**: 4개 (62KB)
- **프로세스**: 2개 (16KB)
- **총**: 10개 파일 + 4개 README

### 코드 (src/)
- **Python 파일**: 20개 모듈
- **핵심 모델**: 1개 (correct_target_design.py)

### 데이터 (data/)
- **JSON 결과**: 37개
- **CSV 학습**: 4개
- **XAI 분석**: 1개

### 격리 (archive/)
- **실패 실험**: 9개 파일
- **오래된 폴더**: 7개 폴더 (100+ 파일)

---

## 🚀 빠른 접근

### 발표 준비
```bash
cat docs/presentation/FINAL_PRESENTATION.md
cat docs/presentation/QUICK_REFERENCE.md
```

### 코드 실행
```bash
PYTHONPATH=/root/workspace python3 src/models/correct_target_design.py
PYTHONPATH=/root/workspace python3 src/utils/system_orchestrator.py
```

### 대시보드
```bash
cd dashboard && npm run dev
open http://localhost:8080/index.html
```

### 문서 탐색
```bash
cat docs/INDEX.md                # 문서 인덱스
cat PROJECT_STRUCTURE.md         # 프로젝트 구조 (이 파일)
```

---

## 🔄 정리 히스토리

### 2025-11-04: 대규모 정리
1. **문서 재구성**: 12개 파일 → docs/ 3개 카테고리
2. **폴더 정리**: 7개 중복 폴더 → archive/old_folders/
3. **삭제**: logs/, tests/ 폴더
4. **병합**: paper_figures/ → paper/figures/
5. **README 생성**: 4개 신규 가이드

### 이전 정리
- Random 데이터 실험 격리 (failed_experiments/)
- 데이터 무결성 검증 완료

---

## ✅ 정리 완료 체크리스트

- [x] 루트 파일 정리 (12개 → 2개)
- [x] 문서 재구성 (docs/ 3개 카테고리)
- [x] 중복 폴더 제거 (7개 → archive)
- [x] 불필요 폴더 삭제 (2개)
- [x] README 작성 (5개 신규)
- [x] 프로젝트 구조 문서화 (이 파일)

---

## 📞 지원

**문서 인덱스**: `docs/INDEX.md`
**프로젝트 구조**: `PROJECT_STRUCTURE.md` (이 파일)
**개발 가이드**: `CLAUDE.md`
**메인 README**: `README.md`

---

**최종 정리 날짜**: 2025-11-04
**상태**: ✅ 완전히 정리 완료
**폴더 수**: 8개 핵심 디렉토리
**문서**: 체계적으로 분류 완료
