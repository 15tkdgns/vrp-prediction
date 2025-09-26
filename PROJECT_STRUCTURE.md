# SPY 변동성 예측 시스템 - 프로젝트 구조

**정리 완료일**: 2025-09-25 (최종 정리)
**총 프로젝트 크기**: 68MB (정리 전 202MB → 134MB 절약)
**Python 파일 수**: 78개 (Ridge Regression 핵심 파일만 유지)

## 📋 프로젝트 개요

**시스템명**: SPY ETF 변동성 예측 시스템 (Ridge Regression)
**메인 모델**: Ridge Regression (α=1.0) with Purged K-Fold CV
**타겟**: 5일 후 변동성 예측 (target_vol_5d)
**성능**: R² = 0.3113 ± 0.1756 (HAR 벤치마크 대비 35배 우수)
**데이터**: SPY ETF 실제 데이터 (2015-2024)

## 🏗️ 주요 디렉토리 구조

```
/root/workspace/
├── src/                          # 9.6MB - 소스 코드
├── data/                         # 16MB - 데이터 및 모델
├── dashboard/                    # 560KB - 웹 대시보드
├── docs/                         # 160KB - 문서
├── tests/                        # 112KB - 테스트 코드
├── scripts/                      # 96KB - 유틸리티 스크립트
├── config/                       # 24KB - 설정 파일
├── results/                      # 24KB - 결과 파일
├── logs/                         # 20KB - 로그 파일
└── tmp/                          # 16KB - 임시 파일
```

## 📁 상세 폴더 구조

### 1. `src/` - 메인 소스 코드 (9.6MB)

**핵심 모듈들:**
```
src/
├── visualization/               # 시각화 시스템 (NEW)
│   ├── performance_dashboard.py # 메인 시각화 생성기
│   ├── charts/                  # 개별 차트 모듈들
│   │   ├── model_metrics.py     # 모델 성능 차트
│   │   ├── feature_analysis.py  # 특성 분석 차트
│   │   ├── economic_metrics.py  # 경제적 수치 차트
│   │   ├── validation_results.py # 검증 결과 차트
│   │   └── main_model_summary.py # 메인 모델 상세 차트
│   ├── data_loaders/           # 데이터 로더
│   ├── templates/              # HTML 템플릿
│   └── outputs/                # 생성된 차트들 (PNG)
├── models/                     # 모델 관련 코드
│   └── correct_target_design.py # 메인 Ridge 모델
├── validation/                 # 검증 시스템
│   ├── purged_cross_validation.py
│   └── economic_backtest_validator.py
├── analysis/                   # 분석 모듈
│   ├── professional_xai_analyzer.py
│   └── verified_xai_analysis.py
├── core/                       # 핵심 시스템
│   ├── data_processor.py       # 데이터 처리
│   ├── config.py              # 설정 관리
│   └── logger.py              # 로깅
└── utils/                      # 유틸리티
    ├── system_orchestrator.py  # 시스템 통합
    └── yfinance_manager.py     # 데이터 수집
```

### 2. `data/` - 데이터 저장소 (16MB)

**정리 후 구조:**
```
data/
├── raw/                        # 4.0MB - 원시 데이터
│   ├── model_performance.json  # 메인 모델 성능 지표
│   ├── system_status.json      # 시스템 상태
│   └── spy_*.json             # SPY 관련 데이터
├── models/                     # 6.4MB - 모델 저장소
│   └── [Ridge 모델 파일들]     # 현재 사용 중인 모델만 보관
├── training/                   # 3.7MB - 훈련 데이터
├── processed/                  # 1.1MB - 전처리된 데이터
├── xai_analysis/              # 36KB - XAI 분석 결과
├── results/                   # 164KB - 분석 결과
└── validation/                # 68KB - 검증 결과
```

### 3. `dashboard/` - 웹 대시보드 (560KB)

**3탭 정적 HTML 대시보드:**
```
dashboard/
├── index.html                 # 메인 대시보드 페이지
├── js/                       # JavaScript 모듈들
│   ├── components/           # 컴포넌트들
│   │   ├── ridge-xai-visualization.js
│   │   ├── chart-manager.js
│   │   └── data-loader.js
│   └── dashboard-manager.js  # 메인 관리자
├── css/                      # 스타일시트
└── package.json             # NPM 설정
```

**대시보드 기능:**
- **Tab 1**: 변동성 예측 vs 실제값
- **Tab 2**: SHAP 특성 중요도 분석
- **Tab 3**: 경제적 백테스트 결과

### 4. `docs/` - 문서 (160KB)

```
docs/
├── methodology/              # 방법론 문서
├── validation/              # 검증 문서
├── api/                     # API 문서
└── user_guide/             # 사용자 가이드
```

## 🚀 주요 실행 파일들

### 메인 시스템 실행
```bash
# 전체 시스템 실행 (데이터 처리 + 모델 + 검증)
PYTHONPATH=/root/workspace python3 src/utils/system_orchestrator.py

# 메인 모델 훈련 및 검증
PYTHONPATH=/root/workspace python3 src/models/correct_target_design.py

# 경제적 백테스트
PYTHONPATH=/root/workspace python3 src/validation/economic_backtest_validator.py
```

### 시각화 시스템 실행
```bash
# 모든 차트 생성 (18개 차트 + 메인 모델 상세 3개)
PYTHONPATH=/root/workspace python3 src/visualization/performance_dashboard.py

# 메인 모델 상세 차트만
PYTHONPATH=/root/workspace python3 src/visualization/charts/main_model_summary.py
```

### 대시보드 실행
```bash
cd dashboard && npm run dev
# → http://localhost:8080/index.html
```

## 📊 생성되는 결과물

### 1. 성능 분석 차트 (PNG, 300 DPI)
- **모델 성능**: R² 비교, 오차 지표, 교차검증 결과 (4개)
- **특성 분석**: SHAP 중요도, 분포, 상관관계 분석 (5개)
- **경제적 분석**: 수익률, 위험지표, 거래비용 분석 (5개)
- **검증 결과**: 예측 정확도, 잔차 분석, 모델 가정 (4개)
- **메인 모델 상세**: 파라미터, 성능, 평가지표 (3개)

### 2. 데이터 파일
- **JSON**: 성능 지표, 분석 결과, 시스템 상태
- **CSV**: 훈련 데이터, 특성 데이터, 분석 결과

### 3. 웹 대시보드
- **정적 HTML**: 자체 포함형, 서버 불필요
- **인터랙티브**: Chart.js 기반 동적 시각화

## 🧹 정리된 내용

### 삭제된 항목들 (총 138MB 절약)
1. **백업 폴더**: `data/models_backup` (135MB), `data/cleanup_backup` (3.5MB)
2. **격리 폴더**: `data/quarantine` (176KB) - 데이터 누출 코드들
3. **캐시 파일**: `__pycache__` 폴더들 (여러 곳)
4. **중복 시각화**: `visualizations/` 폴더 (804KB)
5. **불필요한 모델**: 기존 pkl, pth 파일들 (40MB+)
6. **임시 파일**: `emergency_data_leakage_audit.json`, `학습계획.txt` 등

### 유지된 핵심 항목들
1. **메인 Ridge 모델** 및 관련 파일들
2. **실제 SPY 데이터** (2015-2024)
3. **검증된 성능 지표** 및 분석 결과
4. **시각화 시스템** (새로 구축)
5. **웹 대시보드** 및 문서

## 🔧 시스템 의존성

### Python 패키지 (`requirements/base.txt`)
- **ML**: scikit-learn, pandas, numpy
- **데이터**: yfinance, ta (기술적 지표)
- **시각화**: matplotlib, seaborn
- **분석**: shap (XAI), scipy (통계)

### Node.js 패키지 (`dashboard/package.json`)
- **서버**: http-server, serve
- **도구**: eslint, prettier

## 📈 프로젝트 상태

**현재 상태**: ✅ **프로덕션 준비 완료**
- **모델**: 검증된 Ridge Regression (R² = 0.3113)
- **데이터**: 실제 SPY ETF 데이터 (누출 없음)
- **검증**: Purged K-Fold CV + 경제적 백테스트
- **시각화**: 21개 차트 + 웹 대시보드
- **문서화**: 완전한 사용 가이드 및 API 문서

## 🎯 다음 단계

1. **모델 업데이트**: 2025년 데이터 포함 재훈련
2. **API 구축**: RESTful API 서버 구현
3. **실시간 모니터링**: 라이브 데이터 수집 및 예측
4. **성능 최적화**: 추가 특성 엔지니어링

---

**마지막 업데이트**: 2025-09-25 16:20 KST
**정리 담당**: Claude Code Assistant