# 프로젝트 정리 실행 계획

## 현재 상황 분석
- 전체 프로젝트 크기: 매우 큰 상태 (100+ 파일)
- PAPER_SUBMISSION 패키지: 완성된 상태 ✅
- 중복 파일: 다수 존재 (이미지, JSON, 스크립트)
- 임시 파일: logs, cache, tmp 디렉터리 존재

## 정리 대상 파일 목록

### 1단계: 임시/캐시 파일 정리 (즉시 삭제 가능)
```bash
# 디렉터리 삭제
rm -rf logs/
rm -rf cache/
rm -rf tmp/
rm -rf config/  # src/core/에 통합됨

# Python 캐시 정리
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete
```

### 2단계: 중복 이미지 파일 정리
**보존할 핵심 이미지 (5개)**:
- `leak_free_model_performance.png` - 메인 모델 성능
- `data_integrity_summary.png` - 데이터 무결성 요약
- `leak_free_price_prediction_comparison.png` - 가격 예측 비교
- `actual_vs_predicted_sp500_prices.png` - 실제 vs 예측
- `model_performance_dashboard.png` - 대시보드 스크린샷

**정리할 중복 이미지 (15+ 개)**:
- `aggressive_mae_optimization_results.png`
- `baseline_models_comparison.png`
- `corrected_models_comparison.png`
- `historical_actual_vs_predicted_prices.png`
- `local_price_comparison_timeseries.png`
- 기타 중복 PNG 파일들

### 3단계: 중복 HTML 파일 정리
**보존할 핵심 HTML (1개)**:
- PAPER_SUBMISSION에 있는 인터랙티브 차트만 보존

**정리할 HTML 파일**:
- `historical_actual_vs_predicted_interactive.html`
- `local_price_comparison_interactive.html`

### 4단계: 중복 Python 스크립트 정리
**보존할 핵심 스크립트**:
- `model_performance_summary_table.py`
- `create_paper_structure.py`

**정리할 중복 스크립트**:
- `price_comparison_chart_local.py` (중복)
- `historical_price_prediction_model.py` (이전 버전)
- `volatility_prediction_comparison.py` (중복)

### 5단계: 소스코드 디렉터리 정리
**보존할 핵심 src/ 모듈**:
- `src/models/correct_target_design.py` - 메인 모델
- `src/validation/purged_cross_validation.py` - 핵심 검증
- `src/validation/economic_backtest_validator.py` - 경제적 검증
- `src/core/` - 핵심 유틸리티
- `src/analysis/` - 성능 분석

**정리할 src/ 모듈**:
- `src/backtesting/` - 중복 백테스팅 (validation에 포함)
- `src/evaluation/` - 중복 평가 (analysis에 포함)
- `src/features/` 중 사용하지 않는 모듈들
- `src/models/` 중 이전 버전 모델들
- `src/validation/` 중 중복 검증 스크립트들

### 6단계: JSON 결과 파일 통합
**보존할 핵심 JSON**:
- `leak_free_model_results.json` - 최종 모델 결과
- `comprehensive_leakage_analysis_report.json` - 종합 누출 분석

**정리할 JSON 파일**:
- `data_leakage_report.json` (중복)
- `perfect_correlation_analysis.json` (중복)

## 예상 정리 결과

### 정리 전
```
workspace/
├── 50+ 디렉터리
├── 80+ Python 파일
├── 29+ 기타 파일 (루트)
├── 100+ 데이터/결과 파일
└── 크기: ~500MB+
```

### 정리 후
```
workspace/
├── PAPER_SUBMISSION/              # 완전한 논문 패키지
├── src/                          # 핵심 모듈만 (20개 파일)
│   ├── models/correct_target_design.py
│   ├── validation/purged_cross_validation.py
│   ├── validation/economic_backtest_validator.py
│   ├── core/
│   └── analysis/
├── data/                         # 필수 데이터만
│   ├── raw/model_performance.json
│   ├── training/
│   └── models/
├── dashboard/                    # 시각화 대시보드
├── docs/                        # 핵심 문서만
├── requirements/                # 의존성
├── tests/                       # 핵심 테스트만
├── LEAK_FREE_MODEL_FINAL_REPORT.md
├── FINAL_DATA_LEAKAGE_REPORT.md
├── CLAUDE.md
├── README.md
├── PROJECT_FILE_INVENTORY.md    # 이 문서
├── model_performance_summary_table.py
├── leak_free_model_performance.png
├── data_integrity_summary.png
├── leak_free_price_prediction_comparison.png
├── actual_vs_predicted_sp500_prices.png
└── 크기: ~100MB
```

## 안전성 검증
각 정리 단계 후 다음을 확인:
1. PAPER_SUBMISSION 패키지 완성도
2. 메인 모델 실행 가능성
3. 대시보드 동작 확인
4. 핵심 문서 보존 상태

## 복구 방안
- Git 히스토리를 통한 복구 가능
- 중요 파일은 PAPER_SUBMISSION에 백업됨
- 즉시 복원이 필요한 경우를 위한 단계별 진행