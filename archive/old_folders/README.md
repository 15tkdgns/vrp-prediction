# Old Folders Archive

**이동 날짜**: 2025-11-04
**이유**: 프로젝트 정리 - 중복 또는 불필요한 폴더 격리

---

## 이동된 폴더 목록

### 1. **analysis/** (7개 파일)
**이유**: 분석 스크립트 중복 (src/analysis/와 중복)
**내용**:
- comprehensive_performance_analysis.py
- comprehensive_performance_report.json
- final_comprehensive_validation.py
- 기타 분석 스크립트들

### 2. **experiments/** (11개 파일)
**이유**: 실험 문서 중복 (docs/와 중복)
**내용**:
- EXPERIMENTAL_SETUP.md
- EXPERIMENT_SUMMARY.md
- FINAL_PERFORMANCE_ASSESSMENT.md
- 기타 실험 문서들

### 3. **reports/** (9개 파일)
**이유**: 리포트 중복 (docs/와 중복)
**내용**:
- ADVANCED_LLM_PATTERN_DETECTION_FINAL.md
- FINAL_DATA_LEAKAGE_REPORT.md
- LEAK_FREE_MODEL_FINAL_REPORT.md
- 기타 리포트들

### 4. **results/** (15개 파일)
**이유**: 결과 파일 중복 (data/raw/와 중복)
**내용**:
- enhanced_model_v2_lite.json
- final_ensemble_model.json
- garch_enhanced_model.json
- 기타 결과 JSON 파일들

### 5. **paper_figures/** (12개 파일)
**이유**: 논문 피규어 중복 (paper/figures/로 병합됨)
**내용**:
- figure1_model_comparison.pdf/png
- figure2_return_prediction_failure.pdf/png
- figure3~6 피규어들
**조치**: paper/figures/로 복사 후 이동

### 6. **scripts/** (24개 파일)
**이유**: 대부분 중복 스크립트 (src/에 정리된 코드 있음)
**내용**:
- create_paper_figures.py
- run_har_benchmark.py
- rv_economic_backtest.py
- 기타 실험 스크립트들
**참고**: 필요시 복구 가능

### 7. **models/** (12개 파일)
**이유**: 학습된 모델 백업 (src/models/에서 재학습 가능)
**내용**:
- lstm_model.keras
- ridge_volatility_model.pkl
- tft_model 파일들
**참고**: 필요시 복구 가능

---

## 삭제된 폴더

### 1. **logs/**
**이유**: 시스템 로그 불필요
**내용**: system/ 폴더만 존재

### 2. **tests/**
**이유**: 테스트 코드 미사용
**내용**: 테스트 파일들

---

## 복구 방법

필요한 파일이 있다면 이 폴더에서 복구할 수 있습니다:

```bash
# 예: scripts 폴더 복구
cp -r archive/old_folders/scripts /root/workspace/

# 예: 특정 스크립트만 복구
cp archive/old_folders/scripts/run_har_benchmark.py scripts/
```

---

## 현재 활성 폴더 구조

```
workspace/
├── README.md
├── CLAUDE.md
├── docs/               # 문서 (정리됨)
├── paper/              # 논문
├── src/                # 소스 코드
├── data/               # 데이터
├── dashboard/          # 대시보드
├── archive/            # 격리된 실험/폴더
│   ├── failed_experiments/   # Random 데이터 실험
│   └── old_folders/          # 중복/불필요 폴더 (이 폴더)
├── config/             # 설정
└── requirements/       # 의존성
```

---

**정리 날짜**: 2025-11-04
**상태**: 백업 완료 - 필요시 복구 가능
