# 🗄️ Archive (보관)

> 구버전, 미사용, 또는 실험 파일 보관소

---

## 개요

이 디렉토리는 현재 사용하지 않지만 참조 목적으로 보관하는 파일들을 포함합니다.

**삭제하지 마세요** - 필요 시 복구하거나 참조할 수 있습니다.

---

## 📁 구조

```
archive/
│
├── deprecated_models/         # 구버전 모델 파일
│   ├── volatility_predictor_v2.py
│   ├── volatility_predictor_v3.py
│   ├── volatility_predictor_v3_fixed.py
│   ├── volatility_predictor_v4_final.py
│   ├── volatility_predictor_v5_correct.py
│   ├── enhanced_volatility_model_v2.py
│   ├── enhanced_volatility_model_v2_lite.py
│   ├── enhanced_volatility_predictor.py
│   ├── debug_sota_model.py
│   ├── sota_volatility_model.py
│   ├── sota_volatility_model_fixed.py
│   ├── enhanced_performance_model.py
│   ├── fine_tuned_performance_model.py
│   ├── final_ensemble_model.py
│   ├── final_volatility_model.py
│   ├── robust_volatility_model.py
│   └── time_window_optimization.py
│
├── deprecated_validation/     # 구버전 검증 스크립트
│   ├── check_v3_leakage.py
│   ├── deep_leakage_check.py
│   ├── verify_chart_leakage.py
│   ├── verify_v2_proper.py
│   └── simple_economic_backtest.py
│
├── old_logs/                  # 오래된 로그 파일
│   ├── model_training.log
│   ├── model_training_final.log
│   └── model_training_v2.log
│
├── old_results/               # 구버전 결과 파일
│
├── failed_experiments/        # 실패한 실험
│   └── (Random 데이터 사용 실험 등)
│
├── old_folders/               # 구버전 폴더 백업
│
├── old_data/                  # 구버전 데이터
│
├── old_models/                # 구버전 모델 체크포인트
│
├── old_reports/               # 구버전 보고서
│
├── old_figures/               # 구버전 피규어
│
├── exploratory_scripts/       # 탐색적 스크립트
│
└── PROJECT_STRUCTURE.md       # 이전 프로젝트 구조 문서
```

---

## 📋 정리 이력

### 2025-12-04

| 카테고리 | 이동된 파일 수 | 이유 |
|----------|----------------|------|
| deprecated_models | 17개 | 구버전 모델, 현재 사용하지 않음 |
| deprecated_validation | 5개 | 구버전 검증 스크립트 |
| old_logs | 3개 | 오래된 로그 파일 |

**현재 활성 모델:**
- `src/models/correct_target_design.py` ⭐
- `src/models/train_final_reproducible_model.py` ⭐
- `src/models/garch_enhanced_model.py`

**현재 활성 검증:**
- `src/validation/purged_cross_validation.py` ⭐
- `src/validation/economic_backtest_validator.py` ⭐
- `src/validation/walk_forward_validation.py`
- `src/validation/advanced_leakage_analysis.py`

---

## ⚠️ 주의사항

1. **삭제 금지**: 이 파일들은 필요 시 참조용으로 보관
2. **Git 추적**: archive 폴더도 Git에 포함됨
3. **복구**: 필요한 파일은 원래 위치로 이동 가능

---

## 🔄 복구 방법

```bash
# 예시: 특정 모델 복구
Move-Item -Path "archive/deprecated_models/some_model.py" -Destination "src/models/"

# 또는 Git으로 복구
git checkout HEAD~10 -- src/models/some_model.py
```

---

**최종 정리일**: 2025-12-04
