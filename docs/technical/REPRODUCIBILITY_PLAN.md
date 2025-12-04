# 재현성 확보 & 논문 데이터 고정 계획

**목적**: 이미 존재하는 산출물을 최대한 활용하면서, 논문 제출 시 필요한 결과를 단계적으로 검증하고 고정합니다. 새 실험은 실제 데이터가 아닌 경우에만 수행합니다.

---

## 1. 환경 및 데이터 소스 확인

1. `requirements/base.txt` 기반 가상환경을 사용합니다.
2. 네트워크가 허용된 환경에서 다음 명령을 실행하여 실제 SPY 데이터를 불러옵니다.
   ```bash
   PYTHONPATH=/root/workspace python3 src/models/correct_target_design.py
   ```
3. 콘솔에 `✅ 실제 SPY 데이터: XXXX개 관측치` 로그가 표시되는지 확인합니다. 실패 시 로그만 남기고 기존 산출물을 유지합니다.

**산출물 (자동 생성/갱신)**  
- `data/validation/comprehensive_model_validation.json`  
- `data/raw/model_performance.json`

이 두 파일이 논문/README/Streamlit에서 참조하는 유일한 소스가 되도록 합니다.

---

## 2. ElasticNet Grid 결과 관리

1. 이미 존재하는 `experiments/elasticnet_grid/run_elasticnet_grid.py`를 다음과 같이 실행합니다.
   ```bash
    PYTHONPATH=/root/workspace python3 experiments/elasticnet_grid/run_elasticnet_grid.py \
      --feature-variants base extended \
      --alphas 0.05 0.08 0.10 0.15 \
      --l1-ratios 0.5 0.6 0.7 0.8
   ```
2. 결과 JSON/CSV는 `experiments/elasticnet_grid/results/elasticnet_grid_<timestamp>.{json,csv}`에 저장됩니다.
3. 파일 내 `data_source` 필드가 `real_spy`인 경우만 논문 근거로 사용합니다. 시뮬레이션(`simulated`) 결과는 `results/archive/`로 이동해 혼동을 방지합니다.

**논문용 데이터 고정 방법**
- 실제 데이터로 생성된 CSV/JSON을 `paper/data/elasticnet_grid_real_spy_{timestamp}.csv` 등으로 복사합니다.
- README나 논문에서 이 경로를 직접 참조합니다.

---

## 3. 경제적 백테스트 재사용

이미 검증된 결과가 `data/raw/rv_economic_backtest_results.json`에 저장되어 있으므로, 동일 스크립트를 다시 실행할 필요가 없습니다. 논문에서는 이 JSON을 Appendix/표로 옮기고, 생성 명령(`src/validation/economic_backtest_validator.py`)만 문서에 명시합니다.

필요 시 향후 개선:
- Sharpe/수익률이 B&H 대비 낮으므로, 본문에서 “경제적 가치” 대신 “리스크 관리 지표” 등으로 표현을 조정하거나, 거래비용 민감도 분석을 추가합니다.

---

## 4. 문서 및 대시보드 정합성

- README와 `docs/technical/VALIDATION_METHODOLOGY.md`에서 인용하는 모든 수치는 `data/validation/comprehensive_model_validation.json`과 동일해야 합니다. 값이 변경되지 않았다면 문서 본문을 수정할 필요는 없으며, 단지 “결과 파일 위치”를 명시합니다.
- Streamlit은 `data/raw/model_performance.json`을 읽으므로, 위 파일만 업데이트하면 대시보드 수치도 자동으로 맞춰집니다.

---

## 5. 논문 제출 패키지 구성

1. 최종적으로 사용할 파일을 `paper/data/`에 복사합니다.
   - 예: `paper/data/model_performance_paper.json`, `paper/data/comprehensive_model_validation_paper.json`, `paper/data/elasticnet_grid_real_spy.csv`, `paper/data/rv_economic_backtest_results.json`
2. `paper/PAPER_STRUCTURE.md` 또는 README에서 “논문용 데이터 패키지” 경로를 언급해 reviewer가 즉시 접근할 수 있도록 합니다.

---

## 6. 반복 작업 최소화 요약

- **실제 데이터가 성공적으로 로드된 경우**: 기존 산출물(JSON/CSV)을 그대로 사용하고, 타임스탬프/경로만 명시합니다.
- **실패한 경우**: 네트워크 문제만 해결한 뒤 동일 명령을 한 번 더 실행하면 됩니다.
- **경제적 백테스트**: 이미 저장된 결과를 인용하고, 추가 실행은 하지 않습니다.
- **문서/대시보드**: 값 변경 없이 파일 경로 링크만 업데이트합니다.

이 계획을 따르면 불필요한 반복 실험 없이, 기존 산출물을 최대한 활용하면서 재현성과 논문 제출 요구를 충족할 수 있습니다.
