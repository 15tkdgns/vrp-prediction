# ElasticNet Grid Experiment

단계적으로 ElasticNet 변동성 모델의 가능성을 확인하기 위한 실험 전용 디렉터리입니다.  
기존 메인 파이프라인을 건드리지 않고, 새로운 구성요소는 모두 `experiments/elasticnet_grid/` 하위에서 진행합니다.

## 구조

```
experiments/elasticnet_grid/
├── README.md                        # 이 문서
├── run_elasticnet_grid.py           # 그리드 실험 러너
└── results/                         # 실행 결과(JSON/CSV)가 자동 저장되는 폴더
```

## 실험 개요

- **목적**: Purged K-Fold 기준으로 ElasticNet의 파라미터(alpha, l1_ratio)와 간단한 피처 확장안의 성능을 비교합니다.
- **데이터**: `src/models/correct_target_design.py`가 사용하는 SPY 2015-2024 데이터 로더/피처 생성 함수를 그대로 재사용합니다.
- **검증 방법**: Purged K-Fold (5-fold, purge=5, embargo=5).
- **출력**:
  - `results/elasticnet_grid_<타임스탬프>.json`: 각 조합별 CV 평균/표준편차, fold별 점수, 피처 세트 정보.
  - `results/elasticnet_grid_<타임스탬프>.csv`: JSON 데이터를 한눈에 볼 수 있는 표.

## 실행 방법

```bash
PYTHONPATH=/root/workspace python3 experiments/elasticnet_grid/run_elasticnet_grid.py \
  --feature-variants base extended \
  --alphas 0.05 0.08 0.10 0.15 \
  --l1-ratios 0.5 0.6 0.7 0.8
```

옵션은 모두 선택 사항이며, 값을 생략하면 위와 동일한 기본 그리드를 사용합니다.

## 결과 해석

- CSV/JSON에는 `cv_r2_mean ≥ 0.30` 또는 `cv_r2_std ≤ 0.18` 조건을 만족하는 조합이 자동으로 `candidate` 플래그(`true/false`)로 표시됩니다.
- 이 플래그가 `true`인 케이스만 후속 실험(추가 피처, Walk-Forward, 경제적 백테스트) 대상으로 삼아 비용을 줄일 수 있습니다.

## 다음 단계

1. `run_elasticnet_grid.py` 실행 → 가능성 있는 파라미터/피처 조합 식별.
2. 상위 조합만 별도 폴더(예: `experiments/walk_forward_eval/`)에서 Walk-Forward 실험.
3. 최종 후보를 경제적 백테스트에 연결.

이 디렉터리의 파일/결과만으로 실험을 재현할 수 있도록, 다른 모듈 수정 없이 이곳에서 작업을 이어가세요.
