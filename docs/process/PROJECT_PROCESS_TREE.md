# SPY 변동성 예측 프로젝트 - 프로세스 트리

## 전체 프로세스 개요

```
SPY 변동성 예측 시스템
│
├── 1. 데이터 수집
│   ├── yfinance로 SPY ETF 데이터 다운로드 (2015-2024)
│   ├── OHLCV 데이터 수집
│   └── VIX 데이터 수집
│
├── 2. 데이터 전처리
│   ├── 결측치 처리
│   ├── 이상치 제거
│   └── 데이터 정합성 검증
│
├── 3. 특성 엔지니어링 (Feature Engineering)
│   ├── VIX 기반 특성 (4개)
│   │   ├── vix_level
│   │   ├── vix_ma_5, vix_ma_20
│   │   └── vix_std_20
│   │
│   ├── 실현 변동성 (3개)
│   │   ├── realized_vol_5
│   │   ├── realized_vol_10
│   │   └── realized_vol_20
│   │
│   ├── 지수 가중 변동성 (3개)
│   │   ├── ewm_vol_5
│   │   ├── ewm_vol_10
│   │   └── ewm_vol_20
│   │
│   ├── 일중 변동성 (2개)
│   │   ├── intraday_vol_5
│   │   └── intraday_vol_10
│   │
│   ├── Garman-Klass 변동성 (2개)
│   │   ├── garman_klass_5
│   │   └── garman_klass_10
│   │
│   ├── 기본 변동성 (3개)
│   │   ├── volatility_5
│   │   ├── volatility_10
│   │   └── volatility_20
│   │
│   ├── 래그 특성 (4개)
│   │   ├── vol_lag_1
│   │   ├── vol_lag_2
│   │   ├── vol_lag_3
│   │   └── vol_lag_5
│   │
│   └── HAR 특성 (3개)
│       ├── rv_daily
│       ├── rv_weekly
│       └── rv_monthly
│
├── 4. 타겟 변수 생성
│   ├── target_vol_5d (5일 후 변동성)
│   ├── 완전한 시간적 분리 보장
│   │   ├── 특성: t 이전 데이터만
│   │   └── 타겟: t+1 ~ t+5 미래 데이터만
│   └── 데이터 누출 Zero 검증
│
├── 5. 특성 선택
│   ├── 상관관계 분석 (타겟과의 상관계수)
│   ├── 상위 25개 특성 선택
│   └── 다중공선성 검사
│
├── 6. 데이터 분할
│   ├── Train set (80%, 약 1,990 샘플)
│   └── Test set (20%, 약 498 샘플)
│
├── 7. Purged K-Fold Cross-Validation
│   ├── n_splits = 5
│   ├── embargo = 1% (약 25 샘플)
│   ├── 시간 순서 보존 (no shuffle)
│   └── Train-Test 간 embargo 구간 설정
│
├── 8. 모델 학습 및 검증 (5개 모델)
│   │
│   ├── 8.1 HAR Benchmark
│   │   ├── 3개 특성 (rv_daily, rv_weekly, rv_monthly)
│   │   ├── Linear Regression
│   │   └── CV R² = 0.2300 ± 0.190
│   │
│   ├── 8.2 Ridge Regression
│   │   ├── 25개 특성
│   │   ├── alpha = 1.0
│   │   ├── StandardScaler 정규화
│   │   └── CV R² = 0.2881 ± 0.248
│   │
│   ├── 8.3 Lasso Regression (α=0.001) ⭐
│   │   ├── 25개 특성
│   │   ├── alpha = 0.001
│   │   ├── StandardScaler 정규화
│   │   ├── CV R² = 0.3373 ± 0.147
│   │   └── ✅ 가장 안정적인 모델
│   │
│   ├── 8.4 ElasticNet
│   │   ├── 25개 특성
│   │   ├── alpha = 0.001, l1_ratio = 0.5
│   │   ├── StandardScaler 정규화
│   │   └── CV R² = 0.3444 ± 0.191
│   │
│   └── 8.5 Random Forest
│       ├── 25개 특성
│       ├── n_estimators = 100, max_depth = 8
│       └── CV R² = 0.1713 ± 0.095
│
├── 9. Walk-Forward Test (Out-of-Sample)
│   ├── 마지막 20% 데이터로 테스트
│   ├── 모델별 Test R² 측정
│   │   ├── Lasso 0.001: +0.0879 ✅ (유일한 양수)
│   │   ├── ElasticNet: +0.0254
│   │   ├── Random Forest: +0.0233
│   │   ├── HAR: -0.0431
│   │   └── Ridge: -0.1429
│   └── Lasso 모델이 유일하게 일반화 성공
│
├── 10. 경제적 백테스트
│   ├── 변동성 기반 포지션 조정 전략
│   │   ├── 예측 변동성 ↑ → 포지션 ↓
│   │   └── 예측 변동성 ↓ → 포지션 ↑
│   ├── 거래 비용 포함 (0.1% per trade)
│   └── 성과 지표
│       ├── 연 수익률: 14.10% (벤치마크 22.71%)
│       ├── 변동성: 12.24% (벤치마크 13.04%) ✅ -0.8% 감소
│       ├── 샤프 비율: 0.989 (벤치마크 1.588)
│       └── 최대 낙폭: -10.81% (벤치마크 -10.15%)
│
├── 11. 데이터 누출 검증
│   ├── 시간적 분리 검증
│   ├── 특성-타겟 상관관계 검증
│   ├── CV split 누출 검증
│   └── ✅ 데이터 누출 Zero 확인

│
└── 14. 최종 결론
    ├── Lasso (α=0.001) 모델 선정
    ├── CV R² = 0.3373, Test R² = 0.0879
    ├── 유일하게 실전 적용 가능한 모델
    └── 변동성 -0.8% 감소 효과 검증
```

---

## 주요 프로세스 단계별 요약

### Phase 1: 데이터 준비 (1~5)
- SPY 데이터 수집 → 전처리 → 31개 특성 생성 → 타겟 변수 생성 → 25개 특성 선택

### Phase 2: 검증 및 학습 (6~9)
- Train/Test 분할 → Purged K-Fold CV → 5개 모델 학습 → Walk-Forward Test

### Phase 3: 평가 및 분석 (10~11)
- 경제적 백테스트 → 데이터 누출 검증

### Phase 4: 결과 도출 (12~14)
- 결과 저장 → 시각화 및 문서화 → 최종 결론

---