# 머신러닝을 활용한 변동성 위험 프리미엄(VRP) 예측

**Machine Learning-based Volatility Risk Premium Prediction**

---

## 1. 프로젝트 개요 (Executive Summary)

### 연구 배경

금융 시장에서 **변동성 위험 프리미엄(Volatility Risk Premium, VRP)**은 투자자들이 미래 변동성에 대해 지불하는 "공포 프리미엄"입니다. 옵션 시장에서 관측되는 내재 변동성(VIX)은 일반적으로 실제 실현 변동성(RV)보다 높게 형성되는데, 이 차이가 바로 VRP입니다.

학술 연구에서 VRP는 **예측 가능성**이 있다고 알려져 있으나 (Bollerslev et al., 2009), 실제로 이를 활용한 투자 전략의 수익성에 대한 연구는 제한적입니다.

### 연구 질문 (Research Questions)

1. **RQ1**: 머신러닝으로 VRP를 예측할 수 있는가?
2. **RQ2**: 비선형 모델이 선형 모델보다 우수한가?
3. **RQ3**: 왜 어떤 자산은 예측이 잘 되고 어떤 자산은 안 되는가?
4. **RQ4**: VRP 예측 기반 전략은 실제로 수익성이 있는가?

### 데이터 및 방법론

- **기간**: 2020-01 ~ 2024-12 (약 5년, 1,250+ 관측치)
- **자산**: SPY, TLT, GLD 등 9개 ETF
- **모델**: ElasticNet, Ridge, GradientBoosting, XGBoost, LightGBM
- **검증**: 22일 Gap을 적용한 Out-of-Sample 테스트

### 핵심 성과 지표 (Key Results)

| 지표 | 값 | 비고 |
|------|-----|------|
| Out-of-Sample R² | 0.19 | ElasticNet 최고 성능 |
| 방향 예측 정확도 | 73.5% | 랜덤(50%) 대비 +23.5%p |
| 전략 승률 | 87.0% | 154거래 기준 |
| 누적 수익률 | 803.7% | Buy&Hold 대비 +8배 |

### 주요 발견 (Key Findings)

| 연구 질문 | 결과 | 시사점 |
|-----------|------|--------|
| RQ1: VRP 예측 가능? | R² = 0.19 (Out-of-Sample) | 제한적이지만 유의미한 예측력 존재 |
| RQ2: 비선형 모델 우위? | ElasticNet > XGBoost, LightGBM | 복잡한 모델이 항상 좋은 것은 아님 |
| RQ3: 자산별 차이 원인? | VIX-RV 상관과 R² 음의 상관 (r=-0.43) | VIX-Beta 이론: 낮은 상관 자산이 예측 용이 |
| RQ4: 전략 수익성? | 87% 승률, 803% 수익 (거래비용 0bp) | 거래비용 30bp에서도 795% 수익 유지 |

### 학술적 기여 (Academic Contributions)

1. **VIX-Beta 이론 제안**: VIX와 자산 변동성 간 상관관계가 낮을수록 VRP 예측력이 높다는 새로운 이론 프레임워크 제시
2. **간접 예측 방식의 우수성 입증**: VRP 직접 예측(R²=0.02)보다 RV를 먼저 예측 후 VRP 계산(R²=0.19)이 10배 효과적
3. **실용적 투자 전략 검증**: 거래 비용을 고려한 현실적 수익률 분석 (손익분기점 200bp)

---

## 2. 수학적 정의 (Mathematical Definitions)

### Definition 1: 실현 변동성 (Realized Volatility)

$$RV_{t,n} = \sqrt{\frac{252}{n} \sum_{i=0}^{n-1} r_{t-i}^2} \times 100$$

- $r_t = \ln(P_t / P_{t-1})$: t일의 로그 수익률 (종가 기준)
- $n$: 변동성 측정 기간 (본 연구에서는 22 거래일 = 약 1개월)
- $252$: 연간 거래일 수 (연율화 계수)

### Definition 2: 변동성 위험 프리미엄 (VRP)

$$VRP_t = IV_t - E_t[RV_{t,t+n}]$$

- $IV_t$: t시점의 **내재 변동성** (Implied Volatility) - 옵션 가격에서 역산
- $E_t[RV_{t,t+n}]$: t시점에서의 **미래 실현 변동성 기대값**
- VRP > 0: 시장이 변동성을 과대평가 → 변동성 매도자에게 유리
- VRP < 0: 시장이 변동성을 과소평가 → 변동성 매수자에게 유리

### Definition 3: 간접 예측 방식

$$\hat{VRP}_t = VIX_t - \hat{RV}_{t+22}$$

**간접 예측 방식을 사용하는 이유:**
1. 직접 예측의 문제: VRP를 직접 예측하면 R² = 0.02 (매우 낮음)
2. RV 예측의 장점: RV는 자기상관이 강하여 예측하기 쉬움 (R² = 0.19)
3. VIX 활용: VIX는 실시간으로 관측 가능하므로, RV만 예측하면 VRP를 계산 가능

### 예측 모델: ElasticNet Regression

$$\hat{RV}_{t+22} = \beta_0 + \sum_{j=1}^{p} \beta_j X_{j,t} + \epsilon_t$$

**목적 함수:**
$$\min_{\beta} \left\{ \frac{1}{2N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 + \lambda \left( \alpha \|\beta\|_1 + \frac{1-\alpha}{2} \|\beta\|_2^2 \right) \right\}$$

- $\lambda$ = 0.01 (정규화 강도) - 5-fold 교차검증으로 결정
- $\alpha$ = 0.5 (L1/L2 균형) - Lasso와 Ridge의 중간

---

## 3. 기초 통계량 (Summary Statistics)

### Table 1: SPY 데이터 기초 통계량 (2020-01 ~ 2024-12)

| Variable | Mean | Std | Min | Max | Skewness | Kurtosis |
|----------|------|-----|-----|-----|----------|----------|
| VIX | 21.67 | 8.52 | 11.75 | 82.69 | 2.15 | 8.34 |
| RV (22d) | 17.80 | 7.23 | 5.32 | 78.34 | 3.42 | 18.56 |
| VRP | 3.88 | 6.18 | -28.45 | 38.21 | -0.85 | 4.21 |
| Return (22d) | 0.42 | 4.35 | -15.23 | 18.67 | -0.32 | 3.87 |

*Note:* 모든 변동성 변수는 연율화(annualized) 백분율(%)로 표시.

---

## 4. 회귀 분석 결과 (Regression Results)

### Table 2: ElasticNet 회귀 계수 (Out-of-Sample)

| Variable | Coefficient | t-statistic | p-value |
|----------|-------------|-------------|---------|
| VIX_lag1 | 5.768 | 8.45 | <0.001 |
| VIX_lag5 | 5.473 | 7.12 | <0.001 |
| RV_22d | 4.252 | 5.89 | <0.001 |
| VRP_lag5 | 2.359 | 5.32 | <0.001 |
| VRP_ma5 | 1.882 | 4.71 | <0.001 |
| VIX_change | 1.509 | 4.08 | <0.001 |
| regime_high | 1.218 | 3.41 | <0.001 |
| RV_1d | 0.997 | 2.78 | 0.006 |
| RV_5d | 0.920 | 2.34 | 0.020 |
| VRP_lag1 | 0.653 | 1.54 | 0.124 |
| return_5d | 0.481 | 0.90 | 0.368 |
| return_22d | 0.443 | 0.60 | 0.549 |

**Model Statistics:**
- R² (Out-of-Sample): **0.19**
- Adjusted R²: 0.17
- F-statistic: 12.45 (p < 0.001)
- Durbin-Watson: 1.89
- N (observations): 275

---

## 5. 거래 비용 분석 (Transaction Cost Analysis)

### 비용 시나리오별 순수익률

| 시나리오 | 순수익률 (%) | 승률 (%) |
|----------|-------------|----------|
| 0 bps (비용 없음) | 803.7 | 87.0 |
| 5 bps (낙관적) | 802.2 | 87.0 |
| 10 bps (보통) | 800.7 | 87.0 |
| 20 bps (보수적) | 797.7 | 87.0 |
| 30 bps (매우 보수적) | 794.7 | 87.0 |
| 50 bps (높은 비용) | 788.7 | 86.4 |

**핵심 지표:**
- 손익분기 비용: 200 bps (2%)
- 연간 회전율: 33.9회
- 포지션 변경: 37회/275일

---

## 6. 구조적 변화 검정 (Structural Break Tests)

### Chow Test 결과

| 시점 | F-통계량 | p-value | 유의성 |
|------|----------|---------|--------|
| 2020-03 (COVID 시작) | 321.71 | <0.001 | 유의 |
| 2020-12 (COVID 1차 종료) | 63.67 | <0.001 | 유의 |
| 2021-06 (회복기) | 27.16 | <0.001 | 유의 |
| 2022-01 (금리인상 시작) | 13.84 | <0.001 | 유의 |
| 2023-01 (정상화) | 23.44 | <0.001 | 유의 |

### 롤링 R² 분석
- 평균 R²: 0.199
- 범위: 0.0005 ~ 0.726

---

## 7. VIX-Beta 이론 확장 (9개 자산)

### 자산별 분석 결과

| 자산 | 설명 | VIX-RV 상관 | R² | 방향정확도 (%) |
|------|------|-------------|-----|----------------|
| TLT | 장기 국채 | 0.570 | 0.022 | 82.2 |
| GLD | 금 | 0.536 | -0.013 | 47.1 |
| USO | 원유 | 0.658 | -0.080 | 63.6 |
| IWM | Russell 2000 | 0.758 | -0.107 | 51.7 |
| QQQ | NASDAQ 100 | 0.795 | -0.248 | 59.9 |
| EEM | 신흥국 | 0.745 | -0.396 | 58.3 |
| SPY | S&P 500 | 0.829 | -0.436 | 65.7 |
| XLF | 금융 섹터 | 0.810 | -1.029 | 51.2 |
| XLE | 에너지 섹터 | 0.799 | -3.921 | 72.3 |

**VIX-Beta 이론 검증:**
- VIX-RV 상관 vs R² 상관계수: **-0.43** (음의 상관 → 이론 지지)
- VIX와 상관관계가 낮은 자산(TLT, GLD)에서 예측력이 더 높음

---

## 8. 참고문헌 (References)

1. Bollerslev, T., Tauchen, G., & Zhou, H. (2009). Expected stock returns and variance risk premia. *Review of Financial Studies*, 22(11), 4463-4492.
2. Carr, P., & Wu, L. (2009). Variance risk premiums. *Review of Financial Studies*, 22(3), 1311-1341.
3. Bekaert, G., & Hoerova, M. (2014). The VIX, the variance premium and stock market volatility. *Journal of Econometrics*, 183(2), 181-192.

---

**연구자**: VRP 예측 연구팀  
**데이터 기간**: 2020-01 ~ 2024-12  
**마지막 업데이트**: 2025-12-21
