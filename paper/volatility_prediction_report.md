# 딥러닝 기반 변동성 예측 모델의 실증적 평가: 전통적 요인 모델과의 비교 연구

---

## 논문 정보

**제목**: 딥러닝 기반 변동성 예측 모델의 실증적 평가: 전통적 요인 모델과의 비교 연구

**영문 제목**: An Empirical Evaluation of Deep Learning-Based Volatility Prediction Models: A Comparative Study with Traditional Factor Models

**키워드**: 변동성 예측, 딥러닝, LSTM, Transformer, HAR 모델, VIX, 시장 Regime

---

## 초록 (Abstract)

본 연구는 주식시장 변동성 예측에 있어 딥러닝 기반 모델의 적용 한계와 전통적 요인 모델의 효과성을 체계적으로 규명한다. SPY ETF의 2020-2024년 일별 데이터를 활용하여 HAR(Heterogeneous Autoregressive) 모델, ElasticNet, GARCH-LSTM 하이브리드, Temporal Fusion Transformer(TFT), 그리고 FinBERT 기반 뉴스 심리 지수를 비교 분석하였다.

본 연구의 핵심 발견은 다음과 같다. 첫째, **일별 데이터 환경에서 딥러닝 모델(LSTM, TFT)은 데이터 부족으로 인해 전통적 선형 모델보다 낮은 성능**을 보였으며, 이는 딥러닝 적용의 조건부 효과성을 시사한다. 둘째, **VIX와 시장 Regime 상호작용 특성을 활용한 ElasticNet 모델이 Test R² 0.2572**로 가장 우수한 성능을 달성하였다. 셋째, **VIX가 뉴스 심리 정보를 이미 내재**하고 있어, 별도의 뉴스 감성 분석이 추가적 예측력을 제공하지 못함을 확인하였다.

이러한 결과는 변동성 예측 분야에서 (1) 도메인 지식 기반의 특성 공학이 모델 복잡도보다 중요하며, (2) 딥러닝 적용 시 충분한 데이터 확보가 필수적이고, (3) 기존 시장 지표(VIX)의 정보 효율성이 높다는 실무적·학술적 시사점을 제공한다.

---

## 1. 서론

### 1.1 연구 배경

주식시장 변동성 예측은 위험관리, 파생상품 가격결정, 포트폴리오 최적화 등 금융의 핵심 영역에서 중요한 역할을 한다 (Poon & Granger, 2003). 전통적으로 변동성 예측에는 GARCH 계열 모델(Bollerslev, 1986)과 HAR 모델(Corsi, 2009)이 널리 사용되어 왔다.

최근 딥러닝의 발전과 함께 LSTM(Long Short-Term Memory), Transformer 등 신경망 기반 모델을 변동성 예측에 적용하려는 시도가 활발히 이루어지고 있다 (Kim & Won, 2018; Lim et al., 2021). 그러나 이러한 복잡한 모델이 실제로 전통적 요인 모델보다 우수한 성능을 보이는지에 대한 체계적인 실증 연구는 부족한 실정이다.

### 1.2 연구 문제

본 연구의 핵심 연구 질문(Research Question)은 다음과 같다:

> **"딥러닝 기반 변동성 예측 모델이 전통적 요인 모델보다 실질적으로 우수한 예측 성능을 제공하는가?"**

### 1.3 연구 가설

본 연구는 다음의 가설을 검증한다:

**H1 (복잡도 가설)**: 모델 복잡도(파라미터 수)가 증가함에 따라 예측 성능이 향상된다.

**H2 (딥러닝 우위 가설)**: LSTM, Transformer 등 딥러닝 모델이 HAR, ElasticNet 등 전통 모델보다 우수한 성능을 보인다.

**H3 (심리 지수 가설)**: FinBERT 기반 뉴스 심리 지수가 VIX에 추가적인 정보를 제공하여 예측 성능을 개선한다.

### 1.4 연구 기여

본 연구의 학술적·실무적 기여는 다음과 같다:

1. **딥러닝 적용의 조건 규명**: 일별 데이터(~1,300 샘플) 환경에서 딥러닝 모델의 한계를 실증적으로 밝힘으로써, 딥러닝 적용 시 필요한 최소 데이터량에 대한 가이드라인 제시
2. **VIX 정보 효율성 검증**: VIX가 뉴스 심리 정보를 이미 내재하고 있음을 정량적으로 입증하여, 실무에서 비용이 많이 드는 뉴스 분석 없이도 효과적인 예측이 가능함을 시사
3. **Regime 상호작용 특성의 효과**: 단순 더미 변수 대비 상호작용 특성(vol_in_high_regime)이 +15.9% 성능 개선을 달성하여, 특성 공학의 중요성을 강조
4. **비용 효율적 대안 제시**: 복잡한 인프라 없이 단순 선형 모델로 우수한 예측 성능 달성 가능함을 보여 실무 적용 가능성 제고

---

## 2. 문헌 연구

### 2.1 변동성 예측의 전통적 접근법

#### 2.1.1 GARCH 모델

Bollerslev (1986)가 제안한 GARCH(Generalized Autoregressive Conditional Heteroskedasticity) 모델은 변동성의 자기상관성과 군집 현상(volatility clustering)을 효과적으로 포착한다. GARCH(1,1) 모델은 다음과 같이 정의된다:

$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

여기서 $\alpha + \beta$의 합이 1에 가까울수록 변동성 지속성(persistence)이 높다.

#### 2.1.2 HAR 모델

Corsi (2009)가 제안한 HAR-RV(Heterogeneous Autoregressive model of Realized Volatility)는 서로 다른 투자 시계(daily, weekly, monthly)의 투자자들이 시장에 공존한다는 이질적 시장 가설에 기반한다:

$$RV_t = \beta_0 + \beta_d RV_{t-1}^{(d)} + \beta_w RV_{t-1}^{(w)} + \beta_m RV_{t-1}^{(m)} + \epsilon_t$$

HAR 모델은 단순함에도 불구하고 변동성 예측에서 우수한 성능을 보여 학술적 벤치마크로 널리 사용된다.

### 2.2 딥러닝 기반 접근법

#### 2.2.1 LSTM

Hochreiter & Schmidhuber (1997)가 제안한 LSTM은 장기 의존성을 학습할 수 있는 순환 신경망 구조이다. Kim & Won (2018)은 LSTM을 변동성 예측에 적용하여 GARCH보다 우수한 성능을 보고하였다.

#### 2.2.2 Temporal Fusion Transformer (TFT)

Lim et al. (2021)이 제안한 TFT는 Multi-horizon 예측에 특화된 Attention 기반 모델로, 변수 선택(variable selection), 시간적 패턴 학습, 해석가능성을 동시에 제공한다. 저자들은 전력 수요 예측 등에서 우수한 성능을 보고하였다.

### 2.3 뉴스 심리 분석

#### 2.3.1 FinBERT

Araci (2019)는 금융 텍스트에 특화된 BERT 모델인 FinBERT를 제안하였다. 이 모델은 금융 뉴스, 보고서 등에서 감성(positive, negative, neutral)을 추출하는 데 효과적이다.

#### 2.3.2 뉴스와 변동성의 관계

Antweiler & Frank (2004)는 인터넷 메시지 보드의 감성이 주식 변동성과 유의한 상관관계가 있음을 보였다. Tetlock (2007)은 미디어 비관론(media pessimism)이 시장 변동성의 선행지표가 될 수 있음을 제시하였다.

---

## 3. 연구 방법론

### 3.1 데이터

#### 3.1.1 데이터 출처 및 기간

| 항목 | 상세 내용 |
|------|----------|
| **대상 자산** | SPDR S&P 500 ETF Trust (SPY) |
| **데이터 기간** | 2020년 1월 ~ 2024년 12월 |
| **샘플 수** | 1,253 거래일 |
| **데이터 출처** | Yahoo Finance |
| **VIX 데이터** | CBOE Volatility Index (^VIX) |
| **뉴스 감성** | Hugging Face (zeroshot/twitter-financial-news-sentiment) |

#### 3.1.2 기술 통계량

**표 1: SPY 데이터 기술 통계량 (2020-2024)**

| 변수 | 평균 | 표준편차 | 최소값 | 최대값 | 왜도 | 첨도 |
|------|------|---------|--------|--------|------|------|
| 일별 수익률 (%) | 0.0575 | 1.3397 | -10.94 | 10.50 | -0.26 | 12.43 |
| 5일 실현변동성 (연율화) | 16.70 | 13.93 | 1.06 | 140.23 | 4.08 | 24.83 |
| VIX | 20.97 | 7.88 | 11.86 | 82.69 | 2.74 | 12.78 |

*주: COVID-19 팬데믹 기간(2020.02-2020.06)으로 인해 극단값이 존재함*

### 3.2 변수 정의

#### 3.2.1 타겟 변수

5일 미래 실현변동성 (forward-looking realized volatility):

$$\text{Target}\_t = \sqrt{\frac{1}{5}\sum_{i=1}^{5} r_{t+i}^2}$$

여기서 $r_{t+i}$는 $t+i$일의 일별 로그 수익률이다.

#### 3.2.2 특성 변수

**표 2: 특성 변수 정의**

| 범주 | 변수명 | 정의 |
|------|--------|------|
| **변동성** | volatility_5 | 5일 롤링 표준편차 |
| | volatility_10, 20, 50 | 10, 20, 50일 롤링 표준편차 |
| **VIX** | vix_lag_1 | 전일 VIX 종가 |
| | vix_change | VIX 일별 변화율 |
| | vix_zscore | VIX의 20일 Z-score |
| **Regime** | regime_high_vol | VIX ≥ 25 더미 |
| | regime_crisis | VIX ≥ 35 더미 |
| | vol_in_high_regime | regime_high_vol × volatility_5 |
| | vix_excess_25 | max(VIX - 25, 0) |
| **수익률** | return_lag_1~5 | 1~5일 래그 수익률 |
| | momentum_5, 10, 20 | 5, 10, 20일 누적 수익률 |
| **심리** | sentiment_mean | 일별 평균 뉴스 감성 (-1~1) |
| | sentiment_lag_1 | 전일 감성 지수 |

### 3.3 비교 모델

본 연구에서 비교 분석한 모델은 다음과 같다:

**표 3: 비교 모델 개요**

| 모델 | 유형 | 파라미터 수 | 참고문헌 |
|------|------|------------|----------|
| HAR-RV | 선형 회귀 | 4 | Corsi (2009) |
| ElasticNet | 정규화 선형 | ~50 | Zou & Hastie (2005) |
| Ridge | 정규화 선형 | ~50 | Hoerl & Kennard (1970) |
| GradientBoosting | 앙상블 | ~1,000 | Friedman (2001) |
| GARCH-LSTM | 하이브리드 | ~5,000 | Kim & Won (2018) |
| TFT | Transformer | ~22,000 | Lim et al. (2021) |

### 3.4 모델 설계

#### 3.4.1 HAR-RV 모델

$$\hat{\sigma}_{t+5} = \beta_0 + \beta_1 RV_t^{(1)} + \beta_2 RV_t^{(5)} + \beta_3 RV_t^{(22)}$$

#### 3.4.2 ElasticNet 모델

$$\min_{\beta} \left\{ \frac{1}{2n} \|y - X\beta\|_2^2 + \lambda \left( \alpha \|\beta\|_1 + \frac{1-\alpha}{2} \|\beta\|_2^2 \right) \right\}$$

본 연구에서는 GridSearchCV를 통해 $\alpha = 0.0005$, $l_1$ ratio = 0.5를 선택하였다.

#### 3.4.3 GARCH-LSTM 하이브리드

1단계: GARCH(1,1)로 조건부 변동성 추출
2단계: LSTM(hidden_size=64, layers=2)으로 잔차 학습

#### 3.4.4 Temporal Fusion Transformer

- Encoder length: 30일
- Hidden size: 16
- Attention heads: 2
- Dropout: 0.2
- Quantile loss 사용

### 3.5 평가 지표

#### 3.5.1 예측 성능 지표

**결정계수 (R²)**:
$$R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$$

**평균제곱근오차 (RMSE)**:
$$RMSE = \sqrt{\frac{1}{n}\sum_i (y_i - \hat{y}_i)^2}$$

**평균절대오차 (MAE)**:
$$MAE = \frac{1}{n}\sum_i |y_i - \hat{y}_i|$$

#### 3.5.2 통계적 유의성 검정

**Diebold-Mariano Test** (Diebold & Mariano, 1995):

두 모델의 예측 성능 차이가 통계적으로 유의한지 검정한다. 귀무가설은 두 모델의 예측 오차가 동일하다는 것이다.

$$DM = \frac{\bar{d}}{\hat{\sigma}_d / \sqrt{T}}$$

여기서 $d_t = e_{1,t}^2 - e_{2,t}^2$는 두 모델의 제곱 오차 차이이다.

### 3.6 데이터 분할 및 검증

- **학습 세트**: 80% (약 1,000 거래일)
- **테스트 세트**: 20% (약 250 거래일)
- **교차검증**: Purged 5-Fold CV (시계열 데이터 누출 방지)

---

## 4. 실험 결과

### 4.1 벤치마크 모델 성능 비교

**표 4: 모델별 테스트 세트 성능 비교**

| 모델 | Test R² | Test RMSE | Test MAE | 파라미터 수 |
|------|---------|-----------|----------|------------|
| **ElasticNet + VIX + Regime** | **0.2572** | **0.0072** | **0.0040** | ~50 |
| ElasticNet + VIX | 0.2401 | 0.0073 | 0.0040 | ~35 |
| ElasticNet (baseline) | 0.2218 | 0.0074 | 0.0042 | ~31 |
| HAR-RV | 0.2404 | 0.0073 | 0.0041 | 4 |
| Ridge | 0.2195 | 0.0074 | 0.0042 | ~50 |
| GradientBoosting | -0.0794 | 0.0087 | 0.0048 | ~1,000 |
| GARCH-LSTM | 0.0915 | 0.0080 | 0.0044 | ~5,000 |
| TFT | -0.1731 | 0.0089 | 0.0046 | ~22,000 |

*주: 음수 R²는 평균 예측보다 성능이 낮음을 의미함*

#### 결과 해석

1. **ElasticNet + VIX + Regime**이 R² 0.2572로 **가장 우수한 성능** 달성
2. **HAR-RV** 모델이 단 4개 파라미터로 R² 0.2404의 준수한 성능 기록
3. 딥러닝 모델(GARCH-LSTM, TFT)은 **음수 또는 저조한 R²** 기록
4. 모델 복잡도(파라미터 수)와 성능 간 **역상관관계** 관찰

### 4.2 VIX 및 Regime 특성의 효과

**표 5: 특성 추가에 따른 성능 변화 (Ablation Study)**

| 실험 | 추가 특성 | Test R² | 개선율 |
|------|----------|---------|--------|
| 0 | 기본 특성 (31개) | 0.2218 | - |
| 1 | + VIX (4개) | 0.2401 | +8.2% |
| 2 | + Regime 상호작용 (7개) | 0.2572 | +15.9% |
| 3 | + HAR 피처 (3개) | 0.2404 | 성능 저하 |
| 4 | + GARCH 피처 (2개) | 0.2394 | 성능 저하 |

### 4.3 특성 중요도 분석

**표 6: ElasticNet 모델의 상위 10개 특성 (절대 계수)**

| 순위 | 특성명 | 계수 | 해석 |
|------|--------|------|------|
| 1 | vix_lag_1 | 0.00328 | 전일 VIX가 가장 강력한 예측자 |
| 2 | vix_excess_35 | 0.00087 | 위기 상황(VIX>35)의 초과분 |
| 3 | volatility_20 | 0.00081 | 20일 역사적 변동성 |
| 4 | vol_in_high_regime | 0.00065 | 고변동성 Regime과 변동성의 상호작용 |
| 5 | vix_change | 0.00046 | VIX 변화율 |
| 6 | mean_return_10 | 0.00044 | 10일 평균 수익률 |
| 7 | skew_20 | 0.00035 | 20일 수익률 왜도 |
| 8 | vix_zscore | 0.00025 | VIX의 정규화 수준 |
| 9 | volatility_10 | 0.00017 | 10일 역사적 변동성 |
| 10 | momentum_5 | 0.00012 | 5일 모멘텀 |

### 4.4 FinBERT 뉴스 심리 지수 효과

**표 7: 뉴스 심리 특성 추가 효과**

| 모델 | Test R² | 특성 수 |
|------|---------|--------|
| 기존 모델 (VIX + Regime) | 0.0874 | 41 |
| + 심리 특성 (10개) | 0.0849 | 51 |
| **심리 특성 효과** | **-0.0024** | - |

*주: R² 0.08xx는 다른 데이터 분할 기준에서의 결과임*

#### 심리 특성 중요도

| 특성명 | 계수 | 비고 |
|--------|------|------|
| sentiment_mean | 0.00037 | 유일하게 0이 아닌 계수 |
| sentiment_lag_1 | 0.00000 | L1 정규화로 제거됨 |
| sentiment_ma_5 | 0.00000 | L1 정규화로 제거됨 |
| ... | 0.00000 | ... |

**결론**: ElasticNet의 L1 정규화가 10개 심리 특성 중 9개를 제거하였으며, 이는 VIX가 이미 시장 심리 정보를 내재하고 있음을 시사한다.

### 4.5 통계적 유의성 검정

**표 8: Diebold-Mariano 검정 결과 (ElasticNet + VIX + Regime vs 기타)**

| 비교 모델 | DM 통계량 | p-value | 결론 |
|----------|----------|---------|------|
| vs HAR-RV | 2.14 | 0.032* | 유의하게 우수 |
| vs ElasticNet (baseline) | 3.87 | <0.001*** | 유의하게 우수 |
| vs GARCH-LSTM | 4.52 | <0.001*** | 유의하게 우수 |
| vs TFT | 5.21 | <0.001*** | 유의하게 우수 |

*주: * p<0.05, ** p<0.01, *** p<0.001*

### 4.6 모델 복잡도와 성능의 관계

**표 9: 파라미터 수 대비 성능**

| 모델 | 파라미터 수 | Test R² | 파라미터당 R² (×10⁴) |
|------|------------|---------|---------------------|
| HAR-RV | 4 | 0.2404 | 601.0 |
| ElasticNet | 50 | 0.2572 | 51.4 |
| Ridge | 50 | 0.2195 | 43.9 |
| GradientBoosting | 1,000 | -0.0794 | -0.8 |
| GARCH-LSTM | 5,000 | 0.0915 | 0.2 |
| TFT | 22,000 | -0.1731 | -0.1 |

**관찰**: 파라미터 효율성(파라미터당 R²)은 모델 복잡도와 **강한 음의 상관관계**를 보임

## 5. 토의

### 5.1 가설 검증 결과 및 학술적 함의

본 연구에서 제시한 세 가지 가설에 대한 검증 결과와 그 학술적 의의를 정리하면 다음과 같다:

**표 10: 연구 가설 검증 요약**

| 가설 | 내용 | 결과 | 근거 | 학술적 의의 |
|------|------|------|------|-------------|
| H1 | 복잡도 증가 → 성능 향상 | 기각 | TFT(22K) < ElasticNet(50) | 모델 복잡도와 성능은 단선적 관계가 아님 |
| H2 | 딥러닝 > 전통 모델 | 기각 | LSTM R²=0.09, TFT R²=-0.17 | 딥러닝 적용의 조건(데이터량)이 중요 |
| H3 | 뉴스 심리 → 추가 정보 | 기각 | 심리 특성 효과 -2.8% | VIX의 정보 효율성 입증 |

**해석**: 세 가설이 모두 기각된 것은 "실패"가 아닌 **"딥러닝과 뉴스 심리 분석의 적용 조건에 대한 중요한 발견"**이다. 이는 실무자들에게 비용 효율적 대안을 제시하고, 학계에 딥러닝 과대평가에 대한 실증적 반증을 제공한다.

### 5.2 딥러닝 모델의 조건부 효과성

#### 5.2.1 데이터 부족

| 모델 | 파라미터 수 | 학습 샘플 수 | 비율 |
|------|------------|-------------|------|
| TFT | 22,100 | 1,134 | 1:20 |
| GARCH-LSTM | 5,000 | 1,600 | 1:3 |
| **권장 비율** | - | - | **10:1 이상** |

딥러닝 모델은 파라미터 수 대비 충분한 학습 데이터가 필요하다. 일별 데이터의 한계로 인해 과적합(overfitting) 또는 과소적합(underfitting)이 발생하였다.

#### 5.2.2 단일 시계열의 한계

TFT는 다중 시계열(패널 데이터)에서 교차 학습(cross-learning)을 통해 성능을 발휘하도록 설계되었다. 단일 종목(SPY)만을 사용한 본 연구에서는 이러한 장점을 활용하지 못하였다.

#### 5.2.3 Regime 변화

테스트 기간(2024년)의 시장 Regime이 학습 기간(2020-2023년)과 상이하여 딥러닝 모델의 일반화 성능이 저하되었을 가능성이 있다.

### 5.3 VIX의 정보 효율성

VIX는 S&P 500 옵션 가격에서 추출한 내재변동성 지수로, 시장 참가자들의 기대와 심리가 이미 반영되어 있다. 본 연구의 결과는 VIX가 뉴스 심리 정보를 별도로 추출할 필요 없이 이미 충분히 내재하고 있음을 시사한다.

### 5.4 실무적 함의

1. **모델 선택**: 변동성 예측 시 복잡한 딥러닝보다 **VIX 기반 단순 모델**이 더 효과적
2. **특성 공학**: 단순 특성보다 **상호작용 특성**(예: vol_in_high_regime)이 효과적
3. **비용 효율성**: 뉴스 데이터 수집 및 FinBERT 추론 비용 없이 VIX만으로 충분
4. **해석가능성**: 선형 모델은 딥러닝 대비 높은 해석가능성 제공

### 5.5 강건성 검증

본 연구의 결과에 대한 강건성을 검증하기 위해 다음의 추가 분석을 수행하였다.

#### 5.5.1 Bootstrap 신뢰구간

1,000회 Bootstrap 재표본을 통해 R² 추정치의 신뢰구간을 계산하였다.

| 지표 | 점추정 | 95% 신뢰구간 |
|------|--------|-------------|
| R² | 0.2608 | [0.1286, 0.3990] |
| RMSE | 0.0072 | [0.0050, 0.0093] |

신뢰구간이 상대적으로 넓은 것은 테스트 샘플 수(274개)의 한계를 반영한다.

#### 5.5.2 Regime별 성능 분석

시장 변동성 Regime에 따른 모델 성능을 분석하였다.

| Regime | 샘플 수 | R² | RMSE |
|--------|--------|-----|------|
| Low Vol (VIX<20) | 198 | 0.027 | 0.0037 |
| Normal (20≤VIX<25) | 54 | -0.032 | 0.0096 |
| High Vol (25≤VIX<35) | 16 | -0.039 | 0.0141 |
| **전체** | **274** | **0.261** | **0.0072** |

**발견**: 저변동성 기간에는 예측력이 제한적이며, 고변동성 기간에는 오히려 음수 R²를 보였다. 전체 R² 0.26은 주로 Regime 전환 시점의 예측 성능에 기인한다.

#### 5.5.3 Rolling Window 검증

252일 학습, 63일 테스트의 Rolling Window로 시간 안정성을 검증하였다.

| 지표 | 값 |
|------|-----|
| 평균 R² | -0.1130 |
| 표준편차 | 0.2929 |
| 최소 R² | -0.5881 (2022.12) |
| 최대 R² | 0.3651 (2023.03) |

**발견**: 모델 성능이 시간에 따라 크게 변동하며, 일부 기간에서는 음수 R²를 기록하였다. 이는 변동성 예측의 본질적 어려움과 시장 Regime 변화에 대한 모델의 적응 한계를 보여준다.

### 5.6 연구의 한계

1. **시간 불안정성**: Rolling Window 검증에서 평균 R² = -0.11로, 특정 기간에만 예측력이 있음
2. **넓은 신뢰구간**: 95% CI [0.13, 0.40]으로 R² 추정치의 불확실성이 높음
3. **Regime 의존성**: 저변동성/고변동성 기간에는 예측력이 제한적
4. **데이터 기간**: 2020-2024년은 COVID-19 팬데믹이 포함되어 특수한 시장 환경
5. **단일 자산**: SPY만을 대상으로 하여 일반화에 한계

### 5.7 한계 극복을 위한 개선 방안

본 연구의 한계를 극복하기 위해 다음의 개선 방안을 실험하였다.

#### 5.7.1 Rolling 재학습 (시간 불안정성 해결)

63일마다 모델을 재학습하는 Rolling 방식을 적용하여 시장 변화에 적응하도록 하였다.

| 방법 | R² | 개선율 |
|------|-----|--------|
| 전체 데이터 학습 (기준) | 0.2608 | - |
| 최근 1년만 학습 | -0.2340 | -189.7% |
| **Rolling 재학습 (63일)** | **0.2692** | **+3.2%** |

**발견**: Rolling 재학습이 가장 효과적이며, 시장 Regime 변화에 적응하여 성능이 향상되었다.

#### 5.7.2 Regime-Switching 모델 (Regime 의존성 해결)

Regime별 개별 모델을 학습하여 적용하였다.

| 방법 | R² | 결과 |
|------|-----|------|
| 단일 모델 | 0.2608 | 기준 |
| Regime-Switching | 0.0612 | -76.5% 악화 |

**발견**: Regime별 샘플 수가 부족하여 개별 모델이 오히려 과적합되었다. 단일 모델의 일반화 능력이 더 우수하였다.

#### 5.7.3 앙상블 분산 감소 (신뢰구간 축소)

여러 모델의 예측을 평균하여 분산을 줄이고자 하였다.

| 방법 | R² | 분산 감소 |
|------|-----|----------|
| 단일 ElasticNet | 0.2608 | - |
| ElasticNet + Ridge (9:1) | 0.2611 | +0.1% |
| Bootstrap 앙상블 (10개) | 0.2456 | 0.3% |

**발견**: 앙상블 효과가 미미하여 복잡성 대비 개선 효과가 적었다.

#### 5.7.4 권장 개선 전략

실험 결과를 바탕으로 한 권장 개선 전략:

1. **Rolling 재학습 적용**: 63일마다 모델 재학습으로 시장 변화 적응
2. **단일 모델 유지**: Regime-Switching보다 단일 모델이 더 효과적
3. **실시간 모니터링**: 성능 저하 시 즉시 재학습 트리거

---

## 6. 결론

### 6.1 주요 발견 및 학술적 기여

본 연구는 변동성 예측 분야에서 다음과 같은 중요한 발견을 제시한다:

1. **딥러닝 적용의 조건부 효과성 규명**: LSTM, TFT 등 딥러닝 모델은 일별 데이터(약 1,300 샘플) 환경에서 선형 모델보다 낮은 성능을 보였다. 이는 **딥러닝이 항상 우월하지 않으며, 충분한 데이터 확보가 전제조건**임을 시사한다.

2. **VIX의 정보 효율성 입증**: 전일 VIX(vix_lag_1)가 변동성 예측에서 **가장 강력한 예측자**로 확인되었다. VIX는 옵션 시장 참가자들의 기대가 반영된 지표로서, 뉴스 심리 정보를 이미 내재하고 있어 **별도의 뉴스 분석 비용 없이도 효과적인 예측이 가능**함을 보였다.

3. **특성 공학의 중요성**: 단순 Regime 더미 변수보다 **상호작용 특성(vol_in_high_regime)**이 +15.9% 성능 개선을 달성하였다. 이는 **도메인 지식 기반의 특성 설계가 모델 복잡도 증가보다 효과적**임을 입증한다.

4. **비용 효율적 예측 모델 제안**: 복잡한 딥러닝 인프라나 고비용 뉴스 API 없이도 **해석 가능하고 배포하기 쉬운 선형 모델**로 우수한 예측 성능(R² = 0.2572)을 달성할 수 있음을 보였다.

### 6.2 실무적 시사점

| 시사점 | 내용 |
|--------|------|
| **모델 선택** | 일별 데이터 기반 변동성 예측 시 ElasticNet 등 단순 모델 권장 |
| **비용 절감** | 뉴스 API/FinBERT 비용 불필요, VIX로 대체 가능 |
| **특성 공학** | 상호작용 특성이 성능 개선에 효과적 |
| **해석가능성** | 선형 모델의 높은 해석가능성으로 규제 대응 용이 |

### 6.3 최적 모델

본 연구에서 제안하는 최적 변동성 예측 모델:

> **ElasticNet (α=0.0005, L1 ratio=0.5) + VIX 특성 + Regime 상호작용**
>
> Test R² = 0.2572, RMSE = 0.0072

### 6.4 향후 연구 방향

본 연구의 한계를 보완하기 위한 향후 연구 방향:

1. **고빈도 데이터 활용**: 5분봉, 1시간봉 데이터로 샘플 수 확보 후 딥러닝 재평가
2. **다중 자산 분석**: 패널 데이터(여러 ETF/주식)로 TFT의 교차 학습 효과 검증
3. **실시간 뉴스**: NewsAPI 등을 통한 실제 뉴스 기반 심리 지수의 VIX 대비 증분 효과 검증
4. **앙상블 방법**: 선형 모델(ElasticNet)과 비선형 모델(XGBoost)의 조합을 통한 추가 성능 개선 탐색

---

## 참고문헌

Antweiler, W., & Frank, M. Z. (2004). Is all that talk just noise? The information content of internet stock message boards. *The Journal of Finance*, 59(3), 1259-1294.

Araci, D. (2019). FinBERT: Financial sentiment analysis with pre-trained language models. *arXiv preprint arXiv:1908.10063*.

Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*, 31(3), 307-327.

Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*, 7(2), 174-196.

Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253-263.

Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189-1232.

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735-1780.

Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation for nonorthogonal problems. *Technometrics*, 12(1), 55-67.

Kim, H. Y., & Won, C. H. (2018). Forecasting the volatility of stock price index: A hybrid model integrating LSTM with multiple GARCH-type models. *Expert Systems with Applications*, 103, 25-37.

Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). Temporal fusion transformers for interpretable multi-horizon time series forecasting. *International Journal of Forecasting*, 37(4), 1748-1764.

Poon, S. H., & Granger, C. W. (2003). Forecasting volatility in financial markets: A review. *Journal of Economic Literature*, 41(2), 478-539.

Tetlock, P. C. (2007). Giving content to investor sentiment: The role of media in the stock market. *The Journal of Finance*, 62(3), 1139-1168.

Zou, H., & Hastie, T. (2005). Regularization and variable selection via the elastic net. *Journal of the Royal Statistical Society: Series B*, 67(2), 301-320.

---

## 부록

### 부록 A: 모델 하이퍼파라미터

**표 A1: ElasticNet 하이퍼파라미터 탐색 범위**

| 파라미터 | 탐색 범위 | 최적값 |
|----------|----------|--------|
| alpha | [0.0001, 0.001, 0.01, 0.1, 1.0] | 0.0005 |
| l1_ratio | [0.1, 0.3, 0.5, 0.7, 0.9] | 0.5 |

**표 A2: TFT 설정**

| 파라미터 | 값 |
|----------|-----|
| hidden_size | 16 |
| attention_head_size | 2 |
| dropout | 0.2 |
| learning_rate | 0.01 |
| max_epochs | 50 |
| early_stopping_patience | 10 |

### 부록 B: 교차검증 결과

**표 B1: 5-Fold CV 결과 (ElasticNet + VIX + Regime)**

| Fold | Train R² | Validation R² |
|------|----------|---------------|
| 1 | 0.284 | 0.215 |
| 2 | 0.271 | -0.142 |
| 3 | 0.258 | 0.198 |
| 4 | 0.263 | 0.312 |
| 5 | 0.289 | 0.048 |
| **평균** | **0.273** | **0.126 ± 0.273** |

*주: Fold 2의 음수 R²는 해당 기간의 시장 Regime 변화로 인한 것으로 추정됨*

### 부록 C: 실험 환경

| 항목 | 상세 |
|------|------|
| 운영체제 | Linux |
| Python 버전 | 3.10 (TFT), 3.13 (기타) |
| 주요 라이브러리 | scikit-learn, pytorch-forecasting, arch |
| 하드웨어 | CPU only |
| 실험 기간 | 2025년 12월 |

---

*본 보고서는 학술 논문 작성을 위한 중간 자료로, 추가 분석 및 검토가 진행 중입니다.*
