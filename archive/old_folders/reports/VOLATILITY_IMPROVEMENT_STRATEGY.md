# 변동성 예측 성능 향상 전략

**현재 성능**: Ridge R² = 0.30 (HAR 대비 35배)
**목표**: R² > 0.40 달성

---

## 1. 현재 시스템 분석

### 1.1 현재 모델 구조

**모델**: Ridge Regression (alpha=1.0)
**특성**: 31개 (변동성 lag, 롤링 통계)
**타겟**: 5일 후 변동성

**장점**:
- ✅ 단순하고 해석 가능
- ✅ 과적합 방지 (L2 regularization)
- ✅ HAR 벤치마크 압도

**한계**:
- ❌ 선형 관계만 포착
- ❌ 변동성 클러스터링 미흡
- ❌ 비대칭 효과 무시 (leverage effect)
- ❌ 일별 데이터만 사용 (고빈도 정보 손실)

---

## 2. 개선 전략 (우선순위별)

### 전략 1: GARCH 계열 모델 추가 ★★★★★

**핵심 아이디어**: 변동성의 자기회귀 특성 + 조건부 이분산성

#### GARCH(1,1) 기본
```
σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
```
- **장점**: 변동성 클러스터링 포착
- **예상 개선**: R² +0.05~0.10

#### EGARCH (비대칭 효과)
```
log(σ²_t) = ω + β·log(σ²_{t-1}) + α·|z_{t-1}| + γ·z_{t-1}
```
- **장점**: 레버리지 효과 (하락 시 변동성↑)
- **예상 개선**: R² +0.10~0.15

#### GJR-GARCH (임계값 효과)
```
σ²_t = ω + α·ε²_{t-1} + γ·ε²_{t-1}·I_{ε<0} + β·σ²_{t-1}
```
- **장점**: 음의 충격에 더 큰 가중치
- **예상 개선**: R² +0.08~0.12

**구현 우선순위**:
1. GARCH(1,1) - 기본
2. EGARCH - 비대칭
3. GJR-GARCH - 임계값

**예상 최종 성능**: R² = 0.40~0.45

---

### 전략 2: Realized Volatility (RV) 활용 ★★★★☆

**핵심 아이디어**: 고빈도 데이터로 실제 변동성 정확히 측정

#### Realized Volatility 계산
```
RV_t = Σ r²_{t,i}  (i = 1분 단위 수익률)
```

**장점**:
- 일별 변동성보다 정확
- 노이즈 감소
- 예측 정보 증가

**데이터 요구사항**:
- 분봉/틱 데이터 필요
- yfinance: 최대 1분봉 (최근 7일만)
- 대안: Alpha Vantage, Polygon.io (유료)

**예상 개선**: R² +0.05~0.08

**단기 구현**:
- Intraday range 프록시: `(High - Low) / Close`
- Parkinson volatility: `√(1/(4·log2) · log²(High/Low))`

---

### 전략 3: 비대칭 및 레버리지 효과 특성 ★★★★☆

**핵심 아이디어**: 하락장에서 변동성 급등 패턴 포착

#### 새 특성 추가

1. **Leverage Effect**
   ```python
   leverage = (returns < 0).astype(int) * abs(returns)
   ```

2. **Asymmetric Volatility**
   ```python
   vol_down = volatility[returns < 0].mean()
   vol_up = volatility[returns > 0].mean()
   asym_ratio = vol_down / vol_up
   ```

3. **Downside Risk**
   ```python
   downside_vol = returns[returns < 0].std()
   ```

4. **Jump Detection**
   ```python
   jump = (abs(returns) > 3 * volatility).astype(int)
   ```

**예상 개선**: R² +0.03~0.05

---

### 전략 4: 앙상블 모델 ★★★★☆

**핵심 아이디어**: 여러 모델의 강점 결합

#### 모델 조합

1. **Ridge** (현재)
   - 장점: 안정적, 해석 가능
   - 가중치: 30%

2. **GARCH** (새)
   - 장점: 변동성 클러스터링
   - 가중치: 40%

3. **XGBoost** (새)
   - 장점: 비선형 패턴
   - 가중치: 30%

#### 앙상블 방법

**가중 평균**:
```python
prediction = 0.3·Ridge + 0.4·GARCH + 0.3·XGBoost
```

**Stacking**:
- Level 0: Ridge, GARCH, XGBoost
- Level 1: Meta-learner (Linear Regression)

**예상 개선**: R² +0.08~0.12

---

### 전략 5: HAR-RV 확장 모델 ★★★☆☆

**핵심 아이디어**: HAR 모델 + Realized Volatility

#### HAR-RV 구조
```
RV_t = β₀ + β_d·RV_{t-1} + β_w·RV_{t-5:t-1} + β_m·RV_{t-22:t-1} + ε_t
```

**확장 버전 (HAR-RV-J)**:
- Jump component 추가
- Continuous + Jump variance 분리

**예상 개선**: R² +0.05~0.08

---

### 전략 6: 외부 요인 추가 ★★★☆☆

**핵심 아이디어**: VIX, 금리, 거래량 등 거시 변수

#### 새 특성

1. **VIX Index**
   - 실제 VIX 지수 (공포 지표)
   - 예상 효과: R² +0.02~0.03

2. **Interest Rate Spread**
   - 10Y - 2Y Treasury
   - 경기 사이클 반영

3. **Market Breadth**
   - Advance/Decline ratio
   - 시장 전체 심리

4. **Option Implied Volatility**
   - ATM volatility surface
   - 시장 기대 반영

**예상 개선**: R² +0.03~0.05

---

### 전략 7: 딥러닝 (LSTM/Transformer) ★★☆☆☆

**핵심 아이디어**: 장기 의존성 포착

#### LSTM Architecture
```
Input → LSTM(64) → LSTM(32) → Dense(16) → Output
```

**장점**:
- 시계열 패턴 자동 학습
- 비선형 관계 포착

**단점**:
- 데이터 많이 필요 (>10,000 샘플)
- 해석 어려움
- 과적합 위험

**예상 개선**: R² +0.05~0.10 (불확실)

**권장**: 데이터 충분할 때만

---

## 3. 구현 로드맵

### Phase 1: Quick Wins (1주)

1. **비대칭 특성 추가**
   - Leverage effect
   - Downside volatility
   - Jump detection
   - 예상: R² +0.03

2. **Intraday Range 활용**
   - Parkinson volatility
   - High-Low ratio
   - 예상: R² +0.02

**목표**: R² = 0.35

---

### Phase 2: GARCH 통합 (2주)

1. **GARCH(1,1) 구현**
   - `arch` 라이브러리 사용
   - 조건부 분산 예측

2. **EGARCH 확장**
   - 비대칭 효과 포착

3. **앙상블 (Ridge + GARCH)**
   - 가중 평균

**목표**: R² = 0.42

---

### Phase 3: 고급 모델 (3주)

1. **HAR-RV 구현**
   - Realized volatility 계산
   - Multi-horizon 특성

2. **XGBoost 추가**
   - 3-way 앙상블
   - Feature importance 분석

**목표**: R² = 0.45~0.50

---

### Phase 4: 외부 요인 (4주)

1. **VIX 통합**
2. **금리 스프레드**
3. **옵션 데이터** (가능 시)

**목표**: R² = 0.50+

---

## 4. 예상 성능 로드맵

| Phase | 모델 | 예상 R² | 개선폭 |
|-------|------|---------|--------|
| **현재** | Ridge | 0.30 | - |
| **Phase 1** | Ridge + 비대칭 | 0.35 | +0.05 |
| **Phase 2** | Ridge + GARCH | 0.42 | +0.12 |
| **Phase 3** | Ensemble (3-way) | 0.48 | +0.18 |
| **Phase 4** | Full System | 0.52 | +0.22 |

---

## 5. 핵심 기술 구현 예시

### 5.1 GARCH(1,1) 구현

```python
from arch import arch_model

# GARCH(1,1) 모델
model = arch_model(returns, vol='Garch', p=1, q=1)
result = model.fit()

# 변동성 예측
forecasts = result.forecast(horizon=5)
predicted_vol = forecasts.variance.values[-1, :]
```

### 5.2 비대칭 특성

```python
# Leverage effect
df['leverage'] = (df['returns'] < 0) * df['returns'].abs()

# Downside volatility
df['downside_vol'] = df.groupby(df['returns'] < 0)['returns'].transform('std')

# Jump detection
df['jump'] = (df['returns'].abs() > 3 * df['volatility']).astype(int)
```

### 5.3 앙상블

```python
# 예측 결합
ensemble_pred = (
    0.3 * ridge_pred +
    0.4 * garch_pred +
    0.3 * xgb_pred
)
```

---

## 6. 위험 요소 및 대응

### 위험 1: 과적합
**증상**: 학습 성능↑, 테스트 성능↓
**대응**:
- Purged K-Fold CV 유지
- Regularization 강화
- 앙상블로 분산 감소

### 위험 2: 데이터 누출
**증상**: 비현실적 R² (>0.7)
**대응**:
- 시간적 분리 엄격 검증
- Feature engineering 재검토

### 위험 3: 모델 복잡도
**증상**: 해석 불가, 유지보수 어려움
**대응**:
- 단순 모델 우선 (GARCH)
- Feature importance 분석
- SHAP values 활용

---

## 7. 성공 지표

### 정량 지표
- **R² > 0.40**: 최소 목표
- **R² > 0.45**: 우수
- **R² > 0.50**: 탁월

### 정성 지표
- ✅ Purged CV 통과
- ✅ 데이터 누출 없음
- ✅ 경제적 가치 증명 (백테스트)
- ✅ 해석 가능성 유지

---

## 8. 학술 근거

### 핵심 논문

1. **GARCH 기본**
   - Bollerslev (1986): "Generalized Autoregressive Conditional Heteroskedasticity"

2. **비대칭 효과**
   - Nelson (1991): "Conditional Heteroskedasticity in Asset Returns: A New Approach"

3. **Realized Volatility**
   - Andersen & Bollerslev (1998): "Answering the Skeptics: Yes, Standard Volatility Models do Provide Accurate Forecasts"

4. **HAR-RV**
   - Corsi (2009): "A Simple Approximate Long-Memory Model of Realized Volatility"

5. **앙상블**
   - Hansen et al. (2011): "The Model Confidence Set"

---

## 9. 실무 적용 전략

### 단기 (1개월)
- Phase 1-2 완료
- R² = 0.40 달성
- 백테스트 검증

### 중기 (3개월)
- Phase 3 완료
- R² = 0.45 달성
- 실시간 시스템 구축

### 장기 (6개월)
- Phase 4 완료
- R² = 0.50+ 도전
- 학술 논문 발표

---

## 10. 결론

**현실적 목표**: R² = 0.42 (GARCH 통합)
**도전적 목표**: R² = 0.50 (Full ensemble)

**핵심 전략**:
1. ✅ GARCH 모델 (변동성 클러스터링)
2. ✅ 비대칭 특성 (레버리지 효과)
3. ✅ 앙상블 (강건성)

**주의사항**:
- 과적합 경계
- 데이터 누출 방지
- 해석 가능성 유지

**다음 단계**:
Phase 1부터 순차적 구현 시작
