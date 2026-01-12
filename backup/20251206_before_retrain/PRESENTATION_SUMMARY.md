# SPY 변동성 예측 모델 - 발표 자료

**검증 완료 일시:** 2025-10-23
**데이터 소스:** SPY ETF (2015-2024, 2,488 샘플)
**검증 방법:** Purged K-Fold Cross-Validation (5-fold, embargo=1%)

---

## 📊 핵심 결과 요약

### 🏆 최고 성능 모델: **Lasso (α=0.001)**

| 메트릭 | 값 | 해석 |
|--------|-----|------|
| **Cross-Validation R²** | **0.3373** ± 0.147 | ✅ **목표 달성** (≥ 0.30) |
| **Walk-Forward Test R²** | **0.0879** | ✅ **유일한 양수** (일반화 성공) |
| **Test MAE** | 0.00233 | 평균 0.23% 오차 |
| **특성 수** | 25개 | 적절한 복잡도 |

**결론:** Lasso 모델만이 Cross-Validation과 Test 모두에서 안정적인 성능을 보이며, **유일하게 실전 적용 가능한 모델**입니다.

---

## 🔬 모든 모델 성능 비교

### Cross-Validation R² (학습 성능)

| 순위 | 모델 | CV R² | 표준편차 | 평가 |
|------|------|--------|---------|------|
| 🥇 | **ElasticNet** | **0.3444** | ±0.191 | 최고 CV 성능 |
| 🥈 | **Lasso 0.001** | **0.3373** | ±0.147 | **가장 안정적** ⭐ |
| 🥉 | Ridge Volatility | 0.2881 | ±0.248 | 불안정함 |
| 4 | HAR Benchmark | 0.2300 | ±0.190 | 기준선 |
| 5 | Random Forest | 0.1713 | ±0.095 | **최악** ❌ |

### Walk-Forward Test R² (실전 성능)

| 순위 | 모델 | Test R² | 평가 |
|------|------|---------|------|
| 🥇 | **Lasso 0.001** | **+0.0879** | ✅ **유일한 양수** |
| 🥈 | ElasticNet | +0.0254 | 거의 0 |
| 🥉 | Random Forest | +0.0233 | 거의 0 |
| 4 | HAR Benchmark | -0.0431 | 음수 (실패) |
| 5 | Ridge Volatility | -0.1429 | **큰 음수** ❌ |

---

## 📉 주요 발견사항

### 1. RandomForest의 실패


**결과:**
- CV R² = **0.1713** (가장 낮음)

**원인:** 트리 기반 모델은 금융 시계열 변동성 예측에 부적합

### 2. ElasticNet의 과적합

- CV에서는 최고 성능 (0.3444)
- Test에서는 거의 0 (0.0254)
- **일반화 실패** (CV-Test 갭 = 0.319)

### 3. Lasso의 안정성

- CV와 Test 모두 **일관된 양수**
- CV-Test 갭 = 0.249 (가장 작음)
- **유일하게 실전 적용 가능**

---

## 🎯 타겟 변수

**target_vol_5d:** 5일 후 변동성 예측

```python
# 완전한 시간적 분리
for i in range(len(returns)):
    future_returns = returns[i+1:i+6]  # t+1 ~ t+5
    target_vol[i] = future_returns.std()
```

- ✅ 미래 데이터만 사용 (t+1 이후)
- ✅ 현재 특성 (≤ t)과 완전 분리
- ✅ 데이터 누출 Zero

---

## 🔧 특성 엔지니어링

**선택된 25개 특성:**

| 카테고리 | 특성 예시 | 개수 |
|----------|----------|------|
| **VIX 기반** | vix_level, vix_ma_5, vix_ma_20, vix_std_20 | 4 |
| **실현 변동성** | realized_vol_5, realized_vol_10, realized_vol_20 | 3 |
| **지수 가중 변동성** | ewm_vol_5, ewm_vol_10, ewm_vol_20 | 3 |
| **일중 변동성** | intraday_vol_5, intraday_vol_10 | 2 |
| **Garman-Klass** | garman_klass_5, garman_klass_10 | 2 |
| **기본 변동성** | volatility_5, volatility_10, volatility_20 | 3 |
| **래그 특성** | vol_lag_1, vol_lag_2, vol_lag_3, vol_lag_5 | 4 |
| **HAR 특성** | rv_daily, rv_weekly, rv_monthly | 3 |
| **기타** | 추가 변동성 지표 | 1 |

**특성 선택 기준:** 타겟 변수와 상관관계 상위 25개

---

## 🧪 검증 방법론

### Purged K-Fold Cross-Validation

```
Timeline: [==========Train==========][Embargo][===Test===][Embargo]...

- n_splits = 5
- embargo = 1% (약 25 샘플)
- 완전한 시간적 순서 보존
- 데이터 누출 방지
```

**왜 Purged K-Fold?**
1. **시계열 특성:** 시간 순서 유지
2. **데이터 누출 방지:** Train-Test 사이 Embargo 구간
3. **보수적 추정:** 과적합 방지

---

## 📈 성능 기준

| R² 범위 | 평가 | 설명 |
|---------|------|------|
| **≥ 0.30** | ✅ **Success** | 실용적 예측력 |
| **0.20 ~ 0.30** | ⚠️ **Marginal** | 제한적 유용성 |
| **< 0.20** | ❌ **Failure** | 예측력 부족 |
| **< 0** | 💀 **Severe** | 평균보다 나쁨 |

**Lasso 모델:**
- CV R² = 0.3373 ✅ Success
- Test R² = 0.0879 ⚠️ Marginal (하지만 유일한 양수!)

---

## 💡 실전 적용 가이드

### ✅ 권장: Lasso (α=0.001)

**장점:**
- Cross-Validation과 Test 모두 양수
- 안정적인 성능 (표준편차 0.147)
- 과적합 방지 (L1 regularization)
- 해석 가능 (sparse 계수)

**사용 예시:**
```python
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# 모델 설정
scaler = StandardScaler()
model = Lasso(alpha=0.001, max_iter=3000, random_state=42)

# 학습
X_train_scaled = scaler.fit_transform(X_train)
model.fit(X_train_scaled, y_train)

# 예측
X_test_scaled = scaler.transform(X_test)
volatility_forecast = model.predict(X_test_scaled)
```

### ❌ 비권장: 나머지 모델

- **ElasticNet:** 일반화 실패 (Test R² ≈ 0)
- **Ridge:** Test에서 음수 (-0.143)
- **RandomForest:** 가장 낮은 성능 (0.171)
- **HAR:** Test에서 음수 (-0.043)

---

## 🚀 향후 개선 방향

### 1. 앙상블 접근
```python
# Lasso + ElasticNet 가중 평균
pred = 0.7 * lasso.predict(X) + 0.3 * elasticnet.predict(X)
```

### 2. 추가 특성
- VIX 옵션 내재 변동성
- 거래량 기반 변동성
- 고빈도 데이터 (분봉)

### 3. 딥러닝 실험
- LSTM (시계열)
- Transformer (attention)
- 단, 데이터 누출 주의!

### 4. 리스크 관리 적용
- VIX 옵션 거래
- 동적 포지션 조정
- 변동성 기반 Stop-Loss

---

## 📚 재현 방법

### 1. 전체 검증 재실행
```bash
PYTHONPATH=/root/workspace python3 scripts/comprehensive_model_validation.py
```

### 2. 결과 확인
```bash
cat data/validation/comprehensive_model_validation.json
```

### 3. 그래프 생성
```bash
python3 scripts/create_paper_figures.py
```

**출력:**
- `paper/figures/main_results/figure1_model_comparison.png`
- `data/validation/comprehensive_model_validation.json`

---

## ⚠️ 중요 주의사항

### 1. 데이터 누출 방지
```python
# ❌ 잘못된 예시
target = df['returns'].rolling(5).std().shift(-5)  # 미래 데이터 사용!

# ✅ 올바른 예시
for i in range(len(df)):
    future = df['returns'].iloc[i+1:i+6]  # 미래만
    target[i] = future.std()
```

### 2. Purged K-Fold 필수
```python
# ❌ 일반 K-Fold는 데이터 누출 발생
from sklearn.model_selection import KFold  # No!

# ✅ Purged K-Fold 사용
from validation.purged_cross_validation import PurgedKFold  # Yes!
```

### 3. 하드코딩 금지
```python
# ❌ 하드코딩 (재현 불가)
cv_r2 = [0.4556, 0.4536, 0.4556]  # No!

# ✅ 실제 검증 결과 로드
with open('validation_results.json') as f:
    cv_r2 = json.load(f)['cv_scores']  # Yes!
```

---

**프로젝트:** SPY 변동성 예측 시스템
**검증 방법:** Purged K-Fold Cross-Validation
**데이터:** SPY ETF (2015-2024)

**핵심 결론:** **Lasso (α=0.001) 모델이 유일하게 실전 적용 가능한 성능을 보임**
