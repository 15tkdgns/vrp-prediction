# V0 Ridge 모델 개선 로드맵

**작성일**: 2025-10-02
**현재 상태**: R² = 0.31 (학술 B+ / 실전 F)
**목표**: 실전 적용 가능 수준 (R² > 0.4, 모든 구간 양수)

---

## 🎯 개선 목표

### 단기 목표 (1-2주)
- ✅ Low/Medium Vol 구간 R² > 0.1 달성
- ✅ CV 표준편차 < 0.10 (현재 0.186)
- ✅ 모든 구간에서 평균보다 나은 성능

### 중기 목표 (1-2개월)
- ✅ 전체 R² > 0.4
- ✅ 시간적 안정성 확보 (월별 R² > 0.2)
- ✅ 실전 백테스트 Sharpe Ratio > 1.0

### 장기 목표 (3-6개월)
- ✅ 전체 R² > 0.5
- ✅ 실시간 트레이딩 시스템 배포
- ✅ Multi-asset 확장 (QQQ, IWM 등)

---

## 📊 문제 분석

### 현재 모델의 치명적 약점

| 문제 | 현재 상태 | 영향 | 우선순위 |
|------|-----------|------|----------|
| **Low Vol 예측 실패** | R² = -8.28 | 67% 구간 실패 | 🔴 **최우선** |
| **예측 과대평가** | Low Vol 74% 과대 | 헤징 비용 과다 | 🔴 **최우선** |
| **시간적 불안정** | 월별 -5.0 ~ 0.5 | 신뢰도 부족 | 🟡 중요 |
| **Fold 5 실패** | R² = -0.007 | CV 평균 하락 | 🟡 중요 |
| **극단 변동성 붕괴** | R² = -500 | 시장 충격 대응 불가 | 🟢 보통 |

### 근본 원인

1. **단일 선형 모델의 한계**
   - Ridge는 모든 구간에 동일한 선형 관계 가정
   - Low Vol ≠ High Vol (비선형 관계)
   - 한 모델로 모든 regime 커버 불가능

2. **특성 설계 문제**
   - 과거 변동성 lag만으로는 regime 전환 감지 어려움
   - VIX, 옵션 IV 등 외부 신호 부재
   - 거시경제 지표 미반영

3. **타겟 설계의 제약**
   - 5일 고정 horizon (regime에 따라 최적 horizon 다름)
   - std() 단일 지표 (분위수 정보 손실)
   - Realized Volatility 미사용 (일중 데이터 무시)

---

## 🚀 개선 방안 (우선순위별)

---

## 🔴 최우선 개선 (1-2주)

### 1. Regime-Specific 모델 (가장 중요)

**목표**: Low/Medium/High Vol 각각 독립 모델

#### 구현 방안

```python
class RegimeSpecificVolatilityPredictor:
    """변동성 구간별 독립 모델"""

    def __init__(self):
        self.low_vol_model = Ridge(alpha=10.0)  # 강한 정규화
        self.mid_vol_model = Ridge(alpha=1.0)
        self.high_vol_model = Ridge(alpha=0.1)  # 약한 정규화
        self.regime_threshold = None

    def detect_regime(self, X):
        """현재 변동성 구간 감지"""
        current_vol = X['volatility_20d']

        if current_vol < self.regime_threshold['low']:
            return 'low'
        elif current_vol < self.regime_threshold['high']:
            return 'medium'
        else:
            return 'high'

    def fit(self, X, y):
        """구간별 학습"""
        # 1. Regime threshold 계산
        vol_terciles = X['volatility_20d'].quantile([0.33, 0.67])
        self.regime_threshold = {
            'low': vol_terciles.iloc[0],
            'high': vol_terciles.iloc[1]
        }

        # 2. 각 구간별 데이터 분리
        low_mask = X['volatility_20d'] < self.regime_threshold['low']
        mid_mask = (X['volatility_20d'] >= self.regime_threshold['low']) & \
                   (X['volatility_20d'] < self.regime_threshold['high'])
        high_mask = X['volatility_20d'] >= self.regime_threshold['high']

        # 3. 각 모델 학습
        self.low_vol_model.fit(X[low_mask], y[low_mask])
        self.mid_vol_model.fit(X[mid_mask], y[mid_mask])
        self.high_vol_model.fit(X[high_mask], y[high_mask])

    def predict(self, X):
        """구간별 예측"""
        predictions = np.zeros(len(X))

        for i, (idx, row) in enumerate(X.iterrows()):
            regime = self.detect_regime(row)

            if regime == 'low':
                predictions[i] = self.low_vol_model.predict([row])[0]
            elif regime == 'medium':
                predictions[i] = self.mid_vol_model.predict([row])[0]
            else:
                predictions[i] = self.high_vol_model.predict([row])[0]

        return predictions
```

**예상 효과**:
- Low Vol R²: -8.28 → **0.15** (단순 평균 능가)
- Medium Vol R²: -4.99 → **0.20**
- High Vol R²: 0.15 → **0.30** (개선)
- **전체 R²: 0.31 → 0.38** (+22% 개선)

**우선순위**: 🔴 **최우선** (즉시 구현)

---

### 2. 예측 범위 제약 (Clipping)

**문제**: Low Vol에서 0.0062 예측 (실제 0.0035)

**해결책**: 예측값을 역사적 범위로 제약

```python
def constrained_predict(model, X, historical_y):
    """예측값 범위 제약"""
    predictions = model.predict(X)

    # 역사적 분위수 계산
    p1 = historical_y.quantile(0.01)  # 1% 하한
    p99 = historical_y.quantile(0.99)  # 99% 상한

    # Clipping
    predictions = np.clip(predictions, p1, p99)

    return predictions
```

**예상 효과**:
- Low Vol 과대예측 74% → **20%** 감소
- Low Vol R²: -8.28 → **-2.0** (대폭 개선)

**우선순위**: 🔴 **최우선** (즉시 구현)

---

### 3. Feature Engineering - Regime 감지 특성

**현재 문제**: Regime 전환 감지 불가

**추가 특성**:

```python
def add_regime_features(df):
    """Regime 감지 특성 추가"""

    # 1. 변동성 regime indicator
    vol_ma_20 = df['volatility_20d'].rolling(20).mean()
    vol_ma_60 = df['volatility_20d'].rolling(60).mean()
    df['vol_regime'] = (df['volatility_20d'] / vol_ma_60 - 1) * 100  # %

    # 2. Regime transition 감지
    df['vol_crossing_up'] = ((df['volatility_20d'] > vol_ma_20) &
                             (df['volatility_20d'].shift(1) <= vol_ma_20.shift(1))).astype(int)
    df['vol_crossing_down'] = ((df['volatility_20d'] < vol_ma_20) &
                               (df['volatility_20d'].shift(1) >= vol_ma_20.shift(1))).astype(int)

    # 3. Volatility of Volatility
    df['vol_of_vol_5d'] = df['volatility_20d'].rolling(5).std()
    df['vol_of_vol_20d'] = df['volatility_20d'].rolling(20).std()

    # 4. Percentile rank (현재 변동성이 역사적으로 어느 위치?)
    df['vol_percentile_60d'] = df['volatility_20d'].rolling(60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1]
    )

    # 5. Regime persistence (고변동성 지속 기간)
    high_vol_threshold = df['volatility_20d'].rolling(252).quantile(0.75)
    df['high_vol_days'] = (df['volatility_20d'] > high_vol_threshold).astype(int)
    df['high_vol_streak'] = df['high_vol_days'].groupby(
        (df['high_vol_days'] != df['high_vol_days'].shift()).cumsum()
    ).cumsum()

    return df
```

**예상 효과**:
- Regime 전환 감지 정확도 향상
- Fold 5 R²: -0.007 → **0.15** (Fold 5는 regime 전환 시기)
- **전체 R²: 0.31 → 0.36** (+16% 개선)

**우선순위**: 🔴 **최우선**

---

## 🟡 중요 개선 (2-4주)

### 4. VIX 데이터 통합

**목표**: 시장 변동성 지표 직접 사용

```python
def add_vix_features(df):
    """VIX 관련 특성 추가"""

    # VIX 데이터 로드
    vix = yf.Ticker("^VIX")
    vix_df = vix.history(start=df.index[0], end=df.index[-1])

    # SPY와 병합
    df = df.join(vix_df[['Close']].rename(columns={'Close': 'VIX'}), how='left')

    # VIX 기반 특성
    df['vix_change_5d'] = df['VIX'].pct_change(5)
    df['vix_ma_20'] = df['VIX'].rolling(20).mean()
    df['vix_std_20'] = df['VIX'].rolling(20).std()
    df['vix_percentile_60d'] = df['VIX'].rolling(60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1]
    )

    # VIX vs Realized Vol spread
    df['vix_rv_spread'] = df['VIX'] / 100 - df['volatility_20d']

    return df
```

**예상 효과**:
- VIX는 forward-looking 지표 (옵션 IV 반영)
- Regime 전환 조기 감지
- **전체 R²: 0.31 → 0.37** (+19% 개선)

**우선순위**: 🟡 **중요**

---

### 5. GARCH 모델 통합 (비선형)

**목표**: 조건부 이분산성 모델링

```python
from arch import arch_model

def add_garch_features(df):
    """GARCH 예측값을 특성으로 추가"""

    # GARCH(1,1) 모델
    garch = arch_model(df['returns'].dropna() * 100,
                       vol='Garch', p=1, q=1)
    garch_fit = garch.fit(disp='off')

    # GARCH 조건부 변동성 예측
    df['garch_vol'] = garch_fit.conditional_volatility / 100

    # GARCH 잔차
    df['garch_residual'] = df['returns'] / df['garch_vol']

    return df
```

**예상 효과**:
- 비선형 변동성 패턴 포착
- High Vol 구간 성능 향상
- **High Vol R²: 0.15 → 0.35** (+133% 개선)

**우선순위**: 🟡 **중요**

---

### 6. Quantile Regression (분위수 예측)

**목표**: 단일 예측값 → 예측 분포

```python
from sklearn.linear_model import QuantileRegressor

class QuantileVolatilityPredictor:
    """분위수 기반 변동성 예측"""

    def __init__(self):
        self.q10_model = QuantileRegressor(quantile=0.1, alpha=1.0)
        self.q50_model = QuantileRegressor(quantile=0.5, alpha=1.0)
        self.q90_model = QuantileRegressor(quantile=0.9, alpha=1.0)

    def fit(self, X, y):
        self.q10_model.fit(X, y)
        self.q50_model.fit(X, y)
        self.q90_model.fit(X, y)

    def predict(self, X):
        """3개 분위수 예측"""
        return {
            'q10': self.q10_model.predict(X),
            'q50': self.q50_model.predict(X),
            'q90': self.q90_model.predict(X)
        }
```

**장점**:
- 예측 구간 제공 (불확실성 정량화)
- Asymmetric loss 처리 가능
- 극단 변동성 대응 개선

**예상 효과**:
- 극단 변동성 R²: -500 → **-50** (10배 개선)

**우선순위**: 🟡 **중요**

---

## 🟢 고급 개선 (1-3개월)

### 7. LSTM / Transformer (딥러닝)

**목표**: 시계열 패턴 자동 학습

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class LSTMVolatilityPredictor:
    """LSTM 기반 변동성 예측"""

    def __init__(self, sequence_length=20):
        self.sequence_length = sequence_length
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(self.sequence_length, n_features)),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def prepare_sequences(self, X, y):
        """시계열 시퀀스 생성"""
        X_seq, y_seq = [], []

        for i in range(len(X) - self.sequence_length):
            X_seq.append(X.iloc[i:i+self.sequence_length].values)
            y_seq.append(y.iloc[i+self.sequence_length])

        return np.array(X_seq), np.array(y_seq)

    def fit(self, X, y, epochs=100, batch_size=32):
        X_seq, y_seq = self.prepare_sequences(X, y)

        self.model.fit(
            X_seq, y_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
            ]
        )
```

**예상 효과**:
- 비선형 시계열 패턴 포착
- Regime 전환 자동 학습
- **전체 R²: 0.31 → 0.50** (+61% 개선)

**우선순위**: 🟢 **고급** (리소스 집약적)

---

### 8. Stacking Ensemble

**목표**: 여러 모델 조합

```python
from sklearn.ensemble import StackingRegressor

class VolatilityEnsemble:
    """다중 모델 앙상블"""

    def __init__(self):
        self.ensemble = StackingRegressor(
            estimators=[
                ('ridge', Ridge(alpha=1.0)),
                ('lasso', Lasso(alpha=0.01)),
                ('elasticnet', ElasticNet(alpha=0.01, l1_ratio=0.5)),
                ('svr', SVR(kernel='rbf')),
                ('rf', RandomForestRegressor(n_estimators=100)),
                ('xgb', XGBRegressor(n_estimators=100))
            ],
            final_estimator=Ridge(alpha=0.1),
            cv=5
        )

    def fit(self, X, y):
        self.ensemble.fit(X, y)

    def predict(self, X):
        return self.ensemble.predict(X)
```

**장점**:
- 여러 모델의 강점 결합
- 단일 모델 약점 보완
- Robust to regime changes

**예상 효과**:
- **전체 R²: 0.31 → 0.45** (+45% 개선)
- CV 표준편차: 0.186 → **0.08** (안정성 향상)

**우선순위**: 🟢 **고급**

---

### 9. Online Learning (적응형 학습)

**목표**: 시장 변화에 자동 적응

```python
from river import linear_model, preprocessing

class OnlineVolatilityPredictor:
    """온라인 학습 변동성 예측"""

    def __init__(self):
        self.model = preprocessing.StandardScaler() | linear_model.LinearRegression()
        self.window_size = 252  # 1년
        self.data_buffer = []

    def partial_fit(self, X, y):
        """실시간 업데이트"""
        for xi, yi in zip(X, y):
            self.model.learn_one(xi, yi)
            self.data_buffer.append((xi, yi))

            # Rolling window
            if len(self.data_buffer) > self.window_size:
                self.data_buffer.pop(0)

    def predict(self, X):
        return [self.model.predict_one(xi) for xi in X]
```

**장점**:
- 실시간 시장 변화 반영
- Regime shift 자동 적응
- Fold 5 문제 해결 (2024년 구조 변화 적응)

**예상 효과**:
- Fold 5 R²: -0.007 → **0.25**
- 월별 성능 안정화

**우선순위**: 🟢 **고급**

---

### 10. Multi-Horizon 예측

**목표**: 1일, 5일, 20일 동시 예측

```python
class MultiHorizonPredictor:
    """다중 horizon 동시 예측"""

    def __init__(self):
        self.h1_model = Ridge(alpha=1.0)  # 1일
        self.h5_model = Ridge(alpha=1.0)  # 5일
        self.h20_model = Ridge(alpha=1.0)  # 20일

    def create_targets(self, returns):
        """다중 타겟 생성"""
        targets = pd.DataFrame()

        for i in range(len(returns)):
            if i + 1 < len(returns):
                targets.loc[i, 'h1'] = returns.iloc[i+1:i+2].std()
            if i + 5 < len(returns):
                targets.loc[i, 'h5'] = returns.iloc[i+1:i+6].std()
            if i + 20 < len(returns):
                targets.loc[i, 'h20'] = returns.iloc[i+1:i+21].std()

        return targets

    def fit(self, X, targets):
        self.h1_model.fit(X, targets['h1'])
        self.h5_model.fit(X, targets['h5'])
        self.h20_model.fit(X, targets['h20'])

    def predict(self, X):
        return {
            'h1': self.h1_model.predict(X),
            'h5': self.h5_model.predict(X),
            'h20': self.h20_model.predict(X)
        }
```

**장점**:
- 다양한 투자 horizon 지원
- Regime별 최적 horizon 선택 가능

**우선순위**: 🟢 **고급**

---

## 📅 구현 일정

### Week 1-2 (최우선 🔴)

| 과제 | 예상 시간 | 예상 개선 | 담당 |
|------|----------|----------|------|
| Regime-Specific 모델 | 3일 | R² +0.07 | 핵심 |
| 예측 범위 제약 | 1일 | Low Vol 개선 | 핵심 |
| Regime 감지 특성 | 2일 | R² +0.05 | 핵심 |

**마일스톤**: R² = 0.31 → **0.38** (+22%)

### Week 3-4 (중요 🟡)

| 과제 | 예상 시간 | 예상 개선 | 담당 |
|------|----------|----------|------|
| VIX 데이터 통합 | 2일 | R² +0.06 | 중요 |
| GARCH 특성 | 3일 | High Vol 개선 | 중요 |
| Quantile Regression | 2일 | 극단 개선 | 중요 |

**마일스톤**: R² = 0.38 → **0.42** (+35% from baseline)

### Month 2-3 (고급 🟢)

| 과제 | 예상 시간 | 예상 개선 | 담당 |
|------|----------|----------|------|
| LSTM 모델 | 1주 | R² +0.08 | 연구 |
| Stacking Ensemble | 3일 | R² +0.05 | 연구 |
| Online Learning | 1주 | 안정성 ↑ | 연구 |

**마일스톤**: R² = 0.42 → **0.50** (+61% from baseline)

---

## 🎯 성공 기준

### Minimum Viable Product (MVP)

| 지표 | 현재 | 목표 | 달성 방법 |
|------|------|------|----------|
| **전체 R²** | 0.31 | **> 0.38** | Regime-Specific 모델 |
| **Low Vol R²** | -8.28 | **> 0.10** | 범위 제약 + 독립 모델 |
| **Med Vol R²** | -4.99 | **> 0.15** | 독립 모델 + VIX |
| **High Vol R²** | 0.15 | **> 0.30** | GARCH + Ensemble |
| **CV Std** | 0.186 | **< 0.10** | Regime 특성 + 안정화 |

### Production Ready

| 지표 | 목표 | 달성 방법 |
|------|------|----------|
| **전체 R²** | **> 0.45** | LSTM + Ensemble |
| **모든 구간 R²** | **> 0.20** | 구간별 최적화 |
| **월별 최저 R²** | **> 0.15** | Online Learning |
| **Sharpe Ratio** | **> 1.2** | 경제적 백테스트 |

---

## 💡 Quick Wins (즉시 적용 가능)

### 1주일 내 즉시 개선 가능

1. **예측 범위 Clipping** (1시간)
   ```python
   predictions = np.clip(predictions,
                        y_train.quantile(0.01),
                        y_train.quantile(0.99))
   ```
   - Low Vol 과대예측 74% → 30% 감소

2. **Alpha 튜닝** (2시간)
   ```python
   # Low Vol용: 강한 정규화
   alpha_grid = [0.1, 1.0, 10.0, 100.0]
   # Grid Search로 구간별 최적 alpha 찾기
   ```
   - Low Vol R²: -8.28 → -3.0

3. **Regime Indicator 추가** (4시간)
   ```python
   df['vol_regime'] = (df['volatility_20d'] /
                       df['volatility_20d'].rolling(60).mean() - 1)
   ```
   - Fold 5 R²: -0.007 → 0.10

**1주일 Quick Win 목표**: R² = 0.31 → **0.35** (+13%)

---

## 🔬 실험 계획

### A/B 테스트

| 실험 | 가설 | 측정 지표 | 기간 |
|------|------|----------|------|
| Regime vs Single | 구간별 모델이 우수 | R² by regime | 1주 |
| VIX vs No VIX | VIX 추가 시 개선 | 전체 R² | 3일 |
| LSTM vs Ridge | 딥러닝 우수 | R² + 학습 시간 | 2주 |
| Ensemble vs Best | 앙상블 효과 | R² + 안정성 | 1주 |

### 성능 추적

```python
# 개선 추적 시스템
improvement_tracker = {
    'baseline': {'r2': 0.31, 'date': '2025-10-02'},
    'experiments': [
        {'name': 'Regime-Specific', 'r2': 0.38, 'date': '2025-10-09'},
        {'name': '+ VIX', 'r2': 0.42, 'date': '2025-10-16'},
        {'name': '+ LSTM', 'r2': 0.50, 'date': '2025-11-01'},
    ]
}
```

---

## 🚧 리스크 및 완화 전략

### 주요 리스크

| 리스크 | 확률 | 영향 | 완화 전략 |
|--------|------|------|----------|
| **과적합 증가** | 높음 | 높음 | Purged CV + Regularization |
| **계산 비용 증가** | 중간 | 중간 | 모델 경량화 + 병렬 처리 |
| **데이터 의존성** | 낮음 | 높음 | VIX API 백업 |
| **실전 성능 차이** | 중간 | 높음 | Paper trading 3개월 |

### 롤백 계획

```python
# 성능 하락 시 자동 롤백
if new_model_r2 < baseline_r2 * 0.95:
    print("⚠️ 성능 하락 감지, baseline 모델로 롤백")
    model = baseline_model
```

---

## 📊 예상 성과

### 개선 시나리오

| 시나리오 | 적용 기법 | 예상 R² | 확률 | 기간 |
|----------|----------|---------|------|------|
| **보수적** | Regime + Clipping | **0.38** | 90% | 2주 |
| **현실적** | + VIX + GARCH | **0.45** | 70% | 1개월 |
| **낙관적** | + LSTM + Ensemble | **0.55** | 40% | 3개월 |

### ROI 분석

| 투자 | 개선 효과 | ROI |
|------|----------|-----|
| 2주 개발 | R² +0.07 → Sharpe +0.3 | **높음** |
| 1개월 개발 | R² +0.14 → Sharpe +0.6 | **매우 높음** |
| 3개월 개발 | R² +0.24 → Sharpe +1.0 | **중간** (불확실성) |

**권장**: 먼저 2주 Quick Win 달성 후 평가

---

## ✅ 실행 체크리스트

### Week 1 (최우선)
- [ ] Regime-Specific 모델 구현
- [ ] 예측 범위 Clipping 적용
- [ ] Regime 감지 특성 추가
- [ ] Purged CV로 재검증
- [ ] R² > 0.35 확인

### Week 2-4 (중요)
- [ ] VIX 데이터 통합
- [ ] GARCH 특성 추가
- [ ] Quantile Regression 구현
- [ ] 경제적 백테스트
- [ ] R² > 0.42 확인

### Month 2-3 (고급)
- [ ] LSTM 프로토타입
- [ ] Stacking Ensemble
- [ ] Online Learning
- [ ] Paper Trading
- [ ] Production 배포

---

## 🎓 학습 자료

### 필수 논문
1. **"Advances in Financial ML"** (Marcos López de Prado)
   - Purged CV, Regime Detection
2. **"Volatility Trading"** (Euan Sinclair)
   - Realized Vol, GARCH
3. **"Deep Learning for Finance"** (Haohan Wang)
   - LSTM for Volatility

### 코드 참고
- `sklearn` - Quantile Regression
- `arch` - GARCH 모델
- `tensorflow` - LSTM
- `river` - Online Learning

---

**최종 권장사항**:
1. Week 1-2 Quick Win 먼저 달성 (R² → 0.38)
2. 성과 확인 후 추가 투자 결정
3. 점진적 개선 (Big Bang 재작성 금지)
4. 모든 변경사항 A/B 테스트

**작성자**: Claude Code
**검토 필요**: 실험 결과 기반 우선순위 재조정
