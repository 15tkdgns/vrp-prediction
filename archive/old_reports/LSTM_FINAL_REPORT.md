# LSTM 수익률 예측 모델 최종 보고서

**날짜**: 2025-10-01
**목표**: R² ≥ 0.3 달성
**결과**: ❌ **목표 미달성** (R² = 0.0041)

---

## 📊 Executive Summary

**목표 달성 여부**: ❌ 실패
**최종 성능**: R² = 0.0041 (목표 0.3 대비 **73배 낮음**)
**데이터 무결성**: ✅ 확인됨
**결론**: **수익률 예측은 효율적 시장 가설(EMH)로 인해 본질적으로 불가능**

---

## 🤖 모델 아키텍처

### Bidirectional LSTM + Attention Mechanism

```
입력: 시계열 시퀀스 (20일 x 54 피처)
    ↓
Bidirectional LSTM (128 units, dropout=0.2)
    ↓
Bidirectional LSTM (64 units, dropout=0.2)
    ↓
Attention Mechanism (가중 평균)
    ↓
Dense(64, relu, dropout=0.3)
    ↓
Dense(32, relu, dropout=0.2)
    ↓
Output(1) - 5일 평균 수익률
```

**하이퍼파라미터**:
- Optimizer: Adam (lr=0.001)
- Loss: MSE
- Batch size: 32
- Max epochs: 100 (early stopping)
- Sequence length: 20일

---

## 📈 성능 결과

### Cross-Validation (Purged K-Fold, 5-fold)

| Fold | R² | MAE | RMSE |
|------|---------|---------|---------|
| Fold 1 | **0.0001** | 0.002533 | 0.003700 |
| Fold 2 | **0.0071** | 0.002639 | 0.003869 |
| Fold 3 | **0.0014** | 0.004219 | 0.006574 |
| Fold 4 | **0.0120** | 0.003955 | 0.005283 |
| Fold 5 | **-0.0002** | 0.002701 | 0.003499 |

**평균 성능**:
- **R² = 0.0041 ± 0.0048**
- MAE = 0.003209
- RMSE = 0.004585

**평가**: ⚠️ 매우 약한 예측력 (실용성 없음)

---

## 🔬 데이터 무결성 검증

### 3대 금기사항 체크

| 검증 항목 | 결과 | 설명 |
|----------|------|------|
| **1. 데이터 누출** | ✅ 없음 | 시간적 분리 완벽 (입력: t-19~t, 타겟: t+1~t+5) |
| **2. 랜덤 데이터** | ✅ 없음 | 실제 SPY 시장 데이터 사용 (2,428 관측치) |
| **3. 하드코딩** | ✅ 없음 | multi_modal_sp500_dataset.csv 사용 |

### 추가 검증

| 검증 항목 | 결과 | 설명 |
|----------|------|------|
| **CV 성능 분산** | ✅ 정상 | std = 0.0048 (적절한 분산) |
| **시퀀스 생성** | ✅ 정상 | LSTM 시퀀스 로직 올바름 |
| **의심스러운 성능** | ✅ 없음 | R² = 0.0041 (낮음 = 정상) |

**종합 판정**: ✅ **데이터 무결성 확인 - 모델이 정직하게 훈련됨**

---

## 📊 모델 비교

| 모델 | 타겟 | R² | 상태 |
|------|------|---------|------|
| **변동성 예측 (Ridge)** | 5일 후 변동성 | **0.3030** | ✅ 우수 |
| **수익률 예측 (Ridge)** | 5일 평균 수익률 | **-0.0632** | ❌ 예측력 없음 |
| **수익률 예측 (LSTM)** | 5일 평균 수익률 | **0.0041** | ⚠️ 매우 약함 |

### 핵심 통찰

1. **LSTM > Ridge**: LSTM (R² = 0.0041)이 Ridge (R² = -0.063)보다 약간 나음
2. **실용성 없음**: 둘 다 실용적 예측력 없음
3. **변동성만 예측 가능**: 변동성 예측 (R² = 0.303)만 유의미
4. **모델 복잡도 무관**: 수익률 예측은 모델 복잡도와 무관하게 불가능

---

## 🎯 목표 미달 원인 분석

### 1. 효율적 시장 가설 (EMH)
- **이론적 한계**: 공개 정보로는 초과 수익 불가
- **랜덤워크**: 수익률은 본질적으로 예측 불가능
- **시장 효율성**: 모든 정보가 가격에 즉시 반영

### 2. 학술 연구 한계
- **2024년 최신 연구**: Out-of-sample R² = -0.02 ~ -0.016 (음수)
- **In-sample 과적합**: 훈련 R² = 0.75-0.81, 테스트 R² = 음수
- **일관된 실패**: 모든 연구에서 실전 예측력 없음

### 3. 실증 데이터
- **모든 CV Fold 실패**: 5개 Fold 모두 R² ≈ 0
- **표준편차 높음**: ±0.0048 (성능 불안정)
- **음수 Fold 존재**: Fold 5 = -0.0002

---

## 💡 실무적 함의

### R² ≥ 0.3 달성 불가능

**이유**:
1. **금융 이론**: EMH로 인한 근본적 한계
2. **실증 증거**: 모든 Fold에서 체계적 실패
3. **학술 한계**: 최신 LSTM/Transformer도 실패
4. **데이터 누출 신호**: R² ≥ 0.3은 보통 데이터 누출

### 권장 사항

#### ✅ 변동성 예측 활용 (권장)
- **현재 성능**: R² = 0.3030 (우수)
- **활용 분야**:
  - 리스크 관리 (VaR, CVaR)
  - VIX 옵션 거래
  - 동적 헤징
  - 포지션 사이징

#### ❌ 수익률 예측 포기 (권장)
- **LSTM 성능**: R² = 0.0041 (실용성 없음)
- **대안**: 변동성 기반 전략
- **정직성**: "수익률 직접 예측 불가" 명시

---

## 🔧 시도한 개선 기법

### 1. 고급 아키텍처
- ✅ Bidirectional LSTM (양방향 학습)
- ✅ Attention Mechanism (중요 시점 가중)
- ✅ Dropout (0.2-0.3, 과적합 방지)

### 2. 고급 피처 (54개)
- ✅ 시계열 피처 (SMA, EMA, 모멘텀)
- ✅ 변동성 피처 (rolling, EWM)
- ✅ 통계 피처 (skew, kurtosis)
- ✅ 래그 피처 (1, 2, 3, 5, 10일)
- ✅ 거래량 피처
- ✅ OHLC 비율

### 3. 엄격한 검증
- ✅ Purged K-Fold CV (금융 표준)
- ✅ 시간적 분리 (purge=5, embargo=5)
- ✅ Early stopping (overfitting 방지)

### 결과: **모든 기법을 사용해도 R² = 0.0041**

---

## 📚 학술적 근거

### Yale University (2023)
> "The out-of-sample R² from a prediction model is an incomplete measure of its economic value, and a market timer can generate significant economic profits even when the predictive R² is negative."

### December 2024 Study
> "While in-sample R² values were relatively high (0.749 to 0.812), the models demonstrated a poor out-of-sample performance, with negative R² values during testing (-0.020 to -0.016)."

**결론**: R² 중심 평가는 부적절. 경제적 가치(Sharpe ratio, 수익률) 중심 평가 필요.

---

## 🎓 최종 결론

### R² ≥ 0.3 달성: **불가능**

**근거**:
1. ✅ 데이터 누출 없음 (검증 완료)
2. ✅ 최신 LSTM + Attention 사용
3. ✅ 54개 고급 피처
4. ✅ Purged K-Fold CV
5. ❌ 결과: R² = 0.0041 (73배 부족)

### 과학적 정직성

**수익률 예측은 EMH로 인해 본질적으로 불가능합니다.**

- **변동성 예측**: R² = 0.3030 ✅ (가능)
- **수익률 예측**: R² = 0.0041 ❌ (불가능)

### 실무 권장

1. **변동성 예측 모델 사용** (R² = 0.3030)
2. **리스크 관리 전략 구축**
3. **"수익률 직접 예측 불가" 명시**
4. **경제적 가치 중심 평가** (R² 대신 Sharpe ratio)

---

## 📁 생성 파일

- `src/models/lstm_return_prediction.py` - LSTM 모델 코드
- `models/lstm_return_prediction.keras` - 훈련된 모델
- `models/lstm_scaler.pkl` - StandardScaler
- `models/lstm_feature_names.pkl` - 피처 이름
- `models/lstm_model_metadata.json` - 메타데이터
- `data/raw/lstm_model_performance.json` - 성능 데이터
- `data/raw/lstm_integrity_report.json` - 무결성 보고서
- `verify_lstm_integrity.py` - 검증 스크립트

---

## 🚨 중요 경고

**R² ≥ 0.3을 달성했다면 데이터 누출을 의심하세요.**

금융 수익률 예측에서 R² ≥ 0.3은:
1. 데이터 누출
2. 미래 정보 사용
3. 하드코딩된 데이터
4. 극히 드문 예외 (Nobel Prize급)

현재 결과 (R² = 0.0041)는 **정직하고 올바른 결과**입니다.

---

**보고서 작성**: Claude Code
**검증 기준**: 3대 금기사항 (데이터 누출, 랜덤 데이터, 하드코딩)
**결론**: 변동성 예측에 집중할 것을 권장
