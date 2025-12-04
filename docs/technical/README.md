# 기술 문서 (Technical Documentation)

**용도**: 개발자 참조, 코드 리뷰, 기술 검증
**최종 업데이트**: 2025-11-04

---

## 📁 파일 목록

### 1. **ARCHITECTURE.md** (20KB) ⭐
**시스템 아키텍처 및 설계 문서**

#### 주요 내용
- 전체 시스템 구조
- 데이터 파이프라인
- 모델 아키텍처
- 검증 프레임워크
- 대시보드 구조
- 기술 스택

#### 다이어그램
```
yfinance → Data Processing → Feature Engineering →
  → Purged K-Fold CV → Ridge Model → Economic Backtest →
  → Dashboard Visualization
```

---

### 2. **VALIDATION_METHODOLOGY.md** (14KB) ⭐
**검증 방법론 상세 문서**

#### 주요 내용
- Purged K-Fold Cross-Validation 상세
- 시간적 분리 (Temporal Separation)
- Purge & Embargo 메커니즘
- HAR 벤치마크 비교
- 데이터 누출 방지 전략
- Walk-Forward Validation

#### 핵심 개념
```python
# Purged K-Fold CV
n_splits = 5
purge_length = 5   # 훈련 세트 끝 5일 제거
embargo_length = 5 # 검증 세트 시작 전 5일 금지
```

**참고 문헌**: López de Prado (2018) *Advances in Financial Machine Learning*

---

### 3. **VARIABLES_DOCUMENTATION.md** (16KB) ⭐
**변수 정의 및 특성 설명**

#### 주요 내용
- 타겟 변수 정의 (target_vol_5d)
- 31개 특성 상세 설명
- 변동성 특성 (volatility_5, _10, _20, _50)
- 래그 특성 (return_lag_1~5, vol_lag_1~5)
- 통계 특성 (mean, skew, kurt)
- 비율 특성 (vol_ratio_*)

#### 변수 분류
1. **변동성 특성** (8개): 과거 변동성 측정
2. **래그 변수** (10개): 시차 효과 포착
3. **통계 특성** (9개): 분포 특성
4. **비율 특성** (4개): 상대적 변동성

---

### 4. **MODEL_PERFORMANCE_REPORT.md** (12KB) ⭐
**모델 성능 분석 리포트**

#### 주요 내용
- 모델 성능 지표 (R², MAE, RMSE)
- Cross-Validation 결과
- HAR 벤치마크 비교
- 경제적 백테스트
- XAI 분석 (SHAP)
- 한계점 및 개선 방안

#### 핵심 성능
| 지표 | 값 |
|------|-----|
| CV R² | 0.303 |
| Test MAE | 0.00332 |
| Test RMSE | 0.00530 |
| HAR 대비 | 1.41배 개선 |

---

## 🎯 문서 활용 가이드

### 시스템 이해 시작
```bash
# 1. 전체 구조 파악
cat docs/technical/ARCHITECTURE.md

# 2. 변수 이해
cat docs/technical/VARIABLES_DOCUMENTATION.md

# 3. 검증 방법 학습
cat docs/technical/VALIDATION_METHODOLOGY.md

# 4. 성능 확인
cat docs/technical/MODEL_PERFORMANCE_REPORT.md
```

### 코드 리뷰 시
1. **ARCHITECTURE.md** - 모듈 구조 확인
2. **VARIABLES_DOCUMENTATION.md** - 변수명 검증
3. **VALIDATION_METHODOLOGY.md** - 검증 로직 확인

### 논문 작성 시
1. **VALIDATION_METHODOLOGY.md** - Method 섹션
2. **MODEL_PERFORMANCE_REPORT.md** - Results 섹션
3. **VARIABLES_DOCUMENTATION.md** - 변수 설명

---

## 🔍 핵심 개념

### 시간적 분리 (Temporal Separation)
```
특성 (Features): ≤ t 시점 데이터
타겟 (Target): ≥ t+1 시점 데이터
간격 (Gap): 완전 분리
```

### Purged K-Fold CV
```
[Train] | [Purge] | [Embargo] [Test] | ...
        5일 제거   5일 금지
```

### 특성 생성 규칙
```python
# 과거만 사용 (≤ t)
volatility_20 = returns.rolling(20).std()  # t-19 ~ t

# 미래 예측 (≥ t+1)
target_vol_5d = returns[t+1:t+6].std()     # t+1 ~ t+5
```

---

## 📊 데이터 플로우

```
1. SPY 데이터 수집 (yfinance)
   ↓
2. 특성 생성 (31개)
   - 변동성 (volatility_*)
   - 래그 (return_lag_*, vol_lag_*)
   - 통계 (mean_*, skew_*, kurt_*)
   - 비율 (vol_ratio_*)
   ↓
3. 타겟 생성
   - target_vol_5d (5일 후 변동성)
   ↓
4. Purged K-Fold CV
   - n_splits=5
   - purge=5, embargo=5
   ↓
5. Ridge 모델 학습
   - alpha=1.0
   - StandardScaler
   ↓
6. 성능 평가
   - R², MAE, RMSE
   - HAR 벤치마크 비교
   ↓
7. 경제적 백테스트
   - 거래비용 0.1%
   - 동적 포지션 조정
```

---

## 🔗 관련 코드

### 메인 모델
```python
# src/models/correct_target_design.py
- get_real_spy_data()           # 데이터 수집
- create_correct_features()     # 특성 생성
- create_correct_targets()      # 타겟 생성
- PurgedKFold                   # 검증 방법
```

### 검증
```python
# src/validation/purged_cross_validation.py
- PurgedKFold.split()           # CV 분할

# src/validation/economic_backtest_validator.py
- EconomicBacktest.run()        # 백테스트
```

### 데이터 처리
```python
# src/core/data_processor.py
- DataProcessor.load_data()     # 데이터 로드
- DataProcessor.preprocess()    # 전처리
```

---

## ⚠️ 주의사항

### 데이터 무결성
- ✅ 특성은 t 시점 이전 데이터만 사용
- ✅ 타겟은 t+1 시점 이후 데이터만 사용
- ✅ Purge & Embargo로 누출 방지

### 하이퍼파라미터
- ⚠️ alpha=1.0 수동 설정 (Grid Search 미적용)
- ⚠️ 향후 Bayesian Optimization 필요

### 한계점
- ⚠️ 수익률 예측 불가능 (R² ≈ 0)
- ⚠️ 일봉 데이터만 사용 (고주파 미적용)
- ⚠️ 단일 자산 (SPY만)

---

## 📚 참고 문헌

1. **López de Prado (2018)** - Purged K-Fold CV
2. **Corsi (2009)** - HAR Model
3. **Hoerl & Kennard (1970)** - Ridge Regression

전체 참고문헌: `paper/PAPER_REFERENCES.bib`

---

**생성일**: 2025-11-04
**상태**: 기술 문서 완료 ✅
