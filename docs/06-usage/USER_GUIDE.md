# 📖 사용 가이드

> 시스템 설치, 실행, API 사용법

---

## 1. 설치

### 1.1 요구사항

| 항목 | 버전 | 비고 |
|------|------|------|
| Python | 3.9+ | 권장: 3.10 |
| pip | 21+ | 최신 버전 권장 |
| 메모리 | 8GB+ | 대시보드 실행 시 |
| 저장공간 | 500MB+ | 데이터 + 모델 |

### 1.2 환경 설정

```bash
# 1. 리포지토리 클론
git clone https://github.com/15tkdgns/ai-stock-prediction.git
cd ai-stock-prediction

# 2. 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. 의존성 설치
pip install -r requirements/base.txt
```

### 1.3 주요 의존성

```text
# requirements/base.txt
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
yfinance>=0.2.0
matplotlib>=3.7.0
plotly>=5.0.0
streamlit>=1.28.0
shap>=0.42.0
joblib>=1.3.0
```

---

## 2. 빠른 시작

### 2.1 시스템 상태 확인

```bash
# PYTHONPATH 설정 후 실행
PYTHONPATH=/path/to/workspace python3 -c "
from src.utils.system_orchestrator import SystemOrchestrator
orchestrator = SystemOrchestrator()
result = orchestrator.initialize_components()
print(f'System Status: {\"Ready\" if result else \"Error\"}')"
```

**예상 출력:**
```
System Status: Ready
```

### 2.2 대시보드 실행 (가장 쉬운 방법)

```bash
# Streamlit 대시보드 실행
streamlit run app.py

# 브라우저에서 열기
# http://localhost:8501
```

### 2.3 모델 학습 (선택사항)

모델은 이미 학습되어 저장되어 있습니다. 재학습이 필요한 경우:

```bash
# 약 10분 소요
PYTHONPATH=/path/to/workspace python3 src/models/train_final_reproducible_model.py
```

---

## 3. 주요 기능

### 3.1 Streamlit 대시보드

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Streamlit 대시보드 (6탭)                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  [탭 1] 변동성 예측                                                     │
│    • 실제 vs 예측 변동성 시계열 차트                                    │
│    • 날짜 범위 선택 가능                                                │
│                                                                          │
│  [탭 2] 특성 영향                                                       │
│    • SHAP 기반 특성 중요도                                              │
│    • 개별 특성 효과 분석                                                │
│                                                                          │
│  [탭 3] 경제적 가치                                                     │
│    • 백테스트 수익 곡선                                                 │
│    • 성과 지표 비교표                                                   │
│                                                                          │
│  [탭 4] 모델 비교                                                       │
│    • ElasticNet vs Ridge vs RF                                          │
│    • 성능 지표 레이더 차트                                              │
│                                                                          │
│  [탭 5] 통계적 검증                                                     │
│    • 잔차 분석                                                          │
│    • Q-Q Plot, ACF                                                      │
│                                                                          │
│  [탭 6] 특성 분석                                                       │
│    • 상관관계 히트맵                                                    │
│    • 분포 히스토그램                                                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 명령줄 실행

#### 모델 학습

```bash
# 전체 학습 파이프라인
PYTHONPATH=/path/to/workspace python3 src/models/train_final_reproducible_model.py

# 출력 예시:
# Loading data...
# Preprocessing features...
# Training ElasticNet with GridSearchCV...
# Best params: alpha=0.0005, l1_ratio=0.3
# Test R²: 0.2218
# Model saved to data/models/final_elasticnet.pkl
```

#### 백테스트 실행

```bash
PYTHONPATH=/path/to/workspace python3 src/validation/economic_backtest_validator.py

# 출력 예시:
# Running economic backtest...
# Strategy Return: 14.10%
# Strategy Volatility: 12.24%
# Sharpe Ratio: 0.989
# Results saved to data/raw/rv_economic_backtest_results.json
```

#### 검증 실행

```bash
# Purged K-Fold CV
PYTHONPATH=/path/to/workspace python3 src/validation/purged_cross_validation.py

# Walk-Forward Validation
PYTHONPATH=/path/to/workspace python3 src/validation/walk_forward_validation.py
```

---

## 4. API 사용법

### 4.1 모델 로드 및 예측

```python
import joblib
import pandas as pd

# 모델 및 스케일러 로드
model = joblib.load('data/models/final_elasticnet.pkl')
scaler = joblib.load('data/models/final_scaler.pkl')

# 새 데이터 준비 (31개 특성)
new_data = pd.DataFrame({
    'volatility_5': [0.012],
    'volatility_10': [0.015],
    'volatility_20': [0.018],
    'volatility_50': [0.020],
    # ... 나머지 특성들
})

# 스케일링
new_scaled = scaler.transform(new_data)

# 예측
prediction = model.predict(new_scaled)
print(f"5일 후 예측 변동성: {prediction[0]:.4f}")
```

### 4.2 데이터 수집

```python
import yfinance as yf

def get_spy_data(start_date, end_date):
    """SPY 데이터 수집"""
    spy = yf.Ticker("SPY")
    data = spy.history(start=start_date, end=end_date)
    return data

# 사용 예시
data = get_spy_data("2024-01-01", "2024-12-31")
print(data.head())
```

### 4.3 특성 생성

```python
import numpy as np
import pandas as pd

def create_features(df):
    """변동성 예측용 특성 생성"""
    
    # 수익률
    df['return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # 변동성 (여러 기간)
    for window in [5, 10, 20, 50]:
        df[f'volatility_{window}'] = df['return'].rolling(window).std()
    
    # 래그 변수
    for lag in [1, 2, 3, 5]:
        df[f'vol_lag_{lag}'] = df['volatility_20'].shift(lag)
    
    # 변동성 비율
    df['vol_ratio_5_20'] = df['volatility_5'] / df['volatility_20']
    
    # 기타 통계
    df['mean_return_20'] = df['return'].rolling(20).mean()
    df['skew_20'] = df['return'].rolling(20).skew()
    
    return df.dropna()

# 사용 예시
data = get_spy_data("2024-01-01", "2024-12-31")
features = create_features(data)
```

### 4.4 시스템 오케스트레이터 사용

```python
from src.utils.system_orchestrator import SystemOrchestrator

# 오케스트레이터 초기화
orchestrator = SystemOrchestrator()

# 컴포넌트 초기화
if orchestrator.initialize_components():
    print("System ready")
else:
    print("Initialization failed")

# 예측 실행
result = orchestrator.run_prediction()
print(f"Prediction: {result}")
```

---

## 5. 파일 구조

### 5.1 주요 실행 파일

| 파일 | 용도 | 실행 방법 |
|------|------|-----------|
| `app.py` | Streamlit 대시보드 | `streamlit run app.py` |
| `src/models/train_final_reproducible_model.py` | 모델 학습 | `python src/models/...` |
| `src/validation/economic_backtest_validator.py` | 경제적 백테스트 | `python src/validation/...` |
| `src/validation/purged_cross_validation.py` | Purged K-Fold CV | `python src/validation/...` |

### 5.2 데이터 파일

| 파일 | 위치 | 설명 |
|------|------|------|
| `final_elasticnet.pkl` | data/models/ | 학습된 모델 |
| `final_scaler.pkl` | data/models/ | 표준화 스케일러 |
| `test_predictions.csv` | data/raw/ | 테스트 예측 결과 |
| `spy_data_2020_2025.csv` | data/raw/ | SPY 원본 데이터 |

---

## 6. 문제 해결

### 6.1 일반적인 오류

#### ModuleNotFoundError

```bash
# 해결: PYTHONPATH 설정
export PYTHONPATH=/path/to/workspace

# Windows
set PYTHONPATH=C:\path\to\workspace
```

#### yfinance 데이터 수집 실패

```bash
# 해결: yfinance 업그레이드
pip install --upgrade yfinance

# 또는 캐시된 데이터 사용
# data/raw/spy_data_2020_2025.csv
```

#### Streamlit 포트 충돌

```bash
# 다른 포트로 실행
streamlit run app.py --server.port 8502
```

### 6.2 성능 최적화

```python
# 메모리 사용량 줄이기
import pandas as pd
df = pd.read_csv('data.csv', dtype={
    'Close': 'float32',  # float64 대신
    'Volume': 'int32',   # int64 대신
})
```

---

## 7. 자주 묻는 질문

### Q1: 새로운 데이터로 예측하려면?

```python
# 1. 데이터 수집
new_data = yf.download("SPY", start="2024-12-01", end="2024-12-31")

# 2. 특성 생성
features = create_features(new_data)

# 3. 예측
prediction = model.predict(scaler.transform(features))
```

### Q2: 다른 ETF에도 사용할 수 있나요?

현재 모델은 SPY에 최적화되어 있습니다. 다른 ETF에 적용하려면 재학습이 필요합니다.

```python
# 다른 ETF 학습 예시
qqq_data = yf.download("QQQ", start="2015-01-01", end="2024-12-31")
# ... 동일한 파이프라인으로 학습
```

### Q3: 실시간 예측이 가능한가요?

현재는 일간 데이터 기반입니다. 실시간 예측은 추가 개발이 필요합니다.

---

## 8. 참고 자료

- **프로젝트 개요**: `docs/01-overview/PROJECT_OVERVIEW.md`
- **시스템 아키텍처**: `docs/02-architecture/SYSTEM_ARCHITECTURE.md`
- **모델 설명**: `docs/03-models/MODEL_SPECIFICATION.md`
- **검증 방법론**: `docs/04-validation/VALIDATION_METHODOLOGY.md`
- **결과 분석**: `docs/05-results/RESULTS_ANALYSIS.md`

---

**문서 작성일**: 2025-12-04
**버전**: 1.0
