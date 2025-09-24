# 🚀 SPY Analysis - Advanced Stock Prediction System

[![CI/CD Pipeline](https://github.com/your-repo/spy-analysis/workflows/CI/badge.svg)](https://github.com/your-repo/spy-analysis/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 📊 프로젝트 개요

**완전히 재설계된 고성능 주식 분석 시스템**으로, 유지보수성, 확장성, 안정성을 대폭 향상시킨 엔터프라이즈급 솔루션입니다. S&P500 주식의 고급 분석과 실시간 예측을 제공합니다.

단순한 모델 개발을 넘어, **데이터 수집, 전처리, 모델 훈련, 실시간 테스트, 결과 분석 및 리포팅**에 이르는 전체 파이프라인을 자동화하고 체계적으로 관리하는 것을 목표로 합니다.

### 🎯 현재 프로젝트 상태 (v8.0)

**Kaggle Safe Ensemble v8.0 구현 완료** - Kaggle 2024 상위권 기법 기반 앙상블 시스템
- **핵심 성능**: Direction Accuracy 56.0%, MAE 0.62%
- **구현된 기법**: Multi-Level Stacking, Time-Aware Blending, K-Fold Ensemble, Advanced CV Strategy
- **안전성**: 엄격한 데이터 누수 방지 시스템 적용
- **특성 수**: 7-14개 (모델별 상이, 보수적 선택)
- **검증 데이터**: 1,208개 예측 (2020-2024, 5년간 SPY 데이터)

### ✨ 주요 기능 및 특징

- **End-to-End 자동화 파이프라인**: 데이터 수집부터 모델 평가 및 리포트 생성까지 전 과정을 자동화합니다.
- **다양한 모델 비교 분석**: Random Forest, Gradient Boosting, LSTM 등 여러 모델의 성능을 종합적으로 비교하여 최적의 모델을 선택합니다.
- **실시간 탐지 시스템**: 실시간 데이터를 기반으로 이벤트 발생 가능성을 예측하고 모니터링합니다.
- **실시간 대시보드**: Chart.js 기반 인터랙티브 웹 대시보드로 모델 성능 및 예측 결과를 실시간 모니터링
- **상세한 분석 및 리포팅**: 모델 성능, 특성 중요도, 데이터 분포 등을 시각화 자료와 함께 상세 리포트로 제공합니다.
- **체계적인 모듈 구조**: `core`, `models`, `testing`, `analysis` 등 기능별로 코드를 모듈화하여 유지보수성과 확장성을 높였습니다.

## 🛠️ 기술 스택

- **언어**: Python 3, JavaScript (ES6+), HTML5, CSS3
- **핵심 라이브러리**:
  - **데이터 처리**: Pandas, NumPy
  - **머신러닝**: Scikit-learn, TensorFlow (Keras)
  - **데이터 수집**: yfinance, NewsAPI
  - **시각화**: Matplotlib, Seaborn, Plotly, Chart.js
  - **기타**: TA (Technical Analysis), TextBlob, SHAP, LIME

## ⚠️ 개발 가이드라인 및 필수 규칙

### 🚫 절대 금지 사항

#### 1. 하드코딩 금지 (No Hardcoding)
```python
# ❌ 금지: 하드코딩된 값들
accuracy = 89.2  # 실제 계산 없이 고정값 사용
model_name = "XGBoost Model (3.27% MAPE)"  # 가짜 성능 표시
chart_data = [1.2, 1.5, 1.8, 2.1]  # 임의의 차트 데이터

# ✅ 권장: 동적 계산 및 실제 데이터 사용
accuracy = calculate_model_accuracy(predictions, actual)
model_name = f"{best_model.name} ({best_model.performance:.2f}% MAE)"
chart_data = load_real_performance_data()
```

#### 2. Random 사용 금지 (No Random Data)
```python
# ❌ 금지: 무작위 데이터 생성
price = 450 + random.random() * 10  # 가짜 주가 데이터
volume = random.randint(100000, 1000000)  # 임의 거래량
performance = random.uniform(0.8, 0.95)  # 가짜 성능 지표

# ✅ 권장: 실제 데이터 및 결정론적 방법 사용
price = fetch_real_stock_price(symbol, date)
volume = get_actual_trading_volume(symbol, date)
performance = calculate_actual_model_performance(model, test_data)
```

#### 3. 데이터 누출 금지 (No Data Leakage)
```python
# ❌ 금지: 미래 정보 사용 (데이터 누수)
features['future_price'] = df['Close'].shift(-1)  # 미래 가격 정보
features['target_derived'] = df['Returns'] * 0.5  # 타겟 변수 기반 특성
rsi = calculate_rsi(df['Close'])  # 현재 시점 포함 계산

# ✅ 권장: 과거 정보만 사용 (시점 t-1 기준)
features['price_lag1'] = df['Close'].shift(1)  # 과거 가격만 사용
features['volume_lag1'] = df['Volume'].shift(1)  # 과거 거래량만 사용
rsi = calculate_rsi(df['Close'].shift(1))  # t-1 시점까지만 계산
```

### 📋 데이터 누수 방지 체크리스트

#### Time Series 특성 생성 규칙
- ✅ 모든 features는 t-1 시점 이하의 데이터만 사용
- ✅ Technical indicators는 현재 시점 제외하고 계산
- ✅ Moving averages, volatility 등 rolling 계산 시 현재 시점 제외
- ✅ Cross-validation 시 TimeSeriesSplit 사용 (무작위 split 금지)

#### 모델 검증 규칙
- ✅ Walk-Forward Validation 적용
- ✅ Test set은 시간 순서상 가장 최근 데이터
- ✅ Feature selection도 각 fold 내부에서 수행
- ✅ Hyperparameter tuning 시 미래 정보 사용 금지

### 🎯 품질 보증 방법

#### 1. 코드 검증
```bash
# 하드코딩 검출
grep -r "accuracy.*=" src/ | grep -v "calculate\|compute"
grep -r "performance.*=" src/ | grep -v "evaluate\|measure"

# Random 사용 검출
grep -r "random\." src/
grep -r "np\.random" src/

# 데이터 누수 의심 패턴 검출
grep -r "shift(-" src/  # 미래 정보 shift
grep -r "\.iloc\[.*:\]" src/  # 잘못된 인덱싱
```

#### 2. 성능 지표 현실성 검증
- **Direction Accuracy**: 50-95% (일일 수익률 예측 현실적 범위)
- **R²**: -0.8 ~ 0.8

#### 3. 대시보드 데이터 검증
- ✅ 모든 성능 지표는 실제 계산된 값 사용
- ✅ Chart 데이터는 실제 모델 결과에서 로드
- ✅ Mock 데이터는 개발/테스트 목적으로만 사용

### 🔍 리뷰 가이드라인

**Pull Request 전 체크사항:**
1. [ ] 하드코딩된 성능 값 없음
2. [ ] Random 함수 사용 없음 (시드 고정 제외)
3. [ ] 모든 features가 t-1 이하 시점 데이터 사용
4. [ ] TimeSeriesSplit 검증 적용
5. [ ] 현실적인 성능 지표 범위 내
6. [ ] 실제 데이터 기반 대시보드 업데이트

**코드 리뷰 시 확인사항:**
- 성능 지표가 비현실적으로 높지 않은가? (99%+ 정확도 의심)
- Feature engineering에서 미래 정보 사용하지 않았는가?
- Mock 데이터가 production 코드에 포함되지 않았는가?

## 🚀 설치 및 실행 방법

### 1. 환경 설정

먼저, 프로젝트에 필요한 라이브러리들을 설치합니다.

```bash
pip install -r config/requirements.txt
```

### 2. 전체 파이프라인 실행

프로젝트의 모든 과정을 순차적으로 실행하려면 `system_orchestrator.py`를 사용합니다.

```bash
python src/utils/system_orchestrator.py
```

### 3. 대시보드 실행 (개선됨!)

**🚀 권장 방법 (스마트 자동 서버):**
```bash
cd dashboard
npm run dev        # 포트 충돌 자동 해결하는 http-server
```

**🔧 대안 방법:**
```bash
cd dashboard
npm run serve      # serve 사용
npm run dev:force  # 기존 서버 강제 종료 후 시작

# 또는 기존 방식
python3 -m http.server 8080
```

**✨ 새로운 기능들:**
- 포트 충돌 시 자동으로 다른 포트 탐색 (8080 → 8081 → 8082...)
- 기존 서버 프로세스 자동 감지 및 관리
- http-server, serve, Python 서버 중 선택 가능
- 사용자 친화적인 상태 메시지

### 4. 개별 스크립트 실행 (옵션)

특정 부분만 개별적으로 실행할 수도 있습니다. 예를 들어, 모델 훈련만 실행하려면:

```bash
python src/models/model_training.py
``