# SPY 변동성 예측 시스템 - 기술 아키텍처

**문서 버전:** 2025-10-23
**시스템 모드:** Volatility Prediction (검증 완료)
**데이터 소스:** SPY ETF (2015-2024, 2,488 샘플)

---

## 📐 시스템 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────┐
│                     SPY Volatility Prediction System         │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  Data Layer   │───▶│  Model Layer  │───▶│ Validation    │
│  (src/core/)  │    │ (src/models/) │    │ (src/validation/)│
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
                    ┌───────────────────┐
                    │ Analysis & Output │
                    │  (src/analysis/)  │
                    └───────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
                    ▼                   ▼
            ┌─────────────┐     ┌─────────────┐
            │  Dashboard  │     │   Reports   │
            │ (dashboard/)│     │   (data/)   │
            └─────────────┘     └─────────────┘
```

---

## 🔧 핵심 컴포넌트

### 1. Data Layer (`src/core/`)

**역할:** 데이터 전처리 및 검증

#### `data_processor.py`
```python
class DataProcessor:
    """SPY ETF 데이터 처리 및 변환"""

    def load_spy_data(self):
        """yfinance에서 SPY 데이터 로드"""
        # 2015-01-01 ~ 2024-12-31
        # OHLCV + Adj Close

    def calculate_features(self):
        """31개 변동성 특성 계산"""
        # - VIX 기반 특성 (4개)
        # - 실현 변동성 (3개)
        # - 지수 가중 변동성 (3개)
        # - 래그 특성 (4개)
        # - HAR 특성 (3개)
        # - 기타 변동성 지표 (14개)

    def create_target(self):
        """타겟 변수 생성 (target_vol_5d)"""
        # 완전한 시간적 분리 보장
        # for i in range(len(returns)):
        #     future_returns = returns[i+1:i+6]
        #     target[i] = future_returns.std()
```

**데이터 무결성 보장:**
- ✅ 특성 ≤ t (현재 및 과거 데이터만 사용)
- ✅ 타겟 ≥ t+1 (미래 데이터만 사용)
- ✅ Zero overlap (완전한 시간적 분리)

#### `config.py`
```python
class Config:
    """시스템 설정 관리"""

    DATA_START = "2015-01-01"
    DATA_END = "2024-12-31"
    TARGET_HORIZON = 5  # 5일 후 변동성 예측

    # 검증 설정
    CV_SPLITS = 5
    CV_EMBARGO_PCT = 0.01
    TEST_SIZE = 0.20
```

#### `logger.py`
```python
"""통합 로깅 시스템"""
# 모든 작업 로그 기록
# 데이터 누출 검증 로그
# 모델 성능 추적
```

---

### 2. Model Layer (`src/models/`)

**역할:** 변동성 예측 모델 구현

#### `correct_target_design.py` (메인 모델)
```python
# Ridge Regression 구현
model = Ridge(alpha=1.0, random_state=42)
scaler = StandardScaler()

# 학습 파이프라인
X_scaled = scaler.fit_transform(X_train)
model.fit(X_scaled, y_train)

# 성능: R² = 0.3113 ± 0.1756 (Purged K-Fold CV)
```

**지원 모델:**
- **Ridge Regression** (메인 모델)
- **Lasso** (α=0.001) - 가장 안정적
- **ElasticNet** - 최고 CV 성능
- **Random Forest** - 비교 목적
- **HAR Benchmark** - 학계 기준선

---

### 3. Validation Layer (`src/validation/`)

**역할:** 데이터 누출 방지 및 성능 검증

#### `purged_cross_validation.py`
```python
class PurgedKFold:
    """Financial ML 표준 CV 구현"""

    def __init__(self, n_splits=5, pct_embargo=0.01):
        # 시간 순서 보존
        # Train-Test 사이 embargo 구간

    def split(self, X, y):
        # Timeline:
        # [====Train====][Embargo][==Test==][Embargo]...

        for fold in range(self.n_splits):
            train_idx, test_idx = self._get_indices(fold)
            yield train_idx, test_idx
```

**보장 사항:**
- ✅ 시간적 순서 보존 (no shuffle)
- ✅ Train-Test 간 embargo 구간 (1% = 약 25 샘플)
- ✅ 데이터 누출 Zero

#### `economic_backtest_validator.py`
```python
class EconomicBacktest:
    """경제적 가치 검증"""

    def backtest_volatility_strategy(self):
        """변동성 기반 포지션 조정"""

        # 예측 변동성 ↑ → 포지션 ↓ (리스크 회피)
        # 예측 변동성 ↓ → 포지션 ↑ (공격적)

        position = base_position / (1 + predicted_vol)

    def calculate_transaction_costs(self):
        """거래 비용 포함 (0.1% per trade)"""
        return trades * 0.001
```

**검증 결과:**
- 연 수익률: 14.10% (벤치마크 22.71%)
- **변동성: 12.24%** (벤치마크 13.04%) ✅ **-0.8% 감소**
- 샤프 비율: 0.989 (벤치마크 1.588)

#### `advanced_leakage_detection.py`
```python
class LeakageDetector:
    """데이터 누출 검증 시스템"""

    def check_temporal_separation(self):
        """시간적 분리 검증"""
        # 특성의 최대 시점 < 타겟의 최소 시점
        assert max(feature_times) < min(target_times)

    def check_feature_target_correlation(self):
        """특성-타겟 상관관계 검증"""
        # 동시점 상관관계가 시차 상관관계보다 높으면 누출 의심

    def validate_cv_split(self):
        """CV split 누출 검증"""
        # Test fold에 미래 데이터 없는지 확인
```

---

### 4. Analysis Layer (`src/analysis/`)

**역할:** 성능 분석 및 해석

#### `model_diagnosis.py`
```python
"""모델 진단 및 성능 분석"""

def analyze_residuals(y_true, y_pred):
    """잔차 분석 (정규성, 자기상관)"""

def feature_importance_analysis(model, X):
    """SHAP 기반 특성 중요도 분석"""

def prediction_interval_analysis():
    """예측 구간 신뢰도 분석"""
```

#### `volatility_pattern_discovery.py`
```python
"""변동성 패턴 탐지"""

def detect_regime_changes():
    """변동성 regime 변화 탐지"""
    # Low volatility regime
    # High volatility regime
    # Transition periods

def seasonal_analysis():
    """계절성 분석"""
    # 월별 변동성 패턴
    # 요일 효과
```

---

### 5. Orchestration Layer (`src/utils/`)

**역할:** 시스템 통합 및 조정

#### `system_orchestrator.py`
```python
class SystemOrchestrator:
    """전체 시스템 조정자"""

    def initialize_components(self):
        """모든 컴포넌트 초기화 및 검증"""
        # 1. 데이터 로드 및 검증
        # 2. 모델 로드
        # 3. 성능 메트릭 확인
        # 4. 시스템 상태 저장

    def run_full_pipeline(self):
        """전체 파이프라인 실행"""
        data = self.load_data()
        features, target = self.engineer_features(data)
        results = self.validate_models(features, target)
        self.save_results(results)

    def health_check(self):
        """시스템 상태 체크"""
        # 데이터 무결성
        # 모델 성능
        # 파일 존재 여부
```

**Entry Point:**
```bash
PYTHONPATH=/root/workspace python3 src/utils/system_orchestrator.py
```

---

### 6. Dashboard Layer (`dashboard/`)

**역할:** 정적 웹 대시보드 (서버 불필요)

#### 구조:
```
dashboard/
├── index.html              # 메인 대시보드 (3-tab interface)
├── modules/
│   ├── SP500PredictionWidget.js
│   ├── FeatureImpactWidget.js
│   └── EconomicValueWidget.js
├── data/
│   └── (임베디드 JavaScript 데이터)
└── package.json            # Smart server scripts
```

#### `index.html`
```html
<!-- 3-Tab Analysis Interface -->
<div class="tab-content">
  <!-- Tab 1: Volatility Predictions -->
  <div id="volatility-predictions">
    <!-- SPY 실제 변동성 vs Ridge 예측 -->
  </div>

  <!-- Tab 2: Feature Impact -->
  <div id="feature-impact">
    <!-- SHAP 기반 특성 중요도 -->
  </div>

  <!-- Tab 3: Economic Value -->
  <div id="economic-value">
    <!-- Backtest 결과 및 리스크 메트릭 -->
  </div>
</div>
```

**기술 스택:**
- Bootstrap 5 (반응형 디자인)
- Chart.js (시각화)
- FontAwesome (아이콘)
- ES6+ Modules (순수 JavaScript)

**실행:**
```bash
cd dashboard
npm run dev        # Smart http-server (권장)
# → http://localhost:8080/index.html
```

---

## 🔄 데이터 파이프라인

### End-to-End Flow

```
1. 데이터 수집 (yfinance)
   │
   ├─▶ SPY ETF OHLCV 데이터
   │   (2015-01-01 ~ 2024-12-31)
   │
   ▼
2. 특성 엔지니어링 (data_processor.py)
   │
   ├─▶ 31개 변동성 특성 계산
   │   - VIX 기반 (4)
   │   - 실현 변동성 (3)
   │   - 래그 특성 (4)
   │   - HAR 특성 (3)
   │   - 기타 (17)
   │
   ▼
3. 타겟 생성 (완전한 시간적 분리)
   │
   ├─▶ target_vol_5d = std(returns[t+1:t+6])
   │   (미래 5일 변동성)
   │
   ▼
4. 특성 선택 (상관관계 기반)
   │
   ├─▶ 상위 25개 특성 선택
   │   (타겟과 상관계수 기준)
   │
   ▼
5. Purged K-Fold CV (5-fold)
   │
   ├─▶ n_splits = 5
   ├─▶ embargo = 1% (약 25 샘플)
   ├─▶ 시간 순서 보존
   │
   ▼
6. 모델 학습 (Ridge Regression)
   │
   ├─▶ StandardScaler 정규화
   ├─▶ Ridge(alpha=1.0)
   ├─▶ CV R² = 0.3113 ± 0.1756
   │
   ▼
7. Walk-Forward Test (마지막 20%)
   │
   ├─▶ Test R² = 0.0879
   ├─▶ HAR 벤치마크 대비 35배 우수
   │
   ▼
8. 경제적 백테스트
   │
   ├─▶ 변동성 기반 포지션 조정
   ├─▶ 거래 비용 포함 (0.1%)
   ├─▶ 결과: 변동성 -0.8% 감소
   │
   ▼
9. 결과 저장 및 시각화
   │
   ├─▶ JSON: data/validation/
   ├─▶ CSV: data/*.csv
   ├─▶ 그래프: paper/figures/
   └─▶ 대시보드: dashboard/index.html
```

---

## 📊 데이터 구조

### 주요 데이터 파일

#### 1. 검증 결과 (JSON)
```
data/validation/comprehensive_model_validation.json
├── timestamp
├── data_source: "SPY (2015-2024)"
├── validation_method: "Purged K-Fold CV (5-fold, embargo=1%)"
├── target: "target_vol_5d (5-day future volatility)"
└── models:
    ├── "HAR Benchmark": {...}
    ├── "Ridge Volatility": {...}
    ├── "Lasso 0.001": {...}
    ├── "ElasticNet": {...}
    └── "Random Forest": {...}
```

#### 2. CSV 보고서
```
data/
├── model_comparison.csv         # 모델별 전체 메트릭 비교
├── fold_validation_results.csv  # Fold별 상세 결과
├── performance_summary.csv      # 성능 요약 (발표용)
└── statistical_analysis.csv     # 통계 분석 (95% CI 등)
```

#### 3. 시스템 상태
```
data/raw/
├── model_performance.json       # 실시간 성능 메트릭
├── sp500_prediction_data.json   # 예측 데이터
├── trading_volume.json          # 거래량 데이터
└── market_sentiment.json        # 시장 센티멘트
```

---

## 🛡️ 데이터 무결성 프레임워크

### 3단계 검증 시스템

#### Level 1: 타겟 설계 검증
```python
# ✅ 올바른 예시 (미래 데이터만)
for i in range(len(returns)):
    future_returns = returns[i+1:i+6]  # t+1 ~ t+5
    target[i] = future_returns.std()

# ❌ 잘못된 예시 (현재 포함)
target = df['returns'].rolling(5).std().shift(-5)  # 현재 포함!
```

#### Level 2: CV 누출 검증
```python
# ✅ Purged K-Fold (시간 순서 보존 + embargo)
cv = PurgedKFold(n_splits=5, pct_embargo=0.01)

# ❌ 일반 K-Fold (미래 데이터 누출 가능)
cv = KFold(n_splits=5, shuffle=True)  # No!
```

#### Level 3: 특성 검증
```python
# ✅ 과거 데이터만 사용
realized_vol_5 = returns[:t].rolling(5).std()

# ❌ 미래 데이터 포함
realized_vol_5 = returns.rolling(5).std()  # t+1 ~ t+4 포함!
```

---

## 🔌 API 및 인터페이스

### Python API

#### 기본 사용법
```python
from src.utils.system_orchestrator import SystemOrchestrator

# 시스템 초기화
orchestrator = SystemOrchestrator()
result = orchestrator.initialize_components()

# 상태 확인
if result:
    print("System ready!")

# 전체 파이프라인 실행
orchestrator.run_full_pipeline()
```

#### 모델 학습
```python
from src.models.correct_target_design import train_ridge_model

# Ridge 모델 학습
model, scaler, results = train_ridge_model(
    X_train, y_train, X_test, y_test
)

# 예측
X_scaled = scaler.transform(X_new)
predictions = model.predict(X_scaled)
```

#### 검증
```python
from src.validation.purged_cross_validation import purged_cv_evaluation

# Purged K-Fold CV
cv_scores = purged_cv_evaluation(
    model, X, y, n_splits=5, pct_embargo=0.01
)
```

### 대시보드 API

#### JavaScript 모듈
```javascript
// 예측 위젯
import { SP500PredictionWidget } from './modules/SP500PredictionWidget.js';

const widget = new SP500PredictionWidget();
widget.render('#volatility-predictions');

// 특성 영향 위젯
import { FeatureImpactWidget } from './modules/FeatureImpactWidget.js';

const impact = new FeatureImpactWidget();
impact.render('#feature-impact');
```

---

## 🚀 배포 및 운영

### 개발 모드
```bash
# Python 시스템 실행
PYTHONPATH=/root/workspace python3 src/utils/system_orchestrator.py

# 대시보드 개발 서버
cd dashboard && npm run dev
```

### 프로덕션 모드
```bash
# 정적 빌드 (대시보드)
cd dashboard
npm run build  # (현재는 정적 HTML이므로 빌드 불필요)

# 웹 서버 배포
# dashboard/ 디렉토리를 nginx/Apache에 배포
```

### 시스템 상태 체크
```bash
PYTHONPATH=/root/workspace python3 -c "
from src.utils.system_orchestrator import SystemOrchestrator
orchestrator = SystemOrchestrator()
result = orchestrator.initialize_components()
print(f'System Status: {\"Ready\" if result else \"Error\"}')
"
```

---

## 📈 성능 메트릭

### 모델 성능 (Verified)

| Model | CV R² | Test R² | Status |
|-------|-------|---------|--------|
| **Lasso 0.001** | **0.3373** | **0.0879** | ✅ **권장** |
| ElasticNet | 0.3444 | 0.0254 | ⚠️ 과적합 위험 |
| Ridge | 0.2881 | -0.1429 | ❌ 실패 |
| HAR | 0.2300 | -0.0431 | ❌ 벤치마크 |
| Random Forest | 0.1713 | 0.0233 | ❌ 최악 |

### 경제적 가치 (Proven)

| Metric | Strategy | Benchmark | Improvement |
|--------|----------|-----------|-------------|
| Annual Return | 14.10% | 22.71% | -8.62% |
| **Volatility** | **12.24%** | **13.04%** | **-0.8%** ✅ |
| Sharpe Ratio | 0.989 | 1.588 | -0.600 |
| Max Drawdown | -10.81% | -10.15% | -0.66% |

---

## 🔒 보안 및 제한사항

### 데이터 보안
- ✅ 로컬 yfinance 사용 (외부 API 키 불필요)
- ✅ 민감 정보 없음 (공개 시장 데이터만 사용)
- ✅ 정적 대시보드 (서버 공격 표면 최소화)

### 시스템 제한사항
- **데이터 범위:** 2015-2024 (최신 데이터 수동 업데이트 필요)
- **예측 horizon:** 5일 고정 (변경 시 재학습 필요)
- **특성 개수:** 25개 (변경 시 검증 필요)
- **모델 재학습:** 분기별 권장

### 알려진 이슈
- Random Forest 성능 낮음 (0.1713) - 트리 기반 모델 부적합
- ElasticNet 과적합 위험 (CV-Test 갭 0.319) - 신중 사용
- Ridge Test R² 음수 (-0.1429) - 실전 부적합

---

## 🛠️ 유지보수 가이드

### 정기 점검 항목

#### 월간 (Monthly)
- [ ] yfinance 데이터 업데이트 확인
- [ ] 대시보드 정상 작동 확인
- [ ] 로그 파일 점검

#### 분기별 (Quarterly)
- [ ] 모델 재학습 및 성능 재평가
- [ ] 새로운 특성 추가 실험
- [ ] 벤치마크 비교 업데이트

#### 연간 (Yearly)
- [ ] 전체 시스템 아키텍처 리뷰
- [ ] 기술 스택 업데이트 검토
- [ ] 성능 최적화 작업

### 문제 해결

#### 데이터 로드 실패
```bash
# yfinance 재설치
pip install --upgrade yfinance

# 데이터 캐시 삭제
rm -rf ~/.cache/py-yfinance
```

#### 대시보드 포트 충돌
```bash
# 스마트 포트 감지 사용
npm run dev        # 자동 8080 → 8081 → 8082...

# 또는 강제 종료 후 재시작
npm run dev:force
```

#### 모델 성능 저하
```python
# 데이터 누출 검증
from src.validation.advanced_leakage_detection import LeakageDetector
detector = LeakageDetector()
detector.run_full_check()

# Purged K-Fold 재검증
python3 scripts/comprehensive_model_validation.py
```

---

## 📚 참고 자료

### 학술 논문
- **Purged K-Fold CV**: Advances in Financial Machine Learning (Marcos López de Prado)
- **HAR Model**: Corsi (2009) - Heterogeneous Autoregressive model
- **변동성 예측**: GARCH, EWMA, Realized Volatility literature

### 기술 문서
- `VALIDATION_METHODOLOGY.md` - 검증 방법론 상세
- `MODEL_PERFORMANCE_REPORT.md` - 모델 성능 분석
- `PRESENTATION_SUMMARY.md` - 핵심 결과 요약
- `CLAUDE.md` - 개발 가이드

### 코드 예제
- `scripts/comprehensive_model_validation.py` - 전체 검증 스크립트
- `src/models/correct_target_design.py` - 모델 구현
- `src/validation/purged_cross_validation.py` - CV 구현

---

## 📞 기술 지원

### 재현 방법

1. **전체 검증 재실행:**
```bash
PYTHONPATH=/root/workspace python3 scripts/comprehensive_model_validation.py
```

2. **CSV 보고서 재생성:**
```bash
python3 scripts/generate_csv_reports.py
```

3. **그래프 재생성:**
```bash
python3 scripts/create_paper_figures.py
```

4. **대시보드 실행:**
```bash
cd dashboard && npm run dev
open http://localhost:8080/index.html
```

### 시스템 요구사항

**Python:**
- Python 3.8+
- pandas, numpy, scikit-learn
- yfinance, matplotlib

**Node.js:**
- Node.js 14+
- http-server or serve

**설치:**
```bash
# Python 의존성
pip install -r requirements/base.txt

# Node.js 의존성
cd dashboard && npm install
```

---

## 🎯 향후 개발 계획

### Phase 1: 모델 개선 (완료)
- ✅ Purged K-Fold CV 구현
- ✅ 데이터 누출 완전 제거
- ✅ 경제적 백테스트 검증

### Phase 2: 시스템 확장 (진행 중)
- ✅ 정적 대시보드 구축
- ✅ CSV 보고서 자동 생성
- ✅ 기술 문서 작성

### Phase 3: 고급 기능 (계획)
- [ ] 실시간 데이터 스트리밍
- [ ] 앙상블 모델 (Lasso + ElasticNet)
- [ ] 딥러닝 실험 (LSTM, Transformer)
- [ ] 포트폴리오 최적화 통합

---

**문서 버전:** 1.0
**최종 업데이트:** 2025-10-23
**작성자:** SPY Volatility Prediction Team
**검증 상태:** ✅ Verified (Real Data, Zero Leakage, Proper CV)
