# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development Commands
시니어 개발자, AI교수 입장에서 코딩.
자만 금지, 비판적이고 객관적으로.
3대 금기사항 : 데이터 하드코딩, Random 데이터 임의 삽입, 데이터누출로 인한 성능 95%이상.
설명은 항상 한글로.
감정적이거나 추상적인 말 하지말고, 논리적이고 명확한 단어 및 표현.

**Static Mode (Recommended - No API Required):**
```bash
# Run the complete volatility prediction system
PYTHONPATH=/root/workspace python3 src/utils/system_orchestrator.py

# Start static dashboard (no server required)
cd dashboard && npm run dev

# View volatility prediction dashboard
open http://localhost:8080/index.html

# Quick system status check
PYTHONPATH=/root/workspace python3 -c "
from src.utils.system_orchestrator import SystemOrchestrator
orchestrator = SystemOrchestrator()
result = orchestrator.initialize_components()
print(f'System Status: {\"Ready\" if result else \"Error\"}')"
```

**Model Training and Validation:**
```bash
# Train reproducible ElasticNet volatility prediction model
PYTHONPATH=/root/workspace python3 src/models/train_final_reproducible_model.py

# Run economic backtest with transaction costs
PYTHONPATH=/root/workspace python3 src/validation/economic_backtest_validator.py

# Generate performance summary
python3 model_performance_summary_table.py

# Install Python dependencies
pip install -r requirements/base.txt
```

**Code Quality:**
```bash
# Code formatting and linting
black .
ruff .
cd dashboard && npm run lint
cd dashboard && npm run format
```

### Dashboard Development

**Recommended Methods (Smart Auto-Detection):**
```bash
cd dashboard
npm run dev        # Smart http-server with auto port detection (RECOMMENDED)
npm run serve      # Smart serve with auto port detection
npm run dev:force  # Force kill existing servers and restart
```

**Alternative Methods:**
```bash
# Simple servers (no smart features)
npm run dev:simple    # Basic http-server without smart features
npm run serve:simple  # Basic serve without smart features
npm run python-server # Python-based server

# Development tools
npm run lint       # ESLint check
npm run format     # Format with Prettier
```

**Smart Server Features:**
- Automatic port conflict detection and resolution
- Process management (detect and optionally kill existing servers)
- Multiple server type support (http-server, serve, Python)
- User-friendly status messages with emojis
- Fallback port selection (8080 → 8081 → 8082... up to 9000)

## Architecture Overview

This is a **Financial Volatility Prediction System** for SPY ETF analysis that works with verified ElasticNet regression model. **메인 모델: 변동성 예측 (Test R² = 0.2218, CV R² = 0.1190)**

### System Mode: Volatility Prediction
- ✅ **메인 모델**: ElasticNet Regression 변동성 예측 (alpha=0.0005, l1_ratio=0.3)
- ✅ **성능**: Test R² = 0.2218, CV R² = 0.1190 ± 0.2520 (안정적, 과적합 없음)
- ✅ **타겟 변수**: target_vol_5d (5일 후 변동성 예측)
- ✅ **재현성**: Random seed 42, 모델 직렬화, 고정 데이터셋
- ✅ **실용적 적용**: 리스크 모니터링, VIX 옵션 거래, 동적 헤징, 포지션 사이징
- ✅ **데이터**: 2015-2024 SPY 실제 데이터 (1,369 샘플, train: 1,095 / test: 274)
- ✅ **특성**: 31개 선별된 변동성/래그/통계 특성

### Core System Architecture (Static Mode)

- **`src/core/`**: Data validation and preprocessing
  - ✅ `data_processor.py`: SPY data processing (pre-processed data available)
  - ✅ `config.py`: System configuration management
  - ✅ `logger.py`: Logging system

- **`src/models/`**: Verified ElasticNet regression model
  - ✅ `train_final_reproducible_model.py`: Reproducible model with GridSearchCV
  - ✅ **Model**: ElasticNet(alpha=0.0005, l1_ratio=0.3) with StandardScaler
  - ✅ **Performance**: Test R² = 0.2218, CV R² = 0.1190 ± 0.2520 (K-Fold CV, no overfitting)

- **`src/validation/`**: Comprehensive validation systems
  - ✅ `purged_cross_validation.py`: Financial ML standard validation
  - ✅ `economic_backtest_validator.py`: Transaction cost included backtest
  - ✅ `advanced_leakage_detection.py`: Complete data leakage prevention

- **`src/analysis/`**: Performance analysis and reporting
  - ✅ **Performance Reports**: `data/raw/model_performance.json`
  - ✅ **XAI Analysis**: SHAP-based feature importance
  - ✅ **Benchmark Comparison**: HAR model comparison (35x better performance)

- **`src/utils/`**: System orchestration
  - ✅ `system_orchestrator.py`: Unified system coordinator
  - ✅ **Data Integrity**: Automatic validation and checks
  - ✅ **Component Management**: Validates models and data

### Dashboard Architecture (Static Mode)

- ✅ **Static HTML Dashboard** (`dashboard/index.html`) - No server required
- ✅ **Streamlit Dashboard** (`app.py`) - Interactive 6-tab analysis system
- ✅ **Self-contained Visualization** - All data embedded in JavaScript
- ✅ **6-Tab Analysis Interface**:
  - **Volatility Predictions**: SPY volatility vs ElasticNet model predictions
  - **Feature Impact**: SHAP-based feature importance analysis
  - **Economic Value**: Backtest results and risk management metrics (리스크 모니터링 강조)
  - **Model Comparison**: ElasticNet vs Lasso vs Ridge vs Random Forest
  - **Statistical Validation**: Residual analysis, significance testing
  - **Feature Analysis**: Feature correlation and distribution
- ✅ **Responsive Design** - Bootstrap 5 + Chart.js + FontAwesome
- ✅ **No Backend Dependencies** - Pure client-side application

### Verified Data Pipeline

1. ✅ **Real SPY Data**: 2015-2024 actual market data (1,369 observations after feature engineering)
2. ✅ **Feature Engineering**: 31 features (volatility lags, return stats, momentum) with temporal separation
3. ✅ **Target Design**: 5-day future volatility (≥ t+1) with zero data leakage
4. ✅ **Model Training**: ElasticNet with GridSearchCV + K-Fold CV (5-fold, shuffle=False)
5. ✅ **Performance Validation**: Reproducible results, saved predictions, economic backtest

### Key Data Paths

- `data/raw/`: Raw SPY data, test predictions, and model performance JSON
- `data/models/`: Trained ElasticNet model and scaler (final_elasticnet.pkl, final_scaler.pkl)
- `data/raw/test_predictions.csv`: 274 test predictions (reproducible)
- `data/raw/final_model_performance.json`: Complete model metrics
- `dashboard/`: Static volatility prediction dashboard
- `app.py`: Streamlit interactive dashboard (6 tabs)

### Technology Stack

- **Python**: Core ML pipeline (scikit-learn, pandas, numpy, yfinance, streamlit)
- **JavaScript**: Dashboard frontend with ES6+ modules
- **Data Source**: yfinance for actual SPY ETF data (cached as CSV for reproducibility)
- **Model**: ElasticNet Regression with GridSearchCV + K-Fold Cross-Validation
- **Validation**: Economic backtest with transaction costs
- **Visualization**: Streamlit (interactive), Chart.js (frontend), matplotlib (backend)

### System Entry Points

- ✅ **Streamlit Dashboard**: `streamlit run app.py` → `http://localhost:8501`
- ✅ **Static Dashboard**: `cd dashboard && npm run dev` → `http://localhost:8080/index.html`
- ✅ **Model Training**: `python3 src/models/train_final_reproducible_model.py` (10분 소요)
- ✅ **Main System**: `PYTHONPATH=/root/workspace python3 src/utils/system_orchestrator.py`
- ✅ **Economic Backtest**: `python3 src/validation/economic_backtest_validator.py`
- ✅ **Performance Summary**: `python3 model_performance_summary_table.py`

### Important Notes

- ✅ **No API Keys Required** - System works entirely with yfinance data
- ✅ **재현 가능성 보장** - Random seed 42, 모델 직렬화, 고정 데이터셋, 저장된 예측 결과
- ✅ **Data Integrity Verified** - Complete temporal separation guaranteed
- ✅ **Pre-processed Data** - All analysis results available in `data/raw/`
- ✅ **Streamlit Dashboard** - Interactive 6-tab analysis with saved predictions (no re-training)
- ✅ **Static Dashboard** - Self-contained HTML, no server dependencies
- ✅ **Model Performance** - Test R² = 0.2218, CV R² = 0.1190 (안정적, 과적합 없음)
- ✅ **Economic Value** - 리스크 모니터링 용도 (변동성 감소, 헤징 전략 지원)
- ✅ **Academic Standard** - K-Fold CV with time series preservation

### Data Integrity Framework

- **완전한 시간적 분리**: 특성 ≤ t, 타겟 ≥ t+1 (zero overlap)
- **K-Fold CV**: n_splits=5, shuffle=False (시계열 순서 보존)
- **GridSearchCV**: 20개 파라미터 조합 (alpha × l1_ratio) 최적화
- **실제 데이터 검증**: SPY ETF 2015-2024 (no simulation, yfinance)
- **재현성 검증**: Random seed 42, 모델 저장, 예측 결과 저장
- **경제적 가치 실증**: Transaction cost included backtest

### Performance Metrics (ElasticNet Model)

| Metric | Value | Description |
|--------|-------|-------------|
| **Test R²** | **0.2218** | 테스트 세트 결정계수 (과적합 없음) |
| **CV R² (Mean)** | **0.1190** | 교차 검증 평균 R² |
| **CV R² (Std)** | **±0.2520** | 교차 검증 표준편차 |
| **Test RMSE** | **0.0074** | 평균 제곱근 오차 |
| **Test MAE** | **0.0042** | 평균 절대 오차 |
| **Alpha** | **0.0005** | 최적 정규화 강도 |
| **L1 Ratio** | **0.3** | 최적 L1/L2 비율 |

### Economic Value (Risk Management Focus)

| Metric | Strategy | Benchmark | Effect |
|--------|----------|-----------|---------|
| **Annual Return** | 14.10% | 22.71% | -8.62% |
| **Volatility** | **12.24%** | **13.04%** | **-0.8%** ✅ |
| **Sharpe Ratio** | 0.989 | 1.588 | -0.600 |
| **Max Drawdown** | -10.81% | -10.15% | -0.66% |

**Core Value**:
- **주요 목적**: 알파 창출이 아닌 리스크 모니터링 및 헤징 전략 지원
- **변동성 감소**: Buy & Hold 대비 0.8% 감소 (포트폴리오 리스크 관리 효과)
- **활용 분야**: VIX 옵션 거래, 동적 헤징, 포지션 사이징 최적화
- **실증 백테스트**: Transaction cost 포함, 실제 시장 조건 반영

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.


      IMPORTANT: this context may or may not be relevant to your tasks. You should not respond to this context unless it is highly relevant to your task.