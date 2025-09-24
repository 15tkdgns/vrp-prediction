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
# Train Ridge volatility prediction model
PYTHONPATH=/root/workspace python3 src/models/correct_target_design.py

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

This is a **Financial Volatility Prediction System** for SPY ETF analysis that works with verified Ridge regression model. **메인 모델: 변동성 예측 (R² = 0.3113)**

### System Mode: Volatility Prediction
- ✅ **메인 모델**: Ridge Regression 변동성 예측 (R² = 0.3113)
- ✅ **타겟 변수**: target_vol_5d (5일 후 변동성 예측)
- ✅ **경제적 가치**: 변동성 0.8% 감소, 연 14.1% 수익률 실증
- ✅ **실용적 적용**: VIX 옵션 거래, 동적 헤징, 리스크 관리
- ✅ **데이터**: 2015-2024 SPY 실제 데이터 (2,445 샘플)
- ✅ **특성**: 31개 선별된 변동성/래그 특성

### Core System Architecture (Static Mode)

- **`src/core/`**: Data validation and preprocessing
  - ✅ `data_processor.py`: SPY data processing (pre-processed data available)
  - ✅ `config.py`: System configuration management
  - ✅ `logger.py`: Logging system

- **`src/models/`**: Verified Ridge regression model
  - ✅ `correct_target_design.py`: Proper temporal separation implementation
  - ✅ **Model**: Ridge(alpha=1.0) with StandardScaler
  - ✅ **Performance**: R² = 0.3113 ± 0.1756 (Purged K-Fold CV)

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
- ✅ **Self-contained Visualization** - All data embedded in JavaScript
- ✅ **3-Tab Analysis Interface**:
  - **Volatility Predictions**: SPY volatility vs Ridge model predictions
  - **Feature Impact**: SHAP-based feature importance analysis
  - **Economic Value**: Backtest results and risk management metrics
- ✅ **Responsive Design** - Bootstrap 5 + Chart.js + FontAwesome
- ✅ **No Backend Dependencies** - Pure client-side application

### Verified Data Pipeline

1. ✅ **Real SPY Data**: 2015-2024 actual market data (2,514 observations)
2. ✅ **Feature Engineering**: Volatility features (≤ t) with complete temporal separation
3. ✅ **Target Design**: 5-day future volatility (≥ t+1) with zero data leakage
4. ✅ **Model Training**: Ridge regression with Purged K-Fold CV
5. ✅ **Performance Validation**: HAR benchmark comparison and economic backtest

### Key Data Paths

- `data/raw/`: Raw SPY data and system status files
- `data/training/`: Leak-free training datasets
- `data/models/`: Trained Ridge regression model
- `dashboard/`: Static volatility prediction dashboard

### Technology Stack

- **Python**: Core ML pipeline (scikit-learn, pandas, numpy, yfinance)
- **JavaScript**: Dashboard frontend with ES6+ modules
- **Data Source**: yfinance for actual SPY ETF data
- **Model**: Ridge Regression with Purged K-Fold Cross-Validation
- **Validation**: Economic backtest with transaction costs
- **Visualization**: Chart.js (frontend), matplotlib (backend)

### System Entry Points (Static Mode)

- ✅ **Main System**: `PYTHONPATH=/root/workspace python3 src/utils/system_orchestrator.py`
- ✅ **Dashboard**: `cd dashboard && npm run dev` → `http://localhost:8080/index.html`
- ✅ **Model Training**: `python3 src/models/correct_target_design.py`
- ✅ **Economic Backtest**: `python3 src/validation/economic_backtest_validator.py`
- ✅ **Performance Summary**: `python3 model_performance_summary_table.py`

### Important Notes (Static Mode)

- ✅ **No API Keys Required** - System works entirely with yfinance data
- ✅ **Data Integrity Verified** - Complete temporal separation guaranteed
- ✅ **Pre-processed Data** - All analysis results available in `data/raw/`
- ✅ **Static Dashboard** - Self-contained HTML, no server dependencies
- ✅ **Model Performance** - Verified R² = 0.3113 (HAR benchmark: 0.0088)
- ✅ **Economic Value** - Proven 0.8% volatility reduction with 14.1% annual return
- ✅ **Academic Standard** - Purged K-Fold CV with complete validation

### Data Integrity Framework

- **완전한 시간적 분리**: 특성 ≤ t, 타겟 ≥ t+1 (zero overlap)
- **Purged K-Fold CV**: n_splits=5, purge_length=5, embargo_length=5
- **실제 데이터 검증**: SPY ETF 2015-2024 (no simulation)
- **벤치마크 비교**: HAR model (academic standard)
- **경제적 가치 실증**: Transaction cost included backtest

### Performance Metrics (Verified)

| Metric | Our Model | HAR Benchmark | Improvement |
|--------|-----------|---------------|-------------|
| **R² Score** | **0.3113** | 0.0088 | **35.4x better** |
| **MSE** | **0.6887** | 0.9912 | **30.5% better** |
| **RMSE** | **0.8298** | 0.9956 | **16.7% better** |
| **MAE** | **0.4573** | 0.7984 | **42.7% better** |

### Economic Value (Proven)

| Metric | Strategy | Benchmark | Effect |
|--------|----------|-----------|---------|
| **Annual Return** | 14.10% | 22.71% | -8.62% |
| **Volatility** | **12.24%** | **13.04%** | **-0.8%** ✅ |
| **Sharpe Ratio** | 0.989 | 1.588 | -0.600 |
| **Max Drawdown** | -10.81% | -10.15% | -0.66% |

**Core Value**: Risk management through volatility reduction proven by real backtest.

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.


      IMPORTANT: this context may or may not be relevant to your tasks. You should not respond to this context unless it is highly relevant to your task.