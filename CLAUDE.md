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
# Run the complete static analysis system
PYTHONPATH=/root/workspace python3 src/utils/system_orchestrator.py

# Start static dashboard (no server required)
cd dashboard && npm run dev

# View static dashboard in browser
open http://localhost:8080/index.html

# Quick system status check
PYTHONPATH=/root/workspace python3 -c "
from src.utils.system_orchestrator import SystemOrchestrator
orchestrator = SystemOrchestrator()
result = orchestrator.initialize_components()
print(f'System Status: {\"Ready\" if result else \"Error\"}')"
```

**Model Training (if needed):**
```bash
# Train models individually  
PYTHONPATH=/root/workspace python3 src/models/model_training.py

# Install Python dependencies
pip install -r config/requirements.txt
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

# Direct script usage with options
./start-dev.sh http-server         # Use http-server (recommended)
./start-dev.sh serve              # Use serve
./start-dev.sh python             # Use Python server  
./start-dev.sh http-server --force # Force restart
./start-dev.sh --help             # Show all options

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

This is a **Static AI Analysis System** for S&P500 event detection that works entirely with pre-processed data. **No API keys or real-time data feeds required.**

### System Mode: Static Analysis
- ✅ **No External APIs Required** - All analysis uses existing datasets
- ✅ **Pre-trained Models** - Gradient Boosting (0.91% MAPE), Random Forest, Ensemble
- ✅ **Static HTML Dashboard** - Self-contained, works without servers
- ✅ **2025 H1 Data Analysis** - Complete analysis of SPY performance through June 2025
- ✅ **Comprehensive Reports** - Feature importance, model performance, news impact analysis

### Core System Architecture (Static Mode)

- **`src/core/`**: Data validation and preprocessing
  - ⚠️ `data_collection_pipeline.py`: Legacy (not used in static mode)
  - ✅ `advanced_preprocessing.py`: Feature engineering (pre-processed data available)
  - ✅ `api_config.py`: Static mode stub (prevents import errors)

- **`src/models/`**: Pre-trained model management
  - ✅ `model_training.py`: Model training class (models already trained)
  - ✅ **Available Models**: Gradient Boosting (Best: 0.91% MAPE), Random Forest, Ensemble
  - ✅ **Model Files**: Stored in `data/models/` directory (ready to use)

- **`src/testing/`**: Static data validation systems
  - ✅ `validation_checker.py`: Data quality checks and validation
  - ⚠️ `realtime_testing_system.py`: Legacy (static mode uses existing results)
  - ✅ **Static Results**: Available in `data/raw/realtime_results.json`

- **`src/analysis/`**: Pre-generated analysis reports
  - ✅ **Performance Reports**: `data/raw/model_performance.json`
  - ✅ **Feature Analysis**: `data/raw/feature_analysis_enhanced.json`
  - ✅ **XAI Results**: SHAP/LIME analysis completed

- **`src/utils/`**: Static system orchestration
  - ✅ `system_orchestrator.py`: Static mode coordinator
  - ✅ **GPU Detection**: Automatic hardware detection
  - ✅ **Component Management**: Validates existing data and models

### Dashboard Architecture (Static Mode)

- ✅ **Static HTML Dashboard** (`dashboard/index.html`) - No server required
- ✅ **Self-contained Visualization** - All data embedded in JavaScript
- ✅ **3-Tab Analysis Interface**:
  - **Price Predictions**: SPY price vs AI model predictions
  - **Feature Impact**: SHAP-based feature importance analysis  
  - **News Impact**: Sentiment analysis and market impact correlation
- ✅ **Responsive Design** - Bootstrap 5 + Chart.js + FontAwesome
- ✅ **No Backend Dependencies** - Pure client-side application

### Static Data Pipeline

1. ✅ **Historical Data**: 2025 H1 SPY data (41 data points) pre-processed
2. ✅ **Feature Engineering**: Technical indicators, volatility, sentiment analysis completed
3. ✅ **Model Training**: 3 optimized models trained and evaluated
4. ✅ **Performance Analysis**: Comprehensive model comparison and validation
5. ✅ **Static Dashboard**: All results visualized in standalone HTML

### Key Data Paths

- `data/raw/`: Raw collected data and system status files
- `data/processed/`: Preprocessed datasets and enhanced features
- `data/models/`: Trained model files (`.pkl`, `.h5`)
- `results/`: Analysis results, visualizations, and reports
- `docs/reports/`: Generated comprehensive reports

### Technology Stack

- **Python**: Core ML pipeline (pandas, scikit-learn, TensorFlow, SHAP)
- **JavaScript**: Dashboard frontend with ES6+ modules
- **Data Sources**: yfinance for stock data, NewsAPI for sentiment analysis
- **Models**: Random Forest, Gradient Boosting, LSTM, XGBoost
- **Visualization**: Chart.js (frontend), matplotlib/plotly (backend)

### System Entry Points (Static Mode)

- ✅ **Static Analysis**: `PYTHONPATH=/root/workspace python3 src/utils/system_orchestrator.py`
- ✅ **Dashboard**: `cd dashboard && npm run dev` → `http://localhost:8080/index.html`
- ✅ **Model Training** (optional): `python3 src/models/model_training.py`
- ⚠️ **Legacy Real-time** (not needed): API-dependent components disabled

### Important Notes (Static Mode)

- ✅ **No API Keys Required** - System works entirely with existing data
- ✅ **GPU Detection** - Automatically handled by system orchestrator  
- ✅ **Pre-processed Data** - All analysis results available in `data/raw/`
- ✅ **Static Dashboard** - Self-contained HTML, no server dependencies
- ✅ **Model Performance** - Best model: Gradient Boosting (0.91% MAPE)
- ✅ **Data Validation** - Comprehensive quality checking included
- ⚠️ **API Warnings** - Expected and harmless (system works without APIs)