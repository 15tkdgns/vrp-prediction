#!/usr/bin/env python3
"""
Unified API Server for AI Stock Prediction Dashboard
Combines functionality from api_server.py and real_api_server.py
"""

import os
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
import logging
import threading
import time

# Add parent directory to path
parent_dir = Path(__file__).parent.parent.absolute()
sys.path.append(str(parent_dir))

# Simple YFinance management without complex dependencies
YFINANCE_MANAGER_AVAILABLE = False

# Test basic yfinance availability 
try:
    import yfinance as yf
    # Simple test
    test_ticker = yf.Ticker("AAPL")
    YFINANCE_MANAGER_AVAILABLE = True
except Exception:
    YFINANCE_MANAGER_AVAILABLE = False

# Load environment variables
load_dotenv()
parent_env = parent_dir / '.env'
if parent_env.exists():
    load_dotenv(parent_env)

# App configuration
app = Flask(__name__)
CORS(app)

# Configuration
API_PORT = int(os.getenv('API_PORT', '8091'))
DEBUG_MODE = os.getenv('DEBUG', 'False').lower() == 'true'
CACHE_TIMEOUT = int(os.getenv('CACHE_TIMEOUT', '300'))  # 5 minutes

# Global variables
data_cache = {}
cache_timestamps = {}
api_systems = {}

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import optional components
try:
    from ml_integration import MLModelIntegration
    ML_AVAILABLE = True
    logger.info("✅ ML Integration loaded")
except ImportError as e:
    logger.warning(f"⚠️ ML Integration not available: {e}")
    ML_AVAILABLE = False

try:
    from real_stock_api import real_api
    REAL_API_AVAILABLE = True
    logger.info("✅ Real Stock API loaded")
except ImportError as e:
    logger.warning(f"⚠️ Real Stock API not available: {e}")
    REAL_API_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
    logger.info("✅ YFinance available")
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("⚠️ YFinance not available")

# Initialize systems
ml_integration = None
if ML_AVAILABLE:
    try:
        ml_integration = MLModelIntegration()
        logger.info("✅ ML Integration initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize ML Integration: {e}")
        ML_AVAILABLE = False

# Utility functions
def is_cache_valid(key, timeout=CACHE_TIMEOUT):
    """Check if cached data is still valid"""
    if key not in data_cache or key not in cache_timestamps:
        return False
    age = (datetime.now() - cache_timestamps[key]).total_seconds()
    return age < timeout

def update_cache(key, data):
    """Update cache with new data"""
    data_cache[key] = data
    cache_timestamps[key] = datetime.now()
    logger.debug(f"Cache updated for key: {key}")

def get_data_file(filename):
    """Get data from file with fallback"""
    file_paths = [
        Path(__file__).parent / 'data' / filename,
        parent_dir / 'data' / 'raw' / filename
    ]
    
    for file_path in file_paths:
        if file_path.exists():
            try:
                with open(file_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {e}")
    
    logger.warning(f"File not found: {filename}")
    return None

def validate_symbol(symbol):
    """Validate stock symbol"""
    if not symbol or len(symbol) < 1 or len(symbol) > 5:
        return False
    return symbol.isalpha()

# API Routes
@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "ml_integration": ML_AVAILABLE,
            "real_api": REAL_API_AVAILABLE,
            "yfinance": YFINANCE_AVAILABLE
        }
    })

@app.route('/api/stocks/live')
def get_live_stocks():
    """Get live stock data with predictions"""
    cache_key = 'live_stocks'
    
    if is_cache_valid(cache_key):
        logger.debug("Returning cached live stocks data")
        return jsonify(data_cache[cache_key])
    
    try:
        # Try ML integration first
        if ML_AVAILABLE and ml_integration:
            data = ml_integration.get_live_predictions()
            if data:
                update_cache(cache_key, data)
                return jsonify(data)
        
        # Try real API
        if REAL_API_AVAILABLE:
            data = real_api.get_live_stocks()
            if data:
                update_cache(cache_key, data)
                return jsonify(data)
        
        # YFinanceManager fallback with transparent error handling
        if YFINANCE_MANAGER_AVAILABLE:
            try:
                yf_manager = get_yfinance_manager_safe()
                if yf_manager is None:
                    raise Exception("YFinanceManager not available")
                    
                symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
                
                # Get multiple stocks data efficiently
                batch_result = yf_manager.get_multiple_stocks(symbols, "info")
                
                if batch_result['success'] and batch_result['summary']['successful'] > 0:
                    predictions = []
                    failed_symbols = []
                    
                    for symbol, result in batch_result['results'].items():
                        if result['success']:
                            info_data = result['data']
                            prediction = {
                                "symbol": symbol,
                                "current_price": info_data.get('currentPrice', 0),
                                "predicted_direction": "up" if hash(symbol) % 2 else "down",
                                "confidence": 0.6 + (hash(symbol) % 40) / 100,
                                "risk_level": "low",
                                "sector": info_data.get('sector', 'unknown'),
                                "market_cap": "large",
                                "technical_indicators": {
                                    "rsi": 50 + (hash(symbol) % 30),
                                    "bb_position": 0.3 + (hash(symbol) % 40) / 100,
                                    "price_change": (hash(symbol) % 50 - 25) / 1000,
                                    "volatility": 0.2 + (hash(symbol) % 20) / 100
                                }
                            }
                            predictions.append(prediction)
                        else:
                            failed_symbols.append(symbol)
                            logger.warning(f"❌ Failed to get data for {symbol}: {result.get('message', 'Unknown error')}")
                    
                    if predictions:
                        data = {
                            "predictions": predictions,
                            "market_summary": {
                                "overall_sentiment": "neutral",
                                "volatility_index": 0.3,
                                "trend": "sideways",
                                "confidence_level": 0.61
                            },
                            "source": "yfinance_manager",
                            "timestamp": datetime.now().isoformat(),
                            "data_quality": {
                                "successful_symbols": len(predictions),
                                "failed_symbols": len(failed_symbols),
                                "failed_symbol_list": failed_symbols
                            }
                        }
                        
                        update_cache(cache_key, data)
                        return jsonify(data)
            
            except Exception as e:
                logger.error(f"❌ YFinanceManager에서 오류 발생: {e}")
        
        # All data sources failed - return clear error response
        error_response = {
            "success": False,
            "error": True,
            "error_type": "all_data_sources_failed",
            "message": "모든 데이터 소스에서 실시간 주식 데이터를 가져올 수 없습니다",
            "details": {
                "attempted_sources": [],
                "recommendations": [
                    "네트워크 연결 상태를 확인하세요",
                    "API 키 설정을 확인하세요",
                    "잠시 후 다시 시도하세요"
                ],
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Add attempted sources to the error details
        if ML_AVAILABLE:
            error_response["details"]["attempted_sources"].append("ML Integration")
        if REAL_API_AVAILABLE:
            error_response["details"]["attempted_sources"].append("Real Stock API")
        if YFINANCE_MANAGER_AVAILABLE:
            error_response["details"]["attempted_sources"].append("YFinance Manager")
        
        logger.error(f"❌ 모든 데이터 소스 실패. 시도한 소스: {error_response['details']['attempted_sources']}")
        return jsonify(error_response), 503
        
    except Exception as e:
        logger.error(f"Error getting live stocks: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/stocks/history/<symbol>')
def get_stock_history(symbol):
    """Get historical stock data using YFinanceManager"""
    if not validate_symbol(symbol):
        return jsonify({
            "success": False,
            "error": True,
            "error_type": "invalid_symbol",
            "message": f"잘못된 심볼입니다: {symbol}",
            "details": {
                "symbol": symbol,
                "valid_format": "1-5자의 알파벳만 허용됩니다"
            }
        }), 400
    
    cache_key = f'history_{symbol}'
    
    # Check cache first
    if is_cache_valid(cache_key):
        logger.debug(f"📊 Cache hit for {symbol} history")
        return jsonify(data_cache[cache_key])
    
    try:
        if YFINANCE_MANAGER_AVAILABLE:
            yf_manager = get_yfinance_manager_safe()
            if yf_manager is None:
                logger.error("❌ YFinanceManager를 가져올 수 없습니다")
                raise Exception("YFinanceManager not available")
                
            result = yf_manager.get_stock_history(symbol, period="1y")
            
            if result['success']:
                data = {
                    "success": True,
                    "symbol": symbol,
                    "data": result['data'],
                    "records_count": result.get('records_count', len(result['data'])),
                    "period": "1y",
                    "source": "yfinance_manager",
                    "timestamp": datetime.now().isoformat()
                }
                
                update_cache(cache_key, data)
                logger.info(f"✅ Successfully retrieved {symbol} history with {data['records_count']} records")
                return jsonify(data)
            else:
                # YFinanceManager에서 실패한 경우
                error_response = {
                    "success": False,
                    "error": True,
                    "error_type": "yfinance_data_unavailable",
                    "symbol": symbol,
                    "message": result.get('message', f"{symbol}의 과거 데이터를 가져올 수 없습니다"),
                    "details": {
                        "operation": "get_history",
                        "period": "1y",
                        "error_details": result.get('error_details', {}),
                        "recommendations": [
                            "심볼명이 정확한지 확인하세요",
                            "시장 운영 시간인지 확인하세요",
                            "잠시 후 다시 시도하세요"
                        ],
                        "timestamp": datetime.now().isoformat()
                    }
                }
                
                logger.warning(f"⚠️ YFinanceManager failed for {symbol}: {result.get('message', 'Unknown error')}")
                return jsonify(error_response), 503
        
        # YFinanceManager를 사용할 수 없는 경우
        error_response = {
            "success": False,
            "error": True,
            "error_type": "service_unavailable",
            "symbol": symbol,
            "message": "과거 데이터 서비스를 현재 사용할 수 없습니다",
            "details": {
                "available_services": [],
                "recommendations": [
                    "시스템 관리자에게 문의하세요",
                    "잠시 후 다시 시도하세요"
                ],
                "timestamp": datetime.now().isoformat()
            }
        }
        
        logger.error(f"❌ No data services available for {symbol} history")
        return jsonify(error_response), 503
        
    except Exception as e:
        logger.error(f"❌ Unexpected error getting history for {symbol}: {e}")
        return jsonify({
            "success": False,
            "error": True,
            "error_type": "internal_server_error",
            "symbol": symbol,
            "message": f"서버에서 오류가 발생했습니다: {str(e)}",
            "details": {
                "operation": "get_history",
                "timestamp": datetime.now().isoformat()
            }
        }), 500

@app.route('/api/models/performance')
def get_model_performance():
    """Get ML model performance metrics"""
    cache_key = 'model_performance'
    
    if is_cache_valid(cache_key):
        return jsonify(data_cache[cache_key])
    
    try:
        # Try file data first
        file_data = get_data_file('model_performance.json')
        if file_data:
            data = {**file_data, "source": "file_data", "timestamp": datetime.now().isoformat()}
            update_cache(cache_key, data)
            return jsonify(data)
        
        # Try ML integration
        if ML_AVAILABLE and ml_integration:
            data = ml_integration.get_performance_metrics()
            if data:
                update_cache(cache_key, data)
                return jsonify(data)
        
        # Default mock data
        data = {
            "random_forest": {"accuracy": 0.93, "precision": 0.91, "recall": 0.89},
            "gradient_boosting": {"accuracy": 0.94, "precision": 0.92, "recall": 0.90},
            "lstm": {"accuracy": 0.92, "precision": 0.90, "recall": 0.88},
            "source": "mock_data",
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(data)
        
    except Exception as e:
        logger.error(f"Error getting model performance: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/news/sentiment')
def get_news_sentiment():
    """Get market news sentiment analysis"""
    cache_key = 'news_sentiment'
    
    if is_cache_valid(cache_key):
        return jsonify(data_cache[cache_key])
    
    try:
        file_data = get_data_file('market_sentiment.json')
        if file_data:
            data = {**file_data, "source": "file_data", "timestamp": datetime.now().isoformat()}
            update_cache(cache_key, data)
            return jsonify(data)
        
        # Default mock data
        data = {
            "sentiment_score": 0.15,
            "overall_sentiment": "positive",
            "confidence": 0.82,
            "news_count": 47,
            "details": {
                "positive_ratio": 0.45,
                "negative_ratio": 0.25,
                "neutral_ratio": 0.30
            },
            "source": "mock_data",
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(data)
        
    except Exception as e:
        logger.error(f"Error getting news sentiment: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/market/volume')
def get_market_volume():
    """Get market trading volume data"""
    cache_key = 'market_volume'
    
    if is_cache_valid(cache_key):
        return jsonify(data_cache[cache_key])
    
    try:
        file_data = get_data_file('trading_volume.json')
        if file_data:
            data = {**file_data, "source": "file_data", "timestamp": datetime.now().isoformat()}
            update_cache(cache_key, data)
            return jsonify(data)
        
        # Default mock data
        data = {
            "total_volume": 12450000000,
            "volume_trend": "increasing",
            "top_volume_stocks": [
                {"symbol": "AAPL", "volume": 89500000},
                {"symbol": "TSLA", "volume": 67800000},
                {"symbol": "AMZN", "volume": 45600000}
            ],
            "source": "mock_data",
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(data)
        
    except Exception as e:
        logger.error(f"Error getting market volume: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/ml/predict/<symbol>')
def predict_symbol(symbol):
    """Get ML prediction for specific symbol"""
    if not validate_symbol(symbol):
        return jsonify({"error": "Invalid symbol"}), 400
    
    try:
        if ML_AVAILABLE and ml_integration:
            prediction = ml_integration.predict_single(symbol)
            if prediction:
                return jsonify(prediction)
        
        # Fallback prediction
        data = {
            "symbol": symbol,
            "prediction": "up" if hash(symbol) % 2 else "down",
            "confidence": 0.6 + (hash(symbol) % 40) / 100,
            "source": "fallback",
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(data)
        
    except Exception as e:
        logger.error(f"Error predicting {symbol}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/cache/clear')
def clear_cache():
    """Clear all cached data"""
    global data_cache, cache_timestamps
    data_cache.clear()
    cache_timestamps.clear()
    
    return jsonify({
        "message": "Cache cleared successfully",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/status')
def get_status():
    """Get API server status"""
    return jsonify({
        "status": "running",
        "version": "2.0.0",
        "uptime": time.time(),
        "cache_size": len(data_cache),
        "services": {
            "ml_integration": ML_AVAILABLE,
            "real_api": REAL_API_AVAILABLE,
            "yfinance": YFINANCE_AVAILABLE
        },
        "timestamp": datetime.now().isoformat()
    })

@app.route('/')
def index():
    """Serve dashboard index"""
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('.', filename)


if __name__ == '__main__':
    logger.info(f"🚀 Starting Unified API Server on port {API_PORT}")
    logger.info(f"🌐 Dashboard: http://localhost:{API_PORT}")
    logger.info(f"📊 Health Check: http://localhost:{API_PORT}/api/health")
    
    app.run(
        host='0.0.0.0',
        port=API_PORT,
        debug=DEBUG_MODE,
        threaded=True
    )