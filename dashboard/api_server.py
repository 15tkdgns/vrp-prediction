#!/usr/bin/env python3
"""
Flask API Server for Real-time Stock Data
Integrates with existing Python API system from src/core/api_config.py
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

# Add the parent directory to Python path to import from src/
parent_dir = Path(__file__).parent.parent.absolute()
sys.path.append(str(parent_dir))

try:
    from src.core.api_config import APIManager
    from src.testing.realtime_testing_system import RealTimeTestingSystem
    from src.models.model_training import SP500EventDetectionModel
except ImportError as e:
    print(f"Warning: Could not import Python API modules: {e}")
    print("Using fallback mode without real API integration")
    APIManager = None
    RealTimeTestingSystem = None
    SP500EventDetectionModel = None

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
API_PORT = int(os.getenv('API_PORT', 8091))
CACHE_TIMEOUT = 30  # seconds
UPDATE_INTERVAL = 60  # seconds for background updates

# Global variables for caching
data_cache = {}
cache_timestamps = {}
api_manager = None
realtime_system = None

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def init_api_systems():
    """Initialize API manager and realtime system"""
    global api_manager, realtime_system
    
    try:
        if APIManager:
            api_manager = APIManager()
            logger.info("✅ API Manager initialized")
        
        if RealTimeTestingSystem:
            realtime_system = RealTimeTestingSystem()
            logger.info("✅ Realtime Testing System initialized")
    except Exception as e:
        logger.error(f"❌ Failed to initialize API systems: {e}")

def is_cache_valid(key, timeout=CACHE_TIMEOUT):
    """Check if cached data is still valid"""
    if key not in cache_timestamps:
        return False
    
    age = (datetime.now() - cache_timestamps[key]).total_seconds()
    return age < timeout

def update_cache(key, data):
    """Update cache with new data"""
    data_cache[key] = data
    cache_timestamps[key] = datetime.now()

def get_mock_stock_data():
    """Fallback mock data for stock predictions"""
    return {
        "predictions": [
            {
                "symbol": "AAPL",
                "current_price": 230.48,
                "predicted_direction": "up",
                "confidence": 0.75,
                "risk_level": "low",
                "sector": "technology",
                "market_cap": "large",
                "technical_indicators": {
                    "rsi": 65.3,
                    "bb_position": 0.8,
                    "price_change": 0.024,
                    "volatility": 0.23
                }
            },
            {
                "symbol": "GOOGL",
                "current_price": 201.53,
                "predicted_direction": "up",
                "confidence": 0.68,
                "risk_level": "medium",
                "sector": "technology",
                "market_cap": "large",
                "technical_indicators": {
                    "rsi": 58.7,
                    "bb_position": 0.6,
                    "price_change": 0.018,
                    "volatility": 0.28
                }
            },
            {
                "symbol": "MSFT",
                "current_price": 509.71,
                "predicted_direction": "down",
                "confidence": 0.62,
                "risk_level": "low",
                "sector": "technology",
                "market_cap": "large",
                "technical_indicators": {
                    "rsi": 45.2,
                    "bb_position": 0.3,
                    "price_change": -0.012,
                    "volatility": 0.21
                }
            },
            {
                "symbol": "AMZN",
                "current_price": 227.74,
                "predicted_direction": "up",
                "confidence": 0.71,
                "risk_level": "medium",
                "sector": "consumer_discretionary",
                "market_cap": "large",
                "technical_indicators": {
                    "rsi": 72.1,
                    "bb_position": 0.85,
                    "price_change": 0.031,
                    "volatility": 0.35
                }
            }
        ],
        "market_summary": {
            "overall_sentiment": "positive",
            "volatility_index": 4.2,
            "trend": "upward",
            "confidence_level": 0.69
        },
        "timestamp": datetime.now().isoformat()
    }

def get_mock_sentiment_data():
    """Fallback mock data for market sentiment"""
    return {
        "sentiment_score": 0.15,
        "overall_sentiment": "positive",
        "confidence": 0.82,
        "news_count": 47,
        "timestamp": datetime.now().isoformat(),
        "details": {
            "positive_ratio": 0.45,
            "negative_ratio": 0.25,
            "neutral_ratio": 0.30
        }
    }

def get_mock_performance_data():
    """Fallback mock data for model performance"""
    return {
        "overall_metrics": {
            "accuracy": 0.847,
            "precision": 0.823,
            "recall": 0.861,
            "f1_score": 0.842
        },
        "model_comparison": {
            "random_forest": {"accuracy": 0.87, "precision": 0.84, "recall": 0.89, "f1_score": 0.86},
            "lstm": {"accuracy": 0.85, "precision": 0.82, "recall": 0.87, "f1_score": 0.84},
            "xgboost": {"accuracy": 0.89, "precision": 0.86, "recall": 0.91, "f1_score": 0.88},
            "gradient_boosting": {"accuracy": 0.84, "precision": 0.81, "recall": 0.83, "f1_score": 0.82}
        },
        "training_time": "2.3분",
        "last_updated": datetime.now().isoformat()
    }

def get_mock_volume_data():
    """Fallback mock data for trading volume"""
    return {
        "labels": ["월", "화", "수", "목", "금"],
        "datasets": [{
            "label": "거래량 (백만)",
            "data": [120, 150, 80, 200, 175],
            "backgroundColor": "#007bff"
        }],
        "timestamp": datetime.now().isoformat()
    }

# API Routes

@app.route('/api/stocks/live')
def get_live_stocks():
    """Get live stock data with predictions"""
    try:
        cache_key = 'live_stocks'
        
        # Check cache first
        if is_cache_valid(cache_key):
            logger.info("📋 Returning cached stock data")
            return jsonify(data_cache[cache_key])
        
        # Try to get real data
        if api_manager and realtime_system:
            try:
                # Get predictions from realtime system
                symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
                predictions = []
                
                for symbol in symbols[:4]:  # Top 4 stocks
                    try:
                        # Get market data
                        market_data = api_manager.get_market_data(symbol, period='1d', interval='5m')
                        
                        if market_data is not None and not market_data.empty:
                            current_price = float(market_data['Close'].iloc[-1])
                            price_change = float((market_data['Close'].iloc[-1] - market_data['Close'].iloc[-2]) / market_data['Close'].iloc[-2])
                            
                            # Mock prediction for now (can be enhanced with actual ML model)
                            prediction = {
                                "symbol": symbol,
                                "current_price": round(current_price, 2),
                                "predicted_direction": "up" if price_change > 0 else "down",
                                "confidence": round(abs(price_change) * 10 + 0.5, 2),  # Mock confidence
                                "risk_level": "low" if abs(price_change) < 0.02 else "medium",
                                "sector": "technology" if symbol in ['AAPL', 'GOOGL', 'MSFT'] else "consumer_discretionary",
                                "market_cap": "large",
                                "technical_indicators": {
                                    "rsi": round(50 + price_change * 100, 1),
                                    "bb_position": round(0.5 + price_change * 2, 3),
                                    "price_change": round(price_change, 3),
                                    "volatility": round(market_data['High'].iloc[-1] - market_data['Low'].iloc[-1], 2) / current_price
                                }
                            }
                            predictions.append(prediction)
                    except Exception as e:
                        logger.warning(f"Failed to get data for {symbol}: {e}")
                        continue
                
                if predictions:
                    result = {
                        "predictions": predictions,
                        "market_summary": {
                            "overall_sentiment": "neutral",
                            "volatility_index": 5.0,
                            "trend": "sideways",
                            "confidence_level": 0.65
                        },
                        "timestamp": datetime.now().isoformat(),
                        "source": "live_api"
                    }
                    update_cache(cache_key, result)
                    logger.info("✅ Live stock data fetched and cached")
                    return jsonify(result)
                    
            except Exception as e:
                logger.error(f"Error fetching live stock data: {e}")
        
        # Fallback to mock data
        result = get_mock_stock_data()
        result["source"] = "mock_data"
        update_cache(cache_key, result)
        logger.info("⚠️ Using mock stock data as fallback")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in get_live_stocks: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/news/sentiment')
def get_news_sentiment():
    """Get news sentiment analysis"""
    try:
        cache_key = 'news_sentiment'
        
        # Check cache first
        if is_cache_valid(cache_key):
            logger.info("📋 Returning cached sentiment data")
            return jsonify(data_cache[cache_key])
        
        # Try to get real data
        if api_manager:
            try:
                # Get news for major stocks
                all_news = []
                symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
                
                for symbol in symbols:
                    news_data = api_manager.get_news_data(symbol, limit=5)
                    all_news.extend(news_data)
                
                if all_news:
                    # Calculate overall sentiment
                    positive_count = sum(1 for news in all_news if news.get('sentiment_label') == 'positive')
                    negative_count = sum(1 for news in all_news if news.get('sentiment_label') == 'negative')
                    neutral_count = len(all_news) - positive_count - negative_count
                    
                    total_sentiment = sum(news.get('polarity', 0) for news in all_news) / len(all_news)
                    
                    result = {
                        "sentiment_score": round(total_sentiment, 3),
                        "overall_sentiment": "positive" if total_sentiment > 0.1 else "negative" if total_sentiment < -0.1 else "neutral",
                        "confidence": round(sum(news.get('sentiment_score', 0) for news in all_news) / len(all_news), 2),
                        "news_count": len(all_news),
                        "timestamp": datetime.now().isoformat(),
                        "details": {
                            "positive_ratio": round(positive_count / len(all_news), 2),
                            "negative_ratio": round(negative_count / len(all_news), 2),
                            "neutral_ratio": round(neutral_count / len(all_news), 2)
                        },
                        "source": "live_api"
                    }
                    
                    update_cache(cache_key, result)
                    logger.info("✅ Live sentiment data fetched and cached")
                    return jsonify(result)
                    
            except Exception as e:
                logger.error(f"Error fetching live sentiment data: {e}")
        
        # Fallback to mock data
        result = get_mock_sentiment_data()
        result["source"] = "mock_data"
        update_cache(cache_key, result)
        logger.info("⚠️ Using mock sentiment data as fallback")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in get_news_sentiment: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/models/performance')
def get_model_performance():
    """Get model performance metrics"""
    try:
        cache_key = 'model_performance'
        
        # Check cache first
        if is_cache_valid(cache_key, timeout=300):  # 5 minute cache
            logger.info("📋 Returning cached performance data")
            return jsonify(data_cache[cache_key])
        
        # Try to get real data from files
        try:
            performance_file = Path(__file__).parent.parent / 'data' / 'raw' / 'model_performance.json'
            if performance_file.exists():
                with open(performance_file, 'r') as f:
                    real_data = json.load(f)
                    real_data["source"] = "file_data"
                    real_data["timestamp"] = datetime.now().isoformat()
                    update_cache(cache_key, real_data)
                    logger.info("✅ Model performance data loaded from file")
                    return jsonify(real_data)
        except Exception as e:
            logger.warning(f"Could not load performance file: {e}")
        
        # Fallback to mock data
        result = get_mock_performance_data()
        result["source"] = "mock_data"
        update_cache(cache_key, result)
        logger.info("⚠️ Using mock performance data as fallback")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in get_model_performance: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/market/volume')
def get_market_volume():
    """Get market volume data"""
    try:
        cache_key = 'market_volume'
        
        # Check cache first
        if is_cache_valid(cache_key):
            logger.info("📋 Returning cached volume data")
            return jsonify(data_cache[cache_key])
        
        # Try to get real data
        if api_manager:
            try:
                # Get volume data for major symbols
                symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
                volume_data = []
                labels = []
                
                for symbol in symbols:
                    market_data = api_manager.get_market_data(symbol, period='5d', interval='1d')
                    if market_data is not None and not market_data.empty:
                        avg_volume = market_data['Volume'].mean() / 1000000  # Convert to millions
                        volume_data.append(round(avg_volume, 1))
                        labels.append(symbol)
                
                if volume_data:
                    result = {
                        "labels": labels,
                        "datasets": [{
                            "label": "평균 거래량 (백만)",
                            "data": volume_data,
                            "backgroundColor": "#007bff"
                        }],
                        "timestamp": datetime.now().isoformat(),
                        "source": "live_api"
                    }
                    
                    update_cache(cache_key, result)
                    logger.info("✅ Live volume data fetched and cached")
                    return jsonify(result)
                    
            except Exception as e:
                logger.error(f"Error fetching live volume data: {e}")
        
        # Fallback to mock data
        result = get_mock_volume_data()
        result["source"] = "mock_data"
        update_cache(cache_key, result)
        logger.info("⚠️ Using mock volume data as fallback")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in get_market_volume: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/status')
def get_api_status():
    """Get API system status"""
    return jsonify({
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "cache_info": {
            "cached_keys": list(data_cache.keys()),
            "cache_count": len(data_cache)
        },
        "api_systems": {
            "api_manager": api_manager is not None,
            "realtime_system": realtime_system is not None
        }
    })

@app.route('/api/cache/clear')
def clear_cache():
    """Clear all cached data"""
    global data_cache, cache_timestamps
    data_cache.clear()
    cache_timestamps.clear()
    logger.info("🧹 Cache cleared")
    return jsonify({"message": "Cache cleared successfully"})

# Serve static files
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)

def background_updater():
    """Background thread to update cache periodically"""
    while True:
        try:
            time.sleep(UPDATE_INTERVAL)
            logger.info("🔄 Background cache update starting...")
            
            # Update all endpoints
            with app.test_client() as client:
                client.get('/api/stocks/live')
                client.get('/api/news/sentiment')
                # Skip model performance as it changes less frequently
                client.get('/api/market/volume')
                
            logger.info("✅ Background cache update completed")
            
        except Exception as e:
            logger.error(f"❌ Background update failed: {e}")
            time.sleep(30)  # Wait 30 seconds before retry

if __name__ == '__main__':
    # Initialize API systems
    init_api_systems()
    
    # Start background updater
    updater_thread = threading.Thread(target=background_updater, daemon=True)
    updater_thread.start()
    
    logger.info(f"🚀 Flask API Server starting on port {API_PORT}")
    logger.info(f"📊 Dashboard available at http://localhost:{API_PORT}")
    logger.info(f"🔗 API endpoints: /api/stocks/live, /api/news/sentiment, /api/models/performance, /api/market/volume")
    
    try:
        app.run(
            host='0.0.0.0',
            port=API_PORT,
            debug=False,
            threaded=True
        )
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server error: {e}")