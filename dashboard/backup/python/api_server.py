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

# API 연동 시스템 import
try:
    from real_stock_api import real_api
    print("✅ 실제 주식 API 연동 활성화됨")
    REAL_API_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import real stock API: {e}")
    print("Using fallback mode with mock data")
    real_api = None
    REAL_API_AVAILABLE = False

# Skip problematic imports for now - focus on ML integration
API_MANAGER_AVAILABLE = False
print("⚠️ APIManager 이시지 생략 (torch 설치 중)")

# ML Integration System
try:
    from ml_integration import MLModelIntegration
    ML_INTEGRATION_AVAILABLE = True
    print("✅ ML Integration 클래스 로드 성공")
except ImportError as e:
    print(f"Warning: Could not import MLModelIntegration: {e}")
    ML_INTEGRATION_AVAILABLE = False

# yfinance fallback
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    print("Warning: yfinance not available")
    YFINANCE_AVAILABLE = False

# RSS and sentiment analysis fallback
try:
    import feedparser
    from textblob import TextBlob
    import requests
    RSS_AVAILABLE = True
    print("✅ RSS 및 감성 분석 라이브러리 로드 성공")
except ImportError as e:
    print(f"Warning: RSS/TextBlob not available: {e}")
    RSS_AVAILABLE = False

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
API_PORT = int(os.getenv('API_PORT', 8091))
CACHE_TIMEOUT = 30  # seconds
UPDATE_INTERVAL = 60  # seconds for background updates

# Global variables for caching and API systems
data_cache = {}
cache_timestamps = {}
real_api_instance = real_api
api_manager = None
ml_integration = None

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def init_api_systems():
    """Initialize all API systems"""
    global real_api_instance, api_manager, ml_integration
    
    # Skip API Manager for now - focus on ML integration
    api_manager = None
    logger.info("⚠️ API Manager 건너뛰기 - ML Integration 우선")
    
    # Initialize ML Integration System
    if ML_INTEGRATION_AVAILABLE:
        try:
            ml_integration = MLModelIntegration()
            status = ml_integration.get_model_status()
            logger.info(f"✅ ML Integration 초기화 완료: {status['model_count']}개 모델 로드")
        except Exception as e:
            logger.error(f"❌ ML Integration 초기화 실패: {e}")
            ml_integration = None
    
    # Initialize Real API
    if REAL_API_AVAILABLE and real_api_instance:
        logger.info("✅ 실제 주식 API 연동 준비 완료")
    else:
        logger.info("⚠️ Mock 데이터 모드로 동작")

def validate_api_keys():
    """Validate environment variables and API keys"""
    required_keys = ['ALPHA_VANTAGE_KEY', 'POLYGON_KEY', 'MARKETAUX_KEY']
    missing_keys = []
    placeholder_keys = []
    
    for key in required_keys:
        value = os.getenv(key)
        if not value:
            missing_keys.append(key)
        elif value.startswith('your_') or value == 'demo':
            placeholder_keys.append(key)
    
    if missing_keys:
        logger.warning(f"⚠️ 누락된 API 키: {missing_keys}")
    if placeholder_keys:
        logger.warning(f"⚠️ 플레이스홀더 API 키: {placeholder_keys}")
    
    return len(missing_keys) == 0 and len(placeholder_keys) == 0

def get_rss_news_with_sentiment():
    """주식 관련 RSS 뉴스를 가져와서 감성 분석 수행"""
    if not RSS_AVAILABLE:
        return []
    
    news_feeds = [
        'https://feeds.finance.yahoo.com/rss/2.0/headline',
        'https://feeds.bloomberg.com/markets/news.rss',
        'https://www.cnbc.com/id/100003114/device/rss/rss.html',
    ]
    
    all_news = []
    
    for feed_url in news_feeds:
        try:
            # Parse RSS feed
            feed = feedparser.parse(feed_url)
            
            for entry in feed.entries[:5]:  # Top 5 from each feed
                try:
                    # Get title and description
                    title = entry.get('title', '')
                    description = entry.get('summary', entry.get('description', ''))
                    text = f"{title}. {description}"
                    
                    # Perform sentiment analysis
                    blob = TextBlob(text)
                    sentiment_score = blob.sentiment.polarity
                    subjectivity = blob.sentiment.subjectivity
                    
                    # Determine sentiment label
                    if sentiment_score > 0.1:
                        sentiment_label = 'positive'
                    elif sentiment_score < -0.1:
                        sentiment_label = 'negative'
                    else:
                        sentiment_label = 'neutral'
                    
                    news_item = {
                        'title': title[:100],  # Limit title length
                        'sentiment_score': round(sentiment_score, 3),
                        'sentiment_label': sentiment_label,
                        'polarity': sentiment_score,
                        'subjectivity': round(subjectivity, 3),
                        'published': entry.get('published', ''),
                        'source': feed_url.split('/')[2]  # Extract domain
                    }
                    all_news.append(news_item)
                    
                except Exception as e:
                    logger.warning(f"Error processing news entry: {e}")
                    continue
                    
        except Exception as e:
            logger.warning(f"Error fetching RSS from {feed_url}: {e}")
            continue
    
    return all_news

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
        
        # Try to get real data with yfinance fallback
        symbols = ['^GSPC', 'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']  # S&P 500 지수 추가
        predictions = []
        
        for symbol in symbols[:5]:  # Top 5 including S&P 500
            try:
                # Try API manager first, then fallback to yfinance
                market_data = None
                
                if api_manager:
                    try:
                        # Try to get data from API manager if available
                        if hasattr(api_manager, 'get_market_data'):
                            market_data = api_manager.get_market_data(symbol, period='1d', interval='5m')
                    except Exception as e:
                        logger.warning(f"API Manager failed for {symbol}: {e}")
                
                # Fallback to yfinance
                if market_data is None or (hasattr(market_data, 'empty') and market_data.empty):
                    if YFINANCE_AVAILABLE:
                        try:
                            ticker = yf.Ticker(symbol)
                            market_data = ticker.history(period='1d', interval='5m')
                            logger.info(f"✅ yfinance data fetched for {symbol}")
                        except Exception as e:
                            logger.warning(f"yfinance failed for {symbol}: {e}")
                            continue
                    else:
                        logger.warning(f"No data source available for {symbol}")
                        continue
                
                if market_data is not None and not market_data.empty:
                    current_price = float(market_data['Close'].iloc[-1])
                    if len(market_data) > 1:
                        price_change = float((market_data['Close'].iloc[-1] - market_data['Close'].iloc[-2]) / market_data['Close'].iloc[-2])
                    else:
                        price_change = 0.0
                            
                    # Enhanced prediction with technical indicators
                    volatility = 0
                    if len(market_data) > 0:
                        high_price = float(market_data['High'].iloc[-1])
                        low_price = float(market_data['Low'].iloc[-1])
                        volatility = (high_price - low_price) / current_price
                    
                    # Determine sector
                    sectors = {
                        '^GSPC': 'market_index',
                        'AAPL': 'technology', 'GOOGL': 'technology', 'MSFT': 'technology',
                        'AMZN': 'consumer_discretionary', 'TSLA': 'automotive'
                    }
                    
                    prediction = {
                        "symbol": symbol,
                        "current_price": round(current_price, 2),
                        "predicted_direction": "up" if price_change > 0 else "down",
                        "confidence": min(round(abs(price_change) * 10 + 0.6, 2), 0.95),
                        "risk_level": "low" if abs(price_change) < 0.02 else "medium" if abs(price_change) < 0.05 else "high",
                        "sector": sectors.get(symbol, "other"),
                        "market_cap": "large",
                        "technical_indicators": {
                            "rsi": round(min(max(50 + price_change * 200, 20), 80), 1),
                            "bb_position": round(min(max(0.5 + price_change * 3, 0), 1), 3),
                            "price_change": round(price_change, 4),
                            "volatility": round(volatility, 4)
                        }
                    }
                    predictions.append(prediction)
            except Exception as e:
                logger.warning(f"Failed to get data for {symbol}: {e}")
                continue
                
        # Generate result if we have predictions
        if predictions:
            # Calculate market summary based on actual data
            avg_change = sum(p['technical_indicators']['price_change'] for p in predictions) / len(predictions)
            avg_volatility = sum(p['technical_indicators']['volatility'] for p in predictions) / len(predictions)
            
            result = {
                "predictions": predictions,
                "market_summary": {
                    "overall_sentiment": "positive" if avg_change > 0.01 else "negative" if avg_change < -0.01 else "neutral",
                    "volatility_index": round(avg_volatility * 100, 1),
                    "trend": "upward" if avg_change > 0.005 else "downward" if avg_change < -0.005 else "sideways",
                    "confidence_level": round(min(sum(p['confidence'] for p in predictions) / len(predictions), 0.9), 2)
                },
                "timestamp": datetime.now().isoformat(),
                "source": "yfinance" if YFINANCE_AVAILABLE else "live_api"
            }
            update_cache(cache_key, result)
            logger.info(f"✅ Live stock data fetched for {len(predictions)} symbols")
            return jsonify(result)
        
        # Ultimate fallback to mock data
        logger.warning("⚠️ No real data available, using mock data")
        result = get_mock_stock_data()
        result["source"] = "mock_data"
        update_cache(cache_key, result)
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
        
        # Try multiple data sources
        all_news = []
        
        # Try API Manager first
        if api_manager:
            try:
                symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
                
                for symbol in symbols:
                    if hasattr(api_manager, 'get_news_data_marketaux'):
                        news_data = api_manager.get_news_data_marketaux(symbol, limit=3)
                        if news_data:
                            all_news.extend(news_data)
                            
            except Exception as e:
                logger.warning(f"API Manager news failed: {e}")
        
        # Fallback to RSS feeds
        if not all_news and RSS_AVAILABLE:
            try:
                all_news = get_rss_news_with_sentiment()
                logger.info(f"✅ RSS news fetched: {len(all_news)} articles")
            except Exception as e:
                logger.warning(f"RSS news failed: {e}")
        
        # Process sentiment data if we have news
        if all_news:
            try:
                # Calculate overall sentiment
                positive_count = sum(1 for news in all_news if news.get('sentiment_label') == 'positive')
                negative_count = sum(1 for news in all_news if news.get('sentiment_label') == 'negative')
                neutral_count = len(all_news) - positive_count - negative_count
                
                total_sentiment = sum(news.get('polarity', 0) for news in all_news) / len(all_news)
                avg_confidence = sum(abs(news.get('sentiment_score', 0)) for news in all_news) / len(all_news)
                
                result = {
                    "sentiment_score": round(total_sentiment, 3),
                    "overall_sentiment": "positive" if total_sentiment > 0.05 else "negative" if total_sentiment < -0.05 else "neutral",
                    "confidence": round(avg_confidence, 2),
                    "news_count": len(all_news),
                    "timestamp": datetime.now().isoformat(),
                    "details": {
                        "positive_ratio": round(positive_count / len(all_news), 2),
                        "negative_ratio": round(negative_count / len(all_news), 2),
                        "neutral_ratio": round(neutral_count / len(all_news), 2)
                    },
                    "sample_headlines": [news.get('title', '')[:80] for news in all_news[:3]],
                    "source": "rss_feeds" if RSS_AVAILABLE else "api_manager"
                }
                
                update_cache(cache_key, result)
                logger.info(f"✅ Sentiment analysis completed for {len(all_news)} articles")
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"Error processing sentiment data: {e}")
        
        # Ultimate fallback to mock data
        logger.warning("⚠️ No real sentiment data available, using mock data")
        result = get_mock_sentiment_data()
        result["source"] = "mock_data"
        update_cache(cache_key, result)
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
        
        # Try multiple data sources for volume data
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
        volume_data = []
        labels = []
        
        for symbol in symbols:
            try:
                market_data = None
                
                # Try API manager first
                if api_manager:
                    try:
                        if hasattr(api_manager, 'get_market_data'):
                            market_data = api_manager.get_market_data(symbol, period='5d', interval='1d')
                    except Exception as e:
                        logger.warning(f"API Manager volume failed for {symbol}: {e}")
                
                # Fallback to yfinance
                if market_data is None or (hasattr(market_data, 'empty') and market_data.empty):
                    if YFINANCE_AVAILABLE:
                        try:
                            ticker = yf.Ticker(symbol)
                            market_data = ticker.history(period='5d', interval='1d')
                        except Exception as e:
                            logger.warning(f"yfinance volume failed for {symbol}: {e}")
                            continue
                    else:
                        continue
                
                # Process volume data
                if market_data is not None and not market_data.empty and 'Volume' in market_data.columns:
                    # Calculate average volume over the period
                    avg_volume = market_data['Volume'].mean() / 1000000  # Convert to millions
                    volume_data.append(round(avg_volume, 1))
                    labels.append(symbol)
                    
            except Exception as e:
                logger.warning(f"Failed to get volume data for {symbol}: {e}")
                continue
        
        # Generate result if we have volume data
        if volume_data and labels:
            # Add some color variety
            colors = ['#007bff', '#28a745', '#ffc107', '#dc3545', '#17a2b8']
            
            result = {
                "labels": labels,
                "datasets": [{
                    "label": "평균 거래량 (백만)",
                    "data": volume_data,
                    "backgroundColor": colors[:len(volume_data)],
                    "borderColor": colors[:len(volume_data)],
                    "borderWidth": 1
                }],
                "summary": {
                    "total_symbols": len(labels),
                    "avg_volume": round(sum(volume_data) / len(volume_data), 1),
                    "highest_volume": {"symbol": labels[volume_data.index(max(volume_data))], "volume": max(volume_data)},
                    "lowest_volume": {"symbol": labels[volume_data.index(min(volume_data))], "volume": min(volume_data)}
                },
                "timestamp": datetime.now().isoformat(),
                "source": "yfinance" if YFINANCE_AVAILABLE else "api_manager"
            }
            
            update_cache(cache_key, result)
            logger.info(f"✅ Volume data fetched for {len(labels)} symbols")
            return jsonify(result)
        
        # Ultimate fallback to mock data
        logger.warning("⚠️ No real volume data available, using mock data")
        result = get_mock_volume_data()
        result["source"] = "mock_data"
        update_cache(cache_key, result)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in get_market_volume: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/status')
def get_api_status():
    """Get API system status"""
    ml_status = None
    if ml_integration:
        ml_status = ml_integration.get_model_status()
    
    return jsonify({
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "cache_info": {
            "cached_keys": list(data_cache.keys()),
            "cache_count": len(data_cache)
        },
        "api_systems": {
            "api_manager": api_manager is not None,
            "ml_integration": ml_integration is not None,
            "ml_models": ml_status
        }
    })

@app.route('/api/ml/predict/<symbol>')
def get_ml_prediction(symbol):
    """Get ML model prediction for specific symbol"""
    try:
        if not ml_integration:
            return jsonify({"error": "ML integration not available"}), 503
        
        result = ml_integration.predict_event(symbol.upper())
        
        if "error" in result:
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in ML prediction for {symbol}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/ml/batch_predict')
def get_batch_predictions():
    """Get ML predictions for multiple symbols"""
    try:
        if not ml_integration:
            return jsonify({"error": "ML integration not available"}), 503
        
        symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
        predictions = {}
        
        for symbol in symbols:
            try:
                result = ml_integration.predict_event(symbol)
                predictions[symbol] = result
            except Exception as e:
                predictions[symbol] = {"error": str(e)}
        
        return jsonify({
            "batch_predictions": predictions,
            "timestamp": datetime.now().isoformat(),
            "total_symbols": len(symbols)
        })
        
    except Exception as e:
        logger.error(f"Error in batch ML prediction: {e}")
        return jsonify({"error": str(e)}), 500

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
    # Validate environment variables
    api_keys_valid = validate_api_keys()
    if not api_keys_valid:
        logger.warning("⚠️ 일부 API 키가 누락되었지만 서버를 시작합니다.")
    
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