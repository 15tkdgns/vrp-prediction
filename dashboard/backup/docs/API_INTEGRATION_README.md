# 📈 AI Stock Dashboard - Real API Data Integration

## 🚀 Overview

This implementation replaces mock/random data with **real-time API integration** using a multi-layered fallback system for maximum reliability.

## 🏗️ Architecture

```
Frontend Dashboard (Port 8080)
        ↓
JavaScript DataManager
        ↓
Flask API Server (Port 8091)
        ↓
Python API System (yFinance, NewsAPI, etc.)
        ↓
Real Stock & News Data
```

## 📁 Key Files

### Backend API Layer

- `api_server.py` - Flask API server with caching and error handling
- `requirements.txt` - Python dependencies for Flask API
- `.env` - Environment configuration (API keys, settings)

### Frontend Integration

- `js/data-manager.js` - Updated to call API endpoints with fallback
- `js/app.js` - Enhanced with real-time refresh and statistics

### Utilities

- `start-api-dashboard.sh` - Complete system startup script
- `stop-servers.sh` - Clean server shutdown script

## 🔌 API Endpoints

### 1. Live Stock Data

```
GET /api/stocks/live
```

**Response:**

```json
{
  "predictions": [
    {
      "symbol": "AAPL",
      "current_price": 230.48,
      "predicted_direction": "up",
      "confidence": 0.75,
      "technical_indicators": {
        "rsi": 65.3,
        "price_change": 0.024,
        "volatility": 0.23
      }
    }
  ],
  "source": "live_api",
  "timestamp": "2025-08-20T18:16:13.285747"
}
```

### 2. News Sentiment Analysis

```
GET /api/news/sentiment
```

**Response:**

```json
{
  "sentiment_score": 0.15,
  "overall_sentiment": "positive",
  "confidence": 0.82,
  "news_count": 47,
  "details": {
    "positive_ratio": 0.45,
    "negative_ratio": 0.25,
    "neutral_ratio": 0.3
  },
  "source": "live_api",
  "timestamp": "2025-08-20T18:16:13.285747"
}
```

### 3. Model Performance

```
GET /api/models/performance
```

### 4. Market Volume

```
GET /api/market/volume
```

### 5. API Status

```
GET /api/status
```

## 🛡️ Fallback System

The system uses a **3-tier fallback approach**:

1. **Primary**: Flask API with live data sources
2. **Secondary**: Existing JSON files (../data/raw/\*.json)
3. **Tertiary**: Mock data (always available)

## ⚙️ Configuration

### Environment Variables (.env)

```bash
# Flask API Server
API_PORT=8091

# Stock Data API Keys (optional - uses free sources if not set)
ALPHA_VANTAGE_KEY=your_key_here
POLYGON_KEY=your_key_here
IEX_CLOUD_KEY=your_key_here

# Cache Settings
CACHE_TIMEOUT=30
UPDATE_INTERVAL=60
```

### Frontend Settings (localStorage)

- `refresh_interval`: Auto-refresh interval in seconds (default: 60)
- `refresh_stats`: Tracks success/failure rates and performance

## 🚀 Quick Start

### Method 1: Automated Script (Recommended)

```bash
cd /root/workspace/dashboard
./start-api-dashboard.sh
```

### Method 2: Manual Setup

```bash
# Terminal 1: Start Flask API Server
cd /root/workspace && source venv/bin/activate
cd dashboard && python api_server.py

# Terminal 2: Start Frontend Dashboard
cd /root/workspace/dashboard
npm run dev
```

## 📊 Real-time Features

### Smart Auto-Refresh

- **Adaptive Timing**: Configurable refresh intervals
- **Visibility Detection**: Pauses when page not visible
- **Focus Resume**: Immediate refresh when page regains focus
- **Statistics Tracking**: Success rates and performance metrics

### Error Handling

- **Exponential Backoff**: Retry failed API calls with increasing delays
- **Circuit Breaker**: Falls back to static data on repeated failures
- **User Feedback**: Real-time status updates and progress indicators

## 🔍 Monitoring & Debugging

### Log Files

```bash
# API Server Logs
tail -f api_server.log

# Frontend Logs
tail -f frontend.log

# Browser Console
# Shows detailed API call logs and fallback information
```

### Health Checks

```bash
# API Server Status
curl http://localhost:8091/api/status

# Frontend Accessibility
curl http://localhost:8080
```

## 🚦 Data Sources Integration

### Stock Data

- **Primary**: yFinance (free, reliable)
- **Backup**: Polygon.io, Alpha Vantage, IEX Cloud
- **Features**: Real-time prices, technical indicators, predictions

### News & Sentiment

- **Primary**: Yahoo Finance RSS feeds
- **Backup**: Marketaux API, NewsData.io
- **Features**: Sentiment analysis with TextBlob, confidence scores

### Model Performance

- **Source**: Existing model training results
- **Features**: Multi-model comparison, real-time metrics

## ⚡ Performance Optimization

### Caching Strategy

- **API Level**: 30-second cache for expensive operations
- **Frontend Level**: Component-level caching with timestamps
- **Background Updates**: Periodic cache refresh in background thread

### Request Optimization

- **Parallel Loading**: Multiple API calls executed concurrently
- **Smart Polling**: Adjusts frequency based on data freshness
- **Connection Pooling**: Reuses HTTP connections for efficiency

## 🔐 Security Considerations

### API Key Management

- Store keys in `.env` file (never commit to git)
- Graceful fallback when keys are missing
- Rate limiting and abuse prevention

### CORS & Headers

- Proper CORS configuration for cross-origin requests
- Security headers for API responses
- Input validation and sanitization

## 🛠️ Customization

### Adding New Data Sources

1. Extend `api_server.py` with new endpoint
2. Update `js/data-manager.js` to call new endpoint
3. Add fallback data in mock methods

### Modifying Refresh Intervals

```javascript
// In browser console or localStorage
localStorage.setItem('refresh_interval', '30'); // 30 seconds
```

### API Key Configuration

```bash
# Edit .env file
ALPHA_VANTAGE_KEY=your_actual_key_here
POLYGON_KEY=your_actual_key_here
```

## 📈 Future Enhancements

### WebSocket Integration

- Real-time streaming data updates
- Reduced server load with push notifications
- Instant price change notifications

### Advanced Caching

- Redis integration for distributed caching
- Cache invalidation strategies
- Partial data updates

### API Rate Limiting

- Intelligent request throttling
- Priority-based API usage
- Cost optimization for paid APIs

## 🐛 Troubleshooting

### Common Issues

**API Server Won't Start**

```bash
# Check if port is in use
lsof -i :8091
# Kill conflicting process
./stop-servers.sh
```

**No Real Data Showing**

```bash
# Check API server logs
cat api_server.log
# Verify API keys in .env
cat .env
```

**Frontend Not Loading**

```bash
# Check npm dependencies
npm install
# Restart frontend server
npm run dev
```

## 📞 Support

For issues or questions:

1. Check browser console for detailed error logs
2. Review `api_server.log` for backend issues
3. Verify `.env` configuration
4. Test individual API endpoints with curl

---

✅ **System Status**: Fully integrated with real API data sources
🔄 **Auto-refresh**: Intelligent, configurable, and efficient
🛡️ **Reliability**: Multi-tier fallback ensures 100% uptime
📊 **Performance**: Optimized caching and request batching
