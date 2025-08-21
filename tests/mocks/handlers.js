// MSW API handlers for mocking dashboard API endpoints
import { http, HttpResponse } from 'msw';

// Mock data generators
const generateMockStockData = (ticker = 'AAPL') => ({
  ticker,
  price: 150.0 + Math.random() * 50,
  change: (Math.random() - 0.5) * 10,
  change_percent: (Math.random() - 0.5) * 5,
  volume: Math.floor(Math.random() * 10000000),
  market_cap: Math.floor(Math.random() * 1000000000000),
  timestamp: new Date().toISOString()
});

const generateMockChartData = (points = 30) => {
  const data = [];
  for (let i = 0; i < points; i++) {
    data.push({
      x: new Date(Date.now() - (points - i) * 60000).toISOString(),
      y: 100 + Math.random() * 100 + i * 0.5
    });
  }
  return data;
};

const generateMockEventData = () => ({
  events: [
    {
      id: 1,
      type: 'price_spike',
      ticker: 'AAPL',
      magnitude: 0.05,
      timestamp: new Date().toISOString(),
      confidence: 0.85
    },
    {
      id: 2,
      type: 'volume_spike',
      ticker: 'GOOGL',
      magnitude: 2.3,
      timestamp: new Date().toISOString(),
      confidence: 0.92
    }
  ]
});

const generateMockModelMetrics = () => ({
  random_forest: {
    accuracy: 0.78,
    precision: 0.82,
    recall: 0.75,
    f1_score: 0.78,
    auc_roc: 0.84
  },
  gradient_boosting: {
    accuracy: 0.81,
    precision: 0.84,
    recall: 0.79,
    f1_score: 0.81,
    auc_roc: 0.87
  },
  lstm: {
    accuracy: 0.76,
    precision: 0.80,
    recall: 0.73,
    f1_score: 0.76,
    auc_roc: 0.82
  }
});

// API handlers
export const handlers = [
  // Stock data endpoints
  http.get('/api/stock/:ticker', ({ params }) => {
    return HttpResponse.json(generateMockStockData(params.ticker));
  }),

  http.get('/api/stocks/sp500', () => {
    const tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'];
    return HttpResponse.json(tickers.map(ticker => generateMockStockData(ticker)));
  }),

  // Chart data endpoints
  http.get('/api/chart/:ticker/:timeframe', ({ params }) => {
    return HttpResponse.json({
      ticker: params.ticker,
      timeframe: params.timeframe,
      data: generateMockChartData()
    });
  }),

  http.get('/api/chart/overview', () => {
    return HttpResponse.json({
      sp500_index: generateMockChartData(),
      volume_overview: generateMockChartData(),
      sector_performance: {
        'Technology': 2.3,
        'Healthcare': 1.8,
        'Finance': -0.5,
        'Energy': -1.2,
        'Consumer': 0.8
      }
    });
  }),

  // Event detection endpoints
  http.get('/api/events/latest', () => {
    return HttpResponse.json(generateMockEventData());
  }),

  http.get('/api/events/:ticker', ({ params }) => {
    return HttpResponse.json({
      ticker: params.ticker,
      events: generateMockEventData().events.filter(e => e.ticker === params.ticker)
    });
  }),

  // Model performance endpoints
  http.get('/api/models/performance', () => {
    return HttpResponse.json(generateMockModelMetrics());
  }),

  http.get('/api/models/predictions', () => {
    return HttpResponse.json({
      predictions: [
        {
          ticker: 'AAPL',
          prediction: 1,
          confidence: 0.87,
          model: 'random_forest',
          timestamp: new Date().toISOString()
        },
        {
          ticker: 'MSFT',
          prediction: 0,
          confidence: 0.92,
          model: 'gradient_boosting',
          timestamp: new Date().toISOString()
        }
      ]
    });
  }),

  // Dashboard status endpoints
  http.get('/api/status', () => {
    return HttpResponse.json({
      status: 'operational',
      models_loaded: ['random_forest', 'gradient_boosting', 'lstm'],
      data_freshness: new Date().toISOString(),
      active_tickers: 20,
      predictions_today: 1247
    });
  }),

  // Real-time data endpoint
  http.get('/api/realtime/:ticker', ({ params }) => {
    return HttpResponse.json({
      ticker: params.ticker,
      price: 150.0 + Math.random() * 50,
      volume: Math.floor(Math.random() * 1000000),
      last_update: new Date().toISOString()
    });
  }),

  // News sentiment endpoint
  http.get('/api/news/:ticker', ({ params }) => {
    return HttpResponse.json({
      ticker: params.ticker,
      sentiment_score: Math.random() * 2 - 1, // -1 to 1
      news_count: Math.floor(Math.random() * 20),
      latest_headlines: [
        'Stock reaches new highs amid positive earnings',
        'Market volatility continues as investors await Fed decision',
        'Technology sector shows strong momentum'
      ]
    });
  }),

  // Error simulation endpoints
  http.get('/api/error/500', () => {
    return new HttpResponse(null, { status: 500 });
  }),

  http.get('/api/error/404', () => {
    return new HttpResponse(null, { status: 404 });
  }),

  http.get('/api/error/timeout', async () => {
    await new Promise(resolve => setTimeout(resolve, 10000));
    return HttpResponse.json({ message: 'This should timeout' });
  })
];

// Export individual generators for test-specific data
export {
  generateMockStockData,
  generateMockChartData,
  generateMockEventData,
  generateMockModelMetrics
};