// Test fixtures for dashboard components
export const mockStockData = {
  sp500: [
    {
      ticker: 'AAPL',
      name: 'Apple Inc.',
      price: 150.25,
      change: 2.15,
      change_percent: 1.45,
      volume: 45623789,
      market_cap: 2500000000000
    },
    {
      ticker: 'MSFT',
      name: 'Microsoft Corporation',
      price: 305.80,
      change: -1.25,
      change_percent: -0.41,
      volume: 32145678,
      market_cap: 2300000000000
    },
    {
      ticker: 'GOOGL',
      name: 'Alphabet Inc.',
      price: 2750.45,
      change: 15.30,
      change_percent: 0.56,
      volume: 1234567,
      market_cap: 1800000000000
    }
  ]
};

export const mockChartData = {
  priceHistory: {
    labels: ['09:30', '10:00', '10:30', '11:00', '11:30', '12:00'],
    datasets: [{
      label: 'AAPL Price',
      data: [148.50, 149.25, 150.10, 149.80, 150.25, 150.50],
      borderColor: 'rgb(75, 192, 192)',
      backgroundColor: 'rgba(75, 192, 192, 0.2)',
      tension: 0.1
    }]
  },
  
  volumeData: {
    labels: ['09:30', '10:00', '10:30', '11:00', '11:30', '12:00'],
    datasets: [{
      label: 'Volume',
      data: [1200000, 980000, 1450000, 1100000, 890000, 1300000],
      backgroundColor: 'rgba(54, 162, 235, 0.5)',
      borderColor: 'rgba(54, 162, 235, 1)',
      borderWidth: 1
    }]
  },

  sectorPerformance: {
    labels: ['Technology', 'Healthcare', 'Finance', 'Energy', 'Consumer'],
    datasets: [{
      label: 'Sector Performance (%)',
      data: [2.3, 1.8, -0.5, -1.2, 0.8],
      backgroundColor: [
        'rgba(255, 99, 132, 0.5)',
        'rgba(54, 162, 235, 0.5)',
        'rgba(255, 205, 86, 0.5)',
        'rgba(75, 192, 192, 0.5)',
        'rgba(153, 102, 255, 0.5)'
      ]
    }]
  },

  candlestickData: {
    labels: ['Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5'],
    datasets: [{
      label: 'AAPL OHLC',
      data: [
        { open: 148.0, high: 150.5, low: 147.5, close: 149.8 },
        { open: 149.8, high: 151.2, low: 149.0, close: 150.5 },
        { open: 150.5, high: 152.0, low: 150.0, close: 151.3 },
        { open: 151.3, high: 151.8, low: 150.5, close: 150.9 },
        { open: 150.9, high: 152.5, low: 150.2, close: 152.1 }
      ]
    }]
  }
};

export const mockEventData = {
  recentEvents: [
    {
      id: 1,
      type: 'price_spike',
      ticker: 'AAPL',
      magnitude: 0.05,
      timestamp: '2024-01-15T14:30:00Z',
      confidence: 0.85,
      description: 'Significant price increase detected'
    },
    {
      id: 2,
      type: 'volume_spike',
      ticker: 'TSLA',
      magnitude: 2.3,
      timestamp: '2024-01-15T13:45:00Z',
      confidence: 0.92,
      description: 'Unusual trading volume detected'
    },
    {
      id: 3,
      type: 'volatility_spike',
      ticker: 'NVDA',
      magnitude: 1.8,
      timestamp: '2024-01-15T12:15:00Z',
      confidence: 0.78,
      description: 'High volatility period detected'
    }
  ]
};

export const mockModelMetrics = {
  random_forest: {
    accuracy: 0.78,
    precision: 0.82,
    recall: 0.75,
    f1_score: 0.78,
    auc_roc: 0.84,
    feature_importance: {
      'price_change': 0.35,
      'volume_change': 0.28,
      'rsi': 0.15,
      'macd': 0.12,
      'bollinger_bands': 0.10
    }
  },
  gradient_boosting: {
    accuracy: 0.81,
    precision: 0.84,
    recall: 0.79,
    f1_score: 0.81,
    auc_roc: 0.87,
    feature_importance: {
      'volume_change': 0.32,
      'price_change': 0.30,
      'volatility': 0.18,
      'rsi': 0.12,
      'macd': 0.08
    }
  },
  lstm: {
    accuracy: 0.76,
    precision: 0.80,
    recall: 0.73,
    f1_score: 0.76,
    auc_roc: 0.82,
    validation_loss: 0.45,
    training_loss: 0.38
  }
};

export const mockSystemStatus = {
  status: 'operational',
  models_loaded: ['random_forest', 'gradient_boosting', 'lstm'],
  data_freshness: '2024-01-15T15:30:00Z',
  active_tickers: 20,
  predictions_today: 1247,
  system_load: {
    cpu: 65,
    memory: 78,
    disk: 45
  },
  api_status: {
    yahoo_finance: 'online',
    news_api: 'online',
    internal_api: 'online'
  }
};

export const mockPredictions = {
  latest: [
    {
      ticker: 'AAPL',
      prediction: 1,
      confidence: 0.87,
      model: 'random_forest',
      timestamp: '2024-01-15T15:00:00Z',
      probability: 0.89
    },
    {
      ticker: 'MSFT',
      prediction: 0,
      confidence: 0.92,
      model: 'gradient_boosting',
      timestamp: '2024-01-15T15:00:00Z',
      probability: 0.08
    },
    {
      ticker: 'GOOGL',
      prediction: 1,
      confidence: 0.76,
      model: 'lstm',
      timestamp: '2024-01-15T15:00:00Z',
      probability: 0.76
    }
  ]
};

export const mockNewsData = {
  AAPL: {
    sentiment_score: 0.65,
    news_count: 12,
    latest_headlines: [
      'Apple reports strong quarterly earnings',
      'iPhone sales exceed expectations',
      'Apple stock reaches new high amid positive sentiment'
    ]
  },
  MSFT: {
    sentiment_score: 0.45,
    news_count: 8,
    latest_headlines: [
      'Microsoft Azure growth slows slightly',
      'Cloud competition intensifies',
      'Microsoft announces new AI initiatives'
    ]
  }
};

// Helper functions for generating dynamic test data
export const generateTimeSeriesData = (points = 30, startValue = 100) => {
  const data = [];
  let currentValue = startValue;
  
  for (let i = 0; i < points; i++) {
    currentValue += (Math.random() - 0.5) * 5;
    data.push({
      x: new Date(Date.now() - (points - i) * 60000).toISOString(),
      y: currentValue
    });
  }
  
  return data;
};

export const generateOHLCData = (points = 10, startPrice = 100) => {
  const data = [];
  let currentPrice = startPrice;
  
  for (let i = 0; i < points; i++) {
    const open = currentPrice;
    const close = open + (Math.random() - 0.5) * 5;
    const high = Math.max(open, close) + Math.random() * 3;
    const low = Math.min(open, close) - Math.random() * 3;
    
    data.push({ open, high, low, close });
    currentPrice = close;
  }
  
  return data;
};

export const generateVolumeData = (points = 30, baseVolume = 1000000) => {
  return Array.from({ length: points }, () => 
    Math.floor(baseVolume * (0.5 + Math.random()))
  );
};

// Error scenarios for testing
export const mockErrorScenarios = {
  networkError: {
    message: 'Network request failed',
    status: 0,
    type: 'network'
  },
  serverError: {
    message: 'Internal server error',
    status: 500,
    type: 'server'
  },
  notFoundError: {
    message: 'Resource not found',
    status: 404,
    type: 'client'
  },
  timeoutError: {
    message: 'Request timeout',
    status: 408,
    type: 'timeout'
  }
};