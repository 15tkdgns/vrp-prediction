// API Data Loader - Real-time data fetching from actual APIs
class APIDataLoader {
  constructor() {
    this.dataCache = new Map();
    this.updateInterval = 60000; // 1 minute
    this.backendApiUrl = 'http://localhost:8090'; // Python backend API
    this.endpoints = {
      predictions: '/api/predictions',
      systemStatus: '/api/system/status',
      tradingVolume: '/api/market/volume',
      marketSentiment: '/api/market/sentiment',
      modelPerformance: '/api/model/performance',
      sp500Data: '/api/stocks/sp500',
      sp500Predictions: '/api/predictions/sp500',
    };

    this.init();
  }

  async init() {
    console.log('[API LOADER] Initializing real-time data loader...');
    await this.loadAllData();
    this.startAutoRefresh();
  }

  async loadAllData() {
    try {
      // Load predictions from Python backend
      const predictionsData = await this.fetchFromBackend('predictions');
      if (predictionsData) {
        this.dataCache.set('realtimeResults', predictionsData);
        window.realtimeResults = predictionsData;
      }

      // Load system status from Python backend
      const systemData = await this.fetchFromBackend('systemStatus');
      if (systemData) {
        this.dataCache.set('systemStatus', systemData);
        window.systemStatus = systemData;
      }

      // Load trading volume from Python backend or stock APIs
      const tradingVolumeData = await this.fetchFromBackend('tradingVolume');
      if (tradingVolumeData) {
        this.dataCache.set('tradingVolume', tradingVolumeData);
        window.tradingVolumeData = tradingVolumeData;
      }

      // Load market sentiment from Python backend
      const marketSentimentData =
        await this.fetchFromBackend('marketSentiment');
      if (marketSentimentData) {
        this.dataCache.set('marketSentiment', marketSentimentData);
        window.marketSentimentData = marketSentimentData;
      }

      // Load S&P 500 data from Python backend
      const sp500Data = await this.fetchFromBackend('sp500Data');
      if (sp500Data) {
        this.dataCache.set('sp500Data', sp500Data);
        window.sp500Data = sp500Data;
      }

      // Load S&P 500 predictions from Python backend
      const sp500Predictions = await this.fetchFromBackend('sp500Predictions');
      if (sp500Predictions) {
        this.dataCache.set('sp500Predictions', sp500Predictions);
        window.sp500Predictions = sp500Predictions;
      }

      console.log('[API LOADER] Real API data loading completed');

      // Notify dashboard of data update
      window.dispatchEvent(
        new CustomEvent('realDataLoaded', {
          detail: {
            predictions: predictionsData,
            systemStatus: systemData,
            tradingVolume: tradingVolumeData,
            marketSentiment: marketSentimentData,
            sp500Data: sp500Data,
            sp500Predictions: sp500Predictions,
          },
        })
      );
    } catch (error) {
      console.error('[API LOADER] Error loading real API data:', error);
      // Show no data instead of mock data
      this.showNoDataAvailable();
    }
  }

  async fetchFromBackend(endpoint) {
    try {
      const url = `${this.backendApiUrl}${this.endpoints[endpoint]}`;
      console.log(`[API LOADER] Fetching from: ${url}`);

      const response = await fetch(url, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
        timeout: 5000, // 5 second timeout
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      console.log(`[API LOADER] Successfully loaded ${endpoint} data`);
      return data;
    } catch (error) {
      console.warn(
        `[API LOADER] Failed to load ${endpoint} from backend:`,
        error.message
      );
      return null;
    }
  }

  showNoDataAvailable() {
    // Set all data to null to trigger no-data displays
    window.realtimeResults = null;
    window.systemStatus = null;
    window.tradingVolumeData = null;
    window.marketSentimentData = null;

    window.dispatchEvent(
      new CustomEvent('noDataAvailable', {
        detail: { message: 'Backend API not available' },
      })
    );
  }

  startAutoRefresh() {
    setInterval(() => {
      this.loadAllData();
    }, this.updateInterval);
  }

  // Get trading volume data for charts
  getTradingVolumeChartData() {
    const data = this.dataCache.get('tradingVolume');
    if (!data || !data.top_volume_stocks) return null;

    return {
      labels: data.top_volume_stocks.map((stock) => stock.symbol),
      datasets: [
        {
          label: 'Trading Volume (Millions)',
          data: data.top_volume_stocks.map((stock) => stock.volume / 1000000),
          backgroundColor: 'rgba(34, 197, 94, 0.8)',
          borderColor: 'rgba(34, 197, 94, 1)',
          borderWidth: 2,
        },
      ],
    };
  }

  // Get market sentiment data for charts
  getMarketSentimentChartData() {
    const data = this.dataCache.get('marketSentiment');
    if (!data || !data.news_analysis) return null;

    const analysis = data.news_analysis;
    return {
      labels: ['Positive', 'Negative', 'Neutral'],
      datasets: [
        {
          data: [
            analysis.positive_articles,
            analysis.negative_articles,
            analysis.neutral_articles,
          ],
          backgroundColor: [
            '#10B981', // Green for positive
            '#EF4444', // Red for negative
            '#6B7280', // Gray for neutral
          ],
        },
      ],
    };
  }

  // Get real-time predictions data
  getRealtimePredictions() {
    const data = this.dataCache.get('realtimeResults');
    return data?.predictions || [];
  }

  // Get system performance metrics
  getSystemMetrics() {
    const data = this.dataCache.get('systemStatus');
    return data?.system_health || {};
  }

  // Get S&P 500 historical data
  getSP500Data() {
    const data = this.dataCache.get('sp500Data');
    return data || null;
  }

  // Get S&P 500 predictions
  getSP500Predictions() {
    const data = this.dataCache.get('sp500Predictions');
    return data || null;
  }

  // Get S&P 500 chart data for Chart.js
  getSP500ChartData() {
    const data = this.getSP500Data();
    if (!data?.historical_data) return null;

    const historicalData = data.historical_data;
    return {
      labels: historicalData.map((item) =>
        new Date(item.date).toLocaleDateString()
      ),
      datasets: [
        {
          label: 'S&P 500 Price',
          data: historicalData.map((item) => item.close),
          borderColor: 'rgba(59, 130, 246, 1)',
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          borderWidth: 2,
          fill: true,
          tension: 0.1,
        },
      ],
    };
  }
}

// Initialize API data loader
if (typeof window !== 'undefined') {
  window.apiDataLoader = new APIDataLoader();
}
