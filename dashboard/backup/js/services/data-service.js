/**
 * Refactored Data Service
 * 데이터 관리 로직을 서비스 레이어로 분리하여 유지보수성 향상
 */

import { apiClient } from '../core/api-client.js';
import { logger } from '../core/logger.js';
import { eventBus, EVENTS } from '../core/event-bus.js';

export class DataService {
  constructor() {
    this.data = {
      stocks: [],
      news: [],
      metrics: {},
      charts: {},
      mlPredictions: {}
    };

    this.loadingStates = new Map();
    this.subscribers = new Map();

    logger.info('DataService initialized');
  }

  /**
   * 데이터 구독 시스템
   */
  subscribe(dataType, callback) {
    if (!this.subscribers.has(dataType)) {
      this.subscribers.set(dataType, new Set());
    }
    
    this.subscribers.get(dataType).add(callback);
    
    // 구독 해제 함수 반환
    return () => {
      this.subscribers.get(dataType)?.delete(callback);
    };
  }

  /**
   * 구독자들에게 데이터 변경 알림
   */
  notify(dataType, data) {
    const subscribers = this.subscribers.get(dataType);
    if (subscribers) {
      subscribers.forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          logger.error(`Subscriber error for ${dataType}:`, error);
        }
      });
    }
    
    eventBus.emit(EVENTS.DATA_UPDATED, { type: dataType, data });
  }

  /**
   * 로딩 상태 관리
   */
  setLoadingState(key, isLoading) {
    this.loadingStates.set(key, isLoading);
    
    if (isLoading) {
      eventBus.emit(EVENTS.DATA_LOADING_START, { key });
    } else {
      eventBus.emit(EVENTS.DATA_LOADING_END, { key });
    }
  }

  isLoading(key) {
    return this.loadingStates.get(key) || false;
  }

  /**
   * 주식 데이터 로드
   */
  async loadStockData() {
    const key = 'stocks';
    
    if (this.isLoading(key)) {
      logger.debug('Stock data already loading, skipping...');
      return this.data.stocks;
    }

    try {
      this.setLoadingState(key, true);
      logger.startPerformance('loadStockData');

      // 로컬 API 우선 시도
      const data = await apiClient.get('/api/stocks/live', {
        timeout: 2000,
        useCache: true
      });

      if (data?.predictions?.length) {
        this.data.stocks = data.predictions;
        logger.info(`✅ Stock data loaded: ${data.predictions.length} items`);
      } else {
        // 파일 폴백
        const fallbackData = await this.loadFallbackData('stocks');
        this.data.stocks = fallbackData;
      }

      this.notify('stocks', this.data.stocks);
      return this.data.stocks;

    } catch (error) {
      logger.error('Failed to load stock data', { error: error.message });
      
      // 에러 시 폴백 데이터 시도
      try {
        const fallbackData = await this.loadFallbackData('stocks');
        this.data.stocks = fallbackData;
        this.notify('stocks', this.data.stocks);
        return this.data.stocks;
      } catch (fallbackError) {
        logger.error('Fallback data also failed', { error: fallbackError.message });
        throw error;
      }

    } finally {
      this.setLoadingState(key, false);
      logger.endPerformance('loadStockData');
    }
  }

  /**
   * ML 예측 데이터 로드
   */
  async loadMLPredictions(symbols = ['AAPL', 'GOOGL', 'MSFT']) {
    const key = 'ml_predictions';
    
    try {
      this.setLoadingState(key, true);
      logger.startPerformance('loadMLPredictions');

      // 배치 예측 API 호출
      const data = await apiClient.get('/api/ml/batch_predict', {
        params: { symbols: symbols.join(',') },
        useCache: true
      });

      this.data.mlPredictions = data;
      logger.info(`✅ ML predictions loaded for ${Object.keys(data).length} symbols`);
      
      this.notify('mlPredictions', this.data.mlPredictions);
      return this.data.mlPredictions;

    } catch (error) {
      logger.error('Failed to load ML predictions', { error: error.message, symbols });
      throw error;

    } finally {
      this.setLoadingState(key, false);
      logger.endPerformance('loadMLPredictions');
    }
  }

  /**
   * 뉴스 및 감정 데이터 로드
   */
  async loadNewsData() {
    const key = 'news';
    
    try {
      this.setLoadingState(key, true);
      
      const data = await apiClient.get('/api/news/sentiment', { useCache: true });
      this.data.news = data;
      
      this.notify('news', this.data.news);
      return this.data.news;

    } catch (error) {
      logger.warn('Failed to load news data', { error: error.message });
      // 뉴스는 필수가 아니므로 에러를 던지지 않음
      return [];

    } finally {
      this.setLoadingState(key, false);
    }
  }

  /**
   * 메트릭 데이터 로드
   */
  async loadMetricsData() {
    const key = 'metrics';
    
    try {
      this.setLoadingState(key, true);

      const [performance, volume] = await Promise.allSettled([
        apiClient.get('/api/models/performance'),
        apiClient.get('/api/market/volume')
      ]);

      this.data.metrics = {
        performance: performance.status === 'fulfilled' ? performance.value : null,
        volume: volume.status === 'fulfilled' ? volume.value : null
      };

      this.notify('metrics', this.data.metrics);
      return this.data.metrics;

    } catch (error) {
      logger.warn('Failed to load metrics data', { error: error.message });
      return {};

    } finally {
      this.setLoadingState(key, false);
    }
  }

  /**
   * 폴백 데이터 로드 (파일 기반)
   */
  async loadFallbackData(dataType) {
    const fileMap = {
      stocks: '../data/raw/realtime_results.json',
      performance: '../data/raw/model_performance.json',
      sentiment: '../data/raw/market_sentiment.json'
    };

    const filePath = fileMap[dataType];
    if (!filePath) {
      throw new Error(`No fallback file defined for ${dataType}`);
    }

    try {
      const response = await fetch(filePath);
      if (!response.ok) throw new Error(`File not found: ${filePath}`);
      
      const data = await response.json();
      logger.info(`📁 Loaded fallback data: ${filePath}`);
      
      // 데이터 형식 정규화
      if (dataType === 'stocks') {
        return data.predictions || [data].filter(Boolean);
      }
      
      return data;

    } catch (error) {
      logger.error(`Failed to load fallback data: ${filePath}`, { error: error.message });
      throw error;
    }
  }

  /**
   * 전체 데이터 새로고침
   */
  async refreshAllData() {
    logger.info('🔄 Refreshing all data...');
    
    try {
      eventBus.emit(EVENTS.REFRESH_TRIGGERED);
      
      const results = await Promise.allSettled([
        this.loadStockData(),
        this.loadMLPredictions(),
        this.loadNewsData(),
        this.loadMetricsData()
      ]);

      const successful = results.filter(r => r.status === 'fulfilled').length;
      const failed = results.length - successful;

      logger.info(`✅ Data refresh completed: ${successful} succeeded, ${failed} failed`);
      
      return { successful, failed, results };

    } catch (error) {
      logger.error('Data refresh failed', { error: error.message });
      throw error;
    }
  }

  /**
   * 특정 데이터 타입 새로고침
   */
  async refreshData(dataType) {
    const loaders = {
      stocks: () => this.loadStockData(),
      ml: () => this.loadMLPredictions(),
      news: () => this.loadNewsData(),
      metrics: () => this.loadMetricsData()
    };

    const loader = loaders[dataType];
    if (!loader) {
      throw new Error(`Unknown data type: ${dataType}`);
    }

    return loader();
  }

  /**
   * 데이터 검증
   */
  validateData(dataType, data) {
    const validators = {
      stocks: (data) => Array.isArray(data) && data.every(item => item.symbol && item.confidence),
      ml: (data) => typeof data === 'object' && Object.keys(data).length > 0,
      news: (data) => Array.isArray(data),
      metrics: (data) => typeof data === 'object'
    };

    const validator = validators[dataType];
    if (!validator) return true;

    const isValid = validator(data);
    if (!isValid) {
      logger.warn(`Invalid data format for ${dataType}`, { data });
    }

    return isValid;
  }

  /**
   * 캐시 상태 조회
   */
  getCacheStatus() {
    return apiClient.getCacheInfo();
  }

  /**
   * 캐시 초기화
   */
  clearCache(pattern) {
    apiClient.clearCache(pattern);
  }

  /**
   * 현재 데이터 스냅샷
   */
  getDataSnapshot() {
    return {
      timestamp: new Date().toISOString(),
      data: { ...this.data },
      loadingStates: Object.fromEntries(this.loadingStates),
      subscriberCounts: Object.fromEntries(
        Array.from(this.subscribers.entries()).map(([key, set]) => [key, set.size])
      )
    };
  }
}

// 싱글톤 인스턴스
export const dataService = new DataService();