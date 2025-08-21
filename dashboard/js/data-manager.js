/**
 * DataManager - 모든 데이터 로드 및 관리
 *
 * 특징:
 * 1. 간단하고 안정적인 데이터 로딩
 * 2. 캐싱으로 중복 요청 방지
 * 3. 에러 처리와 폴백 데이터
 */

class DataManager {
  constructor() {
    this.cache = new Map();
    this.lastFetchTime = new Map();
    this.cacheTimeout = 30000; // 30초 캐시

    // API Configuration
    this.apiBaseUrl = 'http://localhost:8091/api';
    this.apiTimeout = 5000; // 5초 타임아웃
    this.maxRetries = 3;

    // 데이터 저장소
    this.data = {
      stocks: [],
      metrics: {},
      news: [],
      charts: {},
    };

    console.log('📊 DataManager 생성됨 (API Mode)');
    console.log('🔗 API Base URL:', this.apiBaseUrl);
  }

  /**
   * 초기화
   */
  async init() {
    console.log('📊 DataManager 초기화 중...');

    // API 서버 상태 확인
    try {
      const status = await this.fetchAPI('/status');
      console.log('✅ API 서버 상태 확인:', status.status);
    } catch (error) {
      console.warn('⚠️ API 서버 접속 실패, 폴백 모드로 동작:', error.message);
    }
  }

  /**
   * API 호출 메서드 (재시도 로직 포함)
   */
  async fetchAPI(endpoint, options = {}) {
    const url = `${this.apiBaseUrl}${endpoint}`;
    const config = {
      method: 'GET',
      timeout: this.apiTimeout,
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      ...options,
    };

    let lastError;

    for (let attempt = 1; attempt <= this.maxRetries; attempt++) {
      try {
        console.log(`🔄 API 호출 (시도 ${attempt}/${this.maxRetries}): ${url}`);

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), config.timeout);

        const response = await fetch(url, {
          ...config,
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();
        console.log(`✅ API 호출 성공: ${endpoint}`);
        return data;
      } catch (error) {
        lastError = error;
        console.warn(`❌ API 호출 실패 (시도 ${attempt}): ${error.message}`);

        if (attempt < this.maxRetries) {
          const delay = Math.min(1000 * Math.pow(2, attempt - 1), 5000); // Exponential backoff
          console.log(`⏱️ ${delay}ms 후 재시도...`);
          await new Promise((resolve) => setTimeout(resolve, delay));
        }
      }
    }

    throw lastError;
  }

  /**
   * 주식 데이터 로드
   */
  async loadStockData() {
    try {
      const cacheKey = 'stocks';
      const cached = this.getCachedData(cacheKey);
      if (cached) {
        this.data.stocks = cached.predictions || cached;
        return this.data.stocks;
      }

      // API에서 실시간 주식 데이터 로드
      const apiData = await this.fetchAPI('/stocks/live');
      if (apiData && apiData.predictions) {
        this.data.stocks = apiData.predictions.slice(0, 4); // 상위 4개
        this.setCachedData(cacheKey, apiData);
        console.log(
          '✅ 실시간 주식 데이터 로드됨 (소스:',
          apiData.source || 'api',
          ')'
        );
        return this.data.stocks;
      }

      throw new Error('API에서 주식 데이터가 비어있음');
    } catch (error) {
      console.warn('⚠️ API 데이터 로드 실패, 폴백 시도:', error.message);

      // 폴백: 기존 JSON 파일 시도
      try {
        const fallbackData = await this.fetchJSON(
          '../data/raw/realtime_results.json'
        );
        if (fallbackData && fallbackData.predictions) {
          this.data.stocks = fallbackData.predictions.slice(0, 4);
          console.log('✅ 폴백 주식 데이터 로드됨 (JSON 파일)');
          return this.data.stocks;
        }
      } catch (fallbackError) {
        console.warn('⚠️ 폴백 파일도 실패:', fallbackError.message);
      }

      // 최종 폴백: 목업 데이터
      this.data.stocks = this.getMockStockData();
      console.log('⚠️ 목업 주식 데이터 사용');
      return this.data.stocks;
    }
  }

  /**
   * 성능 지표 데이터 로드
   */
  async loadMetrics() {
    try {
      const cacheKey = 'metrics';
      const cached = this.getCachedData(cacheKey);
      if (cached) {
        this.data.metrics = cached;
        return cached;
      }

      // API에서 모델 성능 데이터 로드
      const apiData = await this.fetchAPI('/models/performance');
      if (apiData) {
        this.data.metrics = apiData;
        this.setCachedData(cacheKey, apiData);
        console.log(
          '✅ 실시간 성능 지표 로드됨 (소스:',
          apiData.source || 'api',
          ')'
        );
        return this.data.metrics;
      }

      throw new Error('API에서 성능 데이터가 없음');
    } catch (error) {
      console.warn('⚠️ API 성능 지표 로드 실패, 폴백 시도:', error.message);

      // 폴백: 기존 JSON 파일 시도
      try {
        const fallbackData = await this.fetchJSON(
          '../data/raw/model_performance.json'
        );
        if (fallbackData) {
          this.data.metrics = fallbackData;
          console.log('✅ 폴백 성능 지표 로드됨 (JSON 파일)');
          return this.data.metrics;
        }
      } catch (fallbackError) {
        console.warn('⚠️ 폴백 파일도 실패:', fallbackError.message);
      }

      // 최종 폴백: 목업 데이터
      this.data.metrics = this.getMockMetrics();
      console.log('⚠️ 목업 성능 지표 사용');
      return this.data.metrics;
    }
  }

  /**
   * 뉴스 데이터 로드
   */
  async loadNews() {
    try {
      const cacheKey = 'news';
      const cached = this.getCachedData(cacheKey);
      if (cached) {
        this.data.news = Array.isArray(cached) ? cached : [cached];
        return this.data.news;
      }

      // API에서 뉴스 감정 분석 데이터 로드
      const apiData = await this.fetchAPI('/news/sentiment');
      if (apiData) {
        this.data.news = [apiData]; // 배열로 감싸기
        this.setCachedData(cacheKey, apiData);
        console.log(
          '✅ 실시간 뉴스 감정 분석 로드됨 (소스:',
          apiData.source || 'api',
          ')'
        );
        return this.data.news;
      }

      throw new Error('API에서 뉴스 데이터가 없음');
    } catch (error) {
      console.warn('⚠️ API 뉴스 데이터 로드 실패, 폴백 시도:', error.message);

      // 폴백: 기존 JSON 파일 시도
      try {
        const fallbackData = await this.fetchJSON(
          '../data/raw/market_sentiment.json'
        );
        if (fallbackData) {
          this.data.news = [fallbackData];
          console.log('✅ 폴백 뉴스 데이터 로드됨 (JSON 파일)');
          return this.data.news;
        }
      } catch (fallbackError) {
        console.warn('⚠️ 폴백 파일도 실패:', fallbackError.message);
      }

      // 최종 폴백: 목업 데이터
      this.data.news = this.getMockNews();
      console.log('⚠️ 목업 뉴스 데이터 사용');
      return this.data.news;
    }
  }

  /**
   * 차트 데이터 로드
   */
  async loadChartData() {
    try {
      const cacheKey = 'charts';
      const cached = this.getCachedData(cacheKey);
      if (cached) {
        this.data.charts = cached;
        return cached;
      }

      // API에서 차트 데이터 로드 (병렬로)
      const [volumeResult, trendResult] = await Promise.allSettled([
        this.fetchAPI('/market/volume'),
        this.fetchJSON('../data/raw/model_performance_trend.json'), // 트렌드는 아직 API 없음
      ]);

      this.data.charts = {
        volume: volumeResult.status === 'fulfilled' ? volumeResult.value : null,
        trend: trendResult.status === 'fulfilled' ? trendResult.value : null,
      };

      this.setCachedData(cacheKey, this.data.charts);

      const volumeSource = this.data.charts.volume?.source || 'unknown';
      console.log('✅ 차트 데이터 로드됨 (거래량 소스:', volumeSource, ')');
      return this.data.charts;
    } catch (error) {
      console.warn('⚠️ API 차트 데이터 로드 실패, 폴백 시도:', error.message);

      // 폴백: 기존 JSON 파일들 시도
      try {
        const [volumeData, trendData] = await Promise.allSettled([
          this.fetchJSON('../data/raw/trading_volume.json'),
          this.fetchJSON('../data/raw/model_performance_trend.json'),
        ]);

        this.data.charts = {
          volume: volumeData.status === 'fulfilled' ? volumeData.value : null,
          trend: trendData.status === 'fulfilled' ? trendData.value : null,
        };

        console.log('✅ 폴백 차트 데이터 로드됨 (JSON 파일)');
        return this.data.charts;
      } catch (fallbackError) {
        console.warn('⚠️ 폴백 파일도 실패:', fallbackError.message);
      }

      // 최종 폴백: 목업 데이터
      this.data.charts = this.getMockChartData();
      console.log('⚠️ 목업 차트 데이터 사용');
      return this.data.charts;
    }
  }

  /**
   * JSON 파일 가져오기 (에러 처리 포함)
   */
  async fetchJSON(url) {
    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      return await response.json();
    } catch (error) {
      console.warn(`JSON 가져오기 실패 (${url}):`, error.message);
      throw error;
    }
  }

  /**
   * 캐시된 데이터 가져오기
   */
  getCachedData(key) {
    const cached = this.cache.get(key);
    const lastFetch = this.lastFetchTime.get(key);

    if (cached && lastFetch && Date.now() - lastFetch < this.cacheTimeout) {
      console.log(`📋 캐시에서 ${key} 데이터 사용`);
      return cached;
    }

    return null;
  }

  /**
   * 데이터 캐시에 저장
   */
  setCachedData(key, data) {
    this.cache.set(key, data);
    this.lastFetchTime.set(key, Date.now());
  }

  /**
   * 모든 데이터 새로고침
   */
  async refresh() {
    console.log('🔄 모든 데이터 새로고침 중...');

    // 캐시 클리어
    this.cache.clear();
    this.lastFetchTime.clear();

    // 모든 데이터 다시 로드
    await Promise.allSettled([
      this.loadStockData(),
      this.loadMetrics(),
      this.loadNews(),
      this.loadChartData(),
    ]);

    console.log('✅ 데이터 새로고침 완료');
  }

  /**
   * 목업 주식 데이터
   */
  getMockStockData() {
    return [
      {
        symbol: 'AAPL',
        current_price: 230.45,
        predicted_direction: 'up',
        confidence: 0.75,
        technical_indicators: { price_change: 0.024 },
      },
      {
        symbol: 'GOOGL',
        current_price: 201.53,
        predicted_direction: 'up',
        confidence: 0.68,
        technical_indicators: { price_change: 0.018 },
      },
      {
        symbol: 'MSFT',
        current_price: 509.71,
        predicted_direction: 'down',
        confidence: 0.62,
        technical_indicators: { price_change: -0.012 },
      },
      {
        symbol: 'AMZN',
        current_price: 227.74,
        predicted_direction: 'up',
        confidence: 0.71,
        technical_indicators: { price_change: 0.031 },
      },
    ];
  }

  /**
   * 목업 성능 지표
   */
  getMockMetrics() {
    return {
      accuracy: 0.847,
      precision: 0.823,
      recall: 0.861,
      f1_score: 0.842,
      training_time: '2.3분',
      last_updated: new Date().toISOString(),
    };
  }

  /**
   * 목업 뉴스 데이터
   */
  getMockNews() {
    return [
      {
        sentiment_score: 0.15,
        overall_sentiment: 'positive',
        confidence: 0.82,
        news_count: 47,
        timestamp: new Date().toISOString(),
      },
    ];
  }

  /**
   * 목업 차트 데이터
   */
  getMockChartData() {
    return {
      volume: {
        labels: ['월', '화', '수', '목', '금'],
        data: [120, 150, 80, 200, 175],
      },
      trend: {
        labels: ['1월', '2월', '3월', '4월', '5월', '6월'],
        accuracy: [0.82, 0.85, 0.83, 0.87, 0.84, 0.86],
        loss: [0.45, 0.38, 0.42, 0.33, 0.39, 0.35],
      },
    };
  }

  /**
   * 특정 주식 데이터 가져오기
   */
  getStockBySymbol(symbol) {
    return this.data.stocks.find((stock) => stock.symbol === symbol);
  }

  /**
   * 모든 데이터 가져오기
   */
  getAllData() {
    return {
      stocks: this.data.stocks,
      metrics: this.data.metrics,
      news: this.data.news,
      charts: this.data.charts,
    };
  }

  /**
   * 디버그 정보
   */
  getDebugInfo() {
    return {
      cacheKeys: Array.from(this.cache.keys()),
      dataKeys: Object.keys(this.data),
      stocksCount: this.data.stocks.length,
      lastUpdate: Math.max(...Array.from(this.lastFetchTime.values())),
    };
  }
}

// 전역 변수로 내보내기
window.DataManager = DataManager;
