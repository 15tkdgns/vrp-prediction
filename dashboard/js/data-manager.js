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
    this.cacheTimeout = 60000; // 60초 캐시 (실제 데이터는 좋은 캐시 사용)

    // API Configuration - 실제 데이터 API 서버
    this.apiBaseUrl = 'http://localhost:8092/api';
    this.apiTimeout = 2000; // 2초 타임아웃 (더 단축)
    this.maxRetries = 2; // 재시도 횟수
    this.retryDelay = 300; // 재시도 지연 더 단축
    this.fastTimeout = 1000; // 빠른 요청용 타임아웃
    this.adaptiveRetry = true; // 적응형 재시도

    // 로딩 상태 및 에러 추적
    this.loadingStates = new Map();
    this.loadingStartTimes = new Map();
    this.errorCounts = new Map();
    this.lastErrors = new Map();
    this.userNotifications = []; // 사용자 알림

    // 성능 모니터링
    this.performanceMetrics = {
      totalRequests: 0,
      successfulRequests: 0,
      failedRequests: 0,
      avgResponseTime: 0,
      cacheHitRate: 0,
      loadTimes: new Map(),
    };

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
   * 초기화 (빠른 초기화)
   */
  async init() {
    console.log('📊 DataManager 초기화 중...');

    // API 서버 상태는 백그라운드에서 확인
    this.checkAPIStatusBackground();

    // 즉시 반환하여 초기화 속도 개선
    console.log('✅ DataManager 빠른 초기화 완료');
  }

  /**
   * 백그라운드에서 API 상태 확인
   */
  async checkAPIStatusBackground() {
    try {
      const status = await this.fetchAPI('/status');
      console.log('✅ API 서버 상태 확인:', status.status);
      this.apiAvailable = true;
    } catch (error) {
      console.warn('⚠️ API 서버 접속 실패, 로컬 데이터 모드로 동작');
      this.apiAvailable = false;
    }
  }

  /**
   * 시스템 상태 데이터만 빠르게 로드
   */
  async loadSystemStatus() {
    try {
      // 로컬 파일에서 빠르게 로드
      const data = await this.loadLocalFile('../data/raw/system_status.json');
      this.data.systemStatus = data;
      return data;
    } catch (error) {
      console.warn('시스템 상태 로드 실패:', error);
      return null;
    }
  }

  /**
   * 뉴스 데이터 백그라운드 로드
   */
  async loadNewsData() {
    try {
      const data = await this.loadLocalFile(
        '../data/raw/market_sentiment.json'
      );
      this.data.news = data;
      return data;
    } catch (error) {
      console.warn('뉴스 데이터 로드 실패:', error);
      return null;
    }
  }

  /**
   * 마켓 데이터 백그라운드 로드
   */
  async loadMarketData() {
    try {
      const data = await this.loadLocalFile('../data/raw/trading_volume.json');
      this.data.market = data;
      return data;
    } catch (error) {
      console.warn('마켓 데이터 로드 실패:', error);
      return null;
    }
  }

  /**
   * 로컬 파일 로드 유틸리티
   */
  async loadLocalFile(path) {
    const response = await fetch(path);
    if (!response.ok) {
      throw new Error(`File load failed: ${response.status}`);
    }
    return await response.json();
  }

  /**
   * API 호출 메서드 (개선된 재시도 로직)
   */
  async fetchAPI(endpoint, options = {}) {
    const url = `${this.apiBaseUrl}${endpoint}`;
    const isFastRequest = options.fast || false;
    const timeout = isFastRequest ? this.fastTimeout : this.apiTimeout;

    const config = {
      method: 'GET',
      timeout: timeout,
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      ...options,
    };

    let lastError;
    const startTime = Date.now();

    for (let attempt = 1; attempt <= this.maxRetries; attempt++) {
      try {
        console.log(
          `🔄 API 호출 (시도 ${attempt}/${this.maxRetries}): ${url} (타임아웃: ${timeout}ms)`
        );

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), timeout);

        const response = await fetch(url, {
          ...config,
          signal: controller.signal,
        });

        clearTimeout(timeoutId);

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();
        const duration = Date.now() - startTime;
        console.log(`✅ API 호출 성공: ${endpoint} (${duration}ms)`);

        // 성능 메트릭 업데이트
        this.updatePerformanceMetrics(endpoint, duration, true, false);

        // 성공 시 에러 카운터 리셋
        this.errorCounts.delete(endpoint);

        return data;
      } catch (error) {
        lastError = error;
        this.trackError(endpoint, error);
        console.warn(`❌ API 호출 실패 (시도 ${attempt}): ${error.message}`);

        if (attempt < this.maxRetries) {
          // 적응형 재시도 지연
          let delay = this.retryDelay * attempt;

          if (this.adaptiveRetry) {
            // 에러 유형에 따른 지연 조정
            const errorType = this.classifyError(lastError);
            switch (errorType) {
              case 'timeout':
                delay = this.retryDelay * 2; // 타임아웃은 더 오래 대기
                break;
              case 'network':
                delay = this.retryDelay * 3; // 네트워크 문제는 더 오래 대기
                break;
              case 'server':
                delay = this.retryDelay * 1.5;
                break;
              default:
                delay = this.retryDelay;
            }
          }

          console.log(
            `⏱️ ${delay}ms 후 재시도... (에러: ${this.classifyError(lastError)})`
          );
          await new Promise((resolve) => setTimeout(resolve, delay));
        }
      }
    }

    // 실패 메트릭 업데이트
    const duration = Date.now() - startTime;
    this.updatePerformanceMetrics(endpoint, duration, false, false);

    throw lastError;
  }

  /**
   * 에러 추적 및 분석
   */
  trackError(endpoint, error) {
    const count = this.errorCounts.get(endpoint) || 0;
    this.errorCounts.set(endpoint, count + 1);
    this.lastErrors.set(endpoint, {
      error: error.message,
      timestamp: Date.now(),
      type: this.classifyError(error),
    });

    // 사용자 알림 추가 (다수 오류 시만)
    if (count >= 2) {
      this.addUserNotification({
        type: 'warning',
        message: `데이터 로딩 중 문제가 발생했습니다. 캐시된 데이터를 사용합니다.`,
        timestamp: Date.now(),
        endpoint: endpoint,
      });
    }
  }

  /**
   * 에러 유형 분류
   */
  classifyError(error) {
    if (error.name === 'AbortError') return 'timeout';
    if (error.message.includes('Failed to fetch')) return 'network';
    if (error.message.includes('HTTP 5')) return 'server';
    if (error.message.includes('HTTP 4')) return 'client';
    return 'unknown';
  }

  /**
   * 사용자 알림 추가
   */
  addUserNotification(notification) {
    this.userNotifications.unshift(notification);
    // 최대 5개만 유지
    if (this.userNotifications.length > 5) {
      this.userNotifications = this.userNotifications.slice(0, 5);
    }
  }

  /**
   * 사용자 알림 가져오기
   */
  getUserNotifications() {
    return this.userNotifications;
  }

  /**
   * 알림 클리어
   */
  clearNotifications() {
    this.userNotifications = [];
  }

  /**
   * 성능 메트릭 업데이트
   */
  updatePerformanceMetrics(
    endpoint,
    duration,
    success = true,
    fromCache = false
  ) {
    this.performanceMetrics.totalRequests++;

    if (success) {
      this.performanceMetrics.successfulRequests++;
    } else {
      this.performanceMetrics.failedRequests++;
    }

    // 평균 응답 시간 업데이트
    const currentAvg = this.performanceMetrics.avgResponseTime;
    const totalSuccessful = this.performanceMetrics.successfulRequests;
    this.performanceMetrics.avgResponseTime =
      (currentAvg * (totalSuccessful - 1) + duration) / totalSuccessful;

    // 캐시 히트율 계산
    if (fromCache) {
      const cacheHits = this.performanceMetrics.cacheHits || 0;
      this.performanceMetrics.cacheHits = cacheHits + 1;
    }

    this.performanceMetrics.cacheHitRate =
      ((this.performanceMetrics.cacheHits || 0) /
        this.performanceMetrics.totalRequests) *
      100;

    // 개별 엔드포인트 성능 추적
    if (!this.performanceMetrics.loadTimes.has(endpoint)) {
      this.performanceMetrics.loadTimes.set(endpoint, []);
    }

    const endpointTimes = this.performanceMetrics.loadTimes.get(endpoint);
    endpointTimes.push({ duration, success, timestamp: Date.now(), fromCache });

    // 최근 20개만 유지
    if (endpointTimes.length > 20) {
      this.performanceMetrics.loadTimes.set(endpoint, endpointTimes.slice(-20));
    }
  }

  /**
   * 성능 리포트 생성
   */
  getPerformanceReport() {
    const metrics = this.performanceMetrics;
    const report = {
      summary: {
        totalRequests: metrics.totalRequests,
        successRate:
          ((metrics.successfulRequests / metrics.totalRequests) * 100).toFixed(
            1
          ) + '%',
        avgResponseTime: Math.round(metrics.avgResponseTime) + 'ms',
        cacheHitRate: metrics.cacheHitRate.toFixed(1) + '%',
      },
      endpoints: {},
      issues: [],
    };

    // 각 엔드포인트별 성능 분석
    for (const [endpoint, times] of metrics.loadTimes) {
      const successful = times.filter((t) => t.success);
      const failed = times.filter((t) => !t.success);
      const avgTime =
        successful.length > 0
          ? successful.reduce((sum, t) => sum + t.duration, 0) /
            successful.length
          : 0;

      report.endpoints[endpoint] = {
        requests: times.length,
        successRate:
          ((successful.length / times.length) * 100).toFixed(1) + '%',
        avgResponseTime: Math.round(avgTime) + 'ms',
        lastRequest: new Date(
          Math.max(...times.map((t) => t.timestamp))
        ).toLocaleTimeString(),
      };

      // 문제 점 식별
      if (successful.length / times.length < 0.8) {
        report.issues.push(
          `${endpoint}: 낮은 성공률 (${report.endpoints[endpoint].successRate})`
        );
      }

      if (avgTime > 3000) {
        report.issues.push(
          `${endpoint}: 느린 응답 시간 (${Math.round(avgTime)}ms)`
        );
      }
    }

    return report;
  }

  /**
   * 성능 메트릭 리셋
   */
  resetPerformanceMetrics() {
    this.performanceMetrics = {
      totalRequests: 0,
      successfulRequests: 0,
      failedRequests: 0,
      avgResponseTime: 0,
      cacheHitRate: 0,
      loadTimes: new Map(),
    };
    console.log('📈 성능 메트릭 리셋됨');
  }

  /**
   * 디버그 정보 출력
   */
  logDebugInfo() {
    console.group('📈 DataManager 성능 리포트');

    const report = this.getPerformanceReport();
    console.log('📊 전체 성능:', report.summary);

    if (report.issues.length > 0) {
      console.warn('⚠️ 발견된 문제점:', report.issues);
    }

    console.log('🔍 엔드포인트별 성능:', report.endpoints);

    const notifications = this.getUserNotifications();
    if (notifications.length > 0) {
      console.log('📬 사용자 알림:', notifications);
    }

    console.log('💾 캐시 상태:', {
      keys: Array.from(this.cache.keys()),
      hitRate: report.summary.cacheHitRate,
    });

    console.log('🔄 로딩 상태:', Object.fromEntries(this.loadingStates));

    console.groupEnd();
  }

  /**
   * 주식 데이터 로드
   */
  async loadStockData() {
    const startTime = Date.now();
    const methodName = 'loadStockData';

    try {
      this.loadingStates.set(methodName, 'loading');
      this.loadingStartTimes.set(methodName, startTime);

      const cacheKey = 'stocks';
      const cached = this.getCachedData(cacheKey);
      if (cached) {
        this.data.stocks = cached.predictions || cached;
        console.log(`📋 캐시된 주식 데이터 사용 (${Date.now() - startTime}ms)`);
        this.loadingStates.set(methodName, 'cached');
        return this.data.stocks;
      }

      // API에서 실시간 주식 데이터 로드 (빠른 폴백 병렬 처리)
      const apiPromise = this.fetchAPI('/stocks/live', { fast: true }); // 빠른 요청 옵션
      const fallbackPromise = this.fetchJSON(
        '../data/raw/realtime_results.json'
      );

      // Race 조건: API가 1초 내에 응답하면 사용, 아니면 폴백
      const result = await Promise.race([
        Promise.allSettled([
          apiPromise.then((data) => ({
            source: 'api',
            data,
            timestamp: Date.now(),
          })),
          fallbackPromise.then((data) => ({
            source: 'fallback',
            data,
            timestamp: Date.now(),
          })),
        ]).then((results) => {
          const successful = results.find((r) => r.status === 'fulfilled');
          return successful ? successful.value : null;
        }),
        new Promise((resolve) =>
          setTimeout(() => resolve(null), this.fastTimeout + 500)
        ),
      ]);

      if (result && result.data && result.data.predictions) {
        this.data.stocks = result.data.predictions.slice(0, 4);
        this.setCachedData(cacheKey, result.data);
        console.log(
          `✅ 주식 데이터 로드됨 (소스: ${result.source}, ${Date.now() - startTime}ms)`
        );
        this.loadingStates.set(methodName, 'success');
        return this.data.stocks;
      }

      throw new Error('모든 데이터 소스 실패');
    } catch (error) {
      this.loadingStates.set(methodName, 'error');
      this.lastErrors.set(methodName, {
        error: error.message,
        timestamp: Date.now(),
        duration: Date.now() - startTime,
      });
      console.warn(
        `⚠️ 주식 데이터 로드 실패 (${Date.now() - startTime}ms):`,
        error.message
      );

      // 최종 폴백: 목업 데이터
      this.data.stocks = this.getMockStockData();
      this.loadingStates.set(methodName, 'fallback');
      console.log(`⚠️ 목업 주식 데이터 사용 (${Date.now() - startTime}ms)`);
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
   * 캐시된 데이터 가져오기 (스마트 캐싱)
   */
  getCachedData(key) {
    const cached = this.cache.get(key);
    const lastFetch = this.lastFetchTime.get(key);
    const now = Date.now();

    if (cached && lastFetch) {
      const age = now - lastFetch;

      // 신선한 데이터는 즉시 반환
      if (age < this.cacheTimeout) {
        console.log(
          `📋 실제 데이터 캐시 사용: ${key} (나이: ${Math.round(age / 1000)}s)`
        );
        this.updatePerformanceMetrics(key, age, true, true);
        return cached;
      }

      // 오래된 실제 데이터지만 백그라운드에서 업데이트 시작
      if (age < this.cacheTimeout * 2) {
        console.log(`📋 실제 데이터 캐시 사용 + 백그라운드 업데이트: ${key}`);
        this.backgroundRefresh(key);
        return cached;
      }
    }

    return null;
  }

  /**
   * 데이터 캐시에 저장 (개선된 메타데이터 포함)
   */
  setCachedData(key, data) {
    const metadata = {
      data: data,
      timestamp: Date.now(),
      size: JSON.stringify(data).length,
      accessCount: (this.cache.get(key)?.accessCount || 0) + 1,
    };

    this.cache.set(key, metadata.data);
    this.lastFetchTime.set(key, metadata.timestamp);

    console.log(
      `💾 캐시 저장: ${key} (크기: ${Math.round(metadata.size / 1024)}KB)`
    );
  }

  /**
   * 백그라운드에서 특정 데이터 새로고침
   */
  backgroundRefresh(key) {
    // 이미 백그라운드 업데이트 중이면 건너뛰기
    if (this.loadingStates.get(`background_${key}`) === 'loading') {
      return;
    }

    this.loadingStates.set(`background_${key}`, 'loading');

    setTimeout(async () => {
      try {
        switch (key) {
          case 'stocks':
            await this.loadStockData();
            break;
          case 'metrics':
            await this.loadMetricsData();
            break;
          case 'news':
            await this.loadNewsData();
            break;
        }
        console.log(`✅ 백그라운드 업데이트 완료: ${key}`);
      } catch (error) {
        console.warn(`⚠️ 백그라운드 업데이트 실패: ${key}`, error);
      } finally {
        this.loadingStates.delete(`background_${key}`);
      }
    }, 100); // 100ms 지연 후 실행
  }

  /**
   * 모든 데이터 새로고침
   */
  async refresh() {
    console.log('🔄 모든 데이터 새로고침 중...');
    const startTime = Date.now();

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
      performance: this.getPerformanceReport(),
      notifications: this.getUserNotifications(),
      loadingStates: Object.fromEntries(this.loadingStates),
      dataKeys: Object.keys(this.data),
      stocksCount: this.data.stocks.length,
      lastUpdate: Math.max(...Array.from(this.lastFetchTime.values())),
    };
  }
}

// 전역 변수로 내보내기
window.DataManager = DataManager;
