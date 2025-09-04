/**
 * Unified API Client with Error Handling & Retry Logic
 * 통합 API 클라이언트 - 에러 처리, 재시도, 캐싱 포함
 */

import { logger } from './logger.js';
import { eventBus, EVENTS } from './event-bus.js';

export class APIClient {
  constructor(config = {}) {
    this.baseURL = config.baseURL || 'http://localhost:8090';
    this.timeout = config.timeout || 3000;
    this.maxRetries = config.maxRetries || 2;
    this.retryDelay = config.retryDelay || 1000;
    
    // 캐시 설정
    this.cacheTimeout = config.cacheTimeout || 30000; // 30초
    this.cache = new Map();
    
    // 요청 인터셉터
    this.requestInterceptors = [];
    this.responseInterceptors = [];
    
    // 동시 요청 제어
    this.pendingRequests = new Map();
    this.maxConcurrent = config.maxConcurrent || 5;
    this.activeRequests = 0;
    
    logger.info('APIClient initialized', { baseURL: this.baseURL, timeout: this.timeout });
  }

  /**
   * 캐시 키 생성
   */
  getCacheKey(url, options = {}) {
    const method = options.method || 'GET';
    const params = options.params ? JSON.stringify(options.params) : '';
    return `${method}:${url}:${params}`;
  }

  /**
   * 캐시 확인
   */
  getFromCache(key) {
    const cached = this.cache.get(key);
    if (cached && Date.now() - cached.timestamp < this.cacheTimeout) {
      logger.debug(`Cache hit: ${key}`);
      return cached.data;
    }
    
    if (cached) {
      this.cache.delete(key); // 만료된 캐시 제거
    }
    return null;
  }

  /**
   * 캐시 저장
   */
  setCache(key, data) {
    // 메모리 관리 (최대 50개 캐시)
    if (this.cache.size >= 50) {
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }
    
    this.cache.set(key, {
      data,
      timestamp: Date.now()
    });
  }

  /**
   * 중복 요청 방지
   */
  getDedupedRequest(key, requestFn) {
    if (this.pendingRequests.has(key)) {
      logger.debug(`Deduped request: ${key}`);
      return this.pendingRequests.get(key);
    }

    const promise = requestFn().finally(() => {
      this.pendingRequests.delete(key);
    });

    this.pendingRequests.set(key, promise);
    return promise;
  }

  /**
   * 재시도 로직 포함 fetch
   */
  async fetchWithRetry(url, options = {}, attempt = 1) {
    const fullURL = url.startsWith('http') ? url : `${this.baseURL}${url}`;
    
    // 동시 요청 제한
    if (this.activeRequests >= this.maxConcurrent) {
      await new Promise(resolve => setTimeout(resolve, 100));
      return this.fetchWithRetry(url, options, attempt);
    }

    this.activeRequests++;
    const requestId = `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    
    try {
      logger.startPerformance(`API:${url}`);
      
      // 타임아웃 설정
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout);
      
      const requestOptions = {
        ...options,
        signal: controller.signal,
        headers: {
          'Content-Type': 'application/json',
          ...options.headers
        }
      };

      // 요청 인터셉터 실행
      for (const interceptor of this.requestInterceptors) {
        await interceptor(fullURL, requestOptions);
      }

      const response = await fetch(fullURL, requestOptions);
      clearTimeout(timeoutId);

      // 응답 인터셉터 실행
      for (const interceptor of this.responseInterceptors) {
        await interceptor(response);
      }

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      logger.endPerformance(`API:${url}`, `(${response.status})`);
      
      return data;

    } catch (error) {
      logger.warn(`API request failed (attempt ${attempt}): ${url}`, { 
        error: error.message,
        requestId 
      });

      // 재시도 로직
      if (attempt < this.maxRetries && !error.name.includes('Abort')) {
        await new Promise(resolve => setTimeout(resolve, this.retryDelay * attempt));
        return this.fetchWithRetry(url, options, attempt + 1);
      }

      // 최종 실패
      eventBus.emit(EVENTS.DATA_ERROR, { url, error, requestId });
      throw error;

    } finally {
      this.activeRequests--;
    }
  }

  /**
   * GET 요청
   */
  async get(url, options = {}) {
    const cacheKey = this.getCacheKey(url, options);
    
    // 캐시 확인
    if (options.useCache !== false) {
      const cached = this.getFromCache(cacheKey);
      if (cached) return cached;
    }

    // 중복 요청 방지
    return this.getDedupedRequest(cacheKey, async () => {
      const data = await this.fetchWithRetry(url, { ...options, method: 'GET' });
      
      // 캐시 저장 (useCache가 false가 아닌 경우만)
      if (options.useCache !== false) {
        this.setCache(cacheKey, data);
      }
      
      return data;
    });
  }

  /**
   * POST 요청
   */
  async post(url, body, options = {}) {
    return this.fetchWithRetry(url, {
      ...options,
      method: 'POST',
      body: JSON.stringify(body)
    });
  }

  /**
   * PUT 요청  
   */
  async put(url, body, options = {}) {
    return this.fetchWithRetry(url, {
      ...options,
      method: 'PUT',
      body: JSON.stringify(body)
    });
  }

  /**
   * DELETE 요청
   */
  async delete(url, options = {}) {
    return this.fetchWithRetry(url, { ...options, method: 'DELETE' });
  }

  /**
   * 배치 요청 (병렬 처리)
   */
  async batch(requests) {
    logger.info(`Executing ${requests.length} batch requests`);
    
    const results = await Promise.allSettled(
      requests.map(({ method, url, data, options }) => {
        switch (method.toLowerCase()) {
          case 'get': return this.get(url, options);
          case 'post': return this.post(url, data, options);
          case 'put': return this.put(url, data, options);
          case 'delete': return this.delete(url, options);
          default: throw new Error(`Unsupported method: ${method}`);
        }
      })
    );

    return results.map((result, index) => ({
      success: result.status === 'fulfilled',
      data: result.status === 'fulfilled' ? result.value : null,
      error: result.status === 'rejected' ? result.reason : null,
      request: requests[index]
    }));
  }

  /**
   * 인터셉터 추가
   */
  addRequestInterceptor(interceptor) {
    this.requestInterceptors.push(interceptor);
  }

  addResponseInterceptor(interceptor) {
    this.responseInterceptors.push(interceptor);
  }

  /**
   * 캐시 관리
   */
  clearCache(pattern = null) {
    if (pattern) {
      const regex = new RegExp(pattern);
      for (const key of this.cache.keys()) {
        if (regex.test(key)) {
          this.cache.delete(key);
        }
      }
    } else {
      this.cache.clear();
    }
    
    logger.info('Cache cleared', { pattern });
    eventBus.emit(EVENTS.CACHE_CLEARED, { pattern });
  }

  getCacheInfo() {
    return {
      size: this.cache.size,
      keys: Array.from(this.cache.keys()),
      pendingRequests: this.pendingRequests.size,
      activeRequests: this.activeRequests
    };
  }

  /**
   * 상태 확인
   */
  async healthCheck() {
    try {
      const start = performance.now();
      await this.get('/api/status', { useCache: false });
      const duration = Math.round(performance.now() - start);
      
      logger.info(`API health check passed (${duration}ms)`);
      return { healthy: true, responseTime: duration };
    } catch (error) {
      logger.error('API health check failed', { error: error.message });
      return { healthy: false, error: error.message };
    }
  }
}

// 글로벌 API 클라이언트 인스턴스
export const apiClient = new APIClient({
  baseURL: 'http://localhost:8090',
  timeout: 3000,
  maxRetries: 2,
  cacheTimeout: 30000,
  maxConcurrent: 5
});