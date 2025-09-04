/**
 * Performance Monitoring & Optimization Utilities
 * 성능 모니터링 및 최적화 유틸리티
 */

import { logger } from '../core/logger.js';

export class PerformanceMonitor {
  constructor() {
    this.measurements = new Map();
    this.observers = new Map();
    this.enabled = typeof window !== 'undefined' && 'performance' in window;
    
    if (this.enabled) {
      this.setupPerformanceObservers();
    }
  }

  /**
   * Performance Observer 설정
   */
  setupPerformanceObservers() {
    try {
      // Long Task 모니터링
      if ('PerformanceObserver' in window) {
        const longTaskObserver = new PerformanceObserver((list) => {
          list.getEntries().forEach(entry => {
            logger.warn(`Long task detected: ${Math.round(entry.duration)}ms`, {
              startTime: entry.startTime,
              name: entry.name
            });
          });
        });
        
        longTaskObserver.observe({ entryTypes: ['longtask'] });
        this.observers.set('longtask', longTaskObserver);

        // Paint 이벤트 모니터링
        const paintObserver = new PerformanceObserver((list) => {
          list.getEntries().forEach(entry => {
            logger.info(`Paint event: ${entry.name} at ${Math.round(entry.startTime)}ms`);
          });
        });
        
        paintObserver.observe({ entryTypes: ['paint'] });
        this.observers.set('paint', paintObserver);
      }
    } catch (error) {
      logger.warn('Failed to setup performance observers', { error: error.message });
    }
  }

  /**
   * 함수 실행 시간 측정
   */
  measure(name, fn) {
    if (!this.enabled) return fn();

    const start = performance.now();
    const result = fn();
    
    if (result instanceof Promise) {
      return result.finally(() => {
        const duration = performance.now() - start;
        this.recordMeasurement(name, duration);
      });
    } else {
      const duration = performance.now() - start;
      this.recordMeasurement(name, duration);
      return result;
    }
  }

  /**
   * 비동기 함수 실행 시간 측정
   */
  async measureAsync(name, asyncFn) {
    if (!this.enabled) return asyncFn();

    const start = performance.now();
    try {
      const result = await asyncFn();
      const duration = performance.now() - start;
      this.recordMeasurement(name, duration);
      return result;
    } catch (error) {
      const duration = performance.now() - start;
      this.recordMeasurement(name, duration, { error: error.message });
      throw error;
    }
  }

  /**
   * 측정값 기록
   */
  recordMeasurement(name, duration, metadata = {}) {
    if (!this.measurements.has(name)) {
      this.measurements.set(name, {
        count: 0,
        total: 0,
        min: Infinity,
        max: 0,
        measurements: []
      });
    }

    const stats = this.measurements.get(name);
    stats.count++;
    stats.total += duration;
    stats.min = Math.min(stats.min, duration);
    stats.max = Math.max(stats.max, duration);
    
    // 최근 10개 측정값만 유지 (메모리 최적화)
    stats.measurements.push({ 
      duration: Math.round(duration), 
      timestamp: Date.now(),
      ...metadata
    });
    
    if (stats.measurements.length > 10) {
      stats.measurements = stats.measurements.slice(-10);
    }

    // 성능 경고
    if (duration > 100) {
      logger.warn(`Slow operation: ${name} took ${Math.round(duration)}ms`);
    }
  }

  /**
   * 성능 리포트 생성
   */
  getReport() {
    const report = {};
    
    for (const [name, stats] of this.measurements) {
      report[name] = {
        count: stats.count,
        average: Math.round(stats.total / stats.count),
        min: Math.round(stats.min),
        max: Math.round(stats.max),
        total: Math.round(stats.total),
        recent: stats.measurements.slice(-3) // 최근 3개
      };
    }

    return report;
  }

  /**
   * 메모리 사용량 체크
   */
  getMemoryInfo() {
    if (performance.memory) {
      return {
        used: Math.round(performance.memory.usedJSHeapSize / 1024 / 1024), // MB
        total: Math.round(performance.memory.totalJSHeapSize / 1024 / 1024),
        limit: Math.round(performance.memory.jsHeapSizeLimit / 1024 / 1024)
      };
    }
    return null;
  }

  /**
   * DOM 노드 수 체크
   */
  getDOMInfo() {
    return {
      nodes: document.querySelectorAll('*').length,
      scripts: document.querySelectorAll('script').length,
      stylesheets: document.querySelectorAll('link[rel="stylesheet"], style').length,
      images: document.querySelectorAll('img').length
    };
  }

  /**
   * 성능 최적화 권장사항
   */
  getRecommendations() {
    const recommendations = [];
    const report = this.getReport();
    const memory = this.getMemoryInfo();
    const dom = this.getDOMInfo();

    // 느린 작업 체크
    for (const [name, stats] of Object.entries(report)) {
      if (stats.average > 100) {
        recommendations.push(`⚠️ ${name} 작업이 평균 ${stats.average}ms로 느립니다. 최적화를 고려하세요.`);
      }
    }

    // 메모리 사용량 체크
    if (memory && memory.used > 50) {
      recommendations.push(`💾 메모리 사용량이 ${memory.used}MB입니다. 메모리 정리를 고려하세요.`);
    }

    // DOM 노드 수 체크
    if (dom.nodes > 1000) {
      recommendations.push(`🏗️ DOM 노드가 ${dom.nodes}개로 많습니다. 가상화를 고려하세요.`);
    }

    return recommendations;
  }

  /**
   * 통계 초기화
   */
  clear() {
    this.measurements.clear();
    logger.info('Performance measurements cleared');
  }

  /**
   * 정리 작업
   */
  destroy() {
    for (const observer of this.observers.values()) {
      observer.disconnect();
    }
    this.observers.clear();
    this.measurements.clear();
  }
}

/**
 * 디바운스 유틸리티
 */
export function debounce(func, wait, immediate = false) {
  let timeout;
  
  return function executedFunction(...args) {
    const later = () => {
      timeout = null;
      if (!immediate) func.apply(this, args);
    };
    
    const callNow = immediate && !timeout;
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
    
    if (callNow) func.apply(this, args);
  };
}

/**
 * 스로틀 유틸리티
 */
export function throttle(func, limit) {
  let inThrottle;
  
  return function executedFunction(...args) {
    if (!inThrottle) {
      func.apply(this, args);
      inThrottle = true;
      setTimeout(() => inThrottle = false, limit);
    }
  };
}

/**
 * 지연된 실행 (RequestAnimationFrame 활용)
 */
export function nextFrame(callback) {
  return requestAnimationFrame(callback);
}

export function nextIdle(callback, timeout = 5000) {
  if ('requestIdleCallback' in window) {
    return requestIdleCallback(callback, { timeout });
  } else {
    return setTimeout(callback, 1);
  }
}

/**
 * 이미지 지연 로딩
 */
export class LazyImageLoader {
  constructor(options = {}) {
    this.threshold = options.threshold || 0.1;
    this.rootMargin = options.rootMargin || '50px';
    
    if ('IntersectionObserver' in window) {
      this.observer = new IntersectionObserver(this.handleIntersection.bind(this), {
        threshold: this.threshold,
        rootMargin: this.rootMargin
      });
    }
  }

  observe(img) {
    if (this.observer) {
      this.observer.observe(img);
    } else {
      // 폴백: 즉시 로드
      this.loadImage(img);
    }
  }

  handleIntersection(entries) {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        this.loadImage(entry.target);
        this.observer.unobserve(entry.target);
      }
    });
  }

  loadImage(img) {
    if (img.dataset.src) {
      img.src = img.dataset.src;
      img.classList.add('loaded');
    }
  }
}

// 전역 인스턴스
export const performanceMonitor = new PerformanceMonitor();
export const lazyImageLoader = new LazyImageLoader();