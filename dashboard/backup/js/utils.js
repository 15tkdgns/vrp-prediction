/**
 * Utility Functions and Error Handling
 * 유틸리티 함수와 에러 처리 시스템
 */

import { CONFIG } from '../config.js';

// 로거 클래스
export class Logger {
  constructor() {
    this.logs = [];
    this.maxLogs = CONFIG.LOGGING.maxLogEntries;
    this.level = CONFIG.LOGGING.level;
  }

  log(level, message, data = null) {
    if (this.shouldLog(level)) {
      const logEntry = {
        timestamp: new Date().toISOString(),
        level,
        message,
        data,
      };

      this.logs.push(logEntry);

      // 최대 로그 수 유지
      if (this.logs.length > this.maxLogs) {
        this.logs.shift();
      }

      // 콘솔 출력
      const consoleMethod = this.getConsoleMethod(level);
      if (data) {
        console[consoleMethod](`[${level.toUpperCase()}] ${message}`, data);
      } else {
        console[consoleMethod](`[${level.toUpperCase()}] ${message}`);
      }
    }
  }

  debug(message, data) {
    this.log('debug', message, data);
  }
  info(message, data) {
    this.log('info', message, data);
  }
  warn(message, data) {
    this.log('warn', message, data);
  }
  error(message, data) {
    this.log('error', message, data);
  }

  shouldLog(level) {
    const levels = ['debug', 'info', 'warn', 'error'];
    const currentLevelIndex = levels.indexOf(this.level);
    const messageLevelIndex = levels.indexOf(level);
    return messageLevelIndex >= currentLevelIndex;
  }

  getConsoleMethod(level) {
    const methods = {
      debug: 'debug',
      info: 'info',
      warn: 'warn',
      error: 'error',
    };
    return methods[level] || 'log';
  }

  getLogs(level = null) {
    if (level) {
      return this.logs.filter((log) => log.level === level);
    }
    return [...this.logs];
  }

  clearLogs() {
    this.logs = [];
  }
}

// 전역 로거 인스턴스
export const logger = new Logger();

// 에러 핸들링 클래스
export class ErrorHandler {
  constructor() {
    this.errorCounts = new Map();
    this.setupGlobalErrorHandling();
  }

  setupGlobalErrorHandling() {
    window.addEventListener('error', (event) => {
      this.handleError(event.error, 'Global Error');
    });

    window.addEventListener('unhandledrejection', (event) => {
      this.handleError(event.reason, 'Unhandled Promise Rejection');
    });
  }

  handleError(error, context = 'Unknown') {
    const errorKey = `${context}: ${error.message}`;
    const count = this.errorCounts.get(errorKey) || 0;
    this.errorCounts.set(errorKey, count + 1);

    logger.error(`${context}: ${error.message}`, {
      stack: error.stack,
      count: count + 1,
      timestamp: new Date().toISOString(),
    });

    // 사용자에게 친화적인 에러 메시지 표시
    this.showUserError(error, context);
  }

  showUserError(error, context) {
    // 중복 에러 방지
    const errorKey = `${context}: ${error.message}`;
    if (this.errorCounts.get(errorKey) > 3) {
      return; // 같은 에러가 3번 이상 발생하면 더 이상 표시하지 않음
    }

    const userMessage = this.getUserFriendlyMessage(error, context);
    if (
      window.dashboardApp &&
      typeof window.dashboardApp.showStatus === 'function'
    ) {
      window.dashboardApp.showStatus(userMessage, 'error');
    }
  }

  getUserFriendlyMessage(error, context) {
    const messages = {
      'Network Error': '네트워크 연결을 확인해주세요',
      'JSON Parse Error': '데이터 형식에 오류가 있습니다',
      'API Error': 'API 서버에 일시적인 문제가 있습니다',
      'Timeout Error': '요청 시간이 초과되었습니다',
    };

    for (const [key, message] of Object.entries(messages)) {
      if (error.message.includes(key)) {
        return `⚠️ ${message}`;
      }
    }

    return '⚠️ 일시적인 오류가 발생했습니다';
  }

  getErrorStats() {
    return Array.from(this.errorCounts.entries()).map(([error, count]) => ({
      error,
      count,
    }));
  }
}

// 전역 에러 핸들러
export const errorHandler = new ErrorHandler();

// 성능 모니터링
export class PerformanceMonitor {
  constructor() {
    this.metrics = new Map();
    this.enabled = CONFIG.LOGGING.enablePerformanceMetrics;
  }

  startTimer(name) {
    if (!this.enabled) return;
    this.metrics.set(name, {
      startTime: performance.now(),
      endTime: null,
      duration: null,
    });
  }

  endTimer(name) {
    if (!this.enabled) return;
    const metric = this.metrics.get(name);
    if (metric && !metric.endTime) {
      metric.endTime = performance.now();
      metric.duration = metric.endTime - metric.startTime;

      logger.debug(`Performance: ${name}`, {
        duration: `${metric.duration.toFixed(2)}ms`,
      });
    }
  }

  getMetric(name) {
    return this.metrics.get(name);
  }

  getAllMetrics() {
    return Array.from(this.metrics.entries())
      .map(([name, metric]) => ({
        name,
        duration: metric.duration,
      }))
      .filter((m) => m.duration !== null);
  }

  clearMetrics() {
    this.metrics.clear();
  }
}

// 전역 성능 모니터
export const performanceMonitor = new PerformanceMonitor();

// 유틸리티 함수들
export const Utils = {
  // 안전한 JSON 파싱
  safeJsonParse(text, fallback = null) {
    try {
      return JSON.parse(text);
    } catch (error) {
      logger.warn('JSON parse failed', { text, error: error.message });
      return fallback;
    }
  },

  // 안전한 fetch
  async safeFetch(url, options = {}) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(
        () => controller.abort(),
        CONFIG.API.timeout
      );

      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      return response;
    } catch (error) {
      if (error.name === 'AbortError') {
        throw new Error('Timeout Error: Request timed out');
      }
      throw error;
    }
  },

  // 디바운스
  debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  },

  // 스로틀
  throttle(func, limit) {
    let inThrottle;
    return function executedFunction(...args) {
      if (!inThrottle) {
        func.apply(this, args);
        inThrottle = true;
        setTimeout(() => (inThrottle = false), limit);
      }
    };
  },

  // 깊은 복사
  deepClone(obj) {
    if (obj === null || typeof obj !== 'object') return obj;
    if (obj instanceof Date) return new Date(obj);
    if (obj instanceof Array) return obj.map((item) => this.deepClone(item));
    if (typeof obj === 'object') {
      const clonedObj = {};
      for (const key in obj) {
        if (obj.hasOwnProperty(key)) {
          clonedObj[key] = this.deepClone(obj[key]);
        }
      }
      return clonedObj;
    }
  },

  // 시간 형식화
  formatRelativeTime(timestamp) {
    try {
      const now = new Date();
      const time = new Date(timestamp);
      const diffMs = now - time;

      const units = [
        { name: '년', ms: 365 * 24 * 60 * 60 * 1000 },
        { name: '개월', ms: 30 * 24 * 60 * 60 * 1000 },
        { name: '일', ms: 24 * 60 * 60 * 1000 },
        { name: '시간', ms: 60 * 60 * 1000 },
        { name: '분', ms: 60 * 1000 },
        { name: '초', ms: 1000 },
      ];

      for (const unit of units) {
        const diff = Math.floor(diffMs / unit.ms);
        if (diff > 0) {
          return `${diff}${unit.name} 전`;
        }
      }

      return '방금 전';
    } catch (error) {
      logger.warn('Time formatting failed', {
        timestamp,
        error: error.message,
      });
      return '알 수 없음';
    }
  },

  // 숫자 형식화
  formatNumber(num, decimals = 2) {
    if (typeof num !== 'number' || isNaN(num)) return '0';
    return num.toLocaleString('ko-KR', {
      minimumFractionDigits: decimals,
      maximumFractionDigits: decimals,
    });
  },

  // 퍼센트 형식화
  formatPercent(num, decimals = 1) {
    if (typeof num !== 'number' || isNaN(num)) return '0%';
    return `${(num * 100).toFixed(decimals)}%`;
  },

  // DOM 요소 안전 선택
  safeQuerySelector(selector, parent = document) {
    try {
      return parent.querySelector(selector);
    } catch (error) {
      logger.warn('Query selector failed', { selector, error: error.message });
      return null;
    }
  },

  // 랜덤 ID 생성
  generateId(prefix = 'id') {
    return `${prefix}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  },
};

// 초기화
logger.info('Utils module initialized');
performanceMonitor.startTimer('app_initialization');
