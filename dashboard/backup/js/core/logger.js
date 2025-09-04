/**
 * Centralized Logger Module
 * 통합 로깅 시스템으로 성능 추적 및 디버깅 지원
 */

export class Logger {
  constructor(config = {}) {
    this.level = config.level || 'info';
    this.enablePerformance = config.enablePerformanceMetrics || true;
    this.maxEntries = config.maxLogEntries || 100;
    this.logs = [];
    this.performanceMetrics = new Map();
    
    // 로그 레벨 우선순위
    this.levels = { debug: 0, info: 1, warn: 2, error: 3 };
    this.currentLevel = this.levels[this.level];
  }

  /**
   * 성능 측정 시작
   */
  startPerformance(label) {
    if (!this.enablePerformance) return;
    this.performanceMetrics.set(label, performance.now());
  }

  /**
   * 성능 측정 종료 및 로깅
   */
  endPerformance(label, context = '') {
    if (!this.enablePerformance || !this.performanceMetrics.has(label)) return;
    
    const startTime = this.performanceMetrics.get(label);
    const duration = Math.round(performance.now() - startTime);
    this.performanceMetrics.delete(label);
    
    this.info(`⚡ ${label}: ${duration}ms ${context}`, { performance: true });
    return duration;
  }

  /**
   * 로그 엔트리 생성
   */
  createLogEntry(level, message, data = {}) {
    const entry = {
      timestamp: new Date().toISOString(),
      level,
      message,
      data,
      url: window.location.href,
      userAgent: navigator.userAgent.split(' ')[0] // 간단화된 UA
    };

    // 메모리 관리
    if (this.logs.length >= this.maxEntries) {
      this.logs = this.logs.slice(-this.maxEntries + 10); // 90% 유지
    }
    
    this.logs.push(entry);
    return entry;
  }

  /**
   * 로그 출력 (레벨 필터링)
   */
  log(level, message, data = {}) {
    if (this.levels[level] < this.currentLevel) return;

    const entry = this.createLogEntry(level, message, data);
    
    // 콘솔 출력 (개발 환경에서만)
    if (window.location.hostname === 'localhost') {
      const emoji = { debug: '🔧', info: '📋', warn: '⚠️', error: '❌' }[level];
      const style = {
        debug: 'color: #6c757d',
        info: 'color: #007bff', 
        warn: 'color: #ffc107',
        error: 'color: #dc3545; font-weight: bold'
      }[level];

      console.log(`%c${emoji} ${message}`, style, data.performance ? '' : data);
    }

    return entry;
  }

  debug(message, data) { return this.log('debug', message, data); }
  info(message, data) { return this.log('info', message, data); }
  warn(message, data) { return this.log('warn', message, data); }
  error(message, data) { return this.log('error', message, data); }

  /**
   * 로그 검색
   */
  search(query, level = null) {
    return this.logs.filter(entry => {
      const matchesQuery = entry.message.toLowerCase().includes(query.toLowerCase());
      const matchesLevel = !level || entry.level === level;
      return matchesQuery && matchesLevel;
    });
  }

  /**
   * 성능 리포트 생성
   */
  getPerformanceReport() {
    const perfLogs = this.logs.filter(log => log.data.performance);
    const avgTimes = {};
    
    perfLogs.forEach(log => {
      const match = log.message.match(/⚡ (.+): (\d+)ms/);
      if (match) {
        const [, label, time] = match;
        avgTimes[label] = avgTimes[label] || [];
        avgTimes[label].push(parseInt(time));
      }
    });

    // 평균 계산
    Object.keys(avgTimes).forEach(label => {
      const times = avgTimes[label];
      avgTimes[label] = {
        count: times.length,
        avg: Math.round(times.reduce((a, b) => a + b, 0) / times.length),
        min: Math.min(...times),
        max: Math.max(...times)
      };
    });

    return avgTimes;
  }

  /**
   * 로그 내보내기 (디버깅용)
   */
  export() {
    return {
      config: { level: this.level, enablePerformance: this.enablePerformance },
      logs: this.logs,
      performance: this.getPerformanceReport(),
      timestamp: new Date().toISOString()
    };
  }

  /**
   * 로그 초기화
   */
  clear() {
    this.logs = [];
    this.performanceMetrics.clear();
    this.info('Logger cleared');
  }
}

// 글로벌 인스턴스 (싱글톤 패턴)
export const logger = new Logger({
  level: window.location.hostname === 'localhost' ? 'debug' : 'info',
  enablePerformanceMetrics: true,
  maxLogEntries: 100
});