/**
 * Dashboard Configuration
 * 모든 설정값을 중앙 집중화
 */

export const CONFIG = {
  // 데이터 소스
  DATA_SOURCES: {
    realtime_results: '../data/raw/realtime_results.json',
    sp500_prediction: '../data/raw/sp500_prediction_data.json',
    market_sentiment: '../data/raw/market_sentiment.json',
    model_performance: '../data/raw/model_performance.json',
    trading_volume: '../data/raw/trading_volume.json',
  },

  // API 설정
  API: {
    timeout: 1500,
    maxRetries: 1,
    retryDelay: 300,
  },

  // 캐시 설정
  CACHE: {
    timeout: 30000, // 30초
    maxSize: 50, // 최대 캐시 항목 수
  },

  // UI 설정
  UI: {
    refreshInterval: {
      min: 30000, // 30초
      max: 60000, // 60초
    },
    animationDuration: 300,
    chartHeight: {
      desktop: 400,
      tablet: 300,
      mobile: 250,
    },
  },

  // 차트 설정
  CHARTS: {
    lineWidth: {
      actual: 1.5,
      predicted: 2,
    },
    colors: {
      primary: '#007bff',
      success: '#28a745',
      danger: '#dc3545',
      warning: '#ffc107',
    },
  },

  // 로깅 설정
  LOGGING: {
    level: 'info', // debug, info, warn, error
    enablePerformanceMetrics: true,
    maxLogEntries: 100,
  },

  // 페이지 설정
  PAGES: {
    default: '홈',
    available: ['홈', 'S&P 500', '주식 분석', '뉴스 & 감정', 'XAI 분석'],
  },
};

// 환경별 설정 오버라이드
if (window.location.hostname === 'localhost') {
  CONFIG.LOGGING.level = 'debug';
  CONFIG.CACHE.timeout = 10000; // 개발 시 짧은 캐시
}

// 설정 검증
export function validateConfig() {
  const required = ['DATA_SOURCES', 'API', 'CACHE', 'UI'];
  for (const key of required) {
    if (!CONFIG[key]) {
      throw new Error(`Missing required config: ${key}`);
    }
  }
  console.log('✅ Configuration validated');
}

// 환경 정보
export const ENV = {
  isDevelopment: window.location.hostname === 'localhost',
  isProduction: window.location.hostname !== 'localhost',
  userAgent: navigator.userAgent,
  timestamp: new Date().toISOString(),
};
