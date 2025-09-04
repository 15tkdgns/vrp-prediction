/**
 * Optimized Dashboard Application
 * 최적화된 대시보드 애플리케이션 - 모듈화, 성능 최적화, 유지보수성 향상
 */

import { CONFIG, validateConfig } from '../config.js';
import { logger } from './core/logger.js';
import { eventBus, EVENTS } from './core/event-bus.js';
import { apiClient } from './core/api-client.js';
import { dataService } from './services/data-service.js';
import { performanceMonitor, debounce, throttle, nextIdle } from './utils/performance.js';

class OptimizedDashboardApp {
  constructor() {
    // 상태 관리
    this.state = {
      initialized: false,
      loading: false,
      currentPage: CONFIG.PAGES.default,
      error: null
    };

    // 컴포넌트 인스턴스
    this.components = new Map();
    this.managers = new Map();
    
    // 이벤트 리스너 정리를 위한 참조
    this.cleanup = [];
    
    // 성능 최적화된 메서드들
    this.debouncedRefresh = debounce(this.refresh.bind(this), 1000);
    this.throttledUpdate = throttle(this.updateAllComponents.bind(this), 100);

    logger.info('OptimizedDashboardApp instance created');
  }

  /**
   * 애플리케이션 초기화
   */
  async initialize() {
    try {
      await performanceMonitor.measureAsync('app_initialization', async () => {
        logger.info('🚀 Starting application initialization...');
        
        // 1. 설정 검증
        this.showLoadingProgress('설정 검증 중...', 5);
        validateConfig();

        // 2. DOM 준비 대기
        this.showLoadingProgress('DOM 준비 중...', 15);
        await this.waitForDOM();

        // 3. 핵심 시스템 초기화
        this.showLoadingProgress('핵심 시스템 초기화...', 25);
        await this.initializeCoreModules();

        // 4. API 연결 확인
        this.showLoadingProgress('API 연결 확인...', 40);
        await this.verifyAPIConnection();

        // 5. 컴포넌트 초기화 (병렬 처리)
        this.showLoadingProgress('컴포넌트 로딩...', 60);
        await this.initializeComponents();

        // 6. 초기 데이터 로드
        this.showLoadingProgress('데이터 로딩...', 80);
        await this.loadInitialData();

        // 7. 이벤트 리스너 설정
        this.showLoadingProgress('이벤트 설정...', 90);
        this.setupEventListeners();

        // 8. 초기화 완료
        this.showLoadingProgress('초기화 완료!', 100);
        this.state.initialized = true;
        this.state.loading = false;

        logger.info('✅ Application initialization completed');
        eventBus.emit(EVENTS.APP_READY);
        
        // 로딩 UI 숨기기
        await this.finalizeInitialization();
      });

    } catch (error) {
      logger.error('❌ Application initialization failed', { error: error.message });
      this.handleInitializationError(error);
    }
  }

  /**
   * DOM 준비 대기
   */
  waitForDOM() {
    return new Promise((resolve) => {
      if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', resolve, { once: true });
      } else {
        resolve();
      }
    });
  }

  /**
   * 핵심 모듈 초기화
   */
  async initializeCoreModules() {
    // API 클라이언트 health check
    const health = await apiClient.healthCheck();
    if (!health.healthy) {
      logger.warn('API health check failed, continuing with fallback mode');
    }

    // 데이터 서비스 이벤트 구독
    this.subscribeToDataEvents();

    // 성능 모니터링 시작
    this.startPerformanceMonitoring();
  }

  /**
   * API 연결 확인
   */
  async verifyAPIConnection() {
    try {
      const status = await apiClient.get('/api/status', { useCache: false });
      logger.info('✅ API connection verified', { 
        status: status.status,
        systems: Object.keys(status.api_systems || {})
      });
    } catch (error) {
      logger.warn('⚠️ API connection failed, using fallback mode', { 
        error: error.message 
      });
    }
  }

  /**
   * 컴포넌트 초기화 (지연 로딩)
   */
  async initializeComponents() {
    const componentConfigs = [
      { name: 'stockGrid', selector: '#page-overview .stock-grid', priority: 1 },
      { name: 'chartContainer', selector: '.chart-container', priority: 2 },
      { name: 'newsWidget', selector: '.news-widget', priority: 3 },
      { name: 'metricsPanel', selector: '.metrics-panel', priority: 3 }
    ];

    // 우선순위별로 컴포넌트 초기화
    const priorityGroups = this.groupByPriority(componentConfigs);
    
    for (const [priority, configs] of priorityGroups.entries()) {
      await Promise.allSettled(
        configs.map(config => this.initializeComponent(config))
      );
      
      // 높은 우선순위 컴포넌트 완료 후 잠시 대기
      if (priority === 1) {
        await new Promise(resolve => setTimeout(resolve, 100));
      }
    }
  }

  /**
   * 개별 컴포넌트 초기화
   */
  async initializeComponent(config) {
    try {
      const element = document.querySelector(config.selector);
      if (!element) {
        logger.debug(`Component element not found: ${config.selector}`);
        return;
      }

      // 동적 컴포넌트 로딩 (예시)
      const ComponentClass = await this.loadComponent(config.name);
      if (ComponentClass) {
        const instance = new ComponentClass(element, {
          dataService,
          eventBus,
          logger
        });
        
        this.components.set(config.name, instance);
        logger.debug(`✅ Component initialized: ${config.name}`);
        
        eventBus.emit(EVENTS.COMPONENT_READY, { name: config.name });
      }

    } catch (error) {
      logger.error(`❌ Failed to initialize component: ${config.name}`, {
        error: error.message
      });
      
      eventBus.emit(EVENTS.COMPONENT_ERROR, { 
        name: config.name, 
        error: error.message 
      });
    }
  }

  /**
   * 컴포넌트 동적 로딩
   */
  async loadComponent(name) {
    const componentMap = {
      stockGrid: () => import('./components/stock-grid.js'),
      chartContainer: () => import('./components/chart-container.js'),
      newsWidget: () => import('./components/news-widget.js'),
      metricsPanel: () => import('./components/metrics-panel.js')
    };

    const loader = componentMap[name];
    if (loader) {
      try {
        const module = await loader();
        return module.default || module[Object.keys(module)[0]];
      } catch (error) {
        logger.warn(`Failed to load component: ${name}`, { error: error.message });
        return null;
      }
    }

    return null;
  }

  /**
   * 초기 데이터 로드 (최적화된)
   */
  async loadInitialData() {
    try {
      // 중요한 데이터 먼저 로드
      const criticalData = await Promise.allSettled([
        dataService.loadStockData(),
        dataService.loadMLPredictions()
      ]);

      // 즉시 UI 업데이트
      this.updateCriticalComponents();

      // 덜 중요한 데이터는 백그라운드에서 로드
      nextIdle(() => {
        this.loadSecondaryData();
      });

      logger.info('Initial data loading completed');

    } catch (error) {
      logger.error('Initial data loading failed', { error: error.message });
    }
  }

  /**
   * 보조 데이터 백그라운드 로드
   */
  async loadSecondaryData() {
    try {
      await Promise.allSettled([
        dataService.loadNewsData(),
        dataService.loadMetricsData()
      ]);

      this.updateSecondaryComponents();
      logger.debug('Secondary data loaded');

    } catch (error) {
      logger.warn('Secondary data loading failed', { error: error.message });
    }
  }

  /**
   * 데이터 이벤트 구독
   */
  subscribeToDataEvents() {
    const unsubscribers = [
      eventBus.on(EVENTS.DATA_UPDATED, this.handleDataUpdate.bind(this)),
      eventBus.on(EVENTS.DATA_ERROR, this.handleDataError.bind(this)),
      eventBus.on(EVENTS.DATA_LOADING_START, this.handleLoadingStart.bind(this)),
      eventBus.on(EVENTS.DATA_LOADING_END, this.handleLoadingEnd.bind(this))
    ];

    this.cleanup.push(...unsubscribers);
  }

  /**
   * 이벤트 리스너 설정
   */
  setupEventListeners() {
    // 페이지 가시성 변경
    const visibilityHandler = () => {
      if (!document.hidden && this.state.initialized) {
        logger.debug('Page became visible, refreshing data');
        this.debouncedRefresh();
      }
    };
    
    document.addEventListener('visibilitychange', visibilityHandler);
    this.cleanup.push(() => 
      document.removeEventListener('visibilitychange', visibilityHandler)
    );

    // 윈도우 리사이즈 (스로틀된)
    const resizeHandler = throttle(() => {
      eventBus.emit('window:resize', {
        width: window.innerWidth,
        height: window.innerHeight
      });
    }, 250);

    window.addEventListener('resize', resizeHandler);
    this.cleanup.push(() => window.removeEventListener('resize', resizeHandler));

    // 에러 핸들링
    window.addEventListener('error', this.handleGlobalError.bind(this));
    window.addEventListener('unhandledrejection', this.handleUnhandledRejection.bind(this));
  }

  /**
   * 성능 모니터링 시작
   */
  startPerformanceMonitoring() {
    // 주기적 성능 체크
    const performanceCheck = () => {
      const recommendations = performanceMonitor.getRecommendations();
      if (recommendations.length > 0) {
        logger.info('Performance recommendations:', { recommendations });
      }
    };

    setInterval(performanceCheck, 60000); // 1분마다
  }

  /**
   * 컴포넌트 업데이트
   */
  updateCriticalComponents() {
    const critical = ['stockGrid', 'chartContainer'];
    critical.forEach(name => {
      const component = this.components.get(name);
      if (component && typeof component.update === 'function') {
        component.update();
      }
    });
  }

  updateSecondaryComponents() {
    const secondary = ['newsWidget', 'metricsPanel'];
    secondary.forEach(name => {
      const component = this.components.get(name);
      if (component && typeof component.update === 'function') {
        component.update();
      }
    });
  }

  updateAllComponents() {
    this.components.forEach((component, name) => {
      if (typeof component.update === 'function') {
        try {
          component.update();
        } catch (error) {
          logger.error(`Component update failed: ${name}`, { error: error.message });
        }
      }
    });
  }

  /**
   * 이벤트 핸들러들
   */
  handleDataUpdate(data) {
    logger.debug(`Data updated: ${data.type}`);
    this.throttledUpdate();
  }

  handleDataError(error) {
    logger.error('Data error occurred', error);
    this.showError(`데이터 로드 실패: ${error.error}`);
  }

  handleLoadingStart(data) {
    // 로딩 상태 UI 업데이트
  }

  handleLoadingEnd(data) {
    // 로딩 완료 UI 업데이트
  }

  handleGlobalError(event) {
    logger.error('Global error caught', {
      message: event.message,
      filename: event.filename,
      line: event.lineno,
      column: event.colno
    });
  }

  handleUnhandledRejection(event) {
    logger.error('Unhandled promise rejection', {
      reason: event.reason
    });
  }

  /**
   * 데이터 새로고침
   */
  async refresh() {
    if (this.state.loading) {
      logger.debug('Refresh already in progress, skipping...');
      return;
    }

    try {
      this.state.loading = true;
      eventBus.emit(EVENTS.REFRESH_TRIGGERED);
      
      const result = await dataService.refreshAllData();
      logger.info('Data refresh completed', result);
      
    } catch (error) {
      logger.error('Data refresh failed', { error: error.message });
      this.showError('데이터 새로고침에 실패했습니다');
      
    } finally {
      this.state.loading = false;
    }
  }

  /**
   * UI 헬퍼 메서드들
   */
  showLoadingProgress(message, progress) {
    const progressBar = document.querySelector('.loading-progress');
    const progressText = document.querySelector('.loading-text');
    
    if (progressBar) {
      progressBar.style.width = `${progress}%`;
    }
    
    if (progressText) {
      progressText.textContent = message;
    }
    
    logger.debug(`Loading: ${message} (${progress}%)`);
  }

  showError(message, duration = 5000) {
    const errorEl = document.querySelector('.error-message') || this.createErrorElement();
    errorEl.textContent = message;
    errorEl.style.display = 'block';
    
    setTimeout(() => {
      errorEl.style.display = 'none';
    }, duration);
  }

  createErrorElement() {
    const errorEl = document.createElement('div');
    errorEl.className = 'error-message';
    errorEl.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      background: #dc3545;
      color: white;
      padding: 12px 20px;
      border-radius: 4px;
      z-index: 9999;
      display: none;
    `;
    document.body.appendChild(errorEl);
    return errorEl;
  }

  hideLoading() {
    const loadingEl = document.querySelector('.loading-overlay');
    if (loadingEl) {
      loadingEl.style.opacity = '0';
      setTimeout(() => {
        loadingEl.style.display = 'none';
      }, 300);
    }
  }

  /**
   * 초기화 완료 처리
   */
  async finalizeInitialization() {
    await new Promise(resolve => setTimeout(resolve, 300));
    this.hideLoading();
    
    // 성능 리포트 출력 (개발 모드에서만)
    if (CONFIG.LOGGING.level === 'debug') {
      const report = performanceMonitor.getReport();
      logger.debug('Performance Report:', report);
    }
  }

  /**
   * 초기화 오류 처리
   */
  handleInitializationError(error) {
    this.state.error = error;
    this.state.loading = false;
    
    this.hideLoading();
    this.showError(`시스템 초기화 실패: ${error.message}`);
    
    // 복구 시도
    setTimeout(() => {
      if (confirm('시스템 초기화에 실패했습니다. 다시 시도하시겠습니까?')) {
        window.location.reload();
      }
    }, 2000);
  }

  /**
   * 유틸리티 메서드들
   */
  groupByPriority(items) {
    const groups = new Map();
    items.forEach(item => {
      const priority = item.priority || 999;
      if (!groups.has(priority)) {
        groups.set(priority, []);
      }
      groups.get(priority).push(item);
    });
    return new Map([...groups.entries()].sort(([a], [b]) => a - b));
  }

  /**
   * 정리 작업
   */
  destroy() {
    logger.info('Destroying application...');
    
    // 이벤트 리스너 정리
    this.cleanup.forEach(fn => fn());
    this.cleanup = [];
    
    // 컴포넌트 정리
    this.components.forEach((component, name) => {
      if (typeof component.destroy === 'function') {
        component.destroy();
      }
    });
    this.components.clear();
    
    // 이벤트 버스 정리
    eventBus.clear();
    
    // 성능 모니터 정리
    performanceMonitor.destroy();
    
    this.state.initialized = false;
  }

  /**
   * 상태 조회
   */
  getStatus() {
    return {
      state: this.state,
      components: Array.from(this.components.keys()),
      performance: performanceMonitor.getReport(),
      cache: dataService.getCacheStatus(),
      memory: performanceMonitor.getMemoryInfo()
    };
  }
}

// 글로벌 앱 인스턴스
const app = new OptimizedDashboardApp();

// DOM 로드 시 자동 초기화
document.addEventListener('DOMContentLoaded', () => {
  app.initialize().catch(error => {
    console.error('Failed to initialize application:', error);
  });
});

// 글로벌 접근을 위해 window에 등록 (디버깅용)
if (typeof window !== 'undefined') {
  window.dashboardApp = app;
}

export default app;