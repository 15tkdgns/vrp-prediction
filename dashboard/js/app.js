/**
 * AI Stock Dashboard - 메인 애플리케이션 클래스
 *
 * 설계 원칙:
 * 1. 신뢰성 우선 - 에러 처리 강화
 * 2. 단순성 - 복잡한 기능 제거
 * 3. 가시성 - 명확한 데이터 표시
 */

class DashboardApp {
  constructor() {
    this.isInitialized = false;
    this.components = new Map();
    this.refreshInterval = null;

    // 상태 관리
    this.state = {
      loading: false,
      error: null,
      lastUpdate: null,
    };

    console.log('🚀 Dashboard App 생성됨');
  }

  /**
   * 애플리케이션 초기화 (성능 최적화됨)
   */
  async init() {
    const startTime = Date.now();
    
    try {
      this.showLoadingProgress('시스템 초기화 중...', 0);

      // 1. DOM 준비 대기 (필수 선행)
      await this.waitForDOM();
      this.showLoadingProgress('DOM 준비 완료', 10);

      // 2. 의존성 확인 (빠른 체크)
      await this.checkDependencies();
      this.showLoadingProgress('의존성 확인 완료', 20);

      // 3. 핵심 컴포넌트들을 병렬로 초기화
      this.showLoadingProgress('핵심 모듈 초기화 중...', 30);
      
      const [dataManager, chartManager, sp500Widget] = await Promise.all([
        this.initDataManager(),
        this.initChartManager(),
        this.initSP500Widget()
      ]);
      
      this.dataManager = dataManager;
      this.chartManager = chartManager;
      this.sp500Widget = sp500Widget;
      
      this.showLoadingProgress('핵심 모듈 완료', 60);

      // 4. UI 컴포넌트와 데이터를 병렬로 처리
      this.showLoadingProgress('UI 및 데이터 로딩 중...', 70);
      
      await Promise.all([
        this.initComponents(),
        this.loadInitialDataOptimized()
      ]);
      
      this.showLoadingProgress('컴포넌트 초기화 완료', 90);

      // 5. 이벤트 리스너 설정 (가장 빠름)
      this.setupEventListeners();
      
      this.showLoadingProgress('시스템 준비 완료', 100);

      // 초기화 완료
      this.isInitialized = true;
      
      const duration = Date.now() - startTime;
      console.log(`✅ Dashboard App 초기화 완료 (${duration}ms)`);
      
      // 로딩 완료 후 페이드아웃
      setTimeout(() => {
        this.hideLoading();
        this.showStatus(`🚀 시스템 준비 완료 (${duration}ms)`, 'success');
      }, 300);

      // 자동 새로고침 시작 (백그라운드)
      setTimeout(() => this.startAutoRefresh(), 1000);
      
    } catch (error) {
      console.error('❌ App 초기화 실패:', error);
      this.showError('시스템 초기화에 실패했습니다: ' + error.message);
      this.hideLoading();
    }
  }

  /**
   * DOM 준비 대기
   */
  waitForDOM() {
    return new Promise((resolve) => {
      if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', resolve);
      } else {
        resolve();
      }
    });
  }

  /**
   * 의존성 확인 (Chart.js 등)
   */
  async checkDependencies() {
    // Chart.js 로드 확인
    if (typeof Chart === 'undefined') {
      throw new Error('Chart.js 라이브러리가 로드되지 않았습니다');
    }

    console.log('✅ Chart.js 버전:', Chart.version);
  }

  /**
   * 데이터 매니저 병렬 초기화
   */
  async initDataManager() {
    const dataManager = new DataManager();
    await dataManager.init();
    return dataManager;
  }

  /**
   * 차트 매니저 병렬 초기화
   */
  async initChartManager() {
    const chartManager = new ChartManager();
    await chartManager.init();
    return chartManager;
  }

  /**
   * S&P 500 위젯 병렬 초기화
   */
  async initSP500Widget() {
    const sp500Widget = new SP500Widget();
    await sp500Widget.init();
    return sp500Widget;
  }

  /**
   * 최적화된 초기 데이터 로드
   */
  async loadInitialDataOptimized() {
    try {
      // 중요도 순으로 데이터 로드
      const criticalData = await Promise.allSettled([
        this.loadCriticalData(),
        this.loadChartData()
      ]);

      // 백그라운드에서 추가 데이터 로드
      setTimeout(() => {
        this.loadSecondaryData();
      }, 100);

      console.log('✅ 최적화된 초기 데이터 로드 완료');
    } catch (error) {
      console.warn('⚠️ 일부 데이터 로드 실패:', error);
    }
  }

  /**
   * 중요한 데이터 우선 로드
   */
  async loadCriticalData() {
    // S&P 500 데이터는 위젯에서 자체 로드
    // 여기서는 필수 시스템 데이터만 로드
    if (this.dataManager) {
      await this.dataManager.loadSystemStatus();
    }
  }

  /**
   * 차트 데이터 로드
   */
  async loadChartData() {
    if (this.chartManager) {
      await this.chartManager.loadInitialCharts();
    }
  }

  /**
   * 보조 데이터 백그라운드 로드
   */
  async loadSecondaryData() {
    try {
      if (this.dataManager) {
        await this.dataManager.loadNewsData();
        await this.dataManager.loadMarketData();
      }
    } catch (error) {
      console.warn('보조 데이터 로드 실패:', error);
    }
  }

  /**
   * UI 컴포넌트들 초기화
   */
  async initComponents() {
    const componentsConfig = [
      { name: 'stockGrid', className: 'StockGrid', selector: '.stock-grid' },
      {
        name: 'chartContainer',
        className: 'ChartContainer',
        selector: '.chart-container',
      },
      {
        name: 'metricsPanel',
        className: 'MetricsPanel',
        selector: '.metrics-panel',
      },
      { name: 'newsPanel', className: 'NewsPanel', selector: '.news-panel' },
    ];

    for (const config of componentsConfig) {
      try {
        if (
          window[config.className] &&
          document.querySelector(config.selector)
        ) {
          const component = new window[config.className](
            document.querySelector(config.selector),
            this.dataManager,
            this.chartManager
          );
          this.components.set(config.name, component);
          console.log(`✅ ${config.name} 컴포넌트 초기화됨`);
        }
      } catch (error) {
        console.warn(`⚠️ ${config.name} 컴포넌트 초기화 실패:`, error);
      }
    }
  }

  /**
   * 초기 데이터 로드
   */
  async loadInitialData() {
    this.showLoading('데이터 로딩 중...');

    try {
      // 병렬로 데이터 로드
      const dataPromises = [
        this.dataManager.loadStockData(),
        this.dataManager.loadMetrics(),
        this.dataManager.loadNews(),
        this.dataManager.loadChartData(),
      ];

      await Promise.allSettled(dataPromises);

      // 컴포넌트들에 데이터 전달
      this.updateAllComponents();

      this.state.lastUpdate = new Date();
    } catch (error) {
      console.error('❌ 초기 데이터 로드 실패:', error);
      this.showError('데이터 로딩에 실패했습니다');
    }
  }

  /**
   * 모든 컴포넌트 업데이트
   */
  updateAllComponents() {
    this.components.forEach((component, name) => {
      try {
        if (component && typeof component.update === 'function') {
          component.update();
        }
      } catch (error) {
        console.warn(`⚠️ ${name} 컴포넌트 업데이트 실패:`, error);
      }
    });
  }

  /**
   * 이벤트 리스너 설정
   */
  setupEventListeners() {
    // 새로고침 버튼
    const refreshBtn = document.getElementById('refresh-btn');
    if (refreshBtn) {
      refreshBtn.addEventListener('click', () => this.refresh());
    }

    // 페이지 포커스시 새로고침
    window.addEventListener('focus', () => {
      if (this.isInitialized) {
        this.refresh();
      }
    });

    // 에러 발생시 처리
    window.addEventListener('error', (event) => {
      console.error('전역 에러:', event.error);
      this.showError('예기치 않은 오류가 발생했습니다');
    });
  }

  /**
   * 수동 새로고침
   */
  async refresh() {
    if (this.state.loading) {
      console.log('⏸️ 새로고침 이미 진행 중, 스킵');
      return;
    }

    const startTime = Date.now();

    try {
      this.showLoading('실시간 데이터 업데이트 중...');
      this.state.loading = true;

      // 새로고침 시작 알림
      this.showStatus('🔄 실시간 API 데이터 가져오는 중...', 'info');

      // 데이터 새로고침
      await this.dataManager.refresh();

      this.showStatus('🔧 UI 컴포넌트 업데이트 중...', 'info');

      // S&P 500 위젯 새로고침
      if (this.sp500Widget) {
        await this.sp500Widget.refresh();
      }

      // 컴포넌트 업데이트
      this.updateAllComponents();

      // 성공 처리
      this.state.lastUpdate = new Date();
      const duration = Date.now() - startTime;

      console.log(`✅ 실시간 새로고침 완료 (${duration}ms)`);
      this.showStatus(
        `✅ 실시간 데이터 업데이트 완료 (${duration}ms)`,
        'success'
      );

      // 새로고침 통계 업데이트
      this.updateRefreshStats(true, duration);
    } catch (error) {
      console.error('❌ 새로고침 실패:', error);
      this.showError('❌ 실시간 업데이트 실패: ' + error.message);
      this.updateRefreshStats(false, Date.now() - startTime);
    } finally {
      this.state.loading = false;
      this.hideLoading();

      // 상태 메시지 자동 숨김
      setTimeout(() => {
        const statusEl = document.getElementById('status-message');
        if (statusEl) {
          statusEl.style.display = 'none';
        }
      }, 3000);
    }
  }

  /**
   * 새로고침 통계 업데이트
   */
  updateRefreshStats(success, duration) {
    const stats = JSON.parse(
      localStorage.getItem('refresh_stats') ||
        '{"success": 0, "failure": 0, "avgDuration": 0}'
    );

    if (success) {
      stats.success++;
      stats.avgDuration =
        (stats.avgDuration * (stats.success - 1) + duration) / stats.success;
    } else {
      stats.failure++;
    }

    stats.lastUpdate = new Date().toISOString();
    localStorage.setItem('refresh_stats', JSON.stringify(stats));

    console.log(
      `📊 새로고침 통계: 성공 ${stats.success}회, 실패 ${stats.failure}회, 평균 ${Math.round(stats.avgDuration)}ms`
    );
  }

  /**
   * 자동 새로고침 시작 (스마트 간격 조정)
   */
  startAutoRefresh() {
    if (this.refreshInterval) {
      clearInterval(this.refreshInterval);
    }

    // 환경 변수에서 간격 읽기 (기본값: 60초)
    const refreshInterval =
      parseInt(localStorage.getItem('refresh_interval') || '60') * 1000;

    console.log(`⏰ 자동 새로고침 시작 (${refreshInterval / 1000}초 간격)`);

    this.refreshInterval = setInterval(() => {
      if (this.isInitialized && !this.state.loading && !document.hidden) {
        console.log('🔄 자동 새로고침 실행');
        this.refresh();
      }
    }, refreshInterval);

    // 페이지 가시성 변경 감지
    document.addEventListener('visibilitychange', () => {
      if (!document.hidden && this.isInitialized) {
        console.log('👀 페이지 다시 활성화 - 즉시 새로고침');
        this.refresh();
      }
    });
  }

  /**
   * 자동 새로고침 중지
   */
  stopAutoRefresh() {
    if (this.refreshInterval) {
      clearInterval(this.refreshInterval);
      this.refreshInterval = null;
    }
  }

  /**
   * 로딩 표시
   */
  showLoading(message = '로딩 중...') {
    const loadingEl = document.getElementById('loading-indicator');
    if (loadingEl) {
      loadingEl.textContent = message;
      loadingEl.style.display = 'block';
    }
  }

  /**
   * 진행률과 함께 로딩 표시
   */
  showLoadingProgress(message = '로딩 중...', progress = 0) {
    const loadingEl = document.getElementById('loading-indicator');
    if (loadingEl) {
      // 진행률 바가 없으면 생성
      let progressBar = loadingEl.querySelector('.progress-bar');
      if (!progressBar) {
        loadingEl.innerHTML = `
          <div class="loading-content">
            <div class="loading-text">${message}</div>
            <div class="progress-container">
              <div class="progress-bar"></div>
            </div>
            <div class="loading-percentage">0%</div>
          </div>
        `;
        progressBar = loadingEl.querySelector('.progress-bar');
        
        // 진행률 바 스타일 추가
        this.addProgressBarStyles();
      }
      
      // 텍스트와 진행률 업데이트
      const textEl = loadingEl.querySelector('.loading-text');
      const percentEl = loadingEl.querySelector('.loading-percentage');
      
      if (textEl) textEl.textContent = message;
      if (percentEl) percentEl.textContent = `${progress}%`;
      if (progressBar) {
        progressBar.style.width = `${progress}%`;
        progressBar.style.transition = 'width 0.3s ease';
      }
      
      loadingEl.style.display = 'block';
    }
  }

  /**
   * 진행률 바 스타일 추가
   */
  addProgressBarStyles() {
    if (document.getElementById('progress-bar-styles')) return;
    
    const style = document.createElement('style');
    style.id = 'progress-bar-styles';
    style.textContent = `
      .loading-content {
        text-align: center;
        padding: 2rem;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 8px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        min-width: 300px;
      }
      
      .loading-text {
        font-size: 1.1rem;
        color: #333;
        margin-bottom: 1rem;
        font-weight: 500;
      }
      
      .progress-container {
        width: 100%;
        height: 8px;
        background: #e9ecef;
        border-radius: 4px;
        overflow: hidden;
        margin: 1rem 0;
      }
      
      .progress-bar {
        height: 100%;
        background: linear-gradient(90deg, #007bff, #0056b3);
        border-radius: 4px;
        width: 0%;
        transition: width 0.3s ease;
      }
      
      .loading-percentage {
        font-size: 0.9rem;
        color: #6c757d;
        font-weight: 600;
      }
      
      #loading-indicator {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.5);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 10000;
      }
    `;
    document.head.appendChild(style);
  }

  /**
   * 로딩 숨김
   */
  hideLoading() {
    const loadingEl = document.getElementById('loading-indicator');
    if (loadingEl) {
      loadingEl.style.opacity = '0';
      loadingEl.style.transition = 'opacity 0.3s ease';
      setTimeout(() => {
        loadingEl.style.display = 'none';
        loadingEl.style.opacity = '1';
      }, 300);
    }
  }

  /**
   * 상태 메시지 표시
   */
  showStatus(message, type = 'info') {
    const statusEl = document.getElementById('status-message');
    if (statusEl) {
      statusEl.textContent = message;
      statusEl.className = `status-message ${type}`;
      statusEl.style.display = 'block';

      // 3초 후 자동 숨김
      setTimeout(() => {
        statusEl.style.display = 'none';
      }, 3000);
    }
    console.log(`📢 상태: ${message}`);
  }

  /**
   * 에러 메시지 표시
   */
  showError(message) {
    this.showStatus(message, 'error');
    this.state.error = message;
  }

  /**
   * 앱 정리
   */
  destroy() {
    this.stopAutoRefresh();

    // 컴포넌트들 정리
    this.components.forEach((component) => {
      if (component && typeof component.destroy === 'function') {
        component.destroy();
      }
    });

    this.components.clear();
    this.isInitialized = false;

    console.log('🧹 Dashboard App 정리됨');
  }

  /**
   * 디버그 정보 출력
   */
  getDebugInfo() {
    return {
      initialized: this.isInitialized,
      state: this.state,
      components: Array.from(this.components.keys()),
      dataManager: this.dataManager?.getDebugInfo(),
      chartManager: this.chartManager?.getDebugInfo(),
    };
  }
}

// 전역 변수로 내보내기
window.DashboardApp = DashboardApp;
