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

    console.log('Dashboard App 생성됨');
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
        this.initSP500Widget(),
      ]);

      this.dataManager = dataManager;
      this.chartManager = chartManager;
      this.sp500Widget = sp500Widget;

      this.showLoadingProgress('핵심 모듈 완료', 60);

      // 4. UI 컴포넌트와 데이터를 병렬로 처리
      this.showLoadingProgress('UI 및 데이터 로딩 중...', 70);

      // Use Promise.allSettled to prevent one failure from blocking everything
      const results = await Promise.allSettled([
        this.initComponents(),
        this.loadInitialDataOptimized(),
      ]);
      
      // Log any failures but don't let them block the loading
      results.forEach((result, index) => {
        if (result.status === 'rejected') {
          const taskName = index === 0 ? 'initComponents' : 'loadInitialDataOptimized';
          console.warn(`⚠️ ${taskName} failed:`, result.reason);
        }
      });

      this.showLoadingProgress('컴포넌트 초기화 완료', 90);

      // 5. 이벤트 리스너 설정 (가장 빠름)
      this.setupEventListeners();

      this.showLoadingProgress('시스템 준비 완료', 100);

      // 초기화 완료
      this.isInitialized = true;

      const duration = Date.now() - startTime;
      console.log(`Dashboard App 초기화 완료 (${duration}ms)`);

      // 로딩 완료 후 페이드아웃
      setTimeout(() => {
        this.hideLoading();
        this.showStatus(`시스템 준비 완료 (${duration}ms)`, 'success');

        // 초기 로딩 완료 후 첫 카운트다운 시작
        setTimeout(() => {
          this.startRefreshCountdown();
        }, 2000); // 2초 후 카운트다운 시작
      }, 300);

      // 자동 새로고침 시작 (백그라운드)
      setTimeout(() => this.startAutoRefresh(), 1000);
    } catch (error) {
      console.error('App 초기화 실패:', error);
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

    // APIService 로드 확인
    if (typeof APIService === 'undefined') {
      console.warn('⚠️ APIService가 정의되지 않음 - 강제 초기화 시도');
      // APIService가 없으면 간단한 폴백 생성
      window.APIService = class APIService {
        constructor() {
          console.log('🔄 Fallback APIService 생성됨');
        }
        async getSP500Current() {
          console.log('🌐 Fallback API 호출');
          const response = await fetch('http://localhost:8090/api/sp500-predictions');
          const data = await response.json();
          const sp500 = data.predictions?.find(p => p.symbol === '^GSPC');
          return sp500 ? {
            current: sp500.current_price,
            change: (sp500.technical_indicators?.price_change || 0) * 100,
            timestamp: data.timestamp,
            source: 'Local API'
          } : null;
        }
      };
      console.log('✅ Fallback APIService 생성 완료');
    }

    console.log('Chart.js 버전:', Chart.version);
    console.log('APIService 사용 가능:', typeof APIService !== 'undefined');
  }

  /**
   * 데이터 매니저 병렬 초기화
   */
  async initDataManager() {
    const dataManager = new OptimizedDataManager();
    console.log('OptimizedDataManager 초기화 완료');
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
    // SP500Widget 클래스가 로드될 때까지 대기
    let retries = 0;
    const maxRetries = 50; // 5초 최대 대기

    while (!window.SP500Widget && retries < maxRetries) {
      console.log(
        `⏳ SP500Widget 로딩 대기 중... (${retries + 1}/${maxRetries})`
      );
      await new Promise((resolve) => setTimeout(resolve, 100));
      retries++;
    }

    if (!window.SP500Widget) {
      console.error('SP500Widget 클래스를 로드할 수 없습니다.');
      throw new Error('SP500Widget is not defined after waiting');
    }

    console.log('SP500Widget 클래스 로드 확인됨');
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
        this.loadChartData(),
      ]);

      // 주식 데이터 로드 후 즉시 UI 업데이트
      this.updateAllComponents();

      // 백그라운드에서 추가 데이터 로드 (비동기)
      Promise.resolve().then(() => {
        this.loadSecondaryData();
      });

      console.log('최적화된 초기 데이터 로드 완료');
    } catch (error) {
      console.warn('일부 데이터 로드 실패:', error);
    }
  }

  /**
   * 중요한 데이터 우선 로드
   */
  async loadCriticalData() {
    if (this.dataManager) {
      console.log('주식 데이터 로딩 시작...');

      // 주식 데이터를 가장 먼저 로드 (사용자가 가장 먼저 보는 데이터)
      await this.dataManager.loadStockData();

      console.log('주식 데이터 로드 완료');
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
        console.log('백그라운드 데이터 로딩...');

        // 비중요 데이터들을 병렬로 로드
        await Promise.allSettled([this.dataManager.loadNewsData()]);

        // 컴포넌트 업데이트
        this.updateAllComponents();

        console.log('백그라운드 데이터 로딩 완료');
      }
    } catch (error) {
      console.warn('보조 데이터 로드 실패:', error);
    }
  }

  /**
   * UI 컴포넌트들 초기화 (향상됨 - 모든 페이지의 컴포넌트 초기화)
   */
  async initComponents() {
    const componentsConfig = [
      {
        name: 'chartContainer',
        className: 'ChartContainer',
        selector: '.chart-container',
      },
      {
        name: 'metricsPanel',
        className: 'MetricsPanel',
        selector: '#page-overview .metrics-panel',
      },
      {
        name: 'metricsPanelPerformance',
        className: 'MetricsPanel',
        selector: '#page-performance .metrics-panel',
      },
      { name: 'newsPanel', className: 'NewsPanel', selector: '#page-overview .news-panel' },
      { name: 'newsPanel2', className: 'NewsPanel', selector: '#page-news .news-panel.enhanced' },
    ];

    for (const config of componentsConfig) {
      try {
        const element = document.querySelector(config.selector);
        if (window[config.className] && element) {
          const component = new window[config.className](
            element,
            this.dataManager,
            this.chartManager
          );
          this.components.set(config.name, component);
          console.log(`✅ ${config.name} 컴포넌트 초기화됨 (${config.selector})`);
        } else {
          console.warn(`⚠️ ${config.name} 초기화 스킵: 클래스=${!!window[config.className]}, 요소=${!!element}`);
        }
      } catch (error) {
        console.warn(`❌ ${config.name} 컴포넌트 초기화 실패:`, error);
      }
    }
    
    console.log(`📊 총 ${this.components.size}개 컴포넌트 초기화 완료`);
  }

  /**
   * 초기 데이터 로드
   */
  async loadInitialData() {
    this.showLoading('데이터 로딩 중...');

    try {
      // 병렬로 데이터 로드
      const dataPromises = [this.dataManager.loadAllData()];

      await Promise.allSettled(dataPromises);

      // 컴포넌트들에 데이터 전달
      this.updateAllComponents();

      this.state.lastUpdate = new Date();
    } catch (error) {
      console.error('초기 데이터 로드 실패:', error);
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
        console.warn(`${name} 컴포넌트 업데이트 실패:`, error);
      }
    });
  }

  /**
   * 특정 컴포넌트 업데이트 (페이지 라우터에서 호출)
   */
  updateComponent(componentName) {
    try {
      const component = this.components.get(componentName);
      if (component && typeof component.update === 'function') {
        console.log(`${componentName} 컴포넌트 업데이트`);
        component.update();
      } else {
        console.warn(`컴포넌트 '${componentName}'을 찾을 수 없거나 업데이트 메서드가 없습니다`);
      }
    } catch (error) {
      console.warn(`${componentName} 컴포넌트 업데이트 실패:`, error);
    }
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
      console.log('새로고침 이미 진행 중, 스킵');
      return;
    }

    const startTime = Date.now();

    try {
      this.showLoading('실시간 데이터 업데이트 중...');
      this.state.loading = true;

      // 새로고침 시작 알림
      this.showStatus('실시간 API 데이터 가져오는 중...', 'info');

      // 데이터 새로고침
      await this.dataManager.loadAllData();

      this.showStatus('UI 컴포넌트 업데이트 중...', 'info');

      // S&P 500 위젯 새로고침
      if (this.sp500Widget) {
        await this.sp500Widget.refresh();
      }

      // 컴포넌트 업데이트
      this.updateAllComponents();

      // 성공 처리
      this.state.lastUpdate = new Date();
      const duration = Date.now() - startTime;

      console.log(`실시간 새로고침 완료 (${duration}ms)`);
      this.showStatus(
        `✅ 실시간 데이터 업데이트 완료 (${duration}ms)`,
        'success'
      );

      // 자동 새로고침 카운트다운 시작
      this.startRefreshCountdown();

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
   * 자동 새로고침 카운트다운 시작
   */
  startRefreshCountdown() {
    // 기존 카운트다운 정리
    if (this.countdownInterval) {
      clearInterval(this.countdownInterval);
    }

    // 랜덤 간격 (3-5분) - 성능 최적화
    const randomInterval = Math.floor(Math.random() * 121) + 180; // 180-300초 (3-5분)
    let remainingSeconds = randomInterval;

    const minutes = Math.floor(remainingSeconds / 60);
    const seconds = remainingSeconds % 60;
    const timeDisplay = minutes > 0 ? `${minutes}분 ${seconds}초` : `${seconds}초`;
    
    console.log(`다음 새로고침까지 ${timeDisplay}`);

    // 초기 메시지 표시
    this.showStatus(`다음 새로고침까지 ${timeDisplay}`, 'info');

    this.countdownInterval = setInterval(() => {
      remainingSeconds--;

      if (remainingSeconds > 0) {
        // 카운트다운 업데이트
        const minutes = Math.floor(remainingSeconds / 60);
        const seconds = remainingSeconds % 60;
        const timeDisplay = minutes > 0 ? `${minutes}분 ${seconds}초` : `${seconds}초`;
        
        if (remainingSeconds <= 30) {
          // 마지막 30초는 주황색으로 표시
          this.showStatus(`${timeDisplay} 후 새로고침`, 'warning');
        } else {
          this.showStatus(`다음 새로고침까지 ${timeDisplay}`, 'info');
        }
      } else {
        // 카운트다운 완료 - 자동 새로고침 실행
        clearInterval(this.countdownInterval);
        this.countdownInterval = null;

        if (this.isInitialized && !this.state.loading && !document.hidden) {
          console.log('자동 새로고침 시간 도달');
          this.refresh();
        }
      }
    }, 1000);
  }

  /**
   * 자동 새로고침 시작 (스마트 간격 조정)
   */
  startAutoRefresh() {
    if (this.refreshInterval) {
      clearInterval(this.refreshInterval);
    }

    // 첫 로딩 완료 후 카운트다운 시작
    console.log('자동 새로고침 시스템 준비 완료');

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

    if (this.countdownInterval) {
      clearInterval(this.countdownInterval);
      this.countdownInterval = null;
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
  showStatus(message, type = 'info', autoHide = true) {
    const statusEl = document.getElementById('status-message');
    if (statusEl) {
      statusEl.textContent = message;
      statusEl.className = `status-message ${type}`;
      statusEl.style.display = 'block';

      // 카운트다운 메시지가 아닐 때만 자동 숨김
      if (
        autoHide &&
        !message.includes('다음 새로고침까지') &&
        !message.includes('초 후 새로고침')
      ) {
        setTimeout(() => {
          statusEl.style.display = 'none';
        }, 3000);
      }
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
      dataManager: 'OptimizedDataManager',
      chartManager: this.chartManager?.getDebugInfo(),
    };
  }
}

// 전역 변수로 내보내기
window.DashboardApp = DashboardApp;
