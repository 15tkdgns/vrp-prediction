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
   * 애플리케이션 초기화
   */
  async init() {
    try {
      this.showLoading('시스템 초기화 중...');

      // 1. DOM 준비 대기
      await this.waitForDOM();

      // 2. 의존성 확인
      await this.checkDependencies();

      // 3. 데이터 매니저 초기화
      this.dataManager = new DataManager();
      await this.dataManager.init();

      // 4. 차트 매니저 초기화
      this.chartManager = new ChartManager();
      await this.chartManager.init();

      // 5. UI 컴포넌트 초기화
      await this.initComponents();

      // 6. 초기 데이터 로드
      await this.loadInitialData();

      // 7. 이벤트 리스너 설정
      this.setupEventListeners();

      this.isInitialized = true;
      this.hideLoading();

      console.log('✅ Dashboard App 초기화 완료');
      this.showStatus('시스템 준비 완료', 'success');

      // 선택적 자동 새로고침 (60초)
      this.startAutoRefresh();
    } catch (error) {
      console.error('❌ App 초기화 실패:', error);
      this.showError('시스템 초기화에 실패했습니다: ' + error.message);
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
   * 로딩 숨김
   */
  hideLoading() {
    const loadingEl = document.getElementById('loading-indicator');
    if (loadingEl) {
      loadingEl.style.display = 'none';
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
