/**
 * PageRouter - 페이지 네비게이션 및 라우팅 시스템
 *
 * 특징:
 * 1. SPA 라우팅 (Single Page Application)
 * 2. 사이드바 네비게이션 관리
 * 3. URL 해시 기반 라우팅
 * 4. 페이지 전환 애니메이션
 */

class PageRouter {
  constructor() {
    this.currentPage = 'overview';
    this.pages = {
      overview: '대시보드 개요',
      stocks: '주식 분석',
      charts: '차트 분석',
      news: '뉴스 & 감정',
      performance: '모델 성능',
      training: '학습 과정 & XAI',
      spy2025: 'SPY 2025 상반기',
      settings: '설정',
    };

    this.sidebarOpen = false;

    console.log('🧭 PageRouter 생성됨');
  }

  /**
   * 라우터 초기화
   */
  init() {
    this.setupEventListeners();
    this.initializePage();
    console.log('✅ PageRouter 초기화됨');
  }

  /**
   * 이벤트 리스너 설정
   */
  setupEventListeners() {
    // 사이드바 토글 버튼
    const sidebarToggle = document.getElementById('sidebar-toggle');
    if (sidebarToggle) {
      sidebarToggle.addEventListener('click', () => this.toggleSidebar());
    }

    // 사이드바 닫기 버튼
    const sidebarClose = document.getElementById('sidebar-close');
    if (sidebarClose) {
      sidebarClose.addEventListener('click', () => this.closeSidebar());
    }

    // 사이드바 오버레이 클릭
    const sidebarOverlay = document.getElementById('sidebar-overlay');
    if (sidebarOverlay) {
      sidebarOverlay.addEventListener('click', () => this.closeSidebar());
    }

    // 메뉴 아이템 클릭
    const menuItems = document.querySelectorAll('.menu-item');
    menuItems.forEach((item) => {
      item.addEventListener('click', (e) => {
        e.preventDefault();
        const page = item.getAttribute('data-page');
        this.navigateToPage(page);
      });
    });

    // 브라우저 뒤로가기/앞으로가기
    window.addEventListener('hashchange', () => {
      this.handleHashChange();
    });

    // 키보드 단축키
    document.addEventListener('keydown', (e) => {
      this.handleKeyboard(e);
    });

    // 반응형 사이드바 (큰 화면에서 자동 열기)
    window.addEventListener('resize', () => {
      this.handleResize();
    });
  }

  /**
   * 초기 페이지 설정
   */
  initializePage() {
    // URL 해시에서 페이지 확인
    const hash = window.location.hash.substring(1);
    const page = hash && this.pages[hash] ? hash : 'overview';

    this.navigateToPage(page, false);

    // 큰 화면에서 사이드바 자동 열기
    this.handleResize();
  }

  /**
   * 페이지 이동
   */
  navigateToPage(page, updateHash = true) {
    if (!this.pages[page] || page === this.currentPage) {
      return;
    }

    console.log(`🧭 페이지 이동: ${this.currentPage} → ${page}`);

    // URL 해시 업데이트
    if (updateHash) {
      window.location.hash = page;
    }

    // 이전 페이지 숨기기
    this.hidePage(this.currentPage);

    // 새 페이지 표시
    this.showPage(page);

    // 메뉴 활성 상태 업데이트
    this.updateActiveMenu(page);

    // 페이지 제목 업데이트
    this.updatePageTitle(page);

    // 현재 페이지 업데이트
    this.currentPage = page;

    // 모바일에서 페이지 이동 시 사이드바 닫기
    if (window.innerWidth <= 768) {
      this.closeSidebar();
    }

    // 페이지별 초기화 함수 호출
    this.initializePageContent(page);
  }

  /**
   * 페이지 숨기기
   */
  hidePage(page) {
    const pageElement = document.getElementById(`page-${page}`);
    if (pageElement) {
      pageElement.classList.remove('active');
    }
  }

  /**
   * 페이지 표시
   */
  showPage(page) {
    const pageElement = document.getElementById(`page-${page}`);
    if (pageElement) {
      pageElement.classList.add('active');

      // 페이지 표시 후 컴포넌트 업데이트
      setTimeout(() => {
        this.updatePageComponents(page);
      }, 100);
    }
  }

  /**
   * 활성 메뉴 업데이트
   */
  updateActiveMenu(page) {
    // 모든 메뉴 아이템에서 active 클래스 제거
    document.querySelectorAll('.menu-item').forEach((item) => {
      item.classList.remove('active');
    });

    // 현재 페이지 메뉴 아이템에 active 클래스 추가
    const activeMenuItem = document.querySelector(
      `.menu-item[data-page="${page}"]`
    );
    if (activeMenuItem) {
      activeMenuItem.classList.add('active');
    }
  }

  /**
   * 페이지 제목 업데이트
   */
  updatePageTitle(page) {
    const title = this.pages[page];
    const pageTitle = document.getElementById('current-page-title');
    if (pageTitle) {
      pageTitle.textContent = title;
    }

    // 브라우저 제목도 업데이트
    document.title = `${title} - AI Stock Dashboard`;
  }

  /**
   * 페이지별 초기화 (이름 중복 해결)
   */
  initializePageContent(page) {
    console.log(`🔧 페이지 컨텐츠 초기화: ${page}`);
    
    // 즉시 컴포넌트 업데이트 시도
    this.updatePageComponents(page);
    
    // 추가적인 페이지별 초기화 로직
    switch (page) {
      case 'stocks':
        // 주식 페이지는 스톡그리드가 가장 중요
        this.ensureStocksPageReady();
        break;
      case 'charts':
        this.ensureChartsPageReady();
        break;
      case 'performance':
        this.ensurePerformancePageReady();
        break;
      case 'spy2025':
        this.ensureSpy2025PageReady();
        break;
    }
  }
  
  /**
   * 주식 페이지 준비 확인
   */
  ensureStocksPageReady() {
    // StockGrid 제거됨 - S&P 500 컴포넌트만 유지
    console.log('📊 주식 페이지 준비 완료 (S&P 500 전용)');
  }
  
  /**
   * 차트 페이지 준비 확인  
   */
  ensureChartsPageReady() {
    setTimeout(() => {
      if (window.app && window.app.components) {
        const chartContainer = window.app.components.get('chartContainer');
        if (chartContainer) {
          console.log('🔄 ChartContainer 컴포넌트 강제 업데이트');
          chartContainer.update();
        }
      }
    }, 200);
  }
  
  /**
   * SPY 2025 페이지 준비 확인
   */
  ensureSpy2025PageReady() {
    setTimeout(() => {
      if (typeof SPY2025Widget !== 'undefined') {
        console.log('🔄 SPY2025Widget 초기화 확인');
        if (!window.spy2025Widget) {
          console.log('🔄 새 SPY2025Widget 생성');
          window.spy2025Widget = new SPY2025Widget();
          window.spy2025Widget.init();
        } else {
          console.log('⚠️ SPY2025Widget 이미 존재 - 재초기화 건너뜀');
        }
      } else {
        console.warn('⚠️ SPY2025Widget 클래스를 찾을 수 없습니다.');
      }
    }, 300);
  }

  /**
   * 성능 페이지 준비 확인
   */
  ensurePerformancePageReady() {
    setTimeout(() => {
      if (window.app && window.app.components) {
        const metricsPanelPerformance = window.app.components.get('metricsPanelPerformance');
        if (metricsPanelPerformance) {
          console.log('🔄 MetricsPanelPerformance 컴포넌트 강제 업데이트');
          metricsPanelPerformance.update();
        } else {
          console.warn('⚠️ metricsPanelPerformance 컴포넌트를 찾을 수 없습니다');
        }
      }
    }, 200);
  }

  /**
   * 페이지별 컴포넌트 업데이트
   */
  updatePageComponents(page) {
    // 앱 인스턴스가 있을 때만 컴포넌트 업데이트
    if (window.app && window.app.isInitialized) {
      switch (page) {
        case 'overview':
          window.app.updateComponent('metricsPanel');
          window.app.updateComponent('newsPanel');
          break;
        case 'stocks':
          // StockGrid 제거됨 - S&P 500 컴포넌트만 활성화
          break;
        case 'charts':
          window.app.updateComponent('chartContainer');
          break;
        case 'news':
          window.app.updateComponent('newsPanel');
          window.app.updateComponent('newsPanel2');
          break;
        case 'performance':
          window.app.updateComponent('metricsPanelPerformance');
          break;
        case 'training':
          this.initializeTrainingPage();
          break;
        case 'settings':
          this.initializeSettings();
          break;
      }
    }
  }

  /**
   * 사이드바 토글
   */
  toggleSidebar() {
    if (this.sidebarOpen) {
      this.closeSidebar();
    } else {
      this.openSidebar();
    }
  }

  /**
   * 사이드바 열기
   */
  openSidebar() {
    const sidebar = document.getElementById('sidebar');
    const overlay = document.getElementById('sidebar-overlay');
    const mainContent = document.getElementById('main-content');

    if (sidebar) sidebar.classList.add('open');
    if (overlay) overlay.classList.add('active');
    if (mainContent && window.innerWidth > 768) {
      mainContent.classList.add('sidebar-open');
    }

    this.sidebarOpen = true;
    console.log('📂 사이드바 열림');
  }

  /**
   * 사이드바 닫기
   */
  closeSidebar() {
    const sidebar = document.getElementById('sidebar');
    const overlay = document.getElementById('sidebar-overlay');
    const mainContent = document.getElementById('main-content');

    if (sidebar) sidebar.classList.remove('open');
    if (overlay) overlay.classList.remove('active');
    if (mainContent) mainContent.classList.remove('sidebar-open');

    this.sidebarOpen = false;
    console.log('📁 사이드바 닫힘');
  }

  /**
   * 해시 변경 처리
   */
  handleHashChange() {
    const hash = window.location.hash.substring(1);
    const page = hash && this.pages[hash] ? hash : 'overview';

    if (page !== this.currentPage) {
      this.navigateToPage(page, false);
    }
  }

  /**
   * 키보드 단축키 처리
   */
  handleKeyboard(e) {
    // ESC로 사이드바 닫기
    if (e.key === 'Escape' && this.sidebarOpen) {
      this.closeSidebar();
    }

    // Ctrl/Cmd + 숫자로 페이지 이동
    if ((e.ctrlKey || e.metaKey) && e.key >= '1' && e.key <= '6') {
      e.preventDefault();
      const pages = Object.keys(this.pages);
      const pageIndex = parseInt(e.key) - 1;
      if (pages[pageIndex]) {
        this.navigateToPage(pages[pageIndex]);
      }
    }
  }

  /**
   * 화면 크기 변경 처리
   */
  handleResize() {
    const isLargeScreen = window.innerWidth >= 1200;
    const sidebar = document.getElementById('sidebar');
    const mainContent = document.getElementById('main-content');

    if (isLargeScreen) {
      // 큰 화면에서는 사이드바 항상 표시
      if (sidebar) sidebar.classList.add('open');
      if (mainContent) mainContent.classList.add('sidebar-open');
      this.sidebarOpen = true;
    } else {
      // 작은 화면에서는 오버레이 방식
      if (mainContent) mainContent.classList.remove('sidebar-open');
      if (!this.sidebarOpen && sidebar) {
        sidebar.classList.remove('open');
      }
    }
  }

  /**
   * 학습 과정 & XAI 페이지 초기화
   */
  async initializeTrainingPage() {
    try {
      // XAI Visualization 인스턴스가 없으면 생성
      if (!window.xaiVisualization) {
        window.xaiVisualization = new XAIVisualization();
      }

      // XAI 차트 초기화 (중복 방지)
      if (!window.xaiVisualization.isInitialized) {
        await window.xaiVisualization.init();
        window.xaiVisualization.isInitialized = true;
        console.log('✅ 학습 과정 & XAI 페이지 초기화 완료');
      } else {
        // 이미 초기화된 경우 차트 업데이트만
        await window.xaiVisualization.updateCharts();
        console.log('🔄 XAI 차트 업데이트 완료');
      }
    } catch (error) {
      console.error('❌ 학습 과정 & XAI 페이지 초기화 실패:', error);
    }
  }

  /**
   * 설정 페이지 초기화
   */
  initializeSettings() {
    const apiSettings = document.querySelector('.api-settings');
    const refreshSettings = document.querySelector('.refresh-settings');
    const displaySettings = document.querySelector('.display-settings');

    if (apiSettings) {
      apiSettings.innerHTML = this.createApiSettingsHTML();
    }

    if (refreshSettings) {
      refreshSettings.innerHTML = this.createRefreshSettingsHTML();
    }

    if (displaySettings) {
      displaySettings.innerHTML = this.createDisplaySettingsHTML();
    }

    this.setupSettingsEventListeners();
  }

  /**
   * API 설정 HTML 생성
   */
  createApiSettingsHTML() {
    return `
      <h3>🔌 API 설정</h3>
      <div class="setting-item">
        <label for="api-refresh-interval">새로고침 간격 (초)</label>
        <input type="number" id="api-refresh-interval" min="10" max="300" 
               value="${localStorage.getItem('refresh_interval') || '60'}">
      </div>
      <div class="setting-item">
        <label for="api-timeout">API 타임아웃 (초)</label>
        <input type="number" id="api-timeout" min="5" max="30" 
               value="${localStorage.getItem('api_timeout') || '5'}">
      </div>
      <button class="btn btn-primary" onclick="pageRouter.saveApiSettings()">설정 저장</button>
    `;
  }

  /**
   * 새로고침 설정 HTML 생성
   */
  createRefreshSettingsHTML() {
    const stats = JSON.parse(
      localStorage.getItem('refresh_stats') ||
        '{"success": 0, "failure": 0, "avgDuration": 0}'
    );

    return `
      <h3>🔄 새로고침 통계</h3>
      <div class="stats-grid">
        <div class="stat-item">
          <div class="stat-value">${stats.success}</div>
          <div class="stat-label">성공</div>
        </div>
        <div class="stat-item">
          <div class="stat-value">${stats.failure}</div>
          <div class="stat-label">실패</div>
        </div>
        <div class="stat-item">
          <div class="stat-value">${Math.round(stats.avgDuration)}ms</div>
          <div class="stat-label">평균 시간</div>
        </div>
      </div>
      <button class="btn btn-secondary" onclick="pageRouter.clearStats()">통계 초기화</button>
    `;
  }

  /**
   * 디스플레이 설정 HTML 생성
   */
  createDisplaySettingsHTML() {
    return `
      <h3>🎨 디스플레이 설정</h3>
      <div class="setting-item">
        <label>
          <input type="checkbox" id="compact-mode" 
                 ${localStorage.getItem('compact_mode') === 'true' ? 'checked' : ''}>
          컴팩트 모드
        </label>
      </div>
      <div class="setting-item">
        <label>
          <input type="checkbox" id="animation-enabled" 
                 ${localStorage.getItem('animation_enabled') !== 'false' ? 'checked' : ''}>
          애니메이션 활성화
        </label>
      </div>
      <button class="btn btn-primary" onclick="pageRouter.saveDisplaySettings()">설정 저장</button>
    `;
  }

  /**
   * 설정 이벤트 리스너 설정
   */
  setupSettingsEventListeners() {
    // 여기에 추가 이벤트 리스너 설정
  }

  /**
   * API 설정 저장
   */
  saveApiSettings() {
    const refreshInterval = document.getElementById(
      'api-refresh-interval'
    )?.value;
    const apiTimeout = document.getElementById('api-timeout')?.value;

    if (refreshInterval) {
      localStorage.setItem('refresh_interval', refreshInterval);
    }
    if (apiTimeout) {
      localStorage.setItem('api_timeout', apiTimeout);
    }

    // 앱 재시작 알림
    if (window.app) {
      window.app.showStatus(
        '✅ 설정이 저장되었습니다. 새로고침을 권장합니다.',
        'success'
      );
    }

    console.log('💾 API 설정 저장됨');
  }

  /**
   * 디스플레이 설정 저장
   */
  saveDisplaySettings() {
    const compactMode = document.getElementById('compact-mode')?.checked;
    const animationEnabled =
      document.getElementById('animation-enabled')?.checked;

    localStorage.setItem('compact_mode', compactMode);
    localStorage.setItem('animation_enabled', animationEnabled);

    // 설정 적용
    document.body.classList.toggle('compact-mode', compactMode);
    document.body.classList.toggle('no-animation', !animationEnabled);

    if (window.app) {
      window.app.showStatus('✅ 디스플레이 설정이 적용되었습니다.', 'success');
    }

    console.log('🎨 디스플레이 설정 저장됨');
  }

  /**
   * 통계 초기화
   */
  clearStats() {
    localStorage.removeItem('refresh_stats');

    if (window.app) {
      window.app.showStatus('📊 통계가 초기화되었습니다.', 'info');
    }

    // 설정 페이지 재초기화
    this.initializeSettings();
  }

  /**
   * 현재 페이지 가져오기
   */
  getCurrentPage() {
    return this.currentPage;
  }

  /**
   * 페이지 목록 가져오기
   */
  getPages() {
    return { ...this.pages };
  }
}

// 전역 변수로 내보내기
window.PageRouter = PageRouter;
