/**
 * UI 컴포넌트들 - 재사용 가능한 컴포넌트들
 *
 * 포함된 컴포넌트:
 * 1. StockGrid - 주요 4개 종목 표시
 * 2. ChartContainer - 차트 렌더링
 * 3. MetricsPanel - 성능 지표 표시
 * 4. NewsPanel - 뉴스/감정 분석
 */

/**
 * 기본 컴포넌트 클래스
 */
class BaseComponent {
  constructor(element, dataManager, chartManager) {
    this.element = element;
    this.dataManager = dataManager;
    this.chartManager = chartManager;
    this.isInitialized = false;
  }

  /**
   * 에러 표시
   */
  showError(message) {
    if (this.element) {
      this.element.innerHTML = `
        <div class="component-error">
          <div class="error-icon">⚠️</div>
          <div class="error-text">${message}</div>
        </div>
      `;
    }
  }

  /**
   * 로딩 표시
   */
  showLoading(message = '로딩 중...') {
    if (this.element) {
      this.element.innerHTML = `
        <div class="component-loading">
          <div class="loading-spinner"></div>
          <div class="loading-text">${message}</div>
        </div>
      `;
    }
  }

  /**
   * 정리
   */
  destroy() {
    this.isInitialized = false;
  }
}

/**
 * StockGrid - 주요 4개 종목 표시 컴포넌트
 */
class StockGrid extends BaseComponent {
  constructor(element, dataManager, chartManager) {
    super(element, dataManager, chartManager);
    this.stocks = [];
    this.init();
  }

  async init() {
    try {
      this.showLoading('주식 데이터 로딩 중...');
      this.isInitialized = true;
      console.log('✅ StockGrid 초기화됨');
    } catch (error) {
      console.error('❌ StockGrid 초기화 실패:', error);
      this.showError('주식 그리드 초기화 실패');
    }
  }

  async update() {
    if (!this.isInitialized) return;

    try {
      this.stocks = this.dataManager.data.stocks || [];
      this.render();
    } catch (error) {
      console.error('❌ StockGrid 업데이트 실패:', error);
      this.showError('주식 데이터 업데이트 실패');
    }
  }

  render() {
    if (!this.element) return;

    if (this.stocks.length === 0) {
      this.showError('주식 데이터가 없습니다');
      return;
    }

    const stockCards = this.stocks
      .map((stock) => this.createStockCard(stock))
      .join('');

    this.element.innerHTML = `
      <div class="stock-grid-container">
        <h2 class="section-title">주요 종목 Top 4 - 실시간 가격 & 예측</h2>
        <div class="stock-cards">
          ${stockCards}
        </div>
      </div>
    `;

    // 미니 차트들 생성
    this.stocks.forEach((stock, index) => {
      const chartId = `stock-mini-${stock.symbol.toLowerCase()}`;
      setTimeout(() => {
        this.chartManager.createStockPriceChart(chartId, stock);
      }, 100 * index);
    });
  }

  createStockCard(stock) {
    const change = stock.technical_indicators?.price_change || 0;
    const changePercent = (change * 100).toFixed(2);
    const changeClass = change >= 0 ? 'positive' : 'negative';
    const direction = stock.predicted_direction || 'neutral';
    const confidence = ((stock.confidence || 0.5) * 100).toFixed(0);

    return `
      <div class="stock-card">
        <div class="stock-header">
          <div class="stock-symbol">${stock.symbol}</div>
          <div class="stock-prediction ${direction}">
            ${direction === 'up' ? '📈' : direction === 'down' ? '📉' : '➡️'}
            ${direction.toUpperCase()}
          </div>
        </div>
        
        <div class="stock-price">
          $${stock.current_price?.toFixed(2) || '0.00'}
        </div>
        
        <div class="stock-change ${changeClass}">
          ${change >= 0 ? '+' : ''}${changePercent}%
        </div>
        
        <div class="stock-confidence">
          신뢰도: ${confidence}%
        </div>
        
        <div class="stock-chart">
          <canvas id="stock-mini-${stock.symbol.toLowerCase()}"></canvas>
        </div>
      </div>
    `;
  }
}

/**
 * ChartContainer - 차트 렌더링 컴포넌트
 */
class ChartContainer extends BaseComponent {
  constructor(element, dataManager, chartManager) {
    super(element, dataManager, chartManager);
    this.charts = new Map();
    this.init();
  }

  async init() {
    try {
      this.isInitialized = true;
      console.log('✅ ChartContainer 초기화됨');
    } catch (error) {
      console.error('❌ ChartContainer 초기화 실패:', error);
      this.showError('차트 컨테이너 초기화 실패');
    }
  }

  async update() {
    if (!this.isInitialized) return;

    try {
      this.render();
      this.createCharts();
    } catch (error) {
      console.error('❌ ChartContainer 업데이트 실패:', error);
      this.showError('차트 업데이트 실패');
    }
  }

  render() {
    if (!this.element) return;

    this.element.innerHTML = `
      <div class="chart-container">
        <h2 class="section-title">차트 분석</h2>
        
        <!-- S&P 500 예측 차트 (최상단, 전체 너비) -->
        <div class="chart-item-featured">
          <h3>📈 S&P 500 예측</h3>
          <div class="chart-wrapper-large">
            <canvas id="sp500-prediction-chart"></canvas>
          </div>
        </div>
        
        <!-- 나머지 차트들 (그리드 형태) -->
        <div class="chart-grid">
          
          <div class="chart-item">
            <h3>모델 성능 비교</h3>
            <div class="chart-wrapper">
              <canvas id="model-performance-chart"></canvas>
            </div>
          </div>
          
          <div class="chart-item">
            <h3>거래량 분석</h3>
            <div class="chart-wrapper">
              <canvas id="volume-analysis-chart"></canvas>
            </div>
          </div>
          
          <div class="chart-item">
            <h3>시장 감정 분석</h3>
            <div class="chart-wrapper">
              <canvas id="sentiment-analysis-chart"></canvas>
            </div>
          </div>
          
        </div>
      </div>
    `;
  }

  createCharts() {
    setTimeout(() => {
      // S&P 500 예측 차트
      const sp500Data = {
        labels: ['1월', '2월', '3월', '4월', '5월', '6월'],
        datasets: [
          {
            label: 'S&P 500 예측',
            data: [4200, 4350, 4180, 4420, 4380, 4500],
          },
        ],
      };
      this.chartManager.createLineChart('sp500-prediction-chart', sp500Data);

      // 모델 성능 차트
      const metrics = this.dataManager.data.metrics;
      this.chartManager.createPerformanceChart(
        'model-performance-chart',
        metrics
      );

      // 거래량 분석 차트
      const volumeData = this.dataManager.data.charts?.volume || {
        labels: ['월', '화', '수', '목', '금'],
        data: [120, 150, 80, 200, 175],
      };

      const volumeChartData = {
        labels: volumeData.labels,
        datasets: [
          {
            label: '거래량 (백만)',
            data: volumeData.data || volumeData,
          },
        ],
      };
      this.chartManager.createBarChart(
        'volume-analysis-chart',
        volumeChartData
      );

      // 감정 분석 차트
      const newsData = this.dataManager.data.news[0] || {};
      const sentimentScore = newsData.sentiment_score || 0.15;

      // 감정 데이터 계산
      const positivePercent = Math.max(
        15,
        sentimentScore > 0 ? sentimentScore * 100 : 20
      );
      const negativePercent = Math.max(
        10,
        sentimentScore < 0 ? Math.abs(sentimentScore) * 100 : 15
      );
      const neutralPercent = 100 - positivePercent - negativePercent;

      const sentimentChartData = {
        labels: ['긍정', '중립', '부정'],
        datasets: [
          {
            data: [positivePercent, neutralPercent, negativePercent],
            backgroundColor: [
              '#007bff', // 파란색 (긍정)
              '#28a745', // 초록색 (중립)
              '#dc3545', // 빨간색 (부정)
            ],
            borderColor: [
              '#0056b3', // 진한 파란색
              '#1e7e34', // 진한 초록색
              '#c82333', // 진한 빨간색
            ],
            borderWidth: 2,
          },
        ],
      };
      this.chartManager.createDoughnutChart(
        'sentiment-analysis-chart',
        sentimentChartData
      );
    }, 200);
  }

  destroy() {
    super.destroy();
    // 차트들 정리
    this.charts.forEach((chart, id) => {
      this.chartManager.destroyChart(id);
    });
    this.charts.clear();
  }
}

/**
 * MetricsPanel - 성능 지표 표시 컴포넌트
 */
class MetricsPanel extends BaseComponent {
  constructor(element, dataManager, chartManager) {
    super(element, dataManager, chartManager);
    this.metrics = {};
    this.init();
  }

  async init() {
    try {
      this.isInitialized = true;
      console.log('✅ MetricsPanel 초기화됨');
    } catch (error) {
      console.error('❌ MetricsPanel 초기화 실패:', error);
      this.showError('메트릭스 패널 초기화 실패');
    }
  }

  async update() {
    if (!this.isInitialized) return;

    try {
      this.metrics = this.dataManager.data.metrics || {};
      this.render();
    } catch (error) {
      console.error('❌ MetricsPanel 업데이트 실패:', error);
      this.showError('성능 지표 업데이트 실패');
    }
  }

  render() {
    if (!this.element) return;

    const accuracy = ((this.metrics.accuracy || 0.85) * 100).toFixed(1);
    const precision = ((this.metrics.precision || 0.82) * 100).toFixed(1);
    const recall = ((this.metrics.recall || 0.86) * 100).toFixed(1);
    const f1Score = ((this.metrics.f1_score || 0.84) * 100).toFixed(1);
    const trainingTime = this.metrics.training_time || '2.3분';

    this.element.innerHTML = `
      <div class="metrics-panel">
        <h2 class="section-title">모델 성능 지표</h2>
        <div class="metrics-grid">
          
          <div class="metric-card">
            <div class="metric-icon">🎯</div>
            <div class="metric-label">정확도</div>
            <div class="metric-value">${accuracy}%</div>
          </div>
          
          <div class="metric-card">
            <div class="metric-icon">🔍</div>
            <div class="metric-label">정밀도</div>
            <div class="metric-value">${precision}%</div>
          </div>
          
          <div class="metric-card">
            <div class="metric-icon">📊</div>
            <div class="metric-label">재현율</div>
            <div class="metric-value">${recall}%</div>
          </div>
          
          <div class="metric-card">
            <div class="metric-icon">⚡</div>
            <div class="metric-label">F1 점수</div>
            <div class="metric-value">${f1Score}%</div>
          </div>
          
          <div class="metric-card">
            <div class="metric-icon">⏱️</div>
            <div class="metric-label">학습 시간</div>
            <div class="metric-value">${trainingTime}</div>
          </div>
          
          <div class="metric-card">
            <div class="metric-icon">🔄</div>
            <div class="metric-label">마지막 업데이트</div>
            <div class="metric-value">${this.formatTime(this.metrics.last_updated)}</div>
          </div>
          
        </div>
      </div>
    `;
  }

  formatTime(timestamp) {
    if (!timestamp) return '방금 전';

    try {
      const date = new Date(timestamp);
      const now = new Date();
      const diff = now - date;

      if (diff < 60000) return '방금 전';
      if (diff < 3600000) return `${Math.floor(diff / 60000)}분 전`;
      if (diff < 86400000) return `${Math.floor(diff / 3600000)}시간 전`;

      return date.toLocaleDateString('ko-KR');
    } catch (error) {
      return '알 수 없음';
    }
  }
}

/**
 * NewsPanel - 뉴스/감정 분석 컴포넌트
 */
class NewsPanel extends BaseComponent {
  constructor(element, dataManager, chartManager) {
    super(element, dataManager, chartManager);
    this.newsData = [];
    this.init();
  }

  async init() {
    try {
      this.isInitialized = true;
      console.log('✅ NewsPanel 초기화됨');
    } catch (error) {
      console.error('❌ NewsPanel 초기화 실패:', error);
      this.showError('뉴스 패널 초기화 실패');
    }
  }

  async update() {
    if (!this.isInitialized) return;

    try {
      this.newsData = this.dataManager.data.news || [];
      this.render();
    } catch (error) {
      console.error('❌ NewsPanel 업데이트 실패:', error);
      this.showError('뉴스 데이터 업데이트 실패');
    }
  }

  render() {
    if (!this.element) return;

    const latestNews = this.newsData[0] || {};
    const sentiment = latestNews.overall_sentiment || 'neutral';
    const sentimentScore = latestNews.sentiment_score || 0;
    const confidence = ((latestNews.confidence || 0.8) * 100).toFixed(0);
    const newsCount = latestNews.news_count || 25;

    const sentimentIcon = this.getSentimentIcon(sentiment);
    const sentimentColor = this.getSentimentColor(sentiment);

    this.element.innerHTML = `
      <div class="news-panel">
        <h2 class="section-title">시장 감정 분석</h2>
        <div class="news-content">
          
          <div class="sentiment-summary">
            <div class="sentiment-main">
              <div class="sentiment-icon" style="color: ${sentimentColor}">
                ${sentimentIcon}
              </div>
              <div class="sentiment-info">
                <div class="sentiment-label">전체 감정</div>
                <div class="sentiment-value" style="color: ${sentimentColor}">
                  ${this.getSentimentText(sentiment)}
                </div>
              </div>
            </div>
            
            <div class="sentiment-score">
              점수: ${(sentimentScore * 100).toFixed(1)}
            </div>
          </div>
          
          <div class="news-stats">
            <div class="news-stat">
              <span class="stat-label">분석된 뉴스</span>
              <span class="stat-value">${newsCount}개</span>
            </div>
            <div class="news-stat">
              <span class="stat-label">신뢰도</span>
              <span class="stat-value">${confidence}%</span>
            </div>
            <div class="news-stat">
              <span class="stat-label">업데이트</span>
              <span class="stat-value">${this.formatTime(latestNews.timestamp)}</span>
            </div>
          </div>
          
        </div>
      </div>
    `;
  }

  getSentimentIcon(sentiment) {
    switch (sentiment) {
      case 'positive':
        return '😊';
      case 'negative':
        return '😟';
      case 'neutral':
      default:
        return '😐';
    }
  }

  getSentimentColor(sentiment) {
    switch (sentiment) {
      case 'positive':
        return '#007bff'; // 파란색 (긍정)
      case 'negative':
        return '#dc3545'; // 빨간색 (부정)
      case 'neutral':
      default:
        return '#28a745'; // 초록색 (중립)
    }
  }

  getSentimentText(sentiment) {
    switch (sentiment) {
      case 'positive':
        return '긍정적';
      case 'negative':
        return '부정적';
      case 'neutral':
      default:
        return '중립적';
    }
  }

  formatTime(timestamp) {
    if (!timestamp) return '방금 전';

    try {
      const date = new Date(timestamp);
      return date.toLocaleString('ko-KR', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      });
    } catch (error) {
      return '알 수 없음';
    }
  }
}

// 전역 변수로 내보내기
window.BaseComponent = BaseComponent;
window.StockGrid = StockGrid;
window.ChartContainer = ChartContainer;
window.MetricsPanel = MetricsPanel;
window.NewsPanel = NewsPanel;
