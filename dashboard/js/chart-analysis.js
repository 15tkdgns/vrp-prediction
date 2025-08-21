/**
 * ChartAnalysisWidget - 고급 차트 분석 위젯
 *
 * 특징:
 * 1. 다양한 기술적 지표 차트
 * 2. 실시간 차트 데이터
 * 3. 인터랙티브 차트 기능
 * 4. 주식별 상세 분석
 */

class ChartAnalysisWidget {
  constructor(element, dataManager, chartManager) {
    this.element = element;
    this.dataManager = dataManager;
    this.chartManager = chartManager;
    this.charts = new Map();
    this.currentStock = 'AAPL';
    this.analysisData = {};

    console.log('📈 ChartAnalysisWidget 생성됨');
  }

  /**
   * 위젯 초기화
   */
  async init() {
    try {
      await this.loadAnalysisData();
      this.render();
      console.log('✅ ChartAnalysisWidget 초기화됨');
    } catch (error) {
      console.error('❌ ChartAnalysisWidget 초기화 실패:', error);
      this.showError('차트 분석 위젯 초기화 실패');
    }
  }

  /**
   * 분석 데이터 로드
   */
  async loadAnalysisData() {
    try {
      // 주식 데이터 로드
      const stockData = this.dataManager.data.stocks || [];

      // 차트 데이터 로드
      const chartData = this.dataManager.data.charts || {};

      // 분석 데이터 구성
      this.analysisData = {
        stocks: stockData,
        technicalIndicators: this.generateTechnicalIndicators(stockData),
        volumeAnalysis: chartData.volume || this.getMockVolumeData(),
        trendAnalysis: chartData.trend || this.getMockTrendData(),
        correlationMatrix: this.generateCorrelationMatrix(stockData),
      };

      console.log('📊 차트 분석 데이터 로드됨');
    } catch (error) {
      console.error('❌ 분석 데이터 로드 실패:', error);
      // 폴백 데이터 사용
      this.analysisData = this.getMockAnalysisData();
    }
  }

  /**
   * 위젯 렌더링
   */
  render() {
    if (!this.element) return;

    this.element.innerHTML = `
      <div class="chart-analysis-widget">
        <!-- 차트 분석 헤더 -->
        <div class="analysis-header">
          <h3>📋 차트 분석</h3>
          <div class="analysis-controls">
            <select id="stock-selector" class="stock-selector">
              ${this.renderStockOptions()}
            </select>
            <select id="timeframe-selector" class="timeframe-selector">
              <option value="1d">1일</option>
              <option value="1w" selected>1주</option>
              <option value="1m">1개월</option>
              <option value="3m">3개월</option>
            </select>
            <button id="refresh-analysis" class="btn btn-sm btn-primary">🔄</button>
          </div>
        </div>

        <!-- 주요 지표 요약 -->
        <div class="indicators-summary">
          <div class="indicator-card">
            <div class="indicator-title">RSI (14)</div>
            <div class="indicator-value rsi" id="current-rsi">65.3</div>
            <div class="indicator-status">과매수</div>
          </div>
          <div class="indicator-card">
            <div class="indicator-title">MACD</div>
            <div class="indicator-value macd" id="current-macd">+0.24</div>
            <div class="indicator-status">상승추세</div>
          </div>
          <div class="indicator-card">
            <div class="indicator-title">볼린저 밴드</div>
            <div class="indicator-value bollinger" id="current-bollinger">상단</div>
            <div class="indicator-status">저항선 근접</div>
          </div>
          <div class="indicator-card">
            <div class="indicator-title">거래량</div>
            <div class="indicator-value volume" id="current-volume">+15%</div>
            <div class="indicator-status">평균 대비 증가</div>
          </div>
        </div>

        <!-- 차트 그리드 -->
        <div class="charts-grid">
          <!-- 가격 및 볼린저 밴드 차트 -->
          <div class="chart-container">
            <div class="chart-header">
              <h4>📈 가격 & 볼린저 밴드</h4>
              <div class="chart-legend">
                <span class="legend-item"><span class="legend-color" style="background: #007bff;"></span> 가격</span>
                <span class="legend-item"><span class="legend-color" style="background: #28a745;"></span> 상단밴드</span>
                <span class="legend-item"><span class="legend-color" style="background: #dc3545;"></span> 하단밴드</span>
              </div>
            </div>
            <div class="chart-wrapper">
              <canvas id="price-bollinger-chart"></canvas>
            </div>
          </div>

          <!-- RSI 차트 -->
          <div class="chart-container">
            <div class="chart-header">
              <h4>📊 RSI (상대강도지수)</h4>
              <div class="chart-legend">
                <span class="legend-item"><span class="legend-color" style="background: #ffc107;"></span> RSI</span>
                <span class="legend-text">과매수: 70+ | 과매도: 30-</span>
              </div>
            </div>
            <div class="chart-wrapper">
              <canvas id="rsi-chart"></canvas>
            </div>
          </div>

          <!-- MACD 차트 -->
          <div class="chart-container">
            <div class="chart-header">
              <h4>📉 MACD</h4>
              <div class="chart-legend">
                <span class="legend-item"><span class="legend-color" style="background: #17a2b8;"></span> MACD</span>
                <span class="legend-item"><span class="legend-color" style="background: #6c757d;"></span> Signal</span>
              </div>
            </div>
            <div class="chart-wrapper">
              <canvas id="macd-chart"></canvas>
            </div>
          </div>

          <!-- 거래량 분석 -->
          <div class="chart-container">
            <div class="chart-header">
              <h4>📊 거래량 분석</h4>
              <div class="chart-legend">
                <span class="legend-item"><span class="legend-color" style="background: #fd7e14;"></span> 거래량</span>
                <span class="legend-item"><span class="legend-color" style="background: #e83e8c;"></span> 평균</span>
              </div>
            </div>
            <div class="chart-wrapper">
              <canvas id="volume-analysis-chart"></canvas>
            </div>
          </div>

          <!-- 상관관계 매트릭스 -->
          <div class="chart-container full-width">
            <div class="chart-header">
              <h4>🔗 주식간 상관관계 매트릭스</h4>
              <div class="chart-legend">
                <span class="legend-text">강한 양의 상관관계: 0.7+ | 강한 음의 상관관계: -0.7-</span>
              </div>
            </div>
            <div class="chart-wrapper">
              <canvas id="correlation-matrix-chart"></canvas>
            </div>
          </div>

          <!-- 기술적 분석 요약 -->
          <div class="analysis-summary full-width">
            <h4>📝 기술적 분석 요약</h4>
            <div class="analysis-insights" id="analysis-insights">
              <!-- 분석 결과가 여기에 표시됩니다 -->
            </div>
          </div>
        </div>
      </div>
    `;

    // 이벤트 리스너 설정
    this.setupEventListeners();

    // 차트 생성
    setTimeout(() => {
      this.createCharts();
    }, 100);
  }

  /**
   * 주식 옵션 렌더링
   */
  renderStockOptions() {
    const stocks = this.analysisData.stocks || [];
    return stocks
      .map(
        (stock) =>
          `<option value="${stock.symbol}" ${stock.symbol === this.currentStock ? 'selected' : ''}>${stock.symbol}</option>`
      )
      .join('');
  }

  /**
   * 이벤트 리스너 설정
   */
  setupEventListeners() {
    // 주식 선택 변경
    const stockSelector = document.getElementById('stock-selector');
    if (stockSelector) {
      stockSelector.addEventListener('change', (e) => {
        this.currentStock = e.target.value;
        this.updateAnalysis();
      });
    }

    // 시간대 선택 변경
    const timeframeSelector = document.getElementById('timeframe-selector');
    if (timeframeSelector) {
      timeframeSelector.addEventListener('change', () => {
        this.updateAnalysis();
      });
    }

    // 새로고침 버튼
    const refreshBtn = document.getElementById('refresh-analysis');
    if (refreshBtn) {
      refreshBtn.addEventListener('click', () => {
        this.refresh();
      });
    }
  }

  /**
   * 차트 생성
   */
  createCharts() {
    this.createPriceBollingerChart();
    this.createRSIChart();
    this.createMACDChart();
    this.createVolumeChart();
    this.createCorrelationMatrix();
    this.updateAnalysisInsights();
  }

  /**
   * 가격 & 볼린거 밴드 차트
   */
  createPriceBollingerChart() {
    const stockData = this.getStockData(this.currentStock);
    const priceHistory = this.generatePriceHistory(stockData.current_price);
    const bollingerBands = this.calculateBollingerBands(priceHistory);

    const data = {
      labels: ['6일전', '5일전', '4일전', '3일전', '2일전', '1일전', '현재'],
      datasets: [
        {
          label: '가격',
          data: priceHistory,
          borderColor: '#007bff',
          backgroundColor: 'rgba(0, 123, 255, 0.1)',
          fill: false,
          tension: 0.4,
        },
        {
          label: '상단 밴드',
          data: bollingerBands.upper,
          borderColor: '#28a745',
          backgroundColor: 'rgba(40, 167, 69, 0.1)',
          fill: false,
          borderDash: [5, 5],
        },
        {
          label: '하단 밴드',
          data: bollingerBands.lower,
          borderColor: '#dc3545',
          backgroundColor: 'rgba(220, 53, 69, 0.1)',
          fill: false,
          borderDash: [5, 5],
        },
      ],
    };

    this.chartManager.createLineChart('price-bollinger-chart', data);
    this.charts.set('price-bollinger', 'price-bollinger-chart');
  }

  /**
   * RSI 차트
   */
  createRSIChart() {
    const rsiData = this.generateRSIData();

    const data = {
      labels: ['6일전', '5일전', '4일전', '3일전', '2일전', '1일전', '현재'],
      datasets: [
        {
          label: 'RSI',
          data: rsiData,
          borderColor: '#ffc107',
          backgroundColor: 'rgba(255, 193, 7, 0.2)',
          fill: true,
          tension: 0.4,
        },
      ],
    };

    const options = {
      scales: {
        y: {
          min: 0,
          max: 100,
          ticks: {
            callback: function (value) {
              return value + '%';
            },
          },
        },
      },
      plugins: {
        annotation: {
          annotations: {
            overbought: {
              type: 'line',
              yMin: 70,
              yMax: 70,
              borderColor: '#dc3545',
              borderWidth: 2,
              borderDash: [10, 5],
              label: {
                content: '과매수 (70)',
                enabled: true,
              },
            },
            oversold: {
              type: 'line',
              yMin: 30,
              yMax: 30,
              borderColor: '#28a745',
              borderWidth: 2,
              borderDash: [10, 5],
              label: {
                content: '과매도 (30)',
                enabled: true,
              },
            },
          },
        },
      },
    };

    this.chartManager.createLineChart('rsi-chart', data, options);
    this.charts.set('rsi', 'rsi-chart');
  }

  /**
   * MACD 차트
   */
  createMACDChart() {
    const macdData = this.generateMACDData();

    const data = {
      labels: ['6일전', '5일전', '4일전', '3일전', '2일전', '1일전', '현재'],
      datasets: [
        {
          label: 'MACD',
          data: macdData.macd,
          borderColor: '#17a2b8',
          backgroundColor: 'rgba(23, 162, 184, 0.2)',
          fill: false,
          tension: 0.4,
        },
        {
          label: 'Signal',
          data: macdData.signal,
          borderColor: '#6c757d',
          backgroundColor: 'rgba(108, 117, 125, 0.2)',
          fill: false,
          tension: 0.4,
        },
      ],
    };

    this.chartManager.createLineChart('macd-chart', data);
    this.charts.set('macd', 'macd-chart');
  }

  /**
   * 거래량 차트
   */
  createVolumeChart() {
    const volumeData = this.generateVolumeData();

    const data = {
      labels: ['월', '화', '수', '목', '금', '토', '일'],
      datasets: [
        {
          label: '거래량',
          data: volumeData.volume,
          backgroundColor: '#fd7e14',
          borderColor: '#fd7e14',
          borderWidth: 1,
        },
        {
          label: '평균 거래량',
          data: volumeData.average,
          type: 'line',
          borderColor: '#e83e8c',
          backgroundColor: 'rgba(232, 62, 140, 0.2)',
          fill: false,
          tension: 0.4,
        },
      ],
    };

    this.chartManager.createBarChart('volume-analysis-chart', data);
    this.charts.set('volume', 'volume-analysis-chart');
  }

  /**
   * 상관관계 매트릭스
   */
  createCorrelationMatrix() {
    const correlationData = this.analysisData.correlationMatrix;

    const data = {
      labels: ['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
      datasets: [
        {
          label: 'AAPL',
          data: correlationData.AAPL,
          backgroundColor: this.getCorrelationColors(correlationData.AAPL),
        },
        {
          label: 'GOOGL',
          data: correlationData.GOOGL,
          backgroundColor: this.getCorrelationColors(correlationData.GOOGL),
        },
        {
          label: 'MSFT',
          data: correlationData.MSFT,
          backgroundColor: this.getCorrelationColors(correlationData.MSFT),
        },
        {
          label: 'AMZN',
          data: correlationData.AMZN,
          backgroundColor: this.getCorrelationColors(correlationData.AMZN),
        },
      ],
    };

    this.chartManager.createBarChart('correlation-matrix-chart', data, {
      scales: {
        y: {
          min: -1,
          max: 1,
          ticks: {
            callback: function (value) {
              return value.toFixed(2);
            },
          },
        },
      },
    });
    this.charts.set('correlation', 'correlation-matrix-chart');
  }

  /**
   * 분석 업데이트
   */
  async updateAnalysis() {
    console.log(`🔄 ${this.currentStock} 분석 업데이트 중...`);

    // 지표 업데이트
    this.updateIndicatorsSummary();

    // 차트 업데이트
    this.updateCharts();

    // 분석 인사이트 업데이트
    this.updateAnalysisInsights();
  }

  /**
   * 지표 요약 업데이트
   */
  updateIndicatorsSummary() {
    const stockData = this.getStockData(this.currentStock);
    if (!stockData) return;

    const rsi = stockData.technical_indicators?.rsi || 50;
    const rsiElement = document.getElementById('current-rsi');
    if (rsiElement) {
      rsiElement.textContent = rsi.toFixed(1);
      rsiElement.parentElement.querySelector('.indicator-status').textContent =
        rsi > 70 ? '과매수' : rsi < 30 ? '과매도' : '중립';
    }

    // 다른 지표들도 유사하게 업데이트
    this.updateMACDIndicator(stockData);
    this.updateBollingerIndicator(stockData);
    this.updateVolumeIndicator(stockData);
  }

  /**
   * 차트들 업데이트
   */
  updateCharts() {
    // 각 차트를 새로운 데이터로 업데이트
    this.createPriceBollingerChart();
    this.createRSIChart();
    this.createMACDChart();
    this.createVolumeChart();
  }

  /**
   * 분석 인사이트 업데이트
   */
  updateAnalysisInsights() {
    const stockData = this.getStockData(this.currentStock);
    const insights = this.generateInsights(stockData);

    const insightsElement = document.getElementById('analysis-insights');
    if (insightsElement) {
      insightsElement.innerHTML = insights
        .map(
          (insight) => `
        <div class="insight-item ${insight.type}">
          <div class="insight-icon">${insight.icon}</div>
          <div class="insight-content">
            <div class="insight-title">${insight.title}</div>
            <div class="insight-description">${insight.description}</div>
          </div>
        </div>
      `
        )
        .join('');
    }
  }

  /**
   * 위젯 새로고침
   */
  async refresh() {
    console.log('🔄 ChartAnalysisWidget 새로고침');
    await this.loadAnalysisData();
    this.updateAnalysis();
  }

  // === 유틸리티 메서드들 ===

  getStockData(symbol) {
    return this.analysisData.stocks.find((stock) => stock.symbol === symbol);
  }

  generatePriceHistory(currentPrice) {
    const history = [];
    let price = currentPrice * 0.96;

    for (let i = 0; i < 7; i++) {
      history.push(Number(price.toFixed(2)));
      price *= 1 + (Math.random() - 0.5) * 0.03;
    }

    history[6] = currentPrice;
    return history;
  }

  calculateBollingerBands(prices) {
    const period = 7;
    const stdDev = this.calculateStandardDeviation(prices);
    const average =
      prices.reduce((sum, price) => sum + price, 0) / prices.length;

    return {
      upper: prices.map(() => average + stdDev * 2),
      lower: prices.map(() => average - stdDev * 2),
    };
  }

  calculateStandardDeviation(values) {
    const avg = values.reduce((sum, value) => sum + value, 0) / values.length;
    const squareDiffs = values.map((value) => Math.pow(value - avg, 2));
    const avgSquareDiff =
      squareDiffs.reduce((sum, value) => sum + value, 0) / values.length;
    return Math.sqrt(avgSquareDiff);
  }

  generateRSIData() {
    return [45, 52, 38, 65, 72, 68, 65.3];
  }

  generateMACDData() {
    return {
      macd: [-0.5, -0.2, 0.1, 0.3, 0.2, 0.4, 0.24],
      signal: [-0.3, -0.1, 0.05, 0.2, 0.15, 0.3, 0.18],
    };
  }

  generateVolumeData() {
    return {
      volume: [120, 150, 80, 200, 175, 190, 185],
      average: [160, 160, 160, 160, 160, 160, 160],
    };
  }

  generateCorrelationMatrix(stocks) {
    // 간단한 mock 상관관계 데이터
    return {
      AAPL: [1.0, 0.75, 0.68, 0.52],
      GOOGL: [0.75, 1.0, 0.71, 0.63],
      MSFT: [0.68, 0.71, 1.0, 0.58],
      AMZN: [0.52, 0.63, 0.58, 1.0],
    };
  }

  getCorrelationColors(values) {
    return values.map((value) => {
      if (value > 0.7) return 'rgba(40, 167, 69, 0.8)';
      if (value > 0.3) return 'rgba(255, 193, 7, 0.8)';
      if (value > -0.3) return 'rgba(108, 117, 125, 0.8)';
      return 'rgba(220, 53, 69, 0.8)';
    });
  }

  generateInsights(stockData) {
    const insights = [];
    const rsi = stockData?.technical_indicators?.rsi || 50;
    const priceChange = stockData?.technical_indicators?.price_change || 0;

    if (rsi > 70) {
      insights.push({
        type: 'warning',
        icon: '⚠️',
        title: 'RSI 과매수 구간',
        description: `현재 RSI가 ${rsi.toFixed(1)}로 과매수 구간에 있습니다. 조정 가능성을 고려해보세요.`,
      });
    }

    if (priceChange > 0.02) {
      insights.push({
        type: 'positive',
        icon: '📈',
        title: '강한 상승 모멘텀',
        description: `최근 ${(priceChange * 100).toFixed(1)}% 상승하며 강한 상승세를 보이고 있습니다.`,
      });
    }

    insights.push({
      type: 'info',
      icon: '💡',
      title: '기술적 분석 요약',
      description: `현재 ${stockData?.symbol || this.currentStock}은 ${stockData?.predicted_direction === 'up' ? '상승' : '하락'} 추세로 예측되며, 신뢰도는 ${((stockData?.confidence || 0.5) * 100).toFixed(0)}%입니다.`,
    });

    return insights;
  }

  updateMACDIndicator(stockData) {
    // MACD 지표 업데이트 로직
  }

  updateBollingerIndicator(stockData) {
    // 볼린저 밴드 지표 업데이트 로직
  }

  updateVolumeIndicator(stockData) {
    // 거래량 지표 업데이트 로직
  }

  generateTechnicalIndicators(stocks) {
    // 기술적 지표 생성 로직
    return {};
  }

  getMockVolumeData() {
    return {
      labels: ['월', '화', '수', '목', '금'],
      data: [120, 150, 80, 200, 175],
    };
  }

  getMockTrendData() {
    return {
      labels: ['1월', '2월', '3월', '4월', '5월', '6월'],
      accuracy: [0.82, 0.85, 0.83, 0.87, 0.84, 0.86],
      loss: [0.45, 0.38, 0.42, 0.33, 0.39, 0.35],
    };
  }

  getMockAnalysisData() {
    return {
      stocks: [
        {
          symbol: 'AAPL',
          current_price: 230.48,
          technical_indicators: { rsi: 65.3, price_change: 0.024 },
        },
      ],
      technicalIndicators: {},
      volumeAnalysis: this.getMockVolumeData(),
      trendAnalysis: this.getMockTrendData(),
      correlationMatrix: {
        AAPL: [1.0, 0.75, 0.68, 0.52],
        GOOGL: [0.75, 1.0, 0.71, 0.63],
        MSFT: [0.68, 0.71, 1.0, 0.58],
        AMZN: [0.52, 0.63, 0.58, 1.0],
      },
    };
  }

  showError(message) {
    if (this.element) {
      this.element.innerHTML = `
        <div class="chart-error">
          <div class="error-icon">⚠️</div>
          <div class="error-message">${message}</div>
          <button class="btn btn-primary" onclick="window.app?.refresh()">다시 시도</button>
        </div>
      `;
    }
  }

  destroy() {
    // 생성된 차트들 정리
    this.charts.forEach((chartId, key) => {
      this.chartManager.destroyChart(chartId);
    });
    this.charts.clear();
  }
}

// 전역 변수로 내보내기
window.ChartAnalysisWidget = ChartAnalysisWidget;
