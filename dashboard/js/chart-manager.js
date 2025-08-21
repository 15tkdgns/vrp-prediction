/**
 * ChartManager - Chart.js 래퍼 클래스
 *
 * 특징:
 * 1. 안전한 차트 생성 및 관리
 * 2. 메모리 누수 방지 (적절한 destroy)
 * 3. 에러 처리 및 폴백
 * 4. 일관된 스타일링
 */

class ChartManager {
  constructor() {
    this.charts = new Map();
    this.defaultColors = {
      primary: '#007bff',
      success: '#28a745',
      warning: '#ffc107',
      danger: '#dc3545',
      info: '#17a2b8',
      purple: '#6f42c1',
      orange: '#fd7e14',
      pink: '#e83e8c',
    };

    console.log('📊 ChartManager 생성됨');
  }

  /**
   * 초기화
   */
  async init() {
    try {
      if (typeof Chart === 'undefined') {
        throw new Error('Chart.js가 로드되지 않았습니다');
      }

      // Chart.js 기본 설정
      Chart.defaults.responsive = true;
      Chart.defaults.maintainAspectRatio = false;
      Chart.defaults.plugins.legend.display = true;

      console.log('✅ ChartManager 초기화됨 (Chart.js v' + Chart.version + ')');
    } catch (error) {
      console.error('❌ ChartManager 초기화 실패:', error);
      throw error;
    }
  }

  /**
   * 초기 차트들 로드 (최소한의 필수 차트만)
   */
  async loadInitialCharts() {
    try {
      // 가장 중요한 차트들만 미리 준비
      console.log('📊 필수 차트 구성 요소 준비 중...');
      
      // Chart.js 전역 설정 최적화
      Chart.defaults.animation = {
        duration: 300 // 애니메이션 단축
      };
      
      // 기본 차트 색상 팔레트 준비
      this.prepareChartColors();
      
      console.log('✅ 필수 차트 로드 완료');
    } catch (error) {
      console.warn('⚠️ 차트 초기화 일부 실패:', error);
    }
  }

  /**
   * 차트 색상 팔레트 준비
   */
  prepareChartColors() {
    this.colorPalette = [
      this.colors.primary,
      this.colors.success,
      this.colors.warning,
      this.colors.danger,
      this.colors.info,
      this.colors.purple,
      this.colors.orange,
      this.colors.pink
    ];
  }

  /**
   * 라인 차트 생성
   */
  createLineChart(canvasId, data, options = {}) {
    return this.createChart(canvasId, 'line', data, {
      tension: 0.4,
      fill: false,
      borderWidth: 2,
      pointRadius: 3,
      pointHoverRadius: 5,
      ...options,
    });
  }

  /**
   * 바 차트 생성
   */
  createBarChart(canvasId, data, options = {}) {
    return this.createChart(canvasId, 'bar', data, {
      borderWidth: 1,
      borderRadius: 4,
      ...options,
    });
  }

  /**
   * 도넛 차트 생성
   */
  createDoughnutChart(canvasId, data, options = {}) {
    return this.createChart(canvasId, 'doughnut', data, {
      cutout: '60%',
      ...options,
    });
  }

  /**
   * 레이더 차트 생성
   */
  createRadarChart(canvasId, data, options = {}) {
    return this.createChart(canvasId, 'radar', data, {
      pointRadius: 3,
      pointHoverRadius: 5,
      ...options,
    });
  }

  /**
   * 기본 차트 생성 메서드
   */
  createChart(canvasId, type, data, customOptions = {}) {
    try {
      // 기존 차트 정리
      this.destroyChart(canvasId);

      // Canvas 요소 확인
      const canvas = document.getElementById(canvasId);
      if (!canvas) {
        throw new Error(`Canvas 요소를 찾을 수 없습니다: ${canvasId}`);
      }

      const ctx = canvas.getContext('2d');
      if (!ctx) {
        throw new Error(`2D 컨텍스트를 가져올 수 없습니다: ${canvasId}`);
      }

      // 데이터 전처리
      const processedData = this.processChartData(data, type);

      // 차트 옵션 구성
      const options = this.buildChartOptions(type, customOptions);

      // 차트 생성
      const chart = new Chart(ctx, {
        type: type,
        data: processedData,
        options: options,
      });

      // 차트 저장
      this.charts.set(canvasId, chart);

      console.log(`✅ ${type} 차트 생성됨: ${canvasId}`);
      return chart;
    } catch (error) {
      console.error(`❌ 차트 생성 실패 (${canvasId}):`, error);
      this.showChartError(canvasId, `차트 생성 실패: ${error.message}`);
      return null;
    }
  }

  /**
   * 차트 데이터 전처리
   */
  processChartData(data, type) {
    if (!data || !data.labels || !data.datasets) {
      throw new Error('유효하지 않은 차트 데이터');
    }

    // 데이터셋에 색상 적용
    const processedDatasets = data.datasets.map((dataset, index) => {
      const colorKey = Object.keys(this.defaultColors)[
        index % Object.keys(this.defaultColors).length
      ];
      const color = this.defaultColors[colorKey];

      return {
        ...dataset,
        borderColor: dataset.borderColor || color,
        backgroundColor:
          dataset.backgroundColor || this.getBackgroundColor(color, type),
      };
    });

    return {
      ...data,
      datasets: processedDatasets,
    };
  }

  /**
   * 배경색 생성 (차트 타입에 따라)
   */
  getBackgroundColor(baseColor, chartType) {
    if (chartType === 'line') {
      return baseColor.replace(')', ', 0.1)').replace('rgb', 'rgba');
    } else if (chartType === 'doughnut' || chartType === 'pie') {
      return baseColor;
    } else {
      return baseColor.replace(')', ', 0.8)').replace('rgb', 'rgba');
    }
  }

  /**
   * 차트 옵션 구성
   */
  buildChartOptions(type, customOptions) {
    const baseOptions = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: true,
          position: 'top',
        },
        tooltip: {
          enabled: true,
          mode: 'index',
          intersect: false,
        },
      },
    };

    // 타입별 기본 옵션
    if (type === 'line' || type === 'bar') {
      baseOptions.scales = {
        x: {
          display: true,
          grid: { color: 'rgba(0,0,0,0.1)' },
        },
        y: {
          display: true,
          beginAtZero: true,
          grid: { color: 'rgba(0,0,0,0.1)' },
        },
      };
    }

    // 커스텀 옵션 병합
    return this.deepMerge(baseOptions, customOptions);
  }

  /**
   * 객체 깊은 병합
   */
  deepMerge(target, source) {
    const result = { ...target };

    for (const key in source) {
      if (
        source[key] &&
        typeof source[key] === 'object' &&
        !Array.isArray(source[key])
      ) {
        result[key] = this.deepMerge(target[key] || {}, source[key]);
      } else {
        result[key] = source[key];
      }
    }

    return result;
  }

  /**
   * 차트 업데이트
   */
  updateChart(canvasId, newData, newOptions = {}) {
    try {
      const chart = this.charts.get(canvasId);
      if (!chart) {
        console.warn(`차트를 찾을 수 없습니다: ${canvasId}`);
        return false;
      }

      // 데이터 업데이트
      if (newData) {
        chart.data = this.processChartData(newData, chart.config.type);
      }

      // 옵션 업데이트
      if (newOptions && Object.keys(newOptions).length > 0) {
        chart.options = this.deepMerge(chart.options, newOptions);
      }

      // 차트 다시 그리기
      chart.update();

      console.log(`🔄 차트 업데이트됨: ${canvasId}`);
      return true;
    } catch (error) {
      console.error(`❌ 차트 업데이트 실패 (${canvasId}):`, error);
      return false;
    }
  }

  /**
   * 차트 제거
   */
  destroyChart(canvasId) {
    const chart = this.charts.get(canvasId);
    if (chart) {
      try {
        chart.destroy();
        this.charts.delete(canvasId);
        console.log(`🗑️ 차트 제거됨: ${canvasId}`);
      } catch (error) {
        console.warn(`차트 제거 중 오류 (${canvasId}):`, error);
      }
    }
  }

  /**
   * 모든 차트 제거
   */
  destroyAllCharts() {
    this.charts.forEach((chart, canvasId) => {
      this.destroyChart(canvasId);
    });
    console.log('🧹 모든 차트 정리됨');
  }

  /**
   * 차트 에러 표시
   */
  showChartError(canvasId, message) {
    const canvas = document.getElementById(canvasId);
    if (canvas) {
      const parent = canvas.parentElement;
      if (parent) {
        parent.innerHTML = `
          <div class="chart-error">
            <div class="error-icon">⚠️</div>
            <div class="error-message">${message}</div>
            <button class="retry-btn" onclick="window.app?.refresh()">다시 시도</button>
          </div>
        `;
      }
    }
  }

  /**
   * 차트 존재 확인
   */
  hasChart(canvasId) {
    return this.charts.has(canvasId);
  }

  /**
   * 차트 가져오기
   */
  getChart(canvasId) {
    return this.charts.get(canvasId);
  }

  /**
   * 미리 정의된 차트 템플릿들
   */
  createStockPriceChart(canvasId, stockData) {
    const data = {
      labels: ['6일전', '5일전', '4일전', '3일전', '2일전', '1일전', '현재'],
      datasets: [
        {
          label: stockData.symbol + ' 주가',
          data: this.generatePriceHistory(stockData.current_price),
          borderColor: this.defaultColors.primary,
          backgroundColor: 'transparent',
          fill: false,
          tension: 0.4,
          borderWidth: 2,
          pointRadius: 4,
          pointHoverRadius: 6,
        },
      ],
    };

    return this.createLineChart(canvasId, data);
  }

  createPerformanceChart(canvasId, metricsData) {
    // 모델별 성능 데이터 (실제 데이터가 없을 때의 목업)
    const modelPerformance = {
      accuracy: { rf: 0.87, lstm: 0.85, xgb: 0.89, gb: 0.84 },
      precision: { rf: 0.84, lstm: 0.82, xgb: 0.86, gb: 0.81 },
      recall: { rf: 0.89, lstm: 0.87, xgb: 0.91, gb: 0.83 },
      f1_score: { rf: 0.86, lstm: 0.84, xgb: 0.88, gb: 0.82 },
    };

    const data = {
      labels: ['Random Forest', 'LSTM', 'XGBoost', 'Gradient Boosting'],
      datasets: [
        {
          label: '정확도',
          data: [
            modelPerformance.accuracy.rf,
            modelPerformance.accuracy.lstm,
            modelPerformance.accuracy.xgb,
            modelPerformance.accuracy.gb,
          ],
          backgroundColor: this.defaultColors.primary,
          borderColor: this.defaultColors.primary,
          borderWidth: 1,
        },
        {
          label: '정밀도',
          data: [
            modelPerformance.precision.rf,
            modelPerformance.precision.lstm,
            modelPerformance.precision.xgb,
            modelPerformance.precision.gb,
          ],
          backgroundColor: this.defaultColors.success,
          borderColor: this.defaultColors.success,
          borderWidth: 1,
        },
        {
          label: '재현율',
          data: [
            modelPerformance.recall.rf,
            modelPerformance.recall.lstm,
            modelPerformance.recall.xgb,
            modelPerformance.recall.gb,
          ],
          backgroundColor: this.defaultColors.warning,
          borderColor: this.defaultColors.warning,
          borderWidth: 1,
        },
        {
          label: 'F1 점수',
          data: [
            modelPerformance.f1_score.rf,
            modelPerformance.f1_score.lstm,
            modelPerformance.f1_score.xgb,
            modelPerformance.f1_score.gb,
          ],
          backgroundColor: this.defaultColors.info,
          borderColor: this.defaultColors.info,
          borderWidth: 1,
        },
      ],
    };

    return this.createBarChart(canvasId, data, {
      scales: {
        y: {
          max: 1.0,
          beginAtZero: true,
          ticks: {
            callback: function (value) {
              return (value * 100).toFixed(0) + '%';
            },
          },
        },
      },
      plugins: {
        legend: {
          display: true,
          position: 'top',
        },
      },
    });
  }

  /**
   * 주가 히스토리 생성 (7일 단위)
   */
  generatePriceHistory(currentPrice) {
    const history = [];
    let price = currentPrice * 0.96; // 시작 가격 (일주일 전)

    for (let i = 0; i < 7; i++) {
      history.push(Number(price.toFixed(2)));
      price *= 1 + (Math.random() - 0.5) * 0.03; // ±1.5% 변동 (하루 변동량)
    }

    history[6] = currentPrice; // 마지막은 현재가격
    return history;
  }

  /**
   * 디버그 정보
   */
  getDebugInfo() {
    return {
      chartCount: this.charts.size,
      chartIds: Array.from(this.charts.keys()),
      colors: this.defaultColors,
    };
  }
}

// 전역 변수로 내보내기
window.ChartManager = ChartManager;
