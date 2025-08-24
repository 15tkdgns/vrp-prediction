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
        duration: 300, // 애니메이션 단축
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
      this.colors.pink,
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
  async createStockPriceChart(canvasId, stockData) {
    // 실제 날짜 기반: 7월 22일부터 8월 21일까지 (한달간)
    const startDate = new Date('2025-07-22'); // 7월 22일부터 시작
    const today = new Date(); // 8월 21일

    // 실제 히스토리 데이터와 라벨 가져오기
    const historyResult = await this.generateRealHistoryData(
      stockData,
      startDate,
      today
    );

    // API에서 실제 라벨과 데이터를 받아온 경우
    let labels, actualPriceHistory;
    if (historyResult && historyResult.labels) {
      labels = historyResult.labels;
      actualPriceHistory = historyResult.prices;
    } else {
      // 폴백: 시뮬레이션 데이터와 영업일 라벨
      labels = this.generateBusinessDayLabels(startDate, today);
      actualPriceHistory =
        historyResult ||
        this.generateSimulatedHistoryData(stockData, startDate, today);
    }

    // 상승/하락에 따른 색상 결정
    const priceChange = stockData.technical_indicators?.price_change || 0;
    const trendColor = priceChange >= 0 ? '#28a745' : '#dc3545';

    // 6월 분석 기반 7월 예측 데이터 생성 (7월 22일~8월 21일)
    const predictedPriceHistory = this.generatePredictedHistoryData(
      stockData,
      startDate,
      today
    );

    // 예측 방향에 따른 예측 색상
    const predictedDirection = stockData.predicted_direction || 'neutral';
    const predictionColor =
      predictedDirection === 'up'
        ? '#007bff'
        : predictedDirection === 'down'
          ? '#fd7e14'
          : '#6c757d';

    // 변화량 비율 계산
    const changePercent = Math.abs(priceChange);

    // 변화량에 따른 동적 패딩 계산
    let paddingRatio;
    if (changePercent > 0.05) {
      paddingRatio = 0.15;
    } else if (changePercent > 0.02) {
      paddingRatio = 0.12;
    } else {
      paddingRatio = 0.08;
    }

    // Y축 범위 동적 계산 (실제 + 예측 데이터 모두 포함)
    const allData = [...actualPriceHistory, ...predictedPriceHistory];
    const minPrice = Math.min(...allData);
    const maxPrice = Math.max(...allData);
    const priceRange = maxPrice - minPrice;
    const currentPrice = stockData.current_price;

    const padding = Math.max(priceRange * paddingRatio, currentPrice * 0.01);
    const yMin = Math.max(minPrice * 0.95, minPrice - padding);
    const yMax = maxPrice + padding;

    const bgColor =
      priceChange >= 0 ? 'rgba(40, 167, 69, 0.1)' : 'rgba(220, 53, 69, 0.1)';

    const data = {
      labels: labels,
      datasets: [
        {
          label: `${stockData.symbol} 실제 (7/22-8/21)`,
          data: actualPriceHistory,
          borderColor: trendColor,
          backgroundColor: bgColor,
          fill: false,
          tension: 0.4,
          borderWidth: 3,
          pointRadius: 0,
          pointHoverRadius: 4,
          pointHoverBackgroundColor: trendColor,
          pointHoverBorderColor: '#fff',
          pointHoverBorderWidth: 2,
          borderDash: [], // 실선
        },
        {
          label: `${stockData.symbol} 예측 (6월분석→7월)`,
          data: predictedPriceHistory,
          borderColor: predictionColor,
          backgroundColor: 'transparent',
          fill: false,
          tension: 0.4,
          borderWidth: 2.5,
          pointRadius: 0,
          pointHoverRadius: 4,
          pointHoverBackgroundColor: predictionColor,
          pointHoverBorderColor: '#fff',
          pointHoverBorderWidth: 2,
          borderDash: [8, 4], // 점선
        },
      ],
    };

    const customOptions = {
      responsive: true,
      maintainAspectRatio: false,
      animation: {
        duration: 1500,
        easing: 'easeInOutCubic',
      },
      interaction: {
        intersect: false,
        mode: 'index',
      },
      plugins: {
        legend: {
          display: true,
          position: 'top',
          labels: {
            boxWidth: 12,
            usePointStyle: true,
            font: { size: 10, weight: 'bold' },
            generateLabels: function (chart) {
              const original =
                Chart.defaults.plugins.legend.labels.generateLabels;
              const labels = original.call(this, chart);

              // 실제 데이터 라벨 스타일링
              if (labels[0]) {
                labels[0].lineDash = [];
                labels[0].pointStyle = 'line';
              }

              // 예측 데이터 라벨 스타일링
              if (labels[1]) {
                labels[1].lineDash = [8, 4];
                labels[1].pointStyle = 'line';
              }

              return labels;
            },
          },
        },
        tooltip: {
          backgroundColor: 'rgba(0, 0, 0, 0.8)',
          titleColor: '#ffffff',
          bodyColor: '#ffffff',
          borderColor: trendColor,
          borderWidth: 1,
          displayColors: false,
          callbacks: {
            label: function (context) {
              return `$${context.parsed.y.toLocaleString()}`;
            },
          },
        },
      },
      scales: {
        x: {
          display: true,
          grid: {
            color: 'rgba(0, 0, 0, 0.05)',
            borderDash: [2, 2],
          },
          ticks: {
            color: '#6c757d',
            maxTicksLimit: 6,
            font: { size: 10 },
          },
        },
        y: {
          display: true,
          position: 'right',
          grid: {
            color: 'rgba(0, 0, 0, 0.05)',
            borderDash: [2, 2],
          },
          ticks: {
            color: '#6c757d',
            font: { size: 10 },
            callback: function (value) {
              return '$' + value.toLocaleString();
            },
          },
          // 계산된 동적 Y축 범위 적용
          min: yMin,
          max: yMax,
        },
      },
    };

    return this.createChart(canvasId, 'line', data, customOptions);
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
  /**
   * 실제 히스토리 데이터 생성 (7월 22일 ~ 8월 21일)
   * 가능하면 API에서 실제 데이터를 가져오고, 실패 시 시뮬레이션 사용
   */
  async generateRealHistoryData(stockData, startDate, endDate) {
    try {
      // 실제 yfinance 히스토리 데이터 시도
      const startStr = startDate.toISOString().split('T')[0];
      const endStr = endDate.toISOString().split('T')[0];

      const response = await fetch(
        `http://localhost:8092/api/stocks/history/${stockData.symbol}?start=${startStr}&end=${endStr}`
      );

      if (response.ok) {
        const data = await response.json();
        console.log(
          `✅ ${stockData.symbol} 실제 히스토리 데이터 로드: ${data.prices.length}개, 라벨: ${data.labels.length}개`
        );
        return {
          prices: data.prices,
          labels: data.labels,
        };
      }
    } catch (error) {
      console.warn(
        `⚠️ ${stockData.symbol} API 히스토리 실패, 시뮬레이션 사용:`,
        error
      );
    }

    // 폴백: 시뮬레이션 데이터만 반환 (라벨 없음)
    return this.generateSimulatedHistoryData(stockData, startDate, endDate);
  }

  /**
   * 영업일 기준 라벨 생성
   */
  generateBusinessDayLabels(startDate, endDate) {
    const labels = [];
    const current = new Date(startDate);

    while (current <= endDate) {
      // 주말 제외 (0=일요일, 6=토요일)
      const dayOfWeek = current.getDay();
      if (dayOfWeek !== 0 && dayOfWeek !== 6) {
        const dateStr = current.toLocaleDateString('ko-KR', {
          month: 'short',
          day: 'numeric',
        });
        labels.push(dateStr);
      }
      current.setDate(current.getDate() + 1);
    }

    return labels;
  }

  /**
   * 시뮬레이션 히스토리 데이터 생성
   */
  generateSimulatedHistoryData(stockData, startDate, endDate) {
    const history = [];
    const currentPrice = stockData.current_price;
    const daysDiff =
      Math.ceil((endDate - startDate) / (1000 * 60 * 60 * 24)) + 1;

    // 7월 22일 시작가를 현재가 기준으로 역산
    const startPrice = currentPrice * (0.95 + Math.random() * 0.1); // ±5% 범위

    // 실제 주식의 변동성과 추세 반영
    const volatility = stockData.technical_indicators?.volatility || 0.02;
    const momentum = stockData.technical_indicators?.momentum || 0;

    let price = startPrice;
    for (let i = 0; i < daysDiff; i++) {
      if (i === 0) {
        history.push(Number(price.toFixed(2)));
        continue;
      }

      // 실제 주식 변동 패턴 시뮬레이션
      const trendEffect = momentum * 0.3; // 모멘텀 영향
      const randomChange = (Math.random() - 0.5) * volatility; // 변동성 영향
      const weekendEffect = Math.sin((i / 7) * Math.PI) * 0.005; // 주간 패턴

      const dailyChange = trendEffect + randomChange + weekendEffect;
      price *= 1 + dailyChange;

      history.push(Number(price.toFixed(2)));
    }

    // 마지막 값을 현재가로 조정
    const adjustment = currentPrice / history[history.length - 1];
    return history.map((p) => Number((p * adjustment).toFixed(2)));
  }

  /**
   * 6월 분석 기반 7월 예측 데이터 생성
   */
  generatePredictedHistoryData(stockData, startDate, endDate) {
    const history = [];
    const daysDiff =
      Math.ceil((endDate - startDate) / (1000 * 60 * 60 * 24)) + 1;

    // 6월 분석 결과 기반 예측 파라미터
    const confidence = stockData.confidence || 0.5;
    const predictedDirection = stockData.predicted_direction || 'neutral';
    const rsi = stockData.technical_indicators?.rsi || 50;
    const currentPrice = stockData.current_price;

    // 7월 초 예측 시작가 (6월 말 기준)
    let predictedPrice = currentPrice * (0.98 + Math.random() * 0.04);

    // 예측 모델의 추세 설정
    let trendStrength = 0;
    if (predictedDirection === 'up') {
      trendStrength = confidence * 0.015; // 상승 추세
    } else if (predictedDirection === 'down') {
      trendStrength = -confidence * 0.015; // 하락 추세
    }

    for (let i = 0; i < daysDiff; i++) {
      if (i === 0) {
        history.push(Number(predictedPrice.toFixed(2)));
        continue;
      }

      // AI 예측 모델의 특성 반영
      const trendEffect = trendStrength; // 예측 방향
      const confidenceEffect = (confidence - 0.5) * 0.01; // 신뢰도 영향
      const rsiEffect = (50 - rsi) / 2000; // RSI 조정
      const modelNoise = (Math.random() - 0.5) * 0.008; // 모델 불확실성

      const dailyChange =
        trendEffect + confidenceEffect + rsiEffect + modelNoise;
      predictedPrice *= 1 + dailyChange;

      history.push(Number(predictedPrice.toFixed(2)));
    }

    return history;
  }

  generatePriceHistory(currentPrice) {
    // 레거시 함수 - 호환성 유지용
    const history = [];
    const basePrice = currentPrice * 0.95;

    for (let i = 0; i < 30; i++) {
      let dailyChange;
      if (i < 15) {
        dailyChange = Math.random() * 0.025 - 0.005;
      } else {
        dailyChange = (Math.random() - 0.5) * 0.04;
      }

      const price = i === 0 ? basePrice : history[i - 1] * (1 + dailyChange);
      history.push(Number(price.toFixed(2)));
    }

    const adjustment = currentPrice / history[29];
    for (let i = 0; i < 30; i++) {
      history[i] = Number((history[i] * adjustment).toFixed(2));
    }

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
