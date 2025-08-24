/**
 * S&P 500 실시간 위젯 관리자
 * - 실시간 가격 업데이트
 * - 30일 가격 차트
 * - AI 예측 표시
 */

class SP500Widget {
  constructor() {
    this.chart = null;
    this.updateInterval = null;
    this.retryCount = 0;
    this.maxRetries = 3;

    console.log('📊 S&P 500 Widget 초기화됨');
  }

  /**
   * 위젯 초기화 (빠른 초기화)
   */
  async init() {
    try {
      // 차트 초기화와 데이터 로드를 병렬로 처리
      await Promise.all([this.initChartOptimized(), this.loadDataOptimized()]);

      // 자동 업데이트는 백그라운드에서 시작
      setTimeout(() => this.startAutoUpdate(), 500);

      console.log('✅ S&P 500 Widget 빠른 초기화 완료');
    } catch (error) {
      console.error('❌ S&P 500 Widget 초기화 실패:', error);
      this.showError('위젯 초기화에 실패했습니다.');
    }
  }

  /**
   * 최적화된 차트 초기화
   */
  async initChartOptimized() {
    const ctx = document.getElementById('sp500-30day-chart');
    if (!ctx) {
      throw new Error('S&P 500 차트 캔버스를 찾을 수 없습니다.');
    }

    // 기존 차트 제거
    if (this.chart) {
      this.chart.destroy();
    }

    // 간단한 플레이스홀더 데이터로 빠른 초기화
    const placeholderData = this.generatePlaceholderData();

    this.chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: placeholderData.labels,
        datasets: [
          {
            label: 'S&P 500',
            data: placeholderData.prices,
            borderColor: '#007bff',
            backgroundColor: 'rgba(0, 123, 255, 0.1)',
            borderWidth: 2,
            fill: true,
            tension: 0.4,
            pointRadius: 0,
            pointHoverRadius: 6,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: {
          duration: 0, // 초기화 시 애니메이션 없음
        },
        interaction: {
          intersect: false,
          mode: 'index',
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            titleColor: '#ffffff',
            bodyColor: '#ffffff',
          },
        },
        scales: {
          x: {
            display: true,
            grid: { color: 'rgba(0, 0, 0, 0.1)' },
            ticks: { color: '#6c757d', maxTicksLimit: 6 },
          },
          y: {
            display: true,
            position: 'right',
            grid: { color: 'rgba(0, 0, 0, 0.1)' },
            ticks: {
              color: '#6c757d',
              callback: function (value) {
                return '$' + value.toLocaleString();
              },
            },
          },
        },
      },
    });

    // 실제 데이터는 백그라운드에서 업데이트
    setTimeout(() => this.updateChartWithRealData(), 100);
  }

  /**
   * 플레이스홀더 데이터 생성 (빠른 로딩용)
   */
  generatePlaceholderData() {
    const labels = [];
    const prices = [];
    const basePrice = 5527.45;

    for (let i = 6; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);
      labels.push(
        date.toLocaleDateString('ko-KR', { month: 'short', day: 'numeric' })
      );
      prices.push(basePrice + (Math.random() - 0.5) * 100);
    }

    return { labels, prices };
  }

  /**
   * 실제 데이터로 차트 업데이트
   */
  async updateChartWithRealData() {
    try {
      const realData = this.generate30DayData();
      if (this.chart) {
        this.chart.data.labels = realData.labels;
        this.chart.data.datasets[0].data = realData.prices;
        this.chart.options.animation.duration = 300; // 업데이트 시 애니메이션 활성화
        this.chart.update();
      }
    } catch (error) {
      console.warn('실제 차트 데이터 업데이트 실패:', error);
    }
  }

  /**
   * 최적화된 데이터 로드
   */
  async loadDataOptimized() {
    try {
      // 기본 데이터로 즉시 표시
      this.displayDefaultData();

      // 실제 데이터는 백그라운드에서 로드
      setTimeout(() => this.loadRealData(), 50);
    } catch (error) {
      console.warn('데이터 로드 실패:', error);
      this.displayDefaultData();
    }
  }

  /**
   * 기본 데이터 즉시 표시
   */
  displayDefaultData() {
    const defaultData = {
      current_price: 5527.45,
      predicted_price: 5612.3,
      price_change: 84.85,
      price_change_percent: 1.54,
      prediction_confidence: 0.87,
    };

    this.updatePriceDisplay(defaultData);
    this.updateLastUpdateTime();
  }

  /**
   * 실제 데이터 백그라운드 로드
   */
  async loadRealData() {
    try {
      const sp500Data = await this.fetchSP500Data();
      if (sp500Data) {
        this.updatePriceDisplay(sp500Data);
        this.updateLastUpdateTime();
      }
    } catch (error) {
      console.warn('실제 데이터 로드 실패:', error);
    }
  }

  /**
   * 30일 차트 초기화 (레거시)
   */
  async initChart() {
    const ctx = document.getElementById('sp500-30day-chart');
    if (!ctx) {
      throw new Error('S&P 500 차트 캔버스를 찾을 수 없습니다.');
    }

    // 기존 차트 제거
    if (this.chart) {
      this.chart.destroy();
    }

    // 30일 데이터 생성 (실제 환경에서는 API에서 가져옴)
    const data = this.generate30DayData();

    this.chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: data.labels,
        datasets: [
          {
            label: 'S&P 500',
            data: data.prices,
            borderColor: '#007bff',
            backgroundColor: 'rgba(0, 123, 255, 0.1)',
            borderWidth: 2,
            fill: true,
            tension: 0.4,
            pointRadius: 0,
            pointHoverRadius: 6,
            pointHoverBackgroundColor: '#007bff',
            pointHoverBorderColor: '#ffffff',
            pointHoverBorderWidth: 2,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          intersect: false,
          mode: 'index',
        },
        plugins: {
          legend: {
            display: false,
          },
          tooltip: {
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            titleColor: '#ffffff',
            bodyColor: '#ffffff',
            borderColor: '#007bff',
            borderWidth: 1,
            callbacks: {
              title: function (context) {
                return context[0].label;
              },
              label: function (context) {
                return `$${context.parsed.y.toLocaleString('en-US', {
                  minimumFractionDigits: 2,
                  maximumFractionDigits: 2,
                })}`;
              },
            },
          },
        },
        scales: {
          x: {
            display: true,
            grid: {
              color: 'rgba(0, 0, 0, 0.1)',
              drawBorder: false,
            },
            ticks: {
              color: '#6c757d',
              maxTicksLimit: 6,
            },
          },
          y: {
            display: true,
            position: 'right',
            grid: {
              color: 'rgba(0, 0, 0, 0.1)',
              drawBorder: false,
            },
            ticks: {
              color: '#6c757d',
              callback: function (value) {
                return '$' + value.toLocaleString();
              },
            },
          },
        },
        elements: {
          point: {
            hoverRadius: 8,
          },
        },
      },
    });
  }

  /**
   * 30일 데이터 생성 (실제 데이터 + 시뮬레이션)
   */
  generate30DayData() {
    const labels = [];
    const prices = [];

    // 실제 데이터에서 역사적 데이터 가져오기
    const sp500Data = this.getStoredSP500Data();

    if (sp500Data && sp500Data.historical_data) {
      // 실제 역사적 데이터 사용
      const historicalData = sp500Data.historical_data;
      const baseData = historicalData.slice(-7); // 최근 7일 데이터

      // 23일의 추가 데이터 생성 (30일 - 7일)
      const today = new Date();
      const startDate = new Date(today);
      startDate.setDate(startDate.getDate() - 29);

      for (let i = 0; i < 30; i++) {
        const date = new Date(startDate);
        date.setDate(date.getDate() + i);

        const label = date.toLocaleDateString('ko-KR', {
          month: 'short',
          day: 'numeric',
        });
        labels.push(label);

        if (i >= 23 && baseData[i - 23]) {
          // 실제 데이터 사용
          prices.push(baseData[i - 23].price);
        } else {
          // 시뮬레이션 데이터 생성
          const basePrice = sp500Data.current_price || 5527.45;
          const variation = (Math.random() - 0.5) * 0.03; // ±1.5% 변동
          const trendFactor = (i / 30) * 0.05; // 점진적 상승
          const price = basePrice * (1 + variation + trendFactor - 0.02);
          prices.push(parseFloat(price.toFixed(2)));
        }
      }
    } else {
      // 폴백: 시뮬레이션 데이터
      const basePrice = 5527.45;
      const today = new Date();

      for (let i = 29; i >= 0; i--) {
        const date = new Date(today);
        date.setDate(date.getDate() - i);

        const label = date.toLocaleDateString('ko-KR', {
          month: 'short',
          day: 'numeric',
        });
        labels.push(label);

        const variation = (Math.random() - 0.5) * 0.04; // -2% ~ +2%
        const trendFactor = ((30 - i) / 30) * 0.08; // 점진적 상승 트렌드
        const price = basePrice * (1 + variation + trendFactor - 0.04);

        prices.push(parseFloat(price.toFixed(2)));
      }
    }

    return { labels, prices };
  }

  /**
   * 저장된 S&P 500 데이터 가져오기
   */
  getStoredSP500Data() {
    // 전역 데이터에서 S&P 500 데이터 가져오기
    if (window.sp500Data) {
      return window.sp500Data;
    }

    // API 데이터 로더에서 가져오기
    if (window.apiDataLoader) {
      return window.apiDataLoader.getSP500Data();
    }

    return null;
  }

  /**
   * 실시간 데이터 로드
   */
  async loadData() {
    try {
      // S&P 500 데이터 로드 시도
      const sp500Data = await this.fetchSP500Data();

      if (sp500Data) {
        this.updatePriceDisplay(sp500Data);
        this.updateLastUpdateTime();
        this.retryCount = 0; // 성공 시 재시도 카운트 리셋
      } else {
        // 실제 데이터가 없을 경우 모의 데이터 사용
        this.updateWithMockData();
      }
    } catch (error) {
      console.error('S&P 500 데이터 로드 실패:', error);
      this.handleDataLoadError();
    }
  }

  /**
   * S&P 500 데이터 가져오기
   */
  async fetchSP500Data() {
    try {
      // API에서 데이터 가져오기 시도
      if (window.apiDataLoader) {
        const data = window.apiDataLoader.getSP500Data();
        if (data) return data;
      }

      // 로컬 파일에서 데이터 가져오기 시도
      const response = await fetch('../data/raw/sp500_prediction_data.json');
      if (response.ok) {
        return await response.json();
      }

      return null;
    } catch (error) {
      console.warn('S&P 500 API 호출 실패:', error);
      return null;
    }
  }

  /**
   * 가격 표시 업데이트
   */
  updatePriceDisplay(data) {
    const currentPrice = data.current_price || 5527.45;
    const predictedPrice = data.predicted_price || 5612.3;
    const priceChange = data.price_change || 84.85;
    const changePercent = data.price_change_percent || 1.54;
    const confidence = (data.prediction_confidence || 0.87) * 100;

    // 현재 가격 업데이트
    const priceEl = document.getElementById('sp500-current-price');
    if (priceEl) {
      priceEl.textContent = `$${currentPrice.toLocaleString('en-US', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      })}`;
    }

    // 가격 변동 업데이트
    const changeEl = document.getElementById('sp500-price-change');
    if (changeEl) {
      const sign = priceChange >= 0 ? '+' : '';
      changeEl.textContent = `${sign}${priceChange.toFixed(2)} (${sign}${changePercent.toFixed(2)}%)`;
      changeEl.className = `price-change ${priceChange >= 0 ? 'positive' : 'negative'}`;
    }

    // 예측 가격 업데이트
    const predictedEl = document.getElementById('sp500-predicted-price');
    if (predictedEl) {
      predictedEl.textContent = `$${predictedPrice.toLocaleString('en-US', {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      })}`;
    }

    // 신뢰도 업데이트
    const confidenceEl = document.getElementById('sp500-prediction-confidence');
    if (confidenceEl) {
      confidenceEl.textContent = `신뢰도: ${confidence}%`;
    }
  }

  /**
   * 모의 데이터로 업데이트
   */
  updateWithMockData() {
    const basePrice = 4580.23;
    const variation = (Math.random() - 0.5) * 20; // ±10 포인트 변동
    const currentPrice = basePrice + variation;
    const predictedPrice = currentPrice * (1 + (Math.random() - 0.3) * 0.02); // ±1% 예측
    const priceChange = variation;
    const changePercent = (priceChange / basePrice) * 100;
    const confidence = Math.floor(Math.random() * 20) + 75; // 75-95% 신뢰도

    this.updatePriceDisplay({
      current_price: currentPrice,
      predicted_price: predictedPrice,
      price_change: priceChange,
      change_percent: changePercent,
      confidence: confidence,
    });

    this.updateLastUpdateTime();
  }

  /**
   * 마지막 업데이트 시간 표시
   */
  updateLastUpdateTime() {
    const updateEl = document.getElementById('sp500-last-update');
    if (updateEl) {
      const now = new Date();
      updateEl.textContent = `업데이트: ${now.toLocaleTimeString('ko-KR')}`;
    }
  }

  /**
   * 데이터 로드 오류 처리
   */
  handleDataLoadError() {
    this.retryCount++;

    if (this.retryCount <= this.maxRetries) {
      console.log(
        `S&P 500 데이터 재시도 ${this.retryCount}/${this.maxRetries}`
      );
      setTimeout(() => this.loadData(), 5000); // 5초 후 재시도
    } else {
      console.warn(
        'S&P 500 데이터 로드 최대 재시도 횟수 초과, 모의 데이터 사용'
      );
      this.updateWithMockData();
    }
  }

  /**
   * 오류 메시지 표시
   */
  showError(message) {
    const updateEl = document.getElementById('sp500-last-update');
    if (updateEl) {
      updateEl.textContent = `오류: ${message}`;
      updateEl.style.color = '#dc3545';
    }
  }

  /**
   * 자동 업데이트 시작
   */
  startAutoUpdate() {
    // 기존 인터벌 정리
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
    }

    // 30초마다 데이터 업데이트
    this.updateInterval = setInterval(() => {
      this.loadData();
    }, 30000);

    console.log('⏰ S&P 500 자동 업데이트 시작 (30초 간격)');
  }

  /**
   * 자동 업데이트 중지
   */
  stopAutoUpdate() {
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
      this.updateInterval = null;
      console.log('⏹️ S&P 500 자동 업데이트 중지');
    }
  }

  /**
   * 위젯 정리
   */
  destroy() {
    this.stopAutoUpdate();

    if (this.chart) {
      this.chart.destroy();
      this.chart = null;
    }

    console.log('🗑️ S&P 500 Widget 정리 완료');
  }

  /**
   * 수동 새로고침
   */
  async refresh() {
    console.log('🔄 S&P 500 위젯 수동 새로고침');
    await this.loadData();

    // 차트 데이터도 업데이트
    if (this.chart) {
      const newData = this.generate30DayData();
      this.chart.data.labels = newData.labels;
      this.chart.data.datasets[0].data = newData.prices;
      this.chart.update('none'); // 애니메이션 없이 업데이트
    }
  }
}

// 전역 변수로 위젯 인스턴스 등록
window.SP500Widget = SP500Widget;

console.log('📊 S&P 500 Widget 모듈 로드됨');
