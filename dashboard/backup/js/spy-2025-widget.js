/**
 * SPY 2025년 상반기 데이터 위젯
 * - 2025년 1월~6월 실제 가격 차트
 * - 기술적 분석 기반 예측 결과 표시
 * - 월별 성과 분석
 */

class SPY2025Widget {
  constructor() {
    this.chart = null;
    this.actualData = null;
    this.predictionData = null;
    this.isLoading = false;

    console.log('📊 SPY 2025 Widget 초기화됨');
  }

  /**
   * 위젯 초기화
   */
  async init() {
    // 중복 초기화 방지
    if (this.isLoading || this.actualData) {
      console.log('⚠️ SPY 2025 Widget 이미 초기화됨 - 건너뜀');
      return;
    }

    try {
      this.showLoading(true);
      
      // 데이터 로드
      await Promise.all([
        this.loadActualData(),
        this.loadPredictionData()
      ]);

      // 차트 생성
      await this.createChart();
      
      // 통계 업데이트
      this.updateStats();

      console.log('✅ SPY 2025 Widget 초기화 완료');
    } catch (error) {
      console.error('❌ SPY 2025 Widget 초기화 실패:', error);
      this.showError('SPY 2025 데이터 로드에 실패했습니다.');
    } finally {
      this.showLoading(false);
    }
  }

  /**
   * 실제 가격 데이터 로드
   */
  async loadActualData() {
    try {
      const response = await fetch('../data/raw/spy_2025_h1.json?t=' + Date.now(), {
        headers: {
          'Cache-Control': 'no-cache'
        }
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      
      this.actualData = await response.json();
      console.log('📊 SPY 2025 실제 데이터 로드됨:', this.actualData.total_records, '개 항목');
    } catch (error) {
      console.error('❌ 실제 데이터 로드 실패:', error);
      throw error;
    }
  }

  /**
   * 예측 데이터 로드
   */
  async loadPredictionData() {
    try {
      const response = await fetch('../data/raw/spy_2025_h1_predictions.json?t=' + Date.now(), {
        headers: {
          'Cache-Control': 'no-cache'
        }
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      
      this.predictionData = await response.json();
      console.log('🔮 SPY 2025 예측 데이터 로드됨:', this.predictionData.predictions.length, '개 예측');
    } catch (error) {
      console.error('❌ 예측 데이터 로드 실패:', error);
      throw error;
    }
  }

  /**
   * 차트 생성
   */
  async createChart() {
    if (typeof Chart === 'undefined') {
      console.error('❌ Chart.js가 로드되지 않음');
      return;
    }

    const ctx = document.getElementById('spy-2025-chart');
    if (!ctx) {
      console.error('❌ SPY 2025 차트 캔버스를 찾을 수 없습니다.');
      return;
    }

    // 기존 차트 제거
    if (this.chart) {
      this.chart.destroy();
    }

    // 데이터 준비
    const chartData = this.prepareChartData();

    this.chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: chartData.labels,
        datasets: [
          {
            label: '📈 실제 SPY 가격',
            data: chartData.actualPrices,
            borderColor: '#1976D2',
            backgroundColor: 'rgba(25, 118, 210, 0.1)',
            borderWidth: 2,
            fill: false,
            tension: 0.1,
            pointRadius: 2,
            pointHoverRadius: 6
          },
          {
            label: '🔮 예측 신호 (Up)',
            data: chartData.upPredictions,
            borderColor: '#4CAF50',
            backgroundColor: 'rgba(76, 175, 80, 0.3)',
            borderWidth: 1,
            borderDash: [5, 5],
            fill: false,
            pointRadius: 3,
            pointHoverRadius: 8,
            pointStyle: 'triangle',
            showLine: false // 점만 표시
          },
          {
            label: '🔽 예측 신호 (Down)',
            data: chartData.downPredictions,
            borderColor: '#F44336',
            backgroundColor: 'rgba(244, 67, 54, 0.3)',
            borderWidth: 1,
            borderDash: [5, 5],
            fill: false,
            pointRadius: 3,
            pointHoverRadius: 8,
            pointStyle: 'triangle',
            rotation: 180,
            showLine: false // 점만 표시
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: 'index',
          intersect: false
        },
        plugins: {
          title: {
            display: true,
            text: 'SPY 2025년 상반기 실제 가격 vs AI 예측 신호',
            font: { size: 16 }
          },
          legend: {
            display: true,
            position: 'top'
          },
          tooltip: {
            callbacks: {
              title: function(context) {
                return context[0].label;
              },
              label: function(context) {
                if (context.datasetIndex === 0) {
                  return `실제 가격: $${context.parsed.y.toFixed(2)}`;
                } else if (context.datasetIndex === 1) {
                  return `상승 예측 (신뢰도: ${(context.parsed.y * 0.01).toFixed(1)}%)`;
                } else if (context.datasetIndex === 2) {
                  return `하락 예측 (신뢰도: ${(context.parsed.y * 0.01).toFixed(1)}%)`;
                }
              }
            }
          }
        },
        scales: {
          x: {
            display: true,
            title: {
              display: true,
              text: '날짜'
            }
          },
          y: {
            display: true,
            title: {
              display: true,
              text: 'SPY 가격 ($)'
            },
            beginAtZero: false
          }
        }
      }
    });

    console.log('📊 SPY 2025 차트 생성 완료');
  }

  /**
   * 차트 데이터 준비
   */
  prepareChartData() {
    if (!this.actualData || !this.predictionData) {
      return { labels: [], actualPrices: [], upPredictions: [], downPredictions: [] };
    }

    const labels = [];
    const actualPrices = [];
    const upPredictions = [];
    const downPredictions = [];

    // 실제 데이터를 기준으로 매칭
    for (const actualItem of this.actualData.data) {
      const date = actualItem.date;
      const prediction = this.predictionData.predictions.find(p => p.date === date);
      
      labels.push(date);
      actualPrices.push(actualItem.close);
      
      if (prediction) {
        if (prediction.prediction === 1) { // Up 예측
          upPredictions.push(actualItem.close);
          downPredictions.push(null);
        } else { // Down 예측
          upPredictions.push(null);
          downPredictions.push(actualItem.close);
        }
      } else {
        upPredictions.push(null);
        downPredictions.push(null);
      }
    }

    return { labels, actualPrices, upPredictions, downPredictions };
  }

  /**
   * 통계 업데이트
   */
  updateStats() {
    if (!this.actualData || !this.predictionData) return;

    const totalDays = this.actualData.total_records;
    const startPrice = this.actualData.data[0].close;
    const endPrice = this.actualData.data[this.actualData.data.length - 1].close;
    const totalReturn = ((endPrice - startPrice) / startPrice * 100);
    
    const accuracy = this.predictionData.model_info.accuracy_on_period * 100;
    const correctPredictions = this.predictionData.model_info.correct_predictions;
    const totalPredictions = this.predictionData.model_info.total_predictions;

    // 월별 정확도 계산
    const monthlyAccuracy = this.calculateMonthlyAccuracy();

    // DOM 업데이트
    this.updateStatsDisplay({
      period: '2025년 1월 ~ 6월',
      totalDays,
      startPrice: startPrice.toFixed(2),
      endPrice: endPrice.toFixed(2),
      totalReturn: totalReturn.toFixed(2),
      accuracy: accuracy.toFixed(1),
      correctPredictions,
      totalPredictions,
      monthlyAccuracy
    });
  }

  /**
   * 월별 정확도 계산
   */
  calculateMonthlyAccuracy() {
    const monthlyStats = {};
    
    for (const pred of this.predictionData.predictions) {
      const month = pred.date.substring(0, 7); // YYYY-MM
      if (!monthlyStats[month]) {
        monthlyStats[month] = { correct: 0, total: 0 };
      }
      
      const actualDirection = pred.actual_return > 0 ? 1 : 0;
      if (actualDirection === pred.prediction) {
        monthlyStats[month].correct++;
      }
      monthlyStats[month].total++;
    }

    return monthlyStats;
  }

  /**
   * 통계 표시 업데이트
   */
  updateStatsDisplay(stats) {
    const statsContainer = document.getElementById('spy-2025-stats');
    if (!statsContainer) return;

    let monthlyHtml = '';
    for (const [month, data] of Object.entries(stats.monthlyAccuracy)) {
      const accuracy = (data.correct / data.total * 100).toFixed(1);
      monthlyHtml += `
        <div class="month-stat">
          <span class="month">${month}</span>
          <span class="accuracy">${accuracy}%</span>
          <span class="count">(${data.correct}/${data.total})</span>
        </div>
      `;
    }

    statsContainer.innerHTML = `
      <div class="stats-grid">
        <div class="stat-card">
          <h3>📊 기간 정보</h3>
          <p><strong>분석 기간:</strong> ${stats.period}</p>
          <p><strong>거래일 수:</strong> ${stats.totalDays}일</p>
        </div>
        
        <div class="stat-card">
          <h3>💰 가격 변동</h3>
          <p><strong>시작 가격:</strong> $${stats.startPrice}</p>
          <p><strong>종료 가격:</strong> $${stats.endPrice}</p>
          <p><strong>수익률:</strong> <span class="${stats.totalReturn >= 0 ? 'positive' : 'negative'}">${stats.totalReturn}%</span></p>
        </div>
        
        <div class="stat-card">
          <h3>🔮 예측 성과</h3>
          <p><strong>전체 정확도:</strong> ${stats.accuracy}%</p>
          <p><strong>정확한 예측:</strong> ${stats.correctPredictions}/${stats.totalPredictions}</p>
          <p><strong>모델 유형:</strong> 기술적 분석</p>
        </div>
        
        <div class="stat-card monthly-accuracy">
          <h3>📅 월별 정확도</h3>
          ${monthlyHtml}
        </div>
      </div>
    `;
  }

  /**
   * 로딩 상태 표시
   */
  showLoading(show) {
    this.isLoading = show;
    const loadingEl = document.getElementById('spy-2025-loading');
    if (loadingEl) {
      loadingEl.style.display = show ? 'block' : 'none';
    }
  }

  /**
   * 에러 표시
   */
  showError(message) {
    const errorEl = document.getElementById('spy-2025-error');
    if (errorEl) {
      errorEl.textContent = message;
      errorEl.style.display = 'block';
    }
  }

  /**
   * 위젯 정리
   */
  destroy() {
    if (this.chart) {
      this.chart.destroy();
      this.chart = null;
    }
  }
}

// 전역에서 사용할 수 있도록 내보내기
window.SPY2025Widget = SPY2025Widget;