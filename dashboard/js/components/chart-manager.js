/**
 * Chart Manager
 * Handles all Chart.js chart creation and management
 */

import { logger, createLogger } from '../utils/logger.js';

const chartLogger = createLogger('ChartManager');

export class ChartManager {
    constructor() {
        this.charts = new Map();
        this.defaultOptions = this.getDefaultChartOptions();
        chartLogger.info('Chart Manager initialized');
    }

    getDefaultChartOptions() {
        return {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    position: 'top',
                    labels: {
                        usePointStyle: true,
                        padding: 20
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: 'white',
                    bodyColor: 'white',
                    borderColor: 'rgba(255, 255, 255, 0.1)',
                    borderWidth: 1
                },
                datalabels: {
                    display: false  // 기본 차트 옵션에서 데이터 라벨 비활성화
                }
            },
            scales: {
                x: {
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    }
                }
            },
            animation: {
                duration: 750,
                easing: 'easeInOutQuart'
            }
        };
    }

    createLineChart(canvasId, data, options = {}) {
        const timer = chartLogger.startTimer(`create_line_chart_${canvasId}`);
        
        try {
            const canvas = document.getElementById(canvasId);
            if (!canvas) {
                throw new Error(`Canvas element with ID '${canvasId}' not found`);
            }

            // Destroy existing chart
            if (this.charts.has(canvasId)) {
                this.destroyChart(canvasId);
            }

            const ctx = canvas.getContext('2d');
            const chartOptions = {
                type: 'line',
                data,
                options: {
                    ...this.defaultOptions,
                    ...options,
                    plugins: {
                        ...this.defaultOptions.plugins,
                        ...options.plugins
                    }
                }
            };

            const chart = new Chart(ctx, chartOptions);
            this.charts.set(canvasId, chart);

            chartLogger.info(`Line chart created: ${canvasId}`);
            chartLogger.endTimer(timer);
            return chart;

        } catch (error) {
            chartLogger.error(`Failed to create line chart: ${canvasId}`, error);
            chartLogger.endTimer(timer);
            throw error;
        }
    }

    createBarChart(canvasId, data, options = {}) {
        const timer = chartLogger.startTimer(`create_bar_chart_${canvasId}`);
        
        try {
            const canvas = document.getElementById(canvasId);
            if (!canvas) {
                throw new Error(`Canvas element with ID '${canvasId}' not found`);
            }

            // Destroy existing chart
            if (this.charts.has(canvasId)) {
                this.destroyChart(canvasId);
            }

            const ctx = canvas.getContext('2d');
            const chartOptions = {
                type: 'bar',
                data,
                options: {
                    ...this.defaultOptions,
                    ...options,
                    plugins: {
                        ...this.defaultOptions.plugins,
                        ...options.plugins
                    }
                }
            };

            const chart = new Chart(ctx, chartOptions);
            this.charts.set(canvasId, chart);

            chartLogger.info(`Bar chart created: ${canvasId}`);
            chartLogger.endTimer(timer);
            return chart;

        } catch (error) {
            chartLogger.error(`Failed to create bar chart: ${canvasId}`, error);
            chartLogger.endTimer(timer);
            throw error;
        }
    }

    createDoughnutChart(canvasId, data, options = {}) {
        const timer = chartLogger.startTimer(`create_doughnut_chart_${canvasId}`);
        
        try {
            const canvas = document.getElementById(canvasId);
            if (!canvas) {
                throw new Error(`Canvas element with ID '${canvasId}' not found`);
            }

            // Destroy existing chart
            if (this.charts.has(canvasId)) {
                this.destroyChart(canvasId);
            }

            const ctx = canvas.getContext('2d');
            const chartOptions = {
                type: 'doughnut',
                data,
                options: {
                    ...this.defaultOptions,
                    ...options,
                    plugins: {
                        ...this.defaultOptions.plugins,
                        ...options.plugins
                    }
                }
            };

            const chart = new Chart(ctx, chartOptions);
            this.charts.set(canvasId, chart);

            chartLogger.info(`Doughnut chart created: ${canvasId}`);
            chartLogger.endTimer(timer);
            return chart;

        } catch (error) {
            chartLogger.error(`Failed to create doughnut chart: ${canvasId}`, error);
            chartLogger.endTimer(timer);
            throw error;
        }
    }

    updateChart(canvasId, newData) {
        const chart = this.charts.get(canvasId);
        if (!chart) {
            chartLogger.warn(`Chart '${canvasId}' not found for update`);
            return;
        }

        try {
            chart.data = newData;
            chart.update('active');
            chartLogger.debug(`Chart updated: ${canvasId}`);
        } catch (error) {
            chartLogger.error(`Failed to update chart: ${canvasId}`, error);
        }
    }

    destroyChart(canvasId) {
        const chart = this.charts.get(canvasId);
        if (chart) {
            chart.destroy();
            this.charts.delete(canvasId);
            chartLogger.info(`Chart destroyed: ${canvasId}`);
        }
    }

    destroyAllCharts() {
        for (const [canvasId, chart] of this.charts) {
            chart.destroy();
        }
        this.charts.clear();
        chartLogger.info('All charts destroyed');
    }

    getChart(canvasId) {
        return this.charts.get(canvasId);
    }

    // Specialized chart creators for dashboard

    createStockChart(canvasId, stockData, predictions = null) {
        console.log('Creating stock chart with data:', {
            stockDataLength: stockData?.length,
            predictionsLength: predictions?.length,
            sampleStock: stockData?.[0],
            samplePrediction: predictions?.[0]
        });

        // Extract data array if wrapped in object
        const actualStockData = stockData?.data || stockData;
        const actualPredictions = predictions?.predictions || predictions;

        if (!actualStockData || actualStockData.length === 0) {
            console.error('No stock data available for chart');
            return null;
        }

        // Create labels from dates
        const labels = actualStockData.map(item => {
            const date = new Date(item.date);
            return date.toLocaleDateString('ko-KR', { month: 'short', day: 'numeric' });
        });

        // Process stock price data - use adj_close if available, fallback to close
        const stockPrices = actualStockData.map(item => {
            return parseFloat(item.adj_close || item.close);
        });

        const datasets = [{
            label: 'SPY Actual Price',
            data: stockPrices,
            borderColor: 'rgb(75, 192, 192)',
            backgroundColor: 'rgba(75, 192, 192, 0.1)',
            fill: true,
            tension: 0.2,
            pointRadius: 0,        // 기본 포인트 숨김 (깔끔한 라인)
            pointHoverRadius: 4,   // 호버 시에만 포인트 표시
            borderWidth: 2
        }];

        // Process predictions if available
        if (actualPredictions && actualPredictions.length > 0) {
            console.log('Processing predictions data:', actualPredictions.slice(0, 3));

            // Create prediction visualization by matching dates
            const predictionPrices = [];

            actualStockData.forEach((stockItem, index) => {
                // Find matching prediction by date
                const matchingPred = actualPredictions.find(pred => pred.date === stockItem.date);

                if (matchingPred) {
                    const basePrice = parseFloat(matchingPred.actual_price);
                    // Create visual distinction: up prediction slightly higher, down prediction slightly lower
                    const adjustment = matchingPred.prediction === 1 ? basePrice * 0.005 : basePrice * -0.005;
                    predictionPrices.push(basePrice + adjustment);
                } else {
                    predictionPrices.push(null); // No prediction for this date
                }
            });

            datasets.push({
                label: 'AI Prediction Trend',
                data: predictionPrices,
                borderColor: 'rgb(255, 99, 132)',
                backgroundColor: 'rgba(255, 99, 132, 0.1)',
                borderDash: [5, 5],
                fill: false,
                tension: 0.2,
                pointRadius: 0,        // 기본 포인트 숨김
                pointHoverRadius: 4,   // 호버 시에만 포인트 표시
                borderWidth: 2,
                spanGaps: true // Connect points even with null values
            });
        }

        const options = {
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Date'
                    },
                    ticks: {
                        maxTicksLimit: 8, // 최대 8개 라벨만 표시
                        callback: function(value, index) {
                            // 더 간격을 벌려서 표시
                            const totalTicks = labels.length;
                            const step = Math.ceil(totalTicks / 6);
                            return index % step === 0 ? labels[index] : '';
                        }
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Price ($)'
                    },
                    beginAtZero: false,
                    ticks: {
                        maxTicksLimit: 6, // Y축도 최대 6개 라벨
                        callback: function(value) {
                            return '$' + value.toFixed(0); // 소수점 제거하여 간소화
                        }
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'SPY Stock Price vs AI Predictions (2025 H1)'
                },
                legend: {
                    display: true,
                    position: 'top'
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: 'white',
                    bodyColor: 'white',
                    borderColor: 'rgba(255, 255, 255, 0.1)',
                    borderWidth: 1,
                    callbacks: {
                        label: function(context) {
                            // 툴팁 텍스트 간소화
                            return context.dataset.label + ': $' + context.parsed.y.toFixed(2);
                        }
                    }
                },
                datalabels: {
                    display: false  // 모든 데이터 라벨 완전 비활성화
                }
            },
            interaction: {
                intersect: false,
                mode: 'index'
            },
            responsive: true,
            maintainAspectRatio: false,
            elements: {
                point: {
                    hoverRadius: 6, // 호버 시 포인트 크기
                    hitRadius: 10   // 클릭 감지 영역
                }
            }
        };

        return this.createLineChart(canvasId, { labels, datasets }, options);
    }

    createModelPerformanceChart(canvasId, modelData) {
        const labels = modelData.map(model => model.name);
        const accuracyData = modelData.map(model => model.accuracy * 100);
        const f1Data = modelData.map(model => model.f1_score * 100);

        const data = {
            labels,
            datasets: [
                {
                    label: 'Accuracy (%)',
                    data: accuracyData,
                    backgroundColor: 'rgba(54, 162, 235, 0.8)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                },
                {
                    label: 'F1 Score (%)',
                    data: f1Data,
                    backgroundColor: 'rgba(255, 206, 86, 0.8)',
                    borderColor: 'rgba(255, 206, 86, 1)',
                    borderWidth: 1
                }
            ]
        };

        const options = {
            plugins: {
                title: {
                    display: true,
                    text: 'Model Performance Comparison'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Performance (%)'
                    }
                }
            }
        };

        return this.createBarChart(canvasId, data, options);
    }

    createFeatureImportanceChart(canvasId, featureData) {
        const sortedFeatures = featureData
            .sort((a, b) => b.importance - a.importance)
            .slice(0, 10); // Top 10 features

        const data = {
            labels: sortedFeatures.map(f => f.name),
            datasets: [{
                label: 'Importance',
                data: sortedFeatures.map(f => f.importance),
                backgroundColor: [
                    'rgba(255, 99, 132, 0.8)',
                    'rgba(54, 162, 235, 0.8)',
                    'rgba(255, 206, 86, 0.8)',
                    'rgba(75, 192, 192, 0.8)',
                    'rgba(153, 102, 255, 0.8)',
                    'rgba(255, 159, 64, 0.8)',
                    'rgba(199, 199, 199, 0.8)',
                    'rgba(83, 102, 255, 0.8)',
                    'rgba(255, 99, 255, 0.8)',
                    'rgba(99, 255, 132, 0.8)'
                ]
            }]
        };

        const options = {
            indexAxis: 'y',
            plugins: {
                title: {
                    display: true,
                    text: 'Top 10 Feature Importance'
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Importance Score'
                    }
                }
            }
        };

        return this.createBarChart(canvasId, data, options);
    }
}

// Create default chart manager instance
export const chartManager = new ChartManager();