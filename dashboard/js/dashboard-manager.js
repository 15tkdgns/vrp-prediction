/**
 * Dashboard Manager
 * Main controller that orchestrates all dashboard components
 */

import { logger } from './utils/logger.js';
import { api } from './utils/api.js';
import { chartManager } from './components/chart-manager.js';
import { dataLoader } from './components/data-loader.js';
import { tabManager } from './components/tab-manager.js';

class DashboardManager {
    constructor() {
        this.initialized = false;
        this.data = null;
        this.updateInterval = null;
        this.errorCount = 0;
        this.maxRetries = 3;
        
        logger.info('🚀 Dashboard Manager starting...');
        this.initialize();
    }

    async initialize() {
        const timer = logger.startTimer('dashboard_initialization');

        try {
            // Show loading state
            this.showLoading();

            // Initialize tab manager first
            tabManager.initializeFromURL();

            // Register tab loaders
            this.registerTabLoaders();

            // Load initial data
            await this.loadInitialData();

            // Set up event listeners
            this.setupEventListeners();

            // Initialize charts for current tab
            await this.initializeCurrentTab();

            // Enable auto-refresh
            dataLoader.enableAutoRefresh();

            // Hide loading and show dashboard
            this.showDashboard();

            this.initialized = true;
            logger.success('✅ Dashboard initialized successfully');
            logger.endTimer(timer);

        } catch (error) {
            logger.error('❌ Dashboard initialization failed', error);
            this.showError('Failed to initialize dashboard: ' + error.message);
            logger.endTimer(timer);
        }
    }

    async loadInitialData() {
        try {
            this.data = await dataLoader.loadDashboardData();
            logger.info('Dashboard data loaded', {
                stockDataPoints: this.data.stockData?.length || 0,
                predictions: this.data.predictions?.length || 0,
                models: this.data.modelPerformance?.length || 0
            });

            // Update UI with best model data
            this.updateBestModelDisplay();
        } catch (error) {
            logger.error('Failed to load initial data', error);
            throw error;
        }
    }

    registerTabLoaders() {
        tabManager.registerTab('overview', () => this.loadOverviewTab());
        tabManager.registerTab('predictions', () => this.loadPredictionsTab());
        tabManager.registerTab('models', () => this.loadModelsTab());
        tabManager.registerTab('analytics', () => this.loadAnalyticsTab());
        tabManager.registerTab('monitoring', () => this.loadMonitoringTab());
    }

    setupEventListeners() {
        // Tab switch events
        window.addEventListener('tabSwitch', (event) => {
            this.handleTabSwitch(event.detail);
        });

        // Data refresh events
        window.addEventListener('dashboardDataRefresh', (event) => {
            this.handleDataRefresh();
        });

        // Window resize events
        window.addEventListener('resize', () => {
            this.handleResize();
        });

        // Page visibility changes
        document.addEventListener('visibilitychange', () => {
            this.handleVisibilityChange();
        });

        // Error events
        window.addEventListener('error', (event) => {
            logger.error('Global error occurred', {
                message: event.message,
                filename: event.filename,
                line: event.lineno
            });
        });
    }

    async initializeCurrentTab() {
        const currentTab = tabManager.getCurrentTab();
        await tabManager.loadTabContent(currentTab);
    }

    // Tab loading methods
    async loadOverviewTab() {
        if (!this.data) return;

        try {
            // Update key metrics
            this.updateKeyMetrics();

            // Create main price chart - using priceChart ID to match HTML
            if (this.data.stockData && this.data.stockData.length > 0) {
                chartManager.createStockChart(
                    'priceChart',
                    this.data.stockData,
                    this.data.predictions
                );
            }

            logger.info('Overview tab loaded');
        } catch (error) {
            logger.error('Failed to load overview tab', error);
        }
    }

    async loadPredictionsTab() {
        if (!this.data) return;

        try {
            // Create predictions chart
            if (this.data.predictions && this.data.predictions.length > 0) {
                const chartData = this.preparePredictionsChartData();
                chartManager.createLineChart('predictions-chart', chartData, {
                    plugins: {
                        title: {
                            display: true,
                            text: 'AI Model Predictions'
                        }
                    }
                });
            }

            // Update predictions list
            this.updatePredictionsList();

            logger.info('Predictions tab loaded');
        } catch (error) {
            logger.error('Failed to load predictions tab', error);
        }
    }

    async loadModelsTab() {
        if (!this.data) return;

        try {
            // Create model performance chart
            if (this.data.modelPerformance && this.data.modelPerformance.length > 0) {
                chartManager.createModelPerformanceChart(
                    'model-performance-chart',
                    this.data.modelPerformance
                );
            }

            // Create feature importance chart
            if (this.data.featureAnalysis && this.data.featureAnalysis.length > 0) {
                chartManager.createFeatureImportanceChart(
                    'feature-importance-chart',
                    this.data.featureAnalysis
                );
            }

            logger.info('Models tab loaded');
        } catch (error) {
            logger.error('Failed to load models tab', error);
        }
    }

    async loadAnalyticsTab() {
        try {
            // Load advanced analytics content
            const analyticsContent = document.getElementById('analytics-content');
            if (analyticsContent) {
                analyticsContent.innerHTML = this.generateAnalyticsHTML();
            }

            logger.info('Analytics tab loaded');
        } catch (error) {
            logger.error('Failed to load analytics tab', error);
        }
    }

    async loadMonitoringTab() {
        try {
            await this.updateSystemHealth();
            await this.updatePerformanceMetrics();

            logger.info('Monitoring tab loaded');
        } catch (error) {
            logger.error('Failed to load monitoring tab', error);
        }
    }

    // UI Update methods
    updateKeyMetrics() {
        if (!this.data.stockData || this.data.stockData.length === 0) return;

        const latestPrice = this.data.stockData[this.data.stockData.length - 1];
        const previousPrice = this.data.stockData[this.data.stockData.length - 2];

        // Current price
        const currentPriceEl = document.getElementById('current-price');
        if (currentPriceEl) {
            currentPriceEl.textContent = `$${latestPrice.close.toFixed(2)}`;
        }

        // Price change
        const priceChangeEl = document.getElementById('price-change');
        if (priceChangeEl && previousPrice) {
            const change = latestPrice.close - previousPrice.close;
            const changePercent = (change / previousPrice.close) * 100;
            const changeText = `${change >= 0 ? '+' : ''}${change.toFixed(2)} (${changePercent.toFixed(2)}%)`;
            priceChangeEl.textContent = changeText;
            priceChangeEl.className = `text-muted ${change >= 0 ? 'text-success' : 'text-danger'}`;
        }

        // AI Prediction
        const aiPredictionEl = document.getElementById('ai-prediction');
        if (aiPredictionEl && this.data.predictions && this.data.predictions.length > 0) {
            const nextPrediction = this.data.predictions[0];
            const trend = nextPrediction.predicted_price > latestPrice.close ? '📈' : '📉';
            aiPredictionEl.innerHTML = `${trend} $${nextPrediction.predicted_price.toFixed(2)}`;
        }

        // Model accuracy
        const accuracyEl = document.getElementById('model-accuracy');
        if (accuracyEl && this.data.modelPerformance) {
            const bestModel = this.getBestModel();
            if (bestModel) {
                if (bestModel.scoreType === 'mae') {
                    accuracyEl.textContent = `${(bestModel.score * 100).toFixed(3)}% MAE`;
                } else {
                    accuracyEl.textContent = `${bestModel.score.toFixed(2)}% MAPE`;
                }
            } else {
                accuracyEl.textContent = 'N/A';
            }
        }

        // Risk level (based on recent volatility)
        const riskLevelEl = document.getElementById('risk-level');
        if (riskLevelEl && this.data.stockData.length >= 20) {
            const recentPrices = this.data.stockData.slice(-20).map(d => d.close);
            const volatility = this.calculateVolatility(recentPrices);
            let riskLevel = 'Low';
            let riskClass = 'text-success';
            
            if (volatility > 0.3) {
                riskLevel = 'High';
                riskClass = 'text-danger';
            } else if (volatility > 0.15) {
                riskLevel = 'Medium';
                riskClass = 'text-warning';
            }
            
            riskLevelEl.textContent = riskLevel;
            riskLevelEl.className = riskClass;
        }
    }

    updatePredictionsList() {
        const listEl = document.getElementById('predictions-list');
        if (!listEl || !this.data.predictions) return;

        const html = this.data.predictions.slice(0, 10).map(pred => {
            const date = new Date(pred.date).toLocaleDateString();
            const confidence = (pred.confidence * 100).toFixed(0);
            const trend = pred.predicted_price > this.data.stockData[this.data.stockData.length - 1].close ? 
                'text-success' : 'text-danger';
            
            return `
                <div class="d-flex justify-content-between align-items-center border-bottom py-2">
                    <div>
                        <small class="text-muted">${date}</small><br>
                        <span class="${trend}">$${pred.predicted_price.toFixed(2)}</span>
                    </div>
                    <div class="text-end">
                        <small class="text-muted">${pred.model}</small><br>
                        <span class="badge bg-primary">${confidence}%</span>
                    </div>
                </div>
            `;
        }).join('');

        listEl.innerHTML = html;
    }

    async updateSystemHealth() {
        try {
            const systemStatus = await api.getSystemStatus();
            const healthEl = document.getElementById('system-health');
            
            if (healthEl) {
                healthEl.innerHTML = this.generateSystemHealthHTML(systemStatus);
            }
        } catch (error) {
            logger.warn('Failed to get system status', error);
        }
    }

    async updatePerformanceMetrics() {
        const metricsEl = document.getElementById('performance-metrics');
        if (!metricsEl) return;

        const metrics = {
            'Load Time': `${((Date.now() - performance.timeOrigin) / 1000).toFixed(2)}s`,
            'Memory Usage': `${(performance.memory?.usedJSHeapSize / 1024 / 1024).toFixed(2) || 'N/A'} MB`,
            'API Calls': api.cache.size,
            'Charts Rendered': chartManager.charts.size,
            'Last Update': new Date().toLocaleTimeString()
        };

        const html = Object.entries(metrics).map(([key, value]) => `
            <div class="d-flex justify-content-between py-2 border-bottom">
                <span>${key}:</span>
                <strong>${value}</strong>
            </div>
        `).join('');

        metricsEl.innerHTML = html;
    }

    // Event handlers
    handleTabSwitch(detail) {
        logger.debug(`Tab switched from ${detail.from} to ${detail.to}`);
        
        // Load tab content if not loaded
        tabManager.loadTabContent(detail.to);
    }

    handleDataRefresh() {
        logger.info('Data refreshed, updating dashboard');
        this.data = dataLoader.getCachedData('dashboardData');
        
        // Update current tab
        const currentTab = tabManager.getCurrentTab();
        this[`load${currentTab.charAt(0).toUpperCase() + currentTab.slice(1)}Tab`]();
    }

    handleResize() {
        // Debounce resize events
        clearTimeout(this.resizeTimeout);
        this.resizeTimeout = setTimeout(() => {
            chartManager.charts.forEach(chart => {
                if (chart.resize) {
                    chart.resize();
                }
            });
        }, 250);
    }

    handleVisibilityChange() {
        if (document.hidden) {
            dataLoader.disableAutoRefresh();
            logger.debug('Page hidden, auto-refresh disabled');
        } else {
            dataLoader.enableAutoRefresh();
            logger.debug('Page visible, auto-refresh enabled');
        }
    }

    // Utility methods
    preparePredictionsChartData() {
        const datasets = [];

        if (this.data.predictions) {
            const predictionData = this.data.predictions.map(pred => ({
                x: pred.date,
                y: pred.predicted_price
            }));

            datasets.push({
                label: 'Predictions',
                data: predictionData,
                borderColor: 'rgb(255, 99, 132)',
                backgroundColor: 'rgba(255, 99, 132, 0.1)',
                borderDash: [5, 5]
            });
        }

        return { datasets };
    }

    calculateVolatility(prices) {
        if (prices.length < 2) return 0;
        
        const returns = [];
        for (let i = 1; i < prices.length; i++) {
            returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
        }
        
        const mean = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
        const variance = returns.reduce((sum, ret) => sum + Math.pow(ret - mean, 2), 0) / returns.length;
        
        return Math.sqrt(variance);
    }

    generateAnalyticsHTML() {
        const bestModel = this.getBestModel();

        // Dynamic insights based on actual model performance
        let modelInsights = [];
        let technicalAnalysis = [];

        if (bestModel) {
            const displayName = bestModel.name
                .replace('_ultra_conservative', ' Ultra Conservative')
                .replace('_truly_leak_free', ' Truly Leak-Free')
                .replace('_leak_free', ' Leak-Free')
                .replace('_', ' ')
                .replace(/\b\w/g, l => l.toUpperCase());

            // Model insights based on actual performance
            if (bestModel.scoreType === 'mae') {
                const directionAcc = bestModel.direction_mean || 0;
                modelInsights = [
                    `🎯 Best Model: ${displayName}`,
                    `📊 MAE: ${(bestModel.score * 100).toFixed(3)}%`,
                    `🔍 Direction Acc: ${directionAcc.toFixed(1)}%`,
                    directionAcc > 55 && directionAcc <= 65 ?
                        '✅ Realistic Performance' :
                        directionAcc > 65 ? '⚠️ Check for Data Leakage' : '📈 Conservative Model'
                ];
            } else {
                modelInsights = [
                    `🎯 Best Model: ${displayName}`,
                    `📊 MAPE: ${bestModel.score.toFixed(2)}%`,
                    `🔍 Direction Acc: ${(bestModel.direction_accuracy || 0).toFixed(1)}%`,
                    bestModel.score < 2 ? '⚠️ Check for Data Leakage' : '✅ Performance OK'
                ];
            }

            // Technical analysis based on recent data
            if (this.data.stockData && this.data.stockData.length > 0) {
                const recentPrices = this.data.stockData.slice(-20).map(d => d.close);
                const currentPrice = recentPrices[recentPrices.length - 1];
                const volatility = this.calculateVolatility(recentPrices);

                const support = Math.min(...recentPrices);
                const resistance = Math.max(...recentPrices);
                const trend = recentPrices[recentPrices.length - 1] > recentPrices[0] ? 'Bullish' : 'Bearish';

                technicalAnalysis = [
                    `📈 Trend: ${trend}`,
                    `📊 Support: $${support.toFixed(2)}`,
                    `📊 Resistance: $${resistance.toFixed(2)}`,
                    volatility > 0.3 ? '⚡ High Volatility' :
                    volatility > 0.15 ? '⚡ Medium Volatility' : '⚡ Low Volatility'
                ];
            } else {
                technicalAnalysis = [
                    '📈 Trend: Loading...',
                    '📊 Support: Loading...',
                    '📊 Resistance: Loading...',
                    '⚡ Volatility: Loading...'
                ];
            }
        } else {
            modelInsights = [
                '🎯 Best Model: No data available',
                '📊 Performance: N/A',
                '🔍 Validation: Pending',
                '⚠️ Status: Loading'
            ];
            technicalAnalysis = [
                '📈 Trend: No data',
                '📊 Support: N/A',
                '📊 Resistance: N/A',
                '⚡ Volatility: N/A'
            ];
        }

        return `
            <div class="row">
                <div class="col-md-6">
                    <h6>Technical Analysis</h6>
                    <ul class="list-unstyled">
                        ${technicalAnalysis.map(item => `<li>${item}</li>`).join('')}
                    </ul>
                </div>
                <div class="col-md-6">
                    <h6>Model Insights</h6>
                    <ul class="list-unstyled">
                        ${modelInsights.map(item => `<li>${item}</li>`).join('')}
                    </ul>
                </div>
            </div>
            <div class="row mt-3">
                <div class="col-12">
                    <div class="alert alert-info">
                        <h6><i class="fas fa-info-circle"></i> Ultra-Conservative Model Analysis</h6>
                        <p class="mb-0">
                            Our models use strict data leakage prevention with basic lag features only.
                            Direction accuracy of 55-65% is realistic for financial markets.
                            Higher accuracy may indicate data leakage.
                        </p>
                    </div>
                </div>
            </div>
        `;
    }

    generateSystemHealthHTML(status) {
        const statusBadge = status.status === 'healthy' ? 
            '<span class="badge bg-success">Healthy</span>' : 
            '<span class="badge bg-warning">Issues</span>';

        return `
            <div class="mb-3">
                <h6>Overall Status: ${statusBadge}</h6>
            </div>
            <div class="row">
                ${Object.entries(status.components || {}).map(([component, health]) => `
                    <div class="col-6 mb-2">
                        <div class="d-flex justify-content-between">
                            <span>${component}:</span>
                            <span class="badge ${health === 'healthy' ? 'bg-success' : 'bg-warning'}">${health}</span>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    }

    // UI state management
    showLoading() {
        const loading = document.getElementById('loading-indicator');
        const content = document.getElementById('dashboard-content');
        const error = document.getElementById('error-alert');
        
        if (loading) loading.classList.remove('d-none');
        if (content) content.classList.add('d-none');
        if (error) error.classList.add('d-none');
    }

    showDashboard() {
        const loading = document.getElementById('loading-indicator');
        const content = document.getElementById('dashboard-content');
        const error = document.getElementById('error-alert');
        
        if (loading) loading.classList.add('d-none');
        if (content) content.classList.remove('d-none');
        if (error) error.classList.add('d-none');
    }

    showError(message) {
        const loading = document.getElementById('loading-indicator');
        const content = document.getElementById('dashboard-content');
        const error = document.getElementById('error-alert');
        const errorMessage = document.getElementById('error-message');
        
        if (loading) loading.classList.add('d-none');
        if (content) content.classList.add('d-none');
        if (error) error.classList.remove('d-none');
        if (errorMessage) errorMessage.textContent = message;

        this.errorCount++;
    }

    async retryLoad() {
        if (this.errorCount < this.maxRetries) {
            logger.info(`Retrying dashboard load (attempt ${this.errorCount + 1}/${this.maxRetries})`);
            await this.initialize();
        } else {
            logger.error('Maximum retry attempts reached');
        }
    }

    // Best model selection logic
    getBestModel() {
        if (!this.data?.modelPerformance) return null;

        const models = this.data.modelPerformance;

        // Find model with lowest MAE (best performance for ultra conservative models)
        let bestModel = null;
        let bestScore = Infinity;

        Object.entries(models).forEach(([name, model]) => {
            // Try MAE first (for ultra conservative models), then MAPE (for legacy models)
            const mae = model.mae_mean;
            const mape = model.final_test?.mape || model.mape || model.cv_mape_mean;

            let score;
            let scoreType;

            if (mae !== undefined) {
                score = mae;
                scoreType = 'mae';
            } else if (mape !== undefined && mape !== Infinity) {
                score = mape;
                scoreType = 'mape';
            } else {
                return; // Skip this model
            }

            if (score < bestScore) {
                bestScore = score;
                bestModel = {
                    name,
                    ...model,
                    score,
                    scoreType,
                    mae: mae,
                    mape: mape
                };
            }
        });

        return bestModel;
    }

    updateBestModelDisplay() {
        const bestModel = this.getBestModel();

        if (!bestModel) {
            logger.warn('No best model data available');
            return;
        }

        // Update header badge
        const badgeEl = document.getElementById('best-model-badge');
        if (badgeEl) {
            const displayName = bestModel.name
                .replace('_ultra_conservative', ' Ultra Conservative')
                .replace('_leak_free', ' Leak-Free')
                .replace('_', ' ');

            let scoreText;
            if (bestModel.scoreType === 'mae') {
                scoreText = `${(bestModel.score * 100).toFixed(3)}% MAE`;
            } else {
                scoreText = `${bestModel.score.toFixed(2)}% MAPE`;
            }

            badgeEl.textContent = `Best Model: ${displayName} (${scoreText})`;
        }

        // Update metric card
        const nameEl = document.getElementById('best-model-name');
        const mapeEl = document.getElementById('best-model-mape');

        if (nameEl && mapeEl) {
            const displayName = bestModel.name
                .replace('_ultra_conservative', ' Ultra Conservative')
                .replace('_leak_free', ' Leak-Free')
                .replace('_', ' ');
            nameEl.textContent = displayName;

            if (bestModel.scoreType === 'mae') {
                mapeEl.textContent = `MAE: ${(bestModel.score * 100).toFixed(3)}%`;
            } else {
                mapeEl.textContent = `MAPE: ${bestModel.score.toFixed(2)}%`;
            }
        }

        // Update price chart subtitle
        const priceChartSubtitle = document.getElementById('price-chart-subtitle');
        if (priceChartSubtitle) {
            const displayName = bestModel.name
                .replace('_ultra_conservative', ' Ultra Conservative')
                .replace('_leak_free', ' Leak-Free')
                .replace('_', ' ')
                .replace(/\b\w/g, l => l.toUpperCase());

            let scoreText;
            if (bestModel.scoreType === 'mae') {
                scoreText = `${(bestModel.score * 100).toFixed(3)}% MAE`;
            } else {
                scoreText = `${bestModel.score.toFixed(2)}% MAPE`;
            }

            priceChartSubtitle.textContent = `${displayName} (${scoreText})`;
        }

        // Update performance chart subtitle
        const performanceChartSubtitle = document.getElementById('performance-chart-subtitle');
        if (performanceChartSubtitle) {
            if (bestModel.scoreType === 'mae') {
                performanceChartSubtitle.textContent = 'Ultra-Conservative Model Comparison (MAE Basis)';
            } else {
                performanceChartSubtitle.textContent = 'Model Performance Comparison (MAPE Basis)';
            }
        }

        // Update technical details
        const techBestModelEl = document.getElementById('tech-best-model');
        const techAlgorithmEl = document.getElementById('tech-algorithm');

        if (techBestModelEl) {
            const cleanName = bestModel.name
                .replace('_ultra_conservative', '')
                .replace('_leak_free', '')
                .replace('_', ' ')
                .replace(/\b\w/g, l => l.toUpperCase());

            if (bestModel.scoreType === 'mae') {
                techBestModelEl.textContent = `Best Model: ${cleanName} Ultra Conservative`;
                techBestModelEl.className = 'mb-3 text-success';
            } else {
                techBestModelEl.textContent = `Best Model: ${cleanName}`;
                techBestModelEl.className = 'mb-3 text-success';
            }
        }

        if (techAlgorithmEl) {
            const modelType = bestModel.name.toLowerCase();
            if (modelType.includes('linear')) {
                techAlgorithmEl.textContent = 'LinearRegression (Ordinary Least Squares)';
            } else if (modelType.includes('ridge')) {
                techAlgorithmEl.textContent = 'Ridge Regression (L2 Regularization)';
            } else if (modelType.includes('lasso')) {
                techAlgorithmEl.textContent = 'Lasso Regression (L1 Regularization)';
            } else if (modelType.includes('random_forest')) {
                techAlgorithmEl.textContent = 'RandomForestRegressor (Ensemble Learning)';
            } else if (modelType.includes('xgboost')) {
                techAlgorithmEl.textContent = 'XGBRegressor (Extreme Gradient Boosting)';
            } else {
                techAlgorithmEl.textContent = `${bestModel.name} Algorithm`;
            }
        }

        logger.info('Best model display updated', {
            model: bestModel.name,
            score: bestModel.score,
            scoreType: bestModel.scoreType
        });

        // Update performance table
        this.updatePerformanceTable();

        // Update feature importance
        this.updateFeatureImportance();
    }

    updatePerformanceTable() {
        const tableBody = document.getElementById('performance-table-body');
        if (!tableBody || !this.data?.modelPerformance) return;

        const models = this.data.modelPerformance;

        // Sort models by best available metric (MAE for ultra-conservative, MAPE for others)
        const sortedModels = Object.entries(models).map(([name, model]) => {
            // Handle ultra-conservative models (MAE-based)
            if (model.mae_mean !== undefined) {
                return {
                    name,
                    mae: model.mae_mean,
                    r2: model.r2_mean || 0,
                    direction_accuracy: model.direction_mean || 0,
                    metric_type: 'mae',
                    primary_score: model.mae_mean,
                    validation_method: model.validation_method || 'N/A',
                    features_count: model.features_count || 0
                };
            }
            // Handle legacy models (MAPE-based)
            else {
                const mape = model.final_test?.mape || model.mape || model.cv_mape_mean || Infinity;
                return {
                    name,
                    mape: mape,
                    r2: model.final_test?.r2 || model.r2 || model.cv_r2_mean || 0,
                    direction_accuracy: model.final_test?.direction_accuracy || model.cv_direction_mean || 0,
                    metric_type: 'mape',
                    primary_score: mape,
                    validation_method: model.validation_method || 'N/A',
                    features_count: model.selected_features_count || 0
                };
            }
        }).sort((a, b) => a.primary_score - b.primary_score);

        const html = sortedModels.map((model, index) => {
            const displayName = model.name
                .replace('_ultra_conservative', ' Ultra Conservative')
                .replace('_truly_leak_free', ' Truly Leak-Free')
                .replace('_leak_free', ' Leak-Free')
                .replace('_', ' ')
                .replace(/\b\w/g, l => l.toUpperCase());

            const rankBadge = index === 0 ?
                '<span class="badge bg-success">🏆 Best</span>' :
                `<span class="badge bg-secondary">#${index + 1}</span>`;

            // Format primary metric display
            let metricDisplay;
            if (model.metric_type === 'mae') {
                metricDisplay = `${(model.mae * 100).toFixed(3)}% MAE`;
            } else {
                metricDisplay = `${model.mape.toFixed(2)}% MAPE`;
            }

            // Format direction accuracy with realistic expectations
            let directionDisplay = `${model.direction_accuracy.toFixed(1)}%`;
            let directionClass = '';

            if (model.metric_type === 'mae') {
                // Ultra-conservative models: 50-60% is realistic
                if (model.direction_accuracy >= 55 && model.direction_accuracy <= 65) {
                    directionClass = 'text-success';
                } else if (model.direction_accuracy > 65) {
                    directionClass = 'text-warning';
                    directionDisplay += ' ⚠️';
                } else {
                    directionClass = 'text-muted';
                }
            } else {
                // Legacy models: higher accuracy may indicate leakage
                if (model.direction_accuracy > 95) {
                    directionClass = 'text-danger';
                    directionDisplay += ' 🚨';
                } else if (model.direction_accuracy > 80) {
                    directionClass = 'text-warning';
                    directionDisplay += ' ⚠️';
                } else {
                    directionClass = 'text-success';
                }
            }

            return `
                <tr>
                    <td>
                        <strong>${displayName}</strong><br>
                        <small class="text-muted">${model.validation_method}</small>
                    </td>
                    <td>
                        <span class="fw-bold">${metricDisplay}</span><br>
                        <small class="text-muted">R²: ${model.r2.toFixed(3)}</small>
                    </td>
                    <td>
                        <span class="${directionClass} fw-bold">${directionDisplay}</span><br>
                        <small class="text-muted">${model.features_count} features</small>
                    </td>
                    <td>${rankBadge}</td>
                </tr>
            `;
        }).join('');

        tableBody.innerHTML = html;

        logger.info('Performance table updated', {
            models_count: sortedModels.length,
            best_model: sortedModels[0]?.name,
            best_score: sortedModels[0]?.primary_score
        });
    }

    getOrdinalSuffix(num) {
        const j = num % 10;
        const k = num % 100;
        if (j == 1 && k != 11) return "st";
        if (j == 2 && k != 12) return "nd";
        if (j == 3 && k != 13) return "rd";
        return "th";
    }

    updateFeatureImportance() {
        const tableBody = document.getElementById('feature-importance-table');
        const descriptionEl = document.getElementById('feature-importance-description');

        if (!tableBody || !this.data?.featureAnalysis) {
            if (tableBody) {
                tableBody.innerHTML = '<tr><td colspan="3" class="text-center">No feature data available</td></tr>';
            }
            if (descriptionEl) {
                descriptionEl.textContent = 'Feature importance data not available for ultra-conservative models.';
            }
            return;
        }

        const features = this.data.featureAnalysis;
        const bestModel = this.getBestModel();

        // Sort features by importance (descending)
        const sortedFeatures = features.sort((a, b) => (b.importance || 0) - (a.importance || 0));

        const html = sortedFeatures.slice(0, 5).map((feature, index) => {
            const importance = ((feature.importance || 0) * 100).toFixed(1);
            return `
                <tr>
                    <td>${index + 1}</td>
                    <td>${feature.name || 'Unknown Feature'}</td>
                    <td>${importance}%</td>
                </tr>
            `;
        }).join('');

        tableBody.innerHTML = html;

        // Update description
        if (descriptionEl && bestModel) {
            const modelName = bestModel.name
                .replace('_ultra_conservative', ' Ultra Conservative')
                .replace('_leak_free', ' Leak-Free')
                .replace('_', ' ')
                .replace(/\b\w/g, l => l.toUpperCase());

            if (bestModel.scoreType === 'mae') {
                descriptionEl.textContent = `Basic feature analysis for ${modelName} with leak-free validation.`;
            } else {
                descriptionEl.textContent = `Feature importance based on SHAP values from ${modelName}.`;
            }
        }

        logger.info('Feature importance table updated', {
            features_count: sortedFeatures.length,
            top_feature: sortedFeatures[0]?.name
        });
    }

    // Public API
    refresh() {
        logger.info('Manual refresh requested');
        return this.loadInitialData().then(() => this.handleDataRefresh());
    }

    destroy() {
        dataLoader.disableAutoRefresh();
        chartManager.destroyAllCharts();
        logger.info('Dashboard destroyed');
    }
}

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.dashboardManager = new DashboardManager();
});

// Export for global access
window.DashboardManager = DashboardManager;