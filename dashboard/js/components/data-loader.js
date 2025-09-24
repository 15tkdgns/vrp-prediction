/**
 * Data Loader
 * Handles loading and processing of dashboard data
 */

import { api } from '../utils/api.js';
import { createLogger } from '../utils/logger.js';

const dataLogger = createLogger('DataLoader');

export class DataLoader {
    constructor() {
        this.cache = new Map();
        this.loadingStates = new Map();
        this.refreshInterval = 5 * 60 * 1000; // 5 minutes
        this.autoRefreshEnabled = false;
        
        dataLogger.info('Data Loader initialized');
    }

    async loadDashboardData() {
        const timer = dataLogger.startTimer('load_dashboard_data');
        
        try {
            dataLogger.info('Loading dashboard data...');

            // Load data in parallel
            const [
                stockData,
                predictions,
                modelPerformance,
                featureAnalysis,
                modelComparison
            ] = await Promise.allSettled([
                this.loadStockData(),
                this.loadPredictions(),
                this.loadModelPerformance(),
                this.loadFeatureAnalysis(),
                this.loadModelComparison()
            ]);

            const dashboardData = {
                stockData: this.processSettledResult(stockData, 'Stock Data'),
                predictions: this.processSettledResult(predictions, 'Predictions'),
                modelPerformance: this.processSettledResult(modelPerformance, 'Model Performance'),
                featureAnalysis: this.processSettledResult(featureAnalysis, 'Feature Analysis'),
                modelComparison: this.processSettledResult(modelComparison, 'Model Comparison'),
                loadTime: Date.now()
            };

            // Cache the data
            this.cache.set('dashboardData', dashboardData);

            dataLogger.info('Dashboard data loaded successfully');
            dataLogger.endTimer(timer);
            
            return dashboardData;

        } catch (error) {
            dataLogger.error('Failed to load dashboard data', error);
            dataLogger.endTimer(timer);
            throw error;
        }
    }

    async loadStockData() {
        if (this.loadingStates.get('stockData')) {
            dataLogger.debug('Stock data already loading, waiting...');
            return this.waitForLoad('stockData');
        }

        this.loadingStates.set('stockData', true);

        try {
            const data = await api.getStockData();
            
            if (!data || !Array.isArray(data.prices)) {
                throw new Error('Invalid stock data format');
            }

            const processedData = this.processStockData(data);
            this.cache.set('stockData', processedData);
            this.loadingStates.set('stockData', false);
            
            dataLogger.info(`Loaded ${processedData.length} stock data points`);
            return processedData;

        } catch (error) {
            this.loadingStates.set('stockData', false);
            dataLogger.warn('Failed to load stock data, using mock data', error.message);
            return this.generateMockStockData();
        }
    }

    async loadPredictions() {
        if (this.loadingStates.get('predictions')) {
            return this.waitForLoad('predictions');
        }

        this.loadingStates.set('predictions', true);

        try {
            const data = await api.getPredictions();
            const processedData = this.processPredictions(data);
            this.cache.set('predictions', processedData);
            this.loadingStates.set('predictions', false);
            
            dataLogger.info(`Loaded ${processedData.length} predictions`);
            return processedData;

        } catch (error) {
            this.loadingStates.set('predictions', false);
            dataLogger.warn('Failed to load predictions, using mock data', error.message);
            return this.generateMockPredictions();
        }
    }

    async loadModelPerformance() {
        if (this.loadingStates.get('modelPerformance')) {
            return this.waitForLoad('modelPerformance');
        }

        this.loadingStates.set('modelPerformance', true);

        try {
            const data = await api.getModelPerformance();
            const processedData = this.processModelPerformance(data);
            this.cache.set('modelPerformance', processedData);
            this.loadingStates.set('modelPerformance', false);
            
            dataLogger.info('Loaded model performance data');
            return processedData;

        } catch (error) {
            this.loadingStates.set('modelPerformance', false);
            dataLogger.warn('Failed to load model performance, using mock data', error.message);
            return this.generateMockModelPerformance();
        }
    }

    async loadFeatureAnalysis() {
        if (this.loadingStates.get('featureAnalysis')) {
            return this.waitForLoad('featureAnalysis');
        }

        this.loadingStates.set('featureAnalysis', true);

        try {
            const data = await api.getFeatureAnalysis();
            const processedData = this.processFeatureAnalysis(data);
            this.cache.set('featureAnalysis', processedData);
            this.loadingStates.set('featureAnalysis', false);
            
            dataLogger.info('Loaded feature analysis data');
            return processedData;

        } catch (error) {
            this.loadingStates.set('featureAnalysis', false);
            dataLogger.warn('Failed to load feature analysis, using mock data', error.message);
            return this.generateMockFeatureAnalysis();
        }
    }

    async loadModelComparison() {
        if (this.loadingStates.get('modelComparison')) {
            return this.waitForLoad('modelComparison');
        }

        this.loadingStates.set('modelComparison', true);

        try {
            const data = await api.getModelComparison();
            const processedData = this.processModelComparison(data);
            this.cache.set('modelComparison', processedData);
            this.loadingStates.set('modelComparison', false);
            
            dataLogger.info('Loaded model comparison data');
            return processedData;

        } catch (error) {
            this.loadingStates.set('modelComparison', false);
            dataLogger.warn('Failed to load model comparison, using mock data', error.message);
            return this.generateMockModelComparison();
        }
    }

    // Data processing methods
    processStockData(rawData) {
        if (!rawData.prices) return [];
        
        return rawData.prices.map(item => ({
            date: new Date(item.Date || item.date),
            open: parseFloat(item.Open || item.open),
            high: parseFloat(item.High || item.high),
            low: parseFloat(item.Low || item.low),
            close: parseFloat(item.Close || item.close),
            volume: parseInt(item.Volume || item.volume)
        })).sort((a, b) => a.date - b.date);
    }

    processPredictions(rawData) {
        if (!rawData.predictions) return [];
        
        return rawData.predictions.map(item => ({
            date: new Date(item.date),
            predicted_price: parseFloat(item.predicted_price),
            confidence: parseFloat(item.confidence || 0.8),
            model: item.model || 'Unknown'
        })).sort((a, b) => a.date - b.date);
    }

    processModelPerformance(rawData) {
        const models = rawData.models || rawData;

        // Handle object format (ultra-conservative models)
        if (typeof models === 'object' && !Array.isArray(models)) {
            dataLogger.info('Processing object-format model performance data');
            return models; // Return as-is since dashboard-manager expects object format
        }

        // Handle array format (legacy models)
        if (Array.isArray(models)) {
            dataLogger.info('Processing array-format model performance data');
            return models.map(model => ({
                name: model.name,
                accuracy: parseFloat(model.accuracy || model.test_accuracy || 0),
                f1_score: parseFloat(model.f1_score || model.test_f1 || 0),
                precision: parseFloat(model.precision || 0),
                recall: parseFloat(model.recall || 0),
                training_time: parseFloat(model.training_time || 0)
            }));
        }

        dataLogger.warn('Unknown model performance data format');
        return {};
    }

    processFeatureAnalysis(rawData) {
        const features = rawData.features || rawData.feature_importance || [];
        if (!Array.isArray(features)) return [];

        return features.map(feature => ({
            name: feature.name || feature.feature,
            importance: parseFloat(feature.importance || feature.value || 0),
            correlation: parseFloat(feature.correlation || 0),
            pvalue: parseFloat(feature.pvalue || 1)
        })).sort((a, b) => b.importance - a.importance);
    }

    processModelComparison(rawData) {
        return this.processModelPerformance(rawData);
    }

    // Utility methods
    processSettledResult(settledResult, dataType) {
        if (settledResult.status === 'fulfilled') {
            return settledResult.value;
        } else {
            dataLogger.warn(`Failed to load ${dataType}:`, settledResult.reason?.message);
            return null;
        }
    }

    async waitForLoad(key) {
        while (this.loadingStates.get(key)) {
            await new Promise(resolve => setTimeout(resolve, 100));
        }
        return this.cache.get(key);
    }

    // Mock data generators
    generateMockStockData() {
        const data = [];
        const startDate = new Date('2024-01-01');
        let price = 450;

        for (let i = 0; i < 100; i++) {
            const date = new Date(startDate);
            date.setDate(date.getDate() + i);
            
            const change = (Math.random() - 0.5) * 10;
            price += change;
            
            data.push({
                date,
                open: price - (Math.random() * 5),
                high: price + (Math.random() * 8),
                low: price - (Math.random() * 8),
                close: price,
                volume: Math.floor(Math.random() * 1000000) + 500000
            });
        }

        dataLogger.info('Generated mock stock data');
        return data;
    }

    generateMockPredictions() {
        const predictions = [];
        const startDate = new Date();
        let price = 460;

        for (let i = 1; i <= 30; i++) {
            const date = new Date(startDate);
            date.setDate(date.getDate() + i);
            
            price += (Math.random() - 0.5) * 5;
            
            predictions.push({
                date,
                predicted_price: price,
                confidence: 0.7 + Math.random() * 0.3,
                model: 'RandomForest'
            });
        }

        dataLogger.info('Generated mock predictions');
        return predictions;
    }

    generateMockModelPerformance() {
        const models = [
            { name: 'RandomForest', accuracy: 0.87, f1_score: 0.84, precision: 0.86, recall: 0.82 },
            { name: 'XGBoost', accuracy: 0.89, f1_score: 0.87, precision: 0.88, recall: 0.86 },
            { name: 'GradientBoosting', accuracy: 0.85, f1_score: 0.83, precision: 0.84, recall: 0.82 },
            { name: 'NeuralNetwork', accuracy: 0.88, f1_score: 0.86, precision: 0.87, recall: 0.85 }
        ];

        dataLogger.info('Generated mock model performance');
        return models;
    }

    generateMockFeatureAnalysis() {
        const features = [
            { name: 'RSI_14', importance: 0.15, correlation: -0.23 },
            { name: 'SMA_20', importance: 0.12, correlation: 0.78 },
            { name: 'Volume_Ratio', importance: 0.11, correlation: 0.34 },
            { name: 'Price_Change_5d', importance: 0.10, correlation: -0.12 },
            { name: 'Volatility_20d', importance: 0.09, correlation: -0.45 },
            { name: 'MACD', importance: 0.08, correlation: 0.56 },
            { name: 'Bollinger_Upper', importance: 0.07, correlation: 0.67 },
            { name: 'ATR_14', importance: 0.06, correlation: -0.34 },
            { name: 'Stochastic_K', importance: 0.05, correlation: -0.12 },
            { name: 'Williams_R', importance: 0.04, correlation: 0.23 }
        ];

        dataLogger.info('Generated mock feature analysis');
        return features;
    }

    generateMockModelComparison() {
        return this.generateMockModelPerformance();
    }

    // Auto-refresh functionality
    enableAutoRefresh() {
        if (this.autoRefreshEnabled) return;
        
        this.autoRefreshEnabled = true;
        this.refreshTimer = setInterval(async () => {
            try {
                dataLogger.info('Auto-refreshing dashboard data...');
                await this.loadDashboardData();
                
                // Emit refresh event
                window.dispatchEvent(new CustomEvent('dashboardDataRefresh', {
                    detail: { timestamp: Date.now() }
                }));
            } catch (error) {
                dataLogger.error('Auto-refresh failed', error);
            }
        }, this.refreshInterval);

        dataLogger.info('Auto-refresh enabled');
    }

    disableAutoRefresh() {
        if (!this.autoRefreshEnabled) return;
        
        this.autoRefreshEnabled = false;
        if (this.refreshTimer) {
            clearInterval(this.refreshTimer);
            this.refreshTimer = null;
        }

        dataLogger.info('Auto-refresh disabled');
    }

    getCachedData(key) {
        return this.cache.get(key);
    }

    clearCache() {
        this.cache.clear();
        dataLogger.info('Data cache cleared');
    }
}

// Create default data loader instance
export const dataLoader = new DataLoader();