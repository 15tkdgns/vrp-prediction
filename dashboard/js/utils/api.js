/**
 * API Utility Functions
 * Handles all API communications with proper error handling and retries
 */

import { logger } from './logger.js';

export class APIClient {
    constructor(baseURL = '') {
        this.baseURL = baseURL;
        this.retryAttempts = 3;
        this.retryDelay = 1000; // 1 second
        this.timeout = 30000; // 30 seconds
        this.cache = new Map();
        this.cacheTimeout = 5 * 60 * 1000; // 5 minutes
    }

    async request(endpoint, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        const requestOptions = {
            timeout: this.timeout,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        };

        // Check cache for GET requests
        if (!options.method || options.method === 'GET') {
            const cachedData = this.getFromCache(url);
            if (cachedData) {
                logger.debug(`Cache hit for ${endpoint}`);
                return cachedData;
            }
        }

        let lastError;
        
        for (let attempt = 1; attempt <= this.retryAttempts; attempt++) {
            try {
                logger.debug(`API request attempt ${attempt}: ${endpoint}`);
                
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), this.timeout);
                
                const response = await fetch(url, {
                    ...requestOptions,
                    signal: controller.signal
                });

                clearTimeout(timeoutId);

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                const data = await response.json();
                
                // Cache successful GET requests
                if (!options.method || options.method === 'GET') {
                    this.addToCache(url, data);
                }

                logger.info(`API request successful: ${endpoint}`);
                return data;

            } catch (error) {
                lastError = error;
                logger.warn(`API request failed (attempt ${attempt}/${this.retryAttempts}): ${endpoint}`, error.message);
                
                // Don't retry on certain errors
                if (error.name === 'AbortError' || 
                    (error.message && error.message.includes('404'))) {
                    break;
                }
                
                // Wait before retrying
                if (attempt < this.retryAttempts) {
                    await this.sleep(this.retryDelay * attempt);
                }
            }
        }

        logger.error(`API request failed after ${this.retryAttempts} attempts: ${endpoint}`, lastError);
        throw lastError;
    }

    async get(endpoint, params = {}) {
        const queryString = new URLSearchParams(params).toString();
        const url = queryString ? `${endpoint}?${queryString}` : endpoint;
        return this.request(url, { method: 'GET' });
    }

    async post(endpoint, data = {}) {
        return this.request(endpoint, {
            method: 'POST',
            body: JSON.stringify(data)
        });
    }

    async put(endpoint, data = {}) {
        return this.request(endpoint, {
            method: 'PUT',
            body: JSON.stringify(data)
        });
    }

    async delete(endpoint) {
        return this.request(endpoint, { method: 'DELETE' });
    }

    // Cache management
    addToCache(key, data) {
        this.cache.set(key, {
            data,
            timestamp: Date.now()
        });
    }

    getFromCache(key) {
        const cached = this.cache.get(key);
        if (!cached) return null;
        
        const isExpired = Date.now() - cached.timestamp > this.cacheTimeout;
        if (isExpired) {
            this.cache.delete(key);
            return null;
        }
        
        return cached.data;
    }

    clearCache() {
        this.cache.clear();
        logger.info('API cache cleared');
    }

    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Dashboard-specific API endpoints
export class DashboardAPI extends APIClient {
    constructor() {
        super('');
        this.dataPath = 'data/raw';
    }

    // Get stock data
    async getStockData(symbol = 'SPY', period = '1y') {
        return this.get(`${this.dataPath}/spy_2025_h1.json`, { 
            t: Date.now() // Cache busting
        });
    }

    // Get predictions
    async getPredictions() {
        return this.get(`${this.dataPath}/spy_2025_h1_predictions.json`, {
            t: Date.now()
        });
    }

    // Get model performance
    async getModelPerformance() {
        return this.get(`${this.dataPath}/model_performance.json`, {
            t: Date.now()
        });
    }

    // Get feature analysis
    async getFeatureAnalysis() {
        return this.get(`${this.dataPath}/feature_analysis_enhanced.json`, {
            t: Date.now()
        });
    }

    // Get model comparison
    async getModelComparison() {
        return this.get(`${this.dataPath}/model_comparison_results_enhanced.json`, {
            t: Date.now()
        });
    }

    // Get system status (from API if available)
    async getSystemStatus() {
        try {
            return await this.get('/api/system/status');
        } catch (error) {
            // Fallback to local status
            return {
                status: 'unknown',
                components: {
                    api: 'offline',
                    dashboard: 'online',
                    models: 'unknown'
                },
                timestamp: Date.now()
            };
        }
    }

    // Health check
    async healthCheck() {
        try {
            return await this.get('/api/health');
        } catch (error) {
            return {
                status: 'offline',
                error: error.message,
                timestamp: Date.now()
            };
        }
    }

    // Get real-time predictions (if API is available)
    async getRealtimePredictions(symbol = 'SPY') {
        try {
            return await this.post('/api/predict', {
                symbol,
                period: '1y',
                models: ['RandomForest', 'XGBoost', 'GradientBoosting']
            });
        } catch (error) {
            logger.warn('Real-time API not available, using cached data');
            return this.getPredictions();
        }
    }
}

// Create default API client instance
export const api = new DashboardAPI();

// Utility functions for common operations
export async function fetchWithFallback(primaryFetch, fallbackFetch) {
    try {
        return await primaryFetch();
    } catch (error) {
        logger.warn('Primary fetch failed, trying fallback', error.message);
        return await fallbackFetch();
    }
}

export function addCacheBuster(url) {
    const separator = url.includes('?') ? '&' : '?';
    return `${url}${separator}t=${Date.now()}`;
}