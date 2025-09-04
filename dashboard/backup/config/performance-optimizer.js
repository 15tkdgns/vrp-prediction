/**
 * Dashboard Performance Optimizer
 * 대시보드 로딩 성능 최적화 및 중복 요청 방지
 */

class PerformanceOptimizer {
    constructor() {
        this.requestCache = new Map();
        this.loadingStates = new Map();
        this.cacheTimeout = 30000; // 30초 캐시
        this.requestQueue = new Map();
        
        console.log('🚀 Performance Optimizer 초기화');
    }

    /**
     * 중복 요청 방지 및 캐싱된 fetch
     */
    async optimizedFetch(url, options = {}) {
        const cacheKey = url + JSON.stringify(options);
        
        // 캐시된 응답이 있는지 확인
        if (this.requestCache.has(cacheKey)) {
            const cached = this.requestCache.get(cacheKey);
            const age = Date.now() - cached.timestamp;
            
            if (age < this.cacheTimeout) {
                console.log(`📦 Cache hit: ${url}`);
                return Promise.resolve(cached.response);
            } else {
                this.requestCache.delete(cacheKey);
            }
        }
        
        // 동일한 요청이 진행 중인지 확인
        if (this.loadingStates.has(cacheKey)) {
            console.log(`⏳ Request in progress, waiting: ${url}`);
            return this.loadingStates.get(cacheKey);
        }
        
        // 새로운 요청 시작
        console.log(`🌐 New request: ${url}`);
        const requestPromise = this.performRequest(url, options, cacheKey);
        this.loadingStates.set(cacheKey, requestPromise);
        
        return requestPromise;
    }
    
    async performRequest(url, options, cacheKey) {
        try {
            const response = await fetch(url, options);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            // 캐시에 저장
            this.requestCache.set(cacheKey, {
                response: data,
                timestamp: Date.now()
            });
            
            // 진행 중 상태 제거
            this.loadingStates.delete(cacheKey);
            
            return data;
            
        } catch (error) {
            this.loadingStates.delete(cacheKey);
            console.error(`❌ Request failed: ${url}`, error);
            throw error;
        }
    }

    /**
     * 배치 요청 최적화
     */
    async batchRequests(requests) {
        console.log(`🔄 Batch processing ${requests.length} requests`);
        
        const results = await Promise.allSettled(
            requests.map(req => this.optimizedFetch(req.url, req.options))
        );
        
        const successful = results.filter(r => r.status === 'fulfilled').length;
        console.log(`✅ Batch complete: ${successful}/${requests.length} successful`);
        
        return results;
    }

    /**
     * 불필요한 요청 디바운싱
     */
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    /**
     * 캐시 정리
     */
    clearCache() {
        const size = this.requestCache.size;
        this.requestCache.clear();
        this.loadingStates.clear();
        console.log(`🧹 Cache cleared: ${size} items removed`);
    }

    /**
     * 성능 통계 리포트
     */
    getPerformanceStats() {
        return {
            cacheSize: this.requestCache.size,
            activeRequests: this.loadingStates.size,
            cacheHitRate: this.calculateCacheHitRate(),
            timestamp: new Date().toISOString()
        };
    }

    calculateCacheHitRate() {
        // 간단한 캐시 히트율 계산 로직
        return this.requestCache.size > 0 ? 
            Math.round((this.requestCache.size / (this.requestCache.size + this.loadingStates.size)) * 100) :
            0;
    }

    /**
     * 메모리 사용량 최적화
     */
    optimizeMemory() {
        const now = Date.now();
        let cleaned = 0;
        
        // 오래된 캐시 항목 제거
        for (const [key, value] of this.requestCache.entries()) {
            if (now - value.timestamp > this.cacheTimeout * 2) {
                this.requestCache.delete(key);
                cleaned++;
            }
        }
        
        if (cleaned > 0) {
            console.log(`🧹 Memory optimized: ${cleaned} old cache entries removed`);
        }
    }

    /**
     * 주기적인 최적화 실행
     */
    startPeriodicOptimization() {
        setInterval(() => {
            this.optimizeMemory();
        }, 60000); // 1분마다 실행

        console.log('⚡ Periodic optimization started');
    }
}

// 전역 인스턴스 생성
window.performanceOptimizer = new PerformanceOptimizer();

// 기존 fetch를 최적화된 버전으로 래핑
const originalFetch = window.fetch;
window.fetch = function(url, options) {
    // API 요청만 최적화 (정적 파일은 제외)
    if (url.includes('/api/') || url.includes('/data/')) {
        return window.performanceOptimizer.optimizedFetch(url, options);
    }
    return originalFetch(url, options);
};

// 주기적 최적화 시작
window.performanceOptimizer.startPeriodicOptimization();

console.log('⚡ Performance Optimizer가 활성화되었습니다');

// 개발자 도구용 헬퍼 함수들
window.debugPerformance = {
    stats: () => window.performanceOptimizer.getPerformanceStats(),
    clearCache: () => window.performanceOptimizer.clearCache(),
    optimize: () => window.performanceOptimizer.optimizeMemory()
};