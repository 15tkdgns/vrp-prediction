/**
 * XAI Visualization Component
 * 
 * Chain-of-Thought 추론 과정과 Attention Heatmap을 시각화하는 컴포넌트
 * LLM 기반 뉴스 분석의 해석가능성을 제공합니다.
 * 
 * @author XAI Analysis System
 * @version 1.0.0
 */

class XAIVisualization {
    constructor() {
        this.xaiData = null;
        this.currentSampleIndex = 0;
        this.isInitialized = false;
        
        // DOM Elements
        this.cotSampleSelect = null;
        this.cotReasoningSteps = null;
        this.cotSentimentScore = null;
        this.cotUncertaintyScore = null;
        this.cotMarketSentiment = null;
        this.attentionHeadline = null;
        this.attentionStats = null;
        this.attentionInsights = null;
        
        console.log('🧠 XAI Visualization Component initialized');
    }

    /**
     * 컴포넌트 초기화
     */
    async initialize() {
        try {
            console.log('🚀 Initializing XAI Visualization...');
            
            // DOM Elements 찾기
            this.findDOMElements();
            
            // XAI 데이터 로딩
            await this.loadXAIData();
            
            // 이벤트 리스너 설정
            this.setupEventListeners();
            
            // 초기 샘플 로딩
            if (this.xaiData && this.xaiData.xai_samples && this.xaiData.xai_samples.length > 0) {
                this.populateSampleSelector();
                this.updateCoTAnalysis(0);
            }
            
            this.isInitialized = true;
            console.log('✅ XAI Visualization initialized successfully');
            
        } catch (error) {
            console.error('❌ XAI Visualization initialization failed:', error);
            this.showError('Failed to initialize XAI Visualization');
        }
    }

    /**
     * DOM Elements 찾기
     */
    findDOMElements() {
        this.cotSampleSelect = document.getElementById('cotSampleSelect');
        this.cotReasoningSteps = document.getElementById('cotReasoningSteps');
        this.cotSentimentScore = document.getElementById('cotSentimentScore');
        this.cotUncertaintyScore = document.getElementById('cotUncertaintyScore');
        this.cotMarketSentiment = document.getElementById('cotMarketSentiment');
        this.attentionHeadline = document.getElementById('attentionHeadline');
        this.attentionStats = document.getElementById('attentionStats');
        this.attentionInsights = document.getElementById('attentionInsights');
        
        if (!this.cotSampleSelect || !this.cotReasoningSteps) {
            throw new Error('Required DOM elements not found');
        }
    }

    /**
     * XAI 데이터 로딩
     */
    async loadXAIData() {
        try {
            console.log('📊 Loading XAI data...');
            
            // 정적 XAI 데이터 파일 로딩 시도
            const response = await fetch('data/processed/xai_dashboard_data.json');
            
            if (!response.ok) {
                // 데이터가 없으면 더미 데이터 사용
                console.warn('⚠️ XAI data file not found, using sample data');
                this.xaiData = this.createSampleData();
                return;
            }
            
            this.xaiData = await response.json();
            console.log(`✅ Loaded ${this.xaiData.xai_samples?.length || 0} XAI samples`);
            
        } catch (error) {
            console.warn('⚠️ Failed to load XAI data, using sample data:', error);
            this.xaiData = this.createSampleData();
        }
    }

    /**
     * 샘플 데이터 생성 (테스트용)
     */
    createSampleData() {
        return {
            xai_samples: [
                {
                    id: "sample_1",
                    title: "Apple Inc. reports record quarterly earnings beating analysts expectations",
                    date: "2025-09-13",
                    sentiment_score: 0.75,
                    uncertainty_score: 0.2,
                    market_sentiment: "Bullish",
                    event_category: "Financials",
                    reasoning_steps: [
                        {
                            step: "STEP 1 - KEY TERMS IDENTIFICATION",
                            content: "Key terms identified: 'Apple Inc.', 'record quarterly earnings', 'beating analysts expectations'. These are strong positive financial indicators for a major tech company."
                        },
                        {
                            step: "STEP 2 - SENTIMENT ANALYSIS",
                            content: "Positive indicators: 'record', 'beating expectations' suggest strong performance. Negative indicators: None identified. Neutral elements: Standard earnings reporting language."
                        },
                        {
                            step: "STEP 3 - MARKET IMPACT ASSESSMENT",
                            content: "This is likely to positively affect S&P 500 as Apple is a major component. Uncertainty is low due to concrete financial results. Category: Financial earnings report."
                        }
                    ],
                    attention: {
                        tokens: ["Apple", "Inc", "reports", "record", "quarterly", "earnings", "beating", "analysts", "expectations"],
                        weights: [0.8, 0.3, 0.5, 0.95, 0.6, 0.9, 0.85, 0.4, 0.7]
                    },
                    model_used: "google/flan-t5-base"
                },
                {
                    id: "sample_2", 
                    title: "Federal Reserve raises interest rates by 0.25% amid inflation concerns",
                    date: "2025-09-13",
                    sentiment_score: -0.3,
                    uncertainty_score: 0.6,
                    market_sentiment: "Bearish",
                    event_category: "Regulation",
                    reasoning_steps: [
                        {
                            step: "STEP 1 - KEY TERMS IDENTIFICATION",
                            content: "Key terms: 'Federal Reserve', 'raises interest rates', '0.25%', 'inflation concerns'. Central bank monetary policy action."
                        },
                        {
                            step: "STEP 2 - SENTIMENT ANALYSIS",
                            content: "Negative indicators: Rate hikes generally negative for markets. Positive indicators: Proactive inflation management. Uncertain elements: Market reaction varies."
                        },
                        {
                            step: "STEP 3 - MARKET IMPACT ASSESSMENT",
                            content: "Rate increases typically pressure equity markets. High uncertainty as market reactions to Fed policy are mixed. Category: Regulatory/monetary policy."
                        }
                    ],
                    attention: {
                        tokens: ["Federal", "Reserve", "raises", "interest", "rates", "0.25%", "inflation", "concerns"],
                        weights: [0.9, 0.9, 0.8, 0.7, 0.7, 0.6, 0.85, 0.75]
                    },
                    model_used: "google/flan-t5-base"
                }
            ]
        };
    }

    /**
     * 샘플 선택기 채우기
     */
    populateSampleSelector() {
        if (!this.cotSampleSelect || !this.xaiData?.xai_samples) return;
        
        this.cotSampleSelect.innerHTML = '';
        
        this.xaiData.xai_samples.forEach((sample, index) => {
            const option = document.createElement('option');
            option.value = index;
            option.textContent = `${sample.date} - ${sample.title.substring(0, 60)}...`;
            this.cotSampleSelect.appendChild(option);
        });
    }

    /**
     * 이벤트 리스너 설정
     */
    setupEventListeners() {
        if (this.cotSampleSelect) {
            this.cotSampleSelect.addEventListener('change', (e) => {
                const selectedIndex = parseInt(e.target.value);
                this.updateCoTAnalysis(selectedIndex);
            });
        }
    }

    /**
     * Chain-of-Thought 분석 업데이트
     */
    updateCoTAnalysis(sampleIndex = 0) {
        if (!this.xaiData?.xai_samples || sampleIndex >= this.xaiData.xai_samples.length) {
            console.warn('⚠️ Invalid sample index or no data available');
            return;
        }
        
        const sample = this.xaiData.xai_samples[sampleIndex];
        this.currentSampleIndex = sampleIndex;
        
        // CoT 추론 단계 렌더링
        this.renderReasoningSteps(sample.reasoning_steps || []);
        
        // 요약 메트릭 업데이트
        this.updateSummaryMetrics(sample);
        
        // Attention 시각화 업데이트
        this.updateAttentionVisualization(sample);
        
        console.log(`📝 Updated CoT analysis for sample ${sampleIndex}`);
    }

    /**
     * 추론 단계 렌더링
     */
    renderReasoningSteps(steps) {
        if (!this.cotReasoningSteps) return;
        
        if (!steps || steps.length === 0) {
            this.cotReasoningSteps.innerHTML = `
                <div class="text-center text-muted py-4">
                    <i class="fas fa-exclamation-triangle fa-lg mb-2"></i>
                    <p>No reasoning steps available for this sample</p>
                </div>
            `;
            return;
        }
        
        const stepsHtml = steps.map((step, index) => {
            const stepNumber = index + 1;
            const stepClass = this.getStepClass(stepNumber);
            
            return `
                <div class="reasoning-step mb-3" style="animation: fadeIn 0.5s ease ${index * 0.1}s both;">
                    <div class="step-header p-3 ${stepClass}" style="border-radius: 8px 8px 0 0; border-left: 4px solid ${this.getStepColor(stepNumber)};">
                        <h6 class="mb-0" style="color: #fff;">
                            <i class="fas fa-${this.getStepIcon(stepNumber)} me-2"></i>
                            ${step.step}
                        </h6>
                    </div>
                    <div class="step-content p-3" style="background-color: #f8f9fa; border-radius: 0 0 8px 8px; border-left: 4px solid ${this.getStepColor(stepNumber)};">
                        <p class="mb-0 text-muted">${step.content}</p>
                    </div>
                </div>
            `;
        }).join('');
        
        this.cotReasoningSteps.innerHTML = stepsHtml;
    }

    /**
     * 단계별 클래스 반환
     */
    getStepClass(stepNumber) {
        const classes = ['bg-primary', 'bg-success', 'bg-warning', 'bg-info'];
        return classes[(stepNumber - 1) % classes.length] || 'bg-secondary';
    }

    /**
     * 단계별 색상 반환
     */
    getStepColor(stepNumber) {
        const colors = ['#007bff', '#28a745', '#ffc107', '#17a2b8'];
        return colors[(stepNumber - 1) % colors.length] || '#6c757d';
    }

    /**
     * 단계별 아이콘 반환
     */
    getStepIcon(stepNumber) {
        const icons = ['search', 'heart', 'chart-line', 'balance-scale'];
        return icons[(stepNumber - 1) % icons.length] || 'cog';
    }

    /**
     * 요약 메트릭 업데이트
     */
    updateSummaryMetrics(sample) {
        if (this.cotSentimentScore) {
            this.cotSentimentScore.textContent = sample.sentiment_score?.toFixed(2) || '--';
        }
        
        if (this.cotUncertaintyScore) {
            this.cotUncertaintyScore.textContent = sample.uncertainty_score?.toFixed(2) || '--';
        }
        
        if (this.cotMarketSentiment) {
            this.cotMarketSentiment.textContent = sample.market_sentiment || '--';
        }
    }

    /**
     * Attention 시각화 업데이트
     */
    updateAttentionVisualization(sample) {
        this.renderAttentionHeadline(sample);
        this.updateAttentionStats(sample);
        this.updateAttentionInsights(sample);
    }

    /**
     * Attention 헤드라인 렌더링
     */
    renderAttentionHeadline(sample) {
        if (!this.attentionHeadline || !sample.attention?.tokens || !sample.attention?.weights) {
            if (this.attentionHeadline) {
                this.attentionHeadline.innerHTML = '<em class="text-muted">No attention data available</em>';
            }
            return;
        }
        
        const tokens = sample.attention.tokens;
        const weights = sample.attention.weights;
        
        const highlightedTokens = tokens.map((token, index) => {
            const weight = weights[index] || 0;
            const intensity = Math.min(weight * 0.9 + 0.1, 1); // 0.1 ~ 1.0 범위
            const backgroundColor = `rgba(255, 193, 7, ${intensity})`;
            const textColor = intensity > 0.5 ? '#212529' : '#495057';
            
            return `<span style="background-color: ${backgroundColor}; color: ${textColor}; padding: 2px 4px; margin: 1px; border-radius: 3px; font-weight: ${intensity > 0.7 ? 'bold' : 'normal'};" title="Attention: ${weight.toFixed(3)}">${token}</span>`;
        }).join(' ');
        
        this.attentionHeadline.innerHTML = highlightedTokens;
    }

    /**
     * Attention 통계 업데이트
     */
    updateAttentionStats(sample) {
        if (!this.attentionStats || !sample.attention?.weights) {
            if (this.attentionStats) {
                this.attentionStats.textContent = '--';
            }
            return;
        }
        
        const weights = sample.attention.weights;
        const highAttentionCount = weights.filter(w => w > 0.7).length;
        const totalTokens = weights.length;
        
        this.attentionStats.textContent = `${highAttentionCount}/${totalTokens}`;
    }

    /**
     * Attention Insights 업데이트
     */
    updateAttentionInsights(sample) {
        if (!this.attentionInsights || !sample.attention?.tokens || !sample.attention?.weights) {
            if (this.attentionInsights) {
                this.attentionInsights.innerHTML = `
                    <div class="text-center text-muted py-3">
                        <i class="fas fa-exclamation-circle fa-lg mb-2"></i>
                        <p>No attention insights available</p>
                    </div>
                `;
            }
            return;
        }
        
        const tokens = sample.attention.tokens;
        const weights = sample.attention.weights;
        
        // 상위 3개 어텐션 토큰 찾기
        const tokenWeightPairs = tokens.map((token, index) => ({
            token,
            weight: weights[index] || 0
        })).sort((a, b) => b.weight - a.weight).slice(0, 3);
        
        const insightsHtml = `
            <div class="attention-top-tokens mb-3">
                <h6 class="mb-2">Top Attention Words:</h6>
                ${tokenWeightPairs.map((pair, index) => {
                    const colors = ['#dc3545', '#fd7e14', '#ffc107'];
                    const color = colors[index] || '#6c757d';
                    
                    return `
                        <div class="token-item d-flex justify-content-between align-items-center mb-2 p-2" style="background-color: rgba(${this.hexToRgb(color)}, 0.1); border-radius: 6px; border-left: 3px solid ${color};">
                            <span style="font-weight: bold;">"${pair.token}"</span>
                            <span class="badge" style="background-color: ${color};">${pair.weight.toFixed(3)}</span>
                        </div>
                    `;
                }).join('')}
            </div>
            
            <div class="attention-analysis">
                <h6 class="mb-2">Analysis:</h6>
                <p class="text-muted mb-0" style="font-size: 0.9rem;">
                    The model focused most on <strong>"${tokenWeightPairs[0]?.token}"</strong> 
                    (${(tokenWeightPairs[0]?.weight * 100).toFixed(1)}% attention), 
                    suggesting this word was crucial for the ${sample.market_sentiment.toLowerCase()} sentiment classification.
                </p>
            </div>
        `;
        
        this.attentionInsights.innerHTML = insightsHtml;
    }

    /**
     * Hex 색상을 RGB로 변환
     */
    hexToRgb(hex) {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? 
            `${parseInt(result[1], 16)}, ${parseInt(result[2], 16)}, ${parseInt(result[3], 16)}` 
            : '0, 0, 0';
    }

    /**
     * 오류 메시지 표시
     */
    showError(message) {
        const errorHtml = `
            <div class="alert alert-danger" role="alert">
                <i class="fas fa-exclamation-triangle me-2"></i>
                ${message}
            </div>
        `;
        
        if (this.cotReasoningSteps) {
            this.cotReasoningSteps.innerHTML = errorHtml;
        }
        
        if (this.attentionInsights) {
            this.attentionInsights.innerHTML = errorHtml;
        }
    }

    /**
     * 컴포넌트 새로고침
     */
    async refresh() {
        console.log('🔄 Refreshing XAI Visualization...');
        await this.loadXAIData();
        
        if (this.xaiData?.xai_samples?.length > 0) {
            this.populateSampleSelector();
            this.updateCoTAnalysis(this.currentSampleIndex);
        }
    }

    /**
     * 컴포넌트 상태 반환
     */
    getStatus() {
        return {
            isInitialized: this.isInitialized,
            samplesLoaded: this.xaiData?.xai_samples?.length || 0,
            currentSample: this.currentSampleIndex
        };
    }
}

// 전역 XAI Visualization 인스턴스
let xaiVisualization = null;

/**
 * 전역 함수들 (HTML에서 호출)
 */

// CoT 분석 업데이트 (HTML에서 호출)
function updateCoTAnalysis() {
    if (xaiVisualization) {
        const selectedIndex = parseInt(document.getElementById('cotSampleSelect').value);
        xaiVisualization.updateCoTAnalysis(selectedIndex);
    }
}

// XAI Visualization 초기화 (다른 컴포넌트에서 호출)
async function initializeXAIVisualization() {
    try {
        xaiVisualization = new XAIVisualization();
        await xaiVisualization.initialize();
        return xaiVisualization;
    } catch (error) {
        console.error('❌ Failed to initialize XAI Visualization:', error);
        return null;
    }
}

// 모듈 내보내기 (ES6 모듈 사용시)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { XAIVisualization, initializeXAIVisualization };
}

console.log('📊 XAI Visualization component loaded');