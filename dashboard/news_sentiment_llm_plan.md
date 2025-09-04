# 뉴스 감정 분석 LLM 통합 강화 계획

## 📊 현재 모델 상태 분석

### 기존 모델 한계점
- **현실적 정확도**: 53-56% (검증된 안정적 범위)
- **주요 특성**: 기술적 지표 기반 (RSI, MA, VIX 등)
- **누락 요소**: 시장 심리, 뉴스 이벤트, 거시경제 신호

### 개선 잠재력
- **뉴스 감정 분석 추가**: 1-3% 정확도 향상 예상
- **목표**: 55-58% 달성 (현실적 상한선)
- **안정성**: 기존 견고함 유지하며 점진적 개선

## 🧠 LLM 감정 분석 아키텍처

### 1. 다층 감정 분석 시스템

```python
class NewsLLMSentimentAnalyzer:
    """
    다층 LLM 기반 뉴스 감정 분석기
    - Layer 1: 기본 감정 분석 (Positive/Negative/Neutral)
    - Layer 2: 시장 영향도 분석 (High/Medium/Low Impact)
    - Layer 3: 섹터별 영향 분석 (SPY 관련도)
    """
    
    def __init__(self):
        self.sentiment_model = "claude-3-haiku"  # 빠르고 경제적
        self.impact_model = "gpt-4o-mini"       # 정확한 영향도 분석
        
    def analyze_news_sentiment(self, news_text, date):
        # 3단계 분석 파이프라인
        sentiment = self.get_basic_sentiment(news_text)
        impact = self.get_market_impact(news_text)
        relevance = self.get_spy_relevance(news_text)
        
        return {
            'sentiment_score': sentiment,    # -1 to +1
            'market_impact': impact,         # 0 to 1
            'spy_relevance': relevance,      # 0 to 1
            'composite_score': self.calculate_composite(sentiment, impact, relevance)
        }
```

### 2. 뉴스 소스 및 데이터 수집

#### 주요 뉴스 소스
- **Financial News**: Reuters, Bloomberg, WSJ, CNBC
- **Economic Indicators**: Fed 발표, 경제 지표 발표
- **Corporate News**: SP500 주요 기업 뉴스
- **Geopolitical Events**: 전쟁, 선거, 정책 변화

#### 데이터 수집 전략
```python
class NewsDataCollector:
    def __init__(self):
        self.sources = {
            'newsapi': NewsAPIClient(),      # 실시간 뉴스
            'reddit': RedditAPI(),           # 소셜 감정
            'twitter': TwitterAPI(),         # 실시간 반응
            'fed': FedNewsRSS()             # 연준 발표
        }
    
    def collect_daily_news(self, date):
        # 매일 오전 9시 (시장 개장 전) 뉴스 수집
        # SPY 관련 키워드: "S&P 500", "market", "economy", "Fed"
        pass
```

## 🔧 구현 계획

### Phase 1: 뉴스 데이터 수집 (1주)

#### 1.1 API 설정 및 키워드 정의
```python
# 핵심 키워드 매트릭스
MARKET_KEYWORDS = {
    'positive': ['rally', 'surge', 'bullish', 'growth', 'strong earnings'],
    'negative': ['crash', 'drop', 'bearish', 'recession', 'weak'],
    'neutral': ['sideways', 'mixed', 'stable', 'unchanged']
}

SPY_SPECIFIC_KEYWORDS = [
    'S&P 500', 'SPY ETF', 'broad market', 'large cap',
    'market index', 'equity market'
]
```

#### 1.2 뉴스 필터링 시스템
- **시간 필터**: 시장 시간 외 뉴스는 다음날 적용
- **관련도 필터**: SPY 관련도 0.3 이상만 사용
- **품질 필터**: 신뢰할 수 있는 소스만 활용

### Phase 2: LLM 감정 분석 엔진 (1-2주)

#### 2.1 프롬프트 엔지니어링
```python
SENTIMENT_ANALYSIS_PROMPT = """
다음 뉴스 기사를 분석하여 S&P 500 지수(SPY)에 대한 영향을 평가해주세요:

뉴스: {news_text}
날짜: {date}

다음 기준으로 평가:
1. 감정 점수 (-1.0 = 매우 부정적, 0 = 중립적, +1.0 = 매우 긍정적)
2. 시장 영향도 (0.0 = 영향 없음, 1.0 = 큰 영향)
3. SPY 관련도 (0.0 = 무관, 1.0 = 직접 관련)

JSON 형태로 응답:
{{
    "sentiment_score": float,
    "market_impact": float,
    "spy_relevance": float,
    "reasoning": "분석 근거"
}}
"""
```

#### 2.2 배치 처리 시스템
```python
class BatchNewsProcessor:
    def __init__(self, max_concurrent=10):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
    async def process_daily_news(self, news_list):
        tasks = [self.analyze_single_news(news) for news in news_list]
        results = await asyncio.gather(*tasks)
        return self.aggregate_sentiment_scores(results)
```

### Phase 3: 특성 통합 (1주)

#### 3.1 감정 특성 생성
```python
def create_sentiment_features(sentiment_data, lookback_days=5):
    """
    뉴스 감정 데이터로부터 모델 특성 생성
    """
    features = {}
    
    # 단기 감정 특성 (1-5일)
    for days in [1, 3, 5]:
        recent_sentiment = sentiment_data[-days:].mean()
        features[f'news_sentiment_{days}d'] = recent_sentiment
        
    # 감정 변화율
    features['sentiment_momentum'] = calculate_sentiment_momentum(sentiment_data)
    
    # 뉴스 볼륨 (뉴스 개수)
    features['news_volume'] = len(sentiment_data)
    
    # 감정 변동성
    features['sentiment_volatility'] = sentiment_data.std()
    
    return features
```

#### 3.2 기존 모델과 통합
```python
class EnhancedSPYPredictor:
    def __init__(self):
        self.technical_features = ['returns_lag1', 'vix_change', 'volatility_20']
        self.sentiment_features = ['news_sentiment_1d', 'sentiment_momentum']
        
    def create_enhanced_features(self, price_data, sentiment_data):
        # 기존 기술적 특성 (검증됨)
        tech_features = self.create_technical_features(price_data)
        
        # 새로운 감정 특성
        sent_features = self.create_sentiment_features(sentiment_data)
        
        # 통합 특성 벡터 (총 7-8개 특성 유지)
        return {**tech_features, **sent_features}
```

## 📊 성능 향상 예상

### 정량적 목표
- **기존 성능**: 53-56% 정확도
- **감정 분석 추가 후**: 55-58% 목표
- **개선폭**: 2-3% 향상 (현실적)

### 개선 메커니즘
1. **이벤트 예측**: 뉴스 이벤트 기반 단기 방향성
2. **심리적 요인**: 시장 참여자들의 감정적 반응
3. **타이밍**: 뉴스 발표 직후 시장 반응 예측

### 리스크 관리
- **오버피팅 방지**: 감정 특성 3개 이하로 제한
- **데이터 누수**: 뉴스는 항상 1일 지연 적용
- **노이즈 관리**: 신뢰도 낮은 뉴스 필터링

## 🚀 실행 타임라인

### 1주차: 인프라 구축
- [ ] News API 키 발급 및 설정
- [ ] 뉴스 수집 스크립트 개발
- [ ] 키워드 기반 필터링 시스템

### 2주차: LLM 분석 엔진
- [ ] Claude/GPT API 통합
- [ ] 프롬프트 최적화
- [ ] 배치 처리 시스템 구축

### 3주차: 모델 통합
- [ ] 감정 특성 생성 파이프라인
- [ ] 기존 모델에 통합
- [ ] 백테스팅 및 검증

### 4주차: 성능 검증
- [ ] A/B 테스트: 기존 vs 강화 모델
- [ ] 오버피팅 검증
- [ ] 실시간 시스템 배포

## 💰 비용 분석

### API 비용 (월간)
- **News API**: ~$50/월 (1000 요청/일)
- **Claude API**: ~$30/월 (뉴스 분석)
- **GPT API**: ~$20/월 (보조 분석)
- **총 비용**: ~$100/월

### ROI 분석
- **투자**: $100/월 + 개발 시간
- **기대 수익**: 2-3% 정확도 향상
- **실제 거래 적용 시**: 월 수익률 개선

## 🎯 성공 지표

### 정량적 지표
- **정확도 개선**: 최소 2% 향상
- **안정성 유지**: CV 표준편차 < 3%
- **실시간 성능**: 일일 예측 정확도 55%+

### 정성적 지표
- **해석 가능성**: 감정 점수의 경제적 의미 명확
- **시스템 안정성**: 24/7 무중단 뉴스 수집
- **확장성**: 다른 지수/종목 적용 가능

## 🔬 실험 설계

### 백테스팅 전략
```python
class SentimentEnhancedBacktest:
    def __init__(self):
        self.baseline_features = ['returns_lag1', 'vix_change', 'volatility_20']
        self.enhanced_features = self.baseline_features + ['news_sentiment_1d']
        
    def run_comparison_test(self, start_date, end_date):
        # 1. 기존 모델 성능
        baseline_acc = self.test_model(self.baseline_features)
        
        # 2. 강화 모델 성능  
        enhanced_acc = self.test_model(self.enhanced_features)
        
        # 3. 통계적 유의성 검증
        improvement = enhanced_acc - baseline_acc
        p_value = self.statistical_significance_test(baseline_acc, enhanced_acc)
        
        return {
            'baseline_accuracy': baseline_acc,
            'enhanced_accuracy': enhanced_acc, 
            'improvement': improvement,
            'p_value': p_value
        }
```

이 계획을 통해 현실적이고 검증 가능한 방식으로 뉴스 감정 분석을 SPY 예측 모델에 통합하여 2-3% 성능 향상을 목표로 합니다.