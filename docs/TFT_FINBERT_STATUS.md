# Transformer & FinBERT 구현 현황 보고서

**날짜**: 2025-12-09  
**대상**: Temporal Fusion Transformer (TFT), FinBERT 뉴스 심리 분석  
**상태**: 구현 보류 중

---

## 🔄 Temporal Fusion Transformer (TFT)

### 현황: 미구현

### 기술적 실현 가능성: ⭐⭐⭐⭐ (4/5)

---

### 1. TFT 개요

#### 개발사
Google Research (2020)

#### 핵심 강점
- **Multi-horizon 예측**: 동시에 여러 시점 예측
- **Attention 메커니즘**: 중요 특성/시점 자동 식별
- **해석 가능성**: Attention weight 시각화 가능
- **정적/동적 변수 처리**: 종목 정보 + 시계열 데이터 통합

#### 적용 사례
- 소매 판매 예측 (Walmart)
- 전력 수요 예측
- 금융시계열 예측 (JPMorgan 등)

---

### 2. 구현 계획

#### 필요 라이브러리
```bash
pip install pytorch-forecasting>=0.10.0
pip install pytorch-lightning>=1.9.0
pip install torch>=2.0.0
```

#### 코드 구조
```python
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_lightning import Trainer

# 1. 데이터 준비
training = TimeSeriesDataSet(
    data[lambda x: x.index < "2024-01-01"],
    time_idx="time_idx",
    target="target_vol",
    group_ids=["ticker"],  # SPY
    min_encoder_length=30,  # 최소 30일 이력
    max_encoder_length=60,  # 최대 60일 이력
    min_prediction_length=1,
    max_prediction_length=5,  # 5일 예측
    static_categoricals=[],
    time_varying_known_reals=["time_idx"],  # 알려진 미래 값
    time_varying_unknown_reals=[  # 예측 대상
        "returns", "volatility_5", "vix", "volume"
    ],
    target_normalizer=GroupNormalizer(
        groups=["ticker"], transformation="softplus"
    ),
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

# 2. 모델 정의
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,  # LSTM hidden units
    attention_head_size=4,  # Multi-head attention
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7,  # Quantile outputs (10, 25, 50, 75, 90 percentiles)
    loss=QuantileLoss(),
    log_interval=10,
    reduce_on_plateau_patience=4,
)

# 3. 학습
trainer = Trainer(
    max_epochs=100,
    gpus=0,  # CPU
    gradient_clip_val=0.1,
)

trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

# 4. 예측 및 해석
predictions = tft.predict(val_dataloader)
interpretation = tft.interpret_output(predictions)

# Attention weight 시각화
tft.plot_prediction(x, out, idx=0, add_loss_to_title=True)
```

---

### 3. 구현 장애물

#### A. 데이터 형식 변환 (⭐⭐⭐)
**문제**: 현재 데이터는 단일 시계열(SPY), TFT는 패널 데이터 기대

**현재 구조**:
```
Date        Close   VIX     target_vol
2020-01-01  330.5   12.3    0.008
2020-01-02  331.2   12.1    0.009
```

**필요 구조**:
```
ticker  time_idx  Close   VIX     target_vol
SPY     0         330.5   12.3    0.008
SPY     1         331.2   12.1    0.009
```

**해결책**:
```python
# 단일 시계열을 패널 형식으로 변환
spy['ticker'] = 'SPY'
spy['time_idx'] = np.arange(len(spy))
spy = spy.reset_index()
```

#### B. 학습 시간 (⭐⭐⭐⭐)
**예상 시간**: CPU 기준 2-4시간

**이유**:
- Attention 메커니즘의 O(n²) 복잡도
- 100 epochs 권장
- 현재 데이터: ~1,400 샘플

**최적화 방안**:
1. GPU 사용 (10-20배 속도 향상)
2. Early stopping (patience=10)
3. 더 작은 모델 (hidden_size=8)

#### C. 과적합 위험 (⭐⭐⭐⭐⭐)
**문제**: 복잡한 모델 vs 적은 데이터

**TFT 파라미터 수**:
```
- LSTM layers: ~16K params
- Attention: ~8K params
- FFN: ~4K params
Total: ~30K parameters
```

**현재 데이터**:
- 학습 샘플: ~1,000개
- 파라미터/샘플 비율: 30:1 (과적합 위험)

**권장 비율**: 1:10 이상

**해결책**:
1. 정규화 강화 (dropout=0.3)
2. 더 단순한 모델
3. 데이터 증강 (5분봉 등)

---

### 4. 성능 예측

#### 기대 효과
**낙관적 시나리오** (R² +0.05):
- 조건: GPU, 충분한 정규화, 고빈도 데이터
- 근거: 선행 연구에서 LSTM 대비 5-10% 개선

**현실적 시나리오** (R² -0.03 ~ +0.02):
- 조건: 현재 일별 데이터, CPU
- 근거: 데이터 부족으로 과적합 가능성

**비관적 시나리오** (R² -0.05):
- 조건: 하이퍼파라미터 튜닝 실패
- 근거: GARCH-LSTM처럼 복잡도가 독이 될 수 있음

#### 리스크 평가
```
데이터 부족:       ⚠️⚠️⚠️⚠️⚠️ (매우 높음)
복잡도 과다:       ⚠️⚠️⚠️⚠️ (높음)
학습 시간:         ⚠️⚠️⚠️ (중간)
구현 난이도:       ⚠️⚠️ (낮음)
```

---

### 5. 권장 사항

#### ❌ 현재 구현 권장하지 않음

**이유**:
1. **ROI 낮음**: 2-4시간 투자로 성능 향상 불확실
2. **현재 모델 우수**: R² 0.2572는 이미 양호
3. **데이터 부족**: 일별 데이터로는 TFT의 강점 발휘 어려움

#### ✅ 구현이 가치 있는 경우

**조건**:
1. **고빈도 데이터**: 5분/1분봉 확보
2. **다중 자산**: SPY, QQQ, IWM 등 여러 ETF
3. **더 긴 기간**: 10년+ 데이터
4. **GPU 환경**: 학습 시간 단축

**예상 ROI**:
- 5분봉 데이터 (50배 샘플 증가) → R² +0.10 가능
- 다중 자산 (5개 ETF) → 패널 데이터의 강점 활용

---

### 6. 대안: 경량 Transformer

#### Informer (2021)
```python
# TFT보다 빠르고 단순
from informer import Informer

model = Informer(
    enc_in=43,  # 입력 특성 수
    dec_in=43,
    c_out=1,    # 출력 (target_vol)
    seq_len=30,
    label_len=15,
    out_len=5,
    factor=3,
    d_model=256,
    n_heads=4,
    e_layers=2,
    d_layers=1,
)
```

**장점**:
- TFT의 50% 파라미터
- 긴 시퀀스 처리 효율적
- 구현 더 단순

---

## 📰 FinBERT 뉴스 심리 분석

### 현황: 미구현

### 기술적 실현 가능성: ⭐⭐⭐ (3/5)

---

### 1. FinBERT 개요

#### 개발사
ProsusAI (네덜란드 투자회사 Prosus)

#### 모델 특징
- **기반**: BERT-base (110M 파라미터)
- **훈련 데이터**: 금융 뉴스, 기업 보고서
- **출력**: 긍정/부정/중립 확률

#### 성능
```
금융 감성 분류 정확도: ~97%
일반 BERT 대비: +7% 개선
```

---

### 2. 구현 아키텍처

### Phase 1: 뉴스 수집

#### 옵션 A: NewsAPI (유료)
```python
from newsapi import NewsApiClient

newsapi = NewsApiClient(api_key='YOUR_KEY')

# SPY 관련 뉴스
articles = newsapi.get_everything(
    q='SPY OR "S&P 500" OR market',
    from_param='2020-01-01',
    to='2025-01-01',
    language='en',
    sort_by='publishedAt',
    page_size=100
)

# 비용: $449/month (개발자 플랜)
# 제한: 1,000 requests/day
```

#### 옵션 B: Yahoo Finance News (무료)
```python
import yfinance as yf

ticker = yf.Ticker('SPY')
news = ticker.news  # 최근 뉴스만 (무료)

# 제한: 최근 뉴스만, 역사적 데이터 없음
```

#### 옵션 C: Reddit/Twitter Scraping
```python
import praw  # Reddit API

reddit = praw.Reddit(
    client_id='YOUR_ID',
    client_secret='YOUR_SECRET',
    user_agent='SPY News Collector'
)

# r/wallstreetbets posts
for submission in reddit.subreddit('wallstreetbets').hot(limit=100):
    if 'SPY' in submission.title or 'SPY' in submission.selftext:
        # 수집

# 장점: 무료
# 단점: 법적 그레이 영역, 데이터 품질 낮음
```

---

### Phase 2: 감성 분석

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# FinBERT 모델 로드
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def analyze_sentiment(text):
    """뉴스 텍스트의 감성 분석"""
    inputs = tokenizer(
        text, 
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # [positive, negative, neutral]
    return {
        'positive': probabilities[0][0].item(),
        'negative': probabilities[0][1].item(),
        'neutral': probabilities[0][2].item()
    }

# 예시
text = "S&P 500 surges to record high on strong earnings"
sentiment = analyze_sentiment(text)
# {'positive': 0.89, 'negative': 0.05, 'neutral': 0.06}
```

---

### Phase 3: 특성 생성

```python
import pandas as pd

# 일별 감성 지수 집계
def aggregate_daily_sentiment(news_df):
    """뉴스를 일별로 집계"""
    daily = news_df.groupby('date').agg({
        'positive': ['mean', 'std', 'max'],
        'negative': ['mean', 'std', 'max'],
        'neutral': 'mean',
        'article_count': 'count'
    })
    
    # 특성 생성
    features = pd.DataFrame({
        # 기본 감성
        'news_positive': daily['positive']['mean'],
        'news_negative': daily['negative']['mean'],
        'news_neutral': daily['neutral']['mean'],
        
        # 감성 순위 (긍정 - 부정)
        'news_sentiment_net': daily['positive']['mean'] - daily['negative']['mean'],
        
        # 감성 변동성
        'news_positive_std': daily['positive']['std'],
        'news_negative_std': daily['negative']['std'],
        
        # 뉴스 볼륨
        'news_volume': daily['article_count'],
        'news_volume_spike': daily['article_count'] / daily['article_count'].rolling(7).mean(),
        
        # 극단 감성
        'news_extreme_positive': daily['positive']['max'],
        'news_extreme_negative': daily['negative']['max'],
    })
    
    # 래그 특성
    for lag in [1, 3, 5]:
        features[f'news_sentiment_lag_{lag}'] = features['news_sentiment_net'].shift(lag)
    
    return features

# SPY 데이터와 병합
spy_with_news = spy.merge(features, left_index=True, right_index=True, how='left')
```

---

### 3. 구현 장애물

#### A. 데이터 수집 비용 (⭐⭐⭐⭐⭐)

**NewsAPI 비용 분석**:
```
개발자 플랜: $449/month
- 1,000 requests/day
- 5년 데이터 = 1,825일
- 필요: 최소 2개월 구독 = $898

엔터프라이즈 플랜: Custom pricing
- 제한 없음
- 예상: $2,000-5,000/month
```

**무료 대안의 한계**:
```
Yahoo Finance News:
  ✓ 무료
  ✗ 최근 뉴스만 (역사적 데이터 없음)
  ✗ 5년치 불가

Reddit/Twitter:
  ✓ 무료
  ✗ 법적 리스크
  ✗ 저품질 데이터
  ✗ API 정책 변경 위험
```

#### B. 계산 비용 (⭐⭐⭐⭐)

**추론 시간 측정**:
```python
import time

# 단일 뉴스 처리
start = time.time()
sentiment = analyze_sentiment(text)
elapsed = time.time() - start

print(f"단일 뉴스: {elapsed:.2f}초")
# CPU: ~0.5초/뉴스
# GPU: ~0.05초/뉴스

# 5년치 뉴스 추정
total_news = 1825 * 20  # 일평균 20개 뉴스
total_time_cpu = total_news * 0.5 / 3600
print(f"전체 처리 시간 (CPU): {total_time_cpu:.1f}시간")
# ~10시간

total_time_gpu = total_news * 0.05 / 3600
print(f"전체 처리 시간 (GPU): {total_time_gpu:.1f}시간")
# ~1시간
```

#### C. 데이터 품질 (⭐⭐⭐⭐)

**문제**: 뉴스와 변동성의 시간차

```
뉴스 발행 → 시장 반응 → 변동성 변화
   t=0        t=?           t=?
```

**시간 정렬 이슈**:
- 장 마감 후 뉴스 → 다음날 영향
- 뉴스 타임스탬프 정확도 문제
- 주말 뉴스 → 월요일 적용?

**해결책**:
```python
# 다양한 래그 시도
for lag in [0, 1, 2, 3]:
    spy[f'news_sentiment_lag_{lag}'] = news['sentiment'].shift(lag)
```

#### D. 언어 모델 의존성 (⭐⭐)

**Transformers 라이브러리 크기**:
```
- transformers: ~400MB
- torch: ~800MB
- FinBERT 모델: ~440MB
Total: ~1.6GB 디스크 공간
```

**배포 복잡도**:
- Docker 이미지 크기 증가
- 메모리 사용량 증가 (모델 로딩 시 ~2GB RAM)
- GPU 인프라 고려 시 비용 상승

---

### 4. 성능 예측

#### 선행 연구 결과

**논문**: "FinBERT for Financial Sentiment Analysis" (2020)
```
뉴스 감성 추가 시:
- Stock Return 예측: R² +0.04
- Volatility 예측: R² +0.03-0.07
```

**조건**:
- 고빈도 뉴스 (일 50+ 기사)
- 장기 데이터 (10년+)
- 고빈도 가격 데이터 (분봉)

#### 현재 상황 적용

**낙관적 시나리오** (R² +0.05):
```
조건:
- NewsAPI 유료 구독
- GPU 추론
- 고품질 뉴스 소스
- 적절한 시간 정렬

예상 성능: 0.2572 → 0.3072
```

**현실적 시나리오** (R² +0.01-0.03):
```
조건:
- 무료 뉴스 (Yahoo Finance, Reddit)
- CPU 추론
- 시간 정렬 문제

예상 성능: 0.2572 → 0.2672
```

**비관적 시나리오** (R² -0.02):
```
조건:
- 저품질 뉴스
- 시간 정렬 실패
- 노이즈 증가

예상 성능: 0.2572 → 0.2372
```

---

### 5. 비용-편익 분석

#### 구현 비용
```
개발 시간:       20-40시간
NewsAPI (2개월): $898
GPU 인스턴스:    $100/month (선택)
Total:           $1,000-1,500 + 개발시간
```

#### 예상 편익
```
낙관적: R² +0.05 → 월 수익 +5%?
현실적: R² +0.02 → 월 수익 +2%?
비관적: R² -0.02 → 손실
```

#### ROI 평가
```
투자: $1,500
필요 월수익: 최소 $150 (10% ROI)

현재 불확실:
- 실제 트레이딩 성능 미검증
- R² 향상이 수익으로 직결되지 않을 수 있음
```

---

### 6. 권장 사항

#### ❌ 현재 구현 권장하지 않음

**이유**:
1. **높은 비용**: $1,000+ 투자 필요
2. **불확실한 효과**: R² +0.02 정도 예상
3. **복잡도**: 배포 및 유지보수 부담
4. **대안 존재**: VIX가 이미 시장 심리 반영

#### ✅ MVP 접근 (저비용 테스트)

**Phase 1: 무료 프록시**
```python
# VIX 변화를 공포 지수로 활용
fear_index = spy['VIX'].pct_change()

# 거래량 급증을 뉴스 이벤트 프록시로
news_proxy = (spy['Volume'] - spy['Volume'].rolling(20).mean()) / \
              spy['Volume'].rolling(20).std()

# 극단 수익률
extreme_event = (abs(spy['returns']) > 2 * spy['returns'].rolling(20).std()).astype(int)
```

**Phase 2: Reddit 실험**
```python
# PRAW (무료)로 r/wallstreetbets 데이터
# 1주일치 테스트
# 성능 향상 확인 시 → NewsAPI 고려
```

---

### 7. FinBERT 경량 대안

#### VADER Sentiment (규칙 기반)
```python
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def vader_sentiment(text):
    scores = analyzer.polarity_scores(text)
    return scores['compound']  # -1 to 1

# 장점:
# - 즉시 실행 (모델 로딩 불필요)
# - 빠름 (0.001초/뉴스)
# - 무료

# 단점:
# - 정확도 낮음 (금융 특화 X)
# - FinBERT 대비 -10-15% 정확도
```

---

## 🎯 종합 결론

### Transformer (TFT)

**상태**: 🔄 구현 보류  
**이유**:
- 데이터 부족 (일별 데이터로 강점 발휘 어려움)
- 과적합 위험
- 현재 ElasticNet 성능 충분

**조건부 권장**:
```
IF (고빈도 데이터 AND 다중 자산 AND GPU)
THEN 구현 고려
ELSE 보류
```

---

### FinBERT

**상태**: 🔄 구현 보류  
**이유**:
- 높은 데이터 수집 비용 ($900+)
- 불확실한 ROI
- VIX가 이미 시장 심리 반영

**조건부 권장**:
```
IF (프로덕션 수익 검증 AND 예산 확보)
THEN 무료 프록시로 MVP 테스트
    IF 효과 확인
    THEN NewsAPI 구독 고려
ELSE 보류
```

---

## 📋 최종 우선순위

### 즉시 실행 가능 (높은 ROI)
1. ✅ 현재 모델 (R² 0.2572) 프로덕션 배포
2. 📊 백테스트 및 트레이딩 전략 검증
3. 📈 실시간 모니터링 대시보드

### 단기 (1-2주)
4. 🔧 간단한 앙상블 (ElasticNet + GradientBoosting)
5. 📉 확률적 예측 (예측 구간)
6. 🧪 무료 뉴스 프록시 (VIX 변화, 거래량)

### 중기 (1-2개월)
7. 📰 Reddit/Twitter 뉴스 MVP
8. 🎯 옵션 내재변동성 데이터
9. 📊 고빈도 데이터 수집 (5분봉)

### 장기 (3개월+, 조건부)
10. 🤖 TFT (고빈도 데이터 확보 시)
11. 📰 FinBERT (수익성 검증 후)
12. 🧪 최신 모델 (PatchTST, TimesNet 등)

---

**보고서 작성**: 2025-12-09  
**다음 검토**: 프로덕션 성과 확인 후
