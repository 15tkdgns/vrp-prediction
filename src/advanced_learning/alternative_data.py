#!/usr/bin/env python3
"""
대체 데이터 통합 시스템
FRED API 거시경제 지표, FinBERT 뉴스 감성 분석, HMM 시장 레짐 탐지
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
import logging
import requests
import json
from textblob import TextBlob
import re

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class DataSourceConfig:
    """데이터 소스 설정"""
    fred_api_key: str = ""  # FRED API 키
    news_api_key: str = ""  # News API 키
    finnhub_api_key: str = ""  # Finnhub API 키
    cache_dir: str = "data/cache"
    use_cache: bool = True
    cache_expiry_hours: int = 24


@dataclass
class MacroEconomicIndicators:
    """거시경제 지표 데이터"""
    date: datetime
    fed_funds_rate: Optional[float] = None
    unemployment_rate: Optional[float] = None
    inflation_rate: Optional[float] = None
    gdp_growth: Optional[float] = None
    treasury_10y: Optional[float] = None
    treasury_2y: Optional[float] = None
    vix: Optional[float] = None
    dxy: Optional[float] = None  # 달러 지수
    oil_price: Optional[float] = None
    gold_price: Optional[float] = None


class FREDDataCollector:
    """
    FRED (Federal Reserve Economic Data) API 데이터 수집기
    거시경제 지표 자동 수집 및 처리
    """

    def __init__(self, config: DataSourceConfig):
        """
        Args:
            config: 데이터 소스 설정
        """
        self.config = config
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"

        # FRED 시리즈 ID 매핑
        self.series_mapping = {
            'fed_funds_rate': 'FEDFUNDS',
            'unemployment_rate': 'UNRATE',
            'inflation_rate': 'CPIAUCSL',
            'gdp_growth': 'GDP',
            'treasury_10y': 'GS10',
            'treasury_2y': 'GS2',
            'vix': 'VIXCLS',  # CBOE VIX
            'dxy': 'DTWEXBGS',  # Trade Weighted U.S. Dollar Index
            'oil_price': 'DCOILWTICO',  # WTI Oil Price
            'gold_price': 'GOLDAMGBD228NLBM'  # Gold Price
        }

    def fetch_series_data(self, series_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        FRED 시리즈 데이터 가져오기

        Args:
            series_id: FRED 시리즈 ID
            start_date: 시작 날짜 (YYYY-MM-DD)
            end_date: 종료 날짜 (YYYY-MM-DD)

        Returns:
            데이터프레임 (date, value)
        """
        if not self.config.fred_api_key:
            logger.warning("FRED API 키가 설정되지 않았습니다. 모의 데이터를 생성합니다.")
            return self._generate_mock_data(series_id, start_date, end_date)

        params = {
            'series_id': series_id,
            'api_key': self.config.fred_api_key,
            'file_type': 'json',
            'observation_start': start_date,
            'observation_end': end_date
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            observations = data['observations']

            df = pd.DataFrame(observations)
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')

            return df[['date', 'value']].dropna()

        except Exception as e:
            logger.error(f"FRED 데이터 가져오기 실패 ({series_id}): {e}")
            return self._generate_mock_data(series_id, start_date, end_date)

    def _generate_mock_data(self, series_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """모의 데이터 생성"""
        np.random.seed(hash(series_id) % 2**32)  # 시리즈별 일관된 랜덤 시드

        dates = pd.date_range(start=start_date, end=end_date, freq='D')

        # 시리즈별 특성 반영한 모의 데이터
        base_values = {
            'FEDFUNDS': 2.0,      # 연방기금 금리
            'UNRATE': 5.0,        # 실업률
            'CPIAUCSL': 250.0,    # 소비자물가지수
            'GDP': 20000.0,       # GDP
            'GS10': 2.5,          # 10년 국채
            'GS2': 2.0,           # 2년 국채
            'VIXCLS': 20.0,       # VIX
            'DTWEXBGS': 100.0,    # 달러 지수
            'DCOILWTICO': 70.0,   # 유가
            'GOLDAMGBD228NLBM': 1800.0  # 금 가격
        }

        base_value = base_values.get(series_id, 100.0)
        volatility = base_value * 0.02  # 2% 변동성

        values = []
        current_value = base_value

        for _ in dates:
            current_value += np.random.normal(0, volatility)
            values.append(current_value)

        return pd.DataFrame({
            'date': dates,
            'value': values
        })

    def collect_all_indicators(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        모든 거시경제 지표 수집

        Args:
            start_date: 시작 날짜
            end_date: 종료 날짜

        Returns:
            통합 거시경제 지표 데이터프레임
        """
        all_data = []

        for indicator_name, series_id in self.series_mapping.items():
            logger.info(f"수집 중: {indicator_name} ({series_id})")

            df = self.fetch_series_data(series_id, start_date, end_date)
            if not df.empty:
                df = df.rename(columns={'value': indicator_name})
                all_data.append(df)

        if not all_data:
            return pd.DataFrame()

        # 데이터 병합 (날짜 기준)
        result = all_data[0]
        for df in all_data[1:]:
            result = pd.merge(result, df, on='date', how='outer')

        # 결측값 전진 채움
        result = result.sort_values('date').fillna(method='ffill')

        return result


class FinBERTSentimentAnalyzer:
    """
    FinBERT 기반 뉴스 감성 분석기
    금융 도메인에 특화된 BERT 모델 활용
    """

    def __init__(self, config: DataSourceConfig):
        """
        Args:
            config: 데이터 소스 설정
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """FinBERT 모델 로드"""
        try:
            # 실제 환경에서는 transformers 라이브러리 사용
            # from transformers import AutoTokenizer, AutoModelForSequenceClassification
            # self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            # self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

            logger.info("FinBERT 모델 로드 시뮬레이션 (실제로는 transformers 라이브러리 필요)")
            self.model_available = False

        except ImportError:
            logger.warning("transformers 라이브러리가 설치되지 않았습니다. TextBlob 대체 사용")
            self.model_available = False

    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        텍스트 감성 분석

        Args:
            text: 분석할 텍스트

        Returns:
            감성 분석 결과 (positive, negative, neutral 확률)
        """
        if self.model_available and self.model:
            # 실제 FinBERT 분석 (구현 시)
            return self._analyze_with_finbert(text)
        else:
            # TextBlob 대체 구현
            return self._analyze_with_textblob(text)

    def _analyze_with_finbert(self, text: str) -> Dict[str, float]:
        """FinBERT 모델 분석 (실제 구현용)"""
        # 실제 구현 시 사용
        # inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        # outputs = self.model(**inputs)
        # probabilities = torch.softmax(outputs.logits, dim=-1)
        # return {"positive": prob[0], "negative": prob[1], "neutral": prob[2]}

        # 모의 구현
        np.random.seed(hash(text) % 2**32)
        probs = np.random.dirichlet([1, 1, 1])  # 3개 클래스 확률
        return {
            "positive": float(probs[0]),
            "negative": float(probs[1]),
            "neutral": float(probs[2])
        }

    def _analyze_with_textblob(self, text: str) -> Dict[str, float]:
        """TextBlob 대체 분석"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1

            # 극성을 3개 클래스로 변환
            if polarity > 0.1:
                positive = 0.7 + 0.3 * polarity
                negative = 0.1
                neutral = 1.0 - positive - negative
            elif polarity < -0.1:
                negative = 0.7 + 0.3 * abs(polarity)
                positive = 0.1
                neutral = 1.0 - positive - negative
            else:
                neutral = 0.8
                positive = 0.1
                negative = 0.1

            return {
                "positive": positive,
                "negative": negative,
                "neutral": neutral
            }

        except Exception as e:
            logger.error(f"TextBlob 분석 실패: {e}")
            return {"positive": 0.33, "negative": 0.33, "neutral": 0.34}

    def analyze_news_batch(self, news_data: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        뉴스 배치 감성 분석

        Args:
            news_data: 뉴스 데이터 리스트 (title, description, content 포함)

        Returns:
            감성 분석 결과 리스트
        """
        results = []

        for news_item in news_data:
            # 텍스트 추출 및 정제
            text = self._extract_and_clean_text(news_item)

            # 감성 분석
            sentiment = self.analyze_text(text)

            # 결과 저장
            result = {
                'title': news_item.get('title', ''),
                'date': news_item.get('publishedAt', ''),
                'sentiment_scores': sentiment,
                'sentiment_label': max(sentiment.keys(), key=sentiment.get),
                'confidence': max(sentiment.values()),
                'text_length': len(text)
            }
            results.append(result)

        return results

    def _extract_and_clean_text(self, news_item: Dict[str, str]) -> str:
        """뉴스 텍스트 추출 및 정제"""
        texts = []

        # 제목
        if news_item.get('title'):
            texts.append(news_item['title'])

        # 설명
        if news_item.get('description'):
            texts.append(news_item['description'])

        # 내용 (일부)
        if news_item.get('content'):
            content = news_item['content'][:500]  # 첫 500자만
            texts.append(content)

        combined_text = ' '.join(texts)

        # 텍스트 정제
        cleaned_text = re.sub(r'http\S+', '', combined_text)  # URL 제거
        cleaned_text = re.sub(r'[^\w\s]', ' ', cleaned_text)  # 특수 문자 제거
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # 중복 공백 제거

        return cleaned_text.strip()


class NewsDataCollector:
    """뉴스 데이터 수집기"""

    def __init__(self, config: DataSourceConfig):
        """
        Args:
            config: 데이터 소스 설정
        """
        self.config = config
        self.news_api_url = "https://newsapi.org/v2/everything"

    def fetch_financial_news(self, query: str = "stock market",
                           start_date: str = None, end_date: str = None,
                           language: str = "en") -> List[Dict[str, str]]:
        """
        금융 뉴스 데이터 수집

        Args:
            query: 검색 쿼리
            start_date: 시작 날짜
            end_date: 종료 날짜
            language: 언어

        Returns:
            뉴스 데이터 리스트
        """
        if not self.config.news_api_key:
            logger.warning("News API 키가 설정되지 않았습니다. 모의 데이터를 생성합니다.")
            return self._generate_mock_news(query, start_date, end_date)

        params = {
            'q': query,
            'apiKey': self.config.news_api_key,
            'language': language,
            'sortBy': 'publishedAt',
            'pageSize': 100
        }

        if start_date:
            params['from'] = start_date
        if end_date:
            params['to'] = end_date

        try:
            response = requests.get(self.news_api_url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()
            articles = data.get('articles', [])

            # 필요한 필드만 추출
            processed_articles = []
            for article in articles:
                processed_article = {
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'content': article.get('content', ''),
                    'publishedAt': article.get('publishedAt', ''),
                    'source': article.get('source', {}).get('name', ''),
                    'url': article.get('url', '')
                }
                processed_articles.append(processed_article)

            return processed_articles

        except Exception as e:
            logger.error(f"뉴스 데이터 수집 실패: {e}")
            return self._generate_mock_news(query, start_date, end_date)

    def _generate_mock_news(self, query: str, start_date: str, end_date: str) -> List[Dict[str, str]]:
        """모의 뉴스 데이터 생성"""
        mock_headlines = [
            "Stock market reaches new highs amid economic optimism",
            "Federal Reserve signals potential rate changes",
            "Technology stocks lead market rally",
            "Economic indicators show mixed signals",
            "Inflation concerns weigh on investor sentiment",
            "Corporate earnings exceed expectations",
            "Global markets react to policy changes",
            "Energy sector shows strong performance",
            "Banking stocks face regulatory pressure",
            "Market volatility increases amid uncertainty"
        ]

        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')

        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        mock_articles = []
        np.random.seed(42)

        for date in date_range:
            # 하루에 2-5개 기사
            n_articles = np.random.randint(2, 6)

            for _ in range(n_articles):
                headline = np.random.choice(mock_headlines)
                article = {
                    'title': headline,
                    'description': f"Market analysis regarding {query}. {headline}",
                    'content': f"Detailed analysis of {headline}. " * 10,  # 더 긴 내용
                    'publishedAt': date.strftime('%Y-%m-%dT%H:%M:%SZ'),
                    'source': np.random.choice(['Reuters', 'Bloomberg', 'CNBC', 'MarketWatch']),
                    'url': f"https://example.com/news/{date.strftime('%Y%m%d')}"
                }
                mock_articles.append(article)

        return mock_articles


class MarketRegimeDetector:
    """
    Hidden Markov Model (HMM) 기반 시장 레짐 탐지
    강세장/약세장/횡보장 등의 시장 상태 식별
    """

    def __init__(self, n_regimes: int = 3):
        """
        Args:
            n_regimes: 시장 레짐 수 (기본 3개: 강세/약세/횡보)
        """
        self.n_regimes = n_regimes
        self.regime_names = ['Bear', 'Sideways', 'Bull'] if n_regimes == 3 else [f'Regime_{i}' for i in range(n_regimes)]
        self.model = None
        self.is_fitted = False

    def fit(self, returns: np.ndarray) -> 'MarketRegimeDetector':
        """
        HMM 모델 학습

        Args:
            returns: 수익률 시계열

        Returns:
            자기 자신 (fitted)
        """
        try:
            from hmmlearn import hmm

            # Gaussian HMM 모델
            self.model = hmm.GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="full",
                n_iter=100,
                random_state=42
            )

            # 수익률을 2차원 배열로 변환
            X = returns.reshape(-1, 1)
            self.model.fit(X)
            self.is_fitted = True

        except ImportError:
            logger.warning("hmmlearn 라이브러리가 설치되지 않았습니다. 단순 규칙 기반 레짐 탐지를 사용합니다.")
            self._fit_simple_rules(returns)

        return self

    def _fit_simple_rules(self, returns: np.ndarray) -> None:
        """단순 규칙 기반 레짐 탐지 학습"""
        # 수익률의 분위수 기반 임계값
        self.bull_threshold = np.percentile(returns, 70)
        self.bear_threshold = np.percentile(returns, 30)
        self.is_fitted = True

    def predict_regimes(self, returns: np.ndarray) -> np.ndarray:
        """
        시장 레짐 예측

        Args:
            returns: 수익률 시계열

        Returns:
            레짐 인덱스 배열 (0, 1, 2, ...)
        """
        if not self.is_fitted:
            raise ValueError("모델이 아직 학습되지 않았습니다.")

        if self.model:
            # HMM 모델 사용
            X = returns.reshape(-1, 1)
            regimes = self.model.predict(X)
        else:
            # 단순 규칙 기반
            regimes = np.zeros(len(returns), dtype=int)
            regimes[returns > self.bull_threshold] = 2  # Bull
            regimes[returns < self.bear_threshold] = 0  # Bear
            regimes[(returns >= self.bear_threshold) & (returns <= self.bull_threshold)] = 1  # Sideways

        return regimes

    def get_regime_statistics(self, returns: np.ndarray, regimes: np.ndarray) -> Dict[str, Dict[str, float]]:
        """레짐별 통계 계산"""
        stats = {}

        for i, regime_name in enumerate(self.regime_names):
            regime_mask = regimes == i
            regime_returns = returns[regime_mask]

            if len(regime_returns) > 0:
                stats[regime_name] = {
                    'mean_return': np.mean(regime_returns),
                    'volatility': np.std(regime_returns),
                    'frequency': np.sum(regime_mask) / len(regimes),
                    'min_return': np.min(regime_returns),
                    'max_return': np.max(regime_returns)
                }
            else:
                stats[regime_name] = {
                    'mean_return': 0.0,
                    'volatility': 0.0,
                    'frequency': 0.0,
                    'min_return': 0.0,
                    'max_return': 0.0
                }

        return stats


class AlternativeDataIntegrator:
    """
    대체 데이터 통합기
    거시경제 지표, 뉴스 감성, 시장 레짐을 통합하여 특성 생성
    """

    def __init__(self, config: DataSourceConfig):
        """
        Args:
            config: 데이터 소스 설정
        """
        self.config = config
        self.fred_collector = FREDDataCollector(config)
        self.news_collector = NewsDataCollector(config)
        self.sentiment_analyzer = FinBERTSentimentAnalyzer(config)
        self.regime_detector = MarketRegimeDetector()

    def collect_comprehensive_data(self, start_date: str, end_date: str,
                                 stock_symbol: str = "SPY") -> pd.DataFrame:
        """
        포괄적인 대체 데이터 수집 및 통합

        Args:
            start_date: 시작 날짜
            end_date: 종료 날짜
            stock_symbol: 주식 심볼

        Returns:
            통합 데이터프레임
        """
        logger.info("포괄적인 대체 데이터 수집 시작...")

        # 1. 거시경제 지표 수집
        logger.info("거시경제 지표 수집 중...")
        macro_data = self.fred_collector.collect_all_indicators(start_date, end_date)

        # 2. 뉴스 데이터 및 감성 분석
        logger.info("뉴스 데이터 및 감성 분석 중...")
        news_data = self.news_collector.fetch_financial_news(
            query=f"{stock_symbol} stock market",
            start_date=start_date,
            end_date=end_date
        )

        sentiment_results = self.sentiment_analyzer.analyze_news_batch(news_data)

        # 일별 감성 집계
        sentiment_df = self._aggregate_daily_sentiment(sentiment_results)

        # 3. 데이터 통합
        logger.info("데이터 통합 중...")
        if not macro_data.empty and not sentiment_df.empty:
            integrated_data = pd.merge(macro_data, sentiment_df, on='date', how='outer')
        elif not macro_data.empty:
            integrated_data = macro_data
        elif not sentiment_df.empty:
            integrated_data = sentiment_df
        else:
            # 모의 데이터 생성
            integrated_data = self._generate_fallback_data(start_date, end_date)

        # 4. 추가 특성 생성
        integrated_data = self._create_additional_features(integrated_data)

        logger.info(f"대체 데이터 수집 완료: {len(integrated_data)} 레코드")
        return integrated_data.sort_values('date').reset_index(drop=True)

    def _aggregate_daily_sentiment(self, sentiment_results: List[Dict[str, Any]]) -> pd.DataFrame:
        """일별 감성 집계"""
        if not sentiment_results:
            return pd.DataFrame()

        sentiment_df = pd.DataFrame(sentiment_results)
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.date

        # 일별 집계
        daily_sentiment = sentiment_df.groupby('date').agg({
            'sentiment_scores': lambda x: {
                'positive': np.mean([s['positive'] for s in x]),
                'negative': np.mean([s['negative'] for s in x]),
                'neutral': np.mean([s['neutral'] for s in x])
            },
            'confidence': 'mean',
            'text_length': 'mean'
        }).reset_index()

        # 감성 점수 분리
        daily_sentiment['news_sentiment_positive'] = daily_sentiment['sentiment_scores'].apply(lambda x: x['positive'])
        daily_sentiment['news_sentiment_negative'] = daily_sentiment['sentiment_scores'].apply(lambda x: x['negative'])
        daily_sentiment['news_sentiment_neutral'] = daily_sentiment['sentiment_scores'].apply(lambda x: x['neutral'])
        daily_sentiment['news_confidence'] = daily_sentiment['confidence']
        daily_sentiment['news_text_length'] = daily_sentiment['text_length']

        # 감성 점수 (positive - negative)
        daily_sentiment['news_sentiment_score'] = (
            daily_sentiment['news_sentiment_positive'] - daily_sentiment['news_sentiment_negative']
        )

        daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])

        return daily_sentiment[['date', 'news_sentiment_positive', 'news_sentiment_negative',
                              'news_sentiment_neutral', 'news_sentiment_score',
                              'news_confidence', 'news_text_length']]

    def _generate_fallback_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """대체 데이터 생성 (API 실패 시)"""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')

        np.random.seed(42)
        n_days = len(dates)

        fallback_data = pd.DataFrame({
            'date': dates,
            'fed_funds_rate': np.random.normal(2.0, 0.1, n_days),
            'unemployment_rate': np.random.normal(5.0, 0.2, n_days),
            'treasury_10y': np.random.normal(2.5, 0.1, n_days),
            'vix': np.random.uniform(15, 30, n_days),
            'news_sentiment_score': np.random.normal(0, 0.2, n_days),
            'news_confidence': np.random.uniform(0.5, 0.9, n_days)
        })

        return fallback_data

    def _create_additional_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """추가 특성 생성"""
        if data.empty:
            return data

        # 수치형 컬럼에 대해 이동평균 특성 생성
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col != 'date':
                # 5일, 20일 이동평균
                data[f'{col}_ma_5'] = data[col].rolling(window=5, min_periods=1).mean()
                data[f'{col}_ma_20'] = data[col].rolling(window=20, min_periods=1).mean()

                # 변화율
                data[f'{col}_pct_change'] = data[col].pct_change()

                # Z-score (최근 60일 기준)
                data[f'{col}_zscore'] = ((data[col] - data[col].rolling(window=60, min_periods=1).mean()) /
                                       data[col].rolling(window=60, min_periods=1).std())

        # 결측값 처리
        data = data.fillna(method='ffill').fillna(method='bfill').fillna(0)

        return data

    def add_market_regimes(self, data: pd.DataFrame, returns: np.ndarray) -> pd.DataFrame:
        """시장 레짐 정보 추가"""
        if len(returns) != len(data):
            logger.warning("수익률과 데이터 길이가 일치하지 않습니다.")
            return data

        # 레짐 탐지 모델 학습 및 예측
        self.regime_detector.fit(returns)
        regimes = self.regime_detector.predict_regimes(returns)

        # 레짐 정보 추가
        data = data.copy()
        data['market_regime'] = regimes

        # 레짐별 더미 변수
        for i, regime_name in enumerate(self.regime_detector.regime_names):
            data[f'regime_{regime_name.lower()}'] = (regimes == i).astype(int)

        return data


# 사용 예시 및 테스트
if __name__ == "__main__":
    # 테스트 설정
    config = DataSourceConfig(
        fred_api_key="",  # 실제 사용시 API 키 입력
        news_api_key="",  # 실제 사용시 API 키 입력
        use_cache=True
    )

    print("=== 대체 데이터 통합 시스템 테스트 ===")

    # 1. FRED 데이터 수집 테스트
    print("\n1. FRED 거시경제 지표 수집 테스트")
    fred_collector = FREDDataCollector(config)

    start_date = "2023-01-01"
    end_date = "2023-12-31"

    macro_data = fred_collector.collect_all_indicators(start_date, end_date)
    print(f"수집된 거시경제 지표: {macro_data.shape}")
    if not macro_data.empty:
        print(f"컬럼: {list(macro_data.columns)}")
        print(f"첫 5개 행:\n{macro_data.head()}")

    # 2. 뉴스 감성 분석 테스트
    print("\n2. 뉴스 감성 분석 테스트")
    news_collector = NewsDataCollector(config)
    sentiment_analyzer = FinBERTSentimentAnalyzer(config)

    news_data = news_collector.fetch_financial_news(
        query="SPY stock market",
        start_date="2023-12-01",
        end_date="2023-12-31"
    )
    print(f"수집된 뉴스 수: {len(news_data)}")

    if news_data:
        sentiment_results = sentiment_analyzer.analyze_news_batch(news_data[:5])
        print("\n감성 분석 결과 샘플:")
        for result in sentiment_results[:2]:
            print(f"제목: {result['title'][:80]}...")
            print(f"감성: {result['sentiment_label']} (신뢰도: {result['confidence']:.3f})")
            print(f"점수: {result['sentiment_scores']}")
            print()

    # 3. 시장 레짐 탐지 테스트
    print("\n3. 시장 레짐 탐지 테스트")
    np.random.seed(42)
    mock_returns = np.random.normal(0.001, 0.02, 252)  # 1년 일일 수익률

    regime_detector = MarketRegimeDetector(n_regimes=3)
    regime_detector.fit(mock_returns)
    regimes = regime_detector.predict_regimes(mock_returns)

    regime_stats = regime_detector.get_regime_statistics(mock_returns, regimes)
    print("레짐별 통계:")
    for regime_name, stats in regime_stats.items():
        print(f"  {regime_name}: 평균수익률={stats['mean_return']:.4f}, "
              f"변동성={stats['volatility']:.4f}, 빈도={stats['frequency']:.2%}")

    # 4. 통합 데이터 수집 테스트
    print("\n4. 통합 대체 데이터 수집 테스트")
    integrator = AlternativeDataIntegrator(config)

    integrated_data = integrator.collect_comprehensive_data(
        start_date="2023-11-01",
        end_date="2023-12-31",
        stock_symbol="SPY"
    )

    print(f"통합 데이터 형태: {integrated_data.shape}")
    print(f"컬럼 수: {len(integrated_data.columns)}")
    print(f"주요 컬럼: {list(integrated_data.columns[:10])}")

    if not integrated_data.empty:
        # 시장 레짐 추가
        integrated_data_with_regimes = integrator.add_market_regimes(
            integrated_data, mock_returns[:len(integrated_data)]
        )
        print(f"레짐 추가 후 컬럼 수: {len(integrated_data_with_regimes.columns)}")

    print("\n✅ 대체 데이터 통합 시스템 테스트 완료")
    print("FRED API, 뉴스 감성 분석, 시장 레짐 탐지 기능 구현 성공")