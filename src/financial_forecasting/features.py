"""
Alternative Data Integration for Financial Forecasting

거시경제 지표, 뉴스 센티멘트, 시장 레짐 감지를 통한 고급 특성 생성:
1. AlternativeDataIntegrator: FRED API 거시경제 지표 통합
2. SentimentAnalyzer: FinBERT 기반 뉴스 센티멘트 분석
3. MarketRegimeDetector: HMM 기반 시장 레짐(강세/약세) 감지
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

try:
    import fredapi
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False
    print("⚠️ fredapi not available, using simulated macro indicators")

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False
    print("⚠️ transformers not available, using simplified sentiment analysis")

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("⚠️ hmmlearn not available, using simplified regime detection")

try:
    import requests
    from datetime import datetime, timedelta
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("⚠️ requests not available, news fetching disabled")


@dataclass
class MacroIndicators:
    """거시경제 지표 데이터"""
    fed_rate: float
    unemployment: float
    cpi_inflation: float
    gdp_growth: float
    vix: float
    yield_curve_10y2y: float
    dollar_index: float
    oil_price: float


@dataclass
class SentimentResult:
    """뉴스 센티멘트 분석 결과"""
    sentiment_score: float  # -1 (매우 부정) ~ +1 (매우 긍정)
    confidence: float       # 0 ~ 1
    positive_ratio: float   # 긍정 뉴스 비율
    negative_ratio: float   # 부정 뉴스 비율
    news_volume: int        # 뉴스 기사 수


@dataclass
class MarketRegime:
    """시장 레짐 감지 결과"""
    regime_id: int          # 0: 약세장, 1: 횡보장, 2: 강세장
    regime_name: str        # "Bear", "Sideways", "Bull"
    probability: float      # 해당 레짐일 확률
    volatility_state: str   # "Low", "Medium", "High"


class AlternativeDataIntegrator:
    """
    FRED API를 통한 거시경제 지표 통합

    금융 시계열 예측에 중요한 거시경제 변수들:
    - 연준 기준금리 (Federal Funds Rate)
    - 실업률 (Unemployment Rate)
    - 소비자물가지수 (CPI)
    - GDP 성장률
    - VIX 공포지수
    - 수익률 곡선 (10Y-2Y Spread)
    - 달러 지수 (DXY)
    - 원유 가격 (WTI)
    """

    def __init__(self, fred_api_key: Optional[str] = None):
        self.fred_api_key = fred_api_key
        self.fred = None

        if FRED_AVAILABLE and fred_api_key:
            try:
                self.fred = fredapi.Fred(api_key=fred_api_key)
                print("✅ FRED API 연결 성공")
            except Exception as e:
                print(f"⚠️ FRED API 연결 실패: {e}")
                self.fred = None

    def fetch_macro_indicators(
        self,
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        거시경제 지표 수집

        Args:
            start_date: 시작 날짜
            end_date: 종료 날짜

        Returns:
            거시경제 지표 DataFrame
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        if not self.fred:
            return self._generate_simulated_macro_data(start_date, end_date)

        try:
            # FRED 시리즈 ID 매핑
            series_mapping = {
                'fed_rate': 'FEDFUNDS',           # 연준 기준금리
                'unemployment': 'UNRATE',         # 실업률
                'cpi_inflation': 'CPIAUCSL',      # 소비자물가지수
                'gdp_growth': 'GDP',              # GDP
                'vix': 'VIXCLS',                  # VIX 지수
                'yield_10y': 'DGS10',             # 10년 국채 수익률
                'yield_2y': 'DGS2',               # 2년 국채 수익률
                'dollar_index': 'DEXUSEU',        # 달러/유로 환율
                'oil_price': 'DCOILWTICO'         # WTI 원유 가격
            }

            macro_data = {}

            for indicator, series_id in series_mapping.items():
                try:
                    data = self.fred.get_series(
                        series_id,
                        start=start_date,
                        end=end_date
                    )
                    macro_data[indicator] = data
                    print(f"✅ {indicator} 데이터 수집 완료 ({len(data)}개)")
                except Exception as e:
                    print(f"⚠️ {indicator} 수집 실패: {e}")
                    # 실패한 지표는 시뮬레이션 데이터로 대체
                    dates = pd.date_range(start=start_date, end=end_date, freq='D')
                    macro_data[indicator] = pd.Series(
                        np.random.normal(0, 1, len(dates)),
                        index=dates
                    )

            # 수익률 곡선 스프레드 계산
            if 'yield_10y' in macro_data and 'yield_2y' in macro_data:
                macro_data['yield_curve_10y2y'] = (
                    macro_data['yield_10y'] - macro_data['yield_2y']
                )

            # DataFrame으로 결합
            macro_df = pd.DataFrame(macro_data)
            macro_df = macro_df.dropna()

            # 전년 동월 대비 성장률 계산 (GDP, CPI)
            if 'gdp_growth' in macro_df.columns:
                macro_df['gdp_growth'] = macro_df['gdp_growth'].pct_change(252)  # 연간 성장률

            if 'cpi_inflation' in macro_df.columns:
                macro_df['cpi_inflation'] = macro_df['cpi_inflation'].pct_change(252)  # 연간 인플레이션

            print(f"✅ 거시경제 지표 통합 완료: {len(macro_df)}개 관측치, {len(macro_df.columns)}개 지표")
            return macro_df

        except Exception as e:
            print(f"⚠️ FRED 데이터 수집 실패: {e}")
            return self._generate_simulated_macro_data(start_date, end_date)

    def _generate_simulated_macro_data(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """FRED API 없을 때 시뮬레이션 거시경제 데이터 생성"""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n_days = len(dates)

        # 실제 경제 사이클과 유사한 패턴 생성
        time_trend = np.arange(n_days) / 252  # 연 단위 변환

        macro_data = {
            'fed_rate': 2.0 + 1.5 * np.sin(time_trend * 2 * np.pi / 4) + np.random.normal(0, 0.1, n_days),
            'unemployment': 5.5 + 2.0 * np.sin(time_trend * 2 * np.pi / 6) + np.random.normal(0, 0.2, n_days),
            'cpi_inflation': 0.02 + 0.01 * np.sin(time_trend * 2 * np.pi / 3) + np.random.normal(0, 0.005, n_days),
            'gdp_growth': 0.025 + 0.015 * np.sin(time_trend * 2 * np.pi / 5) + np.random.normal(0, 0.01, n_days),
            'vix': 20 + 10 * np.sin(time_trend * 2 * np.pi / 2) + np.random.normal(0, 2, n_days),
            'yield_curve_10y2y': 1.5 + 1.0 * np.sin(time_trend * 2 * np.pi / 8) + np.random.normal(0, 0.1, n_days),
            'dollar_index': 100 + 5 * np.sin(time_trend * 2 * np.pi / 3) + np.random.normal(0, 0.5, n_days),
            'oil_price': 70 + 20 * np.sin(time_trend * 2 * np.pi / 2) + np.random.normal(0, 2, n_days)
        }

        macro_df = pd.DataFrame(macro_data, index=dates)
        print(f"✅ 시뮬레이션 거시경제 데이터 생성: {len(macro_df)}개 관측치")
        return macro_df


class SentimentAnalyzer:
    """
    FinBERT 기반 뉴스 센티멘트 분석

    금융 도메인 특화 BERT 모델을 사용하여:
    - 뉴스 기사의 감정 점수 계산
    - 시장에 대한 전반적 센티멘트 측정
    - 센티멘트 변화 추세 감지
    """

    def __init__(self):
        self.tokenizer = None
        self.model = None

        if FINBERT_AVAILABLE:
            try:
                model_name = "ProsusAI/finbert"
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                print("✅ FinBERT 모델 로드 성공")
            except Exception as e:
                print(f"⚠️ FinBERT 로드 실패: {e}")
                self.tokenizer = None
                self.model = None

    def analyze_text_sentiment(self, text: str) -> Tuple[float, float]:
        """
        단일 텍스트의 센티멘트 분석

        Args:
            text: 분석할 텍스트

        Returns:
            (sentiment_score, confidence)
        """
        if not self.model or not self.tokenizer:
            # 간단한 키워드 기반 센티멘트 분석
            return self._simple_sentiment_analysis(text)

        try:
            # 텍스트 토큰화
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )

            # 예측 수행
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # FinBERT는 [negative, neutral, positive] 순서
            negative_prob = predictions[0][0].item()
            neutral_prob = predictions[0][1].item()
            positive_prob = predictions[0][2].item()

            # 센티멘트 점수 계산 (-1 ~ +1)
            sentiment_score = positive_prob - negative_prob
            confidence = max(positive_prob, negative_prob, neutral_prob)

            return sentiment_score, confidence

        except Exception as e:
            print(f"⚠️ FinBERT 분석 실패: {e}")
            return self._simple_sentiment_analysis(text)

    def _simple_sentiment_analysis(self, text: str) -> Tuple[float, float]:
        """간단한 키워드 기반 센티멘트 분석"""
        positive_words = [
            'bull', 'bullish', 'rise', 'gain', 'profit', 'up', 'increase',
            'strong', 'buy', 'optimistic', 'positive', 'growth', 'rally'
        ]

        negative_words = [
            'bear', 'bearish', 'fall', 'loss', 'down', 'decrease', 'decline',
            'weak', 'sell', 'pessimistic', 'negative', 'crash', 'correction'
        ]

        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        total_words = len(text.split())
        sentiment_score = (positive_count - negative_count) / max(total_words, 1)
        confidence = min((positive_count + negative_count) / max(total_words, 1), 1.0)

        return np.clip(sentiment_score, -1, 1), confidence

    def analyze_news_batch(
        self,
        news_articles: List[str],
        weights: Optional[List[float]] = None
    ) -> SentimentResult:
        """
        뉴스 기사 배치 센티멘트 분석

        Args:
            news_articles: 뉴스 기사 목록
            weights: 각 기사의 가중치 (중요도)

        Returns:
            종합 센티멘트 결과
        """
        if not news_articles:
            return SentimentResult(0.0, 0.0, 0.0, 0.0, 0)

        if weights is None:
            weights = [1.0] * len(news_articles)

        sentiments = []
        confidences = []

        for article, weight in zip(news_articles, weights):
            sentiment, confidence = self.analyze_text_sentiment(article)
            sentiments.append(sentiment * weight)
            confidences.append(confidence * weight)

        # 가중 평균 계산
        total_weight = sum(weights)
        avg_sentiment = sum(sentiments) / total_weight
        avg_confidence = sum(confidences) / total_weight

        # 긍정/부정 비율 계산
        positive_count = sum(1 for s in sentiments if s > 0)
        negative_count = sum(1 for s in sentiments if s < 0)

        positive_ratio = positive_count / len(sentiments)
        negative_ratio = negative_count / len(sentiments)

        return SentimentResult(
            sentiment_score=avg_sentiment,
            confidence=avg_confidence,
            positive_ratio=positive_ratio,
            negative_ratio=negative_ratio,
            news_volume=len(news_articles)
        )

    def generate_sentiment_time_series(
        self,
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
        simulate: bool = True
    ) -> pd.DataFrame:
        """센티멘트 시계열 데이터 생성 (시뮬레이션 또는 실제 뉴스)"""
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n_days = len(dates)

        if simulate:
            # 시뮬레이션 센티멘트 데이터 (실제 시장 사이클과 유사)
            time_trend = np.arange(n_days) / 252
            base_sentiment = 0.1 * np.sin(time_trend * 2 * np.pi / 2)  # 2년 주기
            noise = np.random.normal(0, 0.05, n_days)

            sentiment_data = {
                'sentiment_score': base_sentiment + noise,
                'confidence': 0.7 + 0.2 * np.random.random(n_days),
                'positive_ratio': 0.4 + 0.3 * np.random.random(n_days),
                'negative_ratio': 0.3 + 0.3 * np.random.random(n_days),
                'news_volume': np.random.poisson(20, n_days)
            }

            sentiment_df = pd.DataFrame(sentiment_data, index=dates)
            print(f"✅ 시뮬레이션 센티멘트 데이터 생성: {len(sentiment_df)}개 관측치")
            return sentiment_df

        else:
            # 실제 뉴스 API를 통한 센티멘트 분석 (향후 구현)
            print("⚠️ 실제 뉴스 API 연동은 향후 구현 예정")
            return self.generate_sentiment_time_series(start_date, end_date, simulate=True)


class MarketRegimeDetector:
    """
    HMM(Hidden Markov Model) 기반 시장 레짐 감지

    시장의 숨겨진 상태를 감지하여:
    - 강세장 (Bull Market): 지속적 상승 트렌드
    - 약세장 (Bear Market): 지속적 하락 트렌드
    - 횡보장 (Sideways Market): 방향성 없는 변동
    """

    def __init__(self, n_regimes: int = 3):
        self.n_regimes = n_regimes
        self.model = None
        self.regime_names = ["Bear", "Sideways", "Bull"]
        self.is_fitted = False

    def fit(self, returns: pd.Series, volatility: pd.Series) -> None:
        """
        HMM 모델 학습

        Args:
            returns: 수익률 시계열
            volatility: 변동성 시계열
        """
        if not HMM_AVAILABLE:
            print("⚠️ hmmlearn 없어서 간단한 레짐 감지 사용")
            self._fit_simple_regime_detector(returns, volatility)
            return

        try:
            # 특성 행렬 구성 (수익률, 변동성)
            features = np.column_stack([
                returns.values,
                volatility.values
            ])

            # 결측치 제거
            mask = ~np.isnan(features).any(axis=1)
            features_clean = features[mask]

            # 정규화
            features_normalized = (features_clean - features_clean.mean(axis=0)) / features_clean.std(axis=0)

            # GaussianHMM 모델 생성
            self.model = hmm.GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="full",
                n_iter=100,
                random_state=42
            )

            # 모델 학습
            self.model.fit(features_normalized)
            self.is_fitted = True

            print(f"✅ HMM 레짐 감지 모델 학습 완료 ({self.n_regimes}개 레짐)")

        except Exception as e:
            print(f"⚠️ HMM 학습 실패: {e}")
            self._fit_simple_regime_detector(returns, volatility)

    def _fit_simple_regime_detector(self, returns: pd.Series, volatility: pd.Series) -> None:
        """간단한 규칙 기반 레짐 감지기"""
        # 이동평균 기반 레짐 분류 기준점 계산
        ma_short = returns.rolling(window=20).mean()
        ma_long = returns.rolling(window=60).mean()
        vol_ma = volatility.rolling(window=30).mean()

        self.simple_thresholds = {
            'bull_return': ma_long.quantile(0.7),
            'bear_return': ma_long.quantile(0.3),
            'high_vol': vol_ma.quantile(0.7)
        }

        self.is_fitted = True
        print("✅ 간단한 레짐 감지기 설정 완료")

    def predict_regime(
        self,
        returns: pd.Series,
        volatility: pd.Series
    ) -> pd.DataFrame:
        """
        시장 레짐 예측

        Args:
            returns: 수익률 시계열
            volatility: 변동성 시계열

        Returns:
            레짐 예측 결과 DataFrame
        """
        if not self.is_fitted:
            self.fit(returns, volatility)

        if self.model is None:
            return self._predict_simple_regime(returns, volatility)

        try:
            # 특성 행렬 구성
            features = np.column_stack([
                returns.values,
                volatility.values
            ])

            # 결측치 처리
            mask = ~np.isnan(features).any(axis=1)

            # 정규화
            features_normalized = (features - features.mean(axis=0)) / features.std(axis=0)

            # 레짐 예측 및 확률 계산
            regimes = self.model.predict(features_normalized)
            regime_probs = self.model.predict_proba(features_normalized)

            # 결과 DataFrame 구성
            results = []
            for i, (regime, probs) in enumerate(zip(regimes, regime_probs)):
                if mask[i]:  # 유효한 데이터만
                    vol_state = "Low" if volatility.iloc[i] < volatility.quantile(0.33) else \
                                "Medium" if volatility.iloc[i] < volatility.quantile(0.67) else "High"

                    results.append({
                        'regime_id': regime,
                        'regime_name': self.regime_names[regime],
                        'probability': probs[regime],
                        'volatility_state': vol_state
                    })
                else:
                    # 결측치는 이전 값으로 채움
                    if results:
                        results.append(results[-1].copy())
                    else:
                        results.append({
                            'regime_id': 1,
                            'regime_name': 'Sideways',
                            'probability': 0.5,
                            'volatility_state': 'Medium'
                        })

            regime_df = pd.DataFrame(results, index=returns.index)
            print(f"✅ HMM 레짐 예측 완료: {len(regime_df)}개 관측치")
            return regime_df

        except Exception as e:
            print(f"⚠️ HMM 예측 실패: {e}")
            return self._predict_simple_regime(returns, volatility)

    def _predict_simple_regime(
        self,
        returns: pd.Series,
        volatility: pd.Series
    ) -> pd.DataFrame:
        """간단한 규칙 기반 레짐 예측"""

        # 이동평균 기반 분류
        ma_short = returns.rolling(window=20).mean()
        ma_long = returns.rolling(window=60).mean()

        regimes = []
        for i in range(len(returns)):
            if ma_long.iloc[i] > self.simple_thresholds['bull_return']:
                regime_id, regime_name = 2, "Bull"
            elif ma_long.iloc[i] < self.simple_thresholds['bear_return']:
                regime_id, regime_name = 0, "Bear"
            else:
                regime_id, regime_name = 1, "Sideways"

            vol_state = "Low" if volatility.iloc[i] < volatility.quantile(0.33) else \
                       "Medium" if volatility.iloc[i] < volatility.quantile(0.67) else "High"

            regimes.append({
                'regime_id': regime_id,
                'regime_name': regime_name,
                'probability': 0.7,  # 간단한 모델이므로 고정 확률
                'volatility_state': vol_state
            })

        regime_df = pd.DataFrame(regimes, index=returns.index)
        print(f"✅ 간단한 레짐 예측 완료: {len(regime_df)}개 관측치")
        return regime_df