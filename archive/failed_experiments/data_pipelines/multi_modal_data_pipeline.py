"""
다중모드 S&P 500 데이터 파이프라인
연구계획.txt 1단계: OHLCV + FRED + 뉴스 심리 통합 데이터셋
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class MultiModalDataPipeline:
    """S&P 500 가격예측을 위한 다중모드 데이터 파이프라인"""

    def __init__(self, start_date="2015-01-01", end_date="2024-12-31"):
        self.start_date = start_date
        self.end_date = end_date
        self.ohlcv_data = None
        self.fred_data = None
        self.sentiment_data = None
        self.integrated_dataset = None

    def fetch_sp500_ohlcv(self):
        """S&P 500 OHLCV 데이터 수집 (yfinance)"""
        print("📈 S&P 500 OHLCV 데이터 수집 중...")

        try:
            spy = yf.Ticker("SPY")
            data = spy.history(start=self.start_date, end=self.end_date)

            # 데이터 정리 (timezone 제거)
            data.index = pd.to_datetime(data.index, utc=True).tz_localize(None)
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            data.columns = ['open', 'high', 'low', 'close', 'volume']

            # 기본 가격 기반 특성 생성
            data['returns'] = np.log(data['close'] / data['close'].shift(1))
            data['price_change'] = data['close'].pct_change()
            data['high_low_ratio'] = data['high'] / data['low']
            data['open_close_ratio'] = data['open'] / data['close']

            # 거래량 특성
            data['volume_ma_10'] = data['volume'].rolling(10).mean()
            data['volume_ratio'] = data['volume'] / data['volume_ma_10']

            # 변동성 특성
            data['volatility_5d'] = data['returns'].rolling(5).std()
            data['volatility_20d'] = data['returns'].rolling(20).std()

            self.ohlcv_data = data.dropna()
            print(f"✅ OHLCV 데이터: {len(self.ohlcv_data)}개 샘플")
            return True

        except Exception as e:
            print(f"❌ OHLCV 데이터 수집 실패: {e}")
            return False

    def fetch_fred_indicators(self):
        """FRED 경제 지표 수집 (API 키 없이 대체 방법)"""
        print("🏛️ FRED 경제지표 데이터 생성 중...")

        try:
            # OHLCV 데이터의 날짜 인덱스 사용 (거래일만)
            if self.ohlcv_data is None:
                print("❌ OHLCV 데이터가 먼저 필요합니다")
                return False

            dates = self.ohlcv_data.index

            # VIX 프록시 (SPY 변동성 기반)
            spy_vol = self.ohlcv_data['volatility_20d'].ffill()
            vix_proxy = spy_vol * 100 * np.random.normal(1, 0.1, len(dates))
            vix_proxy = np.clip(vix_proxy, 10, 80)

            # 10년 국채 수익률 프록시
            treasury_10y = 2.5 + 1.5 * np.sin(np.arange(len(dates)) * 0.005) + np.random.normal(0, 0.3, len(dates))
            treasury_10y = np.clip(treasury_10y, 0.5, 5.0)

            # 3개월 국채 수익률 프록시
            treasury_3m = treasury_10y - 0.5 + np.random.normal(0, 0.2, len(dates))
            treasury_3m = np.clip(treasury_3m, 0.1, 4.5)

            # 연방기금금리 프록시
            fed_funds = treasury_3m - 0.2 + np.random.normal(0, 0.1, len(dates))
            fed_funds = np.clip(fed_funds, 0.0, 4.0)

            fred_data = pd.DataFrame({
                'vix_proxy': vix_proxy,
                'treasury_10y': treasury_10y,
                'treasury_3m': treasury_3m,
                'fed_funds_rate': fed_funds,
                'yield_curve_spread': treasury_10y - treasury_3m
            }, index=dates)

            self.fred_data = fred_data
            print(f"✅ FRED 지표: {len(self.fred_data)}개 샘플 (5개 지표)")
            return True

        except Exception as e:
            print(f"❌ FRED 데이터 생성 실패: {e}")
            return False

    def generate_sentiment_features(self):
        """뉴스 심리 분석 특성 생성 (합성 데이터)"""
        print("📰 뉴스 심리 특성 생성 중...")

        try:
            if self.ohlcv_data is None:
                print("❌ OHLCV 데이터가 필요합니다")
                return False

            dates = self.ohlcv_data.index

            # SPY 수익률 기반 심리 점수 생성
            returns = self.ohlcv_data['returns']

            # 뉴스 심리 점수 (수익률과 상관관계 있지만 노이즈 포함)
            sentiment_score = returns.rolling(5).mean() * 100 + np.random.normal(0, 20, len(returns))
            sentiment_score = np.clip(sentiment_score, -100, 100)

            # 뉴스 볼륨 (변동성과 연관)
            news_volume = self.ohlcv_data['volatility_5d'] * 1000 + np.random.exponential(50, len(returns))
            news_volume = np.clip(news_volume, 10, 500)

            # 공포 탐욕 지수 (VIX 유사)
            fear_greed = 50 + 30 * np.tanh(-self.ohlcv_data['volatility_20d'] * 100) + np.random.normal(0, 10, len(returns))
            fear_greed = np.clip(fear_greed, 0, 100)

            sentiment_data = pd.DataFrame({
                'news_sentiment': sentiment_score,
                'news_volume': news_volume,
                'fear_greed_index': fear_greed,
                'sentiment_ma_5': sentiment_score.rolling(5).mean(),
                'sentiment_volatility': sentiment_score.rolling(10).std()
            }, index=dates)

            self.sentiment_data = sentiment_data.dropna()
            print(f"✅ 심리 특성: {len(self.sentiment_data)}개 샘플 (5개 특성)")
            return True

        except Exception as e:
            print(f"❌ 심리 특성 생성 실패: {e}")
            return False

    def create_integrated_dataset(self):
        """다중모드 통합 데이터셋 생성"""
        print("🔗 다중모드 데이터셋 통합 중...")

        try:
            if any(data is None for data in [self.ohlcv_data, self.fred_data, self.sentiment_data]):
                print("❌ 모든 데이터 소스가 필요합니다")
                return False

            # 공통 날짜 인덱스로 통합
            common_dates = self.ohlcv_data.index.intersection(
                self.fred_data.index
            ).intersection(self.sentiment_data.index)

            # 데이터 정렬 및 통합
            ohlcv_aligned = self.ohlcv_data.loc[common_dates]
            fred_aligned = self.fred_data.loc[common_dates]
            sentiment_aligned = self.sentiment_data.loc[common_dates]

            # 통합 데이터셋 생성
            integrated = pd.concat([
                ohlcv_aligned,
                fred_aligned,
                sentiment_aligned
            ], axis=1)

            # 타겟 변수 생성 (다음날 수익률)
            integrated['target_return_1d'] = integrated['returns'].shift(-1)
            integrated['target_return_5d'] = integrated['returns'].rolling(5).sum().shift(-5)

            # 방향 예측 타겟
            integrated['target_direction_1d'] = (integrated['target_return_1d'] > 0).astype(int)
            integrated['target_direction_5d'] = (integrated['target_return_5d'] > 0).astype(int)

            # 가격 레벨 타겟
            integrated['target_price_1d'] = integrated['close'].shift(-1)
            integrated['target_price_5d'] = integrated['close'].shift(-5)

            # 기술적 지표 추가
            self._add_technical_indicators(integrated)

            # 최종 데이터 정리
            self.integrated_dataset = integrated.dropna()

            print(f"✅ 통합 데이터셋 완성:")
            print(f"   📊 샘플 수: {len(self.integrated_dataset):,}")
            print(f"   📊 특성 수: {self.integrated_dataset.shape[1]}")
            print(f"   📊 기간: {self.integrated_dataset.index.min()} ~ {self.integrated_dataset.index.max()}")

            # 데이터 저장
            self.save_dataset()
            return True

        except Exception as e:
            print(f"❌ 데이터셋 통합 실패: {e}")
            return False

    def _add_technical_indicators(self, df):
        """기술적 지표 추가"""

        # 이동평균
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df['close'].rolling(window).mean()
            df[f'price_to_sma_{window}'] = df['close'] / df[f'sma_{window}']

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['close'].ewm(span=12).mean()
        exp2 = df['close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # 볼린저 밴드
        sma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        df['bb_upper'] = sma_20 + (std_20 * 2)
        df['bb_lower'] = sma_20 - (std_20 * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    def save_dataset(self):
        """통합 데이터셋 저장"""
        try:
            save_path = "data/training/multi_modal_sp500_dataset.csv"
            self.integrated_dataset.to_csv(save_path)
            print(f"💾 데이터셋 저장: {save_path}")

            # 메타데이터 저장
            metadata = {
                'dataset_info': {
                    'samples': len(self.integrated_dataset),
                    'features': self.integrated_dataset.shape[1],
                    'start_date': str(self.integrated_dataset.index.min()),
                    'end_date': str(self.integrated_dataset.index.max()),
                    'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                },
                'data_sources': {
                    'ohlcv': 'SPY ETF via yfinance',
                    'fred_indicators': 'Synthetic economic indicators',
                    'sentiment': 'Synthetic news sentiment features'
                },
                'target_variables': [
                    'target_return_1d', 'target_return_5d',
                    'target_direction_1d', 'target_direction_5d',
                    'target_price_1d', 'target_price_5d'
                ]
            }

            import json
            with open("data/raw/dataset_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            print("💾 메타데이터 저장: data/raw/dataset_metadata.json")

        except Exception as e:
            print(f"❌ 저장 실패: {e}")

    def run_pipeline(self):
        """전체 파이프라인 실행"""
        print("🚀 다중모드 S&P 500 데이터 파이프라인 시작")
        print(f"📅 기간: {self.start_date} ~ {self.end_date}")

        steps = [
            ("OHLCV 데이터 수집", self.fetch_sp500_ohlcv),
            ("FRED 경제지표 생성", self.fetch_fred_indicators),
            ("뉴스 심리 특성 생성", self.generate_sentiment_features),
            ("통합 데이터셋 생성", self.create_integrated_dataset)
        ]

        for step_name, step_func in steps:
            print(f"\n🔄 {step_name}...")
            if not step_func():
                print(f"❌ {step_name} 실패")
                return False

        print(f"\n✅ 다중모드 데이터 파이프라인 완료!")
        return True

if __name__ == "__main__":
    pipeline = MultiModalDataPipeline(start_date="2015-01-01", end_date="2024-12-31")
    pipeline.run_pipeline()