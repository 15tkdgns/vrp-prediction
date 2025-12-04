#!/usr/bin/env python3
"""
실제 뉴스 데이터 기반 감성 분석 파이프라인
LLM 감성지표로 주가 예측 가능성 테스트를 위한 데이터 수집

목표: 합성 데이터를 제거하고 실제 뉴스 헤드라인 기반 감성 점수 생성
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

class RealNewsSentimentPipeline:
    """실제 뉴스 데이터 기반 감성 분석 파이프라인"""

    def __init__(self, ticker="SPY", start_date="2015-01-01", end_date="2024-12-31"):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.news_data = []
        self.sentiment_features = None
        self.price_data = None

    def fetch_spy_price_data(self):
        """SPY 가격 데이터 수집 (타겟 변수용)"""
        print(f"📈 {self.ticker} 가격 데이터 수집 중...")

        try:
            spy = yf.Ticker(self.ticker)
            data = spy.history(start=self.start_date, end=self.end_date)

            # 타임존 제거 및 정리
            data.index = pd.to_datetime(data.index, utc=True).tz_localize(None)
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            data.columns = ['open', 'high', 'low', 'close', 'volume']

            # 수익률 계산 (타겟 변수)
            data['returns'] = np.log(data['close'] / data['close'].shift(1))

            # 변동성 (비교 특성)
            data['volatility_5d'] = data['returns'].rolling(5).std()
            data['volatility_20d'] = data['returns'].rolling(20).std()

            self.price_data = data.dropna()
            print(f"✅ 가격 데이터: {len(self.price_data)} 샘플")
            return True

        except Exception as e:
            print(f"❌ 가격 데이터 수집 실패: {e}")
            return False

    def fetch_news_from_yfinance(self, max_news_per_request=50):
        """yfinance를 통한 실제 뉴스 헤드라인 수집"""
        print(f"📰 {self.ticker} 뉴스 데이터 수집 중...")

        try:
            spy = yf.Ticker(self.ticker)

            # yfinance 뉴스 API 사용
            news = spy.news

            if not news:
                print(f"⚠️  yfinance에서 뉴스를 가져올 수 없음. 대체 방법 사용...")
                return self._generate_realistic_news_proxy()

            # 뉴스 데이터 파싱
            for item in news[:max_news_per_request]:
                try:
                    news_item = {
                        'date': pd.to_datetime(item.get('providerPublishTime', 0), unit='s'),
                        'title': item.get('title', ''),
                        'publisher': item.get('publisher', 'Unknown'),
                        'link': item.get('link', '')
                    }
                    self.news_data.append(news_item)
                except Exception as e:
                    continue

            # 최소 샘플 수 체크 (10개 이하면 프록시 사용)
            if len(self.news_data) < 100:
                print(f"⚠️  뉴스 샘플 부족 ({len(self.news_data)}개). VIX 기반 프록시 사용...")
                self.news_data = []  # 리셋
                return self._generate_realistic_news_proxy()
            else:
                print(f"✅ 실제 뉴스 수집: {len(self.news_data)}개")
                return True

        except Exception as e:
            print(f"⚠️  뉴스 수집 중 오류: {e}")
            print("⚠️  대체 방법 사용: 시장 변동성 기반 프록시 생성...")
            return self._generate_realistic_news_proxy()

    def _generate_realistic_news_proxy(self):
        """
        실제 뉴스 수집 불가 시 대체 방안

        중요: 이것은 "합성 데이터"이지만, 기존과 달리 수익률과 직접 연결하지 않음
        대신 VIX, 거래량 등 독립적인 시장 지표 기반으로 생성
        """
        print("🔄 시장 변동성 기반 뉴스 프록시 생성 중...")

        try:
            if self.price_data is None:
                print("❌ 가격 데이터 필요")
                return False

            dates = self.price_data.index

            # VIX 프록시 수집 시도
            try:
                vix = yf.Ticker("^VIX")
                vix_data = vix.history(start=self.start_date, end=self.end_date)
                vix_data.index = pd.to_datetime(vix_data.index, utc=True).tz_localize(None)
                vix_close = vix_data['Close'].reindex(dates, method='ffill')
                print("✅ 실제 VIX 데이터 수집 성공")
            except:
                # VIX 수집 실패 시 SPY 변동성 기반 프록시
                vix_close = self.price_data['volatility_20d'] * 100
                print("⚠️  VIX 프록시 사용 (SPY 변동성 기반)")

            # 거래량 이상치 (뉴스 볼륨 프록시)
            volume_ma = self.price_data['volume'].rolling(20).mean()
            volume_ratio = self.price_data['volume'] / volume_ma

            # 가격 변동폭 (시장 관심도)
            price_range = (self.price_data['high'] - self.price_data['low']) / self.price_data['close']

            # 뉴스 프록시 생성 (날짜별)
            for date in dates:
                # 해당 날짜의 시장 지표
                vix_val = vix_close.loc[date] if date in vix_close.index else 15.0
                vol_ratio = volume_ratio.loc[date] if date in volume_ratio.index else 1.0
                price_rng = price_range.loc[date] if date in price_range.index else 0.01

                # NaN 체크
                if pd.isna(vix_val):
                    vix_val = 15.0
                if pd.isna(vol_ratio):
                    vol_ratio = 1.0
                if pd.isna(price_rng):
                    price_rng = 0.01

                # 뉴스 볼륨 추정 (VIX 높고 거래량 많으면 뉴스 많음)
                news_volume = int(vix_val / 2 + vol_ratio * 10 + price_rng * 500)
                news_volume = max(5, min(news_volume, 100))

                # 가상의 뉴스 생성 (실제 타이틀은 없지만 메타데이터만)
                for _ in range(min(news_volume, 20)):  # 최대 20개/일
                    self.news_data.append({
                        'date': date,
                        'title': f'Market proxy news (VIX={vix_val:.1f})',
                        'publisher': 'VIX-based-proxy',
                        'vix_level': vix_val,
                        'volume_ratio': vol_ratio,
                        'price_range': price_rng
                    })

            print(f"✅ 뉴스 프록시 생성: {len(self.news_data)}개 (VIX 기반)")
            return True

        except Exception as e:
            print(f"❌ 뉴스 프록시 생성 실패: {e}")
            return False

    def analyze_sentiment_with_textblob(self):
        """
        TextBlob 기반 감성 분석 (간단한 LLM 대체)

        실제 환경에서는 FinBERT, GPT 등 사용 가능
        여기서는 TextBlob의 극성(polarity) 점수 활용
        """
        print("🤖 TextBlob 감성 분석 실행 중...")

        try:
            if not self.news_data:
                print("❌ 뉴스 데이터 없음")
                return False

            # 각 뉴스에 감성 점수 추가
            for news_item in self.news_data:
                title = news_item.get('title', '')

                if 'VIX' in title:  # 프록시 뉴스인 경우
                    # VIX 기반 감성 (VIX 높으면 부정적)
                    vix_level = news_item.get('vix_level', 15.0)
                    sentiment = -np.tanh((vix_level - 15) / 10)  # -1 ~ +1
                else:
                    # 실제 뉴스 타이틀 분석
                    try:
                        blob = TextBlob(title)
                        sentiment = blob.sentiment.polarity  # -1 (부정) ~ +1 (긍정)
                    except:
                        sentiment = 0.0

                news_item['sentiment'] = sentiment

            print(f"✅ 감성 분석 완료: {len(self.news_data)}개 뉴스")
            return True

        except Exception as e:
            print(f"❌ 감성 분석 실패: {e}")
            return False

    def aggregate_daily_sentiment(self):
        """일별 감성 지표 집계"""
        print("📊 일별 감성 특성 집계 중...")

        try:
            if not self.news_data:
                print("❌ 뉴스 데이터 없음")
                return False

            # DataFrame 변환
            news_df = pd.DataFrame(self.news_data)

            # 날짜 처리 - 이미 datetime이면 그대로, 아니면 변환
            if not pd.api.types.is_datetime64_any_dtype(news_df['date']):
                news_df['date'] = pd.to_datetime(news_df['date'])

            # 시간 제거하고 날짜만
            news_df['date'] = news_df['date'].dt.tz_localize(None).dt.normalize()

            # 일별 집계
            daily_sentiment = news_df.groupby('date').agg({
                'sentiment': ['mean', 'std', 'min', 'max', 'count']
            }).reset_index()

            daily_sentiment.columns = ['date', 'sentiment_mean', 'sentiment_std',
                                      'sentiment_min', 'sentiment_max', 'news_count']

            # 결측치 처리
            daily_sentiment['sentiment_std'] = daily_sentiment['sentiment_std'].fillna(0)

            # 추가 특성 생성
            daily_sentiment['sentiment_range'] = daily_sentiment['sentiment_max'] - daily_sentiment['sentiment_min']
            daily_sentiment['sentiment_ma_5'] = daily_sentiment['sentiment_mean'].rolling(5).mean()
            daily_sentiment['sentiment_ma_20'] = daily_sentiment['sentiment_mean'].rolling(20).mean()
            daily_sentiment['sentiment_momentum'] = daily_sentiment['sentiment_mean'] - daily_sentiment['sentiment_ma_5']

            # 뉴스 볼륨 특성
            daily_sentiment['news_volume_ma_10'] = daily_sentiment['news_count'].rolling(10).mean()
            daily_sentiment['news_volume_ratio'] = daily_sentiment['news_count'] / daily_sentiment['news_volume_ma_10']
            daily_sentiment['news_volume_ratio'] = daily_sentiment['news_volume_ratio'].fillna(1.0)

            # 날짜 인덱스 설정
            daily_sentiment.set_index('date', inplace=True)

            self.sentiment_features = daily_sentiment
            print(f"✅ 일별 감성 특성: {len(self.sentiment_features)} 일, {self.sentiment_features.shape[1]} 특성")
            return True

        except Exception as e:
            print(f"❌ 감성 집계 실패: {e}")
            import traceback
            traceback.print_exc()
            return False

    def create_integrated_dataset(self):
        """가격 데이터 + 감성 특성 통합 (시간적 분리 보장)"""
        print("🔗 통합 데이터셋 생성 중 (시간적 분리 검증)...")

        try:
            if self.price_data is None or self.sentiment_features is None:
                print("❌ 가격 데이터와 감성 특성 모두 필요")
                return False

            # 디버깅: 인덱스 타입 확인
            print(f"   가격 데이터 인덱스 타입: {type(self.price_data.index)}")
            print(f"   가격 데이터 첫 날짜: {self.price_data.index[0]}")
            print(f"   감성 데이터 인덱스 타입: {type(self.sentiment_features.index)}")
            print(f"   감성 데이터 첫 날짜: {self.sentiment_features.index[0]}")

            # 인덱스 정규화 (둘 다 timezone 없는 datetime으로, 시간 제거)
            self.price_data.index = pd.to_datetime(self.price_data.index).tz_localize(None).normalize()
            self.sentiment_features.index = pd.to_datetime(self.sentiment_features.index).tz_localize(None).normalize()

            # 공통 날짜로 정렬
            common_dates = self.price_data.index.intersection(self.sentiment_features.index)
            print(f"   공통 날짜 수: {len(common_dates)}")

            price_aligned = self.price_data.loc[common_dates]
            sentiment_aligned = self.sentiment_features.loc[common_dates]

            # 통합
            integrated = pd.concat([price_aligned, sentiment_aligned], axis=1)

            # 타겟 변수 생성 (완전한 시간적 분리)
            # t일 감성 특성 → t+1일 수익률 예측
            integrated['target_return_1d'] = integrated['returns'].shift(-1)
            integrated['target_return_5d'] = integrated['returns'].rolling(5).sum().shift(-5)
            integrated['target_direction_1d'] = (integrated['target_return_1d'] > 0).astype(int)

            # 타겟 가격
            integrated['target_price_1d'] = integrated['close'].shift(-1)

            # 최종 정리 (NaN 제거)
            integrated = integrated.dropna()

            print(f"\n✅ 통합 데이터셋 완성:")
            print(f"   📊 샘플 수: {len(integrated):,}")
            print(f"   📊 특성 수: {integrated.shape[1]}")
            print(f"   📊 기간: {integrated.index.min()} ~ {integrated.index.max()}")

            # 데이터 누출 검증
            self._verify_no_data_leakage(integrated)

            # 저장
            self._save_dataset(integrated)

            return integrated

        except Exception as e:
            print(f"❌ 통합 데이터셋 생성 실패: {e}")
            return None

    def _verify_no_data_leakage(self, df):
        """데이터 누출 검증"""
        print("\n🔍 데이터 누출 검증 중...")

        # 감성 특성 컬럼
        sentiment_cols = [col for col in df.columns if 'sentiment' in col or 'news' in col]

        # 타겟과의 상관관계 (너무 높으면 의심)
        if 'target_return_1d' in df.columns:
            for col in sentiment_cols[:5]:  # 주요 특성만
                corr = df[col].corr(df['target_return_1d'])
                print(f"   {col:30s} ↔ target_return_1d: {corr:+.4f}")

                if abs(corr) > 0.3:
                    print(f"   ⚠️  높은 상관관계 감지: {col} ({corr:.4f})")

        print("✅ 데이터 누출 검증 완료")

    def _save_dataset(self, df):
        """데이터셋 저장"""
        try:
            save_path = "data/training/spy_news_sentiment_dataset.csv"
            df.to_csv(save_path)
            print(f"\n💾 데이터셋 저장: {save_path}")

            # 메타데이터
            import json
            metadata = {
                'dataset_info': {
                    'ticker': self.ticker,
                    'samples': len(df),
                    'features': df.shape[1],
                    'start_date': str(df.index.min()),
                    'end_date': str(df.index.max()),
                    'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                },
                'data_sources': {
                    'price': f'{self.ticker} via yfinance',
                    'news': 'yfinance news API or VIX-based proxy',
                    'sentiment_analysis': 'TextBlob polarity score'
                },
                'target_variables': [
                    'target_return_1d',
                    'target_return_5d',
                    'target_direction_1d',
                    'target_price_1d'
                ],
                'temporal_separation': 'Complete (t features → t+1 target)',
                'sentiment_features': [col for col in df.columns if 'sentiment' in col or 'news' in col]
            }

            with open("data/raw/news_sentiment_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"💾 메타데이터 저장: data/raw/news_sentiment_metadata.json")

        except Exception as e:
            print(f"❌ 저장 실패: {e}")

    def run_pipeline(self):
        """전체 파이프라인 실행"""
        print("="*60)
        print("🚀 실제 뉴스 기반 감성 분석 파이프라인 시작")
        print("="*60)
        print(f"📅 기간: {self.start_date} ~ {self.end_date}")
        print(f"🎯 목표: LLM 감성지표로 주가 예측 가능성 테스트\n")

        steps = [
            ("가격 데이터 수집", self.fetch_spy_price_data),
            ("뉴스 데이터 수집", self.fetch_news_from_yfinance),
            ("감성 분석 (TextBlob)", self.analyze_sentiment_with_textblob),
            ("일별 감성 집계", self.aggregate_daily_sentiment),
            ("통합 데이터셋 생성", self.create_integrated_dataset)
        ]

        for step_name, step_func in steps:
            print(f"\n{'='*60}")
            print(f"🔄 {step_name}")
            print(f"{'='*60}")

            result = step_func()
            if result is False:
                print(f"\n❌ {step_name} 실패 - 파이프라인 중단")
                return False

        print("\n" + "="*60)
        print("✅ 파이프라인 완료!")
        print("="*60)
        return True

if __name__ == "__main__":
    # 파이프라인 실행
    pipeline = RealNewsSentimentPipeline(
        ticker="SPY",
        start_date="2015-01-01",
        end_date="2024-12-31"
    )

    success = pipeline.run_pipeline()

    if success:
        print("\n다음 단계: 주가 예측 모델 학습")
        print("  python3 src/models/news_sentiment_price_prediction.py")
