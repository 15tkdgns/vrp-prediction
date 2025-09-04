#!/usr/bin/env python3
"""
뉴스 감정 분석과 주식 데이터를 통합하는 모듈
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsStockIntegration:
    """뉴스 감정 분석과 SPY 주식 데이터를 통합하는 클래스"""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = data_dir
        self.spy_data = None
        self.news_data = None
        self.integrated_data = None
        
    def load_spy_data(self) -> pd.DataFrame:
        """SPY 주식 데이터 로드"""
        spy_file = os.path.join(self.data_dir, "spy_2025_h1.json")
        
        if not os.path.exists(spy_file):
            raise FileNotFoundError(f"SPY data file not found: {spy_file}")
            
        logger.info(f"Loading SPY data from {spy_file}")
        
        with open(spy_file, 'r') as f:
            spy_json = json.load(f)
            
        # DataFrame으로 변환
        spy_df = pd.DataFrame(spy_json['data'])
        spy_df['date'] = pd.to_datetime(spy_df['date'])
        spy_df = spy_df.sort_values('date')
        
        # 기본적인 기술적 지표 계산
        spy_df = self._calculate_technical_indicators(spy_df)
        
        logger.info(f"Loaded {len(spy_df)} SPY records")
        self.spy_data = spy_df
        return spy_df
    
    def load_news_data(self) -> pd.DataFrame:
        """뉴스 감정 분석 데이터 로드"""
        # 시계열 데이터 로드
        timeseries_file = os.path.join(self.data_dir, "sentiment_timeseries.json")
        summary_file = os.path.join(self.data_dir, "news_sentiment_summary.json")
        
        if not os.path.exists(timeseries_file):
            raise FileNotFoundError(f"News timeseries file not found: {timeseries_file}")
        if not os.path.exists(summary_file):
            raise FileNotFoundError(f"News summary file not found: {summary_file}")
            
        logger.info(f"Loading news data from {timeseries_file} and {summary_file}")
        
        # 시계열 데이터
        with open(timeseries_file, 'r') as f:
            timeseries = json.load(f)
            
        # 요약 데이터
        with open(summary_file, 'r') as f:
            summary = json.load(f)
            
        # 시계열 데이터를 DataFrame으로 변환
        news_df = pd.DataFrame({
            'date': pd.to_datetime(timeseries['dates']),
            'sentiment_score': timeseries['sentiment_scores'],
            'article_count': timeseries['article_counts']
        })
        
        # 요약 데이터에서 추가 특징 추출
        news_df = self._add_news_features(news_df, summary)
        
        logger.info(f"Loaded {len(news_df)} news records")
        self.news_data = news_df
        return news_df
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표 계산"""
        df = df.copy()
        
        # 가격 변화
        df['price_change'] = df['close'].pct_change()
        df['price_change_abs'] = df['price_change'].abs()
        
        # 볼륨 변화
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma_5'] = df['volume'].rolling(5).mean()
        df['unusual_volume'] = (df['volume'] > df['volume_ma_5'] * 1.5).astype(int)
        
        # 이동평균
        for period in [5, 10, 20]:
            df[f'ma_{period}'] = df['close'].rolling(period).mean()
            df[f'price_to_ma{period}'] = df['close'] / df[f'ma_{period}']
        
        # RSI 계산
        df['rsi'] = self._calculate_rsi(df['close'])
        
        # 볼린저 밴드
        df = self._calculate_bollinger_bands(df)
        
        # MACD
        df = self._calculate_macd(df)
        
        # 변동성
        df['volatility_20'] = df['price_change'].rolling(20).std()
        df['volatility_5'] = df['price_change'].rolling(5).std()
        
        # 가격 스파이크 감지
        df['price_spike'] = (df['price_change_abs'] > df['volatility_20'] * 2).astype(int)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame, window: int = 20, std_dev: int = 2) -> pd.DataFrame:
        """볼린저 밴드 계산"""
        df = df.copy()
        df[f'bb_middle'] = df['close'].rolling(window).mean()
        bb_std = df['close'].rolling(window).std()
        df[f'bb_upper'] = df[f'bb_middle'] + (bb_std * std_dev)
        df[f'bb_lower'] = df[f'bb_middle'] - (bb_std * std_dev)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        return df
    
    def _calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """MACD 계산"""
        df = df.copy()
        ema_fast = df['close'].ewm(span=fast).mean()
        ema_slow = df['close'].ewm(span=slow).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        return df
    
    def _add_news_features(self, news_df: pd.DataFrame, summary_data: Dict) -> pd.DataFrame:
        """뉴스 데이터에 추가 특징 생성"""
        news_df = news_df.copy()
        
        # 뉴스 감정 점수 특징
        news_df['sentiment_positive'] = (news_df['sentiment_score'] > 0.1).astype(int)
        news_df['sentiment_negative'] = (news_df['sentiment_score'] < -0.1).astype(int)
        news_df['sentiment_neutral'] = ((news_df['sentiment_score'] >= -0.1) & (news_df['sentiment_score'] <= 0.1)).astype(int)
        news_df['sentiment_abs'] = news_df['sentiment_score'].abs()
        
        # 뉴스 개수 특징
        news_df['high_news_volume'] = (news_df['article_count'] > news_df['article_count'].quantile(0.75)).astype(int)
        news_df['low_news_volume'] = (news_df['article_count'] < news_df['article_count'].quantile(0.25)).astype(int)
        
        # 이동평균 기반 특징
        for window in [3, 7]:
            news_df[f'sentiment_ma_{window}'] = news_df['sentiment_score'].rolling(window).mean()
            news_df[f'news_count_ma_{window}'] = news_df['article_count'].rolling(window).mean()
        
        # 변화율 특징
        news_df['sentiment_change'] = news_df['sentiment_score'].diff()
        news_df['news_count_change'] = news_df['article_count'].pct_change()
        
        # 변동성 특징
        news_df['sentiment_volatility'] = news_df['sentiment_score'].rolling(7).std()
        
        # 고임팩트 뉴스 특징 (요약 데이터에서)
        if 'recent_news' in summary_data:
            high_impact_dates = self._extract_high_impact_dates(summary_data['recent_news'])
            news_df['high_impact_news'] = news_df['date'].dt.date.isin(high_impact_dates).astype(int)
        
        return news_df
    
    def _extract_high_impact_dates(self, recent_news: List[Dict]) -> set:
        """고임팩트 뉴스가 있는 날짜들 추출"""
        high_impact_dates = set()
        
        for news in recent_news:
            impact_score = news.get('market_impact', 0) * (news.get('spy_relevance', 0.8) if 'spy_relevance' in news else 0.8)
            if impact_score > 0.6:  # 임계값
                news_date = datetime.strptime(news['date'], '%Y-%m-%d').date()
                high_impact_dates.add(news_date)
                
        return high_impact_dates
    
    def integrate_data(self) -> pd.DataFrame:
        """뉴스 데이터와 주식 데이터를 통합"""
        if self.spy_data is None:
            self.load_spy_data()
        if self.news_data is None:
            self.load_news_data()
            
        logger.info("Integrating SPY and news data...")
        
        # 날짜별로 조인
        integrated = pd.merge(
            self.spy_data,
            self.news_data,
            on='date',
            how='left'  # SPY 데이터 기준으로 조인
        )
        
        # 뉴스 데이터가 없는 날짜는 기본값으로 채움
        news_columns = self.news_data.columns.drop('date')
        for col in news_columns:
            if col in ['sentiment_score', 'sentiment_change', 'sentiment_ma_3', 'sentiment_ma_7', 'sentiment_volatility']:
                integrated[col] = integrated[col].fillna(0)  # 감정 점수 관련은 0으로
            elif col in ['article_count', 'news_count_change', 'news_count_ma_3', 'news_count_ma_7']:
                integrated[col] = integrated[col].fillna(1)  # 뉴스 개수는 1로 (0은 비현실적)
            else:
                integrated[col] = integrated[col].fillna(0)  # 나머지는 0으로
        
        # 라벨 생성 (다음 날 가격 상승 여부)
        integrated['next_day_return'] = integrated['close'].shift(-1) / integrated['close'] - 1
        integrated['target'] = (integrated['next_day_return'] > 0).astype(int)
        
        # NaN 제거
        integrated = integrated.dropna()
        
        logger.info(f"Integrated dataset created with {len(integrated)} records")
        logger.info(f"Features: {len(integrated.columns)} total")
        
        self.integrated_data = integrated
        return integrated
    
    def save_integrated_data(self, output_file: str = "integrated_spy_news_data.csv"):
        """통합 데이터 저장"""
        if self.integrated_data is None:
            self.integrate_data()
            
        output_path = os.path.join(self.data_dir, output_file)
        self.integrated_data.to_csv(output_path, index=False)
        logger.info(f"Integrated data saved to {output_path}")
        
        # 데이터 요약 정보 출력
        self._print_data_summary()
        
        return output_path
    
    def _print_data_summary(self):
        """데이터 요약 정보 출력"""
        if self.integrated_data is None:
            return
            
        df = self.integrated_data
        
        logger.info("\n=== 통합 데이터 요약 ===")
        logger.info(f"총 레코드 수: {len(df)}")
        logger.info(f"날짜 범위: {df['date'].min()} ~ {df['date'].max()}")
        logger.info(f"총 특성 수: {len(df.columns)}")
        
        logger.info(f"\n=== 기술적 지표 특성 ===")
        technical_features = [col for col in df.columns if any(indicator in col for indicator in 
                            ['rsi', 'macd', 'bb_', 'ma_', 'volatility', 'price_', 'volume_', 'unusual_volume', 'price_spike'])]
        logger.info(f"기술적 지표 특성 수: {len(technical_features)}")
        
        logger.info(f"\n=== 뉴스 관련 특성 ===")
        news_features = [col for col in df.columns if any(keyword in col for keyword in 
                        ['sentiment', 'news', 'article', 'impact'])]
        logger.info(f"뉴스 관련 특성 수: {len(news_features)}")
        logger.info(f"뉴스 특성: {news_features}")
        
        logger.info(f"\n=== 타겟 분포 ===")
        target_dist = df['target'].value_counts()
        logger.info(f"상승 (1): {target_dist.get(1, 0)} ({target_dist.get(1, 0)/len(df)*100:.1f}%)")
        logger.info(f"하락 (0): {target_dist.get(0, 0)} ({target_dist.get(0, 0)/len(df)*100:.1f}%)")
        
        logger.info(f"\n=== 뉴스 감정 분석 통계 ===")
        logger.info(f"평균 감정 점수: {df['sentiment_score'].mean():.3f}")
        logger.info(f"감정 점수 표준편차: {df['sentiment_score'].std():.3f}")
        logger.info(f"평균 일별 뉴스 개수: {df['article_count'].mean():.1f}")
        logger.info(f"고임팩트 뉴스 일수: {df['high_impact_news'].sum()}일")


def main():
    """메인 실행 함수"""
    logger.info("🤖 뉴스-주식 데이터 통합 시작")
    
    try:
        # 통합기 초기화
        integrator = NewsStockIntegration()
        
        # 데이터 통합 및 저장
        output_file = integrator.save_integrated_data()
        
        logger.info("✅ 뉴스-주식 데이터 통합 완료!")
        logger.info(f"📁 저장 파일: {output_file}")
        
        return output_file
        
    except Exception as e:
        logger.error(f"❌ 데이터 통합 실패: {str(e)}")
        raise


if __name__ == "__main__":
    main()