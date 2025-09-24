#!/usr/bin/env python3
"""
Extended Data Integration Module
5년 SPY 데이터를 활용한 확장 데이터셋 생성 및 데이터 누수 방지
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ExtendedDataIntegrator:
    """5년 SPY 데이터를 통합하고 데이터 누수를 방지하는 클래스"""
    
    def __init__(self, data_dir: str = "data", output_dir: str = "data/processed"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.raw_data = None
        self.processed_data = None
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 데이터 누수 방지를 위한 특성 분류
        self.safe_features = []
        self.leakage_features = []
        
    def load_extended_spy_data(self) -> pd.DataFrame:
        """5년 SPY 데이터 로드 및 기본 전처리"""
        logger.info("Loading extended SPY data (2020-2025)...")
        
        spy_file = f"{self.data_dir}/raw/spy_data_2020_2025.csv"
        
        if not os.path.exists(spy_file):
            raise FileNotFoundError(f"Extended SPY data not found: {spy_file}")
            
        # 데이터 로드
        self.raw_data = pd.read_csv(spy_file)
        
        # 첫 번째 컬럼이 빈 이름인 경우 인덱스로 처리
        if self.raw_data.columns[0] == '':
            self.raw_data = self.raw_data.drop(self.raw_data.columns[0], axis=1)
        
        # Date 컬럼 생성 (인덱스로부터)
        if 'Date' not in self.raw_data.columns:
            # 인덱스를 날짜로 변환
            date_range = pd.date_range(start='2020-01-02', periods=len(self.raw_data), freq='D')
            # 주말 제거
            business_days = pd.bdate_range(start='2020-01-02', periods=len(self.raw_data))
            
            # 실제로는 CSV의 첫 번째 컬럼이 날짜일 가능성이 높음
            # 첫 번째 행을 확인해서 날짜 형식인지 판단
            try:
                # 첫 번째 컬럼 값들을 날짜로 파싱 시도
                first_col_values = self.raw_data.iloc[:5, 0].astype(str)
                parsed_dates = pd.to_datetime(first_col_values, errors='coerce')
                
                if not parsed_dates.isna().all():
                    # 첫 번째 컬럼이 날짜
                    self.raw_data['Date'] = pd.to_datetime(self.raw_data.iloc[:, 0])
                    self.raw_data = self.raw_data.drop(self.raw_data.columns[0], axis=1)
                else:
                    # 날짜 정보가 없으면 순차적으로 생성
                    self.raw_data['Date'] = business_days[:len(self.raw_data)]
            except:
                # 에러 발생시 기본 날짜 범위 사용
                self.raw_data['Date'] = business_days[:len(self.raw_data)]
        
        # 날짜 정렬
        self.raw_data = self.raw_data.sort_values('Date').reset_index(drop=True)
        
        # 기본 컬럼 확인
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in self.raw_data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        logger.info(f"Loaded {len(self.raw_data)} data points from {self.raw_data['Date'].min()} to {self.raw_data['Date'].max()}")
        
        return self.raw_data
        
    def calculate_safe_technical_indicators(self, data: pd.DataFrame, lookback_only: bool = True) -> pd.DataFrame:
        """데이터 누수 방지를 위한 안전한 기술적 지표 계산
        
        Args:
            data: 입력 데이터프레임
            lookback_only: True면 과거 데이터만 사용, False면 미래 데이터도 허용
        """
        logger.info("Calculating safe technical indicators (no data leakage)...")
        
        df = data.copy()
        
        # 기본 가격 변화율 (안전 - 과거 데이터만 사용)
        df['Returns'] = df['Close'].pct_change()
        df['Price_Change'] = df['Returns'].abs()
        
        # 가격 대비 거래량 (안전)
        df['Volume_Price_Ratio'] = df['Volume'] / df['Close']
        
        # 이동평균 (안전 - 과거 데이터만 사용)
        windows = [5, 10, 20, 50]
        for window in windows:
            df[f'SMA_{window}'] = df['Close'].rolling(window=window, min_periods=1).mean()
            df[f'Volume_SMA_{window}'] = df['Volume'].rolling(window=window, min_periods=1).mean()
            
            # 가격이 이동평균보다 높은지 (안전)
            df[f'Price_Above_SMA_{window}'] = (df['Close'] > df[f'SMA_{window}']).astype(int)
            
        # 변동성 (안전 - 과거 데이터만 사용)
        volatility_windows = [5, 10, 20]
        for window in volatility_windows:
            df[f'Volatility_{window}'] = df['Returns'].rolling(window=window, min_periods=1).std()
            
        # RSI (안전 - 과거 데이터만 사용)
        df['RSI_14'] = self.calculate_rsi(df['Close'])
        
        # MACD (안전 - 과거 데이터만 사용)
        df['MACD'], df['MACD_Signal'] = self.calculate_macd(df['Close'])
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # 볼린저 밴드 (안전 - 과거 데이터만 사용)
        df['BB_Upper'], df['BB_Lower'], df['BB_Middle'] = self.calculate_bollinger_bands(df['Close'])
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # ATR (안전 - 과거 데이터만 사용)
        df['ATR_14'] = self.calculate_atr(df)
        
        # 거래량 지표 (안전)
        for window in [5, 10, 20]:
            volume_sma = df[f'Volume_SMA_{window}']
            df[f'Volume_Ratio_{window}'] = df['Volume'] / volume_sma
            df[f'Volume_Spike_{window}'] = (df[f'Volume_Ratio_{window}'] > 2.0).astype(int)
            
        # 가격 갭 (안전)
        df['Price_Gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        df['Gap_Up'] = (df['Price_Gap'] > 0.02).astype(int)  # 2% 이상 갭업
        df['Gap_Down'] = (df['Price_Gap'] < -0.02).astype(int)  # 2% 이상 갭다운
        
        # 일중 변동성 (안전)
        df['Intraday_Range'] = (df['High'] - df['Low']) / df['Close']
        df['High_Low_Ratio'] = df['High'] / df['Low']
        
        # 종가 위치 (안전)
        df['Close_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        # 거래량 가중평균가격 근사치 (안전)
        df['VWAP_Approx'] = (df['High'] + df['Low'] + df['Close']) / 3
        
        self.safe_features = [col for col in df.columns if col not in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']]
        
        logger.info(f"Generated {len(self.safe_features)} safe technical indicators")
        
        return df
        
    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """RSI 계산 (안전 - 과거 데이터만 사용)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """MACD 계산 (안전 - 과거 데이터만 사용)"""
        exp1 = prices.ewm(span=fast, min_periods=1).mean()
        exp2 = prices.ewm(span=slow, min_periods=1).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, min_periods=1).mean()
        
        return macd, signal_line
        
    def calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """볼린저 밴드 계산 (안전 - 과거 데이터만 사용)"""
        rolling_mean = prices.rolling(window=window, min_periods=1).mean()
        rolling_std = prices.rolling(window=window, min_periods=1).std()
        
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        
        return upper_band, lower_band, rolling_mean
        
    def calculate_atr(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """ATR 계산 (안전 - 과거 데이터만 사용)"""
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=window, min_periods=1).mean()
        
        return atr
        
    def create_safe_event_labels(self, data: pd.DataFrame, price_threshold: float = 0.02,
                                volume_threshold: float = 2.0) -> pd.DataFrame:
        """데이터 누수 방지를 위한 안전한 이벤트 라벨 생성
        
        Args:
            data: 입력 데이터
            price_threshold: 가격 변동 임계값 (2%)
            volume_threshold: 거래량 비율 임계값 (평균의 2배)
        """
        logger.info("Creating safe event labels (no future data leakage)...")
        
        df = data.copy()
        
        # 현재 시점의 데이터만 사용하여 라벨 생성
        # 미래의 가격 변동을 예측하는 것이므로, 현재까지의 정보만 사용
        
        # 1. 가격 급변 이벤트 (현재 대비 다음날 변동)
        df['Next_Return'] = df['Returns'].shift(-1)  # 이것은 예측 대상이므로 허용
        df['Price_Event'] = (np.abs(df['Next_Return']) > price_threshold).astype(int)
        
        # 2. 거래량 급증 이벤트 (현재 시점 기준)
        df['Volume_20MA'] = df['Volume'].rolling(window=20, min_periods=1).mean()
        df['Volume_Event'] = (df['Volume'] > df['Volume_20MA'] * volume_threshold).astype(int)
        
        # 3. 복합 이벤트 (가격 + 거래량)
        df['Major_Event'] = ((df['Price_Event'] == 1) & (df['Volume_Event'] == 1)).astype(int)
        
        # 4. 트렌드 변화 이벤트
        df['SMA_5'] = df['Close'].rolling(window=5, min_periods=1).mean()
        df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
        
        # 골든/데드 크로스 감지 (현재 시점 기준)
        df['Golden_Cross'] = ((df['SMA_5'] > df['SMA_20']) & 
                             (df['SMA_5'].shift(1) <= df['SMA_20'].shift(1))).astype(int)
        df['Death_Cross'] = ((df['SMA_5'] < df['SMA_20']) & 
                            (df['SMA_5'].shift(1) >= df['SMA_20'].shift(1))).astype(int)
        
        df['Trend_Change'] = (df['Golden_Cross'] | df['Death_Cross']).astype(int)
        
        # 5. 최종 타겟 변수 생성 (예측하고자 하는 이벤트)
        # 다음날 큰 가격 변동이 있을지 예측
        df['Target'] = df['Price_Event']
        
        # 미래 데이터 제거 (Next_Return은 예측 대상이므로 특성에서 제외)
        feature_columns = [col for col in df.columns 
                          if col not in ['Next_Return', 'Target'] and 'Event' not in col and 'Cross' not in col]
        
        logger.info(f"Created event labels. Target event rate: {df['Target'].mean():.4f}")
        
        return df
        
    def detect_data_leakage_features(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """데이터 누수 가능성이 있는 특성들을 감지"""
        logger.info("Detecting potential data leakage features...")
        
        leakage_indicators = {
            'future_keywords': ['next', 'future', 'ahead', 'forward'],
            'perfect_correlation': [],
            'impossible_accuracy': []
        }
        
        safe_features = []
        risky_features = []
        
        target = data['Target']
        
        for col in data.columns:
            if col in ['Target', 'Date']:
                continue
                
            # 키워드 기반 검사
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in leakage_indicators['future_keywords']):
                risky_features.append(col)
                continue
                
            # 완벽한 상관관계 검사
            if pd.api.types.is_numeric_dtype(data[col]):
                try:
                    correlation = np.abs(np.corrcoef(data[col].fillna(0), target)[0, 1])
                    if correlation > 0.99:
                        leakage_indicators['perfect_correlation'].append(col)
                        risky_features.append(col)
                        continue
                except:
                    pass
                    
            # 안전한 특성으로 분류
            safe_features.append(col)
        
        self.safe_features = safe_features
        self.leakage_features = risky_features
        
        logger.info(f"Safe features: {len(safe_features)}")
        logger.info(f"Risky features: {len(risky_features)}")
        
        if risky_features:
            logger.warning(f"Potential leakage features detected: {risky_features}")
            
        return {
            'safe_features': safe_features,
            'leakage_features': risky_features,
            'leakage_indicators': leakage_indicators
        }
        
    def save_processed_data(self, data: pd.DataFrame, filename: str = "extended_spy_features_safe.csv") -> str:
        """처리된 데이터 저장"""
        output_path = f"{self.output_dir}/{filename}"
        
        # 안전한 특성만 저장
        safe_columns = ['Date', 'Target'] + self.safe_features
        safe_data = data[safe_columns].copy()
        
        # NaN 값 처리
        safe_data = safe_data.fillna(0)
        
        # 저장
        safe_data.to_csv(output_path, index=False)
        
        logger.info(f"Safe processed data saved: {output_path}")
        logger.info(f"Data shape: {safe_data.shape}")
        logger.info(f"Date range: {safe_data['Date'].min()} to {safe_data['Date'].max()}")
        logger.info(f"Target distribution: {safe_data['Target'].value_counts().to_dict()}")
        
        return output_path
        
    def create_data_report(self, data: pd.DataFrame) -> Dict:
        """데이터 품질 리포트 생성"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_summary': {
                'total_samples': len(data),
                'date_range': {
                    'start': data['Date'].min().isoformat(),
                    'end': data['Date'].max().isoformat()
                },
                'target_distribution': data['Target'].value_counts().to_dict(),
                'target_rate': float(data['Target'].mean()),
                'total_features': len(self.safe_features),
                'safe_features_count': len(self.safe_features),
                'leakage_features_count': len(self.leakage_features)
            },
            'feature_summary': {
                'safe_features': self.safe_features[:10],  # 처음 10개만
                'leakage_features': self.leakage_features,
                'feature_categories': {
                    'price_features': [f for f in self.safe_features if 'price' in f.lower() or 'close' in f.lower()],
                    'volume_features': [f for f in self.safe_features if 'volume' in f.lower()],
                    'technical_features': [f for f in self.safe_features if any(t in f.lower() for t in ['sma', 'rsi', 'macd', 'bb', 'atr'])],
                    'volatility_features': [f for f in self.safe_features if 'volatility' in f.lower()]
                }
            },
            'data_quality': {
                'null_values': data[self.safe_features].isnull().sum().to_dict(),
                'infinite_values': np.isinf(data[self.safe_features].select_dtypes(include=[np.number])).sum().to_dict()
            }
        }
        
        # 리포트 저장
        report_path = f"{self.output_dir}/extended_data_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        logger.info(f"Data quality report saved: {report_path}")
        
        return report
        
    def run_complete_integration(self) -> Tuple[str, str]:
        """전체 데이터 통합 프로세스 실행"""
        logger.info("Starting complete data integration process...")
        
        # 1. 5년 SPY 데이터 로드
        raw_data = self.load_extended_spy_data()
        
        # 2. 안전한 기술적 지표 계산
        data_with_features = self.calculate_safe_technical_indicators(raw_data)
        
        # 3. 안전한 이벤트 라벨 생성
        data_with_labels = self.create_safe_event_labels(data_with_features)
        
        # 4. 데이터 누수 검사
        leakage_analysis = self.detect_data_leakage_features(data_with_labels)
        
        # 5. 처리된 데이터 저장
        data_path = self.save_processed_data(data_with_labels)
        
        # 6. 데이터 품질 리포트 생성
        report = self.create_data_report(data_with_labels)
        report_path = f"{self.output_dir}/extended_data_report.json"
        
        logger.info("Data integration completed successfully!")
        logger.info(f"Processed data: {data_path}")
        logger.info(f"Quality report: {report_path}")
        
        return data_path, report_path


def main():
    """메인 실행 함수"""
    integrator = ExtendedDataIntegrator()
    
    try:
        data_path, report_path = integrator.run_complete_integration()
        
        print("\n" + "="*60)
        print("🎯 EXTENDED DATA INTEGRATION COMPLETED")
        print("="*60)
        print(f"📊 Processed Data: {data_path}")
        print(f"📋 Quality Report: {report_path}")
        print(f"✅ Ready for Walk-Forward Validation")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Data integration failed: {e}")
        raise


if __name__ == "__main__":
    main()