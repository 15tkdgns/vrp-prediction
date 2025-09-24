#!/usr/bin/env python3
"""
통합 데이터 처리 모듈
안전한 특징 공학 및 데이터 검증
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLeakageValidator:
    """데이터 누출 검증기"""

    @staticmethod
    def validate_temporal_order(df, date_col='Date'):
        """시계열 순서 검증"""
        if date_col in df.columns:
            return df[date_col].is_monotonic_increasing
        return True

    @staticmethod
    def validate_feature_independence(X, y, threshold=0.99):
        """특징-타겟 간 과도한 상관관계 검사"""
        correlations = []
        for i in range(X.shape[1]):
            corr = np.corrcoef(X[:, i], y)[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))

        max_corr = max(correlations) if correlations else 0
        return max_corr < threshold, max_corr

    @staticmethod
    def validate_time_series_split(train_idx, test_idx):
        """시계열 분할 검증"""
        return max(train_idx) < min(test_idx)

class FeatureEngineer:
    """안전한 특징 공학"""

    def __init__(self, lookback_window=20):
        self.lookback_window = lookback_window

    def create_technical_features(self, df):
        """기술적 지표 생성 (미래 정보 누출 방지)"""
        df = df.copy()

        # 기본 가격 특징
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))

        # 모멘텀 (다양한 기간)
        periods = [3, 5, 10, 15, 20]
        for period in periods:
            df[f'momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1
            df[f'roc_{period}'] = (df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)

        # 변동성
        windows = [5, 10, 20, 30]
        for window in windows:
            df[f'volatility_{window}'] = df['returns'].rolling(window, min_periods=1).std()

        # 이동평균
        ma_pairs = [(5, 20), (10, 30), (20, 50)]
        for short, long in ma_pairs:
            df[f'sma_{short}'] = df['Close'].rolling(short, min_periods=1).mean()
            df[f'sma_{long}'] = df['Close'].rolling(long, min_periods=1).mean()
            df[f'sma_ratio_{short}_{long}'] = df[f'sma_{short}'] / df[f'sma_{long}']

        # RSI
        rsi_periods = [7, 14, 21]
        for period in rsi_periods:
            df[f'rsi_{period}'] = self._calculate_rsi(df['Close'], period)

        # 볼륨 특징
        df['volume_sma_20'] = df['Volume'].rolling(20, min_periods=1).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma_20']

        # 타겟 변수 (다음 날 방향)
        df['next_return'] = df['Close'].pct_change().shift(-1)
        df['direction_target'] = (df['next_return'] > 0).astype(int)
        df['return_target'] = df['next_return']

        # 결측값 처리
        df = df.fillna(method='ffill').fillna(0)

        return df

    def _calculate_rsi(self, prices, window=14):
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

class DataProcessor:
    """통합 데이터 처리 시스템"""

    def __init__(self):
        self.validator = DataLeakageValidator()
        self.feature_engineer = FeatureEngineer()
        self.scalers = {}

    def load_and_validate_data(self, data_path):
        """데이터 로딩 및 검증"""
        try:
            df = pd.read_csv(data_path)
            logger.info(f"원본 데이터 로딩: {df.shape}")

            # 시계열 순서 검증
            if not self.validator.validate_temporal_order(df):
                raise ValueError("시계열 순서가 올바르지 않습니다!")

            # 특징 공학
            df = self.feature_engineer.create_technical_features(df)
            logger.info(f"특징 공학 완료: {df.shape}")

            return df

        except Exception as e:
            logger.error(f"데이터 로딩 실패: {e}")
            return None

    def prepare_ml_data(self, df, target_type='direction'):
        """ML용 데이터 준비"""
        # 특징 컬럼 선별
        feature_cols = [col for col in df.columns
                       if col not in ['Date', 'direction_target', 'return_target', 'next_return']
                       and not col.startswith('Unnamed')]

        # 타겟 선택
        if target_type == 'direction':
            target_col = 'direction_target'
        else:
            target_col = 'return_target'

        # 유효한 데이터만 선택
        valid_mask = df[target_col].notna()
        df_clean = df[valid_mask].reset_index(drop=True)

        X = df_clean[feature_cols]
        y = df_clean[target_col]

        logger.info(f"ML 데이터 준비: X={X.shape}, y={y.shape}")

        # 데이터 타입별 분포 확인
        if target_type == 'direction':
            logger.info(f"방향 분포: 상승={y.sum()}, 하락={len(y)-y.sum()}")
        else:
            logger.info(f"수익률 통계: 평균={y.mean():.4f}, 표준편차={y.std():.4f}")

        return X, y, feature_cols

    def prepare_sequence_data(self, df, sequence_length=20, target_type='direction'):
        """시계열 시퀀스 데이터 준비"""
        X, y, feature_cols = self.prepare_ml_data(df, target_type)

        # 정규화
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers[f'{target_type}_scaler'] = scaler

        # 시퀀스 생성
        X_seq, y_seq = [], []
        for i in range(sequence_length, len(X_scaled)):
            X_seq.append(X_scaled[i-sequence_length:i])
            y_seq.append(y.iloc[i])

        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)

        logger.info(f"시퀀스 데이터: X={X_seq.shape}, y={y_seq.shape}")

        return X_seq, y_seq, scaler

    def create_train_val_split(self, X, y, n_splits=5):
        """시계열 분할"""
        tscv = TimeSeriesSplit(n_splits=n_splits)

        splits = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            # 데이터 누출 검증
            if not self.validator.validate_time_series_split(train_idx, val_idx):
                raise ValueError(f"Fold {fold}: 시계열 분할 오류!")

            splits.append((train_idx, val_idx))

        return splits

    def validate_data_integrity(self, X, y):
        """데이터 무결성 검증"""
        # 특징-타겟 상관관계 검증
        is_safe, max_corr = self.validator.validate_feature_independence(X, y)

        if not is_safe:
            logger.warning(f"높은 특징-타겟 상관관계 감지: {max_corr:.3f}")

        # 무한값/결측값 검사
        if np.any(np.isinf(X)) or np.any(np.isnan(X)):
            logger.warning("무한값/결측값 감지됨")

        return {
            'feature_target_correlation_safe': is_safe,
            'max_correlation': max_corr,
            'has_infinite_values': np.any(np.isinf(X)),
            'has_nan_values': np.any(np.isnan(X))
        }

if __name__ == "__main__":
    # 테스트
    processor = DataProcessor()
    data_path = "/root/workspace/data/training/sp500_2020_2024_enhanced.csv"

    df = processor.load_and_validate_data(data_path)
    if df is not None:
        X, y, feature_cols = processor.prepare_ml_data(df, 'direction')
        integrity = processor.validate_data_integrity(X.values, y.values)
        print(f"데이터 무결성: {integrity}")
        print(f"특징 수: {len(feature_cols)}")
        print("✅ 데이터 처리 모듈 정상 작동")