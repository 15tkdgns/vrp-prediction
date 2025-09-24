#!/usr/bin/env python3
"""
초안전 데이터 처리 시스템
가장 엄격한 기준으로 데이터 누출 완전 차단
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UltraSafeDataProcessor:
    """초안전 데이터 처리 시스템"""

    def __init__(self):
        # 극도로 엄격한 기준
        self.MAX_R2 = 0.10           # 10% 이상 금지
        self.MAX_DIRECTION_ACC = 60.0 # 60% 이상 금지
        self.MAX_CORRELATION = 0.20   # 20% 이상 상관관계 금지
        self.MIN_LAG_DAYS = 2         # 최소 2일 지연

        logger.info("초안전 데이터 처리 시스템 초기화")
        logger.info(f"극도로 엄격한 기준 - 상관관계: <{self.MAX_CORRELATION}")

    def create_ultra_safe_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """초안전 특징 생성 (극도로 보수적)"""
        logger.info("=== 초안전 특징 생성 ===")

        df = df.copy()

        # 1. 과거 수익률만 사용 (최소 2일 지연)
        logger.info("1. 과거 수익률 (2일+ 지연)")
        df['returns'] = df['Close'].pct_change()
        df['returns_lag2'] = df['returns'].shift(2)
        df['returns_lag3'] = df['returns'].shift(3)
        df['returns_lag5'] = df['returns'].shift(5)
        df['returns_lag10'] = df['returns'].shift(10)

        # 2. 과거 변동성 (보수적)
        logger.info("2. 과거 변동성 (2일+ 지연)")
        df['vol_5_lag2'] = df['returns_lag2'].rolling(5).std()
        df['vol_10_lag2'] = df['returns_lag2'].rolling(10).std()

        # 3. 과거 수익률 통계 (보수적)
        logger.info("3. 과거 수익률 통계")
        df['returns_mean_5_lag2'] = df['returns_lag2'].rolling(5).mean()
        df['returns_std_5_lag2'] = df['returns_lag2'].rolling(5).std()

        # 4. 단순 지연 차이 (안전한 특징)
        logger.info("4. 단순 지연 차이")
        df['returns_diff_2_5'] = df['returns_lag2'] - df['returns_lag5']
        df['returns_diff_3_10'] = df['returns_lag3'] - df['returns_lag10']

        # 결측값 처리
        df = df.dropna()

        # 안전한 특징만 선택
        safe_features = [
            'returns_lag2', 'returns_lag3', 'returns_lag5', 'returns_lag10',
            'vol_5_lag2', 'vol_10_lag2',
            'returns_mean_5_lag2', 'returns_std_5_lag2',
            'returns_diff_2_5', 'returns_diff_3_10'
        ]

        logger.info(f"초안전 특징: {len(safe_features)}개")

        return df, safe_features

    def validate_ultra_safe_features(self, df: pd.DataFrame, feature_cols: List[str], target_col: str) -> bool:
        """초안전 특징 검증"""
        logger.info("=== 초안전 특징 검증 ===")

        max_corr = 0
        dangerous_features = []

        for col in feature_cols:
            if col in df.columns:
                corr = abs(df[col].corr(df[target_col]))
                max_corr = max(max_corr, corr)

                if corr > self.MAX_CORRELATION:
                    dangerous_features.append((col, corr))

        logger.info(f"최대 상관관계: {max_corr:.3f}")

        if dangerous_features:
            logger.error(f"위험한 특징 발견: {len(dangerous_features)}개")
            for col, corr in dangerous_features:
                logger.error(f"   {col}: {corr:.3f}")
            return False

        logger.info("✅ 모든 특징이 초안전 기준 통과")
        return True

    def prepare_ultra_safe_data(self, data_path: str, target_type: str = 'return') -> Dict:
        """초안전 데이터 준비"""
        logger.info("=" * 80)
        logger.info("🛡️ 초안전 데이터 준비 시작")
        logger.info("🚫 극도로 엄격한 데이터 누출 방지")
        logger.info("=" * 80)

        # 1. 데이터 로딩
        df = pd.read_csv(data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)

        logger.info(f"원본 데이터: {df.shape}")

        # 2. 초안전 특징 생성
        df, safe_features = self.create_ultra_safe_features(df)

        # 3. 타겟 생성
        if target_type == 'return':
            target = df['Close'].pct_change().shift(-1)
            target_name = 'next_return'
        else:  # direction
            returns = df['Close'].pct_change().shift(-1)
            target = (returns > 0).astype(int)
            target_name = 'next_direction'

        df[target_name] = target
        df = df.dropna()

        # 4. 초안전 검증
        is_safe = self.validate_ultra_safe_features(df, safe_features, target_name)

        if not is_safe:
            raise ValueError("초안전 검증 실패")

        # 5. 최종 데이터 준비
        X = df[safe_features].values
        y = df[target_name].values

        # 6. 시간 분할
        tscv = TimeSeriesSplit(n_splits=3, test_size=50, gap=2)  # 2일 간격
        splits = list(tscv.split(X))

        logger.info(f"최종 초안전 데이터: X{X.shape}, y{y.shape}")

        # 7. 기준선 성능 측정 (초안전 확인)
        baseline_r2 = self._measure_baseline_performance(X, y)

        if baseline_r2 > self.MAX_R2:
            logger.warning(f"기준선 R² {baseline_r2:.3f} > {self.MAX_R2} - 여전히 위험할 수 있음")

        logger.info("✅ 초안전 데이터 준비 완료")

        return {
            'X': X,
            'y': y,
            'feature_names': safe_features,
            'splits': splits,
            'target_type': target_type,
            'baseline_r2': baseline_r2,
            'safety_level': 'ULTRA_SAFE'
        }

    def _measure_baseline_performance(self, X: np.ndarray, y: np.ndarray) -> float:
        """기준선 성능 측정"""
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score

        # 간단한 분할
        split_point = int(len(X) * 0.8)
        X_train, X_test = X[:split_point], X[split_point:]
        y_train, y_test = y[:split_point], y[split_point:]

        # 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 선형 회귀
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        r2 = r2_score(y_test, y_pred)
        logger.info(f"기준선 R²: {r2:.4f}")

        return r2

def main():
    """테스트"""
    processor = UltraSafeDataProcessor()

    try:
        data_dict = processor.prepare_ultra_safe_data(
            '/root/workspace/data/training/sp500_2020_2024_enhanced.csv',
            target_type='return'
        )

        print(f"\n✅ 초안전 데이터 준비 성공:")
        print(f"   데이터 크기: {data_dict['X'].shape}")
        print(f"   특징: {data_dict['feature_names']}")
        print(f"   기준선 R²: {data_dict['baseline_r2']:.4f}")
        print(f"   안전 수준: {data_dict['safety_level']}")

        return data_dict

    except Exception as e:
        print(f"\n❌ 초안전 데이터 준비 실패: {e}")
        return None

if __name__ == "__main__":
    result = main()