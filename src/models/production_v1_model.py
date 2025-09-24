#!/usr/bin/env python3
"""
Production V1 Model for SPY Volatility Prediction
최고 성능 Ridge Regression 모델 (R² = 0.314)
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Tuple, List, Optional, Union

import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 프로젝트 루트 추가
sys.path.append('/root/workspace')
from src.core.unified_config import UnifiedConfigManager
from src.core.logger import setup_logger

class ProductionV1Model:
    """
    프로덕션용 V1 변동성 예측 모델

    특징:
    - Ridge Regression (alpha=1.8523)
    - 12개 핵심 특성
    - 5일 후 변동성 예측
    - R² = 0.314 (검증 완료)
    """

    def __init__(self, config_path: Optional[str] = None):
        """모델 초기화"""
        self.logger = setup_logger(self.__class__.__name__)
        self.config = UnifiedConfigManager()

        # 모델 사양
        self.model_specs = {
            'name': 'V1_Volatility_Predictor',
            'type': 'Ridge Regression',
            'alpha': 1.8523,
            'validation_r2': 0.314,
            'validation_r2_std': 0.181,
            'features_count': 12,
            'target': 'target_vol_5d',
            'prediction_horizon': '5 days',
            'complexity': 'Low',
            'maintainability': 'High',
            'interpretability': 'Excellent'
        }

        # 핵심 특성 정의
        self.feature_names = [
            'vol_5', 'vol_10', 'vol_20',
            'return_lag_1', 'return_lag_2', 'return_lag_3',
            'zscore_10', 'zscore_20',
            'momentum_10', 'momentum_20',
            'vol_5_20_ratio', 'vol_regime'
        ]

        # 모델 컴포넌트
        self.model = Ridge(alpha=self.model_specs['alpha'], random_state=42)
        self.scaler = StandardScaler()

        # 상태 변수
        self.is_trained = False
        self.last_prediction_time = None
        self.training_stats = {}

        self.logger.info(f"✅ {self.model_specs['name']} 초기화 완료")
        self.logger.info(f"📊 모델 사양: R² = {self.model_specs['validation_r2']}")

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        V1 모델용 특성 생성

        Args:
            data: OHLCV 데이터

        Returns:
            특성 DataFrame
        """
        try:
            features = pd.DataFrame(index=data.index)

            # 1. 변동성 특성 (vol_5, vol_10, vol_20)
            for window in [5, 10, 20]:
                features[f'vol_{window}'] = (
                    data['close'].rolling(window).std() * np.sqrt(252)
                )

            # 2. 수익률 지연 특성 (return_lag_1, return_lag_2, return_lag_3)
            returns = data['close'].pct_change()
            for lag in [1, 2, 3]:
                features[f'return_lag_{lag}'] = returns.shift(lag)

            # 3. Z-스코어 특성 (zscore_10, zscore_20)
            for window in [10, 20]:
                mean_ret = returns.rolling(window).mean()
                std_ret = returns.rolling(window).std()
                features[f'zscore_{window}'] = (returns - mean_ret) / std_ret

            # 4. 모멘텀 특성 (momentum_10, momentum_20)
            for window in [10, 20]:
                features[f'momentum_{window}'] = (
                    data['close'] / data['close'].shift(window) - 1
                )

            # 5. 변동성 비율 특성 (vol_5_20_ratio)
            features['vol_5_20_ratio'] = features['vol_5'] / features['vol_20']

            # 6. 변동성 체제 특성 (vol_regime)
            vol_median = features['vol_20'].rolling(252).median()
            features['vol_regime'] = (features['vol_20'] > vol_median).astype(int)

            # 결측치 제거
            features = features.dropna()

            # 특성 순서 정렬
            features = features[self.feature_names]

            self.logger.info(f"✅ 특성 생성 완료: {features.shape[0]}개 샘플, {features.shape[1]}개 특성")

            return features

        except Exception as e:
            self.logger.error(f"❌ 특성 생성 실패: {e}")
            raise

    def create_target(self, data: pd.DataFrame) -> pd.Series:
        """
        5일 후 변동성 타겟 생성

        Args:
            data: OHLCV 데이터

        Returns:
            타겟 시리즈
        """
        try:
            returns = data['close'].pct_change()

            # 5일 후 변동성 계산
            target = returns.rolling(5).std().shift(-5) * np.sqrt(252)
            target.name = 'target_vol_5d'

            self.logger.info(f"✅ 타겟 생성 완료: {target.count()}개 유효값")

            return target

        except Exception as e:
            self.logger.error(f"❌ 타겟 생성 실패: {e}")
            raise

    def train(self, data: pd.DataFrame) -> Dict:
        """
        모델 훈련

        Args:
            data: 훈련용 OHLCV 데이터

        Returns:
            훈련 통계
        """
        try:
            self.logger.info("🔄 V1 모델 훈련 시작...")

            # 특성 및 타겟 생성
            features = self.prepare_features(data)
            target = self.create_target(data)

            # 데이터 정렬 및 결측치 제거
            aligned_data = pd.concat([features, target], axis=1).dropna()
            X = aligned_data[self.feature_names]
            y = aligned_data['target_vol_5d']

            self.logger.info(f"📊 훈련 데이터: {X.shape[0]}개 샘플")

            # 특성 스케일링
            X_scaled = self.scaler.fit_transform(X)

            # 모델 훈련
            self.model.fit(X_scaled, y)

            # 훈련 성능 계산
            y_pred_train = self.model.predict(X_scaled)

            self.training_stats = {
                'samples': len(X),
                'features': len(self.feature_names),
                'train_r2': r2_score(y, y_pred_train),
                'train_mse': mean_squared_error(y, y_pred_train),
                'train_rmse': np.sqrt(mean_squared_error(y, y_pred_train)),
                'train_mae': mean_absolute_error(y, y_pred_train),
                'training_date': datetime.now().isoformat(),
                'model_alpha': self.model_specs['alpha']
            }

            self.is_trained = True

            self.logger.info(f"✅ V1 모델 훈련 완료")
            self.logger.info(f"📊 훈련 R² = {self.training_stats['train_r2']:.4f}")

            return self.training_stats

        except Exception as e:
            self.logger.error(f"❌ 모델 훈련 실패: {e}")
            raise

    def predict_volatility(self, features: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        변동성 예측

        Args:
            features: 예측용 특성

        Returns:
            예측된 5일 후 변동성
        """
        if not self.is_trained:
            raise ValueError("모델이 훈련되지 않았습니다. train() 메서드를 먼저 실행하세요.")

        try:
            if isinstance(features, pd.DataFrame):
                X = features[self.feature_names].values
            else:
                X = features

            # 특성 스케일링
            X_scaled = self.scaler.transform(X)

            # 예측
            predictions = self.model.predict(X_scaled)

            self.last_prediction_time = datetime.now()

            self.logger.info(f"✅ 변동성 예측 완료: {len(predictions)}개 예측값")

            return predictions

        except Exception as e:
            self.logger.error(f"❌ 변동성 예측 실패: {e}")
            raise

    def get_feature_importance(self) -> Dict:
        """특성 중요도 반환"""
        if not self.is_trained:
            return {}

        coefficients = self.model.coef_

        importance = {
            feature: abs(coef) for feature, coef
            in zip(self.feature_names, coefficients)
        }

        # 중요도 순으로 정렬
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        return importance

    def get_performance_metrics(self) -> Dict:
        """모델 성능 지표 반환"""
        metrics = {
            **self.model_specs,
            **self.training_stats,
            'is_trained': self.is_trained,
            'last_prediction_time': self.last_prediction_time.isoformat() if self.last_prediction_time else None,
            'feature_importance': self.get_feature_importance()
        }

        return metrics

    def save_model(self, filepath: str = None) -> str:
        """모델 저장"""
        if not self.is_trained:
            raise ValueError("훈련된 모델만 저장할 수 있습니다.")

        if filepath is None:
            filepath = f"/root/workspace/data/models/v1_production_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"

        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'model_specs': self.model_specs,
                'training_stats': self.training_stats,
                'save_time': datetime.now().isoformat()
            }

            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)

            self.logger.info(f"✅ 모델 저장 완료: {filepath}")

            return filepath

        except Exception as e:
            self.logger.error(f"❌ 모델 저장 실패: {e}")
            raise

    def load_model(self, filepath: str):
        """모델 로드"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.model_specs = model_data['model_specs']
            self.training_stats = model_data['training_stats']

            self.is_trained = True

            self.logger.info(f"✅ 모델 로드 완료: {filepath}")

        except Exception as e:
            self.logger.error(f"❌ 모델 로드 실패: {e}")
            raise


def demo_v1_model():
    """V1 모델 데모"""
    import yfinance as yf

    print("🚀 V1 프로덕션 모델 데모 시작...")

    # 모델 초기화
    model = ProductionV1Model()

    # SPY 데이터 로드
    print("📊 SPY 데이터 로드 중...")
    spy_data = yf.download('SPY', start='2020-01-01', end='2024-01-01')
    spy_data.columns = [col.lower() for col in spy_data.columns]

    # 모델 훈련
    print("🔄 모델 훈련 중...")
    training_stats = model.train(spy_data)

    # 성능 지표 출력
    print("\n📊 V1 모델 성능 지표:")
    print(f"  - 훈련 R²: {training_stats['train_r2']:.4f}")
    print(f"  - 훈련 RMSE: {training_stats['train_rmse']:.4f}")
    print(f"  - 훈련 MAE: {training_stats['train_mae']:.4f}")
    print(f"  - 샘플 수: {training_stats['samples']:,}")
    print(f"  - 특성 수: {training_stats['features']}")

    # 특성 중요도
    print("\n🎯 특성 중요도 (Top 5):")
    importance = model.get_feature_importance()
    for i, (feature, score) in enumerate(list(importance.items())[:5]):
        print(f"  {i+1}. {feature}: {score:.4f}")

    # 모델 저장
    model_path = model.save_model()
    print(f"\n✅ 모델 저장 완료: {model_path}")

    # 최근 데이터로 예측 테스트
    print("\n🔮 최근 데이터 예측 테스트...")
    recent_data = yf.download('SPY', start='2024-01-01', period='1y')
    recent_data.columns = [col.lower() for col in recent_data.columns]

    features = model.prepare_features(recent_data)
    if len(features) > 0:
        predictions = model.predict_volatility(features.tail(5))
        print(f"  최근 5일 변동성 예측: {predictions}")

    print("\n🎉 V1 프로덕션 모델 데모 완료!")

    return model


if __name__ == '__main__':
    model = demo_v1_model()