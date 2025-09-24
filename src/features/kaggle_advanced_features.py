#!/usr/bin/env python3
"""
캐글 우승자 기법 기반 고급 특징 엔지니어링
Two Sigma, Jane Street, Optiver 등 대회 우승 솔루션 적용
"""

import sys
sys.path.append('/root/workspace')

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from src.core.ultra_safe_data_processor import UltraSafeDataProcessor
from src.validation.auto_leakage_detector import AutoLeakageDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KaggleAdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    """캐글 우승자 기법 기반 고급 특징 엔지니어링"""

    def __init__(self, safety_mode: bool = True):
        self.safety_mode = safety_mode
        self.MAX_CORRELATION = 0.25 if safety_mode else 0.30
        self.scaler_ = StandardScaler()
        self.feature_names_ = None
        self.leakage_detector = AutoLeakageDetector()

        logger.info("🏆 캐글 우승자 기법 기반 고급 특징 엔지니어링 초기화")
        logger.info(f"안전 모드: {safety_mode}, 최대 상관관계: {self.MAX_CORRELATION}")

    def fit(self, X, y=None):
        """고급 특징 생성 규칙 학습"""
        logger.info("=== 캐글 기법 기반 고급 특징 학습 ===")

        # 기본 특징명 (10개)
        base_features = [
            'returns_lag2', 'returns_lag3', 'returns_lag5', 'returns_lag10',
            'vol_5_lag2', 'vol_10_lag2',
            'returns_mean_5_lag2', 'returns_std_5_lag2',
            'returns_diff_2_5', 'returns_diff_3_10'
        ]

        # 캐글 우승자 기법 특징들
        kaggle_features = [
            # Two Sigma 기법: 시간 구조 특징
            'intraday_pattern_lag2', 'volatility_regime_lag2', 'trend_strength_lag2',

            # Jane Street 기법: 고급 기술적 지표
            'fractal_dimension_lag2', 'hurst_exponent_lag2', 'entropy_measure_lag2',

            # Optiver 기법: 시장 미시구조
            'price_acceleration_lag2', 'volatility_of_volatility_lag2', 'momentum_persistence_lag2',

            # 메타 라벨링 특징
            'prediction_confidence_lag2', 'regime_classification_lag2',

            # 어텐션 기반 특징
            'temporal_attention_score_lag2', 'feature_importance_score_lag2'
        ]

        self.feature_names_ = base_features + kaggle_features

        # 특징 생성 후 스케일러 학습
        X_features = self._generate_kaggle_features(X)
        self.scaler_.fit(X_features)

        logger.info(f"총 고급 특징 수: {len(self.feature_names_)}")
        logger.info(f"추가된 캐글 특징: {len(kaggle_features)}개")

        return self

    def transform(self, X):
        """학습된 규칙으로 고급 특징 생성"""
        if self.feature_names_ is None:
            raise ValueError("먼저 fit해야 합니다")

        X_features = self._generate_kaggle_features(X)
        X_scaled = self.scaler_.transform(X_features)

        logger.info(f"캐글 기법 특징 생성: {X.shape} -> {X_scaled.shape}")
        return X_scaled

    def _generate_kaggle_features(self, X):
        """캐글 우승자 기법으로 고급 특징 생성"""

        # 기존 특징 사용
        features = X.copy()
        n_samples = features.shape[0]

        # 기본 특징들 추출
        returns_lag2 = features[:, 0]
        returns_lag3 = features[:, 1]
        returns_lag5 = features[:, 2]
        vol_5_lag2 = features[:, 4]
        vol_10_lag2 = features[:, 5]
        returns_mean_5_lag2 = features[:, 6]
        returns_std_5_lag2 = features[:, 7]

        logger.info("1. Two Sigma 기법: 시간 구조 특징 생성")

        # 1. Two Sigma 기법: 시간 구조 특징
        # 일중 패턴 (간소화된 버전)
        intraday_pattern_lag2 = np.zeros(n_samples)
        for i in range(20, n_samples):
            # 최근 20일 패턴의 주기성 측정
            recent_returns = returns_lag2[i-20:i]
            if len(recent_returns) > 0:
                intraday_pattern_lag2[i] = np.std(recent_returns) / (np.abs(np.mean(recent_returns)) + 1e-8)

        # 변동성 체제 분류
        volatility_regime_lag2 = np.zeros(n_samples)
        for i in range(30, n_samples):
            recent_vol = vol_5_lag2[i-30:i]
            long_term_vol = vol_10_lag2[i-30:i]
            if len(recent_vol) > 0 and len(long_term_vol) > 0:
                volatility_regime_lag2[i] = np.mean(recent_vol) / (np.mean(long_term_vol) + 1e-8)

        # 추세 강도 측정
        trend_strength_lag2 = np.zeros(n_samples)
        for i in range(15, n_samples):
            recent_returns = returns_lag2[i-15:i]
            if len(recent_returns) > 0:
                # 연속적인 같은 방향 움직임의 비율
                directions = np.sign(recent_returns)
                trend_strength_lag2[i] = np.abs(np.sum(directions)) / len(directions)

        logger.info("2. Jane Street 기법: 고급 기술적 지표 생성")

        # 2. Jane Street 기법: 고급 기술적 지표
        # 프랙탈 차원 (간소화된 Higuchi 방법)
        fractal_dimension_lag2 = np.zeros(n_samples)
        for i in range(25, n_samples):
            series = returns_lag2[i-25:i]
            if len(series) > 10:
                # 간소화된 프랙탈 차원 계산
                diffs = np.abs(np.diff(series))
                fractal_dimension_lag2[i] = np.log(np.sum(diffs)) / np.log(len(diffs))

        # 허스트 지수 (R/S 분석 간소화)
        hurst_exponent_lag2 = np.zeros(n_samples)
        for i in range(30, n_samples):
            series = returns_lag2[i-30:i]
            if len(series) > 5:
                # R/S 통계 계산
                mean_series = np.mean(series)
                deviations = np.cumsum(series - mean_series)
                R = np.max(deviations) - np.min(deviations)
                S = np.std(series)
                if S > 1e-8:
                    hurst_exponent_lag2[i] = np.log(R/S) / np.log(len(series))

        # 엔트로피 측정 (Shannon 엔트로피 근사)
        entropy_measure_lag2 = np.zeros(n_samples)
        for i in range(20, n_samples):
            series = returns_lag2[i-20:i]
            if len(series) > 0:
                # 수익률을 구간으로 나누어 엔트로피 계산
                hist, _ = np.histogram(series, bins=5, density=True)
                hist = hist[hist > 0]  # 0이 아닌 값만
                if len(hist) > 0:
                    entropy_measure_lag2[i] = -np.sum(hist * np.log(hist + 1e-8))

        logger.info("3. Optiver 기법: 시장 미시구조 특징 생성")

        # 3. Optiver 기법: 시장 미시구조
        # 가격 가속도 (2차 차분)
        price_acceleration_lag2 = returns_lag2 - returns_lag3

        # 변동성의 변동성
        volatility_of_volatility_lag2 = np.zeros(n_samples)
        for i in range(15, n_samples):
            recent_vols = vol_5_lag2[i-15:i]
            if len(recent_vols) > 0:
                volatility_of_volatility_lag2[i] = np.std(recent_vols)

        # 모멘텀 지속성
        momentum_persistence_lag2 = np.zeros(n_samples)
        for i in range(10, n_samples):
            recent_returns = returns_lag2[i-10:i]
            if len(recent_returns) > 0:
                # 연속적인 움직임의 일관성 측정
                directions = np.sign(recent_returns)
                momentum_persistence_lag2[i] = np.mean(np.abs(np.diff(directions)) == 0)

        logger.info("4. 메타 라벨링 특징 생성")

        # 4. 메타 라벨링 특징
        # 예측 신뢰도 (변동성 기반 근사)
        prediction_confidence_lag2 = 1 / (1 + vol_5_lag2)

        # 체제 분류 (평균 회귀 vs 추세)
        regime_classification_lag2 = np.zeros(n_samples)
        for i in range(20, n_samples):
            recent_returns = returns_lag2[i-20:i]
            if len(recent_returns) > 0:
                # 자기상관 측정으로 체제 분류
                if len(recent_returns) > 1:
                    autocorr = np.corrcoef(recent_returns[:-1], recent_returns[1:])[0, 1]
                    regime_classification_lag2[i] = autocorr if not np.isnan(autocorr) else 0

        logger.info("5. 어텐션 기반 특징 생성")

        # 5. 어텐션 기반 특징
        # 시간 어텐션 점수 (최근성 가중치)
        temporal_attention_score_lag2 = np.zeros(n_samples)
        for i in range(10, n_samples):
            weights = np.exp(-np.arange(10) / 5)  # 지수 감소 가중치
            recent_returns = returns_lag2[i-10:i]
            if len(recent_returns) > 0:
                temporal_attention_score_lag2[i] = np.average(np.abs(recent_returns), weights=weights)

        # 특징 중요도 점수 (변동성과 수익률의 조합)
        feature_importance_score_lag2 = np.abs(returns_lag2) * vol_5_lag2

        # 모든 새로운 특징들 결합
        kaggle_features = np.column_stack([
            intraday_pattern_lag2, volatility_regime_lag2, trend_strength_lag2,
            fractal_dimension_lag2, hurst_exponent_lag2, entropy_measure_lag2,
            price_acceleration_lag2, volatility_of_volatility_lag2, momentum_persistence_lag2,
            prediction_confidence_lag2, regime_classification_lag2,
            temporal_attention_score_lag2, feature_importance_score_lag2
        ])

        # 기존 특징과 결합
        X_combined = np.column_stack([features, kaggle_features])

        logger.info(f"캐글 기법 특징 생성 완료: {features.shape} -> {X_combined.shape}")

        return X_combined

    def validate_feature_safety(self, X_features, y):
        """특징 안전성 검증"""
        logger.info("=== 캐글 특징 안전성 검증 ===")

        if y is None:
            logger.warning("타겟이 없어 상관관계 검증 생략")
            return True

        dangerous_features = []
        correlations = []

        for i, feature_name in enumerate(self.feature_names_):
            if i < X_features.shape[1]:
                corr = abs(np.corrcoef(X_features[:, i], y)[0, 1])
                correlations.append(corr)

                if not np.isnan(corr) and corr > self.MAX_CORRELATION:
                    dangerous_features.append((feature_name, corr))

        max_corr = max(correlations) if correlations else 0

        logger.info(f"최대 상관관계: {max_corr:.3f}")
        logger.info(f"안전 기준: {self.MAX_CORRELATION}")

        if dangerous_features:
            logger.error(f"위험한 특징 발견: {len(dangerous_features)}개")
            for feature, corr in dangerous_features:
                logger.error(f"   {feature}: {corr:.3f}")
            return False

        logger.info("✅ 모든 캐글 특징이 안전 기준 통과")
        return True

class SafeKaggleEnhancedSystem:
    """안전한 캐글 기법 향상 시스템"""

    def __init__(self):
        self.data_processor = UltraSafeDataProcessor()
        self.kaggle_engineer = KaggleAdvancedFeatureEngineer(safety_mode=True)
        self.leakage_detector = AutoLeakageDetector()

        # 더 엄격한 안전 기준
        self.MAX_R2 = 0.12
        self.MAX_DIRECTION_ACC = 62.0

        logger.info("안전한 캐글 기법 향상 시스템 초기화")

    def run_kaggle_enhancement(self, data_path: str):
        """캐글 기법 기반 안전한 성능 향상"""
        logger.info("=" * 100)
        logger.info("🏆 캐글 기법 기반 안전한 성능 향상 시작")
        logger.info("=" * 100)

        # 1. 기본 안전 데이터 준비
        data_dict = self.data_processor.prepare_ultra_safe_data(data_path)
        X_base, y = data_dict['X'], data_dict['y']

        # 2. 데이터 분할
        split_point = int(len(X_base) * 0.8)
        X_train_base = X_base[:split_point]
        X_test_base = X_base[split_point:]
        y_train = y[:split_point]
        y_test = y[split_point:]

        # 3. 캐글 기법 적용
        logger.info("캐글 고급 특징 엔지니어링 적용")
        X_train_enhanced = self.kaggle_engineer.fit_transform(X_train_base)
        X_test_enhanced = self.kaggle_engineer.transform(X_test_base)

        # 4. 특징 안전성 검증
        is_safe = self.kaggle_engineer.validate_feature_safety(X_train_enhanced, y_train)

        if not is_safe:
            logger.error("❌ 캐글 특징이 안전 기준을 초과함")
            return None

        # 5. 간단한 모델로 성능 평가
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=50.0)  # 강한 정규화로 안전성 확보
        model.fit(X_train_enhanced, y_train)
        y_pred = model.predict(X_test_enhanced)

        # 6. 성능 지표 계산
        from sklearn.metrics import mean_squared_error, mean_absolute_error

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = 1 - (mse / np.var(y_test))

        # 방향 정확도
        direction_actual = (y_test > 0).astype(int)
        direction_pred = (y_pred > 0).astype(int)
        direction_accuracy = np.mean(direction_actual == direction_pred) * 100

        # 7. 안전성 최종 검증
        metrics = {
            'r2': r2,
            'direction_accuracy': direction_accuracy
        }

        final_safe = self.leakage_detector.validate_during_training(0, 'kaggle_enhanced', metrics)

        # 추가 안전 검증
        safe_performance = (r2 <= self.MAX_R2 and direction_accuracy <= self.MAX_DIRECTION_ACC)

        logger.info(f"\n{'='*60}")
        logger.info(f"🏆 캐글 기법 성능 향상 결과")
        logger.info(f"{'='*60}")
        logger.info(f"R²: {r2:.4f} ({'✅ 안전' if r2 <= self.MAX_R2 else '❌ 위험'})")
        logger.info(f"MAE: {mae:.4f}")
        logger.info(f"방향정확도: {direction_accuracy:.1f}% ({'✅ 안전' if direction_accuracy <= self.MAX_DIRECTION_ACC else '❌ 위험'})")
        logger.info(f"특징 수: {X_train_enhanced.shape[1]}개 (기존 대비 +{X_train_enhanced.shape[1] - X_train_base.shape[1]})")

        success = final_safe and safe_performance

        logger.info(f"최종 안전성: {'✅ 성공' if success else '❌ 실패'}")

        return {
            'enhanced_features': X_train_enhanced.shape[1],
            'performance': {
                'r2': r2,
                'mae': mae,
                'direction_accuracy': direction_accuracy
            },
            'safety_check': success,
            'feature_names': self.kaggle_engineer.feature_names_,
            'model': model,
            'kaggle_engineer': self.kaggle_engineer
        }

def main():
    """메인 실행 함수"""
    system = SafeKaggleEnhancedSystem()

    try:
        results = system.run_kaggle_enhancement(
            '/root/workspace/data/training/sp500_2020_2024_enhanced.csv'
        )

        if results is None:
            print("❌ 캐글 기법 적용 실패 - 안전 기준 위반")
            return None

        print(f"\n{'='*80}")
        print(f"🏆 캐글 기법 기반 성능 향상 완료")
        print(f"{'='*80}")

        perf = results['performance']
        print(f"\n📊 향상된 성능:")
        print(f"   R²: {perf['r2']:.4f}")
        print(f"   MAE: {perf['mae']:.4f}")
        print(f"   방향정확도: {perf['direction_accuracy']:.1f}%")

        print(f"\n🔧 특징 엔지니어링:")
        print(f"   총 특징 수: {results['enhanced_features']}개")
        print(f"   캐글 기법 적용: 13개 추가")

        print(f"\n🛡️ 안전성: {'✅ 통과' if results['safety_check'] else '❌ 실패'}")

        if results['safety_check']:
            print(f"\n🎉 성공: 캐글 우승자 기법으로 안전한 성능 향상 달성!")
        else:
            print(f"\n⚠️ 주의: 안전 기준 초과 - 추가 보정 필요")

        return results

    except Exception as e:
        logger.error(f"캐글 기법 향상 실패: {e}")
        return None

if __name__ == "__main__":
    result = main()