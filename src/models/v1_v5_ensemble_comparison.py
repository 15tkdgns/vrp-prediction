"""
V1 + V5 앙상블 성능 비교 시스템
단일 모델 대비 앙상블 효과 검증

비교 대상:
- V1 단독 (R² = 0.314)
- V5 단독 (R² = 0.302)
- V1 + V5 앙상블 (다양한 가중치)

목표: 앙상블로 성능 향상 및 안정성 확보
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json
import logging
from datetime import datetime
import warnings
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import VotingRegressor

warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/root/workspace/data/raw/v1_v5_ensemble_comparison.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PurgedKFoldSklearn:
    """sklearn 호환 Purged K-Fold Cross-Validation"""

    def __init__(self, n_splits=5, purge_length=5, embargo_length=5):
        self.n_splits = n_splits
        self.purge_length = purge_length
        self.embargo_length = embargo_length

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        indices = np.arange(n_samples)
        test_size = n_samples // self.n_splits
        splits = []

        for i in range(self.n_splits):
            test_start = i * test_size
            test_end = min((i + 1) * test_size, n_samples)
            test_indices = indices[test_start:test_end]

            purge_start = test_end
            purge_end = min(test_end + self.purge_length, n_samples)
            embargo_end = min(purge_end + self.embargo_length, n_samples)

            train_indices = np.concatenate([
                indices[:test_start],
                indices[embargo_end:]
            ])

            if len(train_indices) > 0 and len(test_indices) > 0:
                splits.append((train_indices, test_indices))

        return splits

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

class V1V5EnsembleComparator:
    """V1 + V5 앙상블 성능 비교기"""

    def __init__(self):
        self.cv = PurgedKFoldSklearn()
        self.scaler_v1 = StandardScaler()
        self.scaler_v5 = StandardScaler()
        self.results = {}

        # 모델 설정
        self.v1_config = {
            'alpha': 1.8523,
            'expected_r2': 0.314,
            'features': 12
        }

        self.v5_config = {
            'alphas': [31.7068, 17.4421, 161.7161],
            'expected_r2': 0.302,
            'features': 50
        }

    def load_spy_data(self):
        """SPY 데이터 로드"""
        logger.info("📊 SPY 데이터 로딩...")

        spy = yf.Ticker("SPY")
        data = spy.history(start="2015-01-01", end="2024-12-31")
        data['returns'] = np.log(data['Close'] / data['Close'].shift(1))
        data = data.dropna()

        logger.info(f"✅ 데이터 로딩 완료: {len(data)}개 관측치")
        return data

    def create_v1_features(self, data):
        """V1 특성 생성 (12개)"""
        returns = data['returns']
        features = pd.DataFrame(index=data.index)

        # 기본 변동성 (3개)
        features['vol_5'] = returns.rolling(5).std()
        features['vol_10'] = returns.rolling(10).std()
        features['vol_20'] = returns.rolling(20).std()

        # 기본 래그 (3개)
        features['return_lag_1'] = returns.shift(1)
        features['return_lag_2'] = returns.shift(2)
        features['return_lag_3'] = returns.shift(3)

        # 기본 통계 (4개)
        for window in [10, 20]:
            ma = returns.rolling(window).mean()
            std = returns.rolling(window).std()
            features[f'zscore_{window}'] = (returns - ma) / (std + 1e-8)
            features[f'momentum_{window}'] = returns.rolling(window).sum()

        # 기본 비율 (2개)
        features['vol_5_20_ratio'] = features['vol_5'] / (features['vol_20'] + 1e-8)
        features['vol_regime'] = (features['vol_5'] > features['vol_10']).astype(float)

        return self.finalize_features(features, returns)

    def create_v5_features(self, data):
        """V5 특성 생성 (50개 - 간소화)"""
        returns = data['returns']
        prices = data['Close']
        volume = data['Volume']
        features = pd.DataFrame(index=data.index)

        # V1 기본 특성들 포함
        v1_features, _ = self.create_v1_features(data)
        for col in v1_features.columns:
            if col != 'target_vol_5d':
                features[col] = v1_features[col]

        # 추가 변동성 (10개)
        for window in [3, 7, 12, 15, 25, 30, 40, 50, 60, 100]:
            features[f'vol_{window}'] = returns.rolling(window).std()

        # 고차 모멘트 (8개)
        for window in [5, 10, 15, 20]:
            features[f'skew_{window}'] = returns.rolling(window).skew()
            features[f'kurt_{window}'] = returns.rolling(window).kurt()

        # 가격 특성 (10개)
        for window in [5, 10, 20, 30, 50]:
            sma = prices.rolling(window).mean()
            features[f'price_sma_dev_{window}'] = (prices - sma) / sma
            features[f'price_mom_{window}'] = (prices / prices.shift(window)) - 1

        # 거래량 특성 (4개)
        for window in [10, 20]:
            vol_sma = volume.rolling(window).mean()
            features[f'volume_ratio_{window}'] = volume / (vol_sma + 1)
            features[f'price_volume_{window}'] = returns * (volume / vol_sma)

        # 추가 래그 (6개)
        for lag in [5, 7, 10]:
            features[f'return_lag_{lag}'] = returns.shift(lag)
            features[f'vol_lag_{lag}'] = features['vol_5'].shift(lag)

        # 상위 50개만 선택 (중복 제거)
        feature_cols = [col for col in features.columns if col != 'target_vol_5d']
        selected_cols = feature_cols[:50]  # 상위 50개

        final_features = pd.DataFrame(index=data.index)
        for col in selected_cols:
            final_features[col] = features[col]

        return self.finalize_features(final_features, returns)

    def finalize_features(self, features, returns):
        """특성 마무리 처리"""
        # 타겟: 5일 후 변동성
        target = []
        for i in range(len(returns)):
            if i + 5 < len(returns):
                future_vol = returns.iloc[i+1:i+6].std()
                target.append(future_vol)
            else:
                target.append(np.nan)

        features['target_vol_5d'] = target
        features = features.dropna()

        X = features.drop('target_vol_5d', axis=1)
        y = features['target_vol_5d']

        return X, y

    def test_single_models(self):
        """단일 모델 성능 테스트"""
        logger.info("🔍 단일 모델 성능 테스트...")

        data = self.load_spy_data()

        # V1 모델 테스트
        logger.info("   V1 모델 테스트...")
        X_v1, y_v1 = self.create_v1_features(data)
        X_v1_scaled = self.scaler_v1.fit_transform(X_v1)

        v1_model = Ridge(alpha=self.v1_config['alpha'])
        v1_scores = []

        splits = list(self.cv.split(X_v1_scaled))
        for train_idx, val_idx in splits:
            X_train, X_val = X_v1_scaled[train_idx], X_v1_scaled[val_idx]
            y_train, y_val = y_v1.iloc[train_idx], y_v1.iloc[val_idx]

            v1_model.fit(X_train, y_train)
            val_pred = v1_model.predict(X_val)
            r2 = r2_score(y_val, val_pred)
            v1_scores.append(r2)

        v1_performance = {
            'scores': [float(s) for s in v1_scores],
            'mean_r2': float(np.mean(v1_scores)),
            'std_r2': float(np.std(v1_scores)),
            'features_count': X_v1.shape[1],
            'samples_count': len(X_v1)
        }

        logger.info(f"   V1 성능: R² = {v1_performance['mean_r2']:.4f} ± {v1_performance['std_r2']:.4f}")

        # V5 모델 테스트
        logger.info("   V5 모델 테스트...")
        X_v5, y_v5 = self.create_v5_features(data)
        X_v5_scaled = self.scaler_v5.fit_transform(X_v5)

        v5_models = [
            ('ridge1', Ridge(alpha=self.v5_config['alphas'][0], random_state=42)),
            ('ridge2', Ridge(alpha=self.v5_config['alphas'][1], random_state=43)),
            ('ridge3', Ridge(alpha=self.v5_config['alphas'][2], random_state=44))
        ]
        v5_ensemble = VotingRegressor(estimators=v5_models)
        v5_scores = []

        splits = list(self.cv.split(X_v5_scaled))
        for train_idx, val_idx in splits:
            X_train, X_val = X_v5_scaled[train_idx], X_v5_scaled[val_idx]
            y_train, y_val = y_v5.iloc[train_idx], y_v5.iloc[val_idx]

            v5_ensemble.fit(X_train, y_train)
            val_pred = v5_ensemble.predict(X_val)
            r2 = r2_score(y_val, val_pred)
            v5_scores.append(r2)

        v5_performance = {
            'scores': [float(s) for s in v5_scores],
            'mean_r2': float(np.mean(v5_scores)),
            'std_r2': float(np.std(v5_scores)),
            'features_count': X_v5.shape[1],
            'samples_count': len(X_v5)
        }

        logger.info(f"   V5 성능: R² = {v5_performance['mean_r2']:.4f} ± {v5_performance['std_r2']:.4f}")

        return v1_performance, v5_performance, (X_v1_scaled, y_v1), (X_v5_scaled, y_v5)

    def test_ensemble_combinations(self, v1_data, v5_data):
        """다양한 앙상블 조합 테스트"""
        logger.info("🔍 V1 + V5 앙상블 조합 테스트...")

        X_v1, y_v1 = v1_data
        X_v5, y_v5 = v5_data

        # 공통 인덱스 찾기 (두 데이터셋이 다를 수 있음)
        common_indices = set(y_v1.index) & set(y_v5.index)
        common_indices = sorted(list(common_indices))

        if len(common_indices) < 1000:
            logger.warning(f"⚠️ 공통 샘플 수가 적음: {len(common_indices)}개")

        # 공통 샘플로 정렬
        v1_common_mask = y_v1.index.isin(common_indices)
        v5_common_mask = y_v5.index.isin(common_indices)

        X_v1_common = X_v1[v1_common_mask]
        y_v1_common = y_v1[v1_common_mask]
        X_v5_common = X_v5[v5_common_mask]
        y_v5_common = y_v5[v5_common_mask]

        logger.info(f"   공통 샘플: {len(y_v1_common)}개")

        # 다양한 가중치 조합 테스트
        weight_combinations = [
            (1.0, 0.0),    # V1 only
            (0.0, 1.0),    # V5 only
            (0.7, 0.3),    # V1 우세
            (0.6, 0.4),    # V1 우세 (약간)
            (0.5, 0.5),    # 균등
            (0.4, 0.6),    # V5 우세 (약간)
            (0.3, 0.7),    # V5 우세
        ]

        ensemble_results = {}

        for w1, w5 in weight_combinations:
            logger.info(f"   가중치 테스트: V1={w1:.1f}, V5={w5:.1f}")

            ensemble_scores = []
            splits = list(self.cv.split(X_v1_common))

            for train_idx, val_idx in splits:
                # V1 모델
                X_v1_train, X_v1_val = X_v1_common[train_idx], X_v1_common[val_idx]
                y_train, y_val = y_v1_common.iloc[train_idx], y_v1_common.iloc[val_idx]

                v1_model = Ridge(alpha=self.v1_config['alpha'])
                v1_model.fit(X_v1_train, y_train)
                v1_pred = v1_model.predict(X_v1_val)

                # V5 모델
                X_v5_train, X_v5_val = X_v5_common[train_idx], X_v5_common[val_idx]

                v5_models = [
                    ('ridge1', Ridge(alpha=self.v5_config['alphas'][0], random_state=42)),
                    ('ridge2', Ridge(alpha=self.v5_config['alphas'][1], random_state=43)),
                    ('ridge3', Ridge(alpha=self.v5_config['alphas'][2], random_state=44))
                ]
                v5_ensemble = VotingRegressor(estimators=v5_models)
                v5_ensemble.fit(X_v5_train, y_train)
                v5_pred = v5_ensemble.predict(X_v5_val)

                # 가중 앙상블
                ensemble_pred = w1 * v1_pred + w5 * v5_pred

                # 성능 계산
                r2 = r2_score(y_val, ensemble_pred)
                ensemble_scores.append(r2)

            ensemble_performance = {
                'weights': (w1, w5),
                'scores': [float(s) for s in ensemble_scores],
                'mean_r2': float(np.mean(ensemble_scores)),
                'std_r2': float(np.std(ensemble_scores))
            }

            ensemble_results[f'w{w1:.1f}_{w5:.1f}'] = ensemble_performance

            logger.info(f"      결과: R² = {ensemble_performance['mean_r2']:.4f} ± {ensemble_performance['std_r2']:.4f}")

        return ensemble_results

    def find_optimal_ensemble(self, ensemble_results):
        """최적 앙상블 조합 찾기"""
        logger.info("🎯 최적 앙상블 조합 분석...")

        best_combination = None
        best_r2 = -1
        best_stability = float('inf')

        performance_summary = []

        for combo_name, results in ensemble_results.items():
            mean_r2 = results['mean_r2']
            std_r2 = results['std_r2']
            weights = results['weights']

            # 성능-안정성 스코어 (높은 R² + 낮은 표준편차)
            stability_penalty = std_r2 * 2  # 안정성 중요도 2배
            composite_score = mean_r2 - stability_penalty

            performance_summary.append({
                'combination': combo_name,
                'weights': weights,
                'mean_r2': mean_r2,
                'std_r2': std_r2,
                'composite_score': composite_score
            })

            # 최고 R² 업데이트
            if mean_r2 > best_r2:
                best_r2 = mean_r2
                best_combination = combo_name
                best_stability = std_r2

        # 정렬 (composite_score 기준)
        performance_summary.sort(key=lambda x: x['composite_score'], reverse=True)

        logger.info("   📊 성능-안정성 순위:")
        for i, result in enumerate(performance_summary[:5], 1):
            w1, w5 = result['weights']
            logger.info(f"   {i}위: V1={w1:.1f}/V5={w5:.1f} → R²={result['mean_r2']:.4f}±{result['std_r2']:.4f}")

        optimal_result = performance_summary[0]
        logger.info(f"🏆 최적 조합: {optimal_result['combination']} (종합 점수: {optimal_result['composite_score']:.4f})")

        return optimal_result, performance_summary

    def generate_comparison_report(self, v1_perf, v5_perf, ensemble_results, optimal_result):
        """종합 비교 보고서 생성"""
        logger.info("📋 V1 + V5 앙상블 종합 비교 보고서 생성...")

        # 최고 성능 찾기
        best_ensemble = max(ensemble_results.values(), key=lambda x: x['mean_r2'])

        comparison_report = {
            'analysis_date': datetime.now().isoformat(),
            'single_model_performance': {
                'V1': v1_perf,
                'V5': v5_perf
            },
            'ensemble_results': ensemble_results,
            'optimal_ensemble': optimal_result,
            'performance_comparison': {
                'V1_only': v1_perf['mean_r2'],
                'V5_only': v5_perf['mean_r2'],
                'best_ensemble': best_ensemble['mean_r2'],
                'optimal_ensemble': optimal_result['mean_r2']
            },
            'improvement_analysis': {
                'ensemble_vs_v1': float(best_ensemble['mean_r2'] - v1_perf['mean_r2']),
                'ensemble_vs_v5': float(best_ensemble['mean_r2'] - v5_perf['mean_r2']),
                'optimal_vs_best_single': float(optimal_result['mean_r2'] - max(v1_perf['mean_r2'], v5_perf['mean_r2'])),
                'improvement_percentage': float((optimal_result['mean_r2'] / max(v1_perf['mean_r2'], v5_perf['mean_r2']) - 1) * 100)
            },
            'stability_analysis': {
                'V1_std': v1_perf['std_r2'],
                'V5_std': v5_perf['std_r2'],
                'optimal_ensemble_std': optimal_result['std_r2'],
                'stability_improvement': max(v1_perf['std_r2'], v5_perf['std_r2']) - optimal_result['std_r2']
            },
            'recommendations': self.generate_recommendations(v1_perf, v5_perf, optimal_result)
        }

        # 결과 저장
        save_path = '/root/workspace/data/raw/v1_v5_ensemble_comparison_results.json'
        with open(save_path, 'w') as f:
            json.dump(comparison_report, f, indent=2)

        logger.info("="*70)
        logger.info("🎯 V1 + V5 앙상블 비교 완료")
        logger.info(f"📊 V1 단독: R² = {v1_perf['mean_r2']:.4f}")
        logger.info(f"📊 V5 단독: R² = {v5_perf['mean_r2']:.4f}")
        logger.info(f"🏆 최적 앙상블: R² = {optimal_result['mean_r2']:.4f}")
        logger.info(f"📈 성능 향상: {comparison_report['improvement_analysis']['improvement_percentage']:+.2f}%")
        logger.info(f"💾 상세 결과: {save_path}")
        logger.info("="*70)

        return comparison_report

    def generate_recommendations(self, v1_perf, v5_perf, optimal_result):
        """권장사항 생성"""
        recommendations = []

        optimal_weights = optimal_result['weights']
        improvement_pct = (optimal_result['mean_r2'] / max(v1_perf['mean_r2'], v5_perf['mean_r2']) - 1) * 100

        if improvement_pct > 2:
            recommendations.extend([
                f"✅ 앙상블 권장: V1={optimal_weights[0]:.1f}, V5={optimal_weights[1]:.1f}",
                f"✅ 성능 향상: {improvement_pct:+.2f}% 개선",
                "✅ 단일 모델 대비 앙상블 우수"
            ])
        elif improvement_pct > 0:
            recommendations.extend([
                f"🔶 앙상블 고려: 소폭 개선 ({improvement_pct:+.2f}%)",
                f"🔶 가중치: V1={optimal_weights[0]:.1f}, V5={optimal_weights[1]:.1f}",
                "🔶 복잡도 vs 성능 트레이드오프 검토"
            ])
        else:
            recommendations.extend([
                "❌ 앙상블 불필요: 단일 모델이 더 우수",
                f"❌ V1 단독 사용 권장 (R² = {v1_perf['mean_r2']:.4f})",
                "❌ 복잡성 증가 대비 성능 이익 없음"
            ])

        # 안정성 분석
        optimal_std = optimal_result['std_r2']
        best_single_std = min(v1_perf['std_r2'], v5_perf['std_r2'])

        if optimal_std < best_single_std:
            recommendations.append("✅ 앙상블이 안정성도 향상")
        else:
            recommendations.append("⚠️ 앙상블이 변동성 증가")

        return recommendations

    def run_full_comparison(self):
        """전체 비교 분석 실행"""
        logger.info("🚀 V1 + V5 앙상블 전체 비교 분석 시작")

        try:
            # 1. 단일 모델 성능 테스트
            v1_perf, v5_perf, v1_data, v5_data = self.test_single_models()

            # 2. 앙상블 조합 테스트
            ensemble_results = self.test_ensemble_combinations(v1_data, v5_data)

            # 3. 최적 조합 찾기
            optimal_result, performance_summary = self.find_optimal_ensemble(ensemble_results)

            # 4. 종합 보고서 생성
            report = self.generate_comparison_report(v1_perf, v5_perf, ensemble_results, optimal_result)

            logger.info("💡 최종 권장사항:")
            for rec in report['recommendations']:
                logger.info(f"   {rec}")

            return report

        except Exception as e:
            logger.error(f"❌ 앙상블 비교 실패: {str(e)}")
            raise

def main():
    """메인 실행"""
    logger.info("🎯 V1 + V5 앙상블 성능 비교 시작")

    comparator = V1V5EnsembleComparator()

    try:
        results = comparator.run_full_comparison()
        return results

    except Exception as e:
        logger.error(f"❌ 전체 비교 실패: {str(e)}")
        raise

if __name__ == "__main__":
    main()