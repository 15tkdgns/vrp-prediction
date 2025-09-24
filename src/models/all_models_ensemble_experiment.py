"""
전체 모델 앙상블 조합 실험 시스템
V1, V2, V4, V5 모든 가능한 조합 테스트

실험 범위:
1. 2-모델 조합: 6개 (V1+V2, V1+V4, V1+V5, V2+V4, V2+V5, V4+V5)
2. 3-모델 조합: 4개 (V1+V2+V4, V1+V2+V5, V1+V4+V5, V2+V4+V5)
3. 4-모델 조합: 1개 (V1+V2+V4+V5)

각 조합마다 다양한 가중치로 최적화
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
from sklearn.metrics import r2_score
from sklearn.ensemble import VotingRegressor
from itertools import combinations
import time

warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/root/workspace/data/raw/all_models_ensemble_experiment.log'),
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

class AllModelsEnsembleExperimenter:
    """전체 모델 앙상블 실험기"""

    def __init__(self):
        self.cv = PurgedKFoldSklearn()
        self.scalers = {
            'V1': StandardScaler(),
            'V2': StandardScaler(),
            'V4': StandardScaler(),
            'V5': StandardScaler()
        }

        # 모델 설정
        self.model_configs = {
            'V1': {
                'alpha': 1.8523,
                'expected_r2': 0.314,
                'description': '기본 경사하강법 (12개 특성)'
            },
            'V2': {
                'alpha': 19.5029,
                'expected_r2': 0.297,
                'description': '완전 특성 최적화 (30개 특성)'
            },
            'V4': {
                'alpha': 5.0,
                'expected_r2': 0.262,
                'description': '확장 특성 최적화 (55개 특성)'
            },
            'V5': {
                'alphas': [31.7068, 17.4421, 161.7161],
                'expected_r2': 0.302,
                'description': '3-Ridge 앙상블 (50개 특성)'
            }
        }

        self.model_data = {}
        self.experiment_results = {}

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

    def create_v2_features(self, data):
        """V2 특성 생성 (30개)"""
        returns = data['returns']
        features = pd.DataFrame(index=data.index)

        # 변동성 (6개)
        for window in [3, 5, 10, 15, 20, 30]:
            features[f'vol_{window}'] = returns.rolling(window).std()

        # 통계적 모멘트 (6개)
        for window in [5, 10, 20]:
            features[f'skew_{window}'] = returns.rolling(window).skew()
            features[f'kurt_{window}'] = returns.rolling(window).kurt()

        # 래그 (6개)
        for lag in [1, 2, 3]:
            features[f'return_lag_{lag}'] = returns.shift(lag)
            features[f'vol_lag_{lag}'] = features['vol_5'].shift(lag)

        # 변동성 체제 (4개)
        short_vol = features['vol_5']
        medium_vol = features['vol_20']
        long_vol = features['vol_30']

        features['vol_regime_short'] = (short_vol > medium_vol).astype(float)
        features['vol_regime_medium'] = (medium_vol > long_vol).astype(float)
        features['vol_expansion'] = short_vol / (long_vol + 1e-8)
        features['vol_contraction'] = long_vol / (short_vol + 1e-8)

        # 통계 지표 (5개)
        for window in [10, 20, 30]:
            ma = returns.rolling(window).mean()
            std = returns.rolling(window).std()
            features[f'zscore_{window}'] = (returns - ma) / (std + 1e-8)

        for window in [10, 20]:
            ma = returns.rolling(window).mean()
            std = returns.rolling(window).std()
            features[f'sharpe_{window}'] = (ma * np.sqrt(252)) / (std + 1e-8)

        # 상호작용 (3개)
        features['vol_5_20_ratio'] = features['vol_5'] / (features['vol_20'] + 1e-8)
        features['vol_10_30_ratio'] = features['vol_10'] / (features['vol_30'] + 1e-8)
        features['vol_price_interaction'] = features['vol_20'] * returns

        return self.finalize_features(features, returns)

    def create_v4_features(self, data):
        """V4 특성 생성 (55개 - 간소화)"""
        returns = data['returns']
        prices = data['Close']
        volume = data['Volume']
        features = pd.DataFrame(index=data.index)

        # V2 기본 특성 포함
        v2_features, _ = self.create_v2_features(data)
        for col in v2_features.columns:
            if col != 'target_vol_5d':
                features[col] = v2_features[col]

        # 추가 변동성 (10개)
        for window in [7, 12, 25, 40, 50, 60, 100, 120, 150, 200]:
            if len(features) > window:
                features[f'vol_{window}'] = returns.rolling(window).std()

        # 고차 모멘트 (8개)
        for window in [15, 25]:
            features[f'skew_{window}'] = returns.rolling(window).skew()
            features[f'kurt_{window}'] = returns.rolling(window).kurt()
            features[f'skew_{window}_lag1'] = features[f'skew_{window}'].shift(1)
            features[f'kurt_{window}_lag1'] = features[f'kurt_{window}'].shift(1)

        # 가격 특성 (7개)
        for window in [5, 10, 30, 50]:
            sma = prices.rolling(window).mean()
            features[f'price_sma_dev_{window}'] = (prices - sma) / sma

        for window in [20, 40, 60]:
            features[f'price_mom_{window}'] = (prices / prices.shift(window)) - 1

        # 상위 55개만 선택
        feature_cols = [col for col in features.columns if col != 'target_vol_5d']
        selected_cols = feature_cols[:55]

        final_features = pd.DataFrame(index=data.index)
        for col in selected_cols:
            final_features[col] = features[col]

        return self.finalize_features(final_features, returns)

    def create_v5_features(self, data):
        """V5 특성 생성 (50개)"""
        returns = data['returns']
        features = pd.DataFrame(index=data.index)

        # V4 기반에서 50개 선택
        v4_features, _ = self.create_v4_features(data)
        feature_cols = [col for col in v4_features.columns if col != 'target_vol_5d']
        selected_cols = feature_cols[:50]

        for col in selected_cols:
            features[col] = v4_features[col]

        return self.finalize_features(features, returns)

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

    def prepare_all_model_data(self):
        """모든 모델 데이터 준비"""
        logger.info("🔧 모든 모델 데이터 준비 중...")

        data = self.load_spy_data()

        # 각 모델별 데이터 생성
        for model_name in ['V1', 'V2', 'V4', 'V5']:
            logger.info(f"   {model_name} 특성 생성...")

            if model_name == 'V1':
                X, y = self.create_v1_features(data)
            elif model_name == 'V2':
                X, y = self.create_v2_features(data)
            elif model_name == 'V4':
                X, y = self.create_v4_features(data)
            elif model_name == 'V5':
                X, y = self.create_v5_features(data)

            X_scaled = self.scalers[model_name].fit_transform(X)

            self.model_data[model_name] = {
                'X_scaled': X_scaled,
                'y': y,
                'features_count': X.shape[1],
                'samples_count': len(X)
            }

            logger.info(f"   {model_name}: {X.shape[1]}개 특성, {len(X)}개 샘플")

    def get_single_model_prediction(self, model_name, X_train, y_train, X_test):
        """단일 모델 예측"""
        if model_name == 'V5':
            # V5는 앙상블
            models = [
                ('ridge1', Ridge(alpha=self.model_configs['V5']['alphas'][0], random_state=42)),
                ('ridge2', Ridge(alpha=self.model_configs['V5']['alphas'][1], random_state=43)),
                ('ridge3', Ridge(alpha=self.model_configs['V5']['alphas'][2], random_state=44))
            ]
            ensemble = VotingRegressor(estimators=models)
            ensemble.fit(X_train, y_train)
            return ensemble.predict(X_test)
        else:
            # 단일 Ridge 모델
            alpha = self.model_configs[model_name]['alpha']
            model = Ridge(alpha=alpha)
            model.fit(X_train, y_train)
            return model.predict(X_test)

    def find_common_samples(self, model_names):
        """모델들 간 공통 샘플 찾기"""
        common_indices = None

        for model_name in model_names:
            model_indices = set(self.model_data[model_name]['y'].index)
            if common_indices is None:
                common_indices = model_indices
            else:
                common_indices = common_indices & model_indices

        return sorted(list(common_indices))

    def test_ensemble_combination(self, model_names, weight_combinations):
        """특정 모델 조합의 앙상블 테스트"""
        combo_name = '+'.join(model_names)
        logger.info(f"🔍 {combo_name} 앙상블 테스트...")

        # 공통 샘플 찾기
        common_indices = self.find_common_samples(model_names)

        if len(common_indices) < 1000:
            logger.warning(f"⚠️ {combo_name} 공통 샘플 부족: {len(common_indices)}개")
            return None

        logger.info(f"   공통 샘플: {len(common_indices)}개")

        # 공통 샘플로 데이터 정렬
        aligned_data = {}
        for model_name in model_names:
            model_data = self.model_data[model_name]
            mask = model_data['y'].index.isin(common_indices)
            aligned_data[model_name] = {
                'X': model_data['X_scaled'][mask],
                'y': model_data['y'][mask]
            }

        # 각 가중치 조합 테스트
        combo_results = {}

        for weights in weight_combinations:
            weight_name = '_'.join([f'{w:.1f}' for w in weights])
            logger.info(f"   가중치 {weight_name}: {dict(zip(model_names, weights))}")

            ensemble_scores = []
            y_reference = aligned_data[model_names[0]]['y']  # 참조용

            splits = list(self.cv.split(aligned_data[model_names[0]]['X']))

            for train_idx, val_idx in splits:
                # 각 모델 예측
                predictions = {}
                y_val = y_reference.iloc[val_idx]

                for model_name in model_names:
                    X_train = aligned_data[model_name]['X'][train_idx]
                    X_val = aligned_data[model_name]['X'][val_idx]
                    y_train = aligned_data[model_name]['y'].iloc[train_idx]

                    pred = self.get_single_model_prediction(model_name, X_train, y_train, X_val)
                    predictions[model_name] = pred

                # 가중 앙상블
                ensemble_pred = np.zeros_like(predictions[model_names[0]])
                for i, model_name in enumerate(model_names):
                    ensemble_pred += weights[i] * predictions[model_name]

                # 성능 계산
                r2 = r2_score(y_val, ensemble_pred)
                ensemble_scores.append(r2)

            combo_results[weight_name] = {
                'weights': weights,
                'scores': [float(s) for s in ensemble_scores],
                'mean_r2': float(np.mean(ensemble_scores)),
                'std_r2': float(np.std(ensemble_scores))
            }

            logger.info(f"      결과: R² = {combo_results[weight_name]['mean_r2']:.4f} ± {combo_results[weight_name]['std_r2']:.4f}")

        return combo_results

    def generate_weight_combinations(self, n_models):
        """n개 모델용 가중치 조합 생성"""
        if n_models == 2:
            return [
                (1.0, 0.0), (0.0, 1.0),  # 단독
                (0.8, 0.2), (0.7, 0.3), (0.6, 0.4), (0.5, 0.5),  # 첫 번째 우세
                (0.4, 0.6), (0.3, 0.7), (0.2, 0.8)  # 두 번째 우세
            ]
        elif n_models == 3:
            return [
                (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),  # 단독
                (0.6, 0.3, 0.1), (0.6, 0.2, 0.2), (0.5, 0.3, 0.2),  # 첫 번째 우세
                (0.4, 0.4, 0.2), (0.4, 0.3, 0.3), (1/3, 1/3, 1/3),  # 균등 및 혼합
                (0.3, 0.5, 0.2), (0.2, 0.6, 0.2), (0.2, 0.4, 0.4)   # 다양한 조합
            ]
        elif n_models == 4:
            return [
                (1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0),  # 단독
                (0.5, 0.3, 0.1, 0.1), (0.4, 0.3, 0.2, 0.1), (0.4, 0.2, 0.2, 0.2),  # 첫 번째 우세
                (0.3, 0.3, 0.2, 0.2), (0.25, 0.25, 0.25, 0.25),  # 균등
                (0.2, 0.4, 0.3, 0.1), (0.1, 0.4, 0.3, 0.2), (0.2, 0.3, 0.3, 0.2)  # 다양한 조합
            ]

    def run_all_combinations(self):
        """모든 가능한 조합 실험"""
        logger.info("🚀 모든 모델 앙상블 조합 실험 시작")

        # 데이터 준비
        self.prepare_all_model_data()

        model_names = list(self.model_configs.keys())

        # 2-모델 조합
        logger.info("📊 2-모델 조합 실험...")
        two_model_combos = list(combinations(model_names, 2))

        for combo in two_model_combos:
            weights = self.generate_weight_combinations(2)
            result = self.test_ensemble_combination(list(combo), weights)
            if result:
                self.experiment_results[f'2_models_{combo[0]}_{combo[1]}'] = result

        # 3-모델 조합
        logger.info("📊 3-모델 조합 실험...")
        three_model_combos = list(combinations(model_names, 3))

        for combo in three_model_combos:
            weights = self.generate_weight_combinations(3)
            result = self.test_ensemble_combination(list(combo), weights)
            if result:
                self.experiment_results[f'3_models_{combo[0]}_{combo[1]}_{combo[2]}'] = result

        # 4-모델 조합
        logger.info("📊 4-모델 조합 실험...")
        weights = self.generate_weight_combinations(4)
        result = self.test_ensemble_combination(model_names, weights)
        if result:
            self.experiment_results['4_models_V1_V2_V4_V5'] = result

    def find_global_optimum(self):
        """전체 실험에서 최적 조합 찾기"""
        logger.info("🎯 전체 실험 최적 조합 분석...")

        all_results = []

        for combo_name, combo_results in self.experiment_results.items():
            for weight_name, result in combo_results.items():
                all_results.append({
                    'combination': combo_name,
                    'weight_name': weight_name,
                    'weights': result['weights'],
                    'mean_r2': result['mean_r2'],
                    'std_r2': result['std_r2'],
                    'composite_score': result['mean_r2'] - (result['std_r2'] * 2)  # 안정성 고려
                })

        # 성능 순으로 정렬
        all_results.sort(key=lambda x: x['mean_r2'], reverse=True)

        logger.info("📊 전체 실험 상위 10개 결과:")
        for i, result in enumerate(all_results[:10], 1):
            logger.info(f"   {i:2d}위: {result['combination']} (가중치: {result['weight_name']}) → R²={result['mean_r2']:.4f}±{result['std_r2']:.4f}")

        return all_results

    def generate_final_report(self, all_results):
        """최종 종합 보고서 생성"""
        logger.info("📋 전체 앙상블 실험 최종 보고서 생성...")

        # 최고 성능 결과들
        top_10 = all_results[:10]
        best_result = all_results[0]

        # 카테고리별 최고 성능
        category_best = {}
        for result in all_results:
            category = result['combination'].split('_')[0] + '_models'
            if category not in category_best:
                category_best[category] = result

        # 단일 모델 성능 (비교용)
        single_model_performance = {}
        for model_name in self.model_configs.keys():
            for result in all_results:
                if result['combination'].endswith(f'_{model_name}_{model_name}') or \
                   (len(result['weights']) == 1) or \
                   (len(result['weights']) > 1 and sum([w for i, w in enumerate(result['weights']) if i == 0]) == 1.0):
                    continue
            # 단일 모델 결과 찾기 (가중치가 [1,0,...] 형태)
            for combo_name, combo_results in self.experiment_results.items():
                if model_name in combo_name:
                    for weight_name, weight_result in combo_results.items():
                        weights = weight_result['weights']
                        # 해당 모델만 1.0인 경우 찾기
                        model_idx = combo_name.split('_')[2:].index(model_name) if len(combo_name.split('_')) > 2 else None
                        if model_idx is not None and len(weights) > model_idx and weights[model_idx] == 1.0:
                            single_model_performance[model_name] = weight_result['mean_r2']
                            break

        final_report = {
            'experiment_date': datetime.now().isoformat(),
            'total_combinations_tested': len(all_results),
            'best_overall': {
                'combination': best_result['combination'],
                'weights': best_result['weights'],
                'performance': {
                    'mean_r2': best_result['mean_r2'],
                    'std_r2': best_result['std_r2'],
                    'composite_score': best_result['composite_score']
                }
            },
            'top_10_results': top_10,
            'category_best': category_best,
            'single_model_baseline': single_model_performance,
            'model_configs': self.model_configs,
            'experiment_summary': {
                '2_model_combinations': len([r for r in all_results if r['combination'].startswith('2_models')]),
                '3_model_combinations': len([r for r in all_results if r['combination'].startswith('3_models')]),
                '4_model_combinations': len([r for r in all_results if r['combination'].startswith('4_models')])
            },
            'recommendations': self.generate_final_recommendations(best_result, category_best, single_model_performance)
        }

        # 결과 저장
        save_path = '/root/workspace/data/raw/all_models_ensemble_experiment_results.json'
        with open(save_path, 'w') as f:
            json.dump(final_report, f, indent=2)

        logger.info("="*80)
        logger.info("🏆 전체 모델 앙상블 실험 완료")
        logger.info(f"📊 총 테스트 조합: {len(all_results)}개")
        logger.info(f"🥇 최고 성능: {best_result['combination']} → R² = {best_result['mean_r2']:.4f}")
        logger.info(f"💾 상세 결과: {save_path}")
        logger.info("="*80)

        return final_report

    def generate_final_recommendations(self, best_result, category_best, single_baseline):
        """최종 권장사항 생성"""
        recommendations = []

        best_r2 = best_result['mean_r2']
        best_single = max(single_baseline.values()) if single_baseline else 0.31

        improvement = (best_r2 - best_single) / best_single * 100

        recommendations.append(f"🏆 최적 앙상블: {best_result['combination']}")
        recommendations.append(f"🎯 최적 가중치: {best_result['weights']}")
        recommendations.append(f"📈 최고 성능: R² = {best_r2:.4f}")

        if improvement > 5:
            recommendations.append(f"✅ 강력 권장: {improvement:+.1f}% 성능 향상")
        elif improvement > 2:
            recommendations.append(f"✅ 권장: {improvement:+.1f}% 성능 향상")
        elif improvement > 0:
            recommendations.append(f"🔶 고려: {improvement:+.1f}% 소폭 향상")
        else:
            recommendations.append("❌ 앙상블 불필요: 단일 모델이 우수")

        # 카테고리별 권장사항
        for category, result in category_best.items():
            recommendations.append(f"📊 {category} 최고: {result['combination']} (R² = {result['mean_r2']:.4f})")

        return recommendations

    def run_complete_experiment(self):
        """전체 실험 실행"""
        start_time = time.time()

        try:
            # 모든 조합 실험
            self.run_all_combinations()

            # 최적 조합 찾기
            all_results = self.find_global_optimum()

            # 최종 보고서 생성
            final_report = self.generate_final_report(all_results)

            elapsed_time = time.time() - start_time
            logger.info(f"⏱️ 총 실험 시간: {elapsed_time:.1f}초")

            # 권장사항 출력
            logger.info("💡 최종 권장사항:")
            for rec in final_report['recommendations']:
                logger.info(f"   {rec}")

            return final_report

        except Exception as e:
            logger.error(f"❌ 전체 실험 실패: {str(e)}")
            raise

def main():
    """메인 실행"""
    logger.info("🎯 전체 모델 앙상블 조합 실험 시작")

    experimenter = AllModelsEnsembleExperimenter()

    try:
        results = experimenter.run_complete_experiment()
        return results

    except Exception as e:
        logger.error(f"❌ 실험 실행 실패: {str(e)}")
        raise

if __name__ == "__main__":
    main()