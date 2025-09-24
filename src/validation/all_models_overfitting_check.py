"""
전체 모델 (V1-V5) 과적합 종합 검증 시스템
V2에서 과적합이 발견되었으므로, 모든 모델의 안전성을 검증

검증 대상:
- V1: alpha=1.8523, 14개 특성, R²=0.2775
- V2: alpha=19.5029, 30개 특성, R²=0.3256 (과적합 확인됨)
- V3: sklearn, alpha=1.0, 31개 특성, R²=0.2750
- V4: alpha=5.0, 66개 특성, R²=0.3222
- V5: 앙상블, R²=0.2289

목표: 실제 프로덕션에서 안전하게 사용할 수 있는 모델 찾기
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
        logging.FileHandler('/root/workspace/data/raw/all_models_overfitting_check.log'),
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

class AllModelsOverfittingChecker:
    """전체 모델 과적합 검증기"""

    def __init__(self):
        self.cv = PurgedKFoldSklearn()
        self.scaler = StandardScaler()
        self.results = {}

        # 모델 설정들
        self.model_configs = {
            'V1': {
                'alpha': 1.8523,
                'features': 'v1_features',  # 14개
                'reported_r2': 0.2775,
                'description': '기본 경사하강법'
            },
            'V2': {
                'alpha': 19.5029,
                'features': 'v2_features',  # 30개
                'reported_r2': 0.3256,
                'description': '완전 특성 + 최적화 (과적합 확인됨)'
            },
            'V3': {
                'alpha': 1.0,
                'features': 'v3_features',  # 31개
                'reported_r2': 0.2750,
                'description': 'sklearn 호환 최적화'
            },
            'V4': {
                'alpha': 5.0,
                'features': 'v4_features',  # 66개
                'reported_r2': 0.3222,
                'description': '확장 특성 최적화'
            },
            'V5': {
                'alpha': [31.7068, 17.4421, 161.7161],  # 앙상블
                'features': 'v5_features',  # 115개 → 간소화
                'reported_r2': 0.2289,
                'description': '앙상블 최적화'
            }
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
        """V1 특성 (14개) - 기본적인 것만"""
        returns = data['returns']
        features = pd.DataFrame(index=data.index)

        # 기본 변동성
        for window in [5, 10, 20]:
            features[f'vol_{window}'] = returns.rolling(window).std()

        # 기본 래그
        for lag in [1, 2, 3]:
            features[f'return_lag_{lag}'] = returns.shift(lag)

        # 기본 통계
        for window in [10, 20]:
            ma = returns.rolling(window).mean()
            std = returns.rolling(window).std()
            features[f'zscore_{window}'] = (returns - ma) / (std + 1e-8)
            features[f'momentum_{window}'] = returns.rolling(window).sum()

        # 기본 비율
        features['vol_5_20_ratio'] = features['vol_5'] / (features['vol_20'] + 1e-8)
        features['vol_regime'] = (features['vol_5'] > features['vol_10']).astype(float)

        return self.finalize_features(features, returns)

    def create_v2_features(self, data):
        """V2 특성 (30개) - V2와 동일"""
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

    def create_v3_features(self, data):
        """V3 특성 (31개) - V2 + 1개 추가"""
        returns = data['returns']
        features = pd.DataFrame(index=data.index)

        # V2 기반으로 시작
        v2_data = self.create_v2_features(data)
        features = v2_data[0].copy()

        # 추가 특성 1개
        features['momentum_15'] = returns.rolling(15).sum()

        return self.finalize_features(features, returns)

    def create_v4_features(self, data):
        """V4 특성 (66개) - 확장된 특성"""
        returns = data['returns']
        prices = data['Close']
        volume = data['Volume']
        features = pd.DataFrame(index=data.index)

        # V2 기본 특성들
        v2_data = self.create_v2_features(data)
        base_features = v2_data[0].copy()

        for col in base_features.columns:
            if col != 'target_vol_5d':
                features[col] = base_features[col]

        # 추가 변동성
        for window in [7, 12, 25, 40, 50, 60, 100]:
            features[f'vol_{window}'] = returns.rolling(window).std()

        # 고차 모멘트 확장
        for window in [15, 25, 30]:
            features[f'skew_{window}'] = returns.rolling(window).skew()
            features[f'kurt_{window}'] = returns.rolling(window).kurt()

        # 가격 특성
        for window in [5, 10, 20, 30, 50]:
            sma = prices.rolling(window).mean()
            features[f'price_sma_dev_{window}'] = (prices - sma) / sma

        # 거래량 특성 (간소화)
        for window in [10, 20]:
            vol_sma = volume.rolling(window).mean()
            features[f'volume_ratio_{window}'] = volume / (vol_sma + 1)

        # 추가 래그
        for lag in [5, 7, 10]:
            features[f'return_lag_{lag}'] = returns.shift(lag)

        # 더 많은 비율
        features['vol_7_30_ratio'] = features['vol_7'] / (features['vol_30'] + 1e-8)
        features['vol_12_50_ratio'] = features['vol_12'] / (features['vol_50'] + 1e-8)

        return self.finalize_features(features, returns)

    def create_v5_features(self, data):
        """V5 특성 (115개 → 50개로 간소화)"""
        returns = data['returns']
        features = pd.DataFrame(index=data.index)

        # V4 기반으로 시작하되 간소화
        v4_data = self.create_v4_features(data)
        v4_features = v4_data[0].copy()

        # 핵심 특성만 선별 (50개로 제한)
        important_features = []
        for col in v4_features.columns:
            if col != 'target_vol_5d':
                important_features.append(col)

        # 상위 50개만 선택 (임의로)
        selected_features = important_features[:50]

        for col in selected_features:
            features[col] = v4_features[col]

        return self.finalize_features(features, returns)

    def finalize_features(self, features, returns):
        """특성 마무리 처리"""
        # 타겟 추가
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

    def test_model_overfitting(self, model_name, config):
        """개별 모델 과적합 테스트"""
        logger.info(f"🔍 {model_name} 과적합 검증 시작...")
        logger.info(f"   설정: alpha={config['alpha']}, R²={config['reported_r2']:.4f}")

        # 데이터 준비
        data = self.load_spy_data()

        if config['features'] == 'v1_features':
            X, y = self.create_v1_features(data)
        elif config['features'] == 'v2_features':
            X, y = self.create_v2_features(data)
        elif config['features'] == 'v3_features':
            X, y = self.create_v3_features(data)
        elif config['features'] == 'v4_features':
            X, y = self.create_v4_features(data)
        elif config['features'] == 'v5_features':
            X, y = self.create_v5_features(data)

        X_scaled = self.scaler.fit_transform(X)

        logger.info(f"   특성: {X.shape[1]}개, 샘플: {len(X)}개")

        # 모델 생성
        if model_name == 'V5':  # 앙상블
            models = [
                ('ridge1', Ridge(alpha=config['alpha'][0], random_state=42)),
                ('ridge2', Ridge(alpha=config['alpha'][1], random_state=43)),
                ('ridge3', Ridge(alpha=config['alpha'][2], random_state=44))
            ]
            model = VotingRegressor(estimators=models)
        else:  # 단일 Ridge
            model = Ridge(alpha=config['alpha'])

        # 과적합 테스트: 훈련 vs 검증
        splits = list(self.cv.split(X_scaled))
        train_scores = []
        val_scores = []

        for train_idx, val_idx in splits:
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model.fit(X_train, y_train)

            # 훈련 성능
            train_pred = model.predict(X_train)
            train_r2 = r2_score(y_train, train_pred)
            train_scores.append(train_r2)

            # 검증 성능
            val_pred = model.predict(X_val)
            val_r2 = r2_score(y_val, val_pred)
            val_scores.append(val_r2)

        # 결과 계산
        train_mean = np.mean(train_scores)
        val_mean = np.mean(val_scores)
        performance_gap = train_mean - val_mean

        # 과적합 판정 기준
        overfitting_risk = 'LOW'
        if performance_gap > 0.15:
            overfitting_risk = 'HIGH'
        elif performance_gap > 0.08:
            overfitting_risk = 'MEDIUM'

        result = {
            'model': model_name,
            'config': config,
            'features_count': X.shape[1],
            'samples_count': len(X),
            'train_r2_mean': float(train_mean),
            'train_r2_std': float(np.std(train_scores)),
            'val_r2_mean': float(val_mean),
            'val_r2_std': float(np.std(val_scores)),
            'performance_gap': float(performance_gap),
            'overfitting_risk': overfitting_risk,
            'reported_vs_actual': {
                'reported_r2': config['reported_r2'],
                'actual_r2': float(val_mean),
                'difference': float(val_mean - config['reported_r2'])
            }
        }

        logger.info(f"   훈련 R²: {train_mean:.4f} ± {np.std(train_scores):.4f}")
        logger.info(f"   검증 R²: {val_mean:.4f} ± {np.std(val_scores):.4f}")
        logger.info(f"   성능 격차: {performance_gap:.4f}")
        logger.info(f"   보고값 vs 실제: {config['reported_r2']:.4f} → {val_mean:.4f}")
        logger.info(f"   과적합 위험: {overfitting_risk}")

        return result

    def check_all_models(self):
        """모든 모델 과적합 검증"""
        logger.info("🚀 전체 모델 (V1-V5) 과적합 종합 검증 시작")

        all_results = {}
        safe_models = []
        risky_models = []

        for model_name, config in self.model_configs.items():
            try:
                result = self.test_model_overfitting(model_name, config)
                all_results[model_name] = result

                if result['overfitting_risk'] == 'LOW':
                    safe_models.append(model_name)
                else:
                    risky_models.append(model_name)

            except Exception as e:
                logger.error(f"❌ {model_name} 검증 실패: {str(e)}")
                all_results[model_name] = {'error': str(e)}

        # 종합 결과
        summary = {
            'analysis_date': datetime.now().isoformat(),
            'total_models_tested': len(self.model_configs),
            'safe_models': safe_models,
            'risky_models': risky_models,
            'detailed_results': all_results,
            'recommendations': self.generate_recommendations(safe_models, risky_models, all_results)
        }

        # 결과 저장
        save_path = '/root/workspace/data/raw/all_models_overfitting_analysis.json'
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info("="*70)
        logger.info("📊 전체 모델 과적합 검증 완료")
        logger.info(f"✅ 안전한 모델: {len(safe_models)}개 - {safe_models}")
        logger.info(f"⚠️ 위험한 모델: {len(risky_models)}개 - {risky_models}")
        logger.info(f"💾 상세 결과: {save_path}")
        logger.info("="*70)

        return summary

    def generate_recommendations(self, safe_models, risky_models, all_results):
        """권장사항 생성"""
        recommendations = []

        if len(safe_models) > 0:
            best_safe_model = None
            best_safe_r2 = -1

            for model in safe_models:
                if model in all_results and 'val_r2_mean' in all_results[model]:
                    r2 = all_results[model]['val_r2_mean']
                    if r2 > best_safe_r2:
                        best_safe_r2 = r2
                        best_safe_model = model

            recommendations.extend([
                f"✅ 프로덕션 권장: {best_safe_model} (R²={best_safe_r2:.4f})",
                "✅ 과적합 위험 낮은 모델들 우선 사용",
                "✅ 정기적인 성능 모니터링 체계 구축"
            ])
        else:
            recommendations.extend([
                "❌ 안전한 모델 없음 - 모든 모델이 과적합 위험",
                "❌ 새로운 모델 개발 필요",
                "❌ 더 강한 정규화 및 특성 선택 필요"
            ])

        if len(risky_models) > 0:
            recommendations.append(f"⚠️ 위험 모델 사용 금지: {risky_models}")

        return recommendations

def main():
    """메인 실행"""
    logger.info("🎯 전체 모델 과적합 검증 시작")

    checker = AllModelsOverfittingChecker()

    try:
        results = checker.check_all_models()

        logger.info("💡 최종 권장사항:")
        for rec in results['recommendations']:
            logger.info(f"   {rec}")

        return results

    except Exception as e:
        logger.error(f"❌ 전체 모델 검증 실패: {str(e)}")
        raise

if __name__ == "__main__":
    main()