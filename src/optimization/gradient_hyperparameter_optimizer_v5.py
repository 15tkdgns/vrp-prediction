"""
경사하강법 하이퍼파라미터 최적화 시스템 v5
개선사항: 앙상블 접근법 + 다중 alpha 동시 최적화

V2 성공: R² = 0.3256 (alpha = 19.5)
목표: R² > 0.35 달성

V5 혁신적 접근:
- 다중 Ridge 모델 앙상블
- 각각 다른 alpha 값으로 최적화
- 가중 평균으로 최종 예측
- Bayesian Optimization 도입
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json
import time
from datetime import datetime
import logging
import warnings
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.ensemble import VotingRegressor
import skopt
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args
warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/root/workspace/data/raw/gradient_optimization_v5.log'),
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

    def split(self, X, y=None):
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

class BayesianHyperparameterOptimizerV5:
    """Bayesian 최적화 기반 앙상블 하이퍼파라미터 최적화기 v5"""

    def __init__(self, n_calls=100):
        self.n_calls = n_calls
        self.cv = PurgedKFoldSklearn()
        self.scaler = StandardScaler()
        self.history = []
        self.call_count = 0

    def load_spy_data(self):
        """SPY 데이터 로드 및 전처리"""
        logger.info("📊 SPY 데이터 로딩 중...")

        spy = yf.Ticker("SPY")
        data = spy.history(start="2015-01-01", end="2024-12-31")
        data['returns'] = np.log(data['Close'] / data['Close'].shift(1))
        data = data.dropna()

        logger.info(f"✅ 데이터 로딩 완료: {len(data)}개 관측치")
        return data

    def create_premium_features(self, data):
        """프리미엄 특성 엔지니어링 (40개+ 특성)"""
        logger.info("🔧 프리미엄 특성 엔지니어링 (40개+ 특성)...")

        returns = data['returns']
        prices = data['Close']
        features = pd.DataFrame(index=data.index)

        # 1. 다층 변동성 특성
        for window in [3, 5, 7, 10, 15, 20, 30, 50, 100]:
            features[f'vol_{window}'] = returns.rolling(window).std()

        # 2. 고차 모멘트 (풍부한 분포 정보)
        for window in [5, 10, 15, 20, 30]:
            features[f'skew_{window}'] = returns.rolling(window).skew()
            features[f'kurt_{window}'] = returns.rolling(window).kurt()

        # 3. 가격 기반 특성
        for window in [10, 20, 50]:
            sma = prices.rolling(window).mean()
            features[f'price_dev_{window}'] = (prices - sma) / sma
            features[f'price_momentum_{window}'] = (prices / prices.shift(window)) - 1

        # 4. 다층 래그 특성
        for lag in [1, 2, 3, 5, 7, 10]:
            features[f'return_lag_{lag}'] = returns.shift(lag)
            if f'vol_5' in features:
                features[f'vol_lag_{lag}'] = features['vol_5'].shift(lag)

        # 5. 변동성 체제 및 트렌드
        short_vol = features['vol_5']
        medium_vol = features['vol_20']
        long_vol = features['vol_50']

        features['vol_regime_sm'] = (short_vol > medium_vol).astype(float)
        features['vol_regime_ml'] = (medium_vol > long_vol).astype(float)
        features['vol_expansion'] = (short_vol / long_vol)
        features['vol_contraction'] = (long_vol / short_vol)

        # 6. 통계적 지표 확장
        for window in [10, 20, 30, 50]:
            ma = returns.rolling(window).mean()
            std = returns.rolling(window).std()
            features[f'zscore_{window}'] = (returns - ma) / (std + 1e-8)
            features[f'sharpe_{window}'] = (ma * np.sqrt(252)) / (std + 1e-8)
            features[f'cvar_{window}'] = returns.rolling(window).quantile(0.05)

        # 7. 모멘텀 지표
        for window in [3, 5, 10, 15, 20, 30]:
            features[f'momentum_{window}'] = returns.rolling(window).sum()
            features[f'roc_{window}'] = (prices / prices.shift(window) - 1) * 100

        # 8. 변동성 상호작용
        features['vol_5_20_ratio'] = features['vol_5'] / (features['vol_20'] + 1e-8)
        features['vol_10_50_ratio'] = features['vol_10'] / (features['vol_50'] + 1e-8)
        features['vol_momentum_interaction'] = features['vol_5'] * features['momentum_10']
        features['price_vol_interaction'] = features['price_dev_20'] * features['vol_20']

        # 9. 극값 및 드로우다운
        for window in [5, 10, 15, 20]:
            cumret = returns.rolling(window).sum()
            features[f'max_drawdown_{window}'] = (cumret - cumret.rolling(window).max()).min()
            features[f'max_upward_{window}'] = (cumret - cumret.rolling(window).min()).max()

        # 10. 고급 통계적 특성
        for window in [15, 30]:
            features[f'q10_{window}'] = returns.rolling(window).quantile(0.1)
            features[f'q90_{window}'] = returns.rolling(window).quantile(0.9)
            features[f'iqr_{window}'] = (features[f'q90_{window}'] - features[f'q10_{window}'])

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

        logger.info(f"✅ 특성 생성 완료: {X.shape[1]}개 특성, {len(X)}개 샘플")
        return X, y

    def create_ensemble_model(self, alpha1, alpha2, alpha3, alpha4, alpha5):
        """5개 Ridge 모델 앙상블 생성"""
        models = [
            ('ridge1', Ridge(alpha=alpha1, random_state=42)),
            ('ridge2', Ridge(alpha=alpha2, random_state=43)),
            ('ridge3', Ridge(alpha=alpha3, random_state=44)),
            ('ridge4', Ridge(alpha=alpha4, random_state=45)),
            ('ridge5', Ridge(alpha=alpha5, random_state=46))
        ]

        return VotingRegressor(estimators=models)

    def objective_function(self, alpha1, alpha2, alpha3, alpha4, alpha5):
        """Bayesian 최적화용 목적함수"""
        self.call_count += 1

        # 앙상블 모델 생성
        model = self.create_ensemble_model(alpha1, alpha2, alpha3, alpha4, alpha5)

        # Purged K-Fold Cross-Validation
        cv_scores = []
        splits = self.cv.split(self.X_scaled)

        for train_idx, test_idx in splits:
            if len(train_idx) < 20 or len(test_idx) < 10:
                continue

            X_train, X_test = self.X_scaled[train_idx], self.X_scaled[test_idx]
            y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]

            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            r2 = r2_score(y_test, predictions)
            cv_scores.append(r2)

        mean_r2 = np.mean(cv_scores) if cv_scores else -1.0

        # 히스토리 저장
        self.history.append({
            'call': self.call_count,
            'alphas': [alpha1, alpha2, alpha3, alpha4, alpha5],
            'r2_score': mean_r2,
            'timestamp': datetime.now().isoformat()
        })

        # 로깅
        if self.call_count % 5 == 0:
            logger.info(
                f"호출 {self.call_count:3d}: α=[{alpha1:.2f},{alpha2:.2f},{alpha3:.2f},{alpha4:.2f},{alpha5:.2f}] "
                f"R²={mean_r2:.6f}"
            )

        if mean_r2 > 0.35:
            logger.info(f"🚀 V5 목표 달성! R² = {mean_r2:.6f} > 0.35")

        return -mean_r2  # 최소화 문제

    def optimize(self):
        """Bayesian 최적화 실행"""
        logger.info("🚀 Bayesian 최적화 기반 앙상블 하이퍼파라미터 최적화 v5 시작")
        logger.info("🎯 v5 목표: R² > 0.35 (5-Ridge 앙상블)")

        # 데이터 준비
        data = self.load_spy_data()
        X, y = self.create_premium_features(data)
        self.X_scaled = self.scaler.fit_transform(X)
        self.y = y

        logger.info(f"📈 Bayesian 최적화 시작: {self.n_calls}회 호출")

        # 탐색 공간 정의 (V2 성공 기준 확장)
        dimensions = [
            Real(1.0, 50.0, name='alpha1'),    # V2 최적: 19.5 주변
            Real(0.1, 20.0, name='alpha2'),    # 다양한 정규화 강도
            Real(5.0, 100.0, name='alpha3'),   # 강한 정규화
            Real(0.01, 5.0, name='alpha4'),    # 약한 정규화
            Real(10.0, 200.0, name='alpha5')   # 매우 강한 정규화
        ]

        # Bayesian 최적화 실행
        result = gp_minimize(
            func=self.objective_function,
            dimensions=dimensions,
            n_calls=self.n_calls,
            n_initial_points=20,  # 초기 랜덤 탐색
            random_state=42,
            acq_func='EI'  # Expected Improvement
        )

        # 결과 처리
        best_alphas = result.x
        best_score = -result.fun

        logger.info(f"✅ v5 최적화 완료!")
        logger.info(f"📊 최적 alphas: {[f'{a:.4f}' for a in best_alphas]}")
        logger.info(f"📊 최적 R²: {best_score:.6f}")

        return best_alphas, best_score, self.history, result

    def save_results(self, best_alphas, best_score, history, bayesian_result):
        """결과 저장"""
        results = {
            'version': 'v5',
            'approach': 'Bayesian_Optimization_Ensemble',
            'improvements': [
                '5개 Ridge 모델 앙상블',
                'Bayesian Optimization 적용',
                '40개+ 프리미엄 특성',
                '다중 alpha 동시 최적화',
                'Expected Improvement 획득함수'
            ],
            'optimization_completed': datetime.now().isoformat(),
            'best_hyperparameters': {
                'alpha1': float(best_alphas[0]),
                'alpha2': float(best_alphas[1]),
                'alpha3': float(best_alphas[2]),
                'alpha4': float(best_alphas[3]),
                'alpha5': float(best_alphas[4]),
                'model_type': 'VotingRegressor_5_Ridge'
            },
            'best_performance': {
                'r2_score': float(best_score),
                'method': 'Purged_K_Fold_CV_Ensemble'
            },
            'version_comparison': {
                'v1_r2': 0.2775,
                'v2_r2': 0.3256,
                'v3_r2': 0.2750,
                'v5_r2': float(best_score),
                'improvement_from_v2': float(best_score - 0.3256),
                'improvement_percent': float(((best_score - 0.3256) / 0.3256) * 100)
            },
            'bayesian_optimization': {
                'n_calls': self.n_calls,
                'acquisition_function': 'Expected_Improvement',
                'n_initial_points': 20
            },
            'optimization_history': history
        }

        # 결과 저장
        save_path = '/root/workspace/data/raw/gradient_optimization_results_v5.json'
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"💾 v5 결과 저장됨: {save_path}")
        return results

def main():
    """메인 실행 함수"""
    logger.info("🎯 Bayesian 최적화 앙상블 하이퍼파라미터 최적화 v5 시작")

    optimizer = BayesianHyperparameterOptimizerV5(n_calls=50)  # 50회 호출

    try:
        best_alphas, best_score, history, bayesian_result = optimizer.optimize()
        results = optimizer.save_results(best_alphas, best_score, history, bayesian_result)

        # 성능 비교
        v2_r2 = 0.3256
        improvement = best_score - v2_r2
        improvement_pct = (improvement / v2_r2) * 100

        logger.info("📈 v5 성능 비교:")
        logger.info(f"   V2 최고점: R² = {v2_r2:.4f}")
        logger.info(f"   V5 앙상블: R² = {best_score:.4f}")
        logger.info(f"   V2 대비: {improvement:+.4f} ({improvement_pct:+.2f}%)")

        if best_score > 0.35:
            logger.info("🎉 v5 대성공: R² > 0.35 달성!")
        elif best_score > v2_r2:
            logger.info("🎉 v5 성공: V2 성능 초과!")
        else:
            logger.info("⚠️ v5 아쉬움: 추가 최적화 검토")

        return results

    except Exception as e:
        logger.error(f"❌ v5 최적화 실패: {str(e)}")
        raise

if __name__ == "__main__":
    main()