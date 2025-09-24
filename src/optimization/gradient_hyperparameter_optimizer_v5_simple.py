"""
경사하강법 하이퍼파라미터 최적화 시스템 v5 (Simple)
개선사항: Random Search + VotingRegressor 앙상블

V5 Simple 접근:
- scikit-optimize 없이 구현
- Random Search로 다중 alpha 탐색
- VotingRegressor 앙상블 사용
- 더 많은 특성과 더 안정적인 구현
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
warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/root/workspace/data/raw/gradient_optimization_v5_simple.log'),
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

class RandomSearchEnsembleOptimizerV5:
    """Random Search 기반 앙상블 하이퍼파라미터 최적화기 v5"""

    def __init__(self, n_trials=100):
        self.n_trials = n_trials
        self.cv = PurgedKFoldSklearn()
        self.scaler = StandardScaler()
        self.history = []
        self.trial_count = 0

    def load_spy_data(self):
        """SPY 데이터 로드 및 전처리"""
        logger.info("📊 SPY 데이터 로딩 중...")

        spy = yf.Ticker("SPY")
        data = spy.history(start="2015-01-01", end="2024-12-31")
        data['returns'] = np.log(data['Close'] / data['Close'].shift(1))
        data = data.dropna()

        logger.info(f"✅ 데이터 로딩 완료: {len(data)}개 관측치")
        return data

    def create_ultra_features(self, data):
        """초고급 특성 엔지니어링 (50개+ 특성)"""
        logger.info("🔧 초고급 특성 엔지니어링 (50개+ 특성)...")

        returns = data['returns']
        prices = data['Close']
        volume = data['Volume']
        features = pd.DataFrame(index=data.index)

        # 1. 기본 변동성 - 다양한 윈도우
        volatility_windows = [3, 5, 7, 10, 12, 15, 20, 25, 30, 40, 50, 60, 100]
        for window in volatility_windows:
            features[f'vol_{window}'] = returns.rolling(window).std()

        # 2. 고차 모멘트 - 분포의 형태
        moment_windows = [5, 10, 15, 20, 25, 30]
        for window in moment_windows:
            features[f'skew_{window}'] = returns.rolling(window).skew()
            features[f'kurt_{window}'] = returns.rolling(window).kurt()

        # 3. 가격 관련 특성
        price_windows = [5, 10, 20, 30, 50]
        for window in price_windows:
            sma = prices.rolling(window).mean()
            features[f'price_sma_dev_{window}'] = (prices - sma) / sma
            features[f'price_mom_{window}'] = (prices / prices.shift(window)) - 1

        # 4. 거래량 관련 특성
        for window in [10, 20, 30]:
            vol_sma = volume.rolling(window).mean()
            features[f'volume_ratio_{window}'] = volume / (vol_sma + 1)
            features[f'price_volume_{window}'] = returns * (volume / vol_sma)

        # 5. 다층 래그 특성
        lags = [1, 2, 3, 5, 7, 10, 15]
        for lag in lags:
            features[f'return_lag_{lag}'] = returns.shift(lag)
            features[f'vol_lag_{lag}'] = features['vol_5'].shift(lag)

        # 6. 변동성 체제 분석
        short_vol = features['vol_5']
        medium_vol = features['vol_20']
        long_vol = features['vol_50']

        features['vol_regime_short'] = (short_vol > medium_vol).astype(float)
        features['vol_regime_medium'] = (medium_vol > long_vol).astype(float)
        features['vol_expansion_factor'] = short_vol / (long_vol + 1e-8)
        features['vol_contraction_factor'] = long_vol / (short_vol + 1e-8)

        # 7. 통계적 지표
        stat_windows = [10, 15, 20, 30, 50]
        for window in stat_windows:
            ma = returns.rolling(window).mean()
            std = returns.rolling(window).std()
            features[f'zscore_{window}'] = (returns - ma) / (std + 1e-8)
            features[f'sharpe_{window}'] = (ma * np.sqrt(252)) / (std + 1e-8)
            features[f'sortino_{window}'] = (ma * np.sqrt(252)) / (returns[returns < 0].rolling(window).std() + 1e-8)

        # 8. 모멘텀 지표
        momentum_windows = [3, 5, 7, 10, 15, 20, 30]
        for window in momentum_windows:
            features[f'momentum_{window}'] = returns.rolling(window).sum()
            features[f'roc_{window}'] = (prices / prices.shift(window) - 1)

        # 9. 변동성 상호작용
        features['vol_5_20_ratio'] = features['vol_5'] / (features['vol_20'] + 1e-8)
        features['vol_10_50_ratio'] = features['vol_10'] / (features['vol_50'] + 1e-8)
        features['vol_short_long_ratio'] = features['vol_7'] / (features['vol_30'] + 1e-8)
        features['vol_momentum_cross'] = features['vol_5'] * features['momentum_10']

        # 10. 극값 및 리스크 지표
        risk_windows = [5, 10, 15, 20, 30]
        for window in risk_windows:
            # 최대 손실폭
            cumret = returns.rolling(window).sum()
            features[f'max_drawdown_{window}'] = (cumret - cumret.rolling(window).max()).min()

            # 분위수
            features[f'q05_{window}'] = returns.rolling(window).quantile(0.05)
            features[f'q95_{window}'] = returns.rolling(window).quantile(0.95)
            features[f'iqr_{window}'] = features[f'q95_{window}'] - features[f'q05_{window}']

        # 11. 고급 상호작용 특성
        features['vol_price_interaction'] = features['vol_20'] * features['price_mom_20']
        features['momentum_vol_interaction'] = features['momentum_5'] * features['vol_10']
        features['regime_vol_interaction'] = features['vol_regime_short'] * features['vol_20']

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

    def generate_alpha_combinations(self):
        """다양한 alpha 조합 생성"""
        np.random.seed(42)
        combinations = []

        for trial in range(self.n_trials):
            if trial < 20:
                # V2 성공값 주변 탐색
                alphas = [
                    np.random.uniform(10, 30),   # V2 최적: 19.5 주변
                    np.random.uniform(1, 10),    # 중간 정규화
                    np.random.uniform(30, 100),  # 강한 정규화
                ]
            else:
                # 랜덤 탐색
                alphas = [
                    np.random.uniform(0.1, 50),
                    np.random.uniform(0.1, 20),
                    np.random.uniform(5, 200),
                ]

            combinations.append(alphas)

        return combinations

    def evaluate_ensemble(self, alphas):
        """3-Ridge 앙상블 평가"""
        self.trial_count += 1

        # VotingRegressor 앙상블 생성
        models = [
            ('ridge1', Ridge(alpha=alphas[0], random_state=42)),
            ('ridge2', Ridge(alpha=alphas[1], random_state=43)),
            ('ridge3', Ridge(alpha=alphas[2], random_state=44))
        ]
        ensemble = VotingRegressor(estimators=models)

        # Purged K-Fold Cross-Validation
        cv_scores = []
        splits = self.cv.split(self.X_scaled)

        for train_idx, test_idx in splits:
            if len(train_idx) < 20 or len(test_idx) < 10:
                continue

            X_train, X_test = self.X_scaled[train_idx], self.X_scaled[test_idx]
            y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]

            ensemble.fit(X_train, y_train)
            predictions = ensemble.predict(X_test)
            r2 = r2_score(y_test, predictions)
            cv_scores.append(r2)

        mean_r2 = np.mean(cv_scores) if cv_scores else -1.0

        # 히스토리 저장
        self.history.append({
            'trial': self.trial_count,
            'alphas': alphas,
            'r2_score': mean_r2,
            'timestamp': datetime.now().isoformat()
        })

        return mean_r2

    def optimize(self):
        """Random Search 최적화 실행"""
        logger.info("🚀 Random Search 앙상블 하이퍼파라미터 최적화 v5 시작")
        logger.info("🎯 v5 목표: R² > 0.33 (3-Ridge 앙상블)")

        # 데이터 준비
        data = self.load_spy_data()
        X, y = self.create_ultra_features(data)
        self.X_scaled = self.scaler.fit_transform(X)
        self.y = y

        logger.info(f"📈 Random Search 시작: {self.n_trials}회 시도")

        # Alpha 조합 생성
        alpha_combinations = self.generate_alpha_combinations()

        # 최적화 실행
        best_score = -1.0
        best_alphas = None

        for i, alphas in enumerate(alpha_combinations):
            start_time = time.time()
            score = self.evaluate_ensemble(alphas)
            elapsed = time.time() - start_time

            if score > best_score:
                best_score = score
                best_alphas = alphas

            # 로깅
            if (i + 1) % 5 == 0:
                logger.info(
                    f"시도 {i+1:3d}/{self.n_trials}: α=[{alphas[0]:.2f},{alphas[1]:.2f},{alphas[2]:.2f}] "
                    f"R²={score:.6f} (최적={best_score:.6f}) 시간={elapsed:.1f}s"
                )

            # 목표 달성 체크
            if score > 0.33:
                logger.info(f"🚀 V5 목표 달성! R² = {score:.6f} > 0.33")

        logger.info(f"✅ v5 최적화 완료!")
        logger.info(f"📊 최적 alphas: [{best_alphas[0]:.4f}, {best_alphas[1]:.4f}, {best_alphas[2]:.4f}]")
        logger.info(f"📊 최적 R²: {best_score:.6f}")

        return best_alphas, best_score, self.history

    def save_results(self, best_alphas, best_score, history):
        """결과 저장"""
        results = {
            'version': 'v5_simple',
            'approach': 'Random_Search_3_Ridge_Ensemble',
            'improvements': [
                '3개 Ridge 모델 VotingRegressor',
                'Random Search 최적화',
                '50개+ 초고급 특성',
                '다중 alpha Random 탐색',
                'scikit-optimize 의존성 제거'
            ],
            'optimization_completed': datetime.now().isoformat(),
            'best_hyperparameters': {
                'alpha1': float(best_alphas[0]),
                'alpha2': float(best_alphas[1]),
                'alpha3': float(best_alphas[2]),
                'model_type': 'VotingRegressor_3_Ridge'
            },
            'best_performance': {
                'r2_score': float(best_score),
                'method': 'Purged_K_Fold_CV_Ensemble'
            },
            'version_comparison': {
                'v1_r2': 0.2775,
                'v2_r2': 0.3256,
                'v3_r2': 0.2750,
                'v5_simple_r2': float(best_score),
                'improvement_from_v2': float(best_score - 0.3256),
                'improvement_percent': float(((best_score - 0.3256) / 0.3256) * 100)
            },
            'optimization_details': {
                'n_trials': self.n_trials,
                'search_method': 'Random_Search'
            },
            'optimization_history': history
        }

        # 결과 저장
        save_path = '/root/workspace/data/raw/gradient_optimization_results_v5_simple.json'
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"💾 v5 결과 저장됨: {save_path}")
        return results

def main():
    """메인 실행 함수"""
    logger.info("🎯 Random Search 앙상블 하이퍼파라미터 최적화 v5 시작")

    optimizer = RandomSearchEnsembleOptimizerV5(n_trials=60)

    try:
        best_alphas, best_score, history = optimizer.optimize()
        results = optimizer.save_results(best_alphas, best_score, history)

        # 성능 비교
        v2_r2 = 0.3256
        improvement = best_score - v2_r2
        improvement_pct = (improvement / v2_r2) * 100

        logger.info("📈 v5 성능 비교:")
        logger.info(f"   V2 최고점: R² = {v2_r2:.4f}")
        logger.info(f"   V5 앙상블: R² = {best_score:.4f}")
        logger.info(f"   V2 대비: {improvement:+.4f} ({improvement_pct:+.2f}%)")

        if best_score > 0.33:
            logger.info("🎉 v5 대성공: R² > 0.33 달성!")
        elif best_score > v2_r2:
            logger.info("🎉 v5 성공: V2 성능 초과!")
        else:
            logger.info("⚠️ v5 아쉬움: 추가 최적화 고려")

        return results

    except Exception as e:
        logger.error(f"❌ v5 최적화 실패: {str(e)}")
        raise

if __name__ == "__main__":
    main()