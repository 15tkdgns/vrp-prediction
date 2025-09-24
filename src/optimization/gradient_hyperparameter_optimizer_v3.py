"""
경사하강법 하이퍼파라미터 최적화 시스템 v3
개선사항: sklearn 호환 목적함수 사용

v1-v2 문제점:
- PyTorch 기반 Ridge와 sklearn Ridge 구현 차이
- 목적함수의 일관성 부족

v3 개선사항:
- sklearn Ridge Regression을 직접 사용
- 동일한 검증 조건 보장
- 더 안정적인 수치 계산
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
from sklearn.model_selection import cross_val_score
import scipy.optimize as opt
warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/root/workspace/data/raw/gradient_optimization_v3.log'),
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
        """시계열 데이터를 purged 방식으로 분할"""
        n_samples = len(X)
        indices = np.arange(n_samples)

        test_size = n_samples // self.n_splits
        splits = []

        for i in range(self.n_splits):
            # 테스트 세트
            test_start = i * test_size
            test_end = min((i + 1) * test_size, n_samples)
            test_indices = indices[test_start:test_end]

            # Purge: 테스트 세트 직후 데이터 제거
            purge_start = test_end
            purge_end = min(test_end + self.purge_length, n_samples)

            # Embargo: 추가 간격
            embargo_end = min(purge_end + self.embargo_length, n_samples)

            # 훈련 세트 (테스트 전 + embargo 후)
            train_indices = np.concatenate([
                indices[:test_start],
                indices[embargo_end:]
            ])

            if len(train_indices) > 0 and len(test_indices) > 0:
                splits.append((train_indices, test_indices))

        return splits

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

class GradientHyperparameterOptimizerV3:
    """sklearn 기반 하이퍼파라미터 최적화기 v3"""

    def __init__(self):
        self.cv = PurgedKFoldSklearn()
        self.scaler = StandardScaler()
        self.history = []

    def load_spy_data(self):
        """SPY 데이터 로드 및 전처리"""
        logger.info("📊 SPY 데이터 로딩 중...")

        spy = yf.Ticker("SPY")
        data = spy.history(start="2015-01-01", end="2024-12-31")
        data['returns'] = np.log(data['Close'] / data['Close'].shift(1))
        data = data.dropna()

        logger.info(f"✅ 데이터 로딩 완료: {len(data)}개 관측치")
        return data

    def create_features(self, data):
        """완전한 특성 엔지니어링 (기존과 동일한 31개 특성)"""
        logger.info("🔧 완전한 특성 엔지니어링 (31개 특성)...")

        returns = data['returns']
        features = pd.DataFrame(index=data.index)

        # 1. 과거 변동성 특성 (≤ t)
        for window in [5, 10, 20, 50]:
            features[f'volatility_{window}'] = returns.rolling(window).std()

        # 2. 고차 모멘트 (≤ t)
        for window in [5, 10, 20]:
            features[f'skew_{window}'] = returns.rolling(window).skew()
            features[f'kurt_{window}'] = returns.rolling(window).kurt()

        # 3. 래그 특성 (과거만)
        for lag in [1, 2, 3, 5]:
            features[f'return_lag_{lag}'] = returns.shift(lag)
            features[f'vol_lag_{lag}'] = features['volatility_5'].shift(lag)

        # 4. 교차 통계 (과거만)
        features['vol_ratio_5_20'] = features['volatility_5'] / (features['volatility_20'] + 1e-8)
        features['vol_ratio_10_50'] = features['volatility_10'] / (features['volatility_50'] + 1e-8)

        # 5. Z-score (과거만)
        ma_20 = returns.rolling(20).mean()
        std_20 = returns.rolling(20).std()
        features['zscore_20'] = (returns - ma_20) / (std_20 + 1e-8)

        # 6. 모멘텀 (과거만)
        for window in [5, 10, 20]:
            features[f'momentum_{window}'] = returns.rolling(window).sum()

        # 7. 분위수 특성 (과거만)
        for window in [10, 20]:
            features[f'quantile_25_{window}'] = returns.rolling(window).quantile(0.25)
            features[f'quantile_75_{window}'] = returns.rolling(window).quantile(0.75)

        # 8. 극값 특성 (과거만)
        features['max_drawdown_5'] = returns.rolling(5).apply(
            lambda x: (x.cumsum() - x.cumsum().cummax()).min()
        )

        # 타겟: 5일 후 변동성 (≥ t+1)
        target = []
        for i in range(len(returns)):
            if i + 5 < len(returns):
                future_vol = returns.iloc[i+1:i+6].std()
                target.append(future_vol)
            else:
                target.append(np.nan)

        features['target_vol_5d'] = target

        # NaN 제거
        features = features.dropna()

        X = features.drop('target_vol_5d', axis=1)
        y = features['target_vol_5d']

        logger.info(f"✅ 특성 생성 완료: {X.shape[1]}개 특성, {len(X)}개 샘플")
        return X, y

    def objective_function(self, log_alpha):
        """sklearn 기반 목적함수"""
        alpha = np.exp(log_alpha)

        # Ridge 모델 생성 (sklearn 사용)
        model = Ridge(alpha=alpha, random_state=42)

        # Purged K-Fold Cross-Validation
        cv_scores = []
        splits = self.cv.split(self.X_scaled)

        for train_idx, test_idx in splits:
            if len(train_idx) < 10 or len(test_idx) < 5:
                continue

            X_train, X_test = self.X_scaled[train_idx], self.X_scaled[test_idx]
            y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]

            # 모델 훈련 및 예측
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            # R² 계산
            r2 = r2_score(y_test, predictions)
            cv_scores.append(r2)

        mean_r2 = np.mean(cv_scores) if cv_scores else -1.0

        # 히스토리 저장
        self.current_iteration += 1
        self.history.append({
            'iteration': self.current_iteration,
            'alpha': alpha,
            'log_alpha': log_alpha,
            'r2_score': mean_r2,
            'timestamp': datetime.now().isoformat()
        })

        # 로깅
        if self.current_iteration % 5 == 0:
            logger.info(
                f"반복 {self.current_iteration:4d}: α={alpha:.6f}, "
                f"R²={mean_r2:.6f}, 목표=0.3113"
            )

        return -mean_r2  # 최소화 문제로 변환

    def optimize(self):
        """scipy.optimize 기반 최적화"""
        logger.info("🚀 경사하강법 기반 하이퍼파라미터 최적화 v3 시작")
        logger.info("🔧 v3 개선사항: sklearn 호환 목적함수")

        # 데이터 준비
        data = self.load_spy_data()
        X, y = self.create_features(data)

        # 표준화
        self.X_scaled = self.scaler.fit_transform(X)
        self.y = y
        self.current_iteration = 0

        logger.info(f"📈 최적화 시작: sklearn Ridge 기반")
        logger.info(f"📊 목표 성능: R² > 0.3113 (기존 모델)")

        # scipy.optimize를 사용한 최적화
        # L-BFGS-B 알고리즘 사용 (경사하강법 계열)
        result = opt.minimize(
            self.objective_function,
            x0=np.array([0.0]),  # log(alpha) = 0, 즉 alpha = 1.0 시작
            method='L-BFGS-B',
            bounds=[(-4.0, 4.0)],  # alpha 범위: 0.018 ~ 54.6
            options={
                'maxiter': 100,
                'ftol': 1e-6,
                'gtol': 1e-6
            }
        )

        # 결과 처리
        best_log_alpha = result.x[0]
        best_alpha = np.exp(best_log_alpha)
        best_score = -result.fun  # 원래 R² 점수

        logger.info(f"✅ v3 최적화 완료!")
        logger.info(f"📊 최적 alpha: {best_alpha:.6f}")
        logger.info(f"📊 최적 R²: {best_score:.6f}")
        logger.info(f"📊 최적화 성공: {'예' if result.success else '아니오'}")
        logger.info(f"📊 함수 호출 횟수: {result.nfev}")

        return best_alpha, best_score, self.history, result

    def save_results(self, best_alpha, best_score, history, optimization_result):
        """결과 저장"""
        results = {
            'version': 'v3',
            'improvements': [
                'sklearn Ridge Regression 직접 사용',
                'scipy.optimize L-BFGS-B 알고리즘',
                '더 안정적인 수치 계산',
                '목적함수 일관성 보장'
            ],
            'optimization_completed': datetime.now().isoformat(),
            'best_hyperparameters': {
                'alpha': best_alpha,
                'log_alpha': float(np.log(best_alpha)),
                'model_type': 'Ridge_sklearn'
            },
            'best_performance': {
                'r2_score': best_score,
                'method': 'Purged_K_Fold_CV_sklearn'
            },
            'baseline_comparison': {
                'baseline_r2': 0.3113,
                'optimized_r2': best_score,
                'improvement': best_score - 0.3113,
                'improvement_percent': ((best_score - 0.3113) / 0.3113) * 100
            },
            'optimization_details': {
                'algorithm': 'L-BFGS-B',
                'success': optimization_result.success,
                'function_evaluations': int(optimization_result.nfev),
                'message': optimization_result.message
            },
            'optimization_history': history
        }

        # 결과 저장
        save_path = '/root/workspace/data/raw/gradient_optimization_results_v3.json'
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"💾 v3 결과 저장됨: {save_path}")
        return results

def main():
    """메인 실행 함수"""
    logger.info("🎯 경사하강법 하이퍼파라미터 최적화 v3 시작")

    # 최적화기 생성
    optimizer = GradientHyperparameterOptimizerV3()

    try:
        # 최적화 실행
        best_alpha, best_score, history, opt_result = optimizer.optimize()

        # 결과 저장
        results = optimizer.save_results(best_alpha, best_score, history, opt_result)

        # 성능 비교
        baseline_r2 = 0.3113
        improvement = best_score - baseline_r2
        improvement_pct = (improvement / baseline_r2) * 100

        logger.info("📈 v3 성능 비교:")
        logger.info(f"   기존 모델 (α=1.0): R² = {baseline_r2:.4f}")
        logger.info(f"   v3 모델 (α={best_alpha:.4f}): R² = {best_score:.4f}")
        logger.info(f"   성능 변화: {improvement:+.4f} ({improvement_pct:+.2f}%)")

        if best_score > baseline_r2:
            logger.info("🎉 v3 성공: 기존 성능 초과!")
        else:
            logger.info("⚠️ v3 미달: 추가 개선 필요 → v4 준비")

        return results

    except Exception as e:
        logger.error(f"❌ v3 최적화 실패: {str(e)}")
        raise

if __name__ == "__main__":
    main()