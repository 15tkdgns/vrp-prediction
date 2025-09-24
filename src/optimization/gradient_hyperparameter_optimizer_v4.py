"""
경사하강법 하이퍼파라미터 최적화 시스템 v4
개선사항: V2 기반으로 더 정교한 탐색 및 추가 특성

V2 성공 요인:
- 31개 특성 사용
- alpha = 19.5 발견
- R² = 0.3256 달성

V4 개선사항:
- V2 기반으로 더 많은 특성 추가 (35개+)
- 더 정교한 alpha 탐색 범위
- 앙상블 특성 추가
- 더 긴 patience로 글로벌 최적해 탐색
"""

import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import json
import time
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/root/workspace/data/raw/gradient_optimization_v4.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PurgedKFoldPyTorch:
    """PyTorch 기반 Purged K-Fold Cross-Validation"""

    def __init__(self, n_splits=5, purge_length=5, embargo_length=5):
        self.n_splits = n_splits
        self.purge_length = purge_length
        self.embargo_length = embargo_length

    def split(self, X):
        """시계열 데이터를 purged 방식으로 분할"""
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

class DifferentiableRidge(nn.Module):
    """미분가능한 Ridge Regression 구현"""

    def __init__(self, n_features):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(n_features, 1) * 0.01)
        self.bias = nn.Parameter(torch.zeros(1))
        # log(alpha)로 파라미터화
        self.log_alpha = nn.Parameter(torch.tensor(np.log(19.5)))  # V2 최적값에서 시작

    def forward(self, X):
        return X @ self.weights + self.bias

    def get_alpha(self):
        return torch.exp(self.log_alpha)

    def ridge_loss(self, X, y):
        """Ridge regression 손실함수"""
        predictions = self.forward(X)
        mse_loss = torch.mean((predictions - y) ** 2)
        l2_penalty = self.get_alpha() * torch.sum(self.weights ** 2)
        return mse_loss + l2_penalty

class GradientHyperparameterOptimizerV4:
    """경사하강법 기반 하이퍼파라미터 최적화기 v4"""

    def __init__(self, learning_rate=0.02, max_iterations=300, patience=75):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.patience = patience
        self.cv = PurgedKFoldPyTorch()
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

    def create_enhanced_features(self, data):
        """확장된 특성 엔지니어링 (35개+ 특성)"""
        logger.info("🔧 확장된 특성 엔지니어링 (35개+ 특성)...")

        returns = data['returns']
        features = pd.DataFrame(index=data.index)

        # 1. 기본 변동성 특성 (≤ t)
        for window in [3, 5, 10, 15, 20, 30, 50]:
            features[f'volatility_{window}'] = returns.rolling(window).std()

        # 2. 고차 모멘트 (≤ t)
        for window in [5, 10, 15, 20]:
            features[f'skew_{window}'] = returns.rolling(window).skew()
            features[f'kurt_{window}'] = returns.rolling(window).kurt()

        # 3. 래그 특성 (과거만)
        for lag in [1, 2, 3, 5, 7]:
            features[f'return_lag_{lag}'] = returns.shift(lag)
            features[f'vol_lag_{lag}'] = features['volatility_5'].shift(lag)

        # 4. 변동성 비율 (과거만)
        features['vol_ratio_5_20'] = features['volatility_5'] / (features['volatility_20'] + 1e-8)
        features['vol_ratio_10_50'] = features['volatility_10'] / (features['volatility_50'] + 1e-8)
        features['vol_ratio_3_15'] = features['volatility_3'] / (features['volatility_15'] + 1e-8)

        # 5. 통계적 특성 (과거만)
        for window in [10, 20, 30]:
            ma = returns.rolling(window).mean()
            std = returns.rolling(window).std()
            features[f'zscore_{window}'] = (returns - ma) / (std + 1e-8)
            features[f'sharpe_{window}'] = ma / (std + 1e-8)

        # 6. 모멘텀 및 이동평균 (과거만)
        for window in [3, 5, 10, 15, 20]:
            features[f'momentum_{window}'] = returns.rolling(window).sum()
            features[f'sma_{window}'] = returns.rolling(window).mean()

        # 7. 분위수 특성 (과거만)
        for window in [10, 20, 30]:
            features[f'q25_{window}'] = returns.rolling(window).quantile(0.25)
            features[f'q75_{window}'] = returns.rolling(window).quantile(0.75)
            features[f'iqr_{window}'] = features[f'q75_{window}'] - features[f'q25_{window}']

        # 8. 극값 및 드로우다운 (과거만)
        for window in [5, 10, 20]:
            features[f'max_drawdown_{window}'] = returns.rolling(window).apply(
                lambda x: (x.cumsum() - x.cumsum().cummax()).min()
            )
            features[f'max_return_{window}'] = returns.rolling(window).max()
            features[f'min_return_{window}'] = returns.rolling(window).min()

        # 9. 변동성 체제 감지 (과거만)
        short_vol = features['volatility_5']
        long_vol = features['volatility_20']
        features['vol_regime'] = (short_vol > long_vol).astype(int)
        features['vol_expansion'] = (short_vol / long_vol > 1.5).astype(int)

        # 10. 상호작용 특성 (고급)
        features['vol_momentum'] = features['volatility_10'] * features['momentum_10']
        features['return_vol_interaction'] = features['return_lag_1'] * features['volatility_5']

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

    def objective_function(self, X, y, log_alpha_value):
        """목적함수: Purged K-Fold CV의 음의 R² 스코어"""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y.values).reshape(-1, 1)

        cv_scores = []
        splits = self.cv.split(X)

        for train_idx, test_idx in splits:
            if len(train_idx) < 20 or len(test_idx) < 10:  # 더 엄격한 최소 요구사항
                continue

            X_train, X_test = X_tensor[train_idx], X_tensor[test_idx]
            y_train, y_test = y_tensor[train_idx], y_tensor[test_idx]

            # Ridge 모델 생성
            model = DifferentiableRidge(X.shape[1])
            model.log_alpha.data = torch.tensor(log_alpha_value)

            # 더 강력한 최적화
            optimizer = optim.LBFGS([model.weights, model.bias], lr=0.1, max_iter=20)

            def closure():
                optimizer.zero_grad()
                loss = model.ridge_loss(X_train, y_train)
                loss.backward()
                return loss

            # 충분한 훈련
            for _ in range(100):  # V2: 50회 → V4: 100회
                optimizer.step(closure)

            # 예측 및 평가
            with torch.no_grad():
                predictions = model.forward(X_test).numpy().flatten()
                r2 = r2_score(y_test.numpy().flatten(), predictions)
                cv_scores.append(r2)

        mean_r2 = np.mean(cv_scores) if cv_scores else -1.0
        return -mean_r2

    def optimize(self):
        """하이퍼파라미터 최적화 실행"""
        logger.info("🚀 경사하강법 기반 하이퍼파라미터 최적화 v4 시작")
        logger.info("🎯 v4 목표: R² > 0.33 (V2 대비 추가 개선)")

        # 데이터 준비
        data = self.load_spy_data()
        X, y = self.create_enhanced_features(data)

        # 표준화
        X_scaled = self.scaler.fit_transform(X)

        # V2 최적값에서 시작
        log_alpha = torch.tensor(np.log(19.5), requires_grad=True)
        optimizer = optim.Adam([log_alpha], lr=self.learning_rate)

        best_score = float('inf')
        best_alpha = 19.5
        patience_counter = 0

        logger.info(f"📈 최적화 시작: 초기 alpha = {torch.exp(log_alpha).item():.4f}")
        logger.info(f"📊 V2 성능 기준: R² = 0.3256")

        for iteration in range(self.max_iterations):
            start_time = time.time()

            # 목적함수 계산
            score = self.objective_function(X_scaled, y, log_alpha.item())

            # 경사 계산 (수치 미분)
            eps = 1e-7  # 더 작은 eps
            score_plus = self.objective_function(X_scaled, y, log_alpha.item() + eps)
            score_minus = self.objective_function(X_scaled, y, log_alpha.item() - eps)
            gradient = (score_plus - score_minus) / (2 * eps)

            # 적응적 학습률
            if patience_counter > 20:
                current_lr = self.learning_rate * 0.5
            elif patience_counter > 10:
                current_lr = self.learning_rate * 0.8
            else:
                current_lr = self.learning_rate

            # 경사하강법 업데이트
            with torch.no_grad():
                log_alpha -= current_lr * gradient
                # V2 결과 기준 제한된 탐색 (5 ~ 100)
                log_alpha.clamp_(np.log(5.0), np.log(100.0))

            current_alpha = torch.exp(log_alpha).item()
            elapsed_time = time.time() - start_time

            # 최적값 업데이트
            if score < best_score:
                best_score = score
                best_alpha = current_alpha
                patience_counter = 0

                current_r2 = -score
                if current_r2 > 0.33:
                    logger.info(f"🚀 V4 목표 달성! R² = {current_r2:.6f} > 0.33")
            else:
                patience_counter += 1

            # 로깅
            if iteration % 5 == 0:
                current_r2 = -score
                best_r2 = -best_score
                logger.info(
                    f"반복 {iteration:4d}: α={current_alpha:.6f}, "
                    f"R²={current_r2:.6f}, 최적R²={best_r2:.6f}, "
                    f"lr={current_lr:.4f}, 시간={elapsed_time:.2f}s"
                )

            # 히스토리 저장
            self.history.append({
                'iteration': iteration,
                'alpha': current_alpha,
                'score': score,
                'best_score': best_score,
                'best_alpha': best_alpha,
                'learning_rate': current_lr,
                'timestamp': datetime.now().isoformat()
            })

            # 조기 종료
            if patience_counter >= self.patience:
                logger.info(f"🛑 조기 종료: {self.patience}번 연속 개선 없음")
                break

        # 최종 결과
        final_r2 = -best_score
        logger.info(f"✅ v4 최적화 완료!")
        logger.info(f"📊 최적 alpha: {best_alpha:.6f}")
        logger.info(f"📊 최적 R²: {final_r2:.6f}")

        return best_alpha, final_r2, self.history

    def save_results(self, best_alpha, best_score, history):
        """결과 저장"""
        results = {
            'version': 'v4',
            'improvements': [
                '확장된 35개+ 특성 사용',
                'V2 최적값(α=19.5)에서 시작',
                '더 강력한 모델 훈련 (100회)',
                '적응적 학습률 적용',
                '더 긴 patience (75회)'
            ],
            'optimization_completed': datetime.now().isoformat(),
            'best_hyperparameters': {
                'alpha': float(best_alpha),
                'model_type': 'Ridge'
            },
            'best_performance': {
                'r2_score': float(best_score),
                'method': 'Purged_K_Fold_CV'
            },
            'baseline_comparison': {
                'original_baseline_r2': 0.3113,
                'v2_baseline_r2': 0.3256,
                'v4_optimized_r2': float(best_score),
                'improvement_from_original': float(best_score - 0.3113),
                'improvement_from_v2': float(best_score - 0.3256),
                'improvement_percent': float(((best_score - 0.3113) / 0.3113) * 100)
            },
            'optimization_history': history,
            'configuration': {
                'learning_rate': self.learning_rate,
                'max_iterations': self.max_iterations,
                'patience': self.patience
            }
        }

        # 결과 저장
        save_path = '/root/workspace/data/raw/gradient_optimization_results_v4.json'
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"💾 v4 결과 저장됨: {save_path}")
        return results

def main():
    """메인 실행 함수"""
    logger.info("🎯 경사하강법 하이퍼파라미터 최적화 v4 시작")
    logger.info("🔧 v4 개선사항: 확장된 특성 + V2 기반 시작")

    optimizer = GradientHyperparameterOptimizerV4(
        learning_rate=0.02,
        max_iterations=300,
        patience=75
    )

    try:
        best_alpha, best_score, history = optimizer.optimize()
        results = optimizer.save_results(best_alpha, best_score, history)

        # 성능 비교
        v2_r2 = 0.3256
        improvement = best_score - v2_r2
        improvement_pct = (improvement / v2_r2) * 100

        logger.info("📈 v4 성능 비교:")
        logger.info(f"   V2 모델: R² = {v2_r2:.4f}")
        logger.info(f"   V4 모델: R² = {best_score:.4f}")
        logger.info(f"   V2 대비: {improvement:+.4f} ({improvement_pct:+.2f}%)")

        if best_score > v2_r2:
            logger.info("🎉 v4 성공: V2 성능 초과!")
        else:
            logger.info("⚠️ v4 아쉬움: V5로 추가 개선")

        return results

    except Exception as e:
        logger.error(f"❌ v4 최적화 실패: {str(e)}")
        raise

if __name__ == "__main__":
    main()