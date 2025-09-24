"""
경사하강법 기반 하이퍼파라미터 최적화 시스템
Ridge Regression alpha 파라미터를 연속적으로 최적화

핵심 아이디어:
1. Purged K-Fold CV를 미분가능한 목적함수로 구현
2. log(alpha) 공간에서 경사하강법 적용
3. 백그라운드 실행하면서 실시간 모니터링
4. 최적 파라미터 자동 저장 및 추적
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
        logging.FileHandler('/root/workspace/data/raw/gradient_optimization.log'),
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

class DifferentiableRidge(nn.Module):
    """미분가능한 Ridge Regression 구현"""

    def __init__(self, n_features):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(n_features, 1) * 0.01)
        self.bias = nn.Parameter(torch.zeros(1))
        # log(alpha)로 파라미터화 (항상 양수 보장)
        self.log_alpha = nn.Parameter(torch.tensor(0.0))  # alpha = 1.0 시작

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

class GradientHyperparameterOptimizer:
    """경사하강법 기반 하이퍼파라미터 최적화기"""

    def __init__(self, learning_rate=0.01, max_iterations=1000, patience=50):
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

    def create_features(self, data):
        """특성 및 타겟 생성"""
        logger.info("🔧 특성 엔지니어링...")

        returns = data['returns']
        features = pd.DataFrame(index=data.index)

        # 과거 변동성 특성 (≤ t)
        for window in [5, 10, 20, 50]:
            features[f'volatility_{window}'] = returns.rolling(window).std()

        # 래그 특성
        for lag in [1, 2, 3, 5]:
            features[f'return_lag_{lag}'] = returns.shift(lag)
            features[f'vol_lag_{lag}'] = features['volatility_5'].shift(lag)

        # 비율 특성
        features['vol_ratio_5_20'] = features['volatility_5'] / (features['volatility_20'] + 1e-8)
        features['vol_ratio_10_50'] = features['volatility_10'] / (features['volatility_50'] + 1e-8)

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
            if len(train_idx) < 10 or len(test_idx) < 5:
                continue

            X_train, X_test = X_tensor[train_idx], X_tensor[test_idx]
            y_train, y_test = y_tensor[train_idx], y_tensor[test_idx]

            # Ridge 모델 생성
            model = DifferentiableRidge(X.shape[1])
            model.log_alpha.data = torch.tensor(log_alpha_value)

            # 모델 훈련 (가중치만 최적화, alpha는 고정)
            optimizer = optim.LBFGS(model.parameters(), lr=0.1)

            def closure():
                optimizer.zero_grad()
                loss = model.ridge_loss(X_train, y_train)
                loss.backward()
                return loss

            # 간단한 훈련 (몇 번의 iteration만)
            for _ in range(10):
                optimizer.step(closure)

            # 예측 및 평가
            with torch.no_grad():
                predictions = model.forward(X_test).numpy().flatten()
                r2 = r2_score(y_test.numpy().flatten(), predictions)
                cv_scores.append(r2)

        mean_r2 = np.mean(cv_scores) if cv_scores else -1.0
        return -mean_r2  # 최소화 문제로 변환

    def optimize(self):
        """하이퍼파라미터 최적화 실행"""
        logger.info("🚀 경사하강법 기반 하이퍼파라미터 최적화 시작")

        # 데이터 준비
        data = self.load_spy_data()
        X, y = self.create_features(data)

        # 표준화
        X_scaled = self.scaler.fit_transform(X)

        # 최적화 변수 초기화
        log_alpha = torch.tensor(0.0, requires_grad=True)  # alpha = 1.0 시작
        optimizer = optim.Adam([log_alpha], lr=self.learning_rate)

        best_score = float('inf')
        best_alpha = 1.0
        patience_counter = 0

        logger.info(f"📈 최적화 시작: 초기 alpha = {torch.exp(log_alpha).item():.4f}")

        for iteration in range(self.max_iterations):
            start_time = time.time()

            # 목적함수 계산
            score = self.objective_function(X_scaled, y, log_alpha.item())

            # 경사 계산 (수치 미분)
            eps = 1e-6
            score_plus = self.objective_function(X_scaled, y, log_alpha.item() + eps)
            score_minus = self.objective_function(X_scaled, y, log_alpha.item() - eps)
            gradient = (score_plus - score_minus) / (2 * eps)

            # 수동 경사하강법 업데이트
            with torch.no_grad():
                log_alpha -= self.learning_rate * gradient
                # 범위 제한 (alpha: 0.001 ~ 1000)
                log_alpha.clamp_(-6.9, 6.9)  # exp(-6.9) ≈ 0.001, exp(6.9) ≈ 1000

            current_alpha = torch.exp(log_alpha).item()
            elapsed_time = time.time() - start_time

            # 최적값 업데이트
            if score < best_score:
                best_score = score
                best_alpha = current_alpha
                patience_counter = 0
            else:
                patience_counter += 1

            # 로깅
            if iteration % 10 == 0:
                logger.info(
                    f"반복 {iteration:4d}: α={current_alpha:.6f}, "
                    f"손실={score:.6f}, 최적손실={best_score:.6f}, "
                    f"시간={elapsed_time:.2f}s"
                )

            # 히스토리 저장
            self.history.append({
                'iteration': iteration,
                'alpha': current_alpha,
                'score': score,
                'best_score': best_score,
                'best_alpha': best_alpha,
                'timestamp': datetime.now().isoformat()
            })

            # 조기 종료
            if patience_counter >= self.patience:
                logger.info(f"🛑 조기 종료: {self.patience}번 연속 개선 없음")
                break

        # 최종 결과
        logger.info(f"✅ 최적화 완료!")
        logger.info(f"📊 최적 alpha: {best_alpha:.6f}")
        logger.info(f"📊 최적 점수: {-best_score:.6f} (R²)")

        return best_alpha, -best_score, self.history

    def save_results(self, best_alpha, best_score, history):
        """결과 저장"""
        results = {
            'optimization_completed': datetime.now().isoformat(),
            'best_hyperparameters': {
                'alpha': best_alpha,
                'model_type': 'Ridge'
            },
            'best_performance': {
                'r2_score': best_score,
                'method': 'Purged_K_Fold_CV'
            },
            'optimization_history': history,
            'configuration': {
                'learning_rate': self.learning_rate,
                'max_iterations': self.max_iterations,
                'patience': self.patience
            }
        }

        # 결과 저장
        save_path = '/root/workspace/data/raw/gradient_optimization_results.json'
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"💾 결과 저장됨: {save_path}")
        return results

def main():
    """메인 실행 함수"""
    logger.info("🎯 경사하강법 하이퍼파라미터 최적화 시작")

    # 최적화기 생성
    optimizer = GradientHyperparameterOptimizer(
        learning_rate=0.1,      # 학습률
        max_iterations=500,     # 최대 반복 횟수
        patience=30             # 조기 종료 기준
    )

    try:
        # 최적화 실행
        best_alpha, best_score, history = optimizer.optimize()

        # 결과 저장
        results = optimizer.save_results(best_alpha, best_score, history)

        # 현재 모델 성능과 비교
        baseline_r2 = 0.3113
        improvement = best_score - baseline_r2
        improvement_pct = (improvement / baseline_r2) * 100

        logger.info("📈 성능 비교:")
        logger.info(f"   기존 모델 (α=1.0): R² = {baseline_r2:.4f}")
        logger.info(f"   최적화 모델 (α={best_alpha:.4f}): R² = {best_score:.4f}")
        logger.info(f"   성능 향상: {improvement:+.4f} ({improvement_pct:+.2f}%)")

        return results

    except Exception as e:
        logger.error(f"❌ 최적화 실패: {str(e)}")
        raise

if __name__ == "__main__":
    main()