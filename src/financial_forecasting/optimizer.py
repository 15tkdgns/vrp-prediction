"""
Financial Performance Metrics Optimization

금융 성과 지표 기반 모델 최적화:
1. FinancialObjectiveOptimizer: 샤프/소르티노/MDD 기반 목적함수 최적화
2. HyperparameterOptimizer: 금융 지표 기준 하이퍼파라미터 튜닝
3. EnsembleOptimizer: 포트폴리오 이론 기반 모델 앙상블 최적화
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

try:
    from scipy.optimize import minimize, differential_evolution
    from scipy.stats import norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ scipy not available, using simplified optimization")

try:
    from sklearn.model_selection import ParameterGrid
    from sklearn.metrics import make_scorer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ sklearn not available for parameter grid search")

from .core import FinancialMetrics
from .validation import WalkForwardValidator, FinancialBacktester


@dataclass
class OptimizationResult:
    """최적화 결과"""
    best_params: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict]
    convergence_info: Dict
    financial_metrics: Dict


@dataclass
class EnsembleWeights:
    """앙상블 가중치"""
    weights: np.ndarray
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    diversification_ratio: float


class FinancialObjectiveOptimizer:
    """
    Financial Performance Based Objective Function Optimizer

    전통적 ML 지표(MSE, 정확도) 대신 실제 금융 성과 지표로 최적화:
    - 샤프 비율 최대화
    - 소르티노 비율 최대화
    - 최대 낙폭(MDD) 최소화
    - 칼마 비율 최대화
    - 복합 목적함수 최적화
    """

    def __init__(
        self,
        objective_weights: Optional[Dict[str, float]] = None,
        risk_free_rate: float = 0.02
    ):
        """
        Args:
            objective_weights: 목적함수 가중치
                {
                    'sharpe': 0.4,      # 샤프 비율
                    'sortino': 0.3,     # 소르티노 비율
                    'calmar': 0.2,      # 칼마 비율
                    'mdd_penalty': 0.1  # MDD 페널티
                }
            risk_free_rate: 무위험 수익률
        """
        if objective_weights is None:
            objective_weights = {
                'sharpe': 0.5,
                'sortino': 0.3,
                'calmar': 0.2,
                'mdd_penalty': 0.0
            }

        self.objective_weights = objective_weights
        self.risk_free_rate = risk_free_rate

    def calculate_financial_objective(
        self,
        predictions: np.ndarray,
        actual_returns: np.ndarray,
        prices: Optional[np.ndarray] = None
    ) -> float:
        """
        금융 성과 기반 목적함수 계산

        Args:
            predictions: 모델 예측값 (수익률 또는 방향)
            actual_returns: 실제 수익률
            prices: 가격 데이터 (선택적)

        Returns:
            복합 목적함수 점수 (높을수록 좋음)
        """
        try:
            # 예측 기반 포트폴리오 수익률 계산
            if len(predictions) != len(actual_returns):
                # 길이가 다르면 짧은 쪽에 맞춤
                min_len = min(len(predictions), len(actual_returns))
                predictions = predictions[:min_len]
                actual_returns = actual_returns[:min_len]

            # 거래 신호 생성 (예측 값이 임계치 이상이면 매수)
            signal_threshold = np.median(predictions)
            signals = np.where(predictions > signal_threshold, 1, 0)

            # 전략 수익률 계산 (신호 * 실제 수익률)
            strategy_returns = signals * actual_returns

            # 수익률이 0인 경우 처리
            if len(strategy_returns) == 0 or np.std(strategy_returns) == 0:
                return -1.0  # 최악 점수

            # pandas Series로 변환
            returns_series = pd.Series(strategy_returns)

            # 금융 성과 지표 계산
            metrics = FinancialMetrics.calculate_comprehensive_metrics(
                returns_series, self.risk_free_rate
            )

            # 복합 목적함수 계산
            objective_score = 0.0

            # 샤프 비율 (정규화: 일반적으로 -2 ~ 3 범위)
            sharpe_normalized = np.clip(metrics.sharpe_ratio / 3.0, -1, 1)
            objective_score += self.objective_weights.get('sharpe', 0) * sharpe_normalized

            # 소르티노 비율 (정규화: 일반적으로 -3 ~ 4 범위)
            sortino_normalized = np.clip(metrics.sortino_ratio / 4.0, -1, 1)
            objective_score += self.objective_weights.get('sortino', 0) * sortino_normalized

            # 칼마 비율 (정규화: 일반적으로 -2 ~ 2 범위)
            calmar_normalized = np.clip(metrics.calmar_ratio / 2.0, -1, 1)
            objective_score += self.objective_weights.get('calmar', 0) * calmar_normalized

            # MDD 페널티 (0 ~ 1 범위, 1에 가까울수록 낮은 MDD)
            mdd_penalty = 1.0 - np.clip(metrics.max_drawdown, 0, 1)
            objective_score += self.objective_weights.get('mdd_penalty', 0) * mdd_penalty

            return objective_score

        except Exception as e:
            print(f"⚠️ 목적함수 계산 실패: {e}")
            return -1.0  # 최악 점수 반환

    def create_sklearn_scorer(self) -> Callable:
        """scikit-learn 호환 스코어러 생성"""
        def financial_scorer(y_true, y_pred):
            return self.calculate_financial_objective(y_pred, y_true)

        if SKLEARN_AVAILABLE:
            return make_scorer(financial_scorer, greater_is_better=True)
        else:
            return financial_scorer


class HyperparameterOptimizer:
    """
    Financial Metrics Based Hyperparameter Optimization

    금융 성과 지표를 목적함수로 하는 하이퍼파라미터 최적화:
    - Grid Search with Financial Objective
    - Bayesian Optimization (금융 지표 기반)
    - Differential Evolution
    - Walk-Forward Validation 결합
    """

    def __init__(
        self,
        objective_optimizer: FinancialObjectiveOptimizer,
        validator: WalkForwardValidator,
        optimization_method: str = "grid_search"  # "grid_search", "differential_evolution"
    ):
        self.objective_optimizer = objective_optimizer
        self.validator = validator
        self.optimization_method = optimization_method

    def optimize_hyperparameters(
        self,
        model_class: type,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Dict[str, List],
        cv_folds: int = 5,
        n_jobs: int = 1
    ) -> OptimizationResult:
        """
        하이퍼파라미터 최적화 수행

        Args:
            model_class: 모델 클래스
            X: 특성 데이터
            y: 타겟 데이터 (수익률)
            param_grid: 파라미터 그리드
            cv_folds: CV 폴드 수
            n_jobs: 병렬 작업 수

        Returns:
            최적화 결과
        """
        print(f"🔧 하이퍼파라미터 최적화 시작 ({self.optimization_method})")
        print(f"   모델: {model_class.__name__}")
        print(f"   파라미터 조합: {self._count_combinations(param_grid)}개")

        if self.optimization_method == "grid_search":
            return self._grid_search_optimization(model_class, X, y, param_grid, cv_folds)
        elif self.optimization_method == "differential_evolution":
            return self._differential_evolution_optimization(model_class, X, y, param_grid, cv_folds)
        else:
            raise ValueError(f"지원하지 않는 최적화 방법: {self.optimization_method}")

    def _grid_search_optimization(
        self,
        model_class: type,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Dict[str, List],
        cv_folds: int
    ) -> OptimizationResult:
        """그리드 서치 최적화"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn이 필요합니다")

        best_score = -np.inf
        best_params = None
        optimization_history = []

        # 파라미터 조합 생성
        param_combinations = list(ParameterGrid(param_grid))

        for i, params in enumerate(param_combinations):
            try:
                print(f"   진행률: {i+1}/{len(param_combinations)} - {params}")

                # 모델 생성
                model = model_class(**params)

                # Walk-Forward Validation
                cv_results = self.validator.validate(model, X, y)

                if not cv_results:
                    continue

                # 모든 폴드의 예측값과 실제값 결합
                all_predictions = np.concatenate([r.predictions for r in cv_results])
                all_actuals = np.concatenate([r.actuals for r in cv_results])

                # 금융 목적함수 점수 계산
                score = self.objective_optimizer.calculate_financial_objective(
                    all_predictions, all_actuals
                )

                # 기록
                optimization_history.append({
                    'params': params.copy(),
                    'score': score,
                    'cv_results': len(cv_results)
                })

                # 최고 점수 업데이트
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    print(f"   ✅ 새로운 최고 점수: {score:.4f}")

            except Exception as e:
                print(f"   ❌ 파라미터 {params} 실패: {e}")
                continue

        # 최종 결과
        result = OptimizationResult(
            best_params=best_params or {},
            best_score=best_score,
            optimization_history=optimization_history,
            convergence_info={'method': 'grid_search', 'total_combinations': len(param_combinations)},
            financial_metrics={}
        )

        print(f"🎯 최적화 완료:")
        print(f"   최고 점수: {best_score:.4f}")
        print(f"   최적 파라미터: {best_params}")

        return result

    def _differential_evolution_optimization(
        self,
        model_class: type,
        X: pd.DataFrame,
        y: pd.Series,
        param_bounds: Dict[str, Tuple[float, float]],
        cv_folds: int
    ) -> OptimizationResult:
        """Differential Evolution 최적화"""
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy가 필요합니다")

        param_names = list(param_bounds.keys())
        bounds = list(param_bounds.values())

        optimization_history = []

        def objective_function(params_array):
            try:
                # 파라미터 딕셔너리 생성
                params = dict(zip(param_names, params_array))

                # 정수형 파라미터 처리
                for key, value in params.items():
                    if key in ['n_estimators', 'max_depth', 'min_samples_split']:
                        params[key] = int(value)

                # 모델 생성 및 검증
                model = model_class(**params)
                cv_results = self.validator.validate(model, X, y)

                if not cv_results:
                    return 1.0  # 최악 점수 (최소화 문제이므로)

                # 예측값 결합
                all_predictions = np.concatenate([r.predictions for r in cv_results])
                all_actuals = np.concatenate([r.actuals for r in cv_results])

                # 금융 목적함수 점수 (최대화 → 최소화로 변환)
                score = self.objective_optimizer.calculate_financial_objective(
                    all_predictions, all_actuals
                )

                optimization_history.append({
                    'params': params.copy(),
                    'score': score
                })

                return -score  # 최소화 문제로 변환

            except Exception as e:
                print(f"⚠️ 목적함수 평가 실패: {e}")
                return 1.0

        # Differential Evolution 실행
        result = differential_evolution(
            objective_function,
            bounds,
            maxiter=50,
            popsize=10,
            seed=42
        )

        # 최적 파라미터 복원
        best_params = dict(zip(param_names, result.x))
        for key, value in best_params.items():
            if key in ['n_estimators', 'max_depth', 'min_samples_split']:
                best_params[key] = int(value)

        optimization_result = OptimizationResult(
            best_params=best_params,
            best_score=-result.fun,  # 원래 점수로 복원
            optimization_history=optimization_history,
            convergence_info={
                'method': 'differential_evolution',
                'success': result.success,
                'message': result.message,
                'iterations': result.nit
            },
            financial_metrics={}
        )

        print(f"🎯 Differential Evolution 완료:")
        print(f"   최고 점수: {-result.fun:.4f}")
        print(f"   최적 파라미터: {best_params}")

        return optimization_result

    def _count_combinations(self, param_grid: Dict[str, List]) -> int:
        """파라미터 조합 수 계산"""
        count = 1
        for values in param_grid.values():
            count *= len(values)
        return count


class EnsembleOptimizer:
    """
    Portfolio Theory Based Ensemble Optimization

    포트폴리오 이론을 적용한 모델 앙상블 최적화:
    - 마코위츠 평균-분산 최적화
    - 리스크 패리티 (Risk Parity)
    - 최대 다각화 (Maximum Diversification)
    - 최소 분산 (Minimum Variance)
    """

    def __init__(self, risk_aversion: float = 1.0):
        """
        Args:
            risk_aversion: 위험 회피 계수 (높을수록 보수적)
        """
        self.risk_aversion = risk_aversion

    def optimize_ensemble_weights(
        self,
        model_predictions: Dict[str, np.ndarray],
        actual_returns: np.ndarray,
        method: str = "markowitz"  # "markowitz", "risk_parity", "max_diversification", "min_variance"
    ) -> EnsembleWeights:
        """
        앙상블 가중치 최적화

        Args:
            model_predictions: 모델별 예측값 딕셔너리
            actual_returns: 실제 수익률
            method: 최적화 방법

        Returns:
            최적 앙상블 가중치
        """
        model_names = list(model_predictions.keys())
        n_models = len(model_names)

        # 각 모델의 전략 수익률 계산
        strategy_returns_matrix = []
        for model_name, predictions in model_predictions.items():
            # 신호 생성 (임계치 기반)
            threshold = np.median(predictions)
            signals = np.where(predictions > threshold, 1, 0)

            # 전략 수익률
            strategy_returns = signals * actual_returns
            strategy_returns_matrix.append(strategy_returns)

        returns_matrix = np.array(strategy_returns_matrix).T  # (시간, 모델)

        # 수익률 및 공분산 행렬 계산
        mean_returns = np.mean(returns_matrix, axis=0)
        cov_matrix = np.cov(returns_matrix.T)

        # 최적화 방법별 가중치 계산
        if method == "markowitz":
            weights = self._markowitz_optimization(mean_returns, cov_matrix)
        elif method == "risk_parity":
            weights = self._risk_parity_optimization(cov_matrix)
        elif method == "max_diversification":
            weights = self._max_diversification_optimization(mean_returns, cov_matrix)
        elif method == "min_variance":
            weights = self._min_variance_optimization(cov_matrix)
        else:
            # 균등 가중치 (기본값)
            weights = np.ones(n_models) / n_models

        # 포트폴리오 성과 계산
        portfolio_returns = returns_matrix @ weights
        expected_return = np.mean(portfolio_returns) * 252  # 연환산
        expected_risk = np.std(portfolio_returns) * np.sqrt(252)  # 연환산
        sharpe_ratio = expected_return / expected_risk if expected_risk > 0 else 0

        # 다각화 비율 계산
        individual_risks = np.sqrt(np.diag(cov_matrix)) * np.sqrt(252)
        weighted_avg_risk = weights @ individual_risks
        diversification_ratio = weighted_avg_risk / expected_risk if expected_risk > 0 else 1

        result = EnsembleWeights(
            weights=weights,
            expected_return=expected_return,
            expected_risk=expected_risk,
            sharpe_ratio=sharpe_ratio,
            diversification_ratio=diversification_ratio
        )

        print(f"📊 앙상블 최적화 완료 ({method}):")
        print(f"   예상 수익률: {expected_return:.2%}")
        print(f"   예상 위험: {expected_risk:.2%}")
        print(f"   샤프 비율: {sharpe_ratio:.3f}")
        print(f"   다각화 비율: {diversification_ratio:.3f}")
        for i, (name, weight) in enumerate(zip(model_names, weights)):
            print(f"   {name}: {weight:.3f}")

        return result

    def _markowitz_optimization(self, mean_returns: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
        """마코위츠 평균-분산 최적화"""
        if not SCIPY_AVAILABLE:
            return np.ones(len(mean_returns)) / len(mean_returns)

        n = len(mean_returns)

        # 목적함수: 기대수익률 최대화 - 위험회피계수 * 분산
        def objective(weights):
            portfolio_return = weights @ mean_returns
            portfolio_variance = weights @ cov_matrix @ weights
            return -(portfolio_return - 0.5 * self.risk_aversion * portfolio_variance)

        # 제약조건: 가중치 합 = 1, 가중치 >= 0
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n)]

        # 최적화 실행
        result = minimize(
            objective,
            x0=np.ones(n) / n,
            bounds=bounds,
            constraints=constraints,
            method='SLSQP'
        )

        return result.x if result.success else np.ones(n) / n

    def _risk_parity_optimization(self, cov_matrix: np.ndarray) -> np.ndarray:
        """리스크 패리티 최적화 (각 모델의 위험 기여도 동일)"""
        if not SCIPY_AVAILABLE:
            return np.ones(len(cov_matrix)) / len(cov_matrix)

        n = len(cov_matrix)

        def objective(weights):
            # 각 자산의 한계 위험 기여도
            portfolio_variance = weights @ cov_matrix @ weights
            marginal_contrib = cov_matrix @ weights
            contrib = weights * marginal_contrib / portfolio_variance

            # 위험 기여도의 분산 최소화
            target = 1.0 / n
            return np.sum((contrib - target) ** 2)

        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0.01, 1) for _ in range(n)]

        result = minimize(
            objective,
            x0=np.ones(n) / n,
            bounds=bounds,
            constraints=constraints,
            method='SLSQP'
        )

        return result.x if result.success else np.ones(n) / n

    def _max_diversification_optimization(self, mean_returns: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
        """최대 다각화 최적화"""
        if not SCIPY_AVAILABLE:
            return np.ones(len(mean_returns)) / len(mean_returns)

        n = len(cov_matrix)

        def objective(weights):
            # 다각화 비율 = (가중평균 변동성) / (포트폴리오 변동성)
            individual_vols = np.sqrt(np.diag(cov_matrix))
            weighted_avg_vol = weights @ individual_vols
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            return -weighted_avg_vol / portfolio_vol  # 최대화를 위해 음수

        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n)]

        result = minimize(
            objective,
            x0=np.ones(n) / n,
            bounds=bounds,
            constraints=constraints,
            method='SLSQP'
        )

        return result.x if result.success else np.ones(n) / n

    def _min_variance_optimization(self, cov_matrix: np.ndarray) -> np.ndarray:
        """최소 분산 최적화"""
        if not SCIPY_AVAILABLE:
            return np.ones(len(cov_matrix)) / len(cov_matrix)

        n = len(cov_matrix)

        def objective(weights):
            return weights @ cov_matrix @ weights

        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n)]

        result = minimize(
            objective,
            x0=np.ones(n) / n,
            bounds=bounds,
            constraints=constraints,
            method='SLSQP'
        )

        return result.x if result.success else np.ones(n) / n