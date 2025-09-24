#!/usr/bin/env python3
"""
🔢 신뢰구간 계산기
학술 논문을 위한 정확한 신뢰구간 계산 도구

주요 기능:
- 다양한 통계량의 신뢰구간 계산
- Bootstrap 신뢰구간
- 모델 성능 지표별 신뢰구간
- 금융 시계열 특화 신뢰구간
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import t, norm, chi2
from typing import Tuple, Dict, List, Optional, Union
import warnings
from sklearn.utils import resample
warnings.filterwarnings('ignore')

class ConfidenceIntervalCalculator:
    """
    다양한 통계량에 대한 신뢰구간 계산 클래스

    금융 머신러닝 모델 평가에 특화된 신뢰구간 계산 도구
    """

    def __init__(self, confidence_level: float = 0.95):
        """
        초기화

        Args:
            confidence_level: 신뢰수준 (기본값: 0.95, 즉 95%)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

    def mean_ci(self, data: np.ndarray, method: str = "parametric") -> Dict:
        """
        평균의 신뢰구간 계산

        Args:
            data: 데이터 배열
            method: 계산 방법 ("parametric", "bootstrap")

        Returns:
            신뢰구간 정보 딕셔너리
        """
        n = len(data)
        mean = np.mean(data)

        if method == "parametric":
            # 모수적 방법 (t-분포 기반)
            std_err = stats.sem(data)  # 표준오차
            t_critical = t.ppf(1 - self.alpha/2, n-1)
            margin_error = t_critical * std_err

            ci_lower = mean - margin_error
            ci_upper = mean + margin_error

            method_used = f"t-distribution (df={n-1})"

        elif method == "bootstrap":
            # Bootstrap 방법
            ci_lower, ci_upper = self._bootstrap_ci(data, np.mean)
            method_used = "Bootstrap"

        else:
            raise ValueError("method는 'parametric' 또는 'bootstrap'이어야 합니다")

        return {
            'statistic': 'mean',
            'value': mean,
            'confidence_level': self.confidence_level,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'method': method_used,
            'sample_size': n,
            'standard_error': stats.sem(data) if method == "parametric" else None
        }

    def proportion_ci(self, successes: int, total: int, method: str = "wilson") -> Dict:
        """
        비율의 신뢰구간 계산 (예: 정확도, 방향 예측 정확도)

        Args:
            successes: 성공 횟수
            total: 전체 시도 횟수
            method: 계산 방법 ("wilson", "agresti_coull", "clopper_pearson")

        Returns:
            신뢰구간 정보 딕셔너리
        """
        if successes > total:
            raise ValueError("성공 횟수가 전체 시도 횟수보다 클 수 없습니다")

        p = successes / total
        z_critical = norm.ppf(1 - self.alpha/2)

        if method == "wilson":
            # Wilson Score Interval (권장)
            denominator = 1 + z_critical**2 / total
            center = (p + z_critical**2 / (2 * total)) / denominator
            margin = z_critical * np.sqrt(p * (1 - p) / total + z_critical**2 / (4 * total**2)) / denominator

            ci_lower = center - margin
            ci_upper = center + margin
            method_used = "Wilson Score Interval"

        elif method == "agresti_coull":
            # Agresti-Coull Interval
            n_tilde = total + z_critical**2
            p_tilde = (successes + z_critical**2 / 2) / n_tilde
            margin = z_critical * np.sqrt(p_tilde * (1 - p_tilde) / n_tilde)

            ci_lower = p_tilde - margin
            ci_upper = p_tilde + margin
            method_used = "Agresti-Coull Interval"

        elif method == "clopper_pearson":
            # Clopper-Pearson Exact Interval
            if successes == 0:
                ci_lower = 0
            else:
                ci_lower = stats.beta.ppf(self.alpha/2, successes, total - successes + 1)

            if successes == total:
                ci_upper = 1
            else:
                ci_upper = stats.beta.ppf(1 - self.alpha/2, successes + 1, total - successes)

            method_used = "Clopper-Pearson Exact"

        else:
            raise ValueError("method는 'wilson', 'agresti_coull', 또는 'clopper_pearson'이어야 합니다")

        # 0과 1 사이로 제한
        ci_lower = max(0, ci_lower)
        ci_upper = min(1, ci_upper)

        return {
            'statistic': 'proportion',
            'value': p,
            'successes': successes,
            'total': total,
            'confidence_level': self.confidence_level,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'method': method_used
        }

    def correlation_ci(self, correlation: float, n: int) -> Dict:
        """
        상관계수의 신뢰구간 계산 (Fisher's z-transformation)

        Args:
            correlation: 표본 상관계수
            n: 표본 크기

        Returns:
            신뢰구간 정보 딕셔너리
        """
        if abs(correlation) >= 1:
            raise ValueError("상관계수는 -1과 1 사이여야 합니다")

        if n < 4:
            raise ValueError("상관계수 신뢰구간 계산을 위해 최소 4개 관측값이 필요합니다")

        # Fisher's z-transformation
        z_r = 0.5 * np.log((1 + correlation) / (1 - correlation))

        # 표준오차
        se_z = 1 / np.sqrt(n - 3)

        # z 신뢰구간
        z_critical = norm.ppf(1 - self.alpha/2)
        z_lower = z_r - z_critical * se_z
        z_upper = z_r + z_critical * se_z

        # 역변환으로 r 신뢰구간
        ci_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
        ci_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)

        return {
            'statistic': 'correlation',
            'value': correlation,
            'confidence_level': self.confidence_level,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'method': "Fisher's z-transformation",
            'sample_size': n,
            'fisher_z': z_r,
            'standard_error_z': se_z
        }

    def variance_ci(self, data: np.ndarray) -> Dict:
        """
        분산의 신뢰구간 계산 (카이제곱 분포 기반)

        Args:
            data: 데이터 배열

        Returns:
            신뢰구간 정보 딕셔너리
        """
        n = len(data)
        sample_var = np.var(data, ddof=1)
        df = n - 1

        # 카이제곱 분포의 임계값
        chi2_lower = chi2.ppf(self.alpha/2, df)
        chi2_upper = chi2.ppf(1 - self.alpha/2, df)

        # 분산의 신뢰구간
        ci_lower = (df * sample_var) / chi2_upper
        ci_upper = (df * sample_var) / chi2_lower

        return {
            'statistic': 'variance',
            'value': sample_var,
            'confidence_level': self.confidence_level,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'method': f"Chi-square distribution (df={df})",
            'sample_size': n,
            'degrees_of_freedom': df
        }

    def model_performance_ci(self, y_true: np.ndarray, y_pred: np.ndarray,
                           metric: str = "mae", n_bootstrap: int = 1000) -> Dict:
        """
        모델 성능 지표의 신뢰구간 계산 (Bootstrap 방법)

        Args:
            y_true: 실제 값
            y_pred: 예측 값
            metric: 성능 지표 ("mae", "mse", "rmse", "r2", "mape")
            n_bootstrap: Bootstrap 반복 횟수

        Returns:
            신뢰구간 정보 딕셔너리
        """
        n = len(y_true)
        if len(y_pred) != n:
            raise ValueError("y_true와 y_pred의 길이가 일치하지 않습니다")

        # 성능 지표 계산 함수
        def calculate_metric(y_t, y_p):
            if metric == "mae":
                return np.mean(np.abs(y_t - y_p))
            elif metric == "mse":
                return np.mean((y_t - y_p)**2)
            elif metric == "rmse":
                return np.sqrt(np.mean((y_t - y_p)**2))
            elif metric == "r2":
                ss_res = np.sum((y_t - y_p)**2)
                ss_tot = np.sum((y_t - np.mean(y_t))**2)
                return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            elif metric == "mape":
                return np.mean(np.abs((y_t - y_p) / y_t)) * 100
            else:
                raise ValueError(f"지원하지 않는 지표입니다: {metric}")

        # 원본 성능
        original_score = calculate_metric(y_true, y_pred)

        # Bootstrap 신뢰구간
        bootstrap_scores = []
        for _ in range(n_bootstrap):
            # 복원 추출
            indices = np.random.choice(n, size=n, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]

            boot_score = calculate_metric(y_true_boot, y_pred_boot)
            bootstrap_scores.append(boot_score)

        bootstrap_scores = np.array(bootstrap_scores)

        # 백분위수로 신뢰구간 계산
        alpha_lower = (self.alpha / 2) * 100
        alpha_upper = (1 - self.alpha / 2) * 100

        ci_lower = np.percentile(bootstrap_scores, alpha_lower)
        ci_upper = np.percentile(bootstrap_scores, alpha_upper)

        return {
            'statistic': f'{metric.upper()}',
            'value': original_score,
            'confidence_level': self.confidence_level,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'method': f"Bootstrap (n={n_bootstrap})",
            'sample_size': n,
            'bootstrap_std': np.std(bootstrap_scores),
            'bootstrap_mean': np.mean(bootstrap_scores)
        }

    def sharpe_ratio_ci(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> Dict:
        """
        샤프 비율의 신뢰구간 계산

        Args:
            returns: 수익률 배열
            risk_free_rate: 무위험 이자율 (연율)

        Returns:
            신뢰구간 정보 딕셔너리
        """
        n = len(returns)
        excess_returns = returns - risk_free_rate / 252  # 일일 무위험 이자율

        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns, ddof=1)

        if std_excess == 0:
            raise ValueError("수익률의 표준편차가 0입니다")

        sharpe_ratio = mean_excess / std_excess * np.sqrt(252)  # 연환산

        # Bootstrap 신뢰구간
        def calculate_sharpe(sample_returns):
            sample_excess = sample_returns - risk_free_rate / 252
            sample_mean = np.mean(sample_excess)
            sample_std = np.std(sample_excess, ddof=1)
            if sample_std == 0:
                return 0
            return sample_mean / sample_std * np.sqrt(252)

        ci_lower, ci_upper = self._bootstrap_ci(returns, calculate_sharpe)

        return {
            'statistic': 'Sharpe Ratio',
            'value': sharpe_ratio,
            'confidence_level': self.confidence_level,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'method': "Bootstrap",
            'sample_size': n,
            'risk_free_rate': risk_free_rate,
            'mean_excess_return': mean_excess,
            'excess_return_std': std_excess
        }

    def maximum_drawdown_ci(self, returns: np.ndarray) -> Dict:
        """
        최대 낙폭(MDD)의 신뢰구간 계산

        Args:
            returns: 수익률 배열

        Returns:
            신뢰구간 정보 딕셔너리
        """
        def calculate_mdd(sample_returns):
            cumulative = np.cumprod(1 + sample_returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdowns = (cumulative - running_max) / running_max
            return abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0

        mdd = calculate_mdd(returns)
        ci_lower, ci_upper = self._bootstrap_ci(returns, calculate_mdd)

        return {
            'statistic': 'Maximum Drawdown',
            'value': mdd,
            'confidence_level': self.confidence_level,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'method': "Bootstrap",
            'sample_size': len(returns)
        }

    def _bootstrap_ci(self, data: np.ndarray, statistic_func, n_bootstrap: int = 1000) -> Tuple[float, float]:
        """
        Bootstrap 신뢰구간 계산 (내부 메서드)

        Args:
            data: 데이터 배열
            statistic_func: 통계량 계산 함수
            n_bootstrap: Bootstrap 반복 횟수

        Returns:
            (신뢰구간 하한, 신뢰구간 상한)
        """
        bootstrap_stats = []

        for _ in range(n_bootstrap):
            # 복원 추출
            bootstrap_sample = resample(data, n_samples=len(data), replace=True)
            stat = statistic_func(bootstrap_sample)
            bootstrap_stats.append(stat)

        bootstrap_stats = np.array(bootstrap_stats)

        # 백분위수 계산
        alpha_lower = (self.alpha / 2) * 100
        alpha_upper = (1 - self.alpha / 2) * 100

        ci_lower = np.percentile(bootstrap_stats, alpha_lower)
        ci_upper = np.percentile(bootstrap_stats, alpha_upper)

        return ci_lower, ci_upper

    def batch_calculate(self, data_dict: Dict[str, np.ndarray],
                       statistics: List[str] = ["mean", "variance"]) -> Dict:
        """
        여러 데이터에 대해 여러 통계량의 신뢰구간을 일괄 계산

        Args:
            data_dict: 데이터명을 키로 하고 데이터 배열을 값으로 하는 딕셔너리
            statistics: 계산할 통계량 리스트

        Returns:
            전체 결과 딕셔너리
        """
        results = {}

        for data_name, data in data_dict.items():
            results[data_name] = {}

            for stat in statistics:
                try:
                    if stat == "mean":
                        result = self.mean_ci(data)
                    elif stat == "variance":
                        result = self.variance_ci(data)
                    elif stat == "sharpe_ratio":
                        result = self.sharpe_ratio_ci(data)
                    elif stat == "maximum_drawdown":
                        result = self.maximum_drawdown_ci(data)
                    else:
                        print(f"지원하지 않는 통계량입니다: {stat}")
                        continue

                    results[data_name][stat] = result

                except Exception as e:
                    print(f"오류 발생 ({data_name}, {stat}): {str(e)}")
                    results[data_name][stat] = {"error": str(e)}

        return results

def main():
    """테스트 및 예제 실행"""
    print("🔢 신뢰구간 계산기 테스트")
    print("=" * 50)

    # 테스트 데이터 생성
    np.random.seed(42)

    # 모델 성능 데이터 시뮬레이션
    mae_scores = np.random.gamma(2, 0.01, 50)  # MAE 점수들
    accuracy_data = np.random.binomial(1, 0.75, 100)  # 정확도 데이터 (75% 성공률)
    returns = np.random.normal(0.001, 0.02, 252)  # 일일 수익률 (1년)

    calc = ConfidenceIntervalCalculator(confidence_level=0.95)

    # 1. 평균의 신뢰구간
    print("\n1. MAE 점수 평균의 신뢰구간")
    print("-" * 40)
    mean_ci = calc.mean_ci(mae_scores, method="parametric")
    print(f"평균: {mean_ci['value']:.6f}")
    print(f"95% 신뢰구간: [{mean_ci['ci_lower']:.6f}, {mean_ci['ci_upper']:.6f}]")
    print(f"방법: {mean_ci['method']}")

    # 2. 비율의 신뢰구간
    print("\n2. 정확도의 신뢰구간")
    print("-" * 40)
    successes = np.sum(accuracy_data)
    prop_ci = calc.proportion_ci(successes, len(accuracy_data), method="wilson")
    print(f"정확도: {prop_ci['value']:.3f}")
    print(f"95% 신뢰구간: [{prop_ci['ci_lower']:.3f}, {prop_ci['ci_upper']:.3f}]")
    print(f"방법: {prop_ci['method']}")

    # 3. 샤프 비율의 신뢰구간
    print("\n3. 샤프 비율의 신뢰구간")
    print("-" * 40)
    sharpe_ci = calc.sharpe_ratio_ci(returns, risk_free_rate=0.02)
    print(f"샤프 비율: {sharpe_ci['value']:.3f}")
    print(f"95% 신뢰구간: [{sharpe_ci['ci_lower']:.3f}, {sharpe_ci['ci_upper']:.3f}]")

    # 4. 최대 낙폭의 신뢰구간
    print("\n4. 최대 낙폭의 신뢰구간")
    print("-" * 40)
    mdd_ci = calc.maximum_drawdown_ci(returns)
    print(f"MDD: {mdd_ci['value']:.3f}")
    print(f"95% 신뢰구간: [{mdd_ci['ci_lower']:.3f}, {mdd_ci['ci_upper']:.3f}]")

    # 5. 상관계수의 신뢰구간
    print("\n5. 상관계수의 신뢰구간")
    print("-" * 40)
    corr_ci = calc.correlation_ci(0.65, n=50)
    print(f"상관계수: {corr_ci['value']:.3f}")
    print(f"95% 신뢰구간: [{corr_ci['ci_lower']:.3f}, {corr_ci['ci_upper']:.3f}]")

if __name__ == "__main__":
    main()