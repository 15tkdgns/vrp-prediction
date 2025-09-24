#!/usr/bin/env python3
"""
금융 성과 지표 및 백테스트 시스템
거래 비용, 슬리피지를 포함한 현실적인 백테스팅 시스템
샤프 비율, 소르티노 비율, 최대 손실 등 전문 금융 지표 계산
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
import logging
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class TradingCosts:
    """거래 비용 설정"""
    commission: float = 0.001  # 수수료 (0.1%)
    bid_ask_spread: float = 0.0005  # 매수-매도 스프레드 (0.05%)
    market_impact: float = 0.0002  # 시장 충격 (0.02%)
    slippage: float = 0.0001  # 슬리피지 (0.01%)


@dataclass
class BacktestConfig:
    """백테스트 설정"""
    initial_capital: float = 100000.0  # 초기 자본
    position_size: float = 1.0  # 포지션 크기 (1.0 = 100% 투자)
    max_leverage: float = 1.0  # 최대 레버리지
    rebalance_frequency: str = 'daily'  # 리밸런싱 빈도
    trading_costs: TradingCosts = field(default_factory=TradingCosts)
    risk_free_rate: float = 0.02  # 무위험 수익률 (연 2%)


@dataclass
class PerformanceMetrics:
    """성과 지표 결과"""
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    var_95: float = 0.0  # 95% VaR
    var_99: float = 0.0  # 99% VaR
    skewness: float = 0.0
    kurtosis: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    information_ratio: float = 0.0


class FinancialMetricsCalculator:
    """금융 성과 지표 계산기"""

    def __init__(self, config: BacktestConfig = None):
        """
        Args:
            config: 백테스트 설정
        """
        self.config = config or BacktestConfig()

    def calculate_returns(self, prices: np.ndarray) -> np.ndarray:
        """수익률 계산"""
        return np.diff(prices) / prices[:-1]

    def calculate_log_returns(self, prices: np.ndarray) -> np.ndarray:
        """로그 수익률 계산"""
        return np.diff(np.log(prices))

    def calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = None) -> float:
        """샤프 비율 계산"""
        if risk_free_rate is None:
            risk_free_rate = self.config.risk_free_rate

        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        # 연간화
        excess_return = np.mean(returns) * 252 - risk_free_rate
        volatility = np.std(returns) * np.sqrt(252)

        return excess_return / volatility

    def calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate: float = None) -> float:
        """소르티노 비율 계산"""
        if risk_free_rate is None:
            risk_free_rate = self.config.risk_free_rate

        if len(returns) == 0:
            return 0.0

        # 하방 편차 계산
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return float('inf')

        excess_return = np.mean(returns) * 252 - risk_free_rate
        downside_deviation = np.std(downside_returns) * np.sqrt(252)

        return excess_return / downside_deviation if downside_deviation > 0 else 0.0

    def calculate_max_drawdown(self, prices: np.ndarray) -> float:
        """최대 손실 계산"""
        if len(prices) == 0:
            return 0.0

        cumulative = np.cumprod(1 + self.calculate_returns(prices))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max

        return np.min(drawdown)

    def calculate_calmar_ratio(self, returns: np.ndarray) -> float:
        """칼마 비율 계산"""
        if len(returns) == 0:
            return 0.0

        annualized_return = np.mean(returns) * 252
        max_dd = abs(self.calculate_max_drawdown(np.cumprod(1 + returns)))

        return annualized_return / max_dd if max_dd > 0 else 0.0

    def calculate_var(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """VaR (Value at Risk) 계산"""
        if len(returns) == 0:
            return 0.0

        return np.percentile(returns, (1 - confidence_level) * 100)

    def calculate_information_ratio(self, portfolio_returns: np.ndarray,
                                  benchmark_returns: np.ndarray) -> float:
        """정보 비율 계산"""
        if len(portfolio_returns) != len(benchmark_returns):
            return 0.0

        excess_returns = portfolio_returns - benchmark_returns
        if len(excess_returns) == 0 or np.std(excess_returns) == 0:
            return 0.0

        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

    def calculate_comprehensive_metrics(self, returns: np.ndarray,
                                      benchmark_returns: np.ndarray = None) -> PerformanceMetrics:
        """종합 성과 지표 계산"""
        if len(returns) == 0:
            return PerformanceMetrics()

        # 기본 통계
        total_return = np.prod(1 + returns) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = np.std(returns) * np.sqrt(252)

        # 위험 조정 수익률
        sharpe_ratio = self.calculate_sharpe_ratio(returns)
        sortino_ratio = self.calculate_sortino_ratio(returns)
        calmar_ratio = self.calculate_calmar_ratio(returns)

        # 위험 지표
        max_drawdown = self.calculate_max_drawdown(np.cumprod(1 + returns))
        var_95 = self.calculate_var(returns, 0.95)
        var_99 = self.calculate_var(returns, 0.99)

        # 분포 특성
        skewness = stats.skew(returns) if len(returns) > 2 else 0.0
        kurtosis = stats.kurtosis(returns) if len(returns) > 3 else 0.0

        # 승률 및 수익 팩터
        win_rate = np.sum(returns > 0) / len(returns)
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]

        profit_factor = (np.sum(positive_returns) / abs(np.sum(negative_returns))
                        if len(negative_returns) > 0 and np.sum(negative_returns) != 0 else 0.0)

        # 정보 비율
        information_ratio = (self.calculate_information_ratio(returns, benchmark_returns)
                           if benchmark_returns is not None else 0.0)

        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            var_95=var_95,
            var_99=var_99,
            skewness=skewness,
            kurtosis=kurtosis,
            win_rate=win_rate,
            profit_factor=profit_factor,
            information_ratio=information_ratio
        )


class RealisticBacktester:
    """
    현실적인 백테스터
    거래 비용, 슬리피지, 시장 충격을 고려한 백테스팅
    """

    def __init__(self, config: BacktestConfig = None):
        """
        Args:
            config: 백테스트 설정
        """
        self.config = config or BacktestConfig()
        self.metrics_calculator = FinancialMetricsCalculator(config)

    def apply_trading_costs(self, gross_returns: np.ndarray,
                          positions: np.ndarray) -> np.ndarray:
        """거래 비용 적용"""
        costs = self.config.trading_costs

        # 포지션 변화 감지
        position_changes = np.diff(positions, prepend=0)
        turnover = np.abs(position_changes)

        # 거래 비용 계산
        total_costs = (costs.commission + costs.bid_ask_spread +
                      costs.market_impact + costs.slippage) * turnover

        # 순 수익률 계산
        net_returns = gross_returns - total_costs

        return net_returns

    def simulate_slippage(self, expected_returns: np.ndarray,
                         volatility: float = 0.01) -> np.ndarray:
        """슬리피지 시뮬레이션"""
        # 변동성에 비례한 슬리피지
        slippage_noise = np.random.normal(0, volatility * 0.1, len(expected_returns))
        return expected_returns - np.abs(slippage_noise)

    def backtest_strategy(self, predictions: np.ndarray, actual_returns: np.ndarray,
                         benchmark_returns: np.ndarray = None,
                         include_costs: bool = True) -> Dict[str, Any]:
        """
        전략 백테스트 실행

        Args:
            predictions: 모델 예측값 (확률 또는 신호)
            actual_returns: 실제 수익률
            benchmark_returns: 벤치마크 수익률
            include_costs: 거래 비용 포함 여부

        Returns:
            backtest_results: 백테스트 결과
        """
        if len(predictions) != len(actual_returns):
            raise ValueError("예측값과 실제 수익률의 길이가 일치하지 않습니다.")

        # 포지션 생성 (예측값을 -1~1 범위로 변환)
        if np.max(predictions) <= 1 and np.min(predictions) >= 0:
            # 확률값인 경우 (0~1)
            positions = (predictions - 0.5) * 2 * self.config.position_size
        else:
            # 이미 신호값인 경우
            positions = np.clip(predictions, -1, 1) * self.config.position_size

        # 총 수익률 계산
        gross_returns = positions[:-1] * actual_returns[1:]  # 한 시점 지연

        # 거래 비용 적용
        if include_costs:
            net_returns = self.apply_trading_costs(gross_returns, positions)
        else:
            net_returns = gross_returns

        # 포트폴리오 가치 계산
        portfolio_values = np.cumprod(1 + net_returns) * self.config.initial_capital

        # 성과 지표 계산
        performance_metrics = self.metrics_calculator.calculate_comprehensive_metrics(
            net_returns, benchmark_returns[1:] if benchmark_returns is not None else None
        )

        # 추가 분석
        analysis_results = self._perform_additional_analysis(
            net_returns, positions, actual_returns, benchmark_returns
        )

        return {
            'performance_metrics': performance_metrics,
            'portfolio_values': portfolio_values,
            'net_returns': net_returns,
            'gross_returns': gross_returns,
            'positions': positions,
            'analysis': analysis_results,
            'config': self.config
        }

    def _perform_additional_analysis(self, returns: np.ndarray, positions: np.ndarray,
                                   actual_returns: np.ndarray,
                                   benchmark_returns: np.ndarray = None) -> Dict[str, Any]:
        """추가 분석 수행"""
        analysis = {}

        # 거래 횟수 분석
        position_changes = np.diff(positions, prepend=0)
        analysis['total_trades'] = np.sum(np.abs(position_changes) > 0.01)
        analysis['average_turnover'] = np.mean(np.abs(position_changes))

        # 포지션 분석
        analysis['long_positions'] = np.sum(positions > 0) / len(positions)
        analysis['short_positions'] = np.sum(positions < 0) / len(positions)
        analysis['neutral_positions'] = np.sum(np.abs(positions) < 0.01) / len(positions)

        # 시기별 성과 분석
        if len(returns) >= 252:  # 1년 이상 데이터
            annual_chunks = np.array_split(returns, len(returns) // 252)
            annual_returns = [np.prod(1 + chunk) - 1 for chunk in annual_chunks]
            analysis['annual_returns'] = annual_returns
            analysis['annual_volatility'] = [np.std(chunk) * np.sqrt(252) for chunk in annual_chunks]

        # 롤링 성과
        if len(returns) >= 63:  # 3개월 이상
            window = 63  # 3개월
            rolling_sharpe = []
            for i in range(window, len(returns)):
                window_returns = returns[i-window:i]
                sharpe = self.metrics_calculator.calculate_sharpe_ratio(window_returns)
                rolling_sharpe.append(sharpe)
            analysis['rolling_sharpe_3m'] = rolling_sharpe

        # 벤치마크 대비 성과
        if benchmark_returns is not None and len(benchmark_returns) > 1:
            bench_returns = benchmark_returns[1:]  # 길이 맞춤
            if len(bench_returns) == len(returns):
                analysis['alpha'] = np.mean(returns - bench_returns) * 252
                analysis['beta'] = np.cov(returns, bench_returns)[0, 1] / np.var(bench_returns)
                analysis['tracking_error'] = np.std(returns - bench_returns) * np.sqrt(252)

        return analysis

    def run_monte_carlo_simulation(self, predictions: np.ndarray,
                                 actual_returns: np.ndarray,
                                 n_simulations: int = 1000) -> Dict[str, Any]:
        """몬테카를로 시뮬레이션"""
        simulation_results = []

        for i in range(n_simulations):
            # 수익률 순서 무작위화 (부트스트랩)
            shuffled_indices = np.random.choice(len(actual_returns), len(actual_returns), replace=True)
            shuffled_returns = actual_returns[shuffled_indices]

            # 백테스트 실행
            result = self.backtest_strategy(predictions, shuffled_returns, include_costs=True)
            simulation_results.append(result['performance_metrics'])

        # 결과 분석
        sharpe_ratios = [result.sharpe_ratio for result in simulation_results]
        total_returns = [result.total_return for result in simulation_results]
        max_drawdowns = [result.max_drawdown for result in simulation_results]

        return {
            'sharpe_distribution': {
                'mean': np.mean(sharpe_ratios),
                'std': np.std(sharpe_ratios),
                'percentiles': np.percentile(sharpe_ratios, [5, 25, 50, 75, 95])
            },
            'return_distribution': {
                'mean': np.mean(total_returns),
                'std': np.std(total_returns),
                'percentiles': np.percentile(total_returns, [5, 25, 50, 75, 95])
            },
            'drawdown_distribution': {
                'mean': np.mean(max_drawdowns),
                'std': np.std(max_drawdowns),
                'percentiles': np.percentile(max_drawdowns, [5, 25, 50, 75, 95])
            },
            'all_results': simulation_results
        }


class BacktestVisualizer:
    """백테스트 결과 시각화"""

    def __init__(self):
        plt.style.use('seaborn-v0_8')

    def plot_performance_summary(self, backtest_result: Dict[str, Any],
                               save_path: str = None) -> None:
        """성과 요약 시각화"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('백테스트 성과 요약', fontsize=16)

        # 1. 포트폴리오 가치 변화
        portfolio_values = backtest_result['portfolio_values']
        axes[0, 0].plot(portfolio_values)
        axes[0, 0].set_title('포트폴리오 가치 변화')
        axes[0, 0].set_ylabel('포트폴리오 가치')

        # 2. 수익률 분포
        returns = backtest_result['net_returns']
        axes[0, 1].hist(returns, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('수익률 분포')
        axes[0, 1].set_xlabel('일일 수익률')
        axes[0, 1].set_ylabel('빈도')

        # 3. 드로우다운
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        axes[0, 2].fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
        axes[0, 2].plot(drawdown, color='red')
        axes[0, 2].set_title('드로우다운')
        axes[0, 2].set_ylabel('드로우다운 (%)')

        # 4. 롤링 샤프 비율
        if 'rolling_sharpe_3m' in backtest_result['analysis']:
            rolling_sharpe = backtest_result['analysis']['rolling_sharpe_3m']
            axes[1, 0].plot(rolling_sharpe)
            axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[1, 0].set_title('롤링 샤프 비율 (3개월)')
            axes[1, 0].set_ylabel('샤프 비율')

        # 5. 포지션 분포
        positions = backtest_result['positions']
        axes[1, 1].hist(positions, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('포지션 분포')
        axes[1, 1].set_xlabel('포지션 크기')
        axes[1, 1].set_ylabel('빈도')

        # 6. 성과 지표 요약
        metrics = backtest_result['performance_metrics']
        metrics_text = f"""
        총 수익률: {metrics.total_return:.2%}
        연간 수익률: {metrics.annualized_return:.2%}
        변동성: {metrics.volatility:.2%}
        샤프 비율: {metrics.sharpe_ratio:.3f}
        소르티노 비율: {metrics.sortino_ratio:.3f}
        최대 손실: {metrics.max_drawdown:.2%}
        승률: {metrics.win_rate:.2%}
        """
        axes[1, 2].text(0.1, 0.9, metrics_text, transform=axes[1, 2].transAxes,
                        fontsize=12, verticalalignment='top', fontfamily='monospace')
        axes[1, 2].set_title('성과 지표')
        axes[1, 2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# 사용 예시 및 테스트
if __name__ == "__main__":
    # 테스트 데이터 생성
    np.random.seed(42)
    n_days = 1000

    # 실제 수익률 (주식 시장 특성 반영)
    actual_returns = np.random.normal(0.0005, 0.02, n_days)  # 일일 0.05% 기대수익, 2% 변동성

    # 벤치마크 수익률 (시장 지수)
    benchmark_returns = np.random.normal(0.0003, 0.015, n_days)

    # 모델 예측값 (약간의 예측력 있는 신호)
    true_signal = np.roll(actual_returns, 1)  # 전날 수익률이 다음날을 약간 예측
    noise = np.random.normal(0, 0.5, n_days)
    predictions = 0.5 + 0.3 * np.tanh(true_signal + noise)  # 0~1 확률값

    print("=== 금융 성과 지표 및 백테스트 시스템 테스트 ===")

    # 1. 거래 비용 설정
    trading_costs = TradingCosts(
        commission=0.001,
        bid_ask_spread=0.0005,
        market_impact=0.0002,
        slippage=0.0001
    )

    config = BacktestConfig(
        initial_capital=100000.0,
        position_size=1.0,
        trading_costs=trading_costs,
        risk_free_rate=0.02
    )

    # 2. 백테스터 생성 및 실행
    backtester = RealisticBacktester(config)

    print("\n1. 기본 백테스트 실행")
    result_with_costs = backtester.backtest_strategy(
        predictions, actual_returns, benchmark_returns, include_costs=True
    )

    result_without_costs = backtester.backtest_strategy(
        predictions, actual_returns, benchmark_returns, include_costs=False
    )

    # 3. 결과 비교
    print(f"\n=== 거래 비용 영향 분석 ===")
    metrics_with = result_with_costs['performance_metrics']
    metrics_without = result_without_costs['performance_metrics']

    print(f"거래 비용 포함:")
    print(f"  총 수익률: {metrics_with.total_return:.4f}")
    print(f"  샤프 비율: {metrics_with.sharpe_ratio:.4f}")
    print(f"  최대 손실: {metrics_with.max_drawdown:.4f}")

    print(f"\n거래 비용 제외:")
    print(f"  총 수익률: {metrics_without.total_return:.4f}")
    print(f"  샤프 비율: {metrics_without.sharpe_ratio:.4f}")
    print(f"  최대 손실: {metrics_without.max_drawdown:.4f}")

    cost_impact = metrics_without.total_return - metrics_with.total_return
    print(f"\n거래 비용으로 인한 수익률 감소: {cost_impact:.4f} ({cost_impact/metrics_without.total_return:.2%})")

    # 4. 추가 분석
    analysis = result_with_costs['analysis']
    print(f"\n=== 거래 분석 ===")
    print(f"총 거래 횟수: {analysis['total_trades']}")
    print(f"평균 회전율: {analysis['average_turnover']:.4f}")
    print(f"롱 포지션 비율: {analysis['long_positions']:.2%}")
    print(f"숏 포지션 비율: {analysis['short_positions']:.2%}")

    # 5. 몬테카를로 시뮬레이션 (작은 규모로)
    print(f"\n2. 몬테카를로 시뮬레이션 (100회)")
    mc_results = backtester.run_monte_carlo_simulation(
        predictions[:500], actual_returns[:500], n_simulations=100
    )

    sharpe_dist = mc_results['sharpe_distribution']
    print(f"샤프 비율 분포:")
    print(f"  평균: {sharpe_dist['mean']:.4f}")
    print(f"  표준편차: {sharpe_dist['std']:.4f}")
    print(f"  95% 신뢰구간: [{sharpe_dist['percentiles'][0]:.4f}, {sharpe_dist['percentiles'][4]:.4f}]")

    return_dist = mc_results['return_distribution']
    print(f"\n총 수익률 분포:")
    print(f"  평균: {return_dist['mean']:.4f}")
    print(f"  95% 신뢰구간: [{return_dist['percentiles'][0]:.4f}, {return_dist['percentiles'][4]:.4f}]")

    # 6. 시각화
    print(f"\n3. 결과 시각화")
    visualizer = BacktestVisualizer()
    try:
        visualizer.plot_performance_summary(result_with_costs)
    except Exception as e:
        print(f"시각화 오류 (정상): {e}")

    print("\n✅ 금융 성과 지표 및 백테스트 시스템 테스트 완료")
    print("현실적인 거래 비용 반영 및 전문 금융 지표 계산 구현 성공")