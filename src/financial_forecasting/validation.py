"""
Financial Validation and Backtesting Framework

시계열 안전 검증 및 금융 성과 평가:
1. WalkForwardValidator: 시계열 안전 Walk-Forward 검증
2. FinancialBacktester: 실제 거래 전략 백테스팅
3. RiskMetricsCalculator: 포트폴리오 리스크 지표 계산
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

from .core import FinancialMetrics


@dataclass
class ValidationResult:
    """검증 결과 데이터"""
    train_score: float
    test_score: float
    predictions: np.ndarray
    actuals: np.ndarray
    train_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex
    model_params: Dict


@dataclass
class BacktestResult:
    """백테스팅 결과"""
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    trade_count: int
    portfolio_value: pd.Series
    trades: pd.DataFrame


@dataclass
class RiskMetrics:
    """리스크 지표"""
    var_95: float          # Value at Risk 95%
    cvar_95: float         # Conditional VaR 95%
    var_99: float          # Value at Risk 99%
    cvar_99: float         # Conditional VaR 99%
    beta: float            # 시장 베타
    tracking_error: float  # 추적 오차
    information_ratio: float  # 정보 비율
    downside_deviation: float  # 하방 편차


class WalkForwardValidator:
    """
    Walk-Forward Validation for Time Series

    시계열 데이터 누출을 방지하는 시간 순차적 검증:
    - 고정 또는 확장 훈련 윈도우
    - 적절한 재훈련 주기
    - Purged 검증 (간격 설정)
    - 결과 안정성 평가
    """

    def __init__(
        self,
        initial_train_size: int = 252,  # 1년
        test_size: int = 21,            # 1개월
        step_size: int = 21,            # 재훈련 주기
        purge_size: int = 5,            # 간격 일수
        expanding_window: bool = True,   # 확장 윈도우 여부
        min_train_size: int = 126       # 최소 훈련 크기
    ):
        self.initial_train_size = initial_train_size
        self.test_size = test_size
        self.step_size = step_size
        self.purge_size = purge_size
        self.expanding_window = expanding_window
        self.min_train_size = min_train_size

    def validate(
        self,
        model: object,
        X: pd.DataFrame,
        y: pd.Series,
        fit_params: Optional[Dict] = None
    ) -> List[ValidationResult]:
        """
        Walk-Forward 검증 수행

        Args:
            model: 학습할 모델 (fit, predict 메서드 필요)
            X: 특성 데이터
            y: 타겟 데이터
            fit_params: 모델 학습 파라미터

        Returns:
            검증 결과 리스트
        """
        if fit_params is None:
            fit_params = {}

        results = []
        n_samples = len(X)

        # 시작 인덱스 계산
        start_idx = self.initial_train_size + self.purge_size
        end_idx = n_samples - self.test_size

        print(f"🔄 Walk-Forward 검증 시작:")
        print(f"   데이터 크기: {n_samples}")
        print(f"   초기 훈련: {self.initial_train_size}")
        print(f"   테스트 크기: {self.test_size}")
        print(f"   재훈련 주기: {self.step_size}")
        print(f"   간격: {self.purge_size}")

        fold = 0
        for test_start in range(start_idx, end_idx, self.step_size):
            fold += 1
            test_end = min(test_start + self.test_size, n_samples)

            # 훈련 데이터 인덱스 계산
            if self.expanding_window:
                train_start = 0
                train_end = test_start - self.purge_size
            else:
                train_start = max(0, test_start - self.initial_train_size - self.purge_size)
                train_end = test_start - self.purge_size

            # 최소 훈련 크기 확인
            if train_end - train_start < self.min_train_size:
                print(f"⚠️ Fold {fold}: 훈련 데이터 부족 ({train_end - train_start} < {self.min_train_size})")
                continue

            # 데이터 분할
            X_train = X.iloc[train_start:train_end]
            y_train = y.iloc[train_start:train_end]
            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]

            try:
                # 모델 학습
                model.fit(X_train, y_train, **fit_params)

                # 예측
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)

                # 점수 계산
                train_score = self._calculate_score(y_train, y_pred_train)
                test_score = self._calculate_score(y_test, y_pred_test)

                # 결과 저장
                result = ValidationResult(
                    train_score=train_score,
                    test_score=test_score,
                    predictions=y_pred_test,
                    actuals=y_test.values,
                    train_dates=X_train.index,
                    test_dates=X_test.index,
                    model_params=getattr(model, 'get_params', lambda: {})()
                )

                results.append(result)

                print(f"✅ Fold {fold}: 훈련={train_score:.4f}, 테스트={test_score:.4f} "
                      f"({X_train.index[0].strftime('%Y-%m-%d')} ~ {X_test.index[-1].strftime('%Y-%m-%d')})")

            except Exception as e:
                print(f"❌ Fold {fold} 실패: {e}")
                continue

        print(f"🎯 Walk-Forward 검증 완료: {len(results)}개 폴드")
        return results

    def _calculate_score(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        """점수 계산 (기본: 상관계수)"""
        try:
            correlation = np.corrcoef(y_true.values, y_pred)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        except:
            return 0.0

    def summarize_results(self, results: List[ValidationResult]) -> Dict:
        """검증 결과 요약"""
        if not results:
            return {}

        train_scores = [r.train_score for r in results]
        test_scores = [r.test_score for r in results]

        summary = {
            'n_folds': len(results),
            'train_score_mean': np.mean(train_scores),
            'train_score_std': np.std(train_scores),
            'test_score_mean': np.mean(test_scores),
            'test_score_std': np.std(test_scores),
            'overfitting_ratio': np.mean(train_scores) / np.mean(test_scores) if np.mean(test_scores) != 0 else float('inf'),
            'score_stability': 1 - (np.std(test_scores) / abs(np.mean(test_scores))) if np.mean(test_scores) != 0 else 0.0
        }

        return summary


class FinancialBacktester:
    """
    Financial Strategy Backtesting

    실제 거래 전략의 성과를 검증:
    - 신호 기반 매매 전략
    - 거래 비용 고려
    - 포지션 사이징
    - 리스크 관리
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.001,  # 0.1%
        position_size: float = 1.0,       # 전체 자본 사용 비율
        risk_free_rate: float = 0.02      # 무위험 수익률
    ):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.position_size = position_size
        self.risk_free_rate = risk_free_rate

    def backtest_strategy(
        self,
        prices: pd.Series,
        signals: pd.Series,
        strategy_name: str = "Strategy"
    ) -> BacktestResult:
        """
        전략 백테스팅 수행

        Args:
            prices: 가격 시계열
            signals: 매매 신호 (-1: 매도, 0: 보유, 1: 매수)
            strategy_name: 전략 이름

        Returns:
            백테스팅 결과
        """
        if len(prices) != len(signals):
            raise ValueError("가격과 신호 데이터 길이가 다릅니다")

        # 일자별 수익률 계산
        returns = prices.pct_change().fillna(0)

        # 포트폴리오 시뮬레이션
        portfolio_value = [self.initial_capital]
        positions = []
        trades = []

        current_position = 0  # 0: 현금, 1: 주식 보유
        shares_held = 0
        cash = self.initial_capital

        for i, (date, price, signal, daily_return) in enumerate(
            zip(prices.index, prices.values, signals.values, returns.values)
        ):
            trade_occurred = False

            # 매매 신호 처리
            if signal == 1 and current_position == 0:  # 매수
                shares_to_buy = (cash * self.position_size) / price
                transaction_fee = shares_to_buy * price * self.transaction_cost

                if cash >= shares_to_buy * price + transaction_fee:
                    shares_held = shares_to_buy
                    cash -= (shares_to_buy * price + transaction_fee)
                    current_position = 1
                    trade_occurred = True

                    trades.append({
                        'date': date,
                        'action': 'BUY',
                        'price': price,
                        'shares': shares_held,
                        'value': shares_held * price,
                        'cash_after': cash
                    })

            elif signal == -1 and current_position == 1:  # 매도
                transaction_fee = shares_held * price * self.transaction_cost
                cash += (shares_held * price - transaction_fee)

                trades.append({
                    'date': date,
                    'action': 'SELL',
                    'price': price,
                    'shares': shares_held,
                    'value': shares_held * price,
                    'cash_after': cash
                })

                shares_held = 0
                current_position = 0
                trade_occurred = True

            # 포트폴리오 가치 계산
            total_value = cash + (shares_held * price if shares_held > 0 else 0)
            portfolio_value.append(total_value)

            positions.append({
                'date': date,
                'position': current_position,
                'shares': shares_held,
                'cash': cash,
                'portfolio_value': total_value
            })

        # 결과 계산
        portfolio_series = pd.Series(portfolio_value[1:], index=prices.index)
        portfolio_returns = portfolio_series.pct_change().fillna(0)

        # 성과 지표 계산
        total_return = (portfolio_series.iloc[-1] / self.initial_capital) - 1
        annual_return = (1 + total_return) ** (252 / len(portfolio_series)) - 1

        # 금융 지표 계산
        metrics = FinancialMetrics.calculate_comprehensive_metrics(
            portfolio_returns, self.risk_free_rate
        )

        # 거래 통계
        trades_df = pd.DataFrame(trades)
        if len(trades_df) > 0:
            buy_trades = trades_df[trades_df['action'] == 'BUY']
            sell_trades = trades_df[trades_df['action'] == 'SELL']

            if len(buy_trades) > 0 and len(sell_trades) > 0:
                # 매매 쌍 매칭하여 수익/손실 계산
                trade_returns = []
                for i in range(min(len(buy_trades), len(sell_trades))):
                    buy_price = buy_trades.iloc[i]['price']
                    sell_price = sell_trades.iloc[i]['price']
                    trade_return = (sell_price - buy_price) / buy_price
                    trade_returns.append(trade_return)

                win_rate = len([r for r in trade_returns if r > 0]) / len(trade_returns)

                # Profit Factor (총 이익 / 총 손실)
                profits = [r for r in trade_returns if r > 0]
                losses = [abs(r) for r in trade_returns if r < 0]
                profit_factor = sum(profits) / sum(losses) if losses else float('inf')
            else:
                win_rate = 0.0
                profit_factor = 0.0
        else:
            win_rate = 0.0
            profit_factor = 0.0

        result = BacktestResult(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=metrics.sharpe_ratio,
            sortino_ratio=metrics.sortino_ratio,
            max_drawdown=metrics.max_drawdown,
            calmar_ratio=metrics.calmar_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            trade_count=len(trades_df),
            portfolio_value=portfolio_series,
            trades=trades_df
        )

        print(f"📊 {strategy_name} 백테스팅 완료:")
        print(f"   총 수익률: {total_return:.2%}")
        print(f"   연 수익률: {annual_return:.2%}")
        print(f"   샤프 비율: {metrics.sharpe_ratio:.3f}")
        print(f"   최대 낙폭: {metrics.max_drawdown:.2%}")
        print(f"   거래 횟수: {len(trades_df)}")
        print(f"   승률: {win_rate:.2%}")

        return result


class RiskMetricsCalculator:
    """
    Portfolio Risk Metrics Calculator

    포트폴리오 리스크 지표 계산:
    - VaR/CVaR (Value at Risk)
    - 베타 및 추적 오차
    - 정보 비율
    - 하방 위험 지표
    """

    def __init__(self, benchmark_returns: Optional[pd.Series] = None):
        self.benchmark_returns = benchmark_returns

    def calculate_var_cvar(
        self,
        returns: pd.Series,
        confidence_levels: List[float] = [0.95, 0.99]
    ) -> Dict[str, Dict[str, float]]:
        """VaR 및 CVaR 계산"""
        var_cvar_results = {}

        for confidence in confidence_levels:
            # Historical VaR
            var = np.percentile(returns, (1 - confidence) * 100)

            # Conditional VaR (Expected Shortfall)
            tail_losses = returns[returns <= var]
            cvar = tail_losses.mean() if len(tail_losses) > 0 else var

            var_cvar_results[f'{confidence:.0%}'] = {
                'VaR': abs(var),
                'CVaR': abs(cvar)
            }

        return var_cvar_results

    def calculate_beta_metrics(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """베타 및 관련 지표 계산"""
        if benchmark_returns is None:
            benchmark_returns = self.benchmark_returns

        if benchmark_returns is None:
            return {
                'beta': 1.0,
                'alpha': 0.0,
                'tracking_error': 0.0,
                'information_ratio': 0.0,
                'correlation': 0.0
            }

        # 공통 기간으로 정렬
        aligned_returns = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        if len(aligned_returns) < 30:  # 최소 30개 관측치 필요
            return {
                'beta': 1.0,
                'alpha': 0.0,
                'tracking_error': 0.0,
                'information_ratio': 0.0,
                'correlation': 0.0
            }

        port_ret = aligned_returns.iloc[:, 0]
        bench_ret = aligned_returns.iloc[:, 1]

        # 베타 계산 (CAPM)
        covariance = np.cov(port_ret, bench_ret)[0, 1]
        benchmark_variance = np.var(bench_ret)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0

        # 알파 계산 (포트폴리오 수익률 - 베타 * 벤치마크 수익률)
        alpha = port_ret.mean() - beta * bench_ret.mean()

        # 추적 오차 (Tracking Error)
        active_returns = port_ret - bench_ret
        tracking_error = active_returns.std() * np.sqrt(252)  # 연환산

        # 정보 비율 (Information Ratio)
        information_ratio = (alpha * 252) / tracking_error if tracking_error > 0 else 0.0

        # 상관계수
        correlation = np.corrcoef(port_ret, bench_ret)[0, 1]

        return {
            'beta': beta,
            'alpha': alpha * 252,  # 연환산
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'correlation': correlation
        }

    def calculate_downside_metrics(
        self,
        returns: pd.Series,
        target_return: float = 0.0
    ) -> Dict[str, float]:
        """하방 위험 지표 계산"""
        # 하방 편차 (Downside Deviation)
        downside_returns = returns[returns < target_return]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.0

        # 하방 확률 (Downside Probability)
        downside_probability = len(downside_returns) / len(returns)

        # 최대 연속 손실 기간
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak

        # 최대 낙폭 기간
        max_dd_end = drawdown.idxmin()
        max_dd_start = cumulative_returns[:max_dd_end].idxmax()
        max_dd_duration = (max_dd_end - max_dd_start).days if hasattr(max_dd_end, 'days') else 0

        return {
            'downside_deviation': downside_deviation,
            'downside_probability': downside_probability,
            'max_drawdown_duration': max_dd_duration,
            'pain_index': drawdown.mean() * -1  # 평균 낙폭 (Pain Index)
        }

    def calculate_comprehensive_risk_metrics(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None
    ) -> RiskMetrics:
        """종합 리스크 지표 계산"""
        # VaR/CVaR
        var_cvar = self.calculate_var_cvar(returns)

        # 베타 지표
        beta_metrics = self.calculate_beta_metrics(returns, benchmark_returns)

        # 하방 위험 지표
        downside_metrics = self.calculate_downside_metrics(returns)

        return RiskMetrics(
            var_95=var_cvar['95%']['VaR'],
            cvar_95=var_cvar['95%']['CVaR'],
            var_99=var_cvar['99%']['VaR'],
            cvar_99=var_cvar['99%']['CVaR'],
            beta=beta_metrics['beta'],
            tracking_error=beta_metrics['tracking_error'],
            information_ratio=beta_metrics['information_ratio'],
            downside_deviation=downside_metrics['downside_deviation']
        )

    def generate_risk_report(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict:
        """리스크 리포트 생성"""
        comprehensive_metrics = self.calculate_comprehensive_risk_metrics(returns, benchmark_returns)
        var_cvar = self.calculate_var_cvar(returns)
        beta_metrics = self.calculate_beta_metrics(returns, benchmark_returns)
        downside_metrics = self.calculate_downside_metrics(returns)

        report = {
            'portfolio_statistics': {
                'total_observations': len(returns),
                'mean_return': returns.mean() * 252,
                'volatility': returns.std() * np.sqrt(252),
                'skewness': returns.skew(),
                'kurtosis': returns.kurtosis()
            },
            'value_at_risk': var_cvar,
            'market_risk': beta_metrics,
            'downside_risk': downside_metrics,
            'comprehensive_metrics': {
                'var_95': comprehensive_metrics.var_95,
                'cvar_95': comprehensive_metrics.cvar_95,
                'beta': comprehensive_metrics.beta,
                'tracking_error': comprehensive_metrics.tracking_error,
                'information_ratio': comprehensive_metrics.information_ratio,
                'downside_deviation': comprehensive_metrics.downside_deviation
            }
        }

        return report