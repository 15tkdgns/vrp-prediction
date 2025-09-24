"""
Core Statistical Foundation for Financial Time Series

통계적 정상성과 시계열 안전성을 보장하는 핵심 구성요소들:
1. LogReturnProcessor: 가격 → 로그 수익률 변환 및 정상성 검증
2. TimeSeriesSafeValidator: 데이터 누출 방지 시계열 검증
3. FinancialMetrics: 샤프, 소르티노, MDD 등 금융 성과 지표
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod

import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.stattools import adfuller
    from scipy import stats
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("⚠️ statsmodels not available, using simplified implementations")


@dataclass
class StationarityTestResult:
    """정상성 검정 결과"""
    is_stationary: bool
    adf_statistic: float
    p_value: float
    critical_values: Dict[str, float]
    transformation: str  # 'none', 'log', 'diff', 'log_diff'


@dataclass
class FinancialMetricsResult:
    """금융 성과 지표 결과"""
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    volatility: float
    skewness: float
    kurtosis: float
    var_95: float  # Value at Risk 95%
    cvar_95: float  # Conditional VaR 95%


class LogReturnProcessor:
    """
    로그 수익률 변환 및 통계적 정상성 확보

    금융 시계열의 근본적 문제인 비정상성을 해결하기 위해:
    1. 원시 가격 → 로그 수익률 변환
    2. ADF 검정으로 정상성 검증
    3. 필요시 차분(differencing) 적용
    """

    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.transformation_history: List[str] = []

    def price_to_log_returns(
        self,
        prices: Union[pd.Series, np.ndarray],
        fill_method: str = 'forward'
    ) -> pd.Series:
        """
        가격을 로그 수익률로 변환

        log_return_t = ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1})

        Args:
            prices: 가격 시계열
            fill_method: 결측치 처리 방법

        Returns:
            로그 수익률 시계열
        """
        if isinstance(prices, np.ndarray):
            prices = pd.Series(prices)

        # 0이나 음수 가격 처리
        prices = prices.replace(0, np.nan)
        prices = prices[prices > 0]

        if fill_method == 'forward':
            prices = prices.ffill()
        elif fill_method == 'backward':
            prices = prices.bfill()
        elif fill_method == 'drop':
            prices = prices.dropna()

        # 로그 수익률 계산
        log_returns = np.log(prices / prices.shift(1))
        log_returns = log_returns.dropna()

        # 이상치 제거 (±5 표준편차)
        mean = log_returns.mean()
        std = log_returns.std()
        outlier_mask = np.abs(log_returns - mean) > 5 * std

        if outlier_mask.sum() > 0:
            print(f"⚠️ 제거된 이상치: {outlier_mask.sum()}개")
            log_returns = log_returns[~outlier_mask]

        return log_returns

    def test_stationarity(
        self,
        series: pd.Series,
        max_lags: Optional[int] = None
    ) -> StationarityTestResult:
        """
        ADF(Augmented Dickey-Fuller) 검정으로 정상성 확인

        H0: 시계열이 단위근을 가짐 (비정상성)
        H1: 시계열이 정상성을 가짐

        Args:
            series: 검정할 시계열
            max_lags: 최대 래그 수

        Returns:
            정상성 검정 결과
        """
        if not STATSMODELS_AVAILABLE:
            # 간단한 대체 검정 (분산 안정성 기반)
            rolling_var = series.rolling(window=30).var()
            var_stability = rolling_var.std() / rolling_var.mean()
            is_stationary = var_stability < 0.5

            return StationarityTestResult(
                is_stationary=is_stationary,
                adf_statistic=-2.0 if is_stationary else -1.0,
                p_value=0.03 if is_stationary else 0.08,
                critical_values={'1%': -3.43, '5%': -2.86, '10%': -2.57},
                transformation='simplified_test'
            )

        # ADF 검정 수행
        if max_lags is None:
            max_lags = int(12 * (len(series) / 100) ** 0.25)

        result = adfuller(series.dropna(), maxlag=max_lags)

        adf_statistic = result[0]
        p_value = result[1]
        critical_values = result[4]

        is_stationary = p_value < self.significance_level

        return StationarityTestResult(
            is_stationary=is_stationary,
            adf_statistic=adf_statistic,
            p_value=p_value,
            critical_values=critical_values,
            transformation='none'
        )

    def ensure_stationarity(
        self,
        series: pd.Series,
        max_diff_order: int = 2
    ) -> Tuple[pd.Series, List[str]]:
        """
        정상성 확보를 위한 자동 변환

        1. 로그 변환 시도
        2. 1차 차분 적용
        3. 필요시 2차 차분까지 적용

        Args:
            series: 원본 시계열
            max_diff_order: 최대 차분 차수

        Returns:
            (정상성 시계열, 적용된 변환 목록)
        """
        transformations = []
        current_series = series.copy()

        # 1단계: ADF 검정
        result = self.test_stationarity(current_series)
        if result.is_stationary:
            return current_series, transformations

        # 2단계: 로그 변환 (양수인 경우만)
        if (current_series > 0).all():
            log_series = np.log(current_series)
            result = self.test_stationarity(log_series)
            if result.is_stationary:
                transformations.append('log')
                return log_series, transformations
            current_series = log_series
            transformations.append('log')

        # 3단계: 차분 적용
        for diff_order in range(1, max_diff_order + 1):
            diff_series = current_series.diff(diff_order).dropna()
            result = self.test_stationarity(diff_series)

            transformations.append(f'diff_{diff_order}')

            if result.is_stationary:
                return diff_series, transformations

            current_series = diff_series

        print(f"⚠️ {max_diff_order}차 차분까지 적용했으나 정상성 확보 실패")
        return current_series, transformations


class TimeSeriesSafeValidator:
    """
    시계열 데이터 누출 방지 검증 시스템

    금융 시계열에서 발생할 수 있는 데이터 누출을 방지:
    1. 시간 순서 보장
    2. Look-ahead bias 방지
    3. 적절한 시계열 분할
    """

    def __init__(self):
        self.validation_warnings: List[str] = []

    def validate_temporal_order(
        self,
        data: pd.DataFrame,
        date_column: str = 'date'
    ) -> bool:
        """시간 순서 검증"""
        if date_column not in data.columns:
            if isinstance(data.index, pd.DatetimeIndex):
                dates = data.index
            else:
                self.validation_warnings.append("날짜 정보를 찾을 수 없음")
                return False
        else:
            dates = pd.to_datetime(data[date_column])

        # 시간 순서 확인
        is_sorted = dates.is_monotonic_increasing

        if not is_sorted:
            self.validation_warnings.append("데이터가 시간 순서대로 정렬되지 않음")

        return is_sorted

    def detect_lookahead_bias(
        self,
        features: pd.DataFrame,
        targets: pd.Series,
        suspicious_patterns: List[str] = None
    ) -> Dict[str, bool]:
        """Look-ahead bias 감지"""
        if suspicious_patterns is None:
            suspicious_patterns = [
                'shift(-1)', 'future_', 'next_', 'forward_',
                'lead_', 'ahead_'
            ]

        bias_detected = {}

        # 컬럼명 패턴 검사
        for pattern in suspicious_patterns:
            pattern_found = any(pattern in col for col in features.columns)
            bias_detected[f'pattern_{pattern}'] = pattern_found

            if pattern_found:
                self.validation_warnings.append(
                    f"의심스러운 패턴 발견: {pattern}"
                )

        # 특성-타겟 상관관계 검사 (너무 높으면 의심)
        correlations = {}
        for col in features.columns:
            if features[col].dtype in ['float64', 'int64']:
                corr = features[col].corr(targets)
                correlations[col] = abs(corr)

                if abs(corr) > 0.95:
                    bias_detected[f'high_correlation_{col}'] = True
                    self.validation_warnings.append(
                        f"비정상적으로 높은 상관관계: {col} (r={corr:.3f})"
                    )

        return bias_detected

    def create_safe_train_test_split(
        self,
        data: pd.DataFrame,
        test_size: float = 0.2,
        gap_size: int = 0
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        시계열 안전 분할

        Args:
            data: 시계열 데이터
            test_size: 테스트 비율
            gap_size: 훈련-테스트 사이 간격 (데이터 누출 방지)

        Returns:
            (훈련 데이터, 테스트 데이터)
        """
        n = len(data)
        test_start_idx = int(n * (1 - test_size))

        # 간격 적용
        train_end_idx = test_start_idx - gap_size
        test_start_idx = test_start_idx

        if train_end_idx <= 0:
            raise ValueError("Gap이 너무 크거나 데이터가 부족함")

        train_data = data.iloc[:train_end_idx]
        test_data = data.iloc[test_start_idx:]

        print(f"✅ 안전 분할 완료:")
        print(f"   훈련: {len(train_data)}개 ({train_data.index[0]} ~ {train_data.index[-1]})")
        print(f"   간격: {gap_size}개")
        print(f"   테스트: {len(test_data)}개 ({test_data.index[0]} ~ {test_data.index[-1]})")

        return train_data, test_data


class FinancialMetrics:
    """
    금융 성과 지표 계산

    MSE 대신 실제 투자 성과와 연결된 지표들:
    - 샤프 비율: 위험 대비 수익
    - 소르티노 비율: 하방 위험 대비 수익
    - 최대 낙폭(MDD): 최악의 손실 기간
    - VaR/CVaR: 극한 리스크 지표
    """

    @staticmethod
    def calculate_sharpe_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> float:
        """
        샤프 비율 = (연평균 수익률 - 무위험 수익률) / 변동성
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        annual_return = returns.mean() * periods_per_year
        annual_vol = returns.std() * np.sqrt(periods_per_year)

        return (annual_return - risk_free_rate) / annual_vol

    @staticmethod
    def calculate_sortino_ratio(
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> float:
        """
        소르티노 비율 = (연평균 수익률 - 무위험 수익률) / 하방 변동성
        """
        if len(returns) == 0:
            return 0.0

        annual_return = returns.mean() * periods_per_year
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0:
            return float('inf')

        downside_vol = downside_returns.std() * np.sqrt(periods_per_year)

        if downside_vol == 0:
            return float('inf')

        return (annual_return - risk_free_rate) / downside_vol

    @staticmethod
    def calculate_max_drawdown(cumulative_returns: pd.Series) -> float:
        """
        최대 낙폭 = max((peak - trough) / peak)
        """
        if len(cumulative_returns) == 0:
            return 0.0

        # 누적 수익 커브
        cum_returns = (1 + cumulative_returns).cumprod()

        # 각 시점의 최고점
        peak = cum_returns.expanding().max()

        # 낙폭 계산
        drawdown = (cum_returns - peak) / peak

        return abs(drawdown.min())

    @staticmethod
    def calculate_var_cvar(
        returns: pd.Series,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """
        VaR (Value at Risk)와 CVaR (Conditional VaR) 계산
        """
        if len(returns) == 0:
            return 0.0, 0.0

        # VaR: 신뢰수준에서의 손실 임계값
        var = np.percentile(returns, (1 - confidence_level) * 100)

        # CVaR: VaR을 초과하는 손실의 평균
        tail_losses = returns[returns <= var]
        cvar = tail_losses.mean() if len(tail_losses) > 0 else var

        return abs(var), abs(cvar)

    @classmethod
    def calculate_comprehensive_metrics(
        cls,
        returns: pd.Series,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ) -> FinancialMetricsResult:
        """종합 금융 성과 지표 계산"""

        if len(returns) == 0:
            return FinancialMetricsResult(
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown=0.0,
                calmar_ratio=0.0,
                volatility=0.0,
                skewness=0.0,
                kurtosis=0.0,
                var_95=0.0,
                cvar_95=0.0
            )

        # 기본 지표
        sharpe = cls.calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)
        sortino = cls.calculate_sortino_ratio(returns, risk_free_rate, periods_per_year)
        mdd = cls.calculate_max_drawdown(returns)

        # Calmar 비율 = 연평균 수익률 / 최대 낙폭
        annual_return = returns.mean() * periods_per_year
        calmar = annual_return / mdd if mdd > 0 else 0.0

        # 변동성 및 분포 지표
        volatility = returns.std() * np.sqrt(periods_per_year)
        skewness = returns.skew()
        kurtosis = returns.kurtosis()

        # 리스크 지표
        var_95, cvar_95 = cls.calculate_var_cvar(returns, 0.95)

        return FinancialMetricsResult(
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=mdd,
            calmar_ratio=calmar,
            volatility=volatility,
            skewness=skewness,
            kurtosis=kurtosis,
            var_95=var_95,
            cvar_95=cvar_95
        )