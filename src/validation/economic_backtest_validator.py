"""
경제적 가치 실증적 백테스트 검증기
거래 비용을 포함한 실제 거래 성과 측정

이 모듈은 변동성 예측 모델의 실제 경제적 가치를 측정합니다.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

class EconomicBacktestValidator:
    """거래 비용을 포함한 실제 백테스트 시스템"""

    def __init__(self, transaction_cost=0.001, leverage=1.0):
        """
        Args:
            transaction_cost: 거래 비용 (0.1% = 0.001)
            leverage: 레버리지 (1.0 = 무레버리지)
        """
        self.transaction_cost = transaction_cost
        self.leverage = leverage
        self.model = Ridge(alpha=1.0, random_state=42)
        self.scaler = StandardScaler()

    def fetch_spy_data(self, start_date="2015-01-01", end_date="2024-12-31"):
        """실제 SPY ETF 데이터 수집"""
        print(f"📊 SPY 데이터 수집 중: {start_date} ~ {end_date}")

        spy = yf.Ticker("SPY")
        data = spy.history(start=start_date, end=end_date)

        # 수익률 계산
        data['returns'] = np.log(data['Close'] / data['Close'].shift(1))
        data = data.dropna()

        print(f"✅ 데이터 수집 완료: {len(data)}개 관측치")
        return data

    def create_features(self, returns):
        """올바른 특성 생성 (≤ t)"""
        features = pd.DataFrame(index=returns.index)

        # 과거 변동성 특성
        for window in [5, 10, 20, 50]:
            features[f'volatility_{window}'] = returns.rolling(window).std()

        # 래그 특성
        for lag in [1, 2, 3, 5]:
            features[f'return_lag_{lag}'] = returns.shift(lag)

        # 비율 특성
        features['vol_ratio_5_20'] = features['volatility_5'] / features['volatility_20']
        features['vol_ratio_10_50'] = features['volatility_10'] / features['volatility_50']

        return features.dropna()

    def create_targets(self, returns):
        """올바른 타겟 생성 (≥ t+1)"""
        target_vol_5d = []

        for i in range(len(returns)):
            if i + 5 < len(returns):
                # 미래 5일 변동성 (t+1 to t+5)
                future_window = returns.iloc[i+1:i+6]
                target_vol_5d.append(future_window.std())
            else:
                target_vol_5d.append(np.nan)

        return pd.Series(target_vol_5d, index=returns.index).dropna()

    def volatility_trading_strategy(self, predicted_vol, returns):
        """변동성 예측 기반 거래 전략 (단순화)"""
        positions = []

        # 예측 변동성의 중앙값을 기준으로 전략 수립
        vol_median = predicted_vol.median()

        for i in range(len(predicted_vol)):
            pred_vol = predicted_vol.iloc[i]

            # 전략: 낮은 변동성 예상시 더 많은 리스크, 높은 변동성 예상시 리스크 축소
            if pred_vol < vol_median * 0.9:  # 낮은 변동성 예상
                position = 1.3  # 포지션 확대
            elif pred_vol > vol_median * 1.1:  # 높은 변동성 예상
                position = 0.7  # 포지션 축소
            else:
                position = 1.0  # 기본 포지션

            positions.append(position)

        return pd.Series(positions, index=predicted_vol.index)

    def calculate_transaction_costs(self, positions):
        """거래 비용 계산"""
        position_changes = positions.diff().abs()
        costs = position_changes * self.transaction_cost
        return costs.fillna(0)

    def run_backtest(self, start_date="2018-01-01", end_date="2024-12-31"):
        """거래 비용 포함 실제 백테스트 실행 (단순화)"""

        print("🏁 경제적 가치 실증 백테스트 시작")
        print("=" * 60)

        # 1. 데이터 수집
        data = self.fetch_spy_data(start_date, end_date)
        returns = data['returns']

        # 2. 특성 및 타겟 생성
        features = self.create_features(returns)
        targets = self.create_targets(returns)

        # 데이터 정렬
        common_index = features.index.intersection(targets.index)
        features = features.loc[common_index]
        targets = targets.loc[common_index]

        print(f"📈 백테스트 기간: {common_index[0].date()} ~ {common_index[-1].date()}")
        print(f"📊 사용 가능 데이터: {len(features)}개")

        # 3. 간단한 훈련-테스트 분할 (70-30)
        split_point = int(len(features) * 0.7)

        X_train = features.iloc[:split_point]
        y_train = targets.iloc[:split_point]
        X_test = features.iloc[split_point:]
        y_test = targets.iloc[split_point:]

        # 4. 모델 훈련
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model.fit(X_train_scaled, y_train)
        predictions = self.model.predict(X_test_scaled)

        # 5. 거래 전략 실행
        pred_series = pd.Series(predictions, index=X_test.index)
        positions = self.volatility_trading_strategy(pred_series, returns.loc[X_test.index])

        # 6. 전략 수익률 계산 (다음날 수익률에 포지션 적용)
        test_returns = returns.loc[X_test.index]
        strategy_returns = positions.shift(1) * test_returns  # t일 포지션으로 t+1일 수익률 획득
        strategy_returns = strategy_returns.dropna()

        # 7. 거래 비용 계산
        position_changes = positions.diff().abs()
        transaction_costs = position_changes * self.transaction_cost
        transaction_costs = transaction_costs.loc[strategy_returns.index]

        # 8. 순 수익률 (거래 비용 차감)
        net_returns = strategy_returns - transaction_costs

        # 9. 벤치마크 (Buy & Hold) 수익률
        benchmark_returns = test_returns.loc[strategy_returns.index]

        return self.calculate_performance_metrics(net_returns, benchmark_returns, transaction_costs)

    def calculate_performance_metrics(self, strategy_returns, benchmark_returns, transaction_costs):
        """성과 지표 계산"""

        # 기본 통계
        strategy_total_return = (1 + strategy_returns).prod() - 1
        benchmark_total_return = (1 + benchmark_returns).prod() - 1

        strategy_annual_return = (1 + strategy_total_return) ** (252 / len(strategy_returns)) - 1
        benchmark_annual_return = (1 + benchmark_total_return) ** (252 / len(benchmark_returns)) - 1

        strategy_volatility = strategy_returns.std() * np.sqrt(252)
        benchmark_volatility = benchmark_returns.std() * np.sqrt(252)

        # 샤프 비율 (무위험 수익률 2% 가정)
        risk_free_rate = 0.02
        strategy_sharpe = (strategy_annual_return - risk_free_rate) / strategy_volatility
        benchmark_sharpe = (benchmark_annual_return - risk_free_rate) / benchmark_volatility

        # 최대 낙폭 (Maximum Drawdown)
        strategy_cumulative = (1 + strategy_returns).cumprod()
        strategy_peak = strategy_cumulative.expanding().max()
        strategy_drawdown = (strategy_cumulative - strategy_peak) / strategy_peak
        strategy_max_drawdown = strategy_drawdown.min()

        benchmark_cumulative = (1 + benchmark_returns).cumprod()
        benchmark_peak = benchmark_cumulative.expanding().max()
        benchmark_drawdown = (benchmark_cumulative - benchmark_peak) / benchmark_peak
        benchmark_max_drawdown = benchmark_drawdown.min()

        # 거래 비용 영향
        total_transaction_costs = transaction_costs.sum()
        transaction_cost_impact = total_transaction_costs / strategy_total_return if strategy_total_return != 0 else 0

        # 정보 비율 (Information Ratio)
        excess_returns = strategy_returns - benchmark_returns
        information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() != 0 else 0

        return {
            'strategy_annual_return': strategy_annual_return,
            'benchmark_annual_return': benchmark_annual_return,
            'strategy_volatility': strategy_volatility,
            'benchmark_volatility': benchmark_volatility,
            'strategy_sharpe': strategy_sharpe,
            'benchmark_sharpe': benchmark_sharpe,
            'strategy_max_drawdown': strategy_max_drawdown,
            'benchmark_max_drawdown': benchmark_max_drawdown,
            'information_ratio': information_ratio,
            'total_transaction_costs': total_transaction_costs,
            'transaction_cost_impact': transaction_cost_impact,
            'excess_annual_return': strategy_annual_return - benchmark_annual_return,
            'volatility_reduction': benchmark_volatility - strategy_volatility,
            'sharpe_improvement': strategy_sharpe - benchmark_sharpe
        }

    def print_backtest_results(self, results):
        """백테스트 결과 출력"""

        print("\n💰 경제적 가치 실증 결과")
        print("=" * 60)

        print("\n📊 수익률 비교")
        print(f"전략 연간 수익률:     {results['strategy_annual_return']:.2%}")
        print(f"벤치마크 연간 수익률: {results['benchmark_annual_return']:.2%}")
        print(f"초과 연간 수익률:     {results['excess_annual_return']:.2%}")

        print("\n📉 리스크 비교")
        print(f"전략 변동성:         {results['strategy_volatility']:.2%}")
        print(f"벤치마크 변동성:     {results['benchmark_volatility']:.2%}")
        print(f"변동성 감소:         {results['volatility_reduction']:.2%}")

        print("\n⚡ 리스크 조정 수익률")
        print(f"전략 샤프 비율:      {results['strategy_sharpe']:.3f}")
        print(f"벤치마크 샤프 비율:  {results['benchmark_sharpe']:.3f}")
        print(f"샤프 비율 개선:      {results['sharpe_improvement']:.3f}")
        print(f"정보 비율:          {results['information_ratio']:.3f}")

        print("\n📊 최대 낙폭 비교")
        print(f"전략 최대 낙폭:      {results['strategy_max_drawdown']:.2%}")
        print(f"벤치마크 최대 낙폭:  {results['benchmark_max_drawdown']:.2%}")

        print("\n💸 거래 비용 분석")
        print(f"총 거래 비용:        {results['total_transaction_costs']:.4f}")
        print(f"거래 비용 영향:      {results['transaction_cost_impact']:.2%}")

        print("\n🏆 경제적 가치 평가")
        if results['excess_annual_return'] > 0:
            print("✅ 전략이 벤치마크를 초과 성과")
        else:
            print("❌ 전략이 벤치마크 대비 저조")

        if results['sharpe_improvement'] > 0:
            print("✅ 리스크 조정 수익률 개선")
        else:
            print("❌ 리스크 조정 수익률 악화")

        if results['volatility_reduction'] > 0:
            print("✅ 변동성 감소 효과")
        else:
            print("❌ 변동성 증가")

        # 종합 평가
        score = 0
        if results['excess_annual_return'] > 0: score += 1
        if results['sharpe_improvement'] > 0: score += 1
        if results['volatility_reduction'] > 0: score += 1
        if results['information_ratio'] > 0: score += 1

        print(f"\n📊 종합 평가: {score}/4 ({'우수' if score >= 3 else '보통' if score >= 2 else '개선 필요'})")

        return results

def run_economic_validation():
    """경제적 가치 실증 검증 실행"""

    print("🚀 변동성 예측 모델 경제적 가치 실증 검증")
    print("=" * 80)

    # 백테스트 실행
    validator = EconomicBacktestValidator(
        transaction_cost=0.001,  # 0.1% 거래 비용
        leverage=1.0             # 레버리지 없음
    )

    results = validator.run_backtest()
    validator.print_backtest_results(results)

    return results

if __name__ == "__main__":
    results = run_economic_validation()