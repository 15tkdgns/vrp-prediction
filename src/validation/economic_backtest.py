#!/usr/bin/env python3
"""
경제적 백테스트 - 변동성 예측 모델의 실제 거래 가치 검증
다양한 변동성 기반 거래 전략의 수익성 분석
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import warnings
import os
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# 최종 모델 로드
import sys
sys.path.append('/root/workspace/src/models')
from final_volatility_model import VolatilityPredictor

class VolatilityTradingBacktest:
    """변동성 예측 기반 거래 전략 백테스트"""

    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.transaction_cost = 0.001  # 0.1% 거래 비용

    def load_backtest_data(self, start_date='2020-01-01', end_date='2024-12-31'):
        """백테스트용 데이터 로드"""
        print(f"📊 백테스트 데이터 로드: {start_date} ~ {end_date}")

        # SPY 데이터
        spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
        spy['returns'] = spy['Close'].pct_change()

        # VIX 데이터
        vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)
        vix_close = vix['Close'].reindex(spy.index, method='ffill')
        spy['vix'] = vix_close

        # 실제 변동성 계산 (5일 후)
        spy['actual_vol_5d'] = spy['returns'].rolling(5).std().shift(-5)

        spy = spy.dropna()
        print(f"✅ 데이터 로드 완료: {len(spy)} 관측치")
        return spy

    def generate_volatility_predictions(self, data):
        """변동성 예측 생성"""
        print("🔮 변동성 예측 생성 중...")

        # 모델 로드
        predictor = VolatilityPredictor()
        predictor.load_model()

        # 예측 생성 (슬라이딩 윈도우)
        predictions = []
        dates = []

        # 6개월씩 슬라이딩하면서 예측
        for i in range(180, len(data) - 5):  # 180일 이후부터 시작
            # 현재 시점까지의 데이터로 예측
            current_data = data.iloc[:i+1].copy()

            try:
                # 예측 수행
                pred_result = predictor.predict_volatility(current_data)
                if pred_result is not None and len(pred_result) > 0:
                    last_pred = pred_result.iloc[-1]['predicted_volatility']
                    predictions.append(last_pred)
                    dates.append(data.index[i])
                else:
                    predictions.append(np.nan)
                    dates.append(data.index[i])
            except:
                predictions.append(np.nan)
                dates.append(data.index[i])

        # 결과 DataFrame 생성
        pred_df = pd.DataFrame({
            'date': dates,
            'predicted_vol': predictions
        }).set_index('date')

        pred_df = pred_df.dropna()
        print(f"✅ 예측 생성 완료: {len(pred_df)} 개")
        return pred_df

    def strategy_volatility_timing(self, data, predictions):
        """전략 1: 변동성 타이밍 - 낮은 변동성 예측 시 매수"""
        print("📈 전략 1: 변동성 타이밍 백테스트 중...")

        # 데이터 결합
        combined = pd.merge(data, predictions, left_index=True, right_index=True, how='inner')

        if len(combined) < 50:
            print("⚠️ 데이터 부족")
            return None

        # 변동성 임계값 설정 (하위 30%, 상위 30%)
        vol_low_threshold = combined['predicted_vol'].quantile(0.3)
        vol_high_threshold = combined['predicted_vol'].quantile(0.7)

        # 거래 신호 생성
        combined['signal'] = 0
        combined.loc[combined['predicted_vol'] <= vol_low_threshold, 'signal'] = 1  # 매수
        combined.loc[combined['predicted_vol'] >= vol_high_threshold, 'signal'] = -1  # 매도/공매도

        # 포지션 계산
        combined['position'] = combined['signal'].shift(1).fillna(0)

        # 수익률 계산
        combined['strategy_returns'] = combined['position'] * combined['returns']

        # 거래 비용 적용
        position_changes = combined['position'].diff().abs()
        transaction_costs = position_changes * self.transaction_cost
        combined['strategy_returns'] -= transaction_costs

        # 누적 수익률
        combined['cumulative_returns'] = (1 + combined['returns']).cumprod()
        combined['cumulative_strategy'] = (1 + combined['strategy_returns']).cumprod()

        return combined

    def strategy_volatility_scaling(self, data, predictions):
        """전략 2: 변동성 스케일링 - 변동성에 반비례하여 포지션 크기 조절"""
        print("📈 전략 2: 변동성 스케일링 백테스트 중...")

        # 데이터 결합
        combined = pd.merge(data, predictions, left_index=True, right_index=True, how='inner')

        if len(combined) < 50:
            print("⚠️ 데이터 부족")
            return None

        # 기준 변동성 설정
        base_vol = combined['predicted_vol'].median()

        # 포지션 크기 = 기준변동성 / 예측변동성 (변동성이 낮을수록 큰 포지션)
        combined['position_size'] = np.clip(base_vol / combined['predicted_vol'], 0.2, 2.0)

        # 수익률 계산
        combined['strategy_returns'] = combined['position_size'].shift(1) * combined['returns']

        # 거래 비용 적용
        position_changes = combined['position_size'].diff().abs()
        transaction_costs = position_changes * self.transaction_cost * 0.5  # 크기 조절은 거래 비용 적게
        combined['strategy_returns'] -= transaction_costs

        # 누적 수익률
        combined['cumulative_returns'] = (1 + combined['returns']).cumprod()
        combined['cumulative_strategy'] = (1 + combined['strategy_returns']).cumprod()

        return combined

    def strategy_vix_arbitrage(self, data, predictions):
        """전략 3: VIX 차익거래 - 예측 변동성과 VIX 차이 활용"""
        print("📈 전략 3: VIX 차익거래 백테스트 중...")

        # 데이터 결합
        combined = pd.merge(data, predictions, left_index=True, right_index=True, how='inner')

        if len(combined) < 50:
            print("⚠️ 데이터 부족")
            return None

        # 연율화된 예측 변동성
        combined['predicted_vol_annual'] = combined['predicted_vol'] * np.sqrt(252) * 100

        # VIX와 예측 변동성의 차이
        combined['vol_spread'] = combined['vix'] - combined['predicted_vol_annual']

        # 거래 신호 (VIX가 예측보다 높으면 VIX 매도, 낮으면 VIX 매수)
        combined['signal'] = 0
        spread_threshold = combined['vol_spread'].std()

        combined.loc[combined['vol_spread'] > spread_threshold, 'signal'] = -1  # VIX 매도
        combined.loc[combined['vol_spread'] < -spread_threshold, 'signal'] = 1  # VIX 매수

        # VIX 수익률 (변화율)
        combined['vix_returns'] = combined['vix'].pct_change()

        # 포지션 계산
        combined['position'] = combined['signal'].shift(1).fillna(0)

        # 수익률 계산 (VIX 포지션)
        combined['strategy_returns'] = combined['position'] * combined['vix_returns']

        # 거래 비용 적용
        position_changes = combined['position'].diff().abs()
        transaction_costs = position_changes * self.transaction_cost * 2  # VIX 거래는 비용 높음
        combined['strategy_returns'] -= transaction_costs

        # 누적 수익률
        combined['cumulative_vix'] = (1 + combined['vix_returns']).cumprod()
        combined['cumulative_strategy'] = (1 + combined['strategy_returns']).cumprod()

        return combined

    def calculate_performance_metrics(self, returns_series, name="Strategy"):
        """성과 지표 계산"""
        if len(returns_series) == 0 or returns_series.isna().all():
            return {}

        # 기본 통계
        total_return = (1 + returns_series).prod() - 1
        annual_return = (1 + returns_series).mean() * 252
        annual_vol = returns_series.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0

        # 최대 낙폭
        cumulative = (1 + returns_series).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min()

        # 승률
        win_rate = (returns_series > 0).mean()

        return {
            'name': name,
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(returns_series)
        }

    def run_comprehensive_backtest(self):
        """종합 백테스트 실행"""
        print("🚀 종합 경제적 백테스트 시작")
        print("=" * 60)

        # 1. 데이터 로드
        data = self.load_backtest_data()

        # 2. 예측 생성
        predictions = self.generate_volatility_predictions(data)

        if len(predictions) < 50:
            print("❌ 예측 데이터 부족으로 백테스트 중단")
            return None

        # 3. 각 전략 백테스트
        results = {}

        # Buy & Hold 벤치마크
        benchmark_data = pd.merge(data, predictions, left_index=True, right_index=True, how='inner')
        benchmark_metrics = self.calculate_performance_metrics(
            benchmark_data['returns'], "Buy & Hold (SPY)"
        )
        results['benchmark'] = {
            'data': benchmark_data,
            'metrics': benchmark_metrics
        }

        # 전략 1: 변동성 타이밍
        strategy1_data = self.strategy_volatility_timing(data, predictions)
        if strategy1_data is not None:
            strategy1_metrics = self.calculate_performance_metrics(
                strategy1_data['strategy_returns'], "Volatility Timing"
            )
            results['volatility_timing'] = {
                'data': strategy1_data,
                'metrics': strategy1_metrics
            }

        # 전략 2: 변동성 스케일링
        strategy2_data = self.strategy_volatility_scaling(data, predictions)
        if strategy2_data is not None:
            strategy2_metrics = self.calculate_performance_metrics(
                strategy2_data['strategy_returns'], "Volatility Scaling"
            )
            results['volatility_scaling'] = {
                'data': strategy2_data,
                'metrics': strategy2_metrics
            }

        # 전략 3: VIX 차익거래
        strategy3_data = self.strategy_vix_arbitrage(data, predictions)
        if strategy3_data is not None:
            strategy3_metrics = self.calculate_performance_metrics(
                strategy3_data['strategy_returns'], "VIX Arbitrage"
            )
            results['vix_arbitrage'] = {
                'data': strategy3_data,
                'metrics': strategy3_metrics
            }

        return results

    def create_backtest_visualizations(self, results):
        """백테스트 결과 시각화"""
        print("📊 백테스트 결과 시각화 중...")

        # 성과 비교 차트
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1. 누적 수익률 비교
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for i, (strategy_name, result) in enumerate(results.items()):
            data = result['data']
            if 'cumulative_strategy' in data.columns:
                ax1.plot(data.index, data['cumulative_strategy'] * 100,
                        label=result['metrics']['name'], color=colors[i], linewidth=2)
            elif 'cumulative_returns' in data.columns:
                ax1.plot(data.index, data['cumulative_returns'] * 100,
                        label=result['metrics']['name'], color=colors[i], linewidth=2)

        ax1.set_title('Cumulative Returns Comparison', fontweight='bold')
        ax1.set_ylabel('Cumulative Return (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 연간 수익률 비교
        strategy_names = [result['metrics']['name'] for result in results.values()]
        annual_returns = [result['metrics']['annual_return'] * 100 for result in results.values()]

        bars = ax2.bar(strategy_names, annual_returns, color=colors[:len(strategy_names)], alpha=0.7)
        ax2.set_title('Annual Returns Comparison', fontweight='bold')
        ax2.set_ylabel('Annual Return (%)')
        ax2.tick_params(axis='x', rotation=45)

        # 값 표시
        for bar, ret in zip(bars, annual_returns):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{ret:.1f}%', ha='center', va='bottom')

        # 3. 샤프 비율 비교
        sharpe_ratios = [result['metrics']['sharpe_ratio'] for result in results.values()]

        bars = ax3.bar(strategy_names, sharpe_ratios, color=colors[:len(strategy_names)], alpha=0.7)
        ax3.set_title('Sharpe Ratio Comparison', fontweight='bold')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.tick_params(axis='x', rotation=45)

        # 값 표시
        for bar, sharpe in zip(bars, sharpe_ratios):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{sharpe:.2f}', ha='center', va='bottom')

        # 4. 최대 낙폭 비교
        max_drawdowns = [result['metrics']['max_drawdown'] * 100 for result in results.values()]

        bars = ax4.bar(strategy_names, max_drawdowns, color=colors[:len(strategy_names)], alpha=0.7)
        ax4.set_title('Maximum Drawdown Comparison', fontweight='bold')
        ax4.set_ylabel('Maximum Drawdown (%)')
        ax4.tick_params(axis='x', rotation=45)

        # 값 표시
        for bar, dd in zip(bars, max_drawdowns):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height - 1,
                    f'{dd:.1f}%', ha='center', va='top')

        plt.tight_layout()

        # 저장
        os.makedirs('figures', exist_ok=True)
        plt.savefig('figures/economic_backtest_results.png', dpi=300, bbox_inches='tight')
        print("✅ 저장: figures/economic_backtest_results.png")
        plt.close()

def main():
    """메인 백테스트 함수"""
    print("💰 경제적 백테스트 - 변동성 예측 모델의 실제 가치 검증")
    print("=" * 60)

    # 백테스트 실행
    backtest = VolatilityTradingBacktest(initial_capital=100000)
    results = backtest.run_comprehensive_backtest()

    if results is None:
        print("❌ 백테스트 실패")
        return

    # 결과 요약 출력
    print("\n📊 백테스트 결과 요약")
    print("=" * 60)

    metrics_df = pd.DataFrame([result['metrics'] for result in results.values()])
    print(metrics_df.round(4))

    # 시각화
    backtest.create_backtest_visualizations(results)

    # 결과 저장
    os.makedirs('results', exist_ok=True)

    # 메트릭만 저장 (JSON 직렬화 가능)
    metrics_only = {k: v['metrics'] for k, v in results.items()}

    backtest_summary = {
        'backtest_date': datetime.now().isoformat(),
        'initial_capital': backtest.initial_capital,
        'transaction_cost': backtest.transaction_cost,
        'strategies_tested': len(results),
        'performance_metrics': metrics_only
    }

    with open('results/economic_backtest_results.json', 'w') as f:
        json.dump(backtest_summary, f, indent=2, default=str)

    print(f"\n💾 백테스트 결과 저장: results/economic_backtest_results.json")

    # 최고 성과 전략 찾기
    best_strategy = max(metrics_only.items(), key=lambda x: x[1]['sharpe_ratio'])
    print(f"\n🏆 최고 성과 전략: {best_strategy[0]}")
    print(f"   샤프 비율: {best_strategy[1]['sharpe_ratio']:.4f}")
    print(f"   연간 수익률: {best_strategy[1]['annual_return']*100:.1f}%")

    print("=" * 60)

    return results

if __name__ == "__main__":
    results = main()