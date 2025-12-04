#!/usr/bin/env python3
"""
간단한 경제적 백테스트
변동성 예측 모델의 경제적 가치를 빠르게 검증
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

def load_backtest_data():
    """백테스트용 데이터 로드"""
    print("📊 백테스트 데이터 로드 중...")

    # SPY 데이터 (2020-2024)
    spy = yf.download('SPY', start='2020-01-01', end='2024-12-31', progress=False)
    spy['returns'] = spy['Close'].pct_change()

    # VIX 데이터
    vix = yf.download('^VIX', start='2020-01-01', end='2024-12-31', progress=False)
    spy['vix'] = vix['Close'].reindex(spy.index, method='ffill')

    # 실제 변동성 계산
    spy['actual_vol_5d'] = spy['returns'].rolling(5).std().shift(-5)
    spy['realized_vol'] = spy['returns'].rolling(20).std()

    spy = spy.dropna()
    print(f"✅ 데이터 로드 완료: {len(spy)} 관측치")
    return spy

def create_simple_volatility_proxy(data):
    """간단한 변동성 예측 프록시 생성"""
    print("🔮 변동성 예측 프록시 생성 중...")

    # VIX 기반 변동성 예측 (모델의 핵심 특성)
    data['predicted_vol'] = (
        0.4 * data['vix'] / 100 / np.sqrt(252) +  # VIX 기반
        0.3 * data['realized_vol'] +               # 과거 변동성
        0.2 * data['returns'].rolling(5).std() +  # 단기 변동성
        0.1 * data['returns'].rolling(10).std()   # 중기 변동성
    )

    print(f"✅ 예측 프록시 생성 완료")
    return data

def strategy_volatility_timing(data):
    """전략 1: 변동성 타이밍"""
    print("📈 변동성 타이밍 전략 백테스트 중...")

    # 변동성 임계값
    vol_low = data['predicted_vol'].quantile(0.3)
    vol_high = data['predicted_vol'].quantile(0.7)

    # 신호 생성
    data['signal'] = 0
    data.loc[data['predicted_vol'] <= vol_low, 'signal'] = 1   # 낮은 변동성 -> 매수
    data.loc[data['predicted_vol'] >= vol_high, 'signal'] = -0.5  # 높은 변동성 -> 축소

    # 포지션 및 수익률
    data['position'] = data['signal'].shift(1).fillna(0)
    data['strategy_returns'] = data['position'] * data['returns']

    # 거래비용 (0.1%)
    transaction_cost = 0.001
    position_changes = data['position'].diff().abs()
    data['strategy_returns'] -= position_changes * transaction_cost

    return data

def strategy_volatility_scaling(data):
    """전략 2: 변동성 스케일링"""
    print("📈 변동성 스케일링 전략 백테스트 중...")

    # 기준 변동성
    base_vol = data['predicted_vol'].median()

    # 포지션 크기 = 기준변동성 / 예측변동성
    data['position_size'] = np.clip(base_vol / data['predicted_vol'], 0.3, 1.5)

    # 수익률 계산
    data['strategy_returns'] = data['position_size'].shift(1) * data['returns']

    # 거래비용 (크기 조절은 적게)
    transaction_cost = 0.0005
    position_changes = data['position_size'].diff().abs()
    data['strategy_returns'] -= position_changes * transaction_cost

    return data

def strategy_vix_mean_reversion(data):
    """전략 3: VIX 평균회귀"""
    print("📈 VIX 평균회귀 전략 백테스트 중...")

    # VIX Z-score
    vix_ma = data['vix'].rolling(50).mean()
    vix_std = data['vix'].rolling(50).std()
    data['vix_zscore'] = (data['vix'] - vix_ma) / vix_std

    # 신호: VIX가 높으면 매수 (공포 시 매수)
    data['signal'] = np.where(data['vix_zscore'] > 1, 1,   # VIX 높음 -> 매수
                     np.where(data['vix_zscore'] < -1, -0.5, 0))  # VIX 낮음 -> 축소

    # 포지션 및 수익률
    data['position'] = data['signal'].shift(1).fillna(0)
    data['strategy_returns'] = data['position'] * data['returns']

    # 거래비용
    transaction_cost = 0.001
    position_changes = data['position'].diff().abs()
    data['strategy_returns'] -= position_changes * transaction_cost

    return data

def calculate_performance_metrics(data, strategy_col, name):
    """성과 지표 계산"""
    returns = data[strategy_col].dropna()
    if len(returns) == 0:
        return {}

    # 기본 통계
    total_return = (1 + returns).prod() - 1
    annual_return = returns.mean() * 252
    annual_vol = returns.std() * np.sqrt(252)
    sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0

    # 최대 낙폭
    cumulative = (1 + returns).cumprod()
    peak = cumulative.expanding().max()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()

    # 승률
    win_rate = (returns > 0).mean()

    return {
        'name': name,
        'total_return': total_return * 100,
        'annual_return': annual_return * 100,
        'annual_volatility': annual_vol * 100,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown * 100,
        'win_rate': win_rate * 100
    }

def create_performance_chart(results):
    """성과 비교 차트 생성"""
    print("📊 성과 비교 차트 생성 중...")

    # 메트릭 추출
    metrics_df = pd.DataFrame([r['metrics'] for r in results])

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # 1. 총 수익률
    bars1 = ax1.bar(metrics_df['name'], metrics_df['total_return'],
                   color=['blue', 'red', 'green', 'orange'], alpha=0.7)
    ax1.set_title('Total Return Comparison (%)', fontweight='bold')
    ax1.set_ylabel('Total Return (%)')
    for bar, val in zip(bars1, metrics_df['total_return']):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom')

    # 2. 샤프 비율
    bars2 = ax2.bar(metrics_df['name'], metrics_df['sharpe_ratio'],
                   color=['blue', 'red', 'green', 'orange'], alpha=0.7)
    ax2.set_title('Sharpe Ratio Comparison', fontweight='bold')
    ax2.set_ylabel('Sharpe Ratio')
    for bar, val in zip(bars2, metrics_df['sharpe_ratio']):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom')

    # 3. 최대 낙폭
    bars3 = ax3.bar(metrics_df['name'], metrics_df['max_drawdown'],
                   color=['blue', 'red', 'green', 'orange'], alpha=0.7)
    ax3.set_title('Maximum Drawdown (%)', fontweight='bold')
    ax3.set_ylabel('Max Drawdown (%)')
    for bar, val in zip(bars3, metrics_df['max_drawdown']):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() - 1,
                f'{val:.1f}%', ha='center', va='top')

    # 4. 승률
    bars4 = ax4.bar(metrics_df['name'], metrics_df['win_rate'],
                   color=['blue', 'red', 'green', 'orange'], alpha=0.7)
    ax4.set_title('Win Rate (%)', fontweight='bold')
    ax4.set_ylabel('Win Rate (%)')
    for bar, val in zip(bars4, metrics_df['win_rate']):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom')

    # X축 라벨 회전
    for ax in [ax1, ax2, ax3, ax4]:
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    # 저장
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/simple_backtest_results.png', dpi=300, bbox_inches='tight')
    print("✅ 저장: figures/simple_backtest_results.png")
    plt.close()

def main():
    """메인 백테스트 함수"""
    print("💰 간단한 경제적 백테스트 시작")
    print("=" * 50)

    # 1. 데이터 로드
    data = load_backtest_data()

    # 2. 변동성 예측 프록시 생성
    data = create_simple_volatility_proxy(data)

    # 3. 각 전략 백테스트
    results = []

    # Buy & Hold 벤치마크
    benchmark_data = data.copy()
    benchmark_metrics = calculate_performance_metrics(
        benchmark_data, 'returns', 'Buy & Hold (SPY)'
    )
    results.append({
        'data': benchmark_data,
        'metrics': benchmark_metrics
    })

    # 전략 1: 변동성 타이밍
    timing_data = data.copy()
    timing_data = strategy_volatility_timing(timing_data)
    timing_metrics = calculate_performance_metrics(
        timing_data, 'strategy_returns', 'Volatility Timing'
    )
    results.append({
        'data': timing_data,
        'metrics': timing_metrics
    })

    # 전략 2: 변동성 스케일링
    scaling_data = data.copy()
    scaling_data = strategy_volatility_scaling(scaling_data)
    scaling_metrics = calculate_performance_metrics(
        scaling_data, 'strategy_returns', 'Volatility Scaling'
    )
    results.append({
        'data': scaling_data,
        'metrics': scaling_metrics
    })

    # 전략 3: VIX 평균회귀
    vix_data = data.copy()
    vix_data = strategy_vix_mean_reversion(vix_data)
    vix_metrics = calculate_performance_metrics(
        vix_data, 'strategy_returns', 'VIX Mean Reversion'
    )
    results.append({
        'data': vix_data,
        'metrics': vix_metrics
    })

    # 4. 결과 출력
    print("\n📊 백테스트 결과 요약")
    print("=" * 50)

    metrics_df = pd.DataFrame([r['metrics'] for r in results])
    print(metrics_df.round(2))

    # 5. 시각화
    create_performance_chart(results)

    # 6. 결과 저장
    os.makedirs('results', exist_ok=True)

    backtest_summary = {
        'backtest_date': datetime.now().isoformat(),
        'period': '2020-2024',
        'strategies': len(results),
        'performance_metrics': [r['metrics'] for r in results]
    }

    with open('results/simple_backtest_results.json', 'w') as f:
        json.dump(backtest_summary, f, indent=2, default=str)

    print(f"\n💾 결과 저장: results/simple_backtest_results.json")

    # 최고 성과 전략
    best_strategy = max([r['metrics'] for r in results], key=lambda x: x['sharpe_ratio'])
    print(f"\n🏆 최고 샤프 비율: {best_strategy['name']}")
    print(f"   샤프 비율: {best_strategy['sharpe_ratio']:.2f}")
    print(f"   총 수익률: {best_strategy['total_return']:.1f}%")

    print("=" * 50)

    return results

if __name__ == "__main__":
    results = main()