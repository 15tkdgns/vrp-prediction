"""
경제적 수치 차트 생성
백테스트 수익률, 위험 지표, 거래비용 분석
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class EconomicMetricsCharts:
    """경제적 수치 차트 생성기"""

    def __init__(self, output_dir: str = "src/visualization/outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 차트 스타일 설정
        plt.style.use('default')

    def create_return_comparison_chart(self, economic_metrics: Dict[str, float]) -> str:
        """수익률 비교 차트"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        strategy_return = economic_metrics.get('strategy_annual_return', 0) * 100
        benchmark_return = economic_metrics.get('benchmark_annual_return', 0) * 100

        categories = ['Strategy', 'Benchmark']
        returns = [strategy_return, benchmark_return]
        colors = ['#2E8B57', '#708090']

        bars = ax.bar(categories, returns, color=colors, alpha=0.8, edgecolor='black', linewidth=1)

        # 수치 라벨 추가
        for bar, ret in zip(bars, returns):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{ret:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

        ax.set_title('Annual Return Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel('Annual Return (%)', fontsize=12)
        ax.grid(axis='y', alpha=0.3)

        # 0% 기준선
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

        plt.tight_layout()

        # 저장
        output_path = self.output_dir / "return_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(output_path)

    def create_risk_metrics_chart(self, economic_metrics: Dict[str, float]) -> str:
        """위험 지표 비교 차트"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        strategy_vol = economic_metrics.get('strategy_volatility', 0) * 100
        benchmark_vol = economic_metrics.get('benchmark_volatility', 0) * 100
        strategy_sharpe = economic_metrics.get('strategy_sharpe', 0)
        strategy_mdd = economic_metrics.get('strategy_max_drawdown', 0) * 100

        # 1. 변동성 비교
        vol_categories = ['Strategy', 'Benchmark']
        volatilities = [strategy_vol, benchmark_vol]
        bars1 = axes[0].bar(vol_categories, volatilities, color=['#4ECDC4', '#FF6B6B'], alpha=0.8, edgecolor='black')

        for bar, vol in zip(bars1, volatilities):
            axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                        f'{vol:.2f}%', ha='center', va='bottom', fontweight='bold')

        axes[0].set_title('Volatility Comparison', fontweight='bold')
        axes[0].set_ylabel('Volatility (%)', fontsize=11)
        axes[0].grid(axis='y', alpha=0.3)

        # 2. 샤프 비율
        axes[1].bar(['Strategy'], [strategy_sharpe], color='#45B7D1', alpha=0.8, edgecolor='black')
        axes[1].text(0, strategy_sharpe + 0.02, f'{strategy_sharpe:.3f}',
                    ha='center', va='bottom', fontweight='bold')
        axes[1].set_title('Sharpe Ratio', fontweight='bold')
        axes[1].set_ylabel('Sharpe Ratio', fontsize=11)
        axes[1].grid(axis='y', alpha=0.3)

        # 3. 최대 낙폭
        axes[2].bar(['Strategy'], [strategy_mdd], color='#FF6B6B', alpha=0.8, edgecolor='black')
        axes[2].text(0, strategy_mdd - 1, f'{strategy_mdd:.2f}%',
                    ha='center', va='top', fontweight='bold', color='white')
        axes[2].set_title('Maximum Drawdown', fontweight='bold')
        axes[2].set_ylabel('Max Drawdown (%)', fontsize=11)
        axes[2].grid(axis='y', alpha=0.3)

        plt.suptitle('Risk Metrics Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # 저장
        output_path = self.output_dir / "risk_metrics.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(output_path)

    def create_transaction_cost_analysis(self, economic_metrics: Dict[str, float]) -> str:
        """거래비용 분석 차트"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        total_cost = economic_metrics.get('total_transaction_costs', 0) * 100
        vol_reduction = economic_metrics.get('volatility_reduction', 0) * 100

        # 1. 거래비용
        ax1.bar(['Total Transaction Costs'], [total_cost], color='#FF8C00', alpha=0.8, edgecolor='black')
        ax1.text(0, total_cost + 0.1, f'{total_cost:.2f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=12)
        ax1.set_title('Total Transaction Costs', fontweight='bold')
        ax1.set_ylabel('Cost (%)', fontsize=12)
        ax1.grid(axis='y', alpha=0.3)

        # 2. 변동성 감소
        color = '#2E8B57' if vol_reduction > 0 else '#DC143C'
        ax2.bar(['Volatility Reduction'], [vol_reduction], color=color, alpha=0.8, edgecolor='black')
        ax2.text(0, vol_reduction + (0.05 if vol_reduction > 0 else -0.05), f'{vol_reduction:.2f}%',
                ha='center', va='bottom' if vol_reduction > 0 else 'top', fontweight='bold', fontsize=12)
        ax2.set_title('Volatility Reduction', fontweight='bold')
        ax2.set_ylabel('Reduction (%)', fontsize=12)
        ax2.grid(axis='y', alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)

        plt.suptitle('Transaction Cost and Volatility Impact', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # 저장
        output_path = self.output_dir / "transaction_cost_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(output_path)

    def create_cumulative_return_simulation(self, economic_metrics: Dict[str, float]) -> str:
        """누적 수익률 시뮬레이션 차트"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        strategy_annual = economic_metrics.get('strategy_annual_return', 0)
        benchmark_annual = economic_metrics.get('benchmark_annual_return', 0)
        strategy_vol = economic_metrics.get('strategy_volatility', 0)
        benchmark_vol = economic_metrics.get('benchmark_volatility', 0)

        # 시뮬레이션 데이터 생성 (252 거래일)
        np.random.seed(42)
        trading_days = 252

        # 일일 수익률 시뮬레이션
        strategy_daily_return = strategy_annual / trading_days
        benchmark_daily_return = benchmark_annual / trading_days

        strategy_daily_vol = strategy_vol / np.sqrt(trading_days)
        benchmark_daily_vol = benchmark_vol / np.sqrt(trading_days)

        # 일일 수익률 생성
        strategy_returns = np.random.normal(strategy_daily_return, strategy_daily_vol, trading_days)
        benchmark_returns = np.random.normal(benchmark_daily_return, benchmark_daily_vol, trading_days)

        # 누적 수익률 계산
        strategy_cumulative = (1 + strategy_returns).cumprod() - 1
        benchmark_cumulative = (1 + benchmark_returns).cumprod() - 1

        days = np.arange(1, trading_days + 1)

        # 플롯
        ax.plot(days, strategy_cumulative * 100, label='Strategy', color='#2E8B57', linewidth=2)
        ax.plot(days, benchmark_cumulative * 100, label='Benchmark', color='#708090', linewidth=2)

        ax.set_title('Cumulative Return Simulation (1 Year)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Trading Days', fontsize=12)
        ax.set_ylabel('Cumulative Return (%)', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(alpha=0.3)

        # 최종 수익률 표시
        final_strategy = strategy_cumulative[-1] * 100
        final_benchmark = benchmark_cumulative[-1] * 100

        ax.text(0.02, 0.98, f'Strategy Final: {final_strategy:.1f}%\nBenchmark Final: {final_benchmark:.1f}%',
               transform=ax.transAxes, fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        # 저장
        output_path = self.output_dir / "cumulative_return_simulation.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(output_path)

    def create_economic_summary_table(self, economic_metrics: Dict[str, float]) -> str:
        """경제적 지표 요약 테이블"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # 테이블 데이터 준비
        table_data = [
            ['Metric', 'Strategy', 'Benchmark/Target', 'Difference']
        ]

        strategy_return = economic_metrics.get('strategy_annual_return', 0) * 100
        benchmark_return = economic_metrics.get('benchmark_annual_return', 0) * 100
        strategy_vol = economic_metrics.get('strategy_volatility', 0) * 100
        benchmark_vol = economic_metrics.get('benchmark_volatility', 0) * 100
        vol_reduction = economic_metrics.get('volatility_reduction', 0) * 100
        total_costs = economic_metrics.get('total_transaction_costs', 0) * 100
        sharpe_ratio = economic_metrics.get('strategy_sharpe', 0)
        max_dd = economic_metrics.get('strategy_max_drawdown', 0) * 100

        table_data.extend([
            ['Annual Return', f'{strategy_return:.2f}%', f'{benchmark_return:.2f}%', f'{strategy_return - benchmark_return:.2f}%'],
            ['Volatility', f'{strategy_vol:.2f}%', f'{benchmark_vol:.2f}%', f'{strategy_vol - benchmark_vol:.2f}%'],
            ['Volatility Reduction', f'{vol_reduction:.2f}%', '0.00%', f'{vol_reduction:.2f}%'],
            ['Sharpe Ratio', f'{sharpe_ratio:.3f}', '-', '-'],
            ['Max Drawdown', f'{max_dd:.2f}%', '-', '-'],
            ['Transaction Costs', f'{total_costs:.2f}%', '0.00%', f'{total_costs:.2f}%']
        ])

        # 테이블 생성
        table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                        cellLoc='center', loc='center',
                        bbox=[0.1, 0.1, 0.8, 0.8])

        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)

        # 헤더 스타일
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # 데이터 행 스타일
        for i in range(1, len(table_data)):
            for j in range(len(table_data[0])):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#F2F2F2')

                # 차이 열에서 음수는 빨간색, 양수는 파란색
                if j == 3 and table_data[i][j] != '-':
                    try:
                        value = float(table_data[i][j].replace('%', ''))
                        if value < 0:
                            table[(i, j)].set_text_props(color='red', weight='bold')
                        elif value > 0:
                            table[(i, j)].set_text_props(color='blue', weight='bold')
                    except:
                        pass

        ax.set_title('Economic Performance Summary', fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')

        # 저장
        output_path = self.output_dir / "economic_summary.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return str(output_path)

    def generate_all_economic_charts(self, economic_metrics: Dict[str, float]) -> List[str]:
        """모든 경제적 수치 차트 생성"""
        output_files = []

        if not economic_metrics:
            print("⚠️ 경제적 지표 데이터가 없습니다.")
            return output_files

        print("경제적 수치 차트 생성 중...")

        # 수익률 비교 차트
        output_files.append(self.create_return_comparison_chart(economic_metrics))
        print("✓ 수익률 비교 차트 완료")

        # 위험 지표 차트
        output_files.append(self.create_risk_metrics_chart(economic_metrics))
        print("✓ 위험 지표 차트 완료")

        # 거래비용 분석 차트
        output_files.append(self.create_transaction_cost_analysis(economic_metrics))
        print("✓ 거래비용 분석 차트 완료")

        # 누적 수익률 시뮬레이션
        output_files.append(self.create_cumulative_return_simulation(economic_metrics))
        print("✓ 누적 수익률 시뮬레이션 완료")

        # 경제적 지표 요약 테이블
        output_files.append(self.create_economic_summary_table(economic_metrics))
        print("✓ 경제적 지표 요약 테이블 완료")

        return [f for f in output_files if f]  # 빈 문자열 제거


if __name__ == "__main__":
    # 테스트 데이터
    test_economic_metrics = {
        'strategy_annual_return': 0.141,
        'benchmark_annual_return': 0.2271,
        'strategy_volatility': 0.1224,
        'benchmark_volatility': 0.1304,
        'strategy_sharpe': 0.989,
        'strategy_max_drawdown': -0.1081,
        'total_transaction_costs': 0.0474,
        'volatility_reduction': 0.008
    }

    # 차트 생성 테스트
    chart_generator = EconomicMetricsCharts()
    output_files = chart_generator.generate_all_economic_charts(test_economic_metrics)

    print(f"\n생성된 파일들:")
    for file in output_files:
        print(f"  - {file}")