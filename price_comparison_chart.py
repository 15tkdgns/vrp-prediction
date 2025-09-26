"""
실제 SPY 주가 vs 예측 주가 시계열 비교 그래프 생성
- 실제 주가와 예측 주가를 겹쳐서 표시
- 정적 PNG 및 인터랙티브 HTML 출력
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
    print("✅ yfinance 사용 가능")
except ImportError:
    YFINANCE_AVAILABLE = False
    print("⚠️ yfinance 없음 - 시뮬레이션 데이터 사용")

try:
    import plotly.graph_objects as go
    import plotly.offline as pyo
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
    print("✅ plotly 사용 가능 - 인터랙티브 그래프 생성")
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ plotly 없음 - 정적 그래프만 생성")

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class PriceComparisonChart:
    """실제 vs 예측 주가 비교 차트 생성기"""

    def __init__(self):
        self.actual_data = None
        self.prediction_data = None

    def load_actual_spy_data(self, start_date="2024-01-01", end_date="2025-12-31"):
        """실제 SPY 주가 데이터 로드"""
        if YFINANCE_AVAILABLE:
            try:
                spy = yf.Ticker("SPY")
                self.actual_data = spy.history(start=start_date, end=end_date)
                print(f"✅ SPY 실제 데이터 로드 완료: {len(self.actual_data)} 일치")
                return True
            except Exception as e:
                print(f"⚠️ yfinance 데이터 로드 실패: {e}")
                return False
        else:
            # 시뮬레이션 데이터 생성
            self._generate_simulation_actual_data(start_date, end_date)
            return True

    def _generate_simulation_actual_data(self, start_date, end_date):
        """시뮬레이션용 실제 데이터 생성"""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        dates = dates[dates.dayofweek < 5]  # 주말 제외

        # 실제 SPY 패턴과 유사한 데이터 생성
        np.random.seed(42)
        initial_price = 450
        returns = np.random.normal(0.0005, 0.015, len(dates))  # 일일 수익률

        prices = [initial_price]
        for ret in returns[:-1]:
            prices.append(prices[-1] * (1 + ret))

        self.actual_data = pd.DataFrame({
            'Close': prices
        }, index=dates)

        print(f"✅ 시뮬레이션 SPY 데이터 생성 완료: {len(self.actual_data)} 일치")

    def load_prediction_data(self):
        """예측 데이터 로드"""
        prediction_files = [
            "data/validation/predictions_2025_h1.json",
            "data/raw/spy_2025_h1_predictions.json",
            "data/validation/daily_predictions_2025_h1.json"
        ]

        for file_path in prediction_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 월별 예측 데이터 처리
                if 'monthly_predictions' in data:
                    predictions = []
                    for pred in data['monthly_predictions']:
                        # 월 중간일로 날짜 설정
                        month_str = pred['month'] + '-15'
                        date = datetime.strptime(month_str, '%Y-%m-%d')
                        predictions.append({
                            'date': date,
                            'predicted_price': pred['predicted_price'],
                            'confidence': pred.get('confidence', 70)
                        })

                    self.prediction_data = pd.DataFrame(predictions)
                    self.prediction_data.set_index('date', inplace=True)
                    print(f"✅ 예측 데이터 로드 완료: {file_path}")
                    return True

            except Exception as e:
                print(f"⚠️ {file_path} 로드 실패: {e}")
                continue

        # 예측 데이터가 없으면 시뮬레이션 생성
        self._generate_simulation_predictions()
        return True

    def _generate_simulation_predictions(self):
        """시뮬레이션용 예측 데이터 생성"""
        if self.actual_data is None:
            print("⚠️ 실제 데이터가 없어 예측 데이터 생성 불가")
            return

        # 2025년 매월 15일에 예측값 생성
        prediction_dates = pd.date_range('2025-01-15', '2025-06-15', freq='MS') + pd.Timedelta(days=14)

        predictions = []
        base_price = self.actual_data['Close'].iloc[-1] if len(self.actual_data) > 0 else 450

        for i, date in enumerate(prediction_dates):
            # 약간의 상승 트렌드 + 노이즈
            trend_factor = 1.02 + (i * 0.01)  # 월 2% + 추가 상승
            noise = np.random.normal(0, 0.05)  # 5% 노이즈
            predicted_price = base_price * trend_factor * (1 + noise)

            predictions.append({
                'predicted_price': predicted_price,
                'confidence': np.random.randint(60, 85)
            })

        self.prediction_data = pd.DataFrame(predictions, index=prediction_dates)
        print(f"✅ 시뮬레이션 예측 데이터 생성 완료: {len(self.prediction_data)} 포인트")

    def create_static_comparison_chart(self, save_path="price_comparison_timeseries.png"):
        """정적 PNG 비교 차트 생성"""
        if self.actual_data is None or self.prediction_data is None:
            print("⚠️ 데이터가 로드되지 않음")
            return None

        fig, ax = plt.subplots(figsize=(15, 8))

        # 실제 주가 (파란색 실선)
        ax.plot(self.actual_data.index, self.actual_data['Close'],
                color='blue', linewidth=2, label='실제 SPY 주가', alpha=0.8)

        # 예측 주가 (빨간색 점선)
        ax.plot(self.prediction_data.index, self.prediction_data['predicted_price'],
                color='red', linestyle='--', linewidth=3, marker='o', markersize=8,
                label='예측 주가', alpha=0.9)

        # 예측 신뢰구간 (음영)
        if 'confidence' in self.prediction_data.columns:
            confidence = self.prediction_data['confidence'] / 100
            upper_bound = self.prediction_data['predicted_price'] * (1 + (1-confidence) * 0.1)
            lower_bound = self.prediction_data['predicted_price'] * (1 - (1-confidence) * 0.1)

            ax.fill_between(self.prediction_data.index, lower_bound, upper_bound,
                           alpha=0.2, color='red', label='예측 신뢰구간')

        # 차트 스타일링
        ax.set_title('SPY 실제 주가 vs 예측 주가 비교', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('날짜', fontsize=12, fontweight='bold')
        ax.set_ylabel('주가 ($)', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=11)
        ax.grid(True, alpha=0.3)

        # 날짜 포맷팅
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)

        # 통계 정보 추가
        if len(self.prediction_data) > 0:
            latest_actual = self.actual_data['Close'].iloc[-1]
            first_prediction = self.prediction_data['predicted_price'].iloc[0]

            stats_text = f'최근 실제 주가: ${latest_actual:.2f}\n'
            stats_text += f'첫 예측 주가: ${first_prediction:.2f}\n'
            stats_text += f'예측 포인트: {len(self.prediction_data)}개'

            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round',
                   facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 정적 차트 저장 완료: {save_path}")
        plt.close()

        return save_path

    def create_interactive_chart(self, save_path="price_comparison_interactive.html"):
        """인터랙티브 HTML 차트 생성"""
        if not PLOTLY_AVAILABLE:
            print("⚠️ plotly 없음 - 인터랙티브 차트 생성 불가")
            return None

        if self.actual_data is None or self.prediction_data is None:
            print("⚠️ 데이터가 로드되지 않음")
            return None

        fig = go.Figure()

        # 실제 주가
        fig.add_trace(go.Scatter(
            x=self.actual_data.index,
            y=self.actual_data['Close'],
            mode='lines',
            name='실제 SPY 주가',
            line=dict(color='blue', width=2),
            hovertemplate='<b>실제 주가</b><br>날짜: %{x}<br>주가: $%{y:.2f}<extra></extra>'
        ))

        # 예측 주가
        fig.add_trace(go.Scatter(
            x=self.prediction_data.index,
            y=self.prediction_data['predicted_price'],
            mode='lines+markers',
            name='예측 주가',
            line=dict(color='red', width=3, dash='dash'),
            marker=dict(size=8),
            hovertemplate='<b>예측 주가</b><br>날짜: %{x}<br>주가: $%{y:.2f}<br>신뢰도: %{customdata}%<extra></extra>',
            customdata=self.prediction_data['confidence'] if 'confidence' in self.prediction_data.columns else None
        ))

        # 예측 신뢰구간
        if 'confidence' in self.prediction_data.columns:
            confidence = self.prediction_data['confidence'] / 100
            upper_bound = self.prediction_data['predicted_price'] * (1 + (1-confidence) * 0.1)
            lower_bound = self.prediction_data['predicted_price'] * (1 - (1-confidence) * 0.1)

            fig.add_trace(go.Scatter(
                x=self.prediction_data.index,
                y=upper_bound,
                fill=None,
                mode='lines',
                line_color='rgba(255,0,0,0)',
                showlegend=False,
                hoverinfo='skip'
            ))

            fig.add_trace(go.Scatter(
                x=self.prediction_data.index,
                y=lower_bound,
                fill='tonexty',
                mode='lines',
                line_color='rgba(255,0,0,0)',
                name='예측 신뢰구간',
                fillcolor='rgba(255,0,0,0.2)',
                hoverinfo='skip'
            ))

        # 레이아웃 설정
        fig.update_layout(
            title='SPY 실제 주가 vs 예측 주가 비교 (인터랙티브)',
            xaxis_title='날짜',
            yaxis_title='주가 ($)',
            hovermode='x unified',
            template='plotly_white',
            width=1200,
            height=600,
            legend=dict(x=0, y=1)
        )

        # HTML 저장
        pyo.plot(fig, filename=save_path, auto_open=False)
        print(f"✅ 인터랙티브 차트 저장 완료: {save_path}")

        return save_path

    def generate_all_charts(self):
        """모든 차트 생성"""
        print("🚀 SPY 주가 비교 차트 생성 시작")

        # 1. 실제 데이터 로드
        print("\n📊 실제 SPY 데이터 로드 중...")
        self.load_actual_spy_data()

        # 2. 예측 데이터 로드
        print("\n🔮 예측 데이터 로드 중...")
        self.load_prediction_data()

        # 3. 정적 차트 생성
        print("\n📈 정적 비교 차트 생성 중...")
        static_path = self.create_static_comparison_chart()

        # 4. 인터랙티브 차트 생성
        print("\n🎯 인터랙티브 차트 생성 중...")
        interactive_path = self.create_interactive_chart()

        print(f"\n✅ 차트 생성 완료!")
        print(f"📁 정적 차트: {static_path}")
        if interactive_path:
            print(f"🌐 인터랙티브 차트: {interactive_path}")

        return static_path, interactive_path

if __name__ == "__main__":
    chart_generator = PriceComparisonChart()
    chart_generator.generate_all_charts()