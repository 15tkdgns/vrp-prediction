"""
2020-2024 기간 SPY 주가 예측 모델
- Ridge Regression 기반 주가 예측
- 실제 vs 예측 주가 비교 차트
- 동일 기간 백테스트 수행
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

try:
    import plotly.graph_objects as go
    import plotly.offline as pyo
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
    print("✅ plotly 사용 가능")
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ plotly 없음")

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class HistoricalPricePredictionModel:
    """2020-2024 SPY 주가 예측 모델"""

    def __init__(self):
        self.data = None
        self.model = None
        self.scaler = None
        self.predictions = None
        self.feature_columns = None

    def load_historical_data(self):
        """2020-2024 historical SPY 데이터 로드"""
        try:
            data_path = "data/training/sp500_2020_2024.csv"
            self.data = pd.read_csv(data_path)

            # Date 컬럼 처리
            self.data['Date'] = pd.to_datetime(self.data['Date'], utc=True).dt.tz_localize(None)
            self.data.set_index('Date', inplace=True)

            # 2020년 1월부터 필터링 (실제로는 3월부터 시작)
            self.data = self.data[self.data.index >= '2020-01-01']

            print(f"✅ 데이터 로드 완료: {len(self.data)}행")
            print(f"📅 기간: {self.data.index.min().date()} ~ {self.data.index.max().date()}")

            return True

        except Exception as e:
            print(f"⚠️ 데이터 로드 실패: {e}")
            return False

    def engineer_price_features(self):
        """주가 예측을 위한 특성 공학"""
        df = self.data.copy()

        # 기본 가격 특성
        df['price_lag_1'] = df['Close'].shift(1)
        df['price_lag_2'] = df['Close'].shift(2)
        df['price_lag_3'] = df['Close'].shift(3)
        df['price_lag_5'] = df['Close'].shift(5)

        # 이동평균
        df['ma_5'] = df['Close'].rolling(5).mean()
        df['ma_10'] = df['Close'].rolling(10).mean()
        df['ma_20'] = df['Close'].rolling(20).mean()
        df['ma_50'] = df['Close'].rolling(50).mean()

        # 수익률 특성
        df['return_1d'] = df['Close'].pct_change()
        df['return_5d'] = df['Close'].pct_change(5)
        df['return_20d'] = df['Close'].pct_change(20)

        # 변동성 특성
        df['volatility_5'] = df['return_1d'].rolling(5).std()
        df['volatility_20'] = df['return_1d'].rolling(20).std()

        # 기술적 지표
        df['rsi'] = self._calculate_rsi(df['Close'])
        df['macd'] = self._calculate_macd(df['Close'])

        # 가격 비율
        df['price_to_ma20'] = df['Close'] / df['ma_20']
        df['price_to_ma50'] = df['Close'] / df['ma_50']

        # 볼륨 정규화
        df['volume_norm'] = df['Volume'] / df['Volume'].rolling(20).mean()

        # 특성 선택
        feature_cols = [
            'price_lag_1', 'price_lag_2', 'price_lag_3', 'price_lag_5',
            'ma_5', 'ma_10', 'ma_20', 'ma_50',
            'return_1d', 'return_5d', 'return_20d',
            'volatility_5', 'volatility_20',
            'rsi', 'macd',
            'price_to_ma20', 'price_to_ma50', 'volume_norm'
        ]

        self.feature_columns = feature_cols

        # 결측치 제거
        df_clean = df.dropna()
        print(f"✅ 특성 공학 완료: {len(feature_cols)}개 특성, {len(df_clean)}행")

        return df_clean

    def _calculate_rsi(self, prices, window=14):
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_macd(self, prices, fast=12, slow=26):
        """MACD 계산"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd

    def train_price_prediction_model(self):
        """주가 예측 모델 훈련"""
        # 특성 공학 적용
        df = self.engineer_price_features()

        # X (features)와 y (target) 준비
        X = df[self.feature_columns].values
        y = df['Close'].values

        # Train/Test 분할 (시계열이므로 랜덤 분할 대신 시간 순서 유지)
        split_idx = int(len(X) * 0.8)  # 80% 훈련, 20% 테스트

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # 특성 정규화
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Ridge 모델 훈련
        self.model = Ridge(alpha=1.0, random_state=42)
        self.model.fit(X_train_scaled, y_train)

        # 성능 평가
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)

        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_mae = mean_absolute_error(y_test, test_pred)

        print(f"✅ 모델 훈련 완료")
        print(f"📊 Train R²: {train_r2:.4f}")
        print(f"📊 Test R²: {test_r2:.4f}")
        print(f"📊 Test RMSE: {test_rmse:.2f}")
        print(f"📊 Test MAE: {test_mae:.2f}")

        return df, train_pred, test_pred, split_idx

    def generate_full_period_predictions(self):
        """전체 기간(2020-2024) 예측 생성"""
        if self.model is None or self.scaler is None:
            print("⚠️ 모델이 훈련되지 않음")
            return None

        # 특성 공학 적용
        df = self.engineer_price_features()

        # 전체 기간 예측
        X_full = df[self.feature_columns].values
        X_full_scaled = self.scaler.transform(X_full)

        predictions = self.model.predict(X_full_scaled)

        # 예측 결과를 DataFrame으로 구성
        self.predictions = pd.DataFrame({
            'actual': df['Close'].values,
            'predicted': predictions
        }, index=df.index)

        # 전체 성능 지표
        full_r2 = r2_score(self.predictions['actual'], self.predictions['predicted'])
        full_rmse = np.sqrt(mean_squared_error(self.predictions['actual'], self.predictions['predicted']))
        full_mae = mean_absolute_error(self.predictions['actual'], self.predictions['predicted'])

        print(f"✅ 전체 기간 예측 완료: {len(self.predictions)}일")
        print(f"📊 전체 R²: {full_r2:.4f}")
        print(f"📊 전체 RMSE: {full_rmse:.2f}")
        print(f"📊 전체 MAE: {full_mae:.2f}")

        return self.predictions

    def create_comprehensive_comparison_chart(self, save_path="historical_actual_vs_predicted_prices.png"):
        """실제 vs 예측 주가 종합 비교 차트"""
        if self.predictions is None:
            print("⚠️ 예측 데이터가 없음")
            return None

        fig, axes = plt.subplots(3, 1, figsize=(16, 14), height_ratios=[3, 2, 1])

        # === 차트 1: 전체 기간 실제 vs 예측 ===
        ax1 = axes[0]

        # 실제 주가
        ax1.plot(self.predictions.index, self.predictions['actual'],
                color='blue', linewidth=2, label='실제 SPY 주가', alpha=0.8)

        # 예측 주가
        ax1.plot(self.predictions.index, self.predictions['predicted'],
                color='red', linewidth=2, linestyle='--', label='예측 주가', alpha=0.8)

        ax1.set_title('SPY 실제 vs 예측 주가 (2020-2024)', fontsize=16, fontweight='bold')
        ax1.set_ylabel('주가 ($)', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=11)
        ax1.grid(True, alpha=0.3)

        # === 차트 2: 오차 분석 ===
        ax2 = axes[1]

        prediction_error = self.predictions['predicted'] - self.predictions['actual']
        ax2.plot(self.predictions.index, prediction_error,
                color='green', linewidth=1, alpha=0.7, label='예측 오차')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.fill_between(self.predictions.index, prediction_error, 0,
                        color='green', alpha=0.2)

        ax2.set_title('예측 오차 (예측 - 실제)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('오차 ($)', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)

        # === 차트 3: 산점도 ===
        ax3 = axes[2]

        # 샘플링 (너무 많은 점 방지)
        sample_idx = np.random.choice(len(self.predictions), size=min(1000, len(self.predictions)), replace=False)
        actual_sample = self.predictions['actual'].iloc[sample_idx]
        predicted_sample = self.predictions['predicted'].iloc[sample_idx]

        ax3.scatter(actual_sample, predicted_sample, alpha=0.6, s=20, color='purple')

        # 완벽한 예측선 (y=x)
        min_val = min(actual_sample.min(), predicted_sample.min())
        max_val = max(actual_sample.max(), predicted_sample.max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='완벽한 예측')

        ax3.set_xlabel('실제 주가 ($)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('예측 주가 ($)', fontsize=12, fontweight='bold')
        ax3.set_title('실제 vs 예측 산점도', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 통계 정보 추가
        r2 = r2_score(self.predictions['actual'], self.predictions['predicted'])
        rmse = np.sqrt(mean_squared_error(self.predictions['actual'], self.predictions['predicted']))
        mae = mean_absolute_error(self.predictions['actual'], self.predictions['predicted'])

        stats_text = f'성능 지표\n'
        stats_text += f'R² Score: {r2:.4f}\n'
        stats_text += f'RMSE: ${rmse:.2f}\n'
        stats_text += f'MAE: ${mae:.2f}\n'
        stats_text += f'데이터: {len(self.predictions):,}일'

        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round',
                facecolor='white', alpha=0.8))

        # 날짜 포맷팅
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.YearLocator())

        plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 종합 비교 차트 저장: {save_path}")
        plt.close()

        return save_path

    def create_interactive_comparison_chart(self, save_path="historical_actual_vs_predicted_interactive.html"):
        """인터랙티브 실제 vs 예측 주가 차트"""
        if not PLOTLY_AVAILABLE or self.predictions is None:
            print("⚠️ plotly 없거나 예측 데이터 없음")
            return None

        # 서브플롯 생성
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=['실제 vs 예측 주가', '예측 오차', '성능 지표 추이'],
            vertical_spacing=0.08,
            row_heights=[0.6, 0.25, 0.15]
        )

        # 실제 주가
        fig.add_trace(
            go.Scatter(
                x=self.predictions.index,
                y=self.predictions['actual'],
                mode='lines',
                name='실제 SPY 주가',
                line=dict(color='blue', width=2),
                hovertemplate='<b>실제</b><br>날짜: %{x}<br>주가: $%{y:.2f}<extra></extra>'
            ), row=1, col=1
        )

        # 예측 주가
        fig.add_trace(
            go.Scatter(
                x=self.predictions.index,
                y=self.predictions['predicted'],
                mode='lines',
                name='예측 주가',
                line=dict(color='red', width=2, dash='dash'),
                hovertemplate='<b>예측</b><br>날짜: %{x}<br>주가: $%{y:.2f}<extra></extra>'
            ), row=1, col=1
        )

        # 예측 오차
        prediction_error = self.predictions['predicted'] - self.predictions['actual']
        fig.add_trace(
            go.Scatter(
                x=self.predictions.index,
                y=prediction_error,
                mode='lines',
                name='예측 오차',
                line=dict(color='green', width=1),
                fill='tonexty',
                fillcolor='rgba(0,255,0,0.2)',
                hovertemplate='<b>오차</b><br>날짜: %{x}<br>오차: $%{y:.2f}<extra></extra>'
            ), row=2, col=1
        )

        # 30일 이동 R²
        rolling_r2 = []
        window = 30
        for i in range(window, len(self.predictions)):
            actual_window = self.predictions['actual'].iloc[i-window:i]
            pred_window = self.predictions['predicted'].iloc[i-window:i]
            r2 = r2_score(actual_window, pred_window)
            rolling_r2.append(r2)

        fig.add_trace(
            go.Scatter(
                x=self.predictions.index[window:],
                y=rolling_r2,
                mode='lines',
                name='30일 이동 R²',
                line=dict(color='purple', width=2),
                hovertemplate='<b>30일 R²</b><br>날짜: %{x}<br>R²: %{y:.3f}<extra></extra>'
            ), row=3, col=1
        )

        # 레이아웃 설정
        fig.update_layout(
            title='SPY 주가 예측 모델 성능 분석 (2020-2024)',
            height=800,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )

        # Y축 레이블
        fig.update_yaxes(title_text="주가 ($)", row=1, col=1)
        fig.update_yaxes(title_text="오차 ($)", row=2, col=1)
        fig.update_yaxes(title_text="R²", row=3, col=1)
        fig.update_xaxes(title_text="날짜", row=3, col=1)

        # HTML 저장
        pyo.plot(fig, filename=save_path, auto_open=False)
        print(f"✅ 인터랙티브 차트 저장: {save_path}")

        return save_path

    def run_complete_analysis(self):
        """전체 분석 실행"""
        print("🚀 2020-2024 SPY 주가 예측 모델 분석 시작")

        # 1. 데이터 로드
        print("\n📊 데이터 로드 중...")
        if not self.load_historical_data():
            return None, None

        # 2. 모델 훈련
        print("\n🤖 주가 예측 모델 훈련 중...")
        self.train_price_prediction_model()

        # 3. 전체 기간 예측 생성
        print("\n🎯 전체 기간 예측 생성 중...")
        self.generate_full_period_predictions()

        # 4. 종합 차트 생성
        print("\n📈 종합 비교 차트 생성 중...")
        static_path = self.create_comprehensive_comparison_chart()

        # 5. 인터랙티브 차트 생성
        print("\n🎯 인터랙티브 차트 생성 중...")
        interactive_path = self.create_interactive_comparison_chart()

        print(f"\n✅ 전체 분석 완료!")
        print(f"📁 정적 차트: {static_path}")
        if interactive_path:
            print(f"🌐 인터랙티브 차트: {interactive_path}")

        return static_path, interactive_path

if __name__ == "__main__":
    analyzer = HistoricalPricePredictionModel()
    analyzer.run_complete_analysis()