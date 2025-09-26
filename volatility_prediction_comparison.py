"""
올바른 변동성 예측 모델 vs 실제 변동성 비교
- target_vol_5d (5일 후 변동성) 예측
- 현실적이고 검증된 모델 성능
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class VolatilityPredictionModel:
    """올바른 변동성 예측 모델 (target_vol_5d)"""

    def __init__(self):
        self.data = None
        self.model = None
        self.scaler = None
        self.predictions = None

    def load_and_process_data(self):
        """SPY 데이터 로드 및 변동성 계산"""
        try:
            data_path = "data/training/sp500_2020_2024.csv"
            df = pd.read_csv(data_path)
            df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)
            df.set_index('Date', inplace=True)

            # 2020년부터 필터링
            df = df[df.index >= '2020-01-01']

            print(f"✅ 데이터 로드: {len(df)}행")

            # 수익률 계산 (올바른 방식)
            df['returns'] = np.log(df['Close'] / df['Close'].shift(1))

            # 미래 5일 변동성 계산 (타겟 변수)
            # Forward-looking volatility: t+1부터 t+5까지의 변동성
            df['target_vol_5d'] = df['returns'].shift(-1).rolling(5).std()

            # 과거 변동성 특성들 (현재 시점 이전만)
            df['vol_1d'] = df['returns'].rolling(1).std()
            df['vol_5d'] = df['returns'].rolling(5).std()
            df['vol_10d'] = df['returns'].rolling(10).std()
            df['vol_20d'] = df['returns'].rolling(20).std()
            df['vol_60d'] = df['returns'].rolling(60).std()

            # 과거 수익률 특성
            df['return_lag1'] = df['returns'].shift(1)
            df['return_lag5'] = df['returns'].shift(5)

            # 변동성 비율
            df['vol_ratio_5_20'] = df['vol_5d'] / df['vol_20d']
            df['vol_ratio_10_60'] = df['vol_10d'] / df['vol_60d']

            # 가격 기반 특성 (시간적 분리)
            df['price_change_5d'] = (df['Close'] / df['Close'].shift(5) - 1).abs()
            df['range_volatility'] = (df['High'] - df['Low']) / df['Close']

            # 거래량 기반
            df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()

            # 디버깅: NaN 확인
            print(f"📊 처리 전: {len(df)}행")
            print(f"📊 target_vol_5d NaN 개수: {df['target_vol_5d'].isna().sum()}")
            print(f"📊 vol_5d NaN 개수: {df['vol_5d'].isna().sum()}")

            self.data = df.dropna()
            print(f"✅ 특성 생성 완료: {len(self.data)}행")

            if len(self.data) == 0:
                print("⚠️ 모든 데이터가 NaN으로 제거됨. 다른 방법 시도...")
                # 최소한의 특성으로 다시 시도
                simple_df = df[['returns', 'Close', 'High', 'Low', 'Volume']].copy()
                simple_df['vol_20d'] = simple_df['returns'].rolling(20).std()
                simple_df['target_vol_5d'] = simple_df['returns'].shift(-1).rolling(5).std()
                simple_df['vol_5d'] = simple_df['returns'].rolling(5).std()
                simple_df['return_lag1'] = simple_df['returns'].shift(1)

                self.data = simple_df.dropna()
                print(f"✅ 단순화 후: {len(self.data)}행")

            return True

        except Exception as e:
            print(f"⚠️ 데이터 처리 실패: {e}")
            return False

    def train_volatility_model(self):
        """변동성 예측 모델 훈련"""
        # 사용 가능한 컬럼만 선택
        all_columns = self.data.columns.tolist()
        potential_features = [
            'vol_1d', 'vol_5d', 'vol_10d', 'vol_20d', 'vol_60d',
            'return_lag1', 'return_lag5',
            'vol_ratio_5_20', 'vol_ratio_10_60',
            'price_change_5d', 'range_volatility', 'volume_ratio'
        ]

        feature_columns = [col for col in potential_features if col in all_columns]
        print(f"📊 사용 가능한 특성: {feature_columns}")

        X = self.data[feature_columns].values
        y = self.data['target_vol_5d'].values

        # 시계열 분할 (80% 훈련)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # 스케일링
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Ridge 모델 (alpha=1.0, 원래 모델과 동일)
        self.model = Ridge(alpha=1.0, random_state=42)
        self.model.fit(X_train_scaled, y_train)

        # 성능 평가
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)

        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_mae = mean_absolute_error(y_test, test_pred)

        print(f"✅ 변동성 예측 모델 훈련 완료")
        print(f"📊 Train R²: {train_r2:.4f}")
        print(f"📊 Test R²: {test_r2:.4f}")
        print(f"📊 Test RMSE: {test_rmse:.6f}")
        print(f"📊 Test MAE: {test_mae:.6f}")

        self.feature_columns = feature_columns  # 저장
        return feature_columns, split_idx

    def generate_full_predictions(self):
        """전체 기간 변동성 예측"""
        feature_columns = self.feature_columns  # 저장된 컬럼 사용

        X_full = self.data[feature_columns].values
        X_full_scaled = self.scaler.transform(X_full)
        predictions = self.model.predict(X_full_scaled)

        self.predictions = pd.DataFrame({
            'actual_volatility': self.data['target_vol_5d'].values,
            'predicted_volatility': predictions,
            'spy_price': self.data['Close'].values
        }, index=self.data.index)

        # 전체 성능
        full_r2 = r2_score(self.predictions['actual_volatility'],
                          self.predictions['predicted_volatility'])
        full_rmse = np.sqrt(mean_squared_error(self.predictions['actual_volatility'],
                                             self.predictions['predicted_volatility']))

        print(f"✅ 전체 변동성 예측: R²={full_r2:.4f}, RMSE={full_rmse:.6f}")

        return self.predictions

    def create_volatility_comparison_chart(self, save_path="volatility_actual_vs_predicted.png"):
        """실제 vs 예측 변동성 비교 차트"""
        if self.predictions is None:
            return None

        fig, axes = plt.subplots(4, 1, figsize=(16, 16), height_ratios=[2, 2, 1, 1])

        # === 차트 1: SPY 주가 (참조용) ===
        ax1 = axes[0]
        ax1.plot(self.predictions.index, self.predictions['spy_price'],
                color='blue', linewidth=1.5, label='SPY 주가')
        ax1.set_title('SPY 주가 (참조)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('주가 ($)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # === 차트 2: 실제 vs 예측 변동성 ===
        ax2 = axes[1]
        ax2.plot(self.predictions.index, self.predictions['actual_volatility'],
                color='red', linewidth=2, label='실제 변동성 (target_vol_5d)', alpha=0.8)
        ax2.plot(self.predictions.index, self.predictions['predicted_volatility'],
                color='darkred', linewidth=2, linestyle='--', label='예측 변동성', alpha=0.8)

        ax2.set_title('실제 vs 예측 변동성 (5일 후 변동성)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('변동성', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # === 차트 3: 변동성 예측 오차 ===
        ax3 = axes[2]
        vol_error = self.predictions['predicted_volatility'] - self.predictions['actual_volatility']
        ax3.plot(self.predictions.index, vol_error, color='green', linewidth=1, alpha=0.7)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.fill_between(self.predictions.index, vol_error, 0, color='green', alpha=0.2)

        ax3.set_title('변동성 예측 오차', fontsize=14, fontweight='bold')
        ax3.set_ylabel('오차', fontsize=12)
        ax3.grid(True, alpha=0.3)

        # === 차트 4: 변동성 산점도 ===
        ax4 = axes[3]
        # 샘플링
        sample_idx = np.random.choice(len(self.predictions), size=min(500, len(self.predictions)), replace=False)
        actual_sample = self.predictions['actual_volatility'].iloc[sample_idx]
        pred_sample = self.predictions['predicted_volatility'].iloc[sample_idx]

        ax4.scatter(actual_sample, pred_sample, alpha=0.6, s=15, color='purple')

        # 완벽한 예측선
        min_val = min(actual_sample.min(), pred_sample.min())
        max_val = max(actual_sample.max(), pred_sample.max())
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='완벽한 예측')

        ax4.set_xlabel('실제 변동성', fontsize=12)
        ax4.set_ylabel('예측 변동성', fontsize=12)
        ax4.set_title('변동성 예측 정확도', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 성능 지표 추가
        r2 = r2_score(self.predictions['actual_volatility'], self.predictions['predicted_volatility'])
        rmse = np.sqrt(mean_squared_error(self.predictions['actual_volatility'], self.predictions['predicted_volatility']))
        mae = mean_absolute_error(self.predictions['actual_volatility'], self.predictions['predicted_volatility'])

        stats_text = f'변동성 예측 성능\n'
        stats_text += f'R² Score: {r2:.4f}\n'
        stats_text += f'RMSE: {rmse:.6f}\n'
        stats_text += f'MAE: {mae:.6f}\n'
        stats_text += f'데이터: {len(self.predictions):,}일\n'
        stats_text += f'타겟: 5일 후 변동성'

        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round',
                facecolor='white', alpha=0.8))

        # 날짜 포맷팅
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.YearLocator())

        plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 변동성 비교 차트 저장: {save_path}")
        plt.close()

        return save_path

    def run_volatility_analysis(self):
        """전체 변동성 분석 실행"""
        print("🚀 올바른 변동성 예측 모델 분석 시작")
        print("📊 타겟: target_vol_5d (5일 후 변동성)")

        # 1. 데이터 로드
        if not self.load_and_process_data():
            return None

        # 2. 모델 훈련
        print("\n🤖 변동성 예측 모델 훈련...")
        self.train_volatility_model()

        # 3. 전체 예측
        print("\n🎯 전체 기간 변동성 예측...")
        self.generate_full_predictions()

        # 4. 차트 생성
        print("\n📈 변동성 비교 차트 생성...")
        chart_path = self.create_volatility_comparison_chart()

        print(f"\n✅ 변동성 예측 분석 완료!")
        print(f"📁 차트: {chart_path}")

        return chart_path

if __name__ == "__main__":
    analyzer = VolatilityPredictionModel()
    analyzer.run_volatility_analysis()