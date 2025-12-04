#!/usr/bin/env python3
"""
뉴스 감성 지표 기반 주가 수익률 예측 모델

실험 목표: LLM 감성분석 지표를 사용해도 주가(수익률) 예측이 불가능함을 실증
비교: 변동성 예측(R²=0.31) vs 수익률 예측(R²≈0.00)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class NewsSentimentPricePrediction:
    """뉴스 감성 기반 주가 수익률 예측 모델"""

    def __init__(self, dataset_path="data/training/spy_news_sentiment_dataset.csv"):
        self.dataset_path = dataset_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = StandardScaler()
        self.results = {}

    def load_data(self):
        """데이터셋 로드 및 전처리"""
        print("📂 데이터셋 로드 중...")

        try:
            self.data = pd.read_csv(self.dataset_path, index_col=0, parse_dates=True)
            print(f"✅ 데이터 로드 완료: {self.data.shape}")
            print(f"   기간: {self.data.index.min()} ~ {self.data.index.max()}")

            # 타겟 변수 확인
            if 'target_return_1d' not in self.data.columns:
                raise ValueError("타겟 변수 'target_return_1d' 없음")

            # 결측치 제거
            self.data = self.data.dropna()
            print(f"   결측치 제거 후: {len(self.data)} 샘플")

            return True

        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            return False

    def prepare_features(self):
        """특성 준비 및 선택"""
        print("\n📊 특성 준비 중...")

        try:
            # 감성 관련 특성만 선택
            sentiment_features = [
                'sentiment_mean', 'sentiment_std', 'sentiment_min', 'sentiment_max',
                'news_count', 'sentiment_range', 'sentiment_ma_5', 'sentiment_ma_20',
                'sentiment_momentum', 'news_volume_ma_10', 'news_volume_ratio'
            ]

            # 가격 기반 특성 (비교용)
            price_features = [
                'returns', 'volatility_5d', 'volatility_20d'
            ]

            # 실험 1: 감성 특성만 사용
            available_sentiment = [f for f in sentiment_features if f in self.data.columns]
            available_price = [f for f in price_features if f in self.data.columns]

            print(f"   감성 특성: {len(available_sentiment)}개")
            print(f"   가격 특성: {len(available_price)}개")

            # 특성 조합
            all_features = available_sentiment + available_price

            # 특성과 타겟 분리
            X = self.data[all_features].values
            y = self.data['target_return_1d'].values

            print(f"   특성 행렬: {X.shape}")
            print(f"   타겟: {y.shape}")

            return X, y, all_features

        except Exception as e:
            print(f"❌ 특성 준비 실패: {e}")
            return None, None, None

    def create_train_test_split(self, X, y, test_size=0.2):
        """시간 순서 유지한 학습/테스트 분리"""
        print(f"\n✂️ 학습/테스트 분리 (시간 순서 유지)...")

        try:
            split_idx = int(len(X) * (1 - test_size))

            self.X_train = X[:split_idx]
            self.X_test = X[split_idx:]
            self.y_train = y[:split_idx]
            self.y_test = y[split_idx:]

            print(f"   학습 데이터: {self.X_train.shape}")
            print(f"   테스트 데이터: {self.X_test.shape}")

            return True

        except Exception as e:
            print(f"❌ 데이터 분리 실패: {e}")
            return False

    def train_model(self):
        """Ridge 회귀 모델 학습"""
        print("\n🤖 Ridge 회귀 모델 학습 중...")

        try:
            # 데이터 스케일링
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)

            # Ridge 모델 학습 (변동성 예측과 동일한 alpha)
            self.model = Ridge(alpha=1.0, random_state=42)
            self.model.fit(self.X_train_scaled, self.y_train)

            print(f"✅ 모델 학습 완료")
            print(f"   모델: Ridge(alpha=1.0)")
            print(f"   특성 수: {self.X_train.shape[1]}")

            return True

        except Exception as e:
            print(f"❌ 모델 학습 실패: {e}")
            return False

    def evaluate_model(self):
        """모델 성능 평가"""
        print("\n📈 모델 성능 평가 중...")

        try:
            # 예측
            y_train_pred = self.model.predict(self.X_train_scaled)
            y_test_pred = self.model.predict(self.X_test_scaled)

            # 학습 세트 성능
            train_r2 = r2_score(self.y_train, y_train_pred)
            train_mse = mean_squared_error(self.y_train, y_train_pred)
            train_mae = mean_absolute_error(self.y_train, y_train_pred)

            # 테스트 세트 성능
            test_r2 = r2_score(self.y_test, y_test_pred)
            test_mse = mean_squared_error(self.y_test, y_test_pred)
            test_mae = mean_absolute_error(self.y_test, y_test_pred)

            print(f"\n📊 학습 세트 성능:")
            print(f"   R² Score: {train_r2:.4f}")
            print(f"   MSE: {train_mse:.6f}")
            print(f"   RMSE: {np.sqrt(train_mse):.6f}")
            print(f"   MAE: {train_mae:.6f}")

            print(f"\n📊 테스트 세트 성능:")
            print(f"   R² Score: {test_r2:.4f}")
            print(f"   MSE: {test_mse:.6f}")
            print(f"   RMSE: {np.sqrt(test_mse):.6f}")
            print(f"   MAE: {test_mae:.6f}")

            # 결과 저장
            self.results = {
                'model': 'Ridge(alpha=1.0)',
                'target': 'target_return_1d',
                'n_samples_train': len(self.y_train),
                'n_samples_test': len(self.y_test),
                'n_features': self.X_train.shape[1],
                'train_metrics': {
                    'r2_score': float(train_r2),
                    'mse': float(train_mse),
                    'rmse': float(np.sqrt(train_mse)),
                    'mae': float(train_mae)
                },
                'test_metrics': {
                    'r2_score': float(test_r2),
                    'mse': float(test_mse),
                    'rmse': float(np.sqrt(test_mse)),
                    'mae': float(test_mae)
                }
            }

            return True

        except Exception as e:
            print(f"❌ 성능 평가 실패: {e}")
            return False

    def compare_with_volatility_model(self):
        """변동성 예측 모델과 비교"""
        print("\n🔍 변동성 예측 모델과 비교 분석...")

        try:
            # 기존 변동성 모델 성능 (데이터에서 확인)
            volatility_model_r2 = 0.3113  # src/models/correct_target_design.py 결과

            price_model_r2 = self.results['test_metrics']['r2_score']

            print(f"\n📊 모델 성능 비교:")
            print(f"   변동성 예측 (기존): R² = {volatility_model_r2:.4f}")
            print(f"   수익률 예측 (본 실험): R² = {price_model_r2:.4f}")
            print(f"   성능 차이: {volatility_model_r2 - price_model_r2:.4f}")

            if price_model_r2 < 0.05:
                print(f"\n✅ 결론: LLM 감성 지표로 주가 수익률 예측 불가능 (R² ≈ {price_model_r2:.4f})")
                print(f"   변동성 예측(R²=0.31)과 극명한 대조")
            elif price_model_r2 < 0.1:
                print(f"\n⚠️  결론: 매우 낮은 예측력 (R² = {price_model_r2:.4f})")
            else:
                print(f"\n⚠️  경고: 예상보다 높은 R² ({price_model_r2:.4f}) - 데이터 누출 의심")

            self.results['comparison'] = {
                'volatility_model_r2': volatility_model_r2,
                'price_model_r2': price_model_r2,
                'difference': volatility_model_r2 - price_model_r2
            }

            return True

        except Exception as e:
            print(f"❌ 비교 분석 실패: {e}")
            return False

    def run_har_benchmark(self, X, y):
        """HAR 모델 벤치마크 (간단한 autoregressive)"""
        print("\n🔬 HAR 벤치마크 모델 실행...")

        try:
            # HAR 특성: 과거 수익률 lag
            har_features = []
            returns_series = pd.Series(y, index=range(len(y)))

            for lag in [1, 5, 22]:  # 1일, 1주, 1달
                har_features.append(returns_series.shift(lag).fillna(0).values)

            X_har = np.column_stack(har_features)

            # 학습/테스트 분리
            split_idx = int(len(X_har) * 0.8)
            X_har_train = X_har[:split_idx]
            X_har_test = X_har[split_idx:]
            y_har_train = y[:split_idx]
            y_har_test = y[split_idx:]

            # HAR 모델 학습
            har_model = Ridge(alpha=1.0, random_state=42)
            har_model.fit(X_har_train, y_har_train)

            # 예측 및 평가
            y_har_pred = har_model.predict(X_har_test)
            har_r2 = r2_score(y_har_test, y_har_pred)

            print(f"   HAR 벤치마크 R²: {har_r2:.4f}")

            self.results['har_benchmark'] = {
                'r2_score': float(har_r2),
                'features': '1-day, 5-day, 22-day lags'
            }

            return True

        except Exception as e:
            print(f"❌ HAR 벤치마크 실패: {e}")
            return False

    def save_results(self):
        """결과 저장"""
        try:
            # 메타데이터 추가
            self.results['metadata'] = {
                'experiment': 'news_sentiment_price_prediction',
                'hypothesis': 'LLM 감성 지표로 주가 수익률 예측 불가능',
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'dataset': self.dataset_path
            }

            # JSON 저장
            output_path = "data/raw/news_sentiment_price_prediction_results.json"
            with open(output_path, 'w') as f:
                json.dump(self.results, f, indent=2)

            print(f"\n💾 결과 저장: {output_path}")

            return True

        except Exception as e:
            print(f"❌ 결과 저장 실패: {e}")
            return False

    def run_experiment(self):
        """전체 실험 실행"""
        print("="*60)
        print("🧪 뉴스 감성 지표 기반 주가 예측 실험")
        print("="*60)
        print("가설: LLM 감성분석으로 주가 수익률 예측 불가능\n")

        steps = [
            ("데이터 로드", self.load_data),
        ]

        for step_name, step_func in steps:
            if not step_func():
                print(f"\n❌ {step_name} 실패 - 실험 중단")
                return False

        # 특성 준비
        X, y, features = self.prepare_features()
        if X is None:
            return False

        # 학습/테스트 분리
        if not self.create_train_test_split(X, y):
            return False

        # 모델 학습
        if not self.train_model():
            return False

        # 성능 평가
        if not self.evaluate_model():
            return False

        # HAR 벤치마크
        self.run_har_benchmark(X, y)

        # 변동성 모델과 비교
        self.compare_with_volatility_model()

        # 결과 저장
        self.save_results()

        print("\n" + "="*60)
        print("✅ 실험 완료!")
        print("="*60)

        return True

if __name__ == "__main__":
    experiment = NewsSentimentPricePrediction(
        dataset_path="data/training/spy_news_sentiment_dataset.csv"
    )

    experiment.run_experiment()
