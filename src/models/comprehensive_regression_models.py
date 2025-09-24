#!/usr/bin/env python3
"""
🚀 종합 회귀 모델 학습 시스템

SVR, KNN, Gradient Boosting, XGBoost, LightGBM 등 다양한 회귀모델 비교
"""

import sys
sys.path.append('/root/workspace/src')

import pandas as pd
import numpy as np
from datetime import datetime
import json
import time
import warnings
warnings.filterwarnings('ignore')

# Core imports
from core.data_processor import DataProcessor

# ML imports
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet, Ridge, Lasso, BayesianRidge
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import catboost as cb

# Deep learning
import torch
import torch.nn as nn
import torch.optim as optim

class ComprehensiveRegressionModels:
    """종합 회귀 모델 시스템"""

    def __init__(self):
        self.data_processor = DataProcessor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"🚀 종합 회귀 모델 시스템 초기화 (디바이스: {self.device})")

    def prepare_regression_data(self, df, target='returns'):
        """회귀 데이터 준비"""
        print(f"📊 회귀 데이터 준비 중... (target: {target})")

        # 타겟 생성
        if target == 'returns':
            df['target'] = df['Close'].pct_change()
        elif target == 'volatility':
            df['target'] = df['Close'].pct_change().rolling(20).std()
        elif target == 'log_returns':
            df['target'] = np.log(df['Close'] / df['Close'].shift(1))
        else:
            df['target'] = df[target]

        # 특성 선택
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_columns = ['Date', 'target', 'Open', 'High', 'Low', 'Close', 'Volume']
        feature_columns = [col for col in numeric_features if col not in exclude_columns]

        # 데이터 추출 및 정제
        X = df[feature_columns].values
        y = df['target'].values

        # 안전한 전처리
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        # 극값 제거
        valid_idx = (np.abs(y) < 0.5) & ~np.isnan(y) & ~np.isinf(y)
        X = X[valid_idx]
        y = y[valid_idx]

        print(f"   ✅ 데이터 준비 완료: X={X.shape}, y={y.shape}")
        return X, y, feature_columns

    def get_regression_models(self):
        """회귀 모델 딕셔너리 반환"""
        models = {
            # 선형 모델들
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
            'Ridge': Ridge(alpha=1.0, random_state=42),
            'Lasso': Lasso(alpha=0.1, random_state=42),
            'BayesianRidge': BayesianRidge(),

            # 트리 기반 모델들
            'RandomForest': RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=100, max_depth=6, random_state=42
            ),

            # 고급 그래디언트 부스팅
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=42, verbosity=0
            ),
            'CatBoost': cb.CatBoostRegressor(
                iterations=100, depth=6, learning_rate=0.1,
                random_seed=42, verbose=False
            ),

            # 인스턴스 기반 모델들
            'KNN': KNeighborsRegressor(n_neighbors=5),
            'SVR_linear': SVR(kernel='linear', C=1.0),
            'SVR_rbf': SVR(kernel='rbf', C=1.0, gamma='scale'),
        }

        return models

    class DeepRegressionModel(nn.Module):
        """딥러닝 회귀 모델"""

        def __init__(self, input_size, hidden_sizes=[128, 64, 32]):
            super().__init__()
            layers = []

            prev_size = input_size
            for hidden_size in hidden_sizes:
                layers.extend([
                    nn.Linear(prev_size, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.3)
                ])
                prev_size = hidden_size

            layers.append(nn.Linear(prev_size, 1))
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

    def train_deep_model(self, X_train, y_train, X_val, y_val):
        """딥러닝 모델 훈련"""
        model = self.DeepRegressionModel(X_train.shape[1]).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 텐서 변환
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)

        # 훈련
        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs.squeeze(), y_train_tensor)
            loss.backward()
            optimizer.step()

        # 예측
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_tensor).cpu().numpy().squeeze()

        return val_pred

    def run_comprehensive_comparison(self, data_path='/root/workspace/data/training/sp500_2020_2024_enhanced.csv'):
        """종합 회귀 모델 비교 실험"""
        print("🚀 종합 회귀 모델 비교 실험 시작")
        print("="*70)

        start_time = time.time()
        all_results = {}

        try:
            # 데이터 로딩
            df = self.data_processor.load_and_validate_data(data_path)

            # 실험 1: 수익률 예측
            print("\n🎯 실험 1: 수익률 예측")
            X_returns, y_returns, features = self.prepare_regression_data(df, target='returns')
            if X_returns is not None:
                returns_results = self._run_model_comparison(X_returns, y_returns, 'returns')
                all_results['returns'] = returns_results

            # 실험 2: 로그 수익률 예측
            print("\n🎯 실험 2: 로그 수익률 예측")
            X_log, y_log, _ = self.prepare_regression_data(df, target='log_returns')
            if X_log is not None:
                log_results = self._run_model_comparison(X_log, y_log, 'log_returns')
                all_results['log_returns'] = log_results

            # 실험 3: 변동성 예측
            print("\n🎯 실험 3: 변동성 예측")
            X_vol, y_vol, _ = self.prepare_regression_data(df, target='volatility')
            if X_vol is not None:
                vol_results = self._run_model_comparison(X_vol, y_vol, 'volatility')
                all_results['volatility'] = vol_results

            # 결과 정리
            total_time = time.time() - start_time
            self._summarize_comprehensive_results(all_results, total_time)

            return all_results

        except Exception as e:
            print(f"❌ 종합 실험 중 오류: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _run_model_comparison(self, X, y, target_type):
        """모델 비교 실행"""
        models = self.get_regression_models()
        tscv = TimeSeriesSplit(n_splits=3)
        model_results = {}

        # 모델별 결과 저장
        for model_name in models.keys():
            model_results[model_name] = {'mae': [], 'r2': [], 'direction_acc': []}

        # 딥러닝 모델 추가
        model_results['DeepNN'] = {'mae': [], 'r2': [], 'direction_acc': []}

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            print(f"\n📊 Fold {fold+1}/3")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # 데이터 전처리
            imputer = SimpleImputer(strategy='mean')
            X_train_clean = imputer.fit_transform(X_train)
            X_val_clean = imputer.transform(X_val)

            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train_clean)
            X_val_scaled = scaler.transform(X_val_clean)

            # 전통적 ML 모델들
            for model_name, model in models.items():
                try:
                    print(f"   🤖 {model_name} 훈련...")
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_val_scaled)

                    mae = mean_absolute_error(y_val, y_pred)
                    r2 = r2_score(y_val, y_pred)

                    # 방향 정확도 계산
                    direction_acc = self._calculate_direction_accuracy(y_val, y_pred)

                    model_results[model_name]['mae'].append(mae)
                    model_results[model_name]['r2'].append(r2)
                    model_results[model_name]['direction_acc'].append(direction_acc)

                    print(f"      MAE={mae:.6f}, R²={r2:.4f}, 방향정확도={direction_acc:.4f}")

                except Exception as e:
                    print(f"      {model_name} 오류: {e}")
                    model_results[model_name]['mae'].append(np.inf)
                    model_results[model_name]['r2'].append(-np.inf)
                    model_results[model_name]['direction_acc'].append(0.5)

            # 딥러닝 모델
            try:
                print(f"   🧠 DeepNN 훈련...")
                deep_pred = self.train_deep_model(X_train_scaled, y_train, X_val_scaled, y_val)

                deep_mae = mean_absolute_error(y_val, deep_pred)
                deep_r2 = r2_score(y_val, deep_pred)
                deep_direction = self._calculate_direction_accuracy(y_val, deep_pred)

                model_results['DeepNN']['mae'].append(deep_mae)
                model_results['DeepNN']['r2'].append(deep_r2)
                model_results['DeepNN']['direction_acc'].append(deep_direction)

                print(f"      MAE={deep_mae:.6f}, R²={deep_r2:.4f}, 방향정확도={deep_direction:.4f}")

            except Exception as e:
                print(f"      DeepNN 오류: {e}")
                model_results['DeepNN']['mae'].append(np.inf)
                model_results['DeepNN']['r2'].append(-np.inf)
                model_results['DeepNN']['direction_acc'].append(0.5)

            print(f"   ✅ Fold {fold+1} 완료")

        return model_results

    def _calculate_direction_accuracy(self, y_true, y_pred):
        """방향 정확도 계산"""
        if len(y_true) < 2:
            return 0.5

        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0

        if len(true_direction) == 0:
            return 0.5

        return np.mean(true_direction == pred_direction)

    def _summarize_comprehensive_results(self, all_results, total_time):
        """종합 결과 요약"""
        print(f"\n🚀 종합 회귀 모델 비교 결과 요약:")
        print(f"   총 소요시간: {total_time:.1f}초")
        print("="*70)

        for target_type, model_results in all_results.items():
            print(f"\n🎯 {target_type.upper()} 예측 결과:")
            print("-"*50)

            # 모델별 평균 성능 계산 및 순위
            model_performance = []

            for model_name, metrics in model_results.items():
                avg_mae = np.mean(metrics['mae'])
                avg_r2 = np.mean(metrics['r2'])
                avg_direction = np.mean(metrics['direction_acc'])

                model_performance.append({
                    'model': model_name,
                    'mae': avg_mae,
                    'r2': avg_r2,
                    'direction_acc': avg_direction
                })

            # MAE 기준 정렬 (낮을수록 좋음)
            model_performance.sort(key=lambda x: x['mae'])

            print(f"🏆 성능 순위 (MAE 기준):")
            for i, perf in enumerate(model_performance):
                rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1:2d}."
                print(f"   {rank_emoji} {perf['model']:15s}: MAE={perf['mae']:.6f}, R²={perf['r2']:+.4f}, 방향={perf['direction_acc']:.4f}")

            # 최고 성능 모델들
            best_mae = min(p['mae'] for p in model_performance if not np.isinf(p['mae']))
            best_r2 = max(p['r2'] for p in model_performance if not np.isinf(p['r2']))
            best_direction = max(p['direction_acc'] for p in model_performance)

            print(f"\n📊 최고 성능:")
            print(f"   MAE 최고: {best_mae:.6f}")
            print(f"   R² 최고: {best_r2:+.4f}")
            print(f"   방향정확도 최고: {best_direction:.4f}")

        # 결과 저장
        self._save_comprehensive_results(all_results, total_time)

    def _save_comprehensive_results(self, all_results, total_time):
        """결과 저장"""
        output_path = f"/root/workspace/data/results/comprehensive_regression_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        try:
            # JSON 직렬화 가능한 형태로 변환
            serializable_results = {}
            for target_type, model_results in all_results.items():
                serializable_results[target_type] = {}
                for model_name, metrics in model_results.items():
                    serializable_results[target_type][model_name] = {
                        'mae_mean': float(np.mean(metrics['mae'])),
                        'mae_std': float(np.std(metrics['mae'])),
                        'r2_mean': float(np.mean(metrics['r2'])),
                        'r2_std': float(np.std(metrics['r2'])),
                        'direction_acc_mean': float(np.mean(metrics['direction_acc'])),
                        'direction_acc_std': float(np.std(metrics['direction_acc']))
                    }

            with open(output_path, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'total_time': total_time,
                    'results': serializable_results
                }, f, indent=2)

            print(f"\n💾 결과 저장: {output_path}")

        except Exception as e:
            print(f"❌ 결과 저장 실패: {e}")

def main():
    """메인 실행"""
    system = ComprehensiveRegressionModels()
    results = system.run_comprehensive_comparison()

    print("\n🎉 종합 회귀 모델 비교 완료!")
    return results

if __name__ == "__main__":
    main()