#!/usr/bin/env python3
"""
🎯 안정적인 회귀 최적화 시스템

핵심 기능에 집중한 안정적인 구현
MAE, 방향 정확도, Log Loss, MDD, Sharpe 지표 중심
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, log_loss
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleLSTMRegressor(nn.Module):
    """단순하고 안정적인 LSTM 회귀 모델"""

    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])  # 마지막 timestep만 사용
        return self.fc(out).squeeze()

class StableRegressionOptimizer:
    """안정적인 회귀 최적화 시스템"""

    def __init__(self):
        self.data_processor = DataProcessor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 안정적인 회귀 시스템 초기화 (디바이스: {self.device})")

    def calculate_comprehensive_metrics(self, y_true, y_pred, returns_true=None):
        """포괄적인 평가 지표 계산"""

        # 기본 지표
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # 방향 정확도 계산
        if len(y_true) > 1:
            direction_true = np.sign(np.diff(y_true))
            direction_pred = np.sign(np.diff(y_pred))
            direction_accuracy = np.mean(direction_true == direction_pred)
        else:
            direction_accuracy = 0.5

        # Log Loss 계산 (방향 기반)
        try:
            if len(y_true) > 1:
                prob_true = (direction_true + 1) / 2  # -1,1 -> 0,1
                prob_pred = (direction_pred + 1) / 2
                prob_pred = np.clip(prob_pred, 1e-15, 1-1e-15)
                log_loss_val = log_loss(prob_true, prob_pred)
            else:
                log_loss_val = np.nan
        except:
            log_loss_val = np.nan

        # 포트폴리오 성과 지표
        sharpe_ratio = 0
        mdd = 0

        if returns_true is not None and len(returns_true) > 1:
            try:
                # 예측 기반 포지션 생성
                positions = np.sign(y_pred[:-1])  # 마지막 제외
                portfolio_returns = returns_true[1:] * positions  # 다음날 수익률

                if len(portfolio_returns) > 0 and np.std(portfolio_returns) > 0:
                    sharpe_ratio = np.mean(portfolio_returns) / np.std(portfolio_returns) * np.sqrt(252)

                    # MDD 계산
                    cumulative = np.cumprod(1 + portfolio_returns)
                    running_max = np.maximum.accumulate(cumulative)
                    drawdown = (cumulative - running_max) / running_max
                    mdd = np.min(drawdown)
                else:
                    sharpe_ratio = 0
                    mdd = 0
            except:
                sharpe_ratio = 0
                mdd = 0

        return {
            'MAE': mae,
            'MSE': mse,
            'R2': r2,
            'direction_accuracy': direction_accuracy,
            'log_loss': log_loss_val if not np.isnan(log_loss_val) else 0,
            'sharpe_ratio': sharpe_ratio,
            'MDD': mdd
        }

    def prepare_stable_data(self, data_path, target_type='returns', sequence_length=20):
        """안정적인 데이터 준비"""
        print(f"📊 안정적인 데이터 준비 중... (target: {target_type})")

        # 데이터 로딩
        df = self.data_processor.load_and_validate_data(data_path)
        if df is None:
            raise ValueError("데이터 로딩 실패")

        # 수익률 계산
        df['returns'] = df['Close'].pct_change()

        # 타겟 설정
        if target_type == 'returns':
            df['target'] = df['returns'].shift(-1)
        elif target_type == 'price':
            df['target'] = df['Close'].shift(-1)
        elif target_type == 'volatility':
            df['volatility'] = df['returns'].rolling(20).std()
            df['target'] = df['volatility'].shift(-1)

        # 시퀀스 데이터 준비
        X_seq, _, scaler = self.data_processor.prepare_sequence_data(
            df, sequence_length, 'direction'
        )

        # 타겟 추출 (안정적인 방식)
        start_idx = sequence_length
        targets = []
        returns_vals = []

        for i in range(len(X_seq)):
            idx = start_idx + i
            if idx < len(df) and not pd.isna(df['target'].iloc[idx]):
                targets.append(df['target'].iloc[idx])
                return_val = df['returns'].iloc[idx] if not pd.isna(df['returns'].iloc[idx]) else 0
                returns_vals.append(return_val)

        # 길이 맞추기
        min_len = min(len(X_seq), len(targets))
        X_seq = X_seq[:min_len]
        targets = np.array(targets[:min_len])
        returns_vals = np.array(returns_vals[:min_len])

        # NaN 처리
        X_seq = np.nan_to_num(X_seq, nan=0.0, posinf=0.0, neginf=0.0)
        targets = np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
        returns_vals = np.nan_to_num(returns_vals, nan=0.0, posinf=0.0, neginf=0.0)

        print(f"   ✅ 안정적인 데이터 준비 완료: X={X_seq.shape}, y={targets.shape}")

        return X_seq, targets, returns_vals

    def train_pytorch_model_stable(self, model, X_train, y_train, X_val, y_val, epochs=30):
        """안정적인 PyTorch 모델 훈련"""
        model = model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = nn.MSELoss()

        # 데이터 정규화
        y_mean, y_std = np.mean(y_train), np.std(y_train)
        y_std = max(y_std, 1e-8)
        y_train_norm = (y_train - y_mean) / y_std

        # 텐서 변환
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train_norm).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)

        # 훈련
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # 예측
        model.eval()
        with torch.no_grad():
            y_pred_norm = model(X_val_tensor).cpu().numpy()
            y_pred = y_pred_norm * y_std + y_mean

        return y_pred

    def run_stable_experiment(self, X, y, returns_vals, n_splits=3):
        """안정적인 실험 실행"""
        print(f"\n🔬 안정적인 {n_splits}-Fold 교차 검증 실험")

        tscv = TimeSeriesSplit(n_splits=n_splits)
        all_results = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            print(f"\n📊 Fold {fold+1}/{n_splits}")

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            returns_test = returns_vals[test_idx] if len(returns_vals) > max(test_idx) else returns_vals[test_idx[test_idx < len(returns_vals)]]

            # 길이 조정
            min_test_len = min(len(X_test), len(y_test), len(returns_test))
            X_test = X_test[:min_test_len]
            y_test = y_test[:min_test_len]
            returns_test = returns_test[:min_test_len]

            fold_predictions = {}

            # 1. LSTM 모델
            print("   🧠 LSTM 훈련...")
            try:
                lstm_model = SimpleLSTMRegressor(input_size=X.shape[-1])
                lstm_pred = self.train_pytorch_model_stable(
                    lstm_model, X_train, y_train, X_test, y_test
                )
                fold_predictions['LSTM'] = lstm_pred
            except Exception as e:
                print(f"      ⚠️ LSTM 오류: {str(e)}")
                fold_predictions['LSTM'] = np.zeros(len(y_test))

            # 2. Random Forest
            print("   🌲 Random Forest 훈련...")
            try:
                rf_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
                X_train_2d = X_train.reshape(X_train.shape[0], -1)
                X_test_2d = X_test.reshape(X_test.shape[0], -1)
                rf_model.fit(X_train_2d, y_train)
                rf_pred = rf_model.predict(X_test_2d)
                fold_predictions['RandomForest'] = rf_pred
            except Exception as e:
                print(f"      ⚠️ RF 오류: {str(e)}")
                fold_predictions['RandomForest'] = np.zeros(len(y_test))

            # 3. ElasticNet
            print("   📊 ElasticNet 훈련...")
            try:
                elastic_model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000)
                X_train_2d = X_train[:, -1, :]  # 마지막 timestep만 사용
                X_test_2d = X_test[:, -1, :]
                elastic_model.fit(X_train_2d, y_train)
                elastic_pred = elastic_model.predict(X_test_2d)
                fold_predictions['ElasticNet'] = elastic_pred
            except Exception as e:
                print(f"      ⚠️ ElasticNet 오류: {str(e)}")
                fold_predictions['ElasticNet'] = np.zeros(len(y_test))

            # 4. Ridge
            print("   📈 Ridge 훈련...")
            try:
                ridge_model = Ridge(alpha=1.0)
                X_train_2d = X_train[:, -1, :]
                X_test_2d = X_test[:, -1, :]
                ridge_model.fit(X_train_2d, y_train)
                ridge_pred = ridge_model.predict(X_test_2d)
                fold_predictions['Ridge'] = ridge_pred
            except Exception as e:
                print(f"      ⚠️ Ridge 오류: {str(e)}")
                fold_predictions['Ridge'] = np.zeros(len(y_test))

            # 5. 앙상블 (단순 평균)
            print("   🎯 앙상블...")
            valid_predictions = [pred for pred in fold_predictions.values() if len(pred) == len(y_test)]
            if valid_predictions:
                ensemble_pred = np.mean(valid_predictions, axis=0)
                fold_predictions['Ensemble'] = ensemble_pred

            # 결과 평가
            fold_results = {}
            for name, pred in fold_predictions.items():
                if len(pred) == len(y_test):
                    metrics = self.calculate_comprehensive_metrics(y_test, pred, returns_test)
                    fold_results[name] = metrics
                    print(f"      {name}: MAE={metrics['MAE']:.6f}, "
                          f"방향정확도={metrics['direction_accuracy']:.4f}, "
                          f"Sharpe={metrics['sharpe_ratio']:.4f}")

            all_results.append(fold_results)
            print(f"   ✅ Fold {fold+1} 완료")

        return all_results

    def run_stable_optimization(self, data_path='/root/workspace/data/training/sp500_2020_2024_enhanced.csv'):
        """안정적인 최적화 실행"""
        print("🎯 안정적인 회귀 최적화 시스템 시작")
        print("="*60)

        start_time = time.time()

        try:
            # 수익률 예측 실험
            print("\n🎯 실험 1: 수익률 예측")
            X, y, returns_vals = self.prepare_stable_data(
                data_path, target_type='returns', sequence_length=20
            )

            returns_results = self.run_stable_experiment(X, y, returns_vals, n_splits=3)

            # 시간이 허용하면 변동성 예측도 수행
            elapsed = time.time() - start_time
            if elapsed < 180:  # 3분 이내
                print("\n🎯 실험 2: 변동성 예측")
                X2, y2, returns_vals2 = self.prepare_stable_data(
                    data_path, target_type='volatility', sequence_length=15
                )
                volatility_results = self.run_stable_experiment(X2, y2, returns_vals2, n_splits=3)
            else:
                print("\n⏰ 시간 제한으로 변동성 실험 건너뛰기")
                volatility_results = None

        except Exception as e:
            print(f"❌ 실험 중 오류: {str(e)}")
            return None

        # 결과 종합
        total_time = time.time() - start_time

        # 평균 성능 계산
        model_names = ['LSTM', 'RandomForest', 'ElasticNet', 'Ridge', 'Ensemble']
        avg_performance = {}

        for model_name in model_names:
            avg_performance[model_name] = {}
            metrics = ['MAE', 'R2', 'direction_accuracy', 'sharpe_ratio', 'MDD']

            for metric in metrics:
                values = []
                for fold_result in returns_results:
                    if model_name in fold_result and metric in fold_result[model_name]:
                        val = fold_result[model_name][metric]
                        if not (np.isnan(val) or np.isinf(val)):
                            values.append(val)

                avg_performance[model_name][metric] = np.mean(values) if values else 0

        # 결과 출력
        print(f"\n📊 안정적인 최적화 결과 요약:")
        print(f"   총 소요시간: {total_time:.1f}초")
        print(f"\n🏆 모델별 평균 성능:")
        print("-" * 70)

        for model_name, performance in avg_performance.items():
            if any(abs(v) > 1e-10 for v in performance.values()):
                print(f"{model_name:15s}: "
                      f"MAE={performance['MAE']:.6f}, "
                      f"R²={performance['R2']:.4f}, "
                      f"방향정확도={performance['direction_accuracy']:.4f}, "
                      f"Sharpe={performance['sharpe_ratio']:.4f}")

        # 최고 성능 모델
        valid_models = {k: v for k, v in avg_performance.items() if any(abs(val) > 1e-10 for val in v.values())}

        if valid_models:
            best_r2_model = max(valid_models.items(), key=lambda x: x[1]['R2'])
            best_sharpe_model = max(valid_models.items(), key=lambda x: x[1]['sharpe_ratio'])
            best_direction_model = max(valid_models.items(), key=lambda x: x[1]['direction_accuracy'])
            best_mae_model = min(valid_models.items(), key=lambda x: x[1]['MAE'])

            print(f"\n🥇 최고 성능:")
            print(f"   MAE 최고: {best_mae_model[0]} ({best_mae_model[1]['MAE']:.6f})")
            print(f"   R² 최고: {best_r2_model[0]} ({best_r2_model[1]['R2']:.4f})")
            print(f"   방향정확도 최고: {best_direction_model[0]} ({best_direction_model[1]['direction_accuracy']:.4f})")
            print(f"   Sharpe 최고: {best_sharpe_model[0]} ({best_sharpe_model[1]['sharpe_ratio']:.4f})")

        # 결과 저장
        final_result = {
            'timestamp': datetime.now().isoformat(),
            'experiment_type': 'stable_regression_optimization',
            'methodology': {
                'models': model_names,
                'validation': 'TimeSeriesSplit Cross-Validation',
                'metrics': ['MAE', 'R2', 'direction_accuracy', 'log_loss', 'sharpe_ratio', 'MDD']
            },
            'total_time_seconds': total_time,
            'average_performance': avg_performance,
            'detailed_results': {
                'returns_prediction': returns_results,
                'volatility_prediction': volatility_results
            }
        }

        output_path = f"/root/workspace/data/results/stable_regression_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, 'w') as f:
            json.dump(final_result, f, indent=2, default=str)

        print(f"\n💾 결과 저장: {output_path}")
        print(f"🏁 안정적인 최적화 완료 ({total_time:.1f}초)")

        return final_result

def main():
    """메인 실행"""
    optimizer = StableRegressionOptimizer()
    results = optimizer.run_stable_optimization()
    return results

if __name__ == "__main__":
    main()