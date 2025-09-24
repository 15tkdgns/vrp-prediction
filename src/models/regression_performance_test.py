#!/usr/bin/env python3
"""
📈 회귀 기반 성능 테스트

평가지표: MAE, 방향 정확도, Log Loss, MDD, Sharpe Ratio
가격 예측 및 금융 지표 기반 종합 평가
"""

import sys
sys.path.append('/root/workspace/src')

import pandas as pd
import numpy as np
from datetime import datetime
import json
import time
from pathlib import Path

# Core imports
from core.data_processor import DataProcessor
from training.model_trainer import ModelTrainer

# 평가 지표 imports
from sklearn.metrics import mean_absolute_error, log_loss
from sklearn.model_selection import TimeSeriesSplit
import torch
import torch.nn as nn

class RegressionPerformanceTester:
    """회귀 기반 성능 테스트 시스템"""

    def __init__(self):
        self.data_processor = DataProcessor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 디바이스: {self.device}")

    def calculate_financial_metrics(self, y_true, y_pred, returns_true, returns_pred):
        """금융 지표 계산"""
        metrics = {}

        # 1. MAE (Mean Absolute Error)
        metrics['MAE'] = mean_absolute_error(y_true, y_pred)

        # 2. 방향 정확도
        direction_true = np.sign(returns_true)
        direction_pred = np.sign(returns_pred)
        direction_accuracy = np.mean(direction_true == direction_pred)
        metrics['direction_accuracy'] = direction_accuracy

        # 3. Log Loss (확률 기반)
        # 방향을 확률로 변환
        prob_true = (direction_true + 1) / 2  # -1,1 -> 0,1
        prob_pred = (direction_pred + 1) / 2
        # 극값 처리
        prob_pred = np.clip(prob_pred, 1e-15, 1-1e-15)
        try:
            metrics['log_loss'] = log_loss(prob_true, prob_pred)
        except:
            metrics['log_loss'] = np.nan

        # 4. MDD (Maximum Drawdown)
        cumulative_returns = np.cumprod(1 + returns_pred)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        metrics['MDD'] = np.min(drawdown)

        # 5. Sharpe Ratio
        if np.std(returns_pred) > 0:
            metrics['sharpe_ratio'] = np.mean(returns_pred) / np.std(returns_pred) * np.sqrt(252)
        else:
            metrics['sharpe_ratio'] = 0

        return metrics

    def prepare_regression_data(self, data_path, target_type='price', sequence_length=20):
        """회귀용 데이터 준비"""
        print(f"📊 회귀 데이터 준비 중... (target: {target_type})")

        # 기존 검증된 데이터 처리 방식 사용
        df = self.data_processor.load_and_validate_data(data_path)
        if df is None:
            raise ValueError("데이터 로딩 실패")

        # 수익률 계산
        df['returns'] = df['Close'].pct_change()

        # 타겟 변수 설정 (미래 정보 누출 방지)
        if target_type == 'price':
            # 다음날 종가를 타겟으로 (현재는 분류용 direction만 있으므로 Close 사용)
            df['target'] = df['Close'].shift(-1)
        elif target_type == 'returns':
            # 다음날 수익률을 타겟으로
            df['target'] = df['returns'].shift(-1)
        else:
            raise ValueError(f"Unsupported target_type: {target_type}")

        # 기존 시퀀스 데이터 준비 방식 활용
        X_seq, y_seq, scaler = self.data_processor.prepare_sequence_data(
            df, sequence_length, 'direction'  # 기존 방식 활용
        )

        # 회귀용 타겟 생성
        y_reg = []
        returns_vals = []

        # 시퀀스와 매칭되는 타겟 값들 추출
        start_idx = sequence_length
        for i in range(len(y_seq)):
            idx = start_idx + i
            if idx < len(df) and not pd.isna(df['target'].iloc[idx]):
                if target_type == 'price':
                    y_reg.append(df['target'].iloc[idx])
                    returns_vals.append(df['returns'].iloc[idx+1] if idx+1 < len(df) else 0)
                elif target_type == 'returns':
                    y_reg.append(df['target'].iloc[idx])
                    returns_vals.append(df['target'].iloc[idx])

        # 길이 맞추기
        min_len = min(len(X_seq), len(y_reg))
        X_seq = X_seq[:min_len]
        y_reg = np.array(y_reg[:min_len])
        returns_vals = np.array(returns_vals[:min_len])

        print(f"   ✅ 시퀀스 데이터: X={X_seq.shape}, y={y_reg.shape}")
        print(f"   📈 타겟 범위: {np.min(y_reg):.4f} ~ {np.max(y_reg):.4f}")

        return X_seq, y_reg, returns_vals, None

    def create_lstm_regressor(self, input_size, hidden_size=128, num_layers=3, dropout=0.3):
        """LSTM 회귀 모델 생성"""
        class LSTMRegressor(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    batch_first=True,
                    bidirectional=True
                )
                self.attention = nn.MultiheadAttention(
                    embed_dim=hidden_size*2,
                    num_heads=8,
                    dropout=dropout,
                    batch_first=True
                )
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(hidden_size*2, 1)

            def forward(self, x):
                lstm_out, _ = self.lstm(x)

                # Attention
                attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

                # Global average pooling
                pooled = torch.mean(attn_out, dim=1)

                # 최종 출력
                output = self.fc(self.dropout(pooled))
                return output.squeeze()

        return LSTMRegressor(input_size, hidden_size, num_layers, dropout)

    def train_and_evaluate_model(self, X, y, returns, model_name="LSTM_Regressor", n_splits=3):
        """모델 훈련 및 평가"""
        print(f"\n🔧 {model_name} 훈련 및 평가...")

        tscv = TimeSeriesSplit(n_splits=n_splits)
        all_metrics = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            print(f"   📊 Fold {fold+1}/{n_splits}")

            # 데이터 분할
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            returns_train, returns_val = returns[train_idx], returns[val_idx]

            # NaN 값 처리
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
            X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
            y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)
            y_val = np.nan_to_num(y_val, nan=0.0, posinf=0.0, neginf=0.0)

            # 데이터 정규화 (robust scaling)
            y_mean, y_std = np.mean(y_train), np.std(y_train)
            if y_std == 0:
                y_std = 1.0
            y_train_norm = (y_train - y_mean) / y_std
            y_val_norm = (y_val - y_mean) / y_std

            # 특성 정규화
            X_train_flat = X_train.reshape(-1, X_train.shape[-1])
            X_mean, X_std = np.mean(X_train_flat, axis=0), np.std(X_train_flat, axis=0)
            X_std[X_std == 0] = 1.0

            X_train = (X_train - X_mean) / X_std
            X_val = (X_val - X_mean) / X_std

            # 텐서 변환
            X_train_tensor = torch.FloatTensor(X_train).to(self.device)
            y_train_tensor = torch.FloatTensor(y_train_norm).to(self.device)
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)

            # 모델 생성
            model = self.create_lstm_regressor(X.shape[2]).to(self.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()

            # 훈련
            model.train()
            for epoch in range(50):
                optimizer.zero_grad()
                outputs = model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()

                if epoch % 10 == 0:
                    print(f"      Epoch {epoch}: Loss={loss.item():.6f}")

            # 평가
            model.eval()
            with torch.no_grad():
                y_pred_norm = model(X_val_tensor).cpu().numpy()
                y_pred = y_pred_norm * y_std + y_mean

                # 예측 수익률 계산
                if len(y_val) > 1:
                    returns_pred = np.diff(y_pred) / y_pred[:-1]
                    returns_val_aligned = returns_val[1:]
                else:
                    returns_pred = np.array([0])
                    returns_val_aligned = np.array([0])

                # 지표 계산
                metrics = self.calculate_financial_metrics(
                    y_val, y_pred, returns_val_aligned, returns_pred
                )
                metrics['fold'] = fold + 1
                all_metrics.append(metrics)

                print(f"      MAE: {metrics['MAE']:.6f}")
                print(f"      방향정확도: {metrics['direction_accuracy']:.4f}")
                print(f"      Sharpe: {metrics['sharpe_ratio']:.4f}")

        # 평균 성능 계산
        avg_metrics = {}
        for key in ['MAE', 'direction_accuracy', 'log_loss', 'MDD', 'sharpe_ratio']:
            values = [m[key] for m in all_metrics if not np.isnan(m[key])]
            avg_metrics[key] = np.mean(values) if values else np.nan

        return avg_metrics, all_metrics

    def run_regression_test(self, data_path='/root/workspace/data/training/sp500_2020_2024_enhanced.csv'):
        """회귀 테스트 실행"""
        print("📈 회귀 기반 성능 테스트 시작")
        print("="*60)

        start_time = time.time()
        results = {}

        try:
            # 1. 가격 예측 모델
            print("\n🎯 테스트 1: 가격 예측 (Price Regression)")
            X, y, returns, features = self.prepare_regression_data(
                data_path, target_type='price', sequence_length=20
            )

            price_metrics, price_folds = self.train_and_evaluate_model(
                X, y, returns, "Price_LSTM_Regressor"
            )
            results['price_prediction'] = {
                'avg_metrics': price_metrics,
                'fold_details': price_folds
            }

            # 시간 체크
            elapsed = time.time() - start_time
            if elapsed > 480:  # 8분 경과
                print("⏰ 시간 제한 근접, 남은 테스트 건너뛰기")
            else:
                # 2. 수익률 예측 모델
                print("\n🎯 테스트 2: 수익률 예측 (Returns Regression)")
                X2, y2, returns2, _ = self.prepare_regression_data(
                    data_path, target_type='returns', sequence_length=15
                )

                returns_metrics, returns_folds = self.train_and_evaluate_model(
                    X2, y2, returns2, "Returns_LSTM_Regressor"
                )
                results['returns_prediction'] = {
                    'avg_metrics': returns_metrics,
                    'fold_details': returns_folds
                }

        except Exception as e:
            print(f"❌ 테스트 중 오류: {str(e)}")

        # 결과 요약
        total_time = time.time() - start_time
        print(f"\n📊 회귀 테스트 결과 요약:")
        print(f"   총 소요시간: {total_time:.1f}초")

        if 'price_prediction' in results:
            price_perf = results['price_prediction']['avg_metrics']
            print(f"\n   🏆 가격 예측 성능:")
            print(f"      MAE: {price_perf['MAE']:.6f}")
            print(f"      방향 정확도: {price_perf['direction_accuracy']:.4f} ({price_perf['direction_accuracy']*100:.2f}%)")
            print(f"      Log Loss: {price_perf['log_loss']:.6f}")
            print(f"      MDD: {price_perf['MDD']:.6f}")
            print(f"      Sharpe Ratio: {price_perf['sharpe_ratio']:.4f}")

        if 'returns_prediction' in results:
            returns_perf = results['returns_prediction']['avg_metrics']
            print(f"\n   🏆 수익률 예측 성능:")
            print(f"      MAE: {returns_perf['MAE']:.6f}")
            print(f"      방향 정확도: {returns_perf['direction_accuracy']:.4f} ({returns_perf['direction_accuracy']*100:.2f}%)")
            print(f"      Log Loss: {returns_perf['log_loss']:.6f}")
            print(f"      MDD: {returns_perf['MDD']:.6f}")
            print(f"      Sharpe Ratio: {returns_perf['sharpe_ratio']:.4f}")

        # 결과 저장
        final_result = {
            'timestamp': datetime.now().isoformat(),
            'experiment_type': 'regression_performance_test',
            'total_time_seconds': total_time,
            'evaluation_metrics': ['MAE', 'direction_accuracy', 'log_loss', 'MDD', 'sharpe_ratio'],
            'results': results
        }

        output_path = f"/root/workspace/data/results/regression_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, 'w') as f:
            json.dump(final_result, f, indent=2, default=str)

        print(f"\n💾 결과 저장: {output_path}")
        print(f"🏁 회귀 테스트 완료 ({total_time:.1f}초)")

        return final_result

def main():
    """메인 실행"""
    tester = RegressionPerformanceTester()
    results = tester.run_regression_test()
    return results

if __name__ == "__main__":
    main()