#!/usr/bin/env python3
"""
🔧 수정된 고급 회귀 최적화 시스템

NaN 처리 문제 해결 및 안정적인 회귀 모델 학습
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
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim

class FixedAdvancedRegressionOptimizer:
    """수정된 고급 회귀 최적화 시스템"""

    def __init__(self):
        self.data_processor = DataProcessor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"🔧 수정된 고급 최적화 시스템 초기화 (디바이스: {self.device})")

    def prepare_safe_data(self, df, target='returns'):
        """안전한 데이터 준비 (NaN 완전 제거)"""
        print(f"🛡️ 안전한 데이터 준비 중... (target: {target})")

        # 기본 특성 생성
        if target == 'returns':
            df['target'] = df['Close'].pct_change()
        elif target == 'volatility':
            df['target'] = df['Close'].pct_change().rolling(20).std()
        else:
            df['target'] = df[target]

        # 숫자형 특성만 선택
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_columns = ['Date', 'target', 'Open', 'High', 'Low', 'Close', 'Volume']
        feature_columns = [col for col in numeric_features if col not in exclude_columns]

        # 데이터 추출
        X = df[feature_columns].values
        y = df['target'].values

        # 1단계: 무한값 처리
        X = np.where(np.isinf(X), 0, X)
        y = np.where(np.isinf(y), 0, y)

        # 2단계: NaN을 0으로 대체 (더 안전한 방법)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        # 3단계: 유효한 타겟 인덱스만 선택
        valid_idx = ~np.isnan(y) & ~np.isinf(y) & (np.abs(y) < 1.0)  # 극값 제거
        X = X[valid_idx]
        y = y[valid_idx]

        # 4단계: 시퀀스 데이터 생성
        sequence_length = 15
        if len(X) > sequence_length:
            X_seq = []
            y_seq = []

            for i in range(sequence_length, len(X)):
                X_seq.append(X[i-sequence_length:i])
                y_seq.append(y[i])

            X_seq = np.array(X_seq)
            y_seq = np.array(y_seq)

            print(f"   ✅ 안전한 데이터 준비 완료: X={X_seq.shape}, y={y_seq.shape}")
            return X_seq, y_seq, feature_columns
        else:
            print(f"   ⚠️ 데이터 부족: {len(X)} < {sequence_length}")
            return None, None, None

    class SafeLSTM(nn.Module):
        """안전한 LSTM 모델"""

        def __init__(self, input_size, hidden_size=32, num_layers=2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                              batch_first=True, dropout=0.2)
            self.fc = nn.Linear(hidden_size, 1)
            self.dropout = nn.Dropout(0.3)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            last_output = lstm_out[:, -1, :]
            dropped = self.dropout(last_output)
            return self.fc(dropped)

    def train_safe_lstm(self, X_train, y_train, X_val, y_val):
        """안전한 LSTM 훈련"""
        model = self.SafeLSTM(X_train.shape[-1]).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 텐서 변환
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)

        model.train()
        for epoch in range(50):
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

    def train_safe_traditional_ml(self, X_train, y_train, X_val, y_val):
        """안전한 전통적 ML 모델들"""
        results = {}

        # 2D로 변환 (ML 모델용)
        X_train_2d = X_train.reshape(X_train.shape[0], -1)
        X_val_2d = X_val.reshape(X_val.shape[0], -1)

        # 추가 안전 처리
        imputer = SimpleImputer(strategy='mean')
        X_train_2d = imputer.fit_transform(X_train_2d)
        X_val_2d = imputer.transform(X_val_2d)

        # 스케일링
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_2d)
        X_val_scaled = scaler.transform(X_val_2d)

        # Random Forest
        try:
            rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
            rf.fit(X_train_scaled, y_train)
            rf_pred = rf.predict(X_val_scaled)
            results['RandomForest'] = rf_pred
        except Exception as e:
            print(f"      RandomForest 오류: {e}")
            results['RandomForest'] = np.zeros(len(y_val))

        # ElasticNet (강화된 안전 처리)
        try:
            # 추가 안전 검사
            if np.any(np.isnan(X_train_scaled)) or np.any(np.isnan(y_train)):
                print("      ElasticNet: NaN 발견, 제로 패딩 적용")
                X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0)
                X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0)
                y_train_clean = np.nan_to_num(y_train, nan=0.0)
            else:
                y_train_clean = y_train

            elastic = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
            elastic.fit(X_train_scaled, y_train_clean)
            elastic_pred = elastic.predict(X_val_scaled)
            results['ElasticNet'] = elastic_pred
        except Exception as e:
            print(f"      ElasticNet 오류: {e}")
            results['ElasticNet'] = np.zeros(len(y_val))

        # Ridge
        try:
            ridge = Ridge(alpha=1.0, random_state=42)
            ridge.fit(X_train_scaled, y_train)
            ridge_pred = ridge.predict(X_val_scaled)
            results['Ridge'] = ridge_pred
        except Exception as e:
            print(f"      Ridge 오류: {e}")
            results['Ridge'] = np.zeros(len(y_val))

        return results

    def run_fixed_experiments(self, data_path='/root/workspace/data/training/sp500_2020_2024_enhanced.csv'):
        """수정된 실험 실행"""
        print("🔧 수정된 고급 회귀 최적화 시스템 시작")
        print("="*70)

        start_time = time.time()
        all_results = {}

        try:
            # 데이터 로딩
            df = self.data_processor.load_and_validate_data(data_path)

            # 실험 1: 수익률 예측
            print("\n🎯 실험 1: 수익률 예측 (Returns)")
            X, y, features = self.prepare_safe_data(df, target='returns')

            if X is not None:
                returns_results = self._run_cross_validation(X, y, 'returns')
                all_results['returns'] = returns_results

            # 실험 2: 변동성 예측
            print("\n🎯 실험 2: 변동성 예측 (Volatility)")
            X_vol, y_vol, _ = self.prepare_safe_data(df, target='volatility')

            if X_vol is not None:
                volatility_results = self._run_cross_validation(X_vol, y_vol, 'volatility')
                all_results['volatility'] = volatility_results

            # 결과 정리
            total_time = time.time() - start_time
            self._summarize_results(all_results, total_time)

            return all_results

        except Exception as e:
            print(f"❌ 수정된 실험 중 오류: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _run_cross_validation(self, X, y, target_type):
        """교차 검증 실행"""
        tscv = TimeSeriesSplit(n_splits=3)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            print(f"\n📊 Fold {fold+1}/3")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            fold_result = {}

            # LSTM 훈련
            print("   🧠 LSTM 훈련...")
            try:
                lstm_pred = self.train_safe_lstm(X_train, y_train, X_val, y_val)
                lstm_mae = mean_absolute_error(y_val, lstm_pred)
                lstm_r2 = r2_score(y_val, lstm_pred)
                fold_result['LSTM'] = {'mae': lstm_mae, 'r2': lstm_r2, 'pred': lstm_pred}
                print(f"      LSTM: MAE={lstm_mae:.6f}, R²={lstm_r2:.4f}")
            except Exception as e:
                print(f"      LSTM 오류: {e}")
                fold_result['LSTM'] = {'mae': np.inf, 'r2': -np.inf, 'pred': np.zeros(len(y_val))}

            # 전통적 ML 모델들
            print("   📊 전통적 ML 모델들...")
            try:
                ml_results = self.train_safe_traditional_ml(X_train, y_train, X_val, y_val)

                for model_name, pred in ml_results.items():
                    mae = mean_absolute_error(y_val, pred)
                    r2 = r2_score(y_val, pred)
                    fold_result[model_name] = {'mae': mae, 'r2': r2, 'pred': pred}
                    print(f"      {model_name}: MAE={mae:.6f}, R²={r2:.4f}")

            except Exception as e:
                print(f"      ML 모델 오류: {e}")

            # 앙상블
            print("   🎯 앙상블...")
            try:
                ensemble_pred = self._create_ensemble(fold_result, y_val)
                ensemble_mae = mean_absolute_error(y_val, ensemble_pred)
                ensemble_r2 = r2_score(y_val, ensemble_pred)
                fold_result['Ensemble'] = {'mae': ensemble_mae, 'r2': ensemble_r2, 'pred': ensemble_pred}
                print(f"      Ensemble: MAE={ensemble_mae:.6f}, R²={ensemble_r2:.4f}")
            except Exception as e:
                print(f"      앙상블 오류: {e}")

            fold_results.append(fold_result)
            print(f"   ✅ Fold {fold+1} 완료")

        return fold_results

    def _create_ensemble(self, fold_result, y_true):
        """안전한 앙상블 생성"""
        predictions = []
        weights = []

        for model_name, result in fold_result.items():
            if 'pred' in result and len(result['pred']) == len(y_true):
                predictions.append(result['pred'])
                # R²가 높을수록 높은 가중치
                weight = max(0.1, result.get('r2', 0) + 1)  # 최소 0.1 가중치
                weights.append(weight)

        if predictions:
            predictions = np.array(predictions)
            weights = np.array(weights)
            weights = weights / weights.sum()  # 정규화

            ensemble_pred = np.average(predictions, axis=0, weights=weights)
            return ensemble_pred
        else:
            return np.zeros(len(y_true))

    def _summarize_results(self, all_results, total_time):
        """결과 요약"""
        print(f"\n📊 수정된 고급 회귀 최적화 결과 요약:")
        print(f"   총 소요시간: {total_time:.1f}초")

        for target_type, results in all_results.items():
            print(f"\n🎯 {target_type.upper()} 예측 결과:")

            # 모델별 평균 성능 계산
            model_performance = {}

            for fold_result in results:
                for model_name, metrics in fold_result.items():
                    if model_name not in model_performance:
                        model_performance[model_name] = {'mae': [], 'r2': []}

                    model_performance[model_name]['mae'].append(metrics['mae'])
                    model_performance[model_name]['r2'].append(metrics['r2'])

            # 평균 출력
            print("   모델별 평균 성능:")
            for model_name, metrics in model_performance.items():
                avg_mae = np.mean(metrics['mae'])
                avg_r2 = np.mean(metrics['r2'])
                print(f"      {model_name:12s}: MAE={avg_mae:.6f}, R²={avg_r2:.4f}")

        # 결과 저장
        output_path = f"/root/workspace/data/results/fixed_advanced_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            # JSON 직렬화 가능한 형태로 변환
            serializable_results = {}
            for target_type, results in all_results.items():
                serializable_results[target_type] = []
                for fold_result in results:
                    fold_data = {}
                    for model_name, metrics in fold_result.items():
                        fold_data[model_name] = {
                            'mae': float(metrics['mae']),
                            'r2': float(metrics['r2'])
                        }
                    serializable_results[target_type].append(fold_data)

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
    optimizer = FixedAdvancedRegressionOptimizer()
    results = optimizer.run_fixed_experiments()

    print("\n🎉 수정된 고급 회귀 최적화 완료!")
    return results

if __name__ == "__main__":
    main()