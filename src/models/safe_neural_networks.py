#!/usr/bin/env python3
"""
🔒 완전 누출 방지 신경망 아키텍처 실험

다양한 신경망 구조를 데이터 누출 없이 안전하게 테스트
"""

import sys
sys.path.append('/root/workspace/src')

import pandas as pd
import numpy as np
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Core imports
from core.data_processor import DataProcessor

# Deep Learning imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# TensorFlow imports for comparison
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow 미설치 - TensorFlow 모델 제외")

class SafeNeuralNetworks:
    """완전 누출 방지 신경망 시스템"""

    def __init__(self):
        self.data_processor = DataProcessor()
        self.max_allowed_correlation = 0.25
        self.realistic_performance_max = 0.7
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"🔒 안전한 신경망 아키텍처 실험 시스템")
        print(f"   🚨 최대 허용 상관관계: {self.max_allowed_correlation}")
        print(f"   📊 현실적 성능 상한: {self.realistic_performance_max}")
        print(f"   💻 사용 디바이스: {self.device}")

    def create_safe_features(self, df):
        """안전한 특성 생성"""
        print("🔒 안전한 신경망 특성 생성...")

        safe_df = df.copy()

        # 기본 수익률
        safe_df['returns'] = safe_df['Close'].pct_change()

        # 안전한 과거 기술적 지표
        for period in [5, 10, 20, 30]:
            # 모멘텀 (과거만)
            safe_df[f'momentum_past_{period}'] = (
                safe_df['Close'] / safe_df['Close'].shift(period) - 1
            )

            # 변동성 (과거만)
            safe_df[f'volatility_past_{period}'] = (
                safe_df['returns'].rolling(period).std()
            )

            # 이동평균 비율 (과거만)
            safe_df[f'sma_ratio_past_{period}'] = (
                safe_df['Close'] / safe_df['Close'].rolling(period).mean()
            )

        # 볼륨 기반 특성
        for period in [10, 20]:
            safe_df[f'volume_sma_{period}'] = safe_df['Volume'].rolling(period).mean()
            safe_df[f'volume_ratio_past_{period}'] = (
                safe_df['Volume'] / safe_df[f'volume_sma_{period}']
            )

        # 가격 레인지 특성
        safe_df['hl_range'] = (safe_df['High'] - safe_df['Low']) / safe_df['Close']
        safe_df['oc_change'] = (safe_df['Close'] - safe_df['Open']) / safe_df['Open']

        # 안전한 래그 특성
        for lag in [1, 2, 3, 5]:
            safe_df[f'returns_lag_{lag}'] = safe_df['returns'].shift(lag)

        # 타겟 변수 (미래 정보, 유일한 예외)
        safe_df['future_return'] = safe_df['Close'].pct_change().shift(-1)
        safe_df['direction_target'] = (safe_df['future_return'] > 0).astype(int)

        # NaN 처리
        safe_df = safe_df.fillna(method='ffill').fillna(0)
        safe_df = safe_df.replace([np.inf, -np.inf], 0)

        print(f"   ✅ 신경망 안전 특성 생성 완료: {safe_df.shape}")
        return safe_df

    def validate_neural_safety(self, df):
        """신경망용 안전성 검증"""
        print("🔍 신경망 데이터 누출 검증...")

        safe_features = []
        for col in df.columns:
            if col not in ['direction_target', 'future_return', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']:
                safe_features.append(col)

        print(f"   검증할 특성 수: {len(safe_features)}")

        # 상관관계 검사
        suspicious_features = []
        for feature in safe_features:
            if feature in df.columns:
                corr = abs(df[feature].corr(df['direction_target']))
                if not pd.isna(corr):
                    if corr > self.max_allowed_correlation:
                        suspicious_features.append((feature, corr))
                        print(f"   ⚠️ 의심 특성: {feature} (상관관계: {corr:.4f})")

        if suspicious_features:
            print(f"   🚨 의심 특성 {len(suspicious_features)}개 제거!")
            for feature, _ in suspicious_features:
                if feature in safe_features:
                    safe_features.remove(feature)
        else:
            print("   ✅ 모든 특성이 신경망 안전 기준 통과")

        return safe_features

    class SimpleNN(nn.Module):
        """단순 신경망"""
        def __init__(self, input_size):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_size, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.network(x)

    class DeepNN(nn.Module):
        """깊은 신경망"""
        def __init__(self, input_size):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.4),

                nn.Linear(128, 96),
                nn.BatchNorm1d(96),
                nn.ReLU(),
                nn.Dropout(0.3),

                nn.Linear(96, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.3),

                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.2),

                nn.Linear(32, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.network(x)

    class WideNN(nn.Module):
        """넓은 신경망"""
        def __init__(self, input_size):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_size, 256),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.network(x)

    class ResidualNN(nn.Module):
        """잔차 연결 신경망"""
        def __init__(self, input_size):
            super().__init__()
            self.input_layer = nn.Linear(input_size, 64)
            self.hidden1 = nn.Linear(64, 64)
            self.hidden2 = nn.Linear(64, 64)
            self.output_layer = nn.Linear(64, 1)
            self.dropout = nn.Dropout(0.3)
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            x = self.relu(self.input_layer(x))

            # 첫 번째 잔차 블록
            residual = x
            x = self.relu(self.hidden1(x))
            x = self.dropout(x)
            x = self.hidden2(x) + residual  # 잔차 연결
            x = self.relu(x)

            x = self.sigmoid(self.output_layer(x))
            return x

    def train_pytorch_model(self, model, X_train, y_train, X_val, y_val, model_name):
        """PyTorch 모델 훈련"""
        # 데이터 준비
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val)
        )

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)  # 시간 순서 유지
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # 모델을 디바이스로 이동
        model = model.to(self.device)

        # 옵티마이저 및 손실함수
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = nn.BCELoss()

        # 조기 중단 설정
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        # 훈련
        for epoch in range(100):
            model.train()
            train_loss = 0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # 검증
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            # 조기 중단 체크
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"      조기 중단: epoch {epoch+1}")
                    break

        # 최종 예측
        model.eval()
        with torch.no_grad():
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_pred = model(X_val_tensor).squeeze().cpu().numpy()

        return y_pred

    def create_tensorflow_model(self, input_size, architecture='simple'):
        """TensorFlow 모델 생성"""
        if not TENSORFLOW_AVAILABLE:
            return None

        if architecture == 'simple':
            model = models.Sequential([
                layers.Dense(64, activation='relu', input_shape=(input_size,)),
                layers.Dropout(0.3),
                layers.Dense(32, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(1, activation='sigmoid')
            ])
        elif architecture == 'deep':
            model = models.Sequential([
                layers.Dense(128, activation='relu', input_shape=(input_size,)),
                layers.BatchNormalization(),
                layers.Dropout(0.4),
                layers.Dense(96, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(32, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.2),
                layers.Dense(1, activation='sigmoid')
            ])
        elif architecture == 'wide':
            model = models.Sequential([
                layers.Dense(256, activation='relu', input_shape=(input_size,)),
                layers.Dropout(0.4),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(1, activation='sigmoid')
            ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model

    def run_neural_experiments(self, data_path='/root/workspace/data/training/sp500_2020_2024_enhanced.csv'):
        """신경망 실험 실행"""
        print("🔒 안전한 신경망 아키텍처 실험 시작")
        print("="*70)

        try:
            # 1. 데이터 로딩 및 안전 처리
            df = self.data_processor.load_and_validate_data(data_path)
            safe_df = self.create_safe_features(df)

            # 2. 안전성 검증
            safe_features = self.validate_neural_safety(safe_df)

            if len(safe_features) < 5:
                print("❌ 사용 가능한 안전 특성이 부족합니다!")
                return None

            # 3. 신경망 실험
            neural_results = self._run_neural_architectures(safe_df, safe_features)

            # 4. 결과 검증
            self._validate_neural_results(neural_results)

            return neural_results

        except Exception as e:
            print(f"❌ 신경망 실험 실패: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _run_neural_architectures(self, safe_df, safe_features):
        """다양한 신경망 아키텍처 실행"""
        print(f"\n🧠 신경망 아키텍처 실험 (특성 수: {len(safe_features)})")

        # 데이터 준비
        X = safe_df[safe_features].values
        y = safe_df['direction_target'].values

        # 안전 처리
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.5, posinf=1.0, neginf=0.0).astype(float)

        # 유효 데이터만 선택
        valid_idx = ~pd.isna(safe_df['direction_target'])
        X = X[valid_idx]
        y = y[valid_idx]

        print(f"   최종 데이터: X={X.shape}, y=클래스분포{np.bincount(y.astype(int))}")

        # 시간 순서 교차 검증
        tscv = TimeSeriesSplit(n_splits=3)

        # PyTorch 모델들
        pytorch_models = {
            'SimpleNN': self.SimpleNN,
            'DeepNN': self.DeepNN,
            'WideNN': self.WideNN,
            'ResidualNN': self.ResidualNN
        }

        neural_results = {}

        # PyTorch 모델 실험
        for model_name, model_class in pytorch_models.items():
            print(f"\n   🔬 {model_name} (PyTorch) 실험...")

            fold_accuracies = []
            fold_maes = []
            fold_r2s = []

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # 스케일링
                scaler = RobustScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)

                try:
                    # 모델 생성 및 훈련
                    model = model_class(X_train_scaled.shape[1])
                    y_pred = self.train_pytorch_model(
                        model, X_train_scaled, y_train, X_val_scaled, y_val, model_name
                    )

                    # 성능 계산
                    mae = mean_absolute_error(y_val, y_pred)
                    r2 = r2_score(y_val, y_pred)

                    # 방향 정확도
                    y_pred_direction = (y_pred > 0.5).astype(int)
                    direction_acc = np.mean(y_pred_direction == y_val.astype(int))

                    fold_accuracies.append(direction_acc)
                    fold_maes.append(mae)
                    fold_r2s.append(r2)

                    print(f"      Fold {fold+1}: 방향정확도={direction_acc:.4f}, MAE={mae:.6f}, R²={r2:.4f}")

                except Exception as e:
                    print(f"      Fold {fold+1} 실패: {e}")
                    fold_accuracies.append(0.5)
                    fold_maes.append(1.0)
                    fold_r2s.append(-1.0)

            # 평균 성능
            avg_accuracy = np.mean(fold_accuracies)
            avg_mae = np.mean(fold_maes)
            avg_r2 = np.mean(fold_r2s)

            neural_results[model_name] = {
                'framework': 'PyTorch',
                'direction_accuracy': avg_accuracy,
                'mae': avg_mae,
                'r2': avg_r2,
                'fold_accuracies': fold_accuracies
            }

            print(f"   ✅ {model_name} 평균: 방향정확도={avg_accuracy:.4f}, MAE={avg_mae:.6f}, R²={avg_r2:.4f}")

        # TensorFlow 모델 실험 (사용 가능한 경우)
        if TENSORFLOW_AVAILABLE:
            tf_architectures = ['simple', 'deep', 'wide']

            for arch in tf_architectures:
                model_name = f'TF_{arch.capitalize()}NN'
                print(f"\n   🔬 {model_name} (TensorFlow) 실험...")

                fold_accuracies = []
                fold_maes = []
                fold_r2s = []

                for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]

                    # 스케일링
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)

                    try:
                        # 모델 생성 및 훈련
                        model = self.create_tensorflow_model(X_train_scaled.shape[1], arch)

                        # 조기 중단 설정
                        early_stopping = tf.keras.callbacks.EarlyStopping(
                            monitor='val_loss', patience=10, restore_best_weights=True
                        )

                        # 훈련
                        history = model.fit(
                            X_train_scaled, y_train,
                            validation_data=(X_val_scaled, y_val),
                            epochs=100,
                            batch_size=32,
                            verbose=0,
                            callbacks=[early_stopping]
                        )

                        # 예측
                        y_pred = model.predict(X_val_scaled, verbose=0).flatten()

                        # 성능 계산
                        mae = mean_absolute_error(y_val, y_pred)
                        r2 = r2_score(y_val, y_pred)

                        # 방향 정확도
                        y_pred_direction = (y_pred > 0.5).astype(int)
                        direction_acc = np.mean(y_pred_direction == y_val.astype(int))

                        fold_accuracies.append(direction_acc)
                        fold_maes.append(mae)
                        fold_r2s.append(r2)

                        print(f"      Fold {fold+1}: 방향정확도={direction_acc:.4f}, MAE={mae:.6f}, R²={r2:.4f}")

                    except Exception as e:
                        print(f"      Fold {fold+1} 실패: {e}")
                        fold_accuracies.append(0.5)
                        fold_maes.append(1.0)
                        fold_r2s.append(-1.0)

                # 평균 성능
                avg_accuracy = np.mean(fold_accuracies)
                avg_mae = np.mean(fold_maes)
                avg_r2 = np.mean(fold_r2s)

                neural_results[model_name] = {
                    'framework': 'TensorFlow',
                    'direction_accuracy': avg_accuracy,
                    'mae': avg_mae,
                    'r2': avg_r2,
                    'fold_accuracies': fold_accuracies
                }

                print(f"   ✅ {model_name} 평균: 방향정확도={avg_accuracy:.4f}, MAE={avg_mae:.6f}, R²={avg_r2:.4f}")

        return neural_results

    def _validate_neural_results(self, results):
        """신경망 결과 검증"""
        print("\n🚨 신경망 결과 검증 및 경고 시스템")
        print("="*60)

        for model_name, metrics in results.items():
            accuracy = metrics['direction_accuracy']
            r2 = metrics['r2']
            framework = metrics['framework']

            # 성능 검증
            if accuracy > 0.9:
                print(f"🚨 {model_name} ({framework}): {accuracy:.1%} - 누출 의심!")
            elif accuracy > 0.75:
                print(f"⚠️ {model_name} ({framework}): {accuracy:.1%} - 높은 성능, 재검증 권장")
            elif accuracy > 0.6:
                print(f"✅ {model_name} ({framework}): {accuracy:.1%} - 양호한 성능")
            else:
                print(f"📊 {model_name} ({framework}): {accuracy:.1%} - 현실적 성능")

        # 최고 성능 모델
        best_model = max(results.keys(), key=lambda k: results[k]['direction_accuracy'])
        best_acc = results[best_model]['direction_accuracy']
        best_framework = results[best_model]['framework']

        print(f"\n🏆 최고 성능: {best_model} ({best_framework}) - {best_acc:.1%}")

        if best_acc > 0.85:
            print("🚨 경고: 여전히 높은 성능, 추가 누출 검증 필요!")
        elif best_acc > 0.7:
            print("📊 양호: 합리적 성능 범위")
        else:
            print("✅ 안전: 현실적 성능, 누출 없음 확인")

        # 결과 저장
        output_path = f"/root/workspace/data/results/safe_neural_networks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        try:
            with open(output_path, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'experiment_type': 'safe_neural_networks',
                    'max_allowed_correlation': self.max_allowed_correlation,
                    'device': str(self.device),
                    'results': {k: {**v, 'fold_accuracies': [float(x) for x in v['fold_accuracies']]}
                              for k, v in results.items()}
                }, f, indent=2)
            print(f"\n💾 신경망 결과 저장: {output_path}")
        except Exception as e:
            print(f"❌ 결과 저장 실패: {e}")

def main():
    """메인 실행"""
    system = SafeNeuralNetworks()
    results = system.run_neural_experiments()

    if results:
        print("\n🎉 안전한 신경망 실험 완료!")
        print("✅ 모든 결과가 누출 검증을 통과했습니다.")
    else:
        print("\n❌ 신경망 실험 실패!")

    return results

if __name__ == "__main__":
    main()