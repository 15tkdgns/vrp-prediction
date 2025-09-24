#!/usr/bin/env python3
"""
GPU 가속 고성능 모델 학습 시스템
데이터 누출 완전 차단 및 엄격한 시계열 검증 적용
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# GPU 메모리 증가 설정 (TensorFlow)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU 설정 오류: {e}")

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataLeakageValidator:
    """데이터 누출 검증기 - 엄격한 시계열 검증"""

    @staticmethod
    def validate_temporal_order(df, date_col='Date'):
        """시계열 순서 검증"""
        if date_col in df.columns:
            return df[date_col].is_monotonic_increasing
        return True

    @staticmethod
    def validate_feature_independence(X, y, threshold=0.99):
        """특징과 타겟의 과도한 상관관계 검사"""
        correlations = []
        for i in range(X.shape[1]):
            corr = np.corrcoef(X[:, i], y)[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))

        max_corr = max(correlations) if correlations else 0
        return max_corr < threshold, max_corr

    @staticmethod
    def validate_time_series_split(train_idx, test_idx):
        """시계열 분할 검증 - 훈련 데이터가 테스트 데이터보다 항상 과거여야 함"""
        return max(train_idx) < min(test_idx)

class SafeFeatureEngineer:
    """안전한 특징 공학 - 미래 정보 누출 방지"""

    def __init__(self, lookback_window=20):
        self.lookback_window = lookback_window
        self.scalers = {}

    def create_safe_features(self, df):
        """미래 정보 누출 없는 안전한 특징 생성"""
        df = df.copy()

        # 기본 가격 특징 (과거 정보만 사용)
        df['price_change'] = df['Close'].pct_change(1)
        df['volatility'] = df['Close'].rolling(self.lookback_window, min_periods=1).std()

        # 기술적 지표 (과거 정보만 사용)
        df['sma_20'] = df['Close'].rolling(20, min_periods=1).mean()
        df['sma_50'] = df['Close'].rolling(50, min_periods=1).mean()
        df['rsi'] = self._calculate_rsi(df['Close'])

        # 볼륨 특징
        df['volume_sma'] = df['Volume'].rolling(20, min_periods=1).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']

        # 결측값 처리
        df = df.fillna(method='ffill').fillna(0)

        return df

    def _calculate_rsi(self, prices, window=14):
        """RSI 계산 (안전한 방식)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

class GPUEnhancedLSTM(nn.Module):
    """GPU 가속 LSTM 모델"""

    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.2):
        super(GPUEnhancedLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM 레이어
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout)

        # 어텐션 메커니즘
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8,
                                             dropout=dropout, batch_first=True)

        # 출력 레이어
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1)
        )

    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)

        # 어텐션
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # 마지막 시점의 출력
        out = attn_out[:, -1, :]

        # 최종 예측
        out = self.fc_layers(out)
        return out

class TensorFlowTransformer:
    """TensorFlow 기반 Transformer 모델"""

    def __init__(self, sequence_length, num_features, d_model=128):
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.d_model = d_model
        self.model = self._build_model()

    def _build_model(self):
        """Transformer 모델 구축"""
        inputs = tf.keras.Input(shape=(self.sequence_length, self.num_features))

        # 위치 인코딩
        x = tf.keras.layers.Dense(self.d_model)(inputs)

        # Multi-Head Attention
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=8, key_dim=self.d_model//8)(x, x)
        x = tf.keras.layers.Add()([x, attention])
        x = tf.keras.layers.LayerNormalization()(x)

        # Feed Forward
        ff = tf.keras.layers.Dense(self.d_model*4, activation='relu')(x)
        ff = tf.keras.layers.Dense(self.d_model)(ff)
        x = tf.keras.layers.Add()([x, ff])
        x = tf.keras.layers.LayerNormalization()(x)

        # Global Average Pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(x)

        # 출력 레이어
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(1)(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        return model

class GPUModelTrainer:
    """GPU 가속 모델 훈련기"""

    def __init__(self, data_path=None):
        self.data_path = data_path or Path("/root/workspace/data/training/sp500_2020_2024_enhanced.csv")
        self.validator = DataLeakageValidator()
        self.feature_engineer = SafeFeatureEngineer()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"사용 디바이스: {self.device}")

    def load_safe_data(self):
        """안전한 데이터 로딩 (누출 검증)"""
        try:
            df = pd.read_csv(self.data_path)
            logger.info(f"데이터 로딩 완료: {df.shape}")

            # 시계열 순서 검증
            if not self.validator.validate_temporal_order(df):
                raise ValueError("시계열 순서가 올바르지 않습니다!")

            # 특징 공학
            df = self.feature_engineer.create_safe_features(df)

            # 타겟 변수 (다음 날 수익률)
            df['target'] = df['Close'].pct_change(1).shift(-1)

            # 결측값 제거
            df = df.dropna()

            logger.info(f"전처리 완료: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"데이터 로딩 실패: {e}")
            return None

    def prepare_sequences(self, df, sequence_length=20):
        """시계열 시퀀스 준비"""
        feature_cols = ['price_change', 'volatility', 'sma_20', 'sma_50',
                       'rsi', 'volume_sma', 'volume_ratio']

        # 특징 정규화
        scaler = RobustScaler()
        scaled_features = scaler.fit_transform(df[feature_cols])

        # 시퀀스 생성
        X, y = [], []
        for i in range(sequence_length, len(scaled_features)):
            X.append(scaled_features[i-sequence_length:i])
            y.append(df.iloc[i]['target'])

        return np.array(X), np.array(y), scaler

    def train_pytorch_lstm(self, X, y):
        """PyTorch LSTM 모델 훈련"""
        logger.info("PyTorch LSTM 모델 훈련 시작")

        # 시계열 분할
        tscv = TimeSeriesSplit(n_splits=5)
        best_model = None
        best_score = float('inf')

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            # 데이터 누출 검증
            if not self.validator.validate_time_series_split(train_idx, val_idx):
                raise ValueError(f"Fold {fold}: 시계열 분할 오류!")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # 데이터셋 생성
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train),
                torch.FloatTensor(y_train)
            )
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val)
            )

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

            # 모델 초기화
            model = GPUEnhancedLSTM(
                input_size=X.shape[2],
                hidden_size=128,
                num_layers=3
            ).to(self.device)

            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

            # 훈련
            best_val_loss = float('inf')
            patience_counter = 0

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
                        val_loss += criterion(outputs, batch_y).item()

                val_loss /= len(val_loader)
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    if val_loss < best_score:
                        best_score = val_loss
                        best_model = model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= 20:
                        break

            logger.info(f"Fold {fold+1}: 최고 검증 손실 = {best_val_loss:.6f}")

        # 최고 모델 저장
        if best_model is not None:
            final_model = GPUEnhancedLSTM(
                input_size=X.shape[2],
                hidden_size=128,
                num_layers=3
            )
            final_model.load_state_dict(best_model)
            torch.save(final_model, "/root/workspace/data/models/gpu_enhanced_lstm.pth")
            logger.info(f"PyTorch LSTM 훈련 완료: 최고 점수 = {best_score:.6f}")

        return best_score

    def train_tensorflow_transformer(self, X, y):
        """TensorFlow Transformer 모델 훈련"""
        logger.info("TensorFlow Transformer 모델 훈련 시작")

        tscv = TimeSeriesSplit(n_splits=5)
        best_score = float('inf')
        best_model = None

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            # 데이터 누출 검증
            if not self.validator.validate_time_series_split(train_idx, val_idx):
                raise ValueError(f"Fold {fold}: 시계열 분할 오류!")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # 모델 생성
            transformer = TensorFlowTransformer(
                sequence_length=X.shape[1],
                num_features=X.shape[2]
            )

            # 콜백 설정
            callbacks = [
                tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
                tf.keras.callbacks.ReduceLROnPlateau(patience=10)
            ]

            # 훈련
            history = transformer.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=32,
                callbacks=callbacks,
                verbose=0
            )

            val_loss = min(history.history['val_loss'])
            logger.info(f"Fold {fold+1}: 최고 검증 손실 = {val_loss:.6f}")

            if val_loss < best_score:
                best_score = val_loss
                best_model = transformer.model

        # 최고 모델 저장
        if best_model is not None:
            best_model.save("/root/workspace/data/models/gpu_enhanced_transformer.h5")
            logger.info(f"TensorFlow Transformer 훈련 완료: 최고 점수 = {best_score:.6f}")

        return best_score

    def train_xgboost_gpu(self, X, y):
        """XGBoost GPU 가속 모델 훈련"""
        logger.info("XGBoost GPU 모델 훈련 시작")

        # 시퀀스를 플랫 특징으로 변환
        X_flat = X.reshape(X.shape[0], -1)

        tscv = TimeSeriesSplit(n_splits=5)
        best_score = float('inf')
        best_model = None

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_flat)):
            # 데이터 누출 검증
            if not self.validator.validate_time_series_split(train_idx, val_idx):
                raise ValueError(f"Fold {fold}: 시계열 분할 오류!")

            X_train, X_val = X_flat[train_idx], X_flat[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # XGBoost 모델 (GPU 가속)
            model = xgb.XGBRegressor(
                tree_method='gpu_hist',
                gpu_id=0,
                n_estimators=1000,
                max_depth=6,
                learning_rate=0.01,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                early_stopping_rounds=50
            )

            # 훈련
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            # 검증
            y_pred = model.predict(X_val)
            val_score = mean_squared_error(y_val, y_pred)

            logger.info(f"Fold {fold+1}: 검증 MSE = {val_score:.6f}")

            if val_score < best_score:
                best_score = val_score
                best_model = model

        # 최고 모델 저장
        if best_model is not None:
            joblib.dump(best_model, "/root/workspace/data/models/gpu_enhanced_xgboost.pkl")
            logger.info(f"XGBoost GPU 훈련 완료: 최고 점수 = {best_score:.6f}")

        return best_score

    def run_gpu_training(self):
        """전체 GPU 훈련 파이프라인 실행"""
        logger.info("=== GPU 가속 모델 훈련 시작 ===")

        # 안전한 데이터 로딩
        df = self.load_safe_data()
        if df is None:
            return None

        # 시퀀스 준비
        X, y, scaler = self.prepare_sequences(df)

        # 특징-타겟 상관관계 검증
        is_safe, max_corr = self.validator.validate_feature_independence(
            X.reshape(X.shape[0], -1), y
        )
        if not is_safe:
            logger.warning(f"높은 특징-타겟 상관관계 감지: {max_corr:.3f}")

        results = {}

        # PyTorch LSTM 훈련
        try:
            pytorch_score = self.train_pytorch_lstm(X, y)
            results['pytorch_lstm'] = pytorch_score
        except Exception as e:
            logger.error(f"PyTorch LSTM 훈련 실패: {e}")

        # TensorFlow Transformer 훈련
        try:
            tf_score = self.train_tensorflow_transformer(X, y)
            results['tensorflow_transformer'] = tf_score
        except Exception as e:
            logger.error(f"TensorFlow Transformer 훈련 실패: {e}")

        # XGBoost GPU 훈련
        try:
            xgb_score = self.train_xgboost_gpu(X, y)
            results['xgboost_gpu'] = xgb_score
        except Exception as e:
            logger.error(f"XGBoost GPU 훈련 실패: {e}")

        # 결과 저장
        results_path = "/root/workspace/data/raw/gpu_enhanced_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info("=== GPU 훈련 완료 ===")
        logger.info(f"결과 저장: {results_path}")

        return results

if __name__ == "__main__":
    trainer = GPUModelTrainer()
    results = trainer.run_gpu_training()

    if results:
        print("\n🚀 GPU 가속 모델 훈련 결과:")
        for model_name, score in results.items():
            print(f"  {model_name}: MSE = {score:.6f}")
    else:
        print("❌ 훈련 실패")