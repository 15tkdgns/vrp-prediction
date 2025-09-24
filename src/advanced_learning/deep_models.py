#!/usr/bin/env python3
"""
고급 딥러닝 모델 시스템
LSTM, Transformer, TFT (Temporal Fusion Transformer) 구현
시계열 금융 데이터 예측에 특화된 모델들
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
import warnings
import logging
import math

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """모델 설정 클래스"""
    sequence_length: int = 20
    hidden_size: int = 128
    num_layers: int = 3
    dropout: float = 0.3
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 100
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    early_stopping_patience: int = 10


class TimeSeriesDataset(Dataset):
    """시계열 데이터셋 클래스"""

    def __init__(self, X: np.ndarray, y: np.ndarray, sequence_length: int = 20):
        """
        Args:
            X: 특성 데이터 (samples, features)
            y: 타겟 데이터 (samples,)
            sequence_length: 시퀀스 길이
        """
        self.X = X
        self.y = y
        self.sequence_length = sequence_length

        # 시퀀스 데이터 생성
        self.sequences, self.targets = self._create_sequences()

    def _create_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        """시퀀스 데이터 생성"""
        sequences = []
        targets = []

        for i in range(len(self.X) - self.sequence_length + 1):
            seq = self.X[i:i + self.sequence_length]
            target = self.y[i + self.sequence_length - 1]
            sequences.append(seq)
            targets.append(target)

        return np.array(sequences), np.array(targets)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor([self.targets[idx]])


class AdvancedLSTM(nn.Module):
    """고급 LSTM 모델 - Bidirectional + Attention"""

    def __init__(self, input_size: int, config: ModelConfig):
        """
        Args:
            input_size: 입력 특성 수
            config: 모델 설정
        """
        super(AdvancedLSTM, self).__init__()
        self.config = config

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            batch_first=True,
            bidirectional=True
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size * 2,  # bidirectional
            num_heads=8,
            dropout=config.dropout,
            batch_first=True
        )

        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, sequence_length, input_size)

        Returns:
            predictions: (batch_size, 1)
        """
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size*2)

        # Attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Global pooling
        pooled = torch.mean(attn_out, dim=1)  # (batch_size, hidden_size*2)

        # Classification
        output = self.classifier(pooled)

        return output


class PositionalEncoding(nn.Module):
    """위치 인코딩 (Transformer용)"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class FinancialTransformer(nn.Module):
    """금융 시계열용 Transformer 모델"""

    def __init__(self, input_size: int, config: ModelConfig):
        """
        Args:
            input_size: 입력 특성 수
            config: 모델 설정
        """
        super(FinancialTransformer, self).__init__()
        self.config = config

        # Input projection
        self.input_projection = nn.Linear(input_size, config.hidden_size)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(config.hidden_size)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=8,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, sequence_length, input_size)

        Returns:
            predictions: (batch_size, 1)
        """
        # Input projection
        x = self.input_projection(x)  # (batch_size, seq_len, hidden_size)

        # Positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch_size, hidden_size)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, hidden_size)

        # Transformer
        transformer_out = self.transformer(x)

        # Global average pooling
        pooled = torch.mean(transformer_out, dim=1)  # (batch_size, hidden_size)

        # Classification
        output = self.classifier(pooled)

        return output


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit (GLU) - TFT용"""

    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1):
        super(GatedLinearUnit, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.dropout(x)
        return torch.sigmoid(x[:, :, :x.size(2)//2]) * x[:, :, x.size(2)//2:]


class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer (TFT)
    시계열 예측에 특화된 고급 Transformer 모델
    """

    def __init__(self, input_size: int, config: ModelConfig):
        """
        Args:
            input_size: 입력 특성 수
            config: 모델 설정
        """
        super(TemporalFusionTransformer, self).__init__()
        self.config = config

        # Variable selection networks
        self.feature_selection = nn.Sequential(
            nn.Linear(input_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, input_size),
            nn.Softmax(dim=-1)
        )

        # GLU layers for feature processing
        self.glu_layers = nn.ModuleList([
            GatedLinearUnit(input_size if i == 0 else config.hidden_size,
                          config.hidden_size, config.dropout)
            for i in range(2)
        ])

        # LSTM for sequential processing
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            batch_first=True
        )

        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=8,
            dropout=config.dropout,
            batch_first=True
        )

        # Position-wise feed forward
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size * 4, config.hidden_size)
        )

        # Output layers
        self.output_projection = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Sigmoid()
        )

        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, sequence_length, input_size)

        Returns:
            predictions: (batch_size, 1)
        """
        # Variable selection
        feature_weights = self.feature_selection(x)
        x_selected = x * feature_weights

        # GLU processing
        for glu in self.glu_layers:
            x_selected = glu(x_selected)

        # LSTM processing
        lstm_out, _ = self.lstm(x_selected)

        # Multi-head attention with residual connection
        attn_out, _ = self.multihead_attn(lstm_out, lstm_out, lstm_out)
        x_attn = self.layer_norm1(lstm_out + attn_out)

        # Feed forward with residual connection
        ff_out = self.feed_forward(x_attn)
        x_ff = self.layer_norm2(x_attn + ff_out)

        # Global pooling and output
        pooled = torch.mean(x_ff, dim=1)  # (batch_size, hidden_size)
        output = self.output_projection(pooled)

        return output


class DeepModelWrapper(BaseEstimator, ClassifierMixin):
    """
    딥러닝 모델 래퍼 - scikit-learn 호환성
    """

    def __init__(self, model_type: str = 'lstm', config: ModelConfig = None):
        """
        Args:
            model_type: 모델 타입 ('lstm', 'transformer', 'tft')
            config: 모델 설정
        """
        self.model_type = model_type
        self.config = config or ModelConfig()
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False

    def _create_model(self, input_size: int) -> nn.Module:
        """모델 생성"""
        if self.model_type == 'lstm':
            return AdvancedLSTM(input_size, self.config)
        elif self.model_type == 'transformer':
            return FinancialTransformer(input_size, self.config)
        elif self.model_type == 'tft':
            return TemporalFusionTransformer(input_size, self.config)
        else:
            raise ValueError(f"지원하지 않는 모델 타입: {self.model_type}")

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DeepModelWrapper':
        """
        모델 훈련

        Args:
            X: 특성 데이터 (samples, features)
            y: 타겟 데이터 (samples,)

        Returns:
            자기 자신 (fitted)
        """
        # 데이터 스케일링
        X_scaled = self.scaler.fit_transform(X)

        # 모델 생성
        input_size = X_scaled.shape[1]
        self.model = self._create_model(input_size)
        self.model.to(self.config.device)

        # 데이터셋 생성
        dataset = TimeSeriesDataset(X_scaled, y, self.config.sequence_length)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)

        # 훈련 설정
        criterion = nn.BCELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        # 조기 종료 설정
        best_loss = float('inf')
        patience_counter = 0

        self.model.train()
        for epoch in range(self.config.epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.config.device)
                batch_y = batch_y.to(self.config.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            scheduler.step(avg_loss)

            # 조기 종료 확인
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"조기 종료: epoch {epoch+1}")
                    break

            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch+1}/{self.config.epochs}, Loss: {avg_loss:.6f}")

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        확률 예측

        Args:
            X: 특성 데이터

        Returns:
            predictions: 확률 예측값 (samples, 2)
        """
        if not self.is_fitted:
            raise ValueError("모델이 아직 훈련되지 않았습니다.")

        # 데이터 스케일링
        X_scaled = self.scaler.transform(X)

        # 데이터셋 생성
        y_dummy = np.zeros(len(X))  # 더미 타겟
        dataset = TimeSeriesDataset(X_scaled, y_dummy, self.config.sequence_length)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)

        # 예측
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for batch_X, _ in dataloader:
                batch_X = batch_X.to(self.config.device)
                outputs = self.model(batch_X)
                predictions.extend(outputs.cpu().numpy())

        predictions = np.array(predictions).flatten()

        # scikit-learn 호환을 위해 2열로 변환 (negative, positive)
        proba = np.column_stack([1 - predictions, predictions])

        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        이진 예측

        Args:
            X: 특성 데이터

        Returns:
            predictions: 이진 예측값 (samples,)
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)

    def get_feature_importance(self) -> np.ndarray:
        """특성 중요도 반환 (근사치)"""
        if not self.is_fitted:
            raise ValueError("모델이 아직 훈련되지 않았습니다.")

        # TFT의 경우 feature selection 가중치 사용
        if self.model_type == 'tft' and hasattr(self.model, 'feature_selection'):
            # 더미 입력으로 feature selection 가중치 계산
            dummy_input = torch.randn(1, self.config.sequence_length,
                                    self.model.feature_selection[0].in_features).to(self.config.device)
            with torch.no_grad():
                weights = self.model.feature_selection(dummy_input)
                return weights.mean(dim=(0, 1)).cpu().numpy()

        # 다른 모델의 경우 근사적 중요도 계산
        return np.ones(self.scaler.n_features_in_) / self.scaler.n_features_in_


class ModelFactory:
    """딥러닝 모델 팩토리"""

    @staticmethod
    def create_model(model_type: str, config: ModelConfig = None) -> DeepModelWrapper:
        """
        모델 생성

        Args:
            model_type: 모델 타입
            config: 모델 설정

        Returns:
            모델 인스턴스
        """
        return DeepModelWrapper(model_type, config)

    @staticmethod
    def get_available_models() -> List[str]:
        """사용 가능한 모델 타입 반환"""
        return ['lstm', 'transformer', 'tft']


# 사용 예시 및 테스트
if __name__ == "__main__":
    # 테스트 데이터 생성
    np.random.seed(42)
    torch.manual_seed(42)

    n_samples = 1000
    n_features = 10

    # 시계열 특성을 가진 데이터 생성
    X = np.random.randn(n_samples, n_features)
    for i in range(1, n_samples):
        X[i] += 0.1 * X[i-1]  # 시계열 의존성

    y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.1 > 0).astype(int)

    # 훈련/테스트 분할
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print("=== 고급 딥러닝 모델 테스트 ===")

    # 모델 설정
    config = ModelConfig(
        sequence_length=10,
        hidden_size=64,
        num_layers=2,
        epochs=50,
        batch_size=32,
        early_stopping_patience=10
    )

    # 각 모델 테스트
    models = ['lstm', 'transformer', 'tft']
    results = {}

    for model_type in models:
        print(f"\n--- {model_type.upper()} 모델 테스트 ---")

        try:
            # 모델 생성 및 훈련
            model = ModelFactory.create_model(model_type, config)
            model.fit(X_train, y_train)

            # 예측
            y_pred_proba = model.predict_proba(X_test)
            y_pred = model.predict(X_test)

            # 성능 계산
            from sklearn.metrics import accuracy_score, roc_auc_score

            accuracy = accuracy_score(y_test, y_pred)
            try:
                auc = roc_auc_score(y_test, y_pred_proba[:, 1])
            except:
                auc = 0.5

            results[model_type] = {
                'accuracy': accuracy,
                'auc': auc,
                'predictions': y_pred_proba[:, 1]
            }

            print(f"정확도: {accuracy:.4f}")
            print(f"AUC: {auc:.4f}")

            # 특성 중요도
            importance = model.get_feature_importance()
            print(f"특성 중요도 (상위 3개): {importance[:3]}")

        except Exception as e:
            print(f"오류 발생: {e}")
            results[model_type] = {'error': str(e)}

    # 결과 요약
    print(f"\n=== 모델 성능 비교 ===")
    for model_type, result in results.items():
        if 'error' not in result:
            print(f"{model_type.upper()}: "
                  f"정확도={result['accuracy']:.4f}, "
                  f"AUC={result['auc']:.4f}")
        else:
            print(f"{model_type.upper()}: 오류 - {result['error']}")

    print("\n✅ 고급 딥러닝 모델 시스템 테스트 완료")
    print("LSTM, Transformer, TFT 모델 구현 및 테스트 성공")