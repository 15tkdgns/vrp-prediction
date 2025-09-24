#!/usr/bin/env python3
"""
회귀 지표 최적화 훈련 시스템
MAPE, R², MSE, MAE 기준 가격 예측 모델 훈련
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MAELoss(nn.Module):
    """Mean Absolute Error Loss"""
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, y_pred, y_true):
        return torch.mean(torch.abs(y_pred - y_true))

class HuberLoss(nn.Module):
    """Huber Loss - MSE와 MAE의 결합"""
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, y_pred, y_true):
        residual = torch.abs(y_pred - y_true)
        condition = residual < self.delta
        squared_loss = 0.5 * (y_pred - y_true) ** 2
        linear_loss = self.delta * residual - 0.5 * self.delta ** 2
        return torch.mean(torch.where(condition, squared_loss, linear_loss))

class QuantileLoss(nn.Module):
    """Quantile Loss - 분위수 회귀"""
    def __init__(self, quantile=0.5):
        super(QuantileLoss, self).__init__()
        self.quantile = quantile

    def forward(self, y_pred, y_true):
        residual = y_true - y_pred
        loss = torch.max(self.quantile * residual, (self.quantile - 1) * residual)
        return torch.mean(loss)

class MAPELoss(nn.Module):
    """Mean Absolute Percentage Error Loss"""
    def __init__(self, epsilon=1e-8):
        super(MAPELoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        return torch.mean(torch.abs((y_true - y_pred) / (torch.abs(y_true) + self.epsilon))) * 100

class AdvancedRegressionLSTM(nn.Module):
    """회귀용 고급 LSTM 모델"""

    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.3):
        super(AdvancedRegressionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Bidirectional LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout, bidirectional=True)

        # Multi-Head Attention
        self.attention = nn.MultiheadAttention(hidden_size * 2, num_heads=8,
                                             dropout=dropout, batch_first=True)

        # Layer Normalization
        self.layer_norm1 = nn.LayerNorm(hidden_size * 2)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

        # Regression head with residual connections
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1)  # 회귀 출력
        )

        # Residual connection
        self.residual_proj = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm1(lstm_out)

        # Attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Global pooling
        max_pool = torch.max(attn_out, dim=1)[0]
        avg_pool = torch.mean(attn_out, dim=1)
        combined = max_pool + avg_pool

        # Regression prediction
        main_pred = self.regressor(combined)
        residual_pred = self.residual_proj(combined)

        # Residual connection
        return main_pred + 0.1 * residual_pred

class RegressionMetricTrainer:
    """회귀 지표 최적화 훈련 시스템"""

    def __init__(self, gpu_enabled=True):
        self.device = torch.device('cuda' if gpu_enabled and torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.model_configs = self._get_regression_configs()
        logger.info(f"회귀 훈련 디바이스: {self.device}")

    def _get_regression_configs(self):
        """회귀 모델 설정"""
        return {
            # PyTorch 회귀 모델들
            'regression_lstm_mse': {
                'type': 'pytorch',
                'model_class': AdvancedRegressionLSTM,
                'loss_function': 'mse',
                'params': {
                    'hidden_size': 128,
                    'num_layers': 3,
                    'dropout': 0.3
                }
            },
            'regression_lstm_mae': {
                'type': 'pytorch',
                'model_class': AdvancedRegressionLSTM,
                'loss_function': 'mae',
                'params': {
                    'hidden_size': 128,
                    'num_layers': 3,
                    'dropout': 0.3
                }
            },
            'regression_lstm_huber': {
                'type': 'pytorch',
                'model_class': AdvancedRegressionLSTM,
                'loss_function': 'huber',
                'params': {
                    'hidden_size': 128,
                    'num_layers': 3,
                    'dropout': 0.3
                }
            },
            'regression_lstm_mape': {
                'type': 'pytorch',
                'model_class': AdvancedRegressionLSTM,
                'loss_function': 'mape',
                'params': {
                    'hidden_size': 128,
                    'num_layers': 3,
                    'dropout': 0.3
                }
            },
            'regression_lstm_quantile': {
                'type': 'pytorch',
                'model_class': AdvancedRegressionLSTM,
                'loss_function': 'quantile',
                'params': {
                    'hidden_size': 128,
                    'num_layers': 3,
                    'dropout': 0.3
                }
            },
            # Sklearn 회귀 모델들
            'xgboost_regression': {
                'type': 'sklearn',
                'model_class': xgb.XGBRegressor,
                'params': {
                    'tree_method': 'gpu_hist',
                    'gpu_id': 0,
                    'n_estimators': 500,
                    'max_depth': 8,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42,
                    'early_stopping_rounds': 50,
                    'eval_metric': 'rmse'
                }
            },
            'random_forest_regression': {
                'type': 'sklearn',
                'model_class': RandomForestRegressor,
                'params': {
                    'n_estimators': 500,
                    'max_depth': 15,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'max_features': 'sqrt',
                    'random_state': 42,
                    'n_jobs': -1
                }
            },
            'gradient_boosting_regression': {
                'type': 'sklearn',
                'model_class': GradientBoostingRegressor,
                'params': {
                    'n_estimators': 300,
                    'learning_rate': 0.05,
                    'max_depth': 8,
                    'subsample': 0.8,
                    'random_state': 42
                }
            },
            'ridge_regression': {
                'type': 'sklearn',
                'model_class': Ridge,
                'params': {
                    'alpha': 1.0,
                    'random_state': 42
                }
            },
            'lasso_regression': {
                'type': 'sklearn',
                'model_class': Lasso,
                'params': {
                    'alpha': 0.001,
                    'random_state': 42,
                    'max_iter': 2000
                }
            }
        }

    def calculate_regression_metrics(self, y_true, y_pred, model_name):
        """회귀 성능 지표 계산"""
        # 기본 지표
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mse)

        # MAPE 계산 (0으로 나누기 방지)
        epsilon = 1e-8
        mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100

        # 방향 정확도 (회귀에서도 방향은 중요)
        y_true_diff = np.diff(y_true)
        y_pred_diff = np.diff(y_pred)
        direction_accuracy = np.mean(np.sign(y_true_diff) == np.sign(y_pred_diff)) * 100

        logger.info(f"{model_name} 회귀 성능:")
        logger.info(f"  MSE: {mse:.6f}")
        logger.info(f"  MAE: {mae:.6f}")
        logger.info(f"  RMSE: {rmse:.6f}")
        logger.info(f"  MAPE: {mape:.2f}%")
        logger.info(f"  R²: {r2:.4f}")
        logger.info(f"  방향 정확도: {direction_accuracy:.2f}%")

        return {
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2': r2,
            'direction_accuracy': direction_accuracy
        }

    def train_sklearn_regression(self, model_name, X_train, X_val, y_train, y_val):
        """Sklearn 회귀 모델 훈련"""
        config = self.model_configs[model_name]
        model = config['model_class'](**config['params'])

        logger.info(f"{model_name} 회귀 훈련 시작")

        # 훈련
        if 'xgboost' in model_name:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X_train, y_train)

        # 예측 및 평가
        y_pred = model.predict(X_val)
        metrics = self.calculate_regression_metrics(y_val, y_pred, model_name)

        return model, metrics

    def train_pytorch_regression(self, model_name, X_train, X_val, y_train, y_val, epochs=150):
        """PyTorch 회귀 모델 훈련"""
        config = self.model_configs[model_name]

        # 모델 초기화
        input_size = X_train.shape[2] if len(X_train.shape) == 3 else X_train.shape[1]
        model = config['model_class'](input_size, **config['params']).to(self.device)

        # 손실 함수 선택
        loss_type = config['loss_function']
        if loss_type == 'mse':
            criterion = nn.MSELoss()
        elif loss_type == 'mae':
            criterion = MAELoss()
        elif loss_type == 'huber':
            criterion = HuberLoss(delta=1.0)
        elif loss_type == 'mape':
            criterion = MAPELoss()
        elif loss_type == 'quantile':
            criterion = QuantileLoss(quantile=0.5)
        else:
            criterion = nn.MSELoss()

        # 옵티마이저
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)

        # 데이터 로더
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train).to(self.device),
            torch.FloatTensor(y_train).to(self.device)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val).to(self.device),
            torch.FloatTensor(y_val).to(self.device)
        )

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        logger.info(f"{model_name} 회귀 훈련 시작 (손실: {loss_type})")

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # 훈련 단계
            model.train()
            train_loss = 0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()

            # 검증 단계
            model.eval()
            val_loss = 0
            val_predictions = []
            val_targets = []

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

                    val_predictions.extend(outputs.cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())

            avg_val_loss = val_loss / len(val_loader)
            scheduler.step(avg_val_loss)

            # 조기 종료
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 25:
                    break

            if epoch % 25 == 0:
                val_mse = mean_squared_error(val_targets, val_predictions)
                val_r2 = r2_score(val_targets, val_predictions)
                logger.info(f"Epoch {epoch}: Val Loss={avg_val_loss:.6f}, MSE={val_mse:.6f}, R²={val_r2:.4f}")

        # 최종 평가
        model.eval()
        with torch.no_grad():
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            final_predictions = model(X_val_tensor).squeeze().cpu().numpy()

        metrics = self.calculate_regression_metrics(y_val, final_predictions, model_name)

        return model, metrics

    def train_all_regression_models(self, X_train, X_val, y_train, y_val, sequence_data=None):
        """모든 회귀 모델 훈련"""
        results = {}

        # Sklearn 모델들 (플랫 데이터)
        sklearn_models = ['xgboost_regression', 'random_forest_regression',
                         'gradient_boosting_regression', 'ridge_regression', 'lasso_regression']

        for model_name in sklearn_models:
            try:
                # 2D 데이터로 변환
                X_train_flat = X_train.reshape(X_train.shape[0], -1) if len(X_train.shape) > 2 else X_train
                X_val_flat = X_val.reshape(X_val.shape[0], -1) if len(X_val.shape) > 2 else X_val

                model, metrics = self.train_sklearn_regression(
                    model_name, X_train_flat, X_val_flat, y_train, y_val
                )

                self.models[model_name] = model
                results[model_name] = {**metrics, 'type': 'sklearn'}

            except Exception as e:
                logger.error(f"{model_name} 훈련 실패: {e}")

        # PyTorch 모델들 (시퀀스 데이터)
        if sequence_data is not None:
            X_seq_train, X_seq_val = sequence_data

            # 시퀀스 길이에 맞게 타겟 조정
            min_train_len = min(len(X_seq_train), len(y_train))
            min_val_len = min(len(X_seq_val), len(y_val))

            y_seq_train = y_train[:min_train_len]
            y_seq_val = y_val[:min_val_len]
            X_seq_train = X_seq_train[:min_train_len]
            X_seq_val = X_seq_val[:min_val_len]

            pytorch_models = ['regression_lstm_mse', 'regression_lstm_mae', 'regression_lstm_huber',
                             'regression_lstm_mape', 'regression_lstm_quantile']

            for model_name in pytorch_models:
                try:
                    model, metrics = self.train_pytorch_regression(
                        model_name, X_seq_train, X_seq_val, y_seq_train, y_seq_val
                    )

                    self.models[model_name] = model
                    results[model_name] = {**metrics, 'type': 'pytorch'}

                except Exception as e:
                    logger.error(f"{model_name} 훈련 실패: {e}")

        return results

    def save_regression_models(self, save_dir="/root/workspace/data/models/regression"):
        """회귀 모델 저장"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        for model_name, model in self.models.items():
            try:
                if self.model_configs[model_name]['type'] == 'sklearn':
                    joblib.dump(model, save_path / f"{model_name}.pkl")
                else:  # pytorch
                    torch.save(model.state_dict(), save_path / f"{model_name}.pth")

                logger.info(f"{model_name} 회귀 모델 저장 완료")

            except Exception as e:
                logger.error(f"{model_name} 저장 실패: {e}")

if __name__ == "__main__":
    # 테스트
    trainer = RegressionMetricTrainer()
    print("✅ 회귀 지표 훈련 시스템 정상 작동")
    print(f"사용 가능한 회귀 모델: {list(trainer.model_configs.keys())}")
    print(f"디바이스: {trainer.device}")