#!/usr/bin/env python3
"""
통합 모델 훈련 시스템
다양한 알고리즘의 훈련을 통합 관리
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DirectionLSTM(nn.Module):
    """방향 예측용 LSTM"""

    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.3):
        super(DirectionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout, bidirectional=True)

        self.attention = nn.MultiheadAttention(hidden_size * 2, num_heads=8,
                                             dropout=dropout, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 2)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Global pooling
        max_pool = torch.max(attn_out, dim=1)[0]
        avg_pool = torch.mean(attn_out, dim=1)
        combined = max_pool + avg_pool

        return self.classifier(combined)

class ModelTrainer:
    """통합 모델 훈련 시스템"""

    def __init__(self, gpu_enabled=True):
        self.device = torch.device('cuda' if gpu_enabled and torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.model_configs = self._get_model_configs()
        logger.info(f"훈련 디바이스: {self.device}")

    def _get_model_configs(self):
        """모델 설정 정의"""
        return {
            'xgboost_gpu': {
                'type': 'sklearn',
                'model_class': xgb.XGBClassifier,
                'params': {
                    'tree_method': 'gpu_hist',
                    'gpu_id': 0,
                    'n_estimators': 300,
                    'max_depth': 8,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42,
                    'early_stopping_rounds': 50,
                    'eval_metric': 'logloss'
                }
            },
            'random_forest': {
                'type': 'sklearn',
                'model_class': RandomForestClassifier,
                'params': {
                    'n_estimators': 300,
                    'max_depth': 15,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'max_features': 'sqrt',
                    'random_state': 42,
                    'n_jobs': -1
                }
            },
            'gradient_boosting': {
                'type': 'sklearn',
                'model_class': GradientBoostingClassifier,
                'params': {
                    'n_estimators': 200,
                    'learning_rate': 0.1,
                    'max_depth': 8,
                    'subsample': 0.8,
                    'random_state': 42
                }
            },
            'logistic_regression': {
                'type': 'sklearn',
                'model_class': LogisticRegression,
                'params': {
                    'C': 1.0,
                    'max_iter': 1000,
                    'random_state': 42
                }
            },
            'direction_lstm': {
                'type': 'pytorch',
                'model_class': DirectionLSTM,
                'params': {
                    'hidden_size': 128,
                    'num_layers': 3,
                    'dropout': 0.3
                }
            }
        }

    def train_sklearn_model(self, model_name, X_train, X_val, y_train, y_val):
        """Scikit-learn 모델 훈련"""
        config = self.model_configs[model_name]
        model = config['model_class'](**config['params'])

        logger.info(f"{model_name} 훈련 시작")

        # 훈련
        if 'xgboost' in model_name:
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X_train, y_train)

        # 검증
        val_pred = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_pred)

        logger.info(f"{model_name} 검증 정확도: {val_accuracy:.4f}")

        return model, val_accuracy

    def train_pytorch_model(self, model_name, X_train, X_val, y_train, y_val, epochs=100):
        """PyTorch 모델 훈련"""
        config = self.model_configs[model_name]

        # 모델 초기화
        input_size = X_train.shape[2] if len(X_train.shape) == 3 else X_train.shape[1]
        model = config['model_class'](input_size, **config['params']).to(self.device)

        # 클래스 가중치
        class_counts = np.bincount(y_train)
        class_weights = len(y_train) / (2 * class_counts)
        class_weights = torch.FloatTensor(class_weights).to(self.device)

        # 손실함수 및 옵티마이저
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        # 데이터 로더
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val)
        )

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        logger.info(f"{model_name} 훈련 시작")

        best_val_acc = 0
        patience_counter = 0

        for epoch in range(epochs):
            # 훈련 단계
            model.train()
            train_loss = 0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()

            # 검증 단계
            model.eval()
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_X)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()

            val_accuracy = val_correct / val_total
            scheduler.step(1 - val_accuracy)

            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    break

            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}: 검증 정확도 = {val_accuracy:.4f}")

        logger.info(f"{model_name} 최고 검증 정확도: {best_val_acc:.4f}")

        return model, best_val_acc

    def train_all_models(self, X_train, X_val, y_train, y_val, sequence_data=None):
        """모든 모델 훈련"""
        results = {}

        # Sklearn 모델들 (플랫 데이터)
        sklearn_models = ['xgboost_gpu', 'random_forest', 'gradient_boosting', 'logistic_regression']

        for model_name in sklearn_models:
            try:
                # 2D 데이터로 변환 (필요시)
                X_train_flat = X_train.reshape(X_train.shape[0], -1) if len(X_train.shape) > 2 else X_train
                X_val_flat = X_val.reshape(X_val.shape[0], -1) if len(X_val.shape) > 2 else X_val

                model, accuracy = self.train_sklearn_model(
                    model_name, X_train_flat, X_val_flat, y_train, y_val
                )

                self.models[model_name] = model
                results[model_name] = {
                    'accuracy': accuracy,
                    'type': 'sklearn'
                }

            except Exception as e:
                logger.error(f"{model_name} 훈련 실패: {e}")

        # PyTorch 모델들 (시퀀스 데이터)
        if sequence_data is not None:
            X_seq_train, X_seq_val = sequence_data

            # 시퀀스 길이만큼 타겟도 조정 (최소값 사용)
            min_train_len = min(len(X_seq_train), len(y_train))
            min_val_len = min(len(X_seq_val), len(y_val))

            y_seq_train = y_train[:min_train_len]
            y_seq_val = y_val[:min_val_len]
            X_seq_train = X_seq_train[:min_train_len]
            X_seq_val = X_seq_val[:min_val_len]

            pytorch_models = ['direction_lstm']

            for model_name in pytorch_models:
                try:
                    model, accuracy = self.train_pytorch_model(
                        model_name, X_seq_train, X_seq_val, y_seq_train, y_seq_val
                    )

                    self.models[model_name] = model
                    results[model_name] = {
                        'accuracy': accuracy,
                        'type': 'pytorch'
                    }

                except Exception as e:
                    logger.error(f"{model_name} 훈련 실패: {e}")

        return results

    def save_models(self, save_dir="/root/workspace/data/models"):
        """모델 저장"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        for model_name, model in self.models.items():
            try:
                if self.model_configs[model_name]['type'] == 'sklearn':
                    joblib.dump(model, save_path / f"{model_name}.pkl")
                else:  # pytorch
                    torch.save(model.state_dict(), save_path / f"{model_name}.pth")

                logger.info(f"{model_name} 모델 저장 완료")

            except Exception as e:
                logger.error(f"{model_name} 저장 실패: {e}")

    def create_ensemble_prediction(self, X, model_subset=None):
        """앙상블 예측"""
        if model_subset is None:
            model_subset = list(self.models.keys())

        predictions = []

        for model_name in model_subset:
            if model_name not in self.models:
                continue

            try:
                model = self.models[model_name]

                if self.model_configs[model_name]['type'] == 'sklearn':
                    # 2D 데이터로 변환
                    X_flat = X.reshape(X.shape[0], -1) if len(X.shape) > 2 else X
                    pred_proba = model.predict_proba(X_flat)[:, 1]
                else:  # pytorch
                    model.eval()
                    with torch.no_grad():
                        X_tensor = torch.FloatTensor(X).to(self.device)
                        outputs = model(X_tensor)
                        pred_proba = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()

                predictions.append(pred_proba)

            except Exception as e:
                logger.error(f"{model_name} 예측 실패: {e}")

        if predictions:
            # 단순 평균 앙상블
            ensemble_pred = np.mean(predictions, axis=0)
            return ensemble_pred
        else:
            return None

if __name__ == "__main__":
    # 테스트
    trainer = ModelTrainer()
    print("✅ 모델 훈련 시스템 정상 작동")
    print(f"사용 가능한 모델: {list(trainer.model_configs.keys())}")
    print(f"디바이스: {trainer.device}")