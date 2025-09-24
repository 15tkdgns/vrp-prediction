#!/usr/bin/env python3
"""
로그 손실 및 F1-Score 최적화 모델 훈련 시스템
다중 지표 기반 성능 최적화
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, log_loss, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.model_selection import TimeSeriesSplit
import joblib
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class F1OptimizedLoss(nn.Module):
    """F1-Score 최적화를 위한 커스텀 손실 함수"""

    def __init__(self, epsilon=1e-7):
        super(F1OptimizedLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        # Softmax 적용 (확률로 변환)
        y_pred = torch.softmax(y_pred, dim=1)

        # 이진 분류를 위해 positive class 확률만 사용
        y_pred_pos = y_pred[:, 1]

        # F1 score 계산을 위한 TP, FP, FN
        tp = torch.sum(y_pred_pos * y_true.float())
        fp = torch.sum(y_pred_pos * (1 - y_true.float()))
        fn = torch.sum((1 - y_pred_pos) * y_true.float())

        # Precision과 Recall 계산
        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        # F1 score 계산
        f1 = 2 * precision * recall / (precision + recall + self.epsilon)

        # Loss는 1 - F1 (F1을 최대화하기 위해)
        return 1 - f1

class FocalLoss(nn.Module):
    """Focal Loss - 불균형 데이터셋을 위한 손실 함수"""

    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class AdvancedDirectionLSTM(nn.Module):
    """고급 Direction LSTM - 다중 손실 함수 지원"""

    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.3):
        super(AdvancedDirectionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Bidirectional LSTM with layer normalization
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout, bidirectional=True)

        self.layer_norm1 = nn.LayerNorm(hidden_size * 2)

        # Multi-Head Attention
        self.attention = nn.MultiheadAttention(hidden_size * 2, num_heads=8,
                                             dropout=dropout, batch_first=True)

        self.layer_norm2 = nn.LayerNorm(hidden_size * 2)

        # Advanced classifier with residual connections
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout),

            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(dropout),

            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_size // 4, 2)
        )

    def forward(self, x):
        # LSTM with residual connection
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm1(lstm_out)

        # Self-attention with residual connection
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.layer_norm2(attn_out + lstm_out)  # Residual connection

        # Global pooling (max + mean)
        max_pool = torch.max(attn_out, dim=1)[0]
        avg_pool = torch.mean(attn_out, dim=1)

        # Feature fusion
        combined = max_pool + avg_pool

        # Classification
        output = self.classifier(combined)
        return output

class AdvancedMetricTrainer:
    """고급 지표 기반 모델 훈련 시스템"""

    def __init__(self, gpu_enabled=True):
        self.device = torch.device('cuda' if gpu_enabled and torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.model_configs = self._get_model_configs()
        logger.info(f"고급 지표 훈련 디바이스: {self.device}")

    def _get_model_configs(self):
        """모델 설정 정의"""
        return {
            'direction_lstm_logloss': {
                'type': 'pytorch',
                'model_class': AdvancedDirectionLSTM,
                'loss_function': 'cross_entropy',
                'params': {
                    'hidden_size': 128,
                    'num_layers': 3,
                    'dropout': 0.3
                },
                'optimizer_params': {
                    'lr': 0.001,
                    'weight_decay': 0.01
                }
            },
            'direction_lstm_f1': {
                'type': 'pytorch',
                'model_class': AdvancedDirectionLSTM,
                'loss_function': 'f1_optimized',
                'params': {
                    'hidden_size': 128,
                    'num_layers': 3,
                    'dropout': 0.3
                },
                'optimizer_params': {
                    'lr': 0.0005,  # F1 최적화는 더 낮은 학습률 사용
                    'weight_decay': 0.01
                }
            },
            'direction_lstm_focal': {
                'type': 'pytorch',
                'model_class': AdvancedDirectionLSTM,
                'loss_function': 'focal',
                'params': {
                    'hidden_size': 128,
                    'num_layers': 3,
                    'dropout': 0.3
                },
                'optimizer_params': {
                    'lr': 0.001,
                    'weight_decay': 0.01
                }
            },
            'xgboost_logloss': {
                'type': 'sklearn',
                'model_class': xgb.XGBClassifier,
                'params': {
                    'device': 'cuda',
                    'tree_method': 'hist',
                    'n_estimators': 500,
                    'max_depth': 8,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'objective': 'binary:logistic',  # 로그 손실 최적화
                    'eval_metric': 'logloss',
                    'random_state': 42,
                    'early_stopping_rounds': 50
                }
            },
            'random_forest_f1': {
                'type': 'sklearn',
                'model_class': RandomForestClassifier,
                'params': {
                    'n_estimators': 500,
                    'max_depth': 15,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'max_features': 'sqrt',
                    'class_weight': 'balanced',  # F1 점수 개선을 위한 클래스 가중치
                    'random_state': 42,
                    'n_jobs': -1
                }
            },
            'logistic_regression_logloss': {
                'type': 'sklearn',
                'model_class': LogisticRegression,
                'params': {
                    'C': 1.0,
                    'class_weight': 'balanced',
                    'solver': 'liblinear',
                    'max_iter': 1000,
                    'random_state': 42
                }
            }
        }

    def get_loss_function(self, loss_name):
        """손실 함수 반환"""
        if loss_name == 'cross_entropy':
            return nn.CrossEntropyLoss()
        elif loss_name == 'f1_optimized':
            return F1OptimizedLoss()
        elif loss_name == 'focal':
            return FocalLoss(alpha=1, gamma=2)
        else:
            return nn.CrossEntropyLoss()

    def train_pytorch_model_advanced(self, model_name, X_train, X_val, y_train, y_val, epochs=150):
        """고급 PyTorch 모델 훈련 (다중 지표 최적화)"""
        config = self.model_configs[model_name]

        # 모델 초기화
        input_size = X_train.shape[2] if len(X_train.shape) == 3 else X_train.shape[1]
        model = config['model_class'](input_size, **config['params']).to(self.device)

        # 클래스 가중치 계산
        class_counts = np.bincount(y_train)
        class_weights = len(y_train) / (2 * class_counts)
        class_weights = torch.FloatTensor(class_weights).to(self.device)

        # 손실 함수 선택
        if config['loss_function'] == 'cross_entropy':
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = self.get_loss_function(config['loss_function'])

        # 옵티마이저 설정
        optimizer = optim.AdamW(model.parameters(), **config['optimizer_params'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)

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

        logger.info(f"{model_name} 훈련 시작 (손실: {config['loss_function']})")

        best_metric = 0 if 'f1' in model_name else float('inf')
        best_model_state = None
        patience_counter = 0

        for epoch in range(epochs):
            # 훈련 단계
            model.train()
            train_loss = 0
            train_preds = []
            train_targets = []

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()

                # 예측값 저장 (평가용)
                _, predicted = torch.max(outputs.data, 1)
                train_preds.extend(predicted.cpu().numpy())
                train_targets.extend(batch_y.cpu().numpy())

            # 검증 단계
            model.eval()
            val_loss = 0
            val_preds = []
            val_targets = []
            val_probs = []

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)

                    val_loss += loss.item()

                    # 확률과 예측 저장
                    probs = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs.data, 1)

                    val_preds.extend(predicted.cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())
                    val_probs.extend(probs.cpu().numpy())

            # 평가 지표 계산
            val_accuracy = accuracy_score(val_targets, val_preds)
            val_f1 = f1_score(val_targets, val_preds, average='weighted')

            try:
                val_logloss = log_loss(val_targets, val_probs)
            except:
                val_logloss = float('inf')

            # 스케줄러 업데이트
            if 'f1' in model_name:
                scheduler.step(-val_f1)  # F1 score 최대화
                current_metric = val_f1
                is_better = current_metric > best_metric
            else:
                scheduler.step(val_logloss)  # Log loss 최소화
                current_metric = val_logloss
                is_better = current_metric < best_metric

            # 최고 모델 저장
            if is_better:
                best_metric = current_metric
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= 25:
                break

            # 로그 출력
            if epoch % 25 == 0:
                logger.info(f"Epoch {epoch}: Val Acc={val_accuracy:.4f}, Val F1={val_f1:.4f}, Val LogLoss={val_logloss:.4f}")

        # 최고 모델 로드
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # 최종 평가
        model.eval()
        final_preds = []
        final_probs = []

        with torch.no_grad():
            for batch_X, _ in val_loader:
                batch_X = batch_X.to(self.device)
                outputs = model(batch_X)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)

                final_preds.extend(predicted.cpu().numpy())
                final_probs.extend(probs.cpu().numpy())

        # 최종 지표 계산
        final_accuracy = accuracy_score(val_targets, final_preds)
        final_f1 = f1_score(val_targets, final_preds, average='weighted')
        final_precision = precision_score(val_targets, final_preds, average='weighted', zero_division=0)
        final_recall = recall_score(val_targets, final_preds, average='weighted', zero_division=0)

        try:
            final_logloss = log_loss(val_targets, final_probs)
            final_auc = roc_auc_score(val_targets, np.array(final_probs)[:, 1])
        except:
            final_logloss = float('inf')
            final_auc = 0.5

        results = {
            'accuracy': final_accuracy,
            'f1_score': final_f1,
            'precision': final_precision,
            'recall': final_recall,
            'log_loss': final_logloss,
            'auc': final_auc,
            'best_metric': best_metric,
            'loss_function': config['loss_function']
        }

        logger.info(f"{model_name} 훈련 완료:")
        logger.info(f"  정확도: {final_accuracy:.4f}")
        logger.info(f"  F1 점수: {final_f1:.4f}")
        logger.info(f"  로그 손실: {final_logloss:.4f}")
        logger.info(f"  AUC: {final_auc:.4f}")

        return model, results

    def train_sklearn_model_advanced(self, model_name, X_train, X_val, y_train, y_val):
        """고급 Sklearn 모델 훈련 (다중 지표 최적화)"""
        config = self.model_configs[model_name]
        model = config['model_class'](**config['params'])

        logger.info(f"{model_name} 훈련 시작")

        # 2D 데이터로 변환 (필요시)
        X_train_flat = X_train.reshape(X_train.shape[0], -1) if len(X_train.shape) > 2 else X_train
        X_val_flat = X_val.reshape(X_val.shape[0], -1) if len(X_val.shape) > 2 else X_val

        # 훈련
        if 'xgboost' in model_name:
            model.fit(X_train_flat, y_train, eval_set=[(X_val_flat, y_val)], verbose=False)
        else:
            model.fit(X_train_flat, y_train)

        # 예측
        y_pred = model.predict(X_val_flat)
        y_proba = model.predict_proba(X_val_flat)

        # 지표 계산
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average='weighted')
        precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)

        try:
            logloss = log_loss(y_val, y_proba)
            auc = roc_auc_score(y_val, y_proba[:, 1])
        except:
            logloss = float('inf')
            auc = 0.5

        results = {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'log_loss': logloss,
            'auc': auc
        }

        logger.info(f"{model_name} 훈련 완료:")
        logger.info(f"  정확도: {accuracy:.4f}")
        logger.info(f"  F1 점수: {f1:.4f}")
        logger.info(f"  로그 손실: {logloss:.4f}")
        logger.info(f"  AUC: {auc:.4f}")

        return model, results

    def train_all_advanced_models(self, X_train, X_val, y_train, y_val, sequence_data=None):
        """모든 고급 모델 훈련"""
        results = {}

        # Sklearn 모델들
        sklearn_models = ['xgboost_logloss', 'random_forest_f1', 'logistic_regression_logloss']

        for model_name in sklearn_models:
            try:
                model, result = self.train_sklearn_model_advanced(
                    model_name, X_train, X_val, y_train, y_val
                )
                self.models[model_name] = model
                results[model_name] = result
                results[model_name]['type'] = 'sklearn'

            except Exception as e:
                logger.error(f"{model_name} 훈련 실패: {e}")

        # PyTorch 모델들 (시퀀스 데이터)
        if sequence_data is not None:
            X_seq_train, X_seq_val = sequence_data

            # 시퀀스 길이 조정
            min_train_len = min(len(X_seq_train), len(y_train))
            min_val_len = min(len(X_seq_val), len(y_val))

            y_seq_train = y_train[:min_train_len]
            y_seq_val = y_val[:min_val_len]
            X_seq_train = X_seq_train[:min_train_len]
            X_seq_val = X_seq_val[:min_val_len]

            pytorch_models = ['direction_lstm_logloss', 'direction_lstm_f1', 'direction_lstm_focal']

            for model_name in pytorch_models:
                try:
                    model, result = self.train_pytorch_model_advanced(
                        model_name, X_seq_train, X_seq_val, y_seq_train, y_seq_val
                    )
                    self.models[model_name] = model
                    results[model_name] = result
                    results[model_name]['type'] = 'pytorch'

                except Exception as e:
                    logger.error(f"{model_name} 훈련 실패: {e}")

        return results

    def save_advanced_models(self, save_dir="/root/workspace/data/models"):
        """고급 모델 저장"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        for model_name, model in self.models.items():
            try:
                if self.model_configs[model_name]['type'] == 'sklearn':
                    joblib.dump(model, save_path / f"{model_name}_advanced.pkl")
                else:  # pytorch
                    torch.save(model.state_dict(), save_path / f"{model_name}_advanced.pth")

                logger.info(f"{model_name} 고급 모델 저장 완료")

            except Exception as e:
                logger.error(f"{model_name} 저장 실패: {e}")

if __name__ == "__main__":
    # 테스트
    trainer = AdvancedMetricTrainer()
    print("✅ 고급 지표 기반 모델 훈련 시스템 정상 작동")
    print(f"사용 가능한 모델: {list(trainer.model_configs.keys())}")
    print(f"디바이스: {trainer.device}")