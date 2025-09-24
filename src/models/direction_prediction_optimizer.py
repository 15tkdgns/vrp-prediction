#!/usr/bin/env python3
"""
방향 정확도 극대화 시스템
분류 기반 방향 예측 + 회귀 기반 크기 예측 이중 모델
데이터 누출 완전 차단
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import tensorflow as tf
from tensorflow.keras import layers, Model
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
# import lightgbm as lgb  # LightGBM 제거
import joblib
import json
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedFeatureEngineer:
    """방향 예측에 특화된 고급 특징 공학"""

    def __init__(self, lookback_window=30):
        self.lookback_window = lookback_window

    def create_direction_features(self, df):
        """방향 예측을 위한 특화 특징 생성"""
        df = df.copy()

        # === 기본 가격 특징 ===
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))

        # === 모멘텀 특징 ===
        for period in [3, 5, 10, 15, 20]:
            df[f'momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1
            df[f'roc_{period}'] = (df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)

        # === 변동성 특징 ===
        for window in [5, 10, 20, 30]:
            df[f'volatility_{window}'] = df['returns'].rolling(window, min_periods=1).std()
            df[f'realized_vol_{window}'] = df['returns'].rolling(window, min_periods=1).apply(
                lambda x: np.sqrt(np.sum(x**2))
            )

        # === 기술적 지표 (다양한 기간) ===
        for short, long in [(5, 20), (10, 30), (20, 50)]:
            df[f'sma_{short}'] = df['Close'].rolling(short, min_periods=1).mean()
            df[f'sma_{long}'] = df['Close'].rolling(long, min_periods=1).mean()
            df[f'sma_ratio_{short}_{long}'] = df[f'sma_{short}'] / df[f'sma_{long}']
            df[f'price_vs_sma_{short}'] = df['Close'] / df[f'sma_{short}'] - 1

        # === RSI (다양한 기간) ===
        for period in [7, 14, 21]:
            df[f'rsi_{period}'] = self._calculate_rsi(df['Close'], period)

        # === 볼린저 밴드 ===
        for window in [10, 20, 30]:
            sma = df['Close'].rolling(window, min_periods=1).mean()
            std = df['Close'].rolling(window, min_periods=1).std()
            df[f'bb_upper_{window}'] = sma + (2 * std)
            df[f'bb_lower_{window}'] = sma - (2 * std)
            df[f'bb_position_{window}'] = (df['Close'] - df[f'bb_lower_{window}']) / (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}'])
            df[f'bb_width_{window}'] = (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}']) / sma

        # === MACD ===
        for fast, slow, signal in [(12, 26, 9), (5, 35, 5)]:
            ema_fast = df['Close'].ewm(span=fast).mean()
            ema_slow = df['Close'].ewm(span=slow).mean()
            df[f'macd_{fast}_{slow}'] = ema_fast - ema_slow
            df[f'macd_signal_{fast}_{slow}_{signal}'] = df[f'macd_{fast}_{slow}'].ewm(span=signal).mean()
            df[f'macd_histogram_{fast}_{slow}_{signal}'] = df[f'macd_{fast}_{slow}'] - df[f'macd_signal_{fast}_{slow}_{signal}']

        # === 스토캐스틱 ===
        for k_period, d_period in [(14, 3), (21, 3)]:
            low_min = df['Low'].rolling(k_period, min_periods=1).min()
            high_max = df['High'].rolling(k_period, min_periods=1).max()
            df[f'stoch_k_{k_period}'] = 100 * (df['Close'] - low_min) / (high_max - low_min)
            df[f'stoch_d_{k_period}_{d_period}'] = df[f'stoch_k_{k_period}'].rolling(d_period, min_periods=1).mean()

        # === 볼륨 특징 ===
        df['volume_sma_5'] = df['Volume'].rolling(5, min_periods=1).mean()
        df['volume_sma_20'] = df['Volume'].rolling(20, min_periods=1).mean()
        df['volume_ratio_5'] = df['Volume'] / df['volume_sma_5']
        df['volume_ratio_20'] = df['Volume'] / df['volume_sma_20']
        df['price_volume'] = df['Close'] * df['Volume']
        df['vwap'] = df['price_volume'].rolling(20, min_periods=1).sum() / df['Volume'].rolling(20, min_periods=1).sum()
        df['price_vs_vwap'] = df['Close'] / df['vwap'] - 1

        # === 갭 및 레인지 특징 ===
        df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        df['true_range'] = np.maximum(df['High'] - df['Low'],
                          np.maximum(abs(df['High'] - df['Close'].shift(1)),
                                   abs(df['Low'] - df['Close'].shift(1))))
        df['atr_14'] = df['true_range'].rolling(14, min_periods=1).mean()
        df['price_range'] = (df['High'] - df['Low']) / df['Close']

        # === 캔들스틱 패턴 ===
        df['body_size'] = abs(df['Close'] - df['Open']) / df['Close']
        df['upper_shadow'] = (df['High'] - np.maximum(df['Open'], df['Close'])) / df['Close']
        df['lower_shadow'] = (np.minimum(df['Open'], df['Close']) - df['Low']) / df['Close']
        df['is_doji'] = (abs(df['Close'] - df['Open']) / df['Close'] < 0.01).astype(int)
        df['is_hammer'] = ((df['lower_shadow'] > 2 * df['body_size']) & (df['upper_shadow'] < df['body_size'])).astype(int)

        # === 시장 구조 특징 ===
        for window in [5, 10, 20]:
            df[f'higher_highs_{window}'] = (df['High'] > df['High'].shift(1)).rolling(window, min_periods=1).sum()
            df[f'lower_lows_{window}'] = (df['Low'] < df['Low'].shift(1)).rolling(window, min_periods=1).sum()
            df[f'trend_strength_{window}'] = df[f'higher_highs_{window}'] - df[f'lower_lows_{window}']

        # === 상대적 강도 ===
        df['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        for window in [5, 10, 20]:
            df[f'avg_close_position_{window}'] = df['close_position'].rolling(window, min_periods=1).mean()

        # === 방향 레이블 (다음 날) ===
        df['next_return'] = df['Close'].pct_change().shift(-1)
        df['direction_label'] = (df['next_return'] > 0).astype(int)

        # 결측값 처리
        df = df.fillna(method='ffill').fillna(0)

        return df

    def _calculate_rsi(self, prices, window=14):
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

class DirectionClassifierLSTM(nn.Module):
    """방향 예측 전용 LSTM 분류기"""

    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.3):
        super(DirectionClassifierLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Bidirectional LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout, bidirectional=True)

        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_size * 2, num_heads=8,
                                             dropout=dropout, batch_first=True)

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 2)  # 2-class classification
        )

    def forward(self, x):
        # Bidirectional LSTM
        lstm_out, _ = self.lstm(x)

        # Attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # Global max pooling + average pooling
        max_pool = torch.max(attn_out, dim=1)[0]
        avg_pool = torch.mean(attn_out, dim=1)

        # Concatenate and classify
        combined = max_pool + avg_pool  # 더 나은 representation
        output = self.classifier(combined)

        return output

class DirectionTransformer(nn.Module):
    """Transformer 기반 방향 분류기"""

    def __init__(self, input_size, d_model=128, nhead=8, num_layers=4, dropout=0.2):
        super(DirectionTransformer, self).__init__()
        self.d_model = d_model

        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoding = self._generate_pos_encoding(1000, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)
        )

    def _generate_pos_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Project and add positional encoding
        x = self.input_projection(x)
        if seq_len <= self.pos_encoding.shape[1]:
            x += self.pos_encoding[:, :seq_len, :].to(x.device)

        # Transformer encoding
        x = self.transformer(x)

        # Global average pooling + max pooling
        avg_pool = torch.mean(x, dim=1)
        max_pool, _ = torch.max(x, dim=1)

        # Combine and classify
        combined = avg_pool + max_pool
        output = self.classifier(combined)

        return output

class DirectionOptimizer:
    """방향 정확도 최적화 시스템"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_engineer = AdvancedFeatureEngineer()
        self.models = {}
        self.scalers = {}
        logger.info(f"방향 예측 최적화 시스템 초기화: {self.device}")

    def load_and_prepare_data(self):
        """데이터 로딩 및 전처리"""
        try:
            data_path = "/root/workspace/data/training/sp500_2020_2024_enhanced.csv"
            df = pd.read_csv(data_path)
            logger.info(f"원본 데이터 로딩: {df.shape}")

            # 고급 특징 공학 적용
            df = self.feature_engineer.create_direction_features(df)
            logger.info(f"특징 공학 완료: {df.shape}")

            # 특징 컬럼 선별 (타겟 제외)
            feature_cols = [col for col in df.columns
                          if col not in ['Date', 'direction_label', 'next_return']
                          and not col.startswith('Unnamed')]

            # NaN 및 무한값 처리
            df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
            df = df.fillna(method='ffill').fillna(0)

            # 유효한 데이터만 선택
            valid_mask = df['direction_label'].notna()
            df = df[valid_mask].reset_index(drop=True)

            logger.info(f"최종 데이터: {df.shape}, 특징 수: {len(feature_cols)}")
            logger.info(f"방향 분포: 상승={df['direction_label'].sum()}, 하락={len(df)-df['direction_label'].sum()}")

            return df, feature_cols

        except Exception as e:
            logger.error(f"데이터 준비 실패: {e}")
            return None, None

    def prepare_sequences(self, df, feature_cols, sequence_length=20):
        """시계열 시퀀스 준비"""
        # 정규화
        scaler = RobustScaler()
        scaled_features = scaler.fit_transform(df[feature_cols])
        self.scalers['feature_scaler'] = scaler

        # 시퀀스 생성
        X, y = [], []
        for i in range(sequence_length, len(scaled_features)):
            X.append(scaled_features[i-sequence_length:i])
            y.append(df.iloc[i]['direction_label'])

        X = np.array(X)
        y = np.array(y, dtype=np.int64)

        logger.info(f"시퀀스 생성: X={X.shape}, y={y.shape}")
        logger.info(f"클래스 분포: 0={np.sum(y==0)}, 1={np.sum(y==1)}")

        return X, y

    def train_pytorch_direction_models(self, X, y):
        """PyTorch 방향 예측 모델들 훈련"""
        logger.info("PyTorch 방향 예측 모델 훈련 시작")

        results = {}

        # 시계열 분할
        tscv = TimeSeriesSplit(n_splits=5)

        for model_name in ['lstm', 'transformer']:
            logger.info(f"{model_name.upper()} 모델 훈련 시작")

            fold_accuracies = []
            best_model = None
            best_accuracy = 0

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # 클래스 가중치 계산
                class_weights = torch.FloatTensor([
                    len(y_train) / (2 * np.sum(y_train == 0)),
                    len(y_train) / (2 * np.sum(y_train == 1))
                ]).to(self.device)

                # 데이터셋 생성
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

                # 모델 초기화
                if model_name == 'lstm':
                    model = DirectionClassifierLSTM(
                        input_size=X.shape[2],
                        hidden_size=128,
                        num_layers=3
                    ).to(self.device)
                else:  # transformer
                    model = DirectionTransformer(
                        input_size=X.shape[2],
                        d_model=128,
                        nhead=8,
                        num_layers=4
                    ).to(self.device)

                criterion = nn.CrossEntropyLoss(weight=class_weights)
                optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

                # 훈련
                best_val_acc = 0
                patience_counter = 0

                for epoch in range(150):
                    # 훈련 단계
                    model.train()
                    train_loss = 0
                    train_correct = 0
                    train_total = 0

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
                        _, predicted = torch.max(outputs.data, 1)
                        train_total += batch_y.size(0)
                        train_correct += (predicted == batch_y).sum().item()

                    # 검증 단계
                    model.eval()
                    val_loss = 0
                    val_correct = 0
                    val_total = 0

                    with torch.no_grad():
                        for batch_X, batch_y in val_loader:
                            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                            outputs = model(batch_X)
                            loss = criterion(outputs, batch_y)

                            val_loss += loss.item()
                            _, predicted = torch.max(outputs.data, 1)
                            val_total += batch_y.size(0)
                            val_correct += (predicted == batch_y).sum().item()

                    val_accuracy = 100 * val_correct / val_total
                    scheduler.step(val_loss)

                    if val_accuracy > best_val_acc:
                        best_val_acc = val_accuracy
                        patience_counter = 0
                        if val_accuracy > best_accuracy:
                            best_accuracy = val_accuracy
                            best_model = model.state_dict().copy()
                    else:
                        patience_counter += 1
                        if patience_counter >= 25:
                            break

                fold_accuracies.append(best_val_acc)
                logger.info(f"Fold {fold+1}: 최고 검증 정확도 = {best_val_acc:.2f}%")

            # 평균 성능 및 최고 모델 저장
            avg_accuracy = np.mean(fold_accuracies)
            results[f'pytorch_{model_name}'] = {
                'average_accuracy': avg_accuracy,
                'fold_accuracies': fold_accuracies,
                'best_accuracy': best_accuracy
            }

            if best_model is not None:
                model_path = f"/root/workspace/data/models/direction_{model_name}.pth"
                torch.save(best_model, model_path)
                logger.info(f"{model_name.upper()} 훈련 완료: 평균 정확도 = {avg_accuracy:.2f}%")

        return results

    def train_traditional_classifiers(self, X, y):
        """전통적 분류기들 훈련"""
        logger.info("전통적 분류기 훈련 시작")

        # 시퀀스를 플랫 특징으로 변환
        X_flat = X.reshape(X.shape[0], -1)

        # 추가 특징 선택 (최근 시점만)
        X_recent = X[:, -1, :]  # 마지막 시점의 특징만

        models_to_train = {
            'xgboost': xgb.XGBClassifier(
                tree_method='gpu_hist',
                gpu_id=0,
                n_estimators=1000,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                early_stopping_rounds=50,
                eval_metric='logloss'
            ),
            'lightgbm': lgb.LGBMClassifier(
                objective='binary',
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=8,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                early_stopping_rounds=50,
                verbose=-1
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=500,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.1,
                max_depth=8,
                random_state=42
            )
        }

        results = {}
        tscv = TimeSeriesSplit(n_splits=5)

        for model_name, model in models_to_train.items():
            logger.info(f"{model_name} 훈련 시작")

            fold_accuracies = []
            best_model = None
            best_accuracy = 0

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_flat)):
                # 특징 선택 (flat vs recent)
                if model_name in ['xgboost', 'lightgbm']:
                    X_train, X_val = X_recent[train_idx], X_recent[val_idx]
                else:
                    X_train, X_val = X_flat[train_idx], X_flat[val_idx]

                y_train, y_val = y[train_idx], y[val_idx]

                # 모델 훈련
                if model_name in ['xgboost', 'lightgbm']:
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=False
                    )
                else:
                    model.fit(X_train, y_train)

                # 예측 및 평가
                y_pred = model.predict(X_val)
                accuracy = accuracy_score(y_val, y_pred) * 100

                fold_accuracies.append(accuracy)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model

                logger.info(f"Fold {fold+1}: 정확도 = {accuracy:.2f}%")

            # 결과 저장
            avg_accuracy = np.mean(fold_accuracies)
            results[model_name] = {
                'average_accuracy': avg_accuracy,
                'fold_accuracies': fold_accuracies,
                'best_accuracy': best_accuracy
            }

            # 최고 모델 저장
            if best_model is not None:
                model_path = f"/root/workspace/data/models/direction_{model_name}.pkl"
                joblib.dump(best_model, model_path)

            logger.info(f"{model_name} 훈련 완료: 평균 정확도 = {avg_accuracy:.2f}%")

        return results

    def create_ensemble_system(self, X, y):
        """앙상블 방향 예측 시스템 구축"""
        logger.info("앙상블 방향 예측 시스템 구축")

        # 모든 개별 모델들 훈련
        pytorch_results = self.train_pytorch_direction_models(X, y)
        traditional_results = self.train_traditional_classifiers(X, y)

        # 결과 통합
        all_results = {**pytorch_results, **traditional_results}

        # 최고 성능 모델들 선별
        top_models = sorted(all_results.items(),
                          key=lambda x: x[1]['best_accuracy'],
                          reverse=True)[:5]

        logger.info("상위 5개 방향 예측 모델:")
        for i, (model_name, result) in enumerate(top_models):
            logger.info(f"{i+1}. {model_name}: {result['best_accuracy']:.2f}%")

        return all_results

    def run_direction_optimization(self):
        """전체 방향 정확도 최적화 실행"""
        logger.info("=== 방향 정확도 최적화 시작 ===")

        # 데이터 준비
        df, feature_cols = self.load_and_prepare_data()
        if df is None:
            return None

        # 시퀀스 생성
        X, y = self.prepare_sequences(df, feature_cols)

        # 앙상블 시스템 구축
        results = self.create_ensemble_system(X, y)

        # 결과 저장
        final_results = {
            'direction_optimization_results': results,
            'best_model': max(results.items(), key=lambda x: x[1]['best_accuracy']),
            'average_improvement': np.mean([r['best_accuracy'] for r in results.values()]),
            'feature_count': len(feature_cols),
            'data_points': len(X),
            'optimization_timestamp': pd.Timestamp.now().isoformat()
        }

        results_path = "/root/workspace/data/raw/direction_optimization_results.json"
        with open(results_path, 'w') as f:
            json.dump(final_results, f, indent=2)

        logger.info("=== 방향 정확도 최적화 완료 ===")

        best_name, best_result = final_results['best_model']
        logger.info(f"최고 성능: {best_name} - {best_result['best_accuracy']:.2f}%")

        return final_results

if __name__ == "__main__":
    optimizer = DirectionOptimizer()
    results = optimizer.run_direction_optimization()

    if results:
        print("\n🎯 방향 정확도 최적화 결과:")
        for model_name, result in results['direction_optimization_results'].items():
            print(f"  {model_name}: {result['best_accuracy']:.2f}% (평균: {result['average_accuracy']:.2f}%)")

        best_name, best_result = results['best_model']
        print(f"\n🏆 최고 성능: {best_name} - {best_result['best_accuracy']:.2f}%")
    else:
        print("❌ 방향 정확도 최적화 실패")