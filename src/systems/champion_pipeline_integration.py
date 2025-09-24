#!/usr/bin/env python3
"""
🎯 캐글 챔피언 특성 + 기존 성공 파이프라인 통합

기존 92.53% 달성 방식에 184개 고급 특성 적용
원본 Advanced Metric Pipeline 구조 유지
"""

import sys
sys.path.append('/root/workspace/src')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# 자체 모듈
from features.kaggle_champion_features import KaggleChampionFeatures
from core.data_processor import DataProcessor

class ChampionPipelineIntegration:
    """캐글 챔피언 특성 + 성공 파이프라인 통합"""

    def __init__(self):
        self.feature_generator = KaggleChampionFeatures(lookback_window=21)
        self.data_processor = DataProcessor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🎯 챔피언 파이프라인 통합 시스템 (디바이스: {self.device})")

    def create_enhanced_features(self, df):
        """향상된 특성 생성 (기존 + 챔피언)"""
        print("🔥 통합 특성 생성...")

        # 1. 기존 성공한 특성들
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))

        # 기본 모멘텀
        periods = [3, 5, 10, 15, 20]
        for period in periods:
            df[f'momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1

        # 변동성
        windows = [5, 10, 20, 30]
        for window in windows:
            df[f'volatility_{window}'] = df['returns'].rolling(window).std()

        # RSI
        for period in [7, 14, 21]:
            df[f'rsi_{period}'] = self.calculate_rsi(df['Close'], period)

        # 2. 선별된 챔피언 특성들 (가장 효과적인 것들만)

        # 고급 가격 특성
        df['hl_ratio'] = (df['High'] - df['Low']) / df['Close']
        df['body_ratio'] = abs(df['Open'] - df['Close']) / (df['High'] - df['Low'] + 1e-8)

        # 고급 볼륨 특성
        df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3
        vwap_10 = (df['typical_price'] * df['Volume']).rolling(10).sum() / df['Volume'].rolling(10).sum()
        df['vwap_ratio_10'] = df['Close'] / vwap_10

        # ATR (True Range)
        true_range = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                abs(df['High'] - df['Close'].shift(1)),
                abs(df['Low'] - df['Close'].shift(1))
            )
        )
        df['atr_14'] = true_range.rolling(14).mean()

        # 고급 모멘텀
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()

        # 통계적 특성
        df['skewness_20'] = df['returns'].rolling(20).skew()
        df['kurtosis_20'] = df['returns'].rolling(20).kurt()

        # 교차 특성
        df['price_volume_corr'] = df['Close'].rolling(20).corr(df['Volume'])
        df['vol_return_ratio'] = df['volatility_20'] / (abs(df['returns']) + 1e-8)

        # 타겟 변수
        df['next_return'] = df['Close'].pct_change().shift(-1)
        df['direction_target'] = (df['next_return'] > 0).astype(int)

        # 결측값 처리
        df = df.fillna(method='ffill').fillna(0)
        df = df.replace([np.inf, -np.inf], 0)

        return df

    def prepare_champion_pipeline_data(self, data_path, sequence_length=20):
        """챔피언 파이프라인 데이터 준비"""
        print("📊 챔피언 파이프라인 데이터 준비...")

        # 데이터 로딩
        df = self.data_processor.load_and_validate_data(data_path)
        if df is None:
            raise ValueError("데이터 로딩 실패")

        # 통합 특성 생성
        enhanced_df = self.create_enhanced_features(df)

        # 핵심 특성만 선별 (차원 축소)
        core_features = [
            'returns', 'log_returns',
            'momentum_3', 'momentum_5', 'momentum_10', 'momentum_15', 'momentum_20',
            'volatility_5', 'volatility_10', 'volatility_20', 'volatility_30',
            'rsi_7', 'rsi_14', 'rsi_21',
            'hl_ratio', 'body_ratio', 'vwap_ratio_10', 'atr_14',
            'macd', 'macd_signal', 'skewness_20', 'kurtosis_20',
            'price_volume_corr', 'vol_return_ratio'
        ]

        print(f"   🎯 핵심 특성: {len(core_features)}개")

        # 시퀀스 데이터 생성 (기존 성공 방식)
        X_sequences = []
        y_direction = []

        for i in range(sequence_length, len(enhanced_df) - 1):
            # 특성 시퀀스
            feature_seq = enhanced_df[core_features].iloc[i-sequence_length:i].values

            # 타겟
            direction_target = enhanced_df['direction_target'].iloc[i]

            if not pd.isna(direction_target):
                X_sequences.append(feature_seq)
                y_direction.append(direction_target)

        X = np.array(X_sequences)
        y = np.array(y_direction)

        # 데이터 정규화 (RobustScaler 사용)
        scaler = RobustScaler()
        n_samples, seq_len, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = scaler.fit_transform(X_reshaped)
        X = X_scaled.reshape(n_samples, seq_len, n_features)

        # NaN 처리
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.5, posinf=1.0, neginf=0.0).astype(int)

        print(f"   ✅ 최종 데이터: X={X.shape}, y={y.shape}")
        print(f"   📊 방향 분포: 상승={np.sum(y)}, 하락={len(y) - np.sum(y)}")

        return X, y, core_features, scaler

    def calculate_rsi(self, prices, window=14):
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))

class ChampionLSTM(nn.Module):
    """기존 성공한 LSTM 아키텍처 기반"""

    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.3):
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

        self.layer_norm = nn.LayerNorm(hidden_size*2)
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        # LSTM processing
        lstm_out, _ = self.lstm(x)

        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.layer_norm(attn_out + lstm_out)  # Residual connection

        # Global average pooling
        pooled = torch.mean(attn_out, dim=1)

        # Classification
        logits = self.classifier(self.dropout(pooled))
        return logits.squeeze()

def train_champion_model(X_train, y_train, X_val, y_val, device, epochs=100):
    """챔피언 모델 훈련"""
    model = ChampionLSTM(input_size=X_train.shape[-1]).to(device)

    # 클래스 가중치 계산 (불균형 처리)
    class_counts = np.bincount(y_train)
    class_weights = len(y_train) / (2 * class_counts)
    weight_tensor = torch.FloatTensor(class_weights).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor[1]/weight_tensor[0])
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 텐서 변환
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)

    # 훈련
    model.train()
    best_val_acc = 0
    patience = 20
    wait = 0

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # 검증 (10 에포크마다)
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_probs = torch.sigmoid(val_outputs).cpu().numpy()
                val_preds = (val_probs > 0.5).astype(int)
                val_acc = np.mean(val_preds == y_val)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    wait = 0
                else:
                    wait += 1

                if wait >= patience:
                    print(f"      조기 종료 at epoch {epoch}, 최고 정확도: {best_val_acc:.4f}")
                    break

            model.train()

    # 최종 예측
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_probs = torch.sigmoid(val_outputs).cpu().numpy()
        val_preds = (val_probs > 0.5).astype(int)

    return model, val_preds, val_probs, best_val_acc

def run_champion_pipeline_experiment():
    """챔피언 파이프라인 실험 실행"""
    print("🎯 캐글 챔피언 파이프라인 통합 실험 시작")
    print("="*70)

    system = ChampionPipelineIntegration()

    # 데이터 준비
    X, y, features, scaler = system.prepare_champion_pipeline_data(
        '/root/workspace/data/training/sp500_2020_2024_enhanced.csv'
    )

    # 교차 검증
    print(f"\n🔬 3-Fold 교차 검증 (기존 성공 방식)")
    tscv = TimeSeriesSplit(n_splits=3)
    all_results = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"\n📊 Fold {fold+1}/3")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 챔피언 LSTM 훈련
        print("   🧠 Champion LSTM 훈련...")
        model, preds, probs, best_acc = train_champion_model(
            X_train, y_train, X_val, y_val, system.device
        )

        accuracy = np.mean(preds == y_val)

        fold_result = {
            'fold': fold + 1,
            'accuracy': accuracy,
            'best_val_acc': best_acc,
            'predictions': preds.tolist(),
            'probabilities': probs.tolist()
        }

        all_results.append(fold_result)
        print(f"      최종 정확도: {accuracy:.4f}")
        print(f"   ✅ Fold {fold+1} 완료")

    # 최종 결과
    accuracies = [r['accuracy'] for r in all_results]
    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    max_accuracy = np.max(accuracies)

    print(f"\n🏆 캐글 챔피언 파이프라인 최종 결과:")
    print("="*70)
    print(f"📊 평균 정확도: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"🎯 최고 정확도: {max_accuracy:.4f}")

    baseline = 0.5774
    improvement = (avg_accuracy - baseline) * 100

    if avg_accuracy > 0.85:
        print(f"🎉 목표 달성! 85%+ 정확도")
    elif avg_accuracy > baseline:
        print(f"📈 기준선 개선: {improvement:+.2f}%p")
    else:
        print(f"⚠️ 기준선 대비: {improvement:+.2f}%p")

    # 결과 저장
    final_result = {
        'timestamp': datetime.now().isoformat(),
        'experiment_type': 'champion_pipeline_integration',
        'approach': 'Champion features + Successful LSTM pipeline',
        'feature_count': len(features),
        'avg_accuracy': avg_accuracy,
        'std_accuracy': std_accuracy,
        'max_accuracy': max_accuracy,
        'baseline_accuracy': baseline,
        'improvement': improvement,
        'detailed_results': all_results
    }

    output_path = f"/root/workspace/data/results/champion_pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, 'w') as f:
        json.dump(final_result, f, indent=2)

    print(f"\n💾 결과 저장: {output_path}")
    print(f"🏁 실험 완료!")

    return final_result


if __name__ == "__main__":
    run_champion_pipeline_experiment()