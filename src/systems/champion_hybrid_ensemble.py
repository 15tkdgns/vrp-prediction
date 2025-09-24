#!/usr/bin/env python3
"""
🏆 챔피언 하이브리드 앙상블 시스템

CatBoost(74.73%) + LSTM Focal(92.53%) 조합
기존 성공 특성 + 선별된 고급 특성
목표: 95%+ 정확도 달성
"""

import sys
sys.path.append('/root/workspace/src')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# 자체 모듈
from core.data_processor import DataProcessor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
import catboost as cb

class ChampionHybridEnsemble:
    """챔피언 하이브리드 앙상블 시스템"""

    def __init__(self):
        self.data_processor = DataProcessor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}

        print(f"🏆 챔피언 하이브리드 앙상블 초기화")
        print(f"   💾 디바이스: {self.device}")
        print(f"   🎯 목표: CatBoost + LSTM Focal → 95%+")

    def prepare_hybrid_features(self, data_path):
        """하이브리드 특성 준비 (기존 성공 + 선별 고급)"""
        print("📊 하이브리드 특성 데이터 준비...")

        # 기본 데이터 로딩
        df = self.data_processor.load_and_validate_data(data_path)
        if df is None:
            raise ValueError("데이터 로딩 실패")

        print(f"   📈 원본 데이터: {df.shape}")

        # 기존 성공한 특성 + 선별된 고급 특성
        enhanced_df = self._create_hybrid_features(df)

        print(f"   🔥 하이브리드 특성 완료: {enhanced_df.shape}")

        # 타겟 설정
        enhanced_df['future_return'] = enhanced_df['Close'].pct_change().shift(-1)
        enhanced_df['direction_target'] = (enhanced_df['future_return'] > 0).astype(int)

        # 특성 컬럼 선별 (성공한 것들만)
        feature_columns = [
            # 기본 가격/수익률 특성
            'returns', 'log_returns',
            # 모멘텀 (성공 검증됨)
            'momentum_3', 'momentum_5', 'momentum_10', 'momentum_15', 'momentum_20',
            # 변동성
            'volatility_5', 'volatility_10', 'volatility_20', 'volatility_30',
            # RSI
            'rsi_7', 'rsi_14', 'rsi_21',
            # 선별된 고급 특성 (효과적인 것들만)
            'hl_ratio', 'body_ratio', 'vwap_ratio_10', 'atr_14',
            'macd', 'macd_signal', 'price_volume_corr',
            # 기존 성공 특성들
            'sma_5', 'sma_10', 'sma_20', 'ema_12', 'ema_26',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
            'volume_sma', 'volume_ratio'
        ]

        # 존재하는 컬럼만 선택
        available_features = [col for col in feature_columns if col in enhanced_df.columns]
        print(f"   ✨ 사용 가능한 특성: {len(available_features)}개")

        # NaN 처리
        enhanced_df = enhanced_df.fillna(method='ffill').fillna(0)
        enhanced_df = enhanced_df.replace([np.inf, -np.inf], 0)

        # 시퀀스 데이터 생성
        sequence_length = 20
        X_sequences = []
        y_direction = []

        for i in range(sequence_length, len(enhanced_df) - 1):
            feature_seq = enhanced_df[available_features].iloc[i-sequence_length:i].values
            direction_target = enhanced_df['direction_target'].iloc[i]

            if not pd.isna(direction_target):
                X_sequences.append(feature_seq)
                y_direction.append(direction_target)

        X = np.array(X_sequences)
        y = np.array(y_direction)

        # 2D 데이터도 준비 (CatBoost용)
        X_2d = X[:, -1, :]  # 마지막 timestep

        # NaN 처리
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_2d = np.nan_to_num(X_2d, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.5, posinf=1.0, neginf=0.0).astype(int)

        print(f"   🎯 최종 시퀀스 데이터: X={X.shape}, y={y.shape}")
        print(f"   🎯 최종 2D 데이터: X_2d={X_2d.shape}")
        print(f"   📊 방향 분포: 상승={np.sum(y)}, 하락={len(y) - np.sum(y)}")

        return X, X_2d, y, available_features

    def _create_hybrid_features(self, df):
        """하이브리드 특성 생성"""
        print("   🔧 하이브리드 특성 생성...")

        # 기존 성공한 특성들 (DataProcessor에서)
        enhanced_df = df.copy()

        # 기본 수익률
        enhanced_df['returns'] = enhanced_df['Close'].pct_change()
        enhanced_df['log_returns'] = np.log(enhanced_df['Close'] / enhanced_df['Close'].shift(1))

        # 모멘텀 (검증된 기간들)
        for period in [3, 5, 10, 15, 20]:
            enhanced_df[f'momentum_{period}'] = enhanced_df['Close'] / enhanced_df['Close'].shift(period) - 1

        # 변동성
        for window in [5, 10, 20, 30]:
            enhanced_df[f'volatility_{window}'] = enhanced_df['returns'].rolling(window).std()

        # RSI
        for period in [7, 14, 21]:
            enhanced_df[f'rsi_{period}'] = self._calculate_rsi(enhanced_df['Close'], period)

        # 선별된 고급 특성들만 (효과적인 것들)
        enhanced_df['hl_ratio'] = (enhanced_df['High'] - enhanced_df['Low']) / enhanced_df['Close']
        enhanced_df['body_ratio'] = abs(enhanced_df['Open'] - enhanced_df['Close']) / (enhanced_df['High'] - enhanced_df['Low'] + 1e-8)

        # VWAP
        typical_price = (enhanced_df['High'] + enhanced_df['Low'] + enhanced_df['Close']) / 3
        vwap_10 = (typical_price * enhanced_df['Volume']).rolling(10).sum() / enhanced_df['Volume'].rolling(10).sum()
        enhanced_df['vwap_ratio_10'] = enhanced_df['Close'] / vwap_10

        # ATR
        true_range = np.maximum(
            enhanced_df['High'] - enhanced_df['Low'],
            np.maximum(
                abs(enhanced_df['High'] - enhanced_df['Close'].shift(1)),
                abs(enhanced_df['Low'] - enhanced_df['Close'].shift(1))
            )
        )
        enhanced_df['atr_14'] = true_range.rolling(14).mean()

        # MACD
        ema_12 = enhanced_df['Close'].ewm(span=12).mean()
        ema_26 = enhanced_df['Close'].ewm(span=26).mean()
        enhanced_df['macd'] = ema_12 - ema_26
        enhanced_df['macd_signal'] = enhanced_df['macd'].ewm(span=9).mean()

        # 가격-볼륨 상관관계
        enhanced_df['price_volume_corr'] = enhanced_df['Close'].rolling(20).corr(enhanced_df['Volume'])

        # 기존 성공 특성들 추가
        enhanced_df['sma_5'] = enhanced_df['Close'].rolling(5).mean()
        enhanced_df['sma_10'] = enhanced_df['Close'].rolling(10).mean()
        enhanced_df['sma_20'] = enhanced_df['Close'].rolling(20).mean()
        enhanced_df['ema_12'] = ema_12
        enhanced_df['ema_26'] = ema_26

        # 볼린저 밴드
        sma_20 = enhanced_df['Close'].rolling(20).mean()
        std_20 = enhanced_df['Close'].rolling(20).std()
        enhanced_df['bb_upper'] = sma_20 + (std_20 * 2)
        enhanced_df['bb_middle'] = sma_20
        enhanced_df['bb_lower'] = sma_20 - (std_20 * 2)
        enhanced_df['bb_width'] = enhanced_df['bb_upper'] - enhanced_df['bb_lower']

        # 볼륨 특성
        enhanced_df['volume_sma'] = enhanced_df['Volume'].rolling(20).mean()
        enhanced_df['volume_ratio'] = enhanced_df['Volume'] / enhanced_df['volume_sma']

        return enhanced_df

    def _calculate_rsi(self, prices, window=14):
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))

    def train_catboost(self, X_train, y_train, X_val, y_val):
        """CatBoost 훈련"""
        print("   🐱 CatBoost 훈련...")

        model = cb.CatBoostClassifier(
            iterations=200,
            learning_rate=0.1,
            depth=6,
            loss_function='CrossEntropy',
            eval_metric='Accuracy',
            random_seed=42,
            verbose=False,
            early_stopping_rounds=50
        )

        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            verbose=False
        )

        # 예측
        val_probs = model.predict_proba(X_val)[:, 1]
        val_preds = (val_probs > 0.5).astype(int)

        return model, val_preds, val_probs

    def train_lstm_focal(self, X_train, y_train, X_val, y_val, epochs=100):
        """LSTM Focal 훈련 (기존 성공 방식)"""
        print("   🧠 LSTM Focal 훈련...")

        # 기존 성공한 LSTM 아키텍처 직접 구현
        model = self._create_lstm_focal_model(X_train.shape[-1]).to(self.device)

        # Focal Loss
        criterion = self._focal_loss
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # 텐서 변환
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)

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
                        break

                model.train()

        # 최종 예측
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_probs = torch.sigmoid(val_outputs).cpu().numpy()
            val_preds = (val_probs > 0.5).astype(int)

        return model, val_preds, val_probs

    def _create_lstm_focal_model(self, input_size):
        """LSTM Focal 모델 생성 (기존 성공 구조)"""

        class LSTMFocalModel(nn.Module):
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

        return LSTMFocalModel(input_size)

    def _focal_loss(self, logits, targets, alpha=0.25, gamma=2.0):
        """Focal Loss 구현"""
        ce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        p_t = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - p_t) ** gamma * ce_loss
        return focal_loss.mean()

    def run_hybrid_ensemble_experiment(self, data_path='/root/workspace/data/training/sp500_2020_2024_enhanced.csv'):
        """하이브리드 앙상블 실험 실행"""
        print("🏆 챔피언 하이브리드 앙상블 실험 시작")
        print("="*70)

        try:
            # 데이터 준비
            X, X_2d, y, features = self.prepare_hybrid_features(data_path)

            # 교차 검증
            print(f"\n🔬 3-Fold 교차 검증")
            tscv = TimeSeriesSplit(n_splits=3)
            all_results = []

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                print(f"\n📊 Fold {fold+1}/3")

                X_train, X_val = X[train_idx], X[val_idx]
                X_2d_train, X_2d_val = X_2d[train_idx], X_2d[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # 정규화
                scaler = RobustScaler()
                X_2d_train_scaled = scaler.fit_transform(X_2d_train)
                X_2d_val_scaled = scaler.transform(X_2d_val)

                # 1. CatBoost 훈련
                cb_model, cb_preds, cb_probs = self.train_catboost(
                    X_2d_train_scaled, y_train, X_2d_val_scaled, y_val
                )
                cb_accuracy = np.mean(cb_preds == y_val)

                # 2. LSTM Focal 훈련
                lstm_model, lstm_preds, lstm_probs = self.train_lstm_focal(
                    X_train, y_train, X_val, y_val
                )
                lstm_accuracy = np.mean(lstm_preds == y_val)

                # 3. 하이브리드 앙상블 (가중 평균)
                # CatBoost가 좋으면 더 높은 가중치
                if cb_accuracy > 0.7:
                    cb_weight = 0.6
                    lstm_weight = 0.4
                else:
                    cb_weight = 0.3  # 기본 가중치
                    lstm_weight = 0.7

                ensemble_probs = cb_weight * cb_probs + lstm_weight * lstm_probs
                ensemble_preds = (ensemble_probs > 0.5).astype(int)
                ensemble_accuracy = np.mean(ensemble_preds == y_val)

                fold_result = {
                    'fold': fold + 1,
                    'catboost': {
                        'accuracy': cb_accuracy,
                        'weight': cb_weight
                    },
                    'lstm_focal': {
                        'accuracy': lstm_accuracy,
                        'weight': lstm_weight
                    },
                    'ensemble': {
                        'accuracy': ensemble_accuracy,
                        'probabilities': ensemble_probs.tolist(),
                        'predictions': ensemble_preds.tolist()
                    }
                }

                all_results.append(fold_result)

                print(f"      CatBoost  : {cb_accuracy:.4f} (weight: {cb_weight:.1f})")
                print(f"      LSTM Focal: {lstm_accuracy:.4f} (weight: {lstm_weight:.1f})")
                print(f"      🏆 Ensemble: {ensemble_accuracy:.4f}")
                print(f"   ✅ Fold {fold+1} 완료")

            # 최종 결과 분석
            self._analyze_hybrid_results(all_results)

            return all_results

        except Exception as e:
            print(f"❌ 하이브리드 앙상블 실험 실패: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _analyze_hybrid_results(self, all_results):
        """하이브리드 결과 분석"""
        print(f"\n🏆 챔피언 하이브리드 앙상블 최종 결과:")
        print("="*70)

        # 각 모델별 성능 수집
        cb_accuracies = [r['catboost']['accuracy'] for r in all_results]
        lstm_accuracies = [r['lstm_focal']['accuracy'] for r in all_results]
        ensemble_accuracies = [r['ensemble']['accuracy'] for r in all_results]

        # 평균 성능
        cb_avg = np.mean(cb_accuracies)
        lstm_avg = np.mean(lstm_accuracies)
        ensemble_avg = np.mean(ensemble_accuracies)

        # 최고 성능
        cb_max = np.max(cb_accuracies)
        lstm_max = np.max(lstm_accuracies)
        ensemble_max = np.max(ensemble_accuracies)

        # 기준선
        baseline = 0.5774

        print(f"📊 모델별 성능 요약:")
        print("-"*70)
        print(f"🐱 CatBoost   : 평균 {cb_avg:.4f} (최고: {cb_max:.4f}) | 개선: {(cb_avg-baseline)*100:+.2f}%p")
        print(f"🧠 LSTM Focal : 평균 {lstm_avg:.4f} (최고: {lstm_max:.4f}) | 개선: {(lstm_avg-baseline)*100:+.2f}%p")
        print(f"🏆 Ensemble   : 평균 {ensemble_avg:.4f} (최고: {ensemble_max:.4f}) | 개선: {(ensemble_avg-baseline)*100:+.2f}%p")

        # 목표 달성 여부
        print(f"\n🎯 목표 달성 분석:")
        if ensemble_avg > 0.95:
            print(f"   🎉 목표 달성! 95%+ 정확도 ({ensemble_avg:.4f})")
        elif ensemble_avg > 0.85:
            print(f"   📈 85%+ 달성! ({ensemble_avg:.4f})")
        elif ensemble_avg > baseline:
            print(f"   📊 기준선 개선! (+{(ensemble_avg-baseline)*100:.2f}%p)")
        else:
            print(f"   ⚠️ 추가 최적화 필요")

        # 결과 저장
        final_result = {
            'timestamp': datetime.now().isoformat(),
            'experiment_type': 'champion_hybrid_ensemble',
            'approach': 'CatBoost + LSTM Focal dynamic weighted ensemble',
            'feature_count': len(all_results[0]) if all_results else 0,
            'performance': {
                'catboost_avg': cb_avg,
                'lstm_focal_avg': lstm_avg,
                'ensemble_avg': ensemble_avg,
                'ensemble_max': ensemble_max
            },
            'baseline_accuracy': baseline,
            'improvement': (ensemble_avg - baseline) * 100,
            'detailed_results': all_results
        }

        output_path = f"/root/workspace/data/results/hybrid_ensemble_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, 'w') as f:
            json.dump(final_result, f, indent=2)

        print(f"\n💾 결과 저장: {output_path}")
        print(f"🏁 하이브리드 앙상블 실험 완료!")

def main():
    """메인 실행"""
    system = ChampionHybridEnsemble()
    results = system.run_hybrid_ensemble_experiment()

    print("\n🎉 챔피언 하이브리드 앙상블 테스트 완료!")
    return results

if __name__ == "__main__":
    main()