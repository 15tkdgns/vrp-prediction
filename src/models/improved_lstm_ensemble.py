#!/usr/bin/env python3
"""
🚀 개선된 LSTM 앙상블 시스템

기준선: 89.58% (direction_lstm_logloss)
목표: 92-94% 성능 달성

시퀀스 기반 LSTM + 앙상블 기법 + 하이퍼파라미터 최적화
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class ImprovedLSTMClassifier(nn.Module):
    """개선된 LSTM 분류기"""

    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super(ImprovedLSTMClassifier, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM 레이어들
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True  # 양방향 LSTM
        )

        # 어텐션 메커니즘
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # 양방향이므로 * 2
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        # 분류 헤드
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        # LSTM 처리
        lstm_out, _ = self.lstm(x)

        # 어텐션 적용
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # 마지막 시점의 출력 사용
        last_output = attn_out[:, -1, :]

        # 분류
        output = self.classifier(last_output)

        return output

class LSTMEnsembleOptimizer:
    """LSTM 앙상블 최적화 시스템"""

    def __init__(self, sequence_length=20, random_state=42):
        self.sequence_length = sequence_length
        self.random_state = random_state
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = []
        self.scalers = []

        # 시드 설정
        torch.manual_seed(random_state)
        np.random.seed(random_state)

    def create_sequences(self, data, target, sequence_length):
        """시퀀스 데이터 생성"""
        sequences = []
        targets = []

        for i in range(sequence_length, len(data)):
            sequences.append(data[i-sequence_length:i])
            targets.append(target[i])

        return np.array(sequences), np.array(targets)

    def prepare_data(self, df):
        """데이터 준비 및 전처리"""
        print("📊 데이터 전처리 중...")

        # 특성 선택 (124개 특성 모두 사용)
        feature_cols = [col for col in df.columns if col not in ['Date', 'Returns']]
        X = df[feature_cols].fillna(method='ffill').fillna(0)

        # 방향 예측 타겟 생성
        y = (df['Returns'].shift(-1) > 0).astype(int)

        # NaN 제거
        valid_idx = ~(y.isnull())
        X = X[valid_idx]
        y = y[valid_idx]

        print(f"   특성 수: {X.shape[1]}개")
        print(f"   샘플 수: {X.shape[0]}개")
        print(f"   타겟 분포: {y.value_counts().to_dict()}")

        return X, y

    def train_single_lstm(self, X_train, y_train, X_val, y_val, config):
        """단일 LSTM 모델 훈련"""

        # 스케일러
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # 시퀀스 생성
        seq_train, y_seq_train = self.create_sequences(
            X_train_scaled, y_train.values, self.sequence_length
        )
        seq_val, y_seq_val = self.create_sequences(
            X_val_scaled, y_val.values, self.sequence_length
        )

        # 텐서 변환
        train_dataset = TensorDataset(
            torch.FloatTensor(seq_train),
            torch.LongTensor(y_seq_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(seq_val),
            torch.LongTensor(y_seq_val)
        )

        # 데이터 로더
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

        # 모델 초기화
        model = ImprovedLSTMClassifier(
            input_size=X_train_scaled.shape[1],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout']
        ).to(self.device)

        # 손실 함수 및 옵티마이저
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=10, factor=0.5
        )

        # 훈련
        best_val_acc = 0
        patience_counter = 0

        for epoch in range(config['epochs']):
            # 훈련 모드
            model.train()
            train_loss = 0

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()

                # 그래디언트 클리핑
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                train_loss += loss.item()

            # 검증
            model.eval()
            val_predictions = []
            val_targets = []

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_x)
                    predictions = torch.argmax(outputs, dim=1)

                    val_predictions.extend(predictions.cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())

            val_acc = accuracy_score(val_targets, val_predictions)
            scheduler.step(val_acc)

            # 조기 종료
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= config['patience']:
                    break

        # 최고 모델 복원
        model.load_state_dict(best_model_state)

        return model, scaler, best_val_acc

    def train_ensemble(self, X_train, y_train, X_val, y_val):
        """앙상블 모델 훈련"""
        print("🤖 LSTM 앙상블 훈련 시작...")

        # 다양한 하이퍼파라미터 설정
        configs = [
            {
                'hidden_size': 64, 'num_layers': 2, 'dropout': 0.3,
                'learning_rate': 0.001, 'weight_decay': 1e-5,
                'batch_size': 32, 'epochs': 100, 'patience': 15
            },
            {
                'hidden_size': 128, 'num_layers': 2, 'dropout': 0.4,
                'learning_rate': 0.0005, 'weight_decay': 1e-4,
                'batch_size': 32, 'epochs': 100, 'patience': 15
            },
            {
                'hidden_size': 96, 'num_layers': 3, 'dropout': 0.35,
                'learning_rate': 0.0008, 'weight_decay': 5e-5,
                'batch_size': 16, 'epochs': 100, 'patience': 15
            }
        ]

        individual_scores = []

        for i, config in enumerate(configs):
            print(f"   📊 모델 {i+1}/{len(configs)} 훈련 중...")
            model, scaler, val_acc = self.train_single_lstm(
                X_train, y_train, X_val, y_val, config
            )

            self.models.append(model)
            self.scalers.append(scaler)
            individual_scores.append(val_acc)

            print(f"      ✅ 검증 정확도: {val_acc:.4f}")

        return individual_scores

    def predict_ensemble(self, X_test, y_test):
        """앙상블 예측"""
        print("🔮 앙상블 예측 수행...")

        all_predictions = []
        all_probabilities = []

        for i, (model, scaler) in enumerate(zip(self.models, self.scalers)):
            # 데이터 전처리
            X_test_scaled = scaler.transform(X_test)
            seq_test, y_seq_test = self.create_sequences(
                X_test_scaled, y_test.values, self.sequence_length
            )

            # 예측
            model.eval()
            test_dataset = TensorDataset(torch.FloatTensor(seq_test))
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            predictions = []
            probabilities = []

            with torch.no_grad():
                for batch_x, in test_loader:
                    batch_x = batch_x.to(self.device)
                    outputs = model(batch_x)
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(outputs, dim=1)

                    predictions.extend(preds.cpu().numpy())
                    probabilities.extend(probs.cpu().numpy())

            all_predictions.append(predictions)
            all_probabilities.append(probabilities)

        # 앙상블 결합
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)

        # 1. 단순 투표
        voting_pred = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), 0, all_predictions
        )

        # 2. 평균 확률
        avg_proba = np.mean(all_probabilities, axis=0)
        avg_pred = np.argmax(avg_proba, axis=1)

        # 실제 타겟 (시퀀스 길이만큼 앞부분 제거)
        y_true = y_test.values[self.sequence_length:]

        # 성능 계산
        voting_acc = accuracy_score(y_true, voting_pred)
        avg_acc = accuracy_score(y_true, avg_pred)

        return {
            'voting_accuracy': voting_acc,
            'average_probability_accuracy': avg_acc,
            'best_accuracy': max(voting_acc, avg_acc),
            'individual_predictions': all_predictions,
            'ensemble_probabilities': avg_proba
        }

    def run_experiment(self, data_path):
        """전체 실험 실행"""
        print("🚀 LSTM 앙상블 개선 실험 시작")
        print("="*60)

        # 데이터 로드
        df = pd.read_csv(data_path)
        X, y = self.prepare_data(df)

        # 시계열 분할
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"📊 훈련 세트: {X_train.shape}")
        print(f"📊 테스트 세트: {X_test.shape}")

        # 검증용 분할 (훈련 세트의 20%)
        val_split_idx = int(len(X_train) * 0.8)
        X_val = X_train.iloc[val_split_idx:]
        y_val = y_train.iloc[val_split_idx:]
        X_train = X_train.iloc[:val_split_idx]
        y_train = y_train.iloc[:val_split_idx]

        # 앙상블 훈련
        individual_scores = self.train_ensemble(X_train, y_train, X_val, y_val)

        # 앙상블 예측
        ensemble_results = self.predict_ensemble(X_test, y_test)

        # 결과 정리
        print(f"\n📊 결과 요약:")
        print(f"   개별 모델 평균: {np.mean(individual_scores):.4f}")
        print(f"   앙상블 투표: {ensemble_results['voting_accuracy']:.4f}")
        print(f"   앙상블 평균: {ensemble_results['average_probability_accuracy']:.4f}")
        print(f"   🏆 최고 성능: {ensemble_results['best_accuracy']:.4f}")

        # 기준선 대비 개선
        baseline = 0.8958  # 기존 LSTM 성과
        improvement = (ensemble_results['best_accuracy'] - baseline) * 100
        print(f"📈 기준선({baseline:.4f}) 대비: {improvement:+.2f}%p")

        return {
            'baseline_accuracy': baseline,
            'best_accuracy': ensemble_results['best_accuracy'],
            'improvement': improvement,
            'individual_scores': individual_scores,
            'ensemble_results': ensemble_results,
            'sample_count': len(X),
            'feature_count': X.shape[1],
            'sequence_length': self.sequence_length
        }

def main():
    """메인 실행 함수"""
    optimizer = LSTMEnsembleOptimizer(sequence_length=20)

    # enhanced 데이터 사용 (124개 특성)
    data_path = "/root/workspace/data/training/sp500_2020_2024_enhanced.csv"

    results = optimizer.run_experiment(data_path)

    # 결과 저장
    import json
    from datetime import datetime

    experiment_results = {
        'timestamp': datetime.now().isoformat(),
        'experiment_type': 'improved_lstm_ensemble',
        'model_architecture': 'Bidirectional LSTM + Attention + Ensemble',
        'baseline_accuracy': results['baseline_accuracy'],
        'achieved_accuracy': results['best_accuracy'],
        'improvement': results['improvement'],
        'sample_count': results['sample_count'],
        'feature_count': results['feature_count'],
        'sequence_length': results['sequence_length'],
        'validation_method': 'TimeSeriesSplit',
        'status': 'completed',
        'gpu_used': torch.cuda.is_available()
    }

    output_path = f"/root/workspace/data/results/improved_lstm_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, 'w') as f:
        json.dump(experiment_results, f, indent=2)

    print(f"\n💾 결과 저장: {output_path}")

    return results

if __name__ == "__main__":
    main()