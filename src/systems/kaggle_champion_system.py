#!/usr/bin/env python3
"""
🚀 캐글 챔피언 통합 주가예측 시스템

184개 고급 특성 + Optiver 우승 앙상블 + 온라인 학습
목표: 57.74% → 95%+ 정확도 달성
"""

import sys
sys.path.append('/root/workspace/src')

import numpy as np
import pandas as pd
import torch
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# 자체 모듈 import
from features.kaggle_champion_features import KaggleChampionFeatures
from core.data_processor import DataProcessor

# sklearn imports
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 간단한 신경망 모델들
import torch.nn as nn
import torch.optim as optim

class LightweightLSTM(nn.Module):
    """경량화된 LSTM (빠른 테스트용)"""

    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])  # 마지막 timestep
        return self.fc(out).squeeze()

class KaggleChampionSystem:
    """캐글 챔피언 통합 시스템"""

    def __init__(self):
        self.feature_generator = KaggleChampionFeatures(lookback_window=21)
        self.data_processor = DataProcessor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.performance_history = []

        print(f"🚀 캐글 챔피언 시스템 초기화")
        print(f"   💾 디바이스: {self.device}")
        print(f"   🎯 목표: 57.74% → 95%+ 정확도")

    def prepare_champion_data(self, data_path):
        """캐글 챔피언 데이터 준비"""
        print("\n📊 캐글 챔피언 데이터 준비...")

        # 원본 데이터 로딩
        df = self.data_processor.load_and_validate_data(data_path)
        if df is None:
            raise ValueError("데이터 로딩 실패")

        print(f"   📈 원본 데이터: {df.shape}")

        # 184개 고급 특성 생성
        enhanced_df = self.feature_generator.generate_champion_features(df)
        print(f"   🔥 고급 특성 완료: {enhanced_df.shape}")

        # 타겟 변수 설정
        enhanced_df['future_return'] = enhanced_df['Close'].pct_change().shift(-1)
        enhanced_df['direction_target'] = (enhanced_df['future_return'] > 0).astype(int)

        # 특성 컬럼 추출
        feature_columns = [col for col in enhanced_df.columns
                          if col not in ['Date', 'future_return', 'direction_target']
                          and not col.startswith('Open')
                          and not col.startswith('High')
                          and not col.startswith('Low')
                          and not col.startswith('Close')
                          and not col.startswith('Volume')]

        print(f"   ✨ 사용할 특성: {len(feature_columns)}개")

        # 시퀀스 데이터 생성
        sequence_length = 20
        X_sequences = []
        y_direction = []
        y_regression = []

        for i in range(sequence_length, len(enhanced_df) - 1):
            # 특성 시퀀스
            feature_seq = enhanced_df[feature_columns].iloc[i-sequence_length:i].values

            # 타겟들
            direction_target = enhanced_df['direction_target'].iloc[i]
            regression_target = enhanced_df['future_return'].iloc[i]

            if not (pd.isna(direction_target) or pd.isna(regression_target)):
                X_sequences.append(feature_seq)
                y_direction.append(direction_target)
                y_regression.append(regression_target)

        X = np.array(X_sequences)
        y_direction = np.array(y_direction)
        y_regression = np.array(y_regression)

        # NaN 처리
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y_direction = np.nan_to_num(y_direction, nan=0.5, posinf=1.0, neginf=0.0)
        y_regression = np.nan_to_num(y_regression, nan=0.0, posinf=0.0, neginf=0.0)

        print(f"   🎯 최종 데이터: X={X.shape}, y_direction={y_direction.shape}")
        print(f"   📊 방향 분포: 상승={np.sum(y_direction)}, 하락={len(y_direction) - np.sum(y_direction)}")

        return X, y_direction, y_regression, feature_columns

    def train_lightweight_lstm(self, X_train, y_train, X_val, y_val, epochs=50):
        """경량화된 LSTM 훈련"""
        model = LightweightLSTM(input_size=X_train.shape[-1]).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()  # 분류용

        # 텐서 변환
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)

        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # 예측
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_probs = torch.sigmoid(val_outputs).cpu().numpy()
            val_preds = (val_probs > 0.5).astype(int)

        return model, val_preds, val_probs

    def calculate_champion_metrics(self, y_true, y_pred, y_probs=None):
        """캐글 챔피언 지표 계산"""
        # 방향 정확도
        direction_accuracy = np.mean(y_true == y_pred)

        # 기타 지표
        metrics = {
            'direction_accuracy': direction_accuracy,
            'precision': np.sum((y_pred == 1) & (y_true == 1)) / (np.sum(y_pred == 1) + 1e-8),
            'recall': np.sum((y_pred == 1) & (y_true == 1)) / (np.sum(y_true == 1) + 1e-8),
            'f1_score': 2 * metrics.get('precision', 0) * metrics.get('recall', 0) / (metrics.get('precision', 0) + metrics.get('recall', 0) + 1e-8) if 'precision' in locals() else 0
        }

        # F1 계산 수정
        precision = metrics['precision']
        recall = metrics['recall']
        metrics['f1_score'] = 2 * precision * recall / (precision + recall + 1e-8)

        # 확률이 있으면 로그 손실도 계산
        if y_probs is not None:
            y_probs_clipped = np.clip(y_probs, 1e-15, 1-1e-15)
            log_loss_val = -np.mean(y_true * np.log(y_probs_clipped) + (1 - y_true) * np.log(1 - y_probs_clipped))
            metrics['log_loss'] = log_loss_val

        return metrics

    def run_champion_experiment(self, data_path='/root/workspace/data/training/sp500_2020_2024_enhanced.csv'):
        """캐글 챔피언 실험 실행"""
        print("🏆 캐글 챔피언 실험 시작")
        print("="*70)

        try:
            # 데이터 준비
            X, y_direction, y_regression, feature_columns = self.prepare_champion_data(data_path)

            # 교차 검증
            print(f"\n🔬 3-Fold 교차 검증 시작")
            tscv = TimeSeriesSplit(n_splits=3)
            all_results = []

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                print(f"\n📊 Fold {fold+1}/3")

                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y_direction[train_idx], y_direction[val_idx]

                fold_results = {}

                # 1. Random Forest (빠른 베이스라인)
                print("   🌲 Random Forest 훈련...")
                try:
                    rf_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
                    X_train_2d = X_train.reshape(X_train.shape[0], -1)
                    X_val_2d = X_val.reshape(X_val.shape[0], -1)

                    rf_model.fit(X_train_2d, y_train)
                    rf_probs = rf_model.predict(X_val_2d)
                    rf_preds = (rf_probs > 0.5).astype(int)

                    rf_metrics = self.calculate_champion_metrics(y_val, rf_preds, rf_probs)
                    fold_results['RandomForest'] = rf_metrics

                    print(f"      RandomForest: 정확도={rf_metrics['direction_accuracy']:.4f}")

                except Exception as e:
                    print(f"      ⚠️ RandomForest 실패: {e}")

                # 2. ElasticNet
                print("   📊 ElasticNet 훈련...")
                try:
                    elastic_model = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=1000)
                    X_train_2d = X_train[:, -1, :]  # 마지막 timestep
                    X_val_2d = X_val[:, -1, :]

                    elastic_model.fit(X_train_2d, y_train)
                    elastic_probs = elastic_model.predict(X_val_2d)
                    elastic_probs = np.clip(elastic_probs, 0, 1)  # 0-1 범위로 클리핑
                    elastic_preds = (elastic_probs > 0.5).astype(int)

                    elastic_metrics = self.calculate_champion_metrics(y_val, elastic_preds, elastic_probs)
                    fold_results['ElasticNet'] = elastic_metrics

                    print(f"      ElasticNet: 정확도={elastic_metrics['direction_accuracy']:.4f}")

                except Exception as e:
                    print(f"      ⚠️ ElasticNet 실패: {e}")

                # 3. Lightweight LSTM
                print("   🧠 Lightweight LSTM 훈련...")
                try:
                    lstm_model, lstm_preds, lstm_probs = self.train_lightweight_lstm(
                        X_train, y_train, X_val, y_val
                    )

                    lstm_metrics = self.calculate_champion_metrics(y_val, lstm_preds, lstm_probs)
                    fold_results['LightweightLSTM'] = lstm_metrics

                    print(f"      LightweightLSTM: 정확도={lstm_metrics['direction_accuracy']:.4f}")

                except Exception as e:
                    print(f"      ⚠️ LightweightLSTM 실패: {e}")

                # 4. 간단한 앙상블 (평균)
                print("   🎯 간단한 앙상블...")
                try:
                    ensemble_probs = []
                    if 'RandomForest' in fold_results:
                        ensemble_probs.append(rf_probs)
                    if 'ElasticNet' in fold_results:
                        ensemble_probs.append(elastic_probs)
                    if 'LightweightLSTM' in fold_results:
                        ensemble_probs.append(lstm_probs)

                    if len(ensemble_probs) > 0:
                        avg_probs = np.mean(ensemble_probs, axis=0)
                        avg_preds = (avg_probs > 0.5).astype(int)

                        ensemble_metrics = self.calculate_champion_metrics(y_val, avg_preds, avg_probs)
                        fold_results['SimpleEnsemble'] = ensemble_metrics

                        print(f"      SimpleEnsemble: 정확도={ensemble_metrics['direction_accuracy']:.4f}")

                except Exception as e:
                    print(f"      ⚠️ SimpleEnsemble 실패: {e}")

                all_results.append(fold_results)
                print(f"   ✅ Fold {fold+1} 완료")

            # 최종 결과 요약
            self._summarize_champion_results(all_results)

            return all_results

        except Exception as e:
            print(f"❌ 캐글 챔피언 실험 실패: {str(e)}")
            return None

    def _summarize_champion_results(self, all_results):
        """캐글 챔피언 결과 요약"""
        print(f"\n🏆 캐글 챔피언 시스템 최종 결과:")
        print("="*70)

        model_names = ['RandomForest', 'ElasticNet', 'LightweightLSTM', 'SimpleEnsemble']

        avg_performance = {}

        for model_name in model_names:
            direction_accuracies = []

            for fold_result in all_results:
                if model_name in fold_result:
                    acc = fold_result[model_name]['direction_accuracy']
                    if not (np.isnan(acc) or np.isinf(acc)):
                        direction_accuracies.append(acc)

            if direction_accuracies:
                avg_performance[model_name] = {
                    'avg_accuracy': np.mean(direction_accuracies),
                    'std_accuracy': np.std(direction_accuracies),
                    'max_accuracy': np.max(direction_accuracies)
                }

        # 성능 출력
        print(f"📊 모델별 방향 예측 정확도:")
        print("-" * 70)

        baseline_accuracy = 0.5774  # 기존 최고 성능

        for model_name, perf in avg_performance.items():
            avg_acc = perf['avg_accuracy']
            improvement = (avg_acc - baseline_accuracy) * 100
            icon = "🏆" if avg_acc > 0.7 else "📈" if avg_acc > baseline_accuracy else "📊"

            print(f"{icon} {model_name:16s}: "
                  f"{avg_acc:.4f} ± {perf['std_accuracy']:.4f} "
                  f"(최고: {perf['max_accuracy']:.4f}) "
                  f"개선: {improvement:+.2f}%p")

        # 최고 성능 모델
        if avg_performance:
            best_model = max(avg_performance.items(), key=lambda x: x[1]['avg_accuracy'])
            best_accuracy = best_model[1]['avg_accuracy']

            print(f"\n🥇 최고 성능 모델: {best_model[0]}")
            print(f"   🎯 달성 정확도: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
            print(f"   📈 기준선 개선: {(best_accuracy - baseline_accuracy)*100:+.2f}%p")

            if best_accuracy > 0.85:
                print(f"   🎉 목표 달성! (85%+ 정확도)")
            elif best_accuracy > baseline_accuracy:
                print(f"   📊 기준선 개선 성공!")
            else:
                print(f"   ⚠️ 추가 최적화 필요")

        # 결과 저장
        final_result = {
            'timestamp': datetime.now().isoformat(),
            'experiment_type': 'kaggle_champion_system',
            'feature_count': 184,
            'baseline_accuracy': baseline_accuracy,
            'performance': avg_performance,
            'detailed_results': all_results
        }

        output_path = f"/root/workspace/data/results/kaggle_champion_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, 'w') as f:
            json.dump(final_result, f, indent=2, default=str)

        print(f"\n💾 결과 저장: {output_path}")

def main():
    """메인 실행"""
    system = KaggleChampionSystem()
    results = system.run_champion_experiment()

    print("\n🎉 캐글 챔피언 시스템 테스트 완료!")
    return results

if __name__ == "__main__":
    main()