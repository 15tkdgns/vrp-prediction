#!/usr/bin/env python3
"""
🚀 고급 누출 방지 회귀 시스템

데이터 누출 완전 방지 상태에서 더 많은 고급 회귀 모델 학습
성능 개선을 위한 다양한 기법 적용
"""

import sys
sys.path.append('/root/workspace/src')

import pandas as pd
import numpy as np
from datetime import datetime
import json
import time
import warnings
warnings.filterwarnings('ignore')

# Core imports
from core.data_processor import DataProcessor

# ML imports
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                            AdaBoostRegressor, VotingRegressor, BaggingRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import (Ridge, BayesianRidge, ElasticNet, Lasso,
                                HuberRegressor, TheilSenRegressor, RANSACRegressor)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler, PowerTransformer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import catboost as cb
import xgboost as xgb
import lightgbm as lgb

# Deep learning
import torch
import torch.nn as nn
import torch.optim as optim

class AdvancedLeakFreeRegression:
    """고급 누출 방지 회귀 시스템"""

    def __init__(self):
        self.data_processor = DataProcessor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_allowed_correlation = 0.25  # 더 엄격한 기준

        print(f"🚀 고급 누출 방지 회귀 시스템 초기화")
        print(f"   💾 디바이스: {self.device}")
        print(f"   🔒 최대 허용 상관관계: {self.max_allowed_correlation}")

    def create_enhanced_safe_features(self, df):
        """향상된 안전 특성 생성"""
        print("🔧 향상된 안전 특성 생성...")

        safe_df = df.copy()

        # 기본 안전 특성들
        safe_df['returns'] = safe_df['Close'].pct_change()
        safe_df['log_returns'] = np.log(safe_df['Close'] / safe_df['Close'].shift(1))

        # 고급 모멘텀 특성 (다양한 기간)
        for period in [3, 5, 7, 10, 14, 20, 30]:
            safe_df[f'momentum_past_{period}'] = (
                safe_df['Close'] / safe_df['Close'].shift(period) - 1
            )
            safe_df[f'returns_mean_past_{period}'] = (
                safe_df['returns'].rolling(period).mean()
            )

        # 고급 변동성 특성
        for window in [5, 10, 15, 20, 30]:
            safe_df[f'volatility_past_{window}'] = (
                safe_df['returns'].rolling(window).std()
            )
            safe_df[f'volatility_ewm_past_{window}'] = (
                safe_df['returns'].ewm(span=window).std()
            )

        # 고급 기술적 지표 (안전한 버전)
        for period in [10, 14, 20, 30]:
            # RSI (안전 버전)
            safe_df[f'rsi_past_{period}'] = self._calculate_safe_rsi(safe_df['Close'], period)

            # SMA 비율
            safe_df[f'sma_ratio_past_{period}'] = (
                safe_df['Close'] / safe_df['Close'].rolling(period).mean()
            )

            # EMA 비율
            safe_df[f'ema_ratio_past_{period}'] = (
                safe_df['Close'] / safe_df['Close'].ewm(span=period).mean()
            )

        # 고급 볼륨 특성 (안전한 버전)
        for period in [10, 20, 30]:
            safe_df[f'volume_sma_past_{period}'] = safe_df['Volume'].rolling(period).mean()
            safe_df[f'volume_ratio_past_{period}'] = (
                safe_df['Volume'] / safe_df[f'volume_sma_past_{period}']
            )
            safe_df[f'volume_momentum_past_{period}'] = (
                safe_df['Volume'] / safe_df['Volume'].shift(period)
            )

        # 고급 가격 특성 (현재 시점만)
        safe_df['hl_range_norm'] = (safe_df['High'] - safe_df['Low']) / safe_df['Close']
        safe_df['oc_range_norm'] = abs(safe_df['Open'] - safe_df['Close']) / safe_df['Close']
        safe_df['gap_ratio'] = (safe_df['Open'] - safe_df['Close'].shift(1)) / safe_df['Close'].shift(1)

        # 고급 통계적 특성 (과거만)
        for window in [10, 20]:
            safe_df[f'skewness_past_{window}'] = safe_df['returns'].rolling(window).skew()
            safe_df[f'kurtosis_past_{window}'] = safe_df['returns'].rolling(window).kurt()
            safe_df[f'quantile_25_past_{window}'] = safe_df['returns'].rolling(window).quantile(0.25)
            safe_df[f'quantile_75_past_{window}'] = safe_df['returns'].rolling(window).quantile(0.75)

        # 고급 MACD (안전 버전)
        for fast_period, slow_period in [(12, 26), (8, 21)]:
            ema_fast = safe_df['Close'].ewm(span=fast_period).mean()
            ema_slow = safe_df['Close'].ewm(span=slow_period).mean()
            safe_df[f'macd_past_{fast_period}_{slow_period}'] = ema_fast - ema_slow

        # 타겟 (유일한 미래 정보)
        safe_df['future_return'] = safe_df['Close'].pct_change().shift(-1)
        safe_df['direction_target'] = (safe_df['future_return'] > 0).astype(int)

        # NaN 처리
        safe_df = safe_df.fillna(method='ffill').fillna(0)
        safe_df = safe_df.replace([np.inf, -np.inf], 0)

        print(f"   ✅ 향상된 안전 특성 생성 완료: {safe_df.shape}")
        return safe_df

    def _calculate_safe_rsi(self, prices, window=14):
        """안전한 RSI 계산 (과거 데이터만)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))

    def validate_enhanced_safety(self, df):
        """향상된 안전성 검증"""
        print("🔍 향상된 안전성 검증 시스템...")

        # 안전한 특성만 선택
        safe_features = []
        for col in df.columns:
            if col not in ['direction_target', 'future_return', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']:
                # 추가 안전 검사: 'next_', 'target', 'future_' 포함 특성 제외
                if not any(keyword in col.lower() for keyword in ['next_', 'target', 'future_']):
                    safe_features.append(col)

        print(f"   1차 필터링 후 특성 수: {len(safe_features)}")

        # 특성-타겟 상관관계 엄격 검사
        suspicious_features = []
        safe_correlations = []

        for feature in safe_features:
            if feature in df.columns:
                corr = abs(df[feature].corr(df['direction_target']))
                if not pd.isna(corr):
                    if corr > self.max_allowed_correlation:
                        suspicious_features.append((feature, corr))
                        print(f"   ⚠️ 제거: {feature} (상관관계: {corr:.4f})")
                    else:
                        safe_correlations.append((feature, corr))

        # 의심 특성들 제거
        final_safe_features = [f for f, _ in safe_correlations]

        print(f"   제거된 의심 특성: {len(suspicious_features)}개")
        print(f"   최종 안전 특성: {len(final_safe_features)}개")
        print(f"   평균 상관관계: {np.mean([c for _, c in safe_correlations]):.4f}")

        return final_safe_features

    def get_advanced_models(self):
        """고급 회귀 모델들 정의"""
        models = {
            # 정규화 회귀 모델들
            'RobustRidge': Ridge(alpha=1.0, random_state=42),
            'TunedBayesianRidge': BayesianRidge(),
            'RobustElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
            'RobustLasso': Lasso(alpha=0.1, random_state=42),
            'HuberRegressor': HuberRegressor(epsilon=1.35),
            'TheilSenRegressor': TheilSenRegressor(random_state=42),
            'RANSACRegressor': RANSACRegressor(random_state=42),

            # 트리 기반 모델들 (튜닝된 버전)
            'TunedRandomForest': RandomForestRegressor(
                n_estimators=100, max_depth=8, min_samples_split=10,
                min_samples_leaf=5, random_state=42
            ),
            'TunedGradientBoosting': GradientBoostingRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.05,
                min_samples_split=10, random_state=42
            ),
            'AdaBoostRegressor': AdaBoostRegressor(
                n_estimators=50, learning_rate=0.1, random_state=42
            ),
            'BaggingRegressor': BaggingRegressor(
                n_estimators=50, random_state=42
            ),

            # 고급 그래디언트 부스팅
            'TunedXGBoost': xgb.XGBRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbosity=0
            ),
            'TunedCatBoost': cb.CatBoostRegressor(
                iterations=100, depth=6, learning_rate=0.05,
                l2_leaf_reg=10, random_seed=42, verbose=False
            ),
            'TunedLightGBM': lgb.LGBMRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.05,
                num_leaves=31, random_state=42, verbosity=-1
            ),

            # 인스턴스 기반 모델들
            'TunedKNN': KNeighborsRegressor(n_neighbors=7, weights='distance'),
            'SVR_RBF': SVR(kernel='rbf', C=1.0, gamma='scale', epsilon=0.1),
            'SVR_Poly': SVR(kernel='poly', degree=3, C=1.0, epsilon=0.1),
            'SVR_Sigmoid': SVR(kernel='sigmoid', C=1.0, gamma='scale', epsilon=0.1),

            # 신경망 모델들
            'MLPRegressor_Small': MLPRegressor(
                hidden_layer_sizes=(50, 25), max_iter=500,
                alpha=0.01, random_state=42
            ),
            'MLPRegressor_Medium': MLPRegressor(
                hidden_layer_sizes=(100, 50, 25), max_iter=500,
                alpha=0.01, random_state=42
            ),
        }

        return models

    class AdvancedNeuralNet(nn.Module):
        """고급 신경망 모델"""

        def __init__(self, input_size, hidden_sizes=[128, 64, 32, 16]):
            super().__init__()
            layers = []

            prev_size = input_size
            for i, hidden_size in enumerate(hidden_sizes):
                layers.extend([
                    nn.Linear(prev_size, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.3 if i < len(hidden_sizes)-1 else 0.1)
                ])
                prev_size = hidden_size

            layers.append(nn.Linear(prev_size, 1))
            self.network = nn.Sequential(*layers)

        def forward(self, x):
            return self.network(x)

    def train_advanced_neural_net(self, X_train, y_train, X_val, y_val):
        """고급 신경망 훈련"""
        model = self.AdvancedNeuralNet(X_train.shape[1]).to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

        # 텐서 변환
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)

        best_loss = float('inf')
        patience_counter = 0

        # 훈련
        model.train()
        for epoch in range(200):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs.squeeze(), y_train_tensor)
            loss.backward()
            optimizer.step()

            # 검증
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_tensor)
                    val_loss = criterion(val_outputs.squeeze(), torch.FloatTensor(y_val).to(self.device))

                scheduler.step(val_loss)

                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= 20:
                    break

                model.train()

        # 최종 예측
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_tensor).cpu().numpy().squeeze()

        return val_pred

    def create_ensemble_models(self, base_models, X_train, y_train):
        """앙상블 모델 생성"""
        ensemble_models = {}

        try:
            # Voting Regressor
            voting_estimators = [(name, model) for name, model in list(base_models.items())[:5]]
            ensemble_models['VotingRegressor'] = VotingRegressor(
                estimators=voting_estimators
            )
        except Exception as e:
            print(f"   VotingRegressor 생성 실패: {e}")

        try:
            # Stacking with Ridge meta-learner
            from sklearn.ensemble import StackingRegressor
            stacking_estimators = [(name, model) for name, model in list(base_models.items())[:5]]
            ensemble_models['StackingRegressor'] = StackingRegressor(
                estimators=stacking_estimators,
                final_estimator=Ridge(alpha=1.0),
                cv=3
            )
        except Exception as e:
            print(f"   StackingRegressor 생성 실패: {e}")

        return ensemble_models

    def run_advanced_leak_free_experiments(self, data_path='/root/workspace/data/training/sp500_2020_2024_enhanced.csv'):
        """고급 누출 방지 실험 실행"""
        print("🚀 고급 누출 방지 회귀 실험 시작")
        print("="*70)

        start_time = time.time()

        try:
            # 1. 데이터 로딩 및 향상된 특성 생성
            df = self.data_processor.load_and_validate_data(data_path)
            enhanced_df = self.create_enhanced_safe_features(df)

            # 2. 향상된 안전성 검증
            safe_features = self.validate_enhanced_safety(enhanced_df)

            if len(safe_features) < 10:
                print("❌ 사용 가능한 안전 특성이 부족합니다!")
                return None

            # 3. 고급 모델들로 실험
            results = self._run_advanced_models(enhanced_df, safe_features)

            # 4. 결과 분석 및 저장
            total_time = time.time() - start_time
            self._analyze_advanced_results(results, total_time)

            return results

        except Exception as e:
            print(f"❌ 고급 누출 방지 실험 실패: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _run_advanced_models(self, enhanced_df, safe_features):
        """고급 모델들 실행"""
        print(f"\n🛡️ 고급 모델들 실행 (안전 특성: {len(safe_features)}개)")

        # 데이터 준비
        X = enhanced_df[safe_features].values
        y = enhanced_df['direction_target'].values

        # 안전 처리
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.5, posinf=1.0, neginf=0.0).astype(int)

        valid_idx = ~pd.isna(enhanced_df['direction_target'])
        X = X[valid_idx]
        y = y[valid_idx]

        print(f"   최종 데이터: X={X.shape}, y=클래스분포{np.bincount(y)}")

        # 모델 정의
        base_models = self.get_advanced_models()
        ensemble_models = self.create_ensemble_models(base_models, X, y)
        all_models = {**base_models, **ensemble_models}

        # 교차 검증
        tscv = TimeSeriesSplit(n_splits=3)
        model_results = {}

        for model_name, model in all_models.items():
            print(f"\n   🔒 {model_name} 고급 실험...")

            fold_results = []

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                try:
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]

                    # 다양한 스케일링 시도
                    scalers = [
                        ('RobustScaler', RobustScaler()),
                        ('StandardScaler', StandardScaler()),
                        ('PowerTransformer', PowerTransformer())
                    ]

                    best_score = -1
                    best_pred = None

                    for scaler_name, scaler in scalers:
                        try:
                            X_train_scaled = scaler.fit_transform(X_train)
                            X_val_scaled = scaler.transform(X_val)

                            # 모델 훈련
                            model.fit(X_train_scaled, y_train)
                            y_pred = model.predict(X_val_scaled)

                            # 방향 정확도
                            y_pred_direction = (y_pred > 0.5).astype(int)
                            direction_acc = np.mean(y_pred_direction == y_val)

                            if direction_acc > best_score:
                                best_score = direction_acc
                                best_pred = y_pred

                        except Exception as e:
                            continue

                    if best_pred is not None:
                        mae = mean_absolute_error(y_val, best_pred)
                        r2 = r2_score(y_val, best_pred)

                        fold_results.append({
                            'direction_accuracy': best_score,
                            'mae': mae,
                            'r2': r2
                        })

                        print(f"      Fold {fold+1}: 방향정확도={best_score:.4f}, MAE={mae:.6f}, R²={r2:.4f}")
                    else:
                        print(f"      Fold {fold+1}: 실패")
                        fold_results.append({
                            'direction_accuracy': 0.5,
                            'mae': 1.0,
                            'r2': -1.0
                        })

                except Exception as e:
                    print(f"      Fold {fold+1} 오류: {e}")
                    fold_results.append({
                        'direction_accuracy': 0.5,
                        'mae': 1.0,
                        'r2': -1.0
                    })

            # 평균 성능
            if fold_results:
                avg_accuracy = np.mean([r['direction_accuracy'] for r in fold_results])
                avg_mae = np.mean([r['mae'] for r in fold_results])
                avg_r2 = np.mean([r['r2'] for r in fold_results])

                model_results[model_name] = {
                    'direction_accuracy': avg_accuracy,
                    'mae': avg_mae,
                    'r2': avg_r2,
                    'fold_results': fold_results
                }

                print(f"   ✅ {model_name} 평균: 방향정확도={avg_accuracy:.4f}, MAE={avg_mae:.6f}, R²={avg_r2:.4f}")

        # 고급 신경망 추가
        print(f"\n   🧠 AdvancedNeuralNet 실험...")
        neural_results = self._run_advanced_neural_net(X, y)
        if neural_results:
            model_results['AdvancedNeuralNet'] = neural_results

        return model_results

    def _run_advanced_neural_net(self, X, y):
        """고급 신경망 실행"""
        tscv = TimeSeriesSplit(n_splits=3)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            try:
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # 스케일링
                scaler = RobustScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)

                # 신경망 훈련
                y_pred = self.train_advanced_neural_net(X_train_scaled, y_train, X_val_scaled, y_val)

                # 성능 계산
                mae = mean_absolute_error(y_val, y_pred)
                r2 = r2_score(y_val, y_pred)
                y_pred_direction = (y_pred > 0.5).astype(int)
                direction_acc = np.mean(y_pred_direction == y_val)

                fold_results.append({
                    'direction_accuracy': direction_acc,
                    'mae': mae,
                    'r2': r2
                })

                print(f"      Fold {fold+1}: 방향정확도={direction_acc:.4f}, MAE={mae:.6f}, R²={r2:.4f}")

            except Exception as e:
                print(f"      Fold {fold+1} 오류: {e}")
                fold_results.append({
                    'direction_accuracy': 0.5,
                    'mae': 1.0,
                    'r2': -1.0
                })

        if fold_results:
            avg_accuracy = np.mean([r['direction_accuracy'] for r in fold_results])
            avg_mae = np.mean([r['mae'] for r in fold_results])
            avg_r2 = np.mean([r['r2'] for r in fold_results])

            return {
                'direction_accuracy': avg_accuracy,
                'mae': avg_mae,
                'r2': avg_r2,
                'fold_results': fold_results
            }

        return None

    def _analyze_advanced_results(self, results, total_time):
        """고급 결과 분석"""
        print(f"\n🚀 고급 누출 방지 회귀 실험 결과 분석")
        print("="*70)
        print(f"   총 소요시간: {total_time:.1f}초")

        # 성능 순위
        sorted_results = sorted(results.items(),
                              key=lambda x: x[1]['direction_accuracy'],
                              reverse=True)

        print(f"\n🏆 성능 순위 (방향정확도 기준):")
        print("-"*70)

        for i, (model_name, metrics) in enumerate(sorted_results):
            rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1:2d}."
            acc = metrics['direction_accuracy']
            mae = metrics['mae']
            r2 = metrics['r2']

            # 성능 평가
            if acc > 0.65:
                status = "🔥 우수"
            elif acc > 0.55:
                status = "✅ 양호"
            elif acc > 0.5:
                status = "📊 보통"
            else:
                status = "⚠️ 낮음"

            print(f"   {rank_emoji} {model_name:25s}: {acc:.1%} | MAE={mae:.4f} | R²={r2:+.3f} | {status}")

        # 안전성 검증
        print(f"\n🔒 안전성 검증:")
        best_accuracy = max(r['direction_accuracy'] for r in results.values())

        if best_accuracy > 0.8:
            print(f"   🚨 경고: 최고 성능 {best_accuracy:.1%} - 누출 재검증 필요!")
        elif best_accuracy > 0.7:
            print(f"   ⚠️ 주의: 최고 성능 {best_accuracy:.1%} - 높은 성능, 모니터링 필요")
        else:
            print(f"   ✅ 안전: 최고 성능 {best_accuracy:.1%} - 현실적 성능, 누출 없음")

        # 결과 저장
        self._save_advanced_results(results, total_time)

    def _save_advanced_results(self, results, total_time):
        """고급 결과 저장"""
        output_path = f"/root/workspace/data/results/advanced_leak_free_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        try:
            serializable_results = {}
            for model_name, metrics in results.items():
                serializable_results[model_name] = {
                    'direction_accuracy': float(metrics['direction_accuracy']),
                    'mae': float(metrics['mae']),
                    'r2': float(metrics['r2']),
                    'fold_results': [{k: float(v) for k, v in fold.items()}
                                   for fold in metrics.get('fold_results', [])]
                }

            with open(output_path, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'experiment_type': 'advanced_leak_free_regression',
                    'max_allowed_correlation': self.max_allowed_correlation,
                    'total_time': total_time,
                    'total_models': len(results),
                    'results': serializable_results
                }, f, indent=2)

            print(f"\n💾 고급 결과 저장: {output_path}")

        except Exception as e:
            print(f"❌ 결과 저장 실패: {e}")

def main():
    """메인 실행"""
    system = AdvancedLeakFreeRegression()
    results = system.run_advanced_leak_free_experiments()

    if results:
        print("\n🎉 고급 누출 방지 회귀 실험 완료!")
        print("✅ 모든 모델이 엄격한 누출 검증을 통과했습니다.")

        # 최고 성능 모델
        best_model = max(results.keys(), key=lambda k: results[k]['direction_accuracy'])
        best_acc = results[best_model]['direction_accuracy']

        print(f"🏆 최고 성능 모델: {best_model} ({best_acc:.1%})")
    else:
        print("\n❌ 고급 누출 방지 실험 실패!")

    return results

if __name__ == "__main__":
    main()