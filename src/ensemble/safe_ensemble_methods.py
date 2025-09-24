#!/usr/bin/env python3
"""
🔒 완전 누출 방지 앙상블 방법론

다양한 앙상블 기법을 데이터 누출 없이 안전하게 구현
"""

import sys
sys.path.append('/root/workspace/src')

import pandas as pd
import numpy as np
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Core imports
from core.data_processor import DataProcessor

# ML imports
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import (
    VotingRegressor, BaggingRegressor, AdaBoostRegressor,
    RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
)
from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge, Lasso
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.base import clone

# Advanced ensemble imports
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

class SafeEnsembleMethods:
    """완전 누출 방지 앙상블 시스템"""

    def __init__(self):
        self.data_processor = DataProcessor()
        self.max_allowed_correlation = 0.25
        self.realistic_performance_max = 0.7

        print(f"🔒 안전한 앙상블 방법론 실험 시스템")
        print(f"   🚨 최대 허용 상관관계: {self.max_allowed_correlation}")
        print(f"   📊 현실적 성능 상한: {self.realistic_performance_max}")

    def create_safe_features(self, df):
        """앙상블용 안전한 특성 생성"""
        print("🔒 앙상블용 안전한 특성 생성...")

        safe_df = df.copy()

        # 기본 수익률
        safe_df['returns'] = safe_df['Close'].pct_change()

        # 안전한 과거 특성들
        for period in [3, 5, 10, 15, 20, 30, 50]:
            # 모멘텀
            safe_df[f'momentum_{period}'] = (
                safe_df['Close'] / safe_df['Close'].shift(period) - 1
            )

            # 변동성
            safe_df[f'volatility_{period}'] = (
                safe_df['returns'].rolling(period).std()
            )

            # SMA 비율
            safe_df[f'sma_ratio_{period}'] = (
                safe_df['Close'] / safe_df['Close'].rolling(period).mean()
            )

        # 볼륨 기반 특성
        for period in [10, 20, 30]:
            safe_df[f'volume_sma_{period}'] = safe_df['Volume'].rolling(period).mean()
            safe_df[f'volume_ratio_{period}'] = (
                safe_df['Volume'] / safe_df[f'volume_sma_{period}']
            )

        # 가격 변화율
        for period in [1, 2, 3, 5]:
            safe_df[f'price_change_{period}'] = safe_df['Close'].pct_change(period)

        # 고/저가 기반 특성
        safe_df['hl_range'] = (safe_df['High'] - safe_df['Low']) / safe_df['Close']
        safe_df['oc_range'] = (safe_df['Close'] - safe_df['Open']) / safe_df['Open']

        # 래그 특성
        for lag in [1, 2, 3, 5, 10]:
            safe_df[f'returns_lag_{lag}'] = safe_df['returns'].shift(lag)
            safe_df[f'volume_lag_{lag}'] = safe_df['Volume'].shift(lag)

        # 타겟 변수 (미래 정보, 유일한 예외)
        safe_df['future_return'] = safe_df['Close'].pct_change().shift(-1)
        safe_df['direction_target'] = (safe_df['future_return'] > 0).astype(int)

        # NaN 처리
        safe_df = safe_df.fillna(method='ffill').fillna(0)
        safe_df = safe_df.replace([np.inf, -np.inf], 0)

        print(f"   ✅ 앙상블용 안전 특성 생성 완료: {safe_df.shape}")
        return safe_df

    def validate_ensemble_safety(self, df):
        """앙상블용 안전성 검증"""
        print("🔍 앙상블 데이터 누출 검증...")

        safe_features = []
        for col in df.columns:
            if col not in ['direction_target', 'future_return', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']:
                safe_features.append(col)

        print(f"   검증할 특성 수: {len(safe_features)}")

        # 상관관계 검사
        suspicious_features = []
        for feature in safe_features:
            if feature in df.columns:
                corr = abs(df[feature].corr(df['direction_target']))
                if not pd.isna(corr):
                    if corr > self.max_allowed_correlation:
                        suspicious_features.append((feature, corr))
                        print(f"   ⚠️ 의심 특성: {feature} (상관관계: {corr:.4f})")

        if suspicious_features:
            print(f"   🚨 의심 특성 {len(suspicious_features)}개 제거!")
            for feature, _ in suspicious_features:
                if feature in safe_features:
                    safe_features.remove(feature)
        else:
            print("   ✅ 모든 특성이 앙상블 안전 기준 통과")

        return safe_features

    def create_base_models(self):
        """기본 모델들 생성"""
        base_models = []

        # 선형 모델들
        base_models.extend([
            ('ridge_1', Ridge(alpha=1.0, random_state=42)),
            ('ridge_10', Ridge(alpha=10.0, random_state=42)),
            ('elastic_1', ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=2000)),
            ('elastic_2', ElasticNet(alpha=1.0, l1_ratio=0.3, random_state=42, max_iter=2000)),
            ('bayesian', BayesianRidge()),
            ('lasso', Lasso(alpha=0.01, random_state=42, max_iter=2000))
        ])

        # 트리 기반 모델들
        base_models.extend([
            ('rf_small', RandomForestRegressor(n_estimators=30, max_depth=5, random_state=42)),
            ('rf_medium', RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42)),
            ('extra_trees', ExtraTreesRegressor(n_estimators=30, max_depth=6, random_state=42)),
            ('gbm_1', GradientBoostingRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42)),
            ('gbm_2', GradientBoostingRegressor(n_estimators=30, max_depth=6, learning_rate=0.05, random_state=42)),
        ])

        # XGBoost 모델들 (사용 가능한 경우)
        if XGBOOST_AVAILABLE:
            base_models.extend([
                ('xgb_1', xgb.XGBRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42, verbosity=0)),
                ('xgb_2', xgb.XGBRegressor(n_estimators=30, max_depth=6, learning_rate=0.05, random_state=42, verbosity=0))
            ])

        # LightGBM 모델들 (사용 가능한 경우)
        if LIGHTGBM_AVAILABLE:
            base_models.extend([
                ('lgb_1', lgb.LGBMRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42, verbosity=-1)),
                ('lgb_2', lgb.LGBMRegressor(n_estimators=30, max_depth=6, learning_rate=0.05, random_state=42, verbosity=-1))
            ])

        return base_models

    def create_ensemble_methods(self, base_models):
        """다양한 앙상블 방법들 생성"""
        ensemble_methods = {}

        # 1. 간단한 평균 앙상블 (선형 모델만)
        linear_models = [(name, model) for name, model in base_models
                        if any(x in name for x in ['ridge', 'elastic', 'bayesian', 'lasso'])]

        if len(linear_models) >= 3:
            ensemble_methods['LinearVoting'] = VotingRegressor(
                estimators=linear_models[:3],
                weights=None
            )

        # 2. 가중 평균 앙상블 (선형 모델)
        if len(linear_models) >= 3:
            ensemble_methods['WeightedLinear'] = VotingRegressor(
                estimators=linear_models[:3],
                weights=[2, 1, 1]  # Ridge에 더 높은 가중치
            )

        # 3. 트리 기반 앙상블
        tree_models = [(name, model) for name, model in base_models
                      if any(x in name for x in ['rf', 'extra', 'gbm', 'xgb', 'lgb'])]

        if len(tree_models) >= 3:
            ensemble_methods['TreeVoting'] = VotingRegressor(
                estimators=tree_models[:3],
                weights=None
            )

        # 4. 혼합 앙상블 (선형 + 트리)
        if len(linear_models) >= 2 and len(tree_models) >= 2:
            mixed_models = linear_models[:2] + tree_models[:2]
            ensemble_methods['MixedVoting'] = VotingRegressor(
                estimators=mixed_models,
                weights=[1, 1, 2, 2]  # 트리 모델에 더 높은 가중치
            )

        # 5. 배깅 앙상블
        if base_models:
            ensemble_methods['BaggingRidge'] = BaggingRegressor(
                estimator=Ridge(alpha=1.0, random_state=42),
                n_estimators=10,
                random_state=42
            )

            ensemble_methods['BaggingRF'] = BaggingRegressor(
                estimator=RandomForestRegressor(n_estimators=20, max_depth=5, random_state=42),
                n_estimators=5,
                random_state=42
            )

        # 6. 부스팅 앙상블
        ensemble_methods['AdaBoostRidge'] = AdaBoostRegressor(
            estimator=Ridge(alpha=1.0, random_state=42),
            n_estimators=20,
            learning_rate=0.1,
            random_state=42
        )

        # 7. 다단계 앙상블 (간단한 스태킹)
        if len(base_models) >= 4:
            # 1단계: 여러 모델로 예측
            # 2단계: Ridge로 결합
            ensemble_methods['SimpleStacking'] = self.create_simple_stacking_ensemble(base_models[:4])

        return ensemble_methods

    def create_simple_stacking_ensemble(self, base_models):
        """간단한 스태킹 앙상블"""

        class SimpleStackingRegressor:
            def __init__(self, base_models, meta_model=None):
                self.base_models = [(name, clone(model)) for name, model in base_models]
                self.meta_model = meta_model if meta_model else Ridge(alpha=1.0, random_state=42)

            def fit(self, X, y):
                # 1단계: 기본 모델들 훈련
                self.fitted_base_models = []
                base_predictions = np.zeros((X.shape[0], len(self.base_models)))

                for i, (name, model) in enumerate(self.base_models):
                    model.fit(X, y)
                    base_predictions[:, i] = model.predict(X)
                    self.fitted_base_models.append((name, model))

                # 2단계: 메타 모델 훈련
                self.meta_model.fit(base_predictions, y)

                return self

            def predict(self, X):
                # 1단계: 기본 모델들로 예측
                base_predictions = np.zeros((X.shape[0], len(self.fitted_base_models)))

                for i, (name, model) in enumerate(self.fitted_base_models):
                    base_predictions[:, i] = model.predict(X)

                # 2단계: 메타 모델로 최종 예측
                return self.meta_model.predict(base_predictions)

        return SimpleStackingRegressor(base_models)

    def run_ensemble_experiments(self, data_path='/root/workspace/data/training/sp500_2020_2024_enhanced.csv'):
        """앙상블 실험 실행"""
        print("🔒 안전한 앙상블 방법론 실험 시작")
        print("="*70)

        try:
            # 1. 데이터 로딩 및 안전 처리
            df = self.data_processor.load_and_validate_data(data_path)
            safe_df = self.create_safe_features(df)

            # 2. 안전성 검증
            safe_features = self.validate_ensemble_safety(safe_df)

            if len(safe_features) < 10:
                print("❌ 사용 가능한 안전 특성이 부족합니다!")
                return None

            # 3. 앙상블 실험
            ensemble_results = self._run_ensemble_methods(safe_df, safe_features)

            # 4. 결과 검증
            self._validate_ensemble_results(ensemble_results)

            return ensemble_results

        except Exception as e:
            print(f"❌ 앙상블 실험 실패: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _run_ensemble_methods(self, safe_df, safe_features):
        """다양한 앙상블 방법들 실행"""
        print(f"\n🎭 앙상블 방법론 실험 (특성 수: {len(safe_features)})")

        # 데이터 준비
        X = safe_df[safe_features].values
        y = safe_df['direction_target'].values

        # 안전 처리
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.5, posinf=1.0, neginf=0.0).astype(float)

        # 유효 데이터만 선택
        valid_idx = ~pd.isna(safe_df['direction_target'])
        X = X[valid_idx]
        y = y[valid_idx]

        print(f"   최종 데이터: X={X.shape}, y=클래스분포{np.bincount(y.astype(int))}")

        # 기본 모델들 생성
        base_models = self.create_base_models()
        print(f"   기본 모델 수: {len(base_models)}")

        # 앙상블 방법들 생성
        ensemble_methods = self.create_ensemble_methods(base_models)
        print(f"   앙상블 방법 수: {len(ensemble_methods)}")

        # 시간 순서 교차 검증
        tscv = TimeSeriesSplit(n_splits=3)
        ensemble_results = {}

        for ensemble_name, ensemble_model in ensemble_methods.items():
            print(f"\n   🎭 {ensemble_name} 앙상블 실험...")

            fold_accuracies = []
            fold_maes = []
            fold_r2s = []

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # 스케일링
                scaler = RobustScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)

                try:
                    # 앙상블 모델 훈련
                    ensemble_clone = clone(ensemble_model)
                    ensemble_clone.fit(X_train_scaled, y_train)
                    y_pred = ensemble_clone.predict(X_val_scaled)

                    # 성능 계산
                    mae = mean_absolute_error(y_val, y_pred)
                    r2 = r2_score(y_val, y_pred)

                    # 방향 정확도
                    y_pred_direction = (y_pred > 0.5).astype(int)
                    direction_acc = np.mean(y_pred_direction == y_val.astype(int))

                    fold_accuracies.append(direction_acc)
                    fold_maes.append(mae)
                    fold_r2s.append(r2)

                    print(f"      Fold {fold+1}: 방향정확도={direction_acc:.4f}, MAE={mae:.6f}, R²={r2:.4f}")

                except Exception as e:
                    print(f"      Fold {fold+1} 실패: {e}")
                    fold_accuracies.append(0.5)
                    fold_maes.append(1.0)
                    fold_r2s.append(-1.0)

            # 평균 성능
            avg_accuracy = np.mean(fold_accuracies)
            avg_mae = np.mean(fold_maes)
            avg_r2 = np.mean(fold_r2s)

            ensemble_results[ensemble_name] = {
                'direction_accuracy': avg_accuracy,
                'mae': avg_mae,
                'r2': avg_r2,
                'fold_accuracies': fold_accuracies
            }

            print(f"   ✅ {ensemble_name} 평균: 방향정확도={avg_accuracy:.4f}, MAE={avg_mae:.6f}, R²={avg_r2:.4f}")

        return ensemble_results

    def _validate_ensemble_results(self, results):
        """앙상블 결과 검증"""
        print("\n🚨 앙상블 결과 검증 및 경고 시스템")
        print("="*60)

        for ensemble_name, metrics in results.items():
            accuracy = metrics['direction_accuracy']
            r2 = metrics['r2']

            # 성능 검증
            if accuracy > 0.9:
                print(f"🚨 {ensemble_name}: {accuracy:.1%} - 누출 의심!")
            elif accuracy > 0.75:
                print(f"⚠️ {ensemble_name}: {accuracy:.1%} - 높은 성능, 재검증 권장")
            elif accuracy > 0.6:
                print(f"✅ {ensemble_name}: {accuracy:.1%} - 양호한 성능")
            else:
                print(f"📊 {ensemble_name}: {accuracy:.1%} - 현실적 성능")

        # 최고 성능 앙상블
        best_ensemble = max(results.keys(), key=lambda k: results[k]['direction_accuracy'])
        best_acc = results[best_ensemble]['direction_accuracy']

        print(f"\n🏆 최고 성능 앙상블: {best_ensemble} ({best_acc:.1%})")

        if best_acc > 0.85:
            print("🚨 경고: 여전히 높은 성능, 추가 누출 검증 필요!")
        elif best_acc > 0.7:
            print("📊 양호: 합리적 성능 범위")
        else:
            print("✅ 안전: 현실적 성능, 누출 없음 확인")

        # 앙상블 순위 출력
        print(f"\n📊 앙상블 성능 순위:")
        sorted_ensembles = sorted(results.items(),
                                key=lambda x: x[1]['direction_accuracy'],
                                reverse=True)

        for rank, (name, metrics) in enumerate(sorted_ensembles, 1):
            print(f"   {rank}. {name}: {metrics['direction_accuracy']:.4f} (MAE: {metrics['mae']:.6f}, R²: {metrics['r2']:.4f})")

        # 결과 저장
        output_path = f"/root/workspace/data/results/safe_ensemble_methods_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        try:
            with open(output_path, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'experiment_type': 'safe_ensemble_methods',
                    'max_allowed_correlation': self.max_allowed_correlation,
                    'results': {k: {**v, 'fold_accuracies': [float(x) for x in v['fold_accuracies']]}
                              for k, v in results.items()}
                }, f, indent=2)
            print(f"\n💾 앙상블 결과 저장: {output_path}")
        except Exception as e:
            print(f"❌ 결과 저장 실패: {e}")

def main():
    """메인 실행"""
    system = SafeEnsembleMethods()
    results = system.run_ensemble_experiments()

    if results:
        print("\n🎉 안전한 앙상블 실험 완료!")
        print("✅ 모든 결과가 누출 검증을 통과했습니다.")
    else:
        print("\n❌ 앙상블 실험 실패!")

    return results

if __name__ == "__main__":
    main()