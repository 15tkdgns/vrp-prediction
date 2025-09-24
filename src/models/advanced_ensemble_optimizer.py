#!/usr/bin/env python3
"""
🚀 고도화된 앙상블 모델 최적화 시스템

현재 89.5% 성능을 92-94%로 개선하기 위한 고급 앙상블 기법
- Multi-Level Stacking
- Dynamic Weighted Blending
- Uncertainty-Based Voting
- Time-Aware Ensemble
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class AdvancedEnsembleOptimizer:
    """고도화된 앙상블 최적화 시스템"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.base_models = {}
        self.meta_learners = {}
        self.ensemble_weights = {}
        self.performance_history = []

    def initialize_base_models(self):
        """다양한 기본 모델 초기화"""
        self.base_models = {
            'rf_conservative': RandomForestClassifier(
                n_estimators=200, max_depth=8, min_samples_split=20,
                random_state=self.random_state, n_jobs=-1
            ),
            'rf_aggressive': RandomForestClassifier(
                n_estimators=300, max_depth=15, min_samples_split=5,
                random_state=self.random_state, n_jobs=-1
            ),
            'gb_conservative': GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.05, max_depth=6,
                random_state=self.random_state
            ),
            'gb_aggressive': GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=8,
                random_state=self.random_state
            ),
            'xgb_conservative': xgb.XGBClassifier(
                n_estimators=150, learning_rate=0.05, max_depth=6,
                random_state=self.random_state, eval_metric='logloss'
            ),
            'xgb_aggressive': xgb.XGBClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=8,
                random_state=self.random_state, eval_metric='logloss'
            ),
            'mlp_small': MLPClassifier(
                hidden_layer_sizes=(50, 25), learning_rate_init=0.001,
                max_iter=500, random_state=self.random_state
            ),
            'mlp_large': MLPClassifier(
                hidden_layer_sizes=(100, 50, 25), learning_rate_init=0.001,
                max_iter=500, random_state=self.random_state
            )
        }

        # 메타 러너들
        self.meta_learners = {
            'logistic': LogisticRegression(random_state=self.random_state),
            'ridge': RidgeClassifier(random_state=self.random_state),
            'rf_meta': RandomForestClassifier(
                n_estimators=50, max_depth=3, random_state=self.random_state
            )
        }

    def train_base_models(self, X_train, y_train, cv_folds=5):
        """기본 모델들 훈련 및 교차검증"""
        print("🤖 기본 모델들 훈련 시작...")

        tscv = TimeSeriesSplit(n_splits=cv_folds)
        base_scores = {}

        for name, model in self.base_models.items():
            print(f"   📊 훈련 중: {name}")

            # 교차검증 점수
            cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='accuracy')
            base_scores[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores
            }

            # 전체 데이터로 훈련
            model.fit(X_train, y_train)

            print(f"      ✅ CV 점수: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        return base_scores

    def generate_meta_features(self, X, y, cv_folds=5):
        """메타 특성 생성 (스태킹용)"""
        print("🔧 메타 특성 생성 중...")

        tscv = TimeSeriesSplit(n_splits=cv_folds)
        meta_features = np.zeros((len(X), len(self.base_models)))

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            print(f"   📊 Fold {fold + 1}/{cv_folds}")

            X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
            y_fold_train = y.iloc[train_idx]

            for i, (name, model) in enumerate(self.base_models.items()):
                # 폴드별로 새로운 모델 인스턴스 생성
                fold_model = model.__class__(**model.get_params())
                fold_model.fit(X_fold_train, y_fold_train)

                # 검증 세트 예측
                pred_proba = fold_model.predict_proba(X_fold_val)[:, 1]
                meta_features[val_idx, i] = pred_proba

        meta_df = pd.DataFrame(
            meta_features,
            columns=[f'base_{name}' for name in self.base_models.keys()],
            index=X.index
        )

        print(f"   ✅ 메타 특성 생성 완료: {meta_df.shape}")
        return meta_df

    def train_stacking_ensemble(self, X_train, y_train):
        """스태킹 앙상블 훈련"""
        print("🏗️ 스태킹 앙상블 훈련...")

        # 메타 특성 생성
        meta_features = self.generate_meta_features(X_train, y_train)

        # 메타 러너들 훈련
        stacking_scores = {}
        for name, meta_model in self.meta_learners.items():
            meta_model.fit(meta_features, y_train)

            # 메타 모델 교차검증
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = cross_val_score(meta_model, meta_features, y_train, cv=tscv, scoring='accuracy')

            stacking_scores[name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }

            print(f"   📊 {name} 메타러너: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        return stacking_scores

    def dynamic_weighted_blending(self, X_train, y_train, X_test):
        """동적 가중 블렌딩"""
        print("⚖️ 동적 가중 블렌딩 수행...")

        # 각 모델의 최근 성능을 기반으로 가중치 계산
        tscv = TimeSeriesSplit(n_splits=5)
        recent_scores = {}

        for name, model in self.base_models.items():
            scores = []
            for train_idx, val_idx in tscv.split(X_train):
                X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                fold_model = model.__class__(**model.get_params())
                fold_model.fit(X_fold_train, y_fold_train)
                pred = fold_model.predict(X_fold_val)
                score = accuracy_score(y_fold_val, pred)
                scores.append(score)

            # 최근 폴드에 더 높은 가중치
            weighted_score = np.average(scores, weights=[0.1, 0.15, 0.2, 0.25, 0.3])
            recent_scores[name] = weighted_score

        # 성능 기반 가중치 계산 (소프트맥스)
        scores_array = np.array(list(recent_scores.values()))
        weights = np.exp(scores_array * 10) / np.sum(np.exp(scores_array * 10))

        self.ensemble_weights = dict(zip(recent_scores.keys(), weights))

        print("   📊 모델별 가중치:")
        for name, weight in self.ensemble_weights.items():
            print(f"      {name}: {weight:.4f}")

        return self.ensemble_weights

    def uncertainty_based_voting(self, X_test):
        """불확실성 기반 투표"""
        print("🗳️ 불확실성 기반 투표 수행...")

        predictions = {}
        uncertainties = {}

        for name, model in self.base_models.items():
            pred_proba = model.predict_proba(X_test)
            predictions[name] = pred_proba

            # 불확실성 계산 (엔트로피)
            uncertainty = -np.sum(pred_proba * np.log(pred_proba + 1e-8), axis=1)
            uncertainties[name] = uncertainty

        # 불확실성이 낮은 모델에 더 높은 가중치
        final_predictions = np.zeros((len(X_test), 2))

        for i in range(len(X_test)):
            weights = []
            for name in self.base_models.keys():
                # 불확실성이 낮을수록 높은 가중치
                weight = 1.0 / (1.0 + uncertainties[name][i])
                weights.append(weight)

            weights = np.array(weights)
            weights = weights / np.sum(weights)

            # 가중 평균
            for j, name in enumerate(self.base_models.keys()):
                final_predictions[i] += weights[j] * predictions[name][i]

        return final_predictions

    def evaluate_ensemble_methods(self, X_train, y_train, X_test, y_test):
        """다양한 앙상블 방법 평가"""
        print("📊 앙상블 방법들 평가 중...")

        results = {}

        # 1. 단순 투표 (Voting)
        voting_preds = np.zeros(len(X_test))
        for model in self.base_models.values():
            voting_preds += model.predict(X_test)
        voting_preds = (voting_preds > len(self.base_models) / 2).astype(int)
        results['voting'] = accuracy_score(y_test, voting_preds)

        # 2. 평균 확률 (Average Probability)
        avg_proba = np.zeros((len(X_test), 2))
        for model in self.base_models.values():
            avg_proba += model.predict_proba(X_test)
        avg_proba /= len(self.base_models)
        avg_preds = np.argmax(avg_proba, axis=1)
        results['average_probability'] = accuracy_score(y_test, avg_preds)

        # 3. 가중 블렌딩
        if self.ensemble_weights:
            weighted_proba = np.zeros((len(X_test), 2))
            for name, model in self.base_models.items():
                weight = self.ensemble_weights.get(name, 1/len(self.base_models))
                weighted_proba += weight * model.predict_proba(X_test)
            weighted_preds = np.argmax(weighted_proba, axis=1)
            results['weighted_blending'] = accuracy_score(y_test, weighted_preds)

        # 4. 스태킹 (최고 성능 메타러너 사용)
        meta_features_test = np.zeros((len(X_test), len(self.base_models)))
        for i, model in enumerate(self.base_models.values()):
            meta_features_test[:, i] = model.predict_proba(X_test)[:, 1]

        best_meta_name = max(
            self.meta_learners.keys(),
            key=lambda x: cross_val_score(
                self.meta_learners[x],
                self.generate_meta_features(X_train, y_train),
                y_train, cv=3
            ).mean()
        )

        meta_features_test_df = pd.DataFrame(meta_features_test)
        stacking_preds = self.meta_learners[best_meta_name].predict(meta_features_test_df)
        results['stacking'] = accuracy_score(y_test, stacking_preds)

        # 5. 불확실성 기반 투표
        uncertainty_proba = self.uncertainty_based_voting(X_test)
        uncertainty_preds = np.argmax(uncertainty_proba, axis=1)
        results['uncertainty_voting'] = accuracy_score(y_test, uncertainty_preds)

        return results

    def run_advanced_ensemble_experiment(self, data_path):
        """고급 앙상블 실험 전체 실행"""
        print("🚀 고급 앙상블 실험 시작")
        print("="*60)

        # 데이터 로드
        df = pd.read_csv(data_path)

        # 특성과 타겟 분리
        feature_cols = [col for col in df.columns if col not in ['Date', 'Returns']]
        X = df[feature_cols].dropna()

        # 방향 예측을 위한 타겟 생성
        y = (df['Returns'].shift(-1) > 0).astype(int)
        y = y.loc[X.index]  # X와 인덱스 맞춤

        # 최종 NaN 제거
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]

        print(f"📊 데이터 크기: {X.shape}")
        print(f"🎯 타겟 분포: {y.value_counts().to_dict()}")

        # 시계열 분할
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        print(f"📊 훈련 세트: {X_train.shape}")
        print(f"📊 테스트 세트: {X_test.shape}")

        # 특성 정규화
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )

        # 모델 초기화 및 훈련
        self.initialize_base_models()
        base_scores = self.train_base_models(X_train_scaled, y_train)

        # 스태킹 앙상블 훈련
        stacking_scores = self.train_stacking_ensemble(X_train_scaled, y_train)

        # 동적 가중 블렌딩
        self.dynamic_weighted_blending(X_train_scaled, y_train, X_test_scaled)

        # 앙상블 방법들 평가
        ensemble_results = self.evaluate_ensemble_methods(
            X_train_scaled, y_train, X_test_scaled, y_test
        )

        # 결과 정리
        print(f"\n📊 기본 모델 성능:")
        for name, scores in base_scores.items():
            print(f"   {name}: {scores['cv_mean']:.4f} ± {scores['cv_std']:.4f}")

        print(f"\n🏗️ 앙상블 방법 성능:")
        for method, accuracy in ensemble_results.items():
            print(f"   {method}: {accuracy:.4f}")

        # 최고 성능 방법 찾기
        best_method = max(ensemble_results.items(), key=lambda x: x[1])
        print(f"\n🏆 최고 성능: {best_method[0]} = {best_method[1]:.4f}")

        # 기준선 대비 개선량
        baseline_accuracy = 0.895  # 현재 검증된 최고 성과
        improvement = (best_method[1] - baseline_accuracy) * 100
        print(f"📈 기준선 대비 개선: {improvement:+.2f}%p")

        return {
            'base_scores': base_scores,
            'ensemble_results': ensemble_results,
            'best_method': best_method,
            'improvement': improvement,
            'test_accuracy': best_method[1]
        }

def main():
    """메인 실행 함수"""
    optimizer = AdvancedEnsembleOptimizer()

    # SPY 데이터로 실험
    data_path = "/root/workspace/data/training/sp500_leak_free_dataset.csv"

    results = optimizer.run_advanced_ensemble_experiment(data_path)

    # 결과 저장
    import json
    from datetime import datetime

    results_with_metadata = {
        'timestamp': datetime.now().isoformat(),
        'experiment_type': 'advanced_ensemble_optimization',
        'data_source': data_path,
        'baseline_accuracy': 0.895,
        'target_improvement': "92-94%",
        'results': results,
        'validation_status': 'pending_verification'
    }

    output_path = f"/root/workspace/data/results/advanced_ensemble_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, 'w') as f:
        json.dump(results_with_metadata, f, indent=2)

    print(f"\n💾 결과 저장: {output_path}")

    return results

if __name__ == "__main__":
    main()