#!/usr/bin/env python3
"""
하이퍼파라미터 최적화 및 최종 모델 검증
Lasso α=0.001 모델의 성능을 더욱 향상시키기 위한 정밀 튜닝
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, ElasticNet, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
import os
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# 이전 모듈에서 함수 가져오기
import sys
sys.path.append('/root/workspace/src/features')
from improved_volatility_model import load_enhanced_spy_data, create_comprehensive_features, create_future_volatility_targets, create_interaction_features

def optimize_lasso_hyperparameters(X, y):
    """Lasso 모델 하이퍼파라미터 최적화"""
    print("🔍 Lasso 하이퍼파라미터 최적화 중...")

    # 완전한 데이터만 사용
    combined_data = pd.concat([X, y], axis=1).dropna()
    X_clean = combined_data[X.columns]
    y_clean = combined_data[y.name]

    # 하이퍼파라미터 그리드
    alpha_values = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
    max_iter_values = [1000, 2000, 3000, 5000]

    param_grid = {
        'alpha': alpha_values,
        'max_iter': max_iter_values
    }

    tscv = TimeSeriesSplit(n_splits=3)
    best_score = -999
    best_params = None
    best_std = 999

    results = []

    print(f"총 {len(alpha_values) * len(max_iter_values)}개 조합 테스트 중...")

    for params in ParameterGrid(param_grid):
        scores = []
        mae_scores = []

        for train_idx, test_idx in tscv.split(X_clean):
            X_train, X_test = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
            y_train, y_test = y_clean.iloc[train_idx], y_clean.iloc[test_idx]

            # 스케일링
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            try:
                model = Lasso(alpha=params['alpha'], max_iter=params['max_iter'], random_state=42)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)

                score = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)

                scores.append(score)
                mae_scores.append(mae)
            except:
                scores.append(-999)
                mae_scores.append(999)

        avg_score = np.mean(scores)
        std_score = np.std(scores)
        avg_mae = np.mean(mae_scores)

        results.append({
            'params': params,
            'mean_r2': avg_score,
            'std_r2': std_score,
            'mean_mae': avg_mae
        })

        if avg_score > best_score:
            best_score = avg_score
            best_params = params
            best_std = std_score

    # 결과 정렬
    results.sort(key=lambda x: x['mean_r2'], reverse=True)

    print(f"\n🏆 최적 Lasso 파라미터:")
    print(f"   Alpha: {best_params['alpha']}")
    print(f"   Max Iter: {best_params['max_iter']}")
    print(f"   R² = {best_score:.4f} ± {best_std:.4f}")

    print(f"\n상위 5개 결과:")
    for i, result in enumerate(results[:5]):
        params = result['params']
        print(f"  {i+1}. α={params['alpha']:6.4f}, iter={params['max_iter']:4d}: "
              f"R² = {result['mean_r2']:7.4f} ± {result['std_r2']:.4f}")

    return best_params, best_score, results

def optimize_elasticnet_hyperparameters(X, y):
    """ElasticNet 모델 하이퍼파라미터 최적화"""
    print("\n🔍 ElasticNet 하이퍼파라미터 최적화 중...")

    # 완전한 데이터만 사용
    combined_data = pd.concat([X, y], axis=1).dropna()
    X_clean = combined_data[X.columns]
    y_clean = combined_data[y.name]

    # 하이퍼파라미터 그리드
    alpha_values = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
    l1_ratio_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    param_grid = {
        'alpha': alpha_values,
        'l1_ratio': l1_ratio_values
    }

    tscv = TimeSeriesSplit(n_splits=3)
    best_score = -999
    best_params = None

    results = []

    for params in ParameterGrid(param_grid):
        scores = []

        for train_idx, test_idx in tscv.split(X_clean):
            X_train, X_test = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
            y_train, y_test = y_clean.iloc[train_idx], y_clean.iloc[test_idx]

            # 스케일링
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            try:
                model = ElasticNet(alpha=params['alpha'], l1_ratio=params['l1_ratio'],
                                 max_iter=3000, random_state=42)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)

                score = r2_score(y_test, y_pred)
                scores.append(score)
            except:
                scores.append(-999)

        avg_score = np.mean(scores)
        results.append({
            'params': params,
            'mean_r2': avg_score
        })

        if avg_score > best_score:
            best_score = avg_score
            best_params = params

    results.sort(key=lambda x: x['mean_r2'], reverse=True)

    print(f"🏆 최적 ElasticNet 파라미터:")
    print(f"   Alpha: {best_params['alpha']}")
    print(f"   L1 Ratio: {best_params['l1_ratio']}")
    print(f"   R² = {best_score:.4f}")

    return best_params, best_score, results

def final_model_validation(X, y, best_lasso_params):
    """최종 모델 검증 및 성능 비교"""
    print("\n🏁 최종 모델 검증 중...")

    # 완전한 데이터만 사용
    combined_data = pd.concat([X, y], axis=1).dropna()
    X_clean = combined_data[X.columns]
    y_clean = combined_data[y.name]

    # 시간 순서 분할 (최종 검증용)
    split_idx = int(len(X_clean) * 0.8)
    X_train, X_test = X_clean.iloc[:split_idx], X_clean.iloc[split_idx:]
    y_train, y_test = y_clean.iloc[:split_idx], y_clean.iloc[split_idx:]

    print(f"최종 검증 - 훈련: {len(X_train)}, 테스트: {len(X_test)}")

    # 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 최적화된 모델들
    models = {
        'Optimized Lasso': Lasso(
            alpha=best_lasso_params['alpha'],
            max_iter=best_lasso_params['max_iter'],
            random_state=42
        ),
        'Baseline Ridge': Ridge(alpha=1.0),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=8),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
    }

    final_results = {}

    for name, model in models.items():
        # 스케일링 적용 여부
        if 'Forest' in name or 'Boosting' in name:
            X_tr, X_te = X_train.values, X_test.values
        else:
            X_tr, X_te = X_train_scaled, X_test_scaled

        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)

        # 성능 메트릭
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        final_results[name] = {
            'r2_score': r2,
            'mse': mse,
            'rmse': rmse,
            'mae': mae
        }

        print(f"{name:20}: R² = {r2:7.4f}, MAE = {mae:.6f}")

        # 특성 중요도 (Lasso만)
        if name == 'Optimized Lasso':
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'coefficient': np.abs(model.coef_)
            }).sort_values('coefficient', ascending=False)

            print(f"\n상위 10개 중요 특성:")
            for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
                print(f"  {i+1:2d}. {row['feature']:25}: {row['coefficient']:.6f}")

    return final_results

def main():
    """메인 하이퍼파라미터 최적화 함수"""
    print("🚀 하이퍼파라미터 최적화 및 최종 검증 시작")
    print("=" * 60)

    # 1. 데이터 로드 및 특성 생성 (이전과 동일)
    spy_data = load_enhanced_spy_data()
    comprehensive_features = create_comprehensive_features(spy_data)
    targets = create_future_volatility_targets(spy_data)

    # 2. 이전에 발견한 최고 특성들 사용
    if 'target_vol_5d' in targets.columns:
        combined_for_selection = pd.concat([comprehensive_features, targets[['target_vol_5d']]], axis=1).dropna()

        if len(combined_for_selection) > 100:
            # 상관관계 분석
            correlations = combined_for_selection[comprehensive_features.columns].corrwith(
                combined_for_selection['target_vol_5d']
            ).abs().sort_values(ascending=False)

            # 상위 15개 특성 + 상호작용
            top_15_features = correlations.head(15).index
            top_features_df = comprehensive_features[top_15_features]
            interaction_features = create_interaction_features(top_features_df, n_top=8)
            final_features = pd.concat([top_features_df, interaction_features], axis=1)

            print(f"📊 최적화용 특성 수: {len(final_features.columns)}")

            # 3. Lasso 하이퍼파라미터 최적화
            best_lasso_params, best_lasso_score, lasso_results = optimize_lasso_hyperparameters(
                final_features, targets['target_vol_5d']
            )

            # 4. ElasticNet 하이퍼파라미터 최적화 (비교용)
            best_elastic_params, best_elastic_score, elastic_results = optimize_elasticnet_hyperparameters(
                final_features, targets['target_vol_5d']
            )

            # 5. 최종 모델 검증
            final_results = final_model_validation(
                final_features, targets['target_vol_5d'], best_lasso_params
            )

            # 6. 결과 저장
            os.makedirs('results', exist_ok=True)

            optimization_results = {
                'timestamp': datetime.now().isoformat(),
                'optimization_summary': {
                    'best_lasso_r2': best_lasso_score,
                    'best_lasso_params': best_lasso_params,
                    'best_elastic_r2': best_elastic_score,
                    'best_elastic_params': best_elastic_params
                },
                'lasso_optimization': lasso_results[:10],  # 상위 10개만 저장
                'elastic_optimization': elastic_results[:10],
                'final_validation': final_results,
                'feature_count': len(final_features.columns)
            }

            with open('results/hyperparameter_optimization.json', 'w') as f:
                json.dump(optimization_results, f, indent=2, default=str)

            print(f"\n💾 최적화 결과 저장: results/hyperparameter_optimization.json")

            # 최종 성능 요약
            print(f"\n" + "=" * 60)
            print("🎯 최적화 완료 - 성능 요약")
            print("=" * 60)
            print(f"최고 Cross-Validation R²: {best_lasso_score:.4f}")

            best_final = max(final_results.items(), key=lambda x: x[1]['r2_score'])
            print(f"최고 Final Test R²:       {best_final[1]['r2_score']:.4f} ({best_final[0]})")
            print("=" * 60)

    return optimization_results if 'optimization_results' in locals() else {}

if __name__ == "__main__":
    results = main()