"""
빠른 R² 개선 테스트

계산량을 줄이고 핵심적인 개선 방법들만 빠르게 테스트
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.preprocessing import StandardScaler, QuantileTransformer
    from sklearn.feature_selection import SelectKBest, f_regression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import ElasticNet, Ridge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import TimeSeriesSplit
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ sklearn not available")


def create_realistic_financial_data(n_samples=500):
    """현실적인 금융 데이터 생성"""
    np.random.seed(42)
    dates = pd.date_range('2022-01-01', periods=n_samples, freq='D')

    # 자기상관이 있는 수익률 생성
    returns = np.zeros(n_samples)
    volatility = 0.02

    for i in range(1, n_samples):
        # AR(1) + GARCH 스타일
        returns[i] = 0.1 * returns[i-1] + np.random.normal(0, volatility)
        volatility = 0.95 * volatility + 0.05 * abs(returns[i])

    prices = 100 * np.exp(np.cumsum(returns))

    return pd.DataFrame({
        'price': prices,
        'log_returns': returns
    }, index=dates)


def add_smart_features(data):
    """효과적인 특성 추가"""
    enhanced = data.copy()
    returns = data['log_returns']
    prices = data['price']

    # 1. 래그 특성 (과거 정보)
    for lag in [1, 2, 3, 5]:
        enhanced[f'returns_lag_{lag}'] = returns.shift(lag)

    # 2. 이동 통계량
    for window in [5, 10, 20]:
        enhanced[f'ma_{window}'] = returns.rolling(window).mean()
        enhanced[f'std_{window}'] = returns.rolling(window).std()
        enhanced[f'skew_{window}'] = returns.rolling(window).skew()

    # 3. 가격 모멘텀
    for days in [3, 5, 10]:
        enhanced[f'momentum_{days}d'] = prices.pct_change(days)

    # 4. 변동성 특성
    enhanced['realized_vol'] = returns.rolling(10).std() * np.sqrt(252)
    enhanced['vol_ratio'] = enhanced['std_5'] / (enhanced['std_20'] + 1e-8)

    # 5. 트렌드 특성
    enhanced['price_trend'] = prices / prices.rolling(20).mean() - 1
    enhanced['returns_zscore'] = (returns - returns.rolling(20).mean()) / (returns.rolling(20).std() + 1e-8)

    return enhanced.dropna()


def create_better_targets(data):
    """향상된 타겟 변수 생성"""
    enhanced = data.copy()
    returns = data['log_returns']

    # 1. 스무딩된 타겟 (노이즈 감소)
    enhanced['target_smoothed_3d'] = returns.rolling(3).mean().shift(-3)
    enhanced['target_smoothed_5d'] = returns.rolling(5).mean().shift(-5)

    # 2. 누적 수익률
    enhanced['target_cumsum_3d'] = returns.rolling(3).sum().shift(-3)
    enhanced['target_cumsum_5d'] = returns.rolling(5).sum().shift(-5)

    # 3. 변동성 조정 수익률
    vol = returns.rolling(10).std()
    enhanced['target_vol_adj'] = (returns / (vol + 1e-8)).shift(-1)

    # 4. 방향 예측 (확률적 타겟)
    enhanced['target_direction_3d'] = (returns.rolling(3).sum().shift(-3) > 0).astype(float)

    return enhanced


def test_r2_improvements():
    """R² 개선 테스트"""
    print("🚀 빠른 R² 개선 테스트")
    print("=" * 50)

    # 1. 데이터 생성
    print("\n📊 1. 현실적 금융 데이터 생성")
    data = create_realistic_financial_data(n_samples=500)
    print(f"   기본 데이터: {len(data)}개 관측치")

    # 2. 특성 엔지니어링
    print("\n🔧 2. 스마트 특성 엔지니어링")
    enhanced_data = add_smart_features(data)
    print(f"   특성 수: {len(enhanced_data.columns)}개")

    # 3. 타겟 엔지니어링
    print("\n🎯 3. 향상된 타겟 생성")
    enhanced_data = create_better_targets(enhanced_data)
    enhanced_data = enhanced_data.dropna()
    print(f"   최종 데이터: {len(enhanced_data)}개 관측치")

    # 4. 특성 선택
    target_columns = [col for col in enhanced_data.columns if 'target_' in col]
    feature_columns = [col for col in enhanced_data.columns if 'target_' not in col and col != 'price']

    X = enhanced_data[feature_columns]

    results = {}

    # 5. 각 타겟별 모델 테스트
    print("\n🤖 4. 모델 테스트")

    for target_col in target_columns[:3]:  # 상위 3개만 테스트
        print(f"\n   {target_col} 타겟:")

        y = enhanced_data[target_col].dropna()
        X_target = X.loc[y.index]

        # 특성 선택 (상위 15개)
        selector = SelectKBest(score_func=f_regression, k=min(15, len(X_target.columns)))
        X_selected = selector.fit_transform(X_target, y)
        selected_features = X_target.columns[selector.get_support()]

        # 시계열 교차검증
        tscv = TimeSeriesSplit(n_splits=3)

        models = {
            'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=50, random_state=42),
            'ElasticNet': ElasticNet(random_state=42),
            'Ridge': Ridge(random_state=42)
        }

        target_results = {}

        for model_name, model in models.items():
            cv_scores = []

            for train_idx, val_idx in tscv.split(X_selected):
                X_train, X_val = X_selected[train_idx], X_selected[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                score = r2_score(y_val, y_pred)
                cv_scores.append(score)

            avg_score = np.mean(cv_scores)
            target_results[model_name] = avg_score
            print(f"     {model_name:<15} R² = {avg_score:.4f}")

        # 앙상블 테스트
        ensemble_pred = np.zeros(len(y))
        ensemble_weights = {}

        # 성능 기반 가중치 계산
        total_positive_score = sum(max(0, score) for score in target_results.values())

        if total_positive_score > 0:
            for model_name, score in target_results.items():
                weight = max(0, score) / total_positive_score
                ensemble_weights[model_name] = weight
        else:
            # 모든 모델이 음수 성능인 경우 균등 가중치
            for model_name in target_results:
                ensemble_weights[model_name] = 1 / len(target_results)

        # 앙상블 예측 계산
        cv_ensemble_scores = []
        for train_idx, val_idx in tscv.split(X_selected):
            X_train, X_val = X_selected[train_idx], X_selected[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            ensemble_pred_fold = np.zeros(len(y_val))

            for model_name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                ensemble_pred_fold += ensemble_weights[model_name] * y_pred

            ensemble_score = r2_score(y_val, ensemble_pred_fold)
            cv_ensemble_scores.append(ensemble_score)

        ensemble_avg_score = np.mean(cv_ensemble_scores)
        target_results['Ensemble'] = ensemble_avg_score

        print(f"     {'Ensemble':<15} R² = {ensemble_avg_score:.4f}")
        print(f"     선택된 특성: {len(selected_features)}개")

        results[target_col] = {
            'scores': target_results,
            'selected_features': selected_features.tolist(),
            'ensemble_weights': ensemble_weights
        }

    # 6. 결과 요약
    print("\n📈 5. 결과 요약")
    print("=" * 50)

    best_overall_score = -np.inf
    best_target = None
    best_model = None

    print("\n🏆 타겟별 최고 성능:")
    for target_col, result in results.items():
        scores = result['scores']
        best_score = max(scores.values())
        best_model_name = max(scores.keys(), key=lambda k: scores[k])

        print(f"   {target_col:<25} {best_model_name:<15} R² = {best_score:.4f}")

        if best_score > best_overall_score:
            best_overall_score = best_score
            best_target = target_col
            best_model = best_model_name

    print(f"\n🥇 전체 최고 성능:")
    print(f"   타겟: {best_target}")
    print(f"   모델: {best_model}")
    print(f"   R² 점수: {best_overall_score:.4f}")

    # 개선 분석
    baseline_r2 = -0.01  # 기존 시스템 평균
    if best_overall_score > baseline_r2:
        if best_overall_score > 0:
            improvement = ((best_overall_score - baseline_r2) / abs(baseline_r2)) * 100
            print(f"\n📊 성능 개선: {improvement:.1f}% 향상")

            if best_overall_score > 0.2:
                print("🎉 우수한 성능 달성!")
            elif best_overall_score > 0.1:
                print("✅ 상당한 성능 개선!")
            elif best_overall_score > 0.05:
                print("📈 의미있는 개선!")
            else:
                print("📊 점진적 개선")
        else:
            print("📈 기준선 대비 개선되었으나 여전히 음수")
    else:
        print("⚠️ 추가 최적화 필요")

    # 최고 성능 모델의 특성 분석
    if best_target in results:
        best_result = results[best_target]
        print(f"\n🔍 최고 성능 모델 분석:")
        print(f"   선택된 특성 수: {len(best_result['selected_features'])}")
        print(f"   주요 특성: {best_result['selected_features'][:5]}")

        if 'ensemble_weights' in best_result:
            print(f"   앙상블 가중치:")
            for model, weight in best_result['ensemble_weights'].items():
                if weight > 0.01:  # 유의미한 가중치만 표시
                    print(f"     {model}: {weight:.3f}")

    print("\n✅ 빠른 R² 개선 테스트 완료!")

    return results, best_overall_score


if __name__ == "__main__":
    results, best_score = test_r2_improvements()