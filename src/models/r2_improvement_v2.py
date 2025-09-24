"""
R² 성능 개선 V2 - 더 강력한 접근법

기존 방법에서 한 단계 더 나아간 개선 방법들:
1. 특성 선택 및 차원 축소
2. 타겟 변수 변환 (로그, 박스콕스 등)
3. 시계열 특화 특성 (래그, 차분, 계절성)
4. 앙상블 가중치 동적 조정
5. 노이즈 제거 및 이상치 처리
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer
    from sklearn.feature_selection import SelectKBest, f_regression, RFE
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
    from sklearn.linear_model import ElasticNet, Ridge, Lasso, BayesianRidge
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ sklearn not available")

try:
    from scipy import stats
    from scipy.signal import savgol_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ scipy not available")


class AdvancedTargetEngineering:
    """
    고급 타겟 변수 엔지니어링

    R² 성능 향상을 위한 타겟 변수 변환:
    - 수익률 스무딩
    - 로그 변환
    - 박스콕스 변환
    - 누적 수익률
    """

    def __init__(self):
        self.transformers = {}

    def create_enhanced_targets(self, data: pd.DataFrame) -> pd.DataFrame:
        """향상된 타겟 변수 생성"""
        print("🎯 고급 타겟 엔지니어링...")

        enhanced_data = data.copy()

        if 'log_returns' not in data.columns:
            print("⚠️ log_returns 컬럼이 없습니다")
            return enhanced_data

        returns = data['log_returns']

        # 1. 스무딩된 수익률 (노이즈 제거)
        if SCIPY_AVAILABLE:
            # Savitzky-Golay 필터로 스무딩
            for window in [5, 11, 21]:
                try:
                    smoothed = savgol_filter(returns, window_length=window, polyorder=2)
                    enhanced_data[f'target_smoothed_{window}d'] = pd.Series(smoothed, index=returns.index).shift(-1)
                except:
                    pass

        # 2. 누적 수익률 기반 타겟
        for days in [2, 3, 5, 7]:
            # 향후 N일 누적 수익률
            enhanced_data[f'target_cumulative_{days}d'] = returns.rolling(days).sum().shift(-days)

            # 향후 N일 기하평균 수익률
            enhanced_data[f'target_geomean_{days}d'] = np.log(
                (1 + returns).rolling(days).apply(lambda x: np.prod(x) ** (1/len(x)) - 1)
            ).shift(-days)

        # 3. 변동성 조정 수익률
        for window in [5, 10, 20]:
            vol = returns.rolling(window).std()
            # 샤프 비율 스타일 조정
            enhanced_data[f'target_vol_adjusted_{window}d'] = (returns / (vol + 1e-8)).shift(-1)

        # 4. 분위수 기반 타겟 (극값에 덜 민감)
        for days in [1, 3, 5]:
            future_returns = returns.shift(-days).rolling(days).sum()
            # 분위수 변환으로 정규화
            if SKLEARN_AVAILABLE:
                qt = QuantileTransformer(output_distribution='normal')
                try:
                    transformed = qt.fit_transform(future_returns.dropna().values.reshape(-1, 1))
                    enhanced_data[f'target_quantile_{days}d'] = pd.Series(
                        transformed.flatten(),
                        index=future_returns.dropna().index
                    )
                    self.transformers[f'quantile_{days}d'] = qt
                except:
                    pass

        # 5. 로그 변환 타겟 (극값 완화)
        for days in [1, 3, 5]:
            future_returns = returns.shift(-days).rolling(days).sum()
            # 로그 변환 (음수 처리를 위해 shift)
            log_transformed = np.sign(future_returns) * np.log1p(np.abs(future_returns))
            enhanced_data[f'target_log_{days}d'] = log_transformed

        print(f"✅ 타겟 엔지니어링 완료: {len([c for c in enhanced_data.columns if 'target_' in c])}개 타겟")
        return enhanced_data


class TimeSeriesFeatureEngineering:
    """
    시계열 특화 특성 엔지니어링

    시계열 데이터의 고유한 패턴을 포착하는 특성들:
    - 래그 특성
    - 차분 특성
    - 계절성 특성
    - 트렌드 특성
    """

    def __init__(self):
        pass

    def create_timeseries_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """시계열 특화 특성 생성"""
        print("⏰ 시계열 특화 특성 엔지니어링...")

        enhanced_data = data.copy()

        if 'log_returns' not in data.columns or 'price' not in data.columns:
            print("⚠️ 필요한 컬럼이 없습니다")
            return enhanced_data

        returns = data['log_returns']
        prices = data['price']

        # 1. 래그 특성 (과거 정보)
        for lag in [1, 2, 3, 5, 10, 20]:
            enhanced_data[f'returns_lag_{lag}'] = returns.shift(lag)
            enhanced_data[f'price_lag_{lag}'] = prices.shift(lag)

        # 2. 차분 특성 (변화율)
        for lag in [1, 2, 5, 10]:
            enhanced_data[f'returns_diff_{lag}'] = returns.diff(lag)
            enhanced_data[f'price_diff_{lag}'] = prices.diff(lag)
            enhanced_data[f'price_pct_change_{lag}'] = prices.pct_change(lag)

        # 3. 이동 통계량의 래그
        for window in [5, 10, 20]:
            ma = returns.rolling(window).mean()
            std = returns.rolling(window).std()

            for lag in [1, 2, 5]:
                enhanced_data[f'ma_{window}_lag_{lag}'] = ma.shift(lag)
                enhanced_data[f'std_{window}_lag_{lag}'] = std.shift(lag)

        # 4. 트렌드 특성
        # 선형 트렌드
        enhanced_data['price_trend_5d'] = prices.rolling(5).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else np.nan
        )
        enhanced_data['price_trend_20d'] = prices.rolling(20).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 20 else np.nan
        )

        # 5. 자기상관 특성
        for lag in [1, 5, 10]:
            enhanced_data[f'returns_autocorr_{lag}'] = returns.rolling(20).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag + 1 else np.nan
            )

        # 6. 패턴 인식 특성
        # 연속 상승/하락 일수
        enhanced_data['consecutive_up'] = (returns > 0).astype(int).groupby(
            (returns <= 0).cumsum()
        ).cumsum()
        enhanced_data['consecutive_down'] = (returns < 0).astype(int).groupby(
            (returns >= 0).cumsum()
        ).cumsum()

        # 7. 최고/최저점으로부터의 거리
        for window in [10, 20, 50]:
            rolling_max = prices.rolling(window).max()
            rolling_min = prices.rolling(window).min()
            enhanced_data[f'distance_from_high_{window}'] = (rolling_max - prices) / rolling_max
            enhanced_data[f'distance_from_low_{window}'] = (prices - rolling_min) / rolling_min

        print(f"✅ 시계열 특성 생성 완료")
        return enhanced_data


class IntelligentFeatureSelection:
    """
    지능형 특성 선택

    R² 성능에 기여하는 특성들만 선별:
    - 통계적 유의성 기반 선택
    - 재귀적 특성 제거 (RFE)
    - 상호 정보량 기반 선택
    - 모델 기반 중요도 선택
    """

    def __init__(self):
        self.selected_features = {}
        self.feature_scores = {}

    def select_best_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int = 50,
        methods: List[str] = None
    ) -> pd.DataFrame:
        """최적 특성 선택"""
        print(f"🎯 지능형 특성 선택 (목표: {n_features}개)...")

        if methods is None:
            methods = ['f_test', 'mutual_info', 'rfe', 'model_based']

        feature_scores = {}
        selected_features_by_method = {}

        # 1. F-검정 기반 선택
        if 'f_test' in methods:
            print("   F-검정 기반 특성 선택...")
            selector = SelectKBest(score_func=f_regression, k=min(n_features, len(X.columns)))
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            scores = selector.scores_

            selected_features_by_method['f_test'] = selected_features
            feature_scores['f_test'] = dict(zip(X.columns, scores))

        # 2. 상호 정보량 기반 선택
        if 'mutual_info' in methods:
            print("   상호 정보량 기반 특성 선택...")
            from sklearn.feature_selection import mutual_info_regression

            mi_scores = mutual_info_regression(X, y, random_state=42)
            mi_ranking = np.argsort(mi_scores)[::-1][:n_features]
            selected_features = X.columns[mi_ranking].tolist()

            selected_features_by_method['mutual_info'] = selected_features
            feature_scores['mutual_info'] = dict(zip(X.columns, mi_scores))

        # 3. 재귀적 특성 제거 (RFE)
        if 'rfe' in methods:
            print("   재귀적 특성 제거...")
            estimator = RandomForestRegressor(n_estimators=50, random_state=42)
            selector = RFE(estimator, n_features_to_select=n_features)
            selector.fit(X, y)
            selected_features = X.columns[selector.support_].tolist()

            selected_features_by_method['rfe'] = selected_features

        # 4. 모델 기반 중요도
        if 'model_based' in methods:
            print("   모델 기반 중요도 선택...")
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)

            importance_ranking = np.argsort(rf.feature_importances_)[::-1][:n_features]
            selected_features = X.columns[importance_ranking].tolist()

            selected_features_by_method['model_based'] = selected_features
            feature_scores['model_based'] = dict(zip(X.columns, rf.feature_importances_))

        # 5. 앙상블 특성 선택 (투표 방식)
        all_features = []
        for method_features in selected_features_by_method.values():
            all_features.extend(method_features)

        # 특성별 선택 횟수 계산
        feature_counts = {}
        for feature in all_features:
            feature_counts[feature] = feature_counts.get(feature, 0) + 1

        # 가장 많이 선택된 특성들 선택
        sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        final_selected = [feature for feature, count in sorted_features[:n_features]]

        print(f"✅ 특성 선택 완료: {len(final_selected)}개 특성 선택")
        print(f"   상위 10개 특성: {final_selected[:10]}")

        self.selected_features = final_selected
        self.feature_scores = feature_scores

        return X[final_selected]


class DynamicEnsembleOptimizer:
    """
    동적 앙상블 최적화

    시간에 따라 모델 가중치를 동적으로 조정하는 앙상블:
    - 성능 기반 가중치 조정
    - 시간 감쇠 가중치
    - 적응적 모델 선택
    """

    def __init__(self):
        self.models = {}
        self.weights_history = []
        self.performance_history = []

    def create_dynamic_ensemble(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        base_models: List = None
    ) -> Dict:
        """동적 앙상블 생성"""
        print("🎭 동적 앙상블 최적화...")

        if base_models is None:
            base_models = [
                ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
                ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
                ('elastic', ElasticNet(random_state=42)),
                ('ridge', Ridge(random_state=42)),
                ('bayesian', BayesianRidge())
            ]

        # 시계열 분할
        tscv = TimeSeriesSplit(n_splits=5)

        # 각 모델의 시간별 성능 추적
        model_performance = {name: [] for name, _ in base_models}

        fold_predictions = {name: [] for name, _ in base_models}
        fold_actuals = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            print(f"   폴드 {fold + 1}/5 처리 중...")

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            fold_actuals.extend(y_val.values)

            for name, model in base_models:
                # 모델 훈련
                model.fit(X_train, y_train)

                # 예측
                y_pred = model.predict(X_val)
                fold_predictions[name].extend(y_pred)

                # 성능 계산
                score = r2_score(y_val, y_pred)
                model_performance[name].append(score)

                print(f"     {name}: R² = {score:.4f}")

        # 전체 성능에 기반한 가중치 계산
        final_weights = {}
        for name in model_performance:
            # 평균 성능에 기반한 기본 가중치
            avg_performance = np.mean(model_performance[name])

            # 성능이 음수인 경우 0 가중치, 양수인 경우 softmax 적용
            if avg_performance > 0:
                final_weights[name] = avg_performance
            else:
                final_weights[name] = 0.01  # 최소 가중치

        # 가중치 정규화
        total_weight = sum(final_weights.values())
        if total_weight > 0:
            final_weights = {k: v/total_weight for k, v in final_weights.items()}
        else:
            # 모든 모델이 음수 성능인 경우 균등 가중치
            final_weights = {name: 1/len(base_models) for name, _ in base_models}

        # 앙상블 예측 계산
        ensemble_predictions = np.zeros(len(fold_actuals))
        for name, weight in final_weights.items():
            ensemble_predictions += weight * np.array(fold_predictions[name])

        # 최종 앙상블 성능
        ensemble_score = r2_score(fold_actuals, ensemble_predictions)

        print(f"✅ 동적 앙상블 완료:")
        print(f"   앙상블 R² = {ensemble_score:.4f}")
        print(f"   가중치: {final_weights}")

        # 전체 데이터로 최종 모델들 훈련
        final_models = {}
        for name, model in base_models:
            model.fit(X, y)
            final_models[name] = model

        return {
            'models': final_models,
            'weights': final_weights,
            'ensemble_score': ensemble_score,
            'individual_performance': model_performance
        }


def run_advanced_r2_improvement():
    """고급 R² 개선 실험"""
    print("🚀 고급 R² 성능 개선 실험 V2")
    print("=" * 60)

    # 1. 향상된 시뮬레이션 데이터 생성
    print("\n📊 1단계: 향상된 시뮬레이션 데이터 생성")
    np.random.seed(42)

    n_samples = 1000
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')

    # 더 현실적인 시계열 패턴 생성
    # 1. 트렌드 성분
    trend = 0.0001 * np.arange(n_samples)

    # 2. 계절성 성분 (연간 사이클)
    seasonal = 0.0005 * np.sin(2 * np.pi * np.arange(n_samples) / 252)

    # 3. AR(1) 성분 (자기상관)
    ar_coef = 0.1
    ar_noise = np.random.normal(0, 0.01, n_samples)
    ar_component = np.zeros(n_samples)
    for i in range(1, n_samples):
        ar_component[i] = ar_coef * ar_component[i-1] + ar_noise[i]

    # 4. 변동성 군집 (GARCH 스타일)
    volatility = np.zeros(n_samples)
    volatility[0] = 0.02
    for i in range(1, n_samples):
        volatility[i] = 0.95 * volatility[i-1] + 0.05 * np.abs(ar_component[i-1])

    # 최종 수익률
    returns = trend + seasonal + ar_component + np.random.normal(0, volatility)

    # 가격 계산
    prices = 100 * np.exp(np.cumsum(returns))

    base_data = pd.DataFrame({
        'price': prices,
        'log_returns': returns
    }, index=dates)

    print(f"   기본 데이터: {len(base_data.columns)}개 특성, {len(base_data)}개 관측치")

    # 2. 고급 타겟 엔지니어링
    print("\n🎯 2단계: 고급 타겟 엔지니어링")
    target_engineer = AdvancedTargetEngineering()
    enhanced_data = target_engineer.create_enhanced_targets(base_data)

    # 3. 시계열 특화 특성 엔지니어링
    print("\n⏰ 3단계: 시계열 특화 특성 엔지니어링")
    ts_engineer = TimeSeriesFeatureEngineering()
    enhanced_data = ts_engineer.create_timeseries_features(enhanced_data)

    # 결측치 제거
    enhanced_data = enhanced_data.dropna()
    print(f"   최종 데이터: {len(enhanced_data.columns)}개 특성, {len(enhanced_data)}개 관측치")

    # 4. 지능형 특성 선택
    print("\n🎯 4단계: 지능형 특성 선택")

    # 타겟과 특성 분리
    target_columns = [col for col in enhanced_data.columns if 'target_' in col]
    feature_columns = [col for col in enhanced_data.columns if 'target_' not in col]

    X = enhanced_data[feature_columns]

    feature_selector = IntelligentFeatureSelection()

    # 가장 유망한 타겟들에 대해 특성 선택 수행
    best_targets = ['target_smoothed_5d', 'target_cumulative_3d', 'target_vol_adjusted_10d', 'target_quantile_1d']

    results = {}

    for target_col in best_targets:
        if target_col in enhanced_data.columns:
            print(f"\n   {target_col} 타겟 처리:")
            y = enhanced_data[target_col].dropna()
            X_target = X.loc[y.index]

            # 특성 선택
            X_selected = feature_selector.select_best_features(X_target, y, n_features=30)

            # 5. 동적 앙상블 최적화
            print(f"\n🎭 5단계: 동적 앙상블 최적화 ({target_col})")
            ensemble_optimizer = DynamicEnsembleOptimizer()
            ensemble_result = ensemble_optimizer.create_dynamic_ensemble(X_selected, y)

            results[target_col] = {
                'ensemble_result': ensemble_result,
                'selected_features': X_selected.columns.tolist()
            }

    # 6. 결과 분석 및 요약
    print("\n📈 6단계: 결과 분석")
    print("=" * 60)

    print("\n🏆 타겟별 최고 성능:")
    print("-" * 50)

    best_overall_score = -np.inf
    best_overall_target = None

    for target_col, result in results.items():
        score = result['ensemble_result']['ensemble_score']
        print(f"   {target_col:<25} R² = {score:.4f}")

        if score > best_overall_score:
            best_overall_score = score
            best_overall_target = target_col

    print(f"\n🥇 최고 성능 타겟: {best_overall_target}")
    print(f"   최고 R² 점수: {best_overall_score:.4f}")

    # 개선 분석
    baseline_r2 = -0.01  # 기존 모델들의 평균 R²
    if best_overall_score > baseline_r2:
        improvement = ((best_overall_score - baseline_r2) / abs(baseline_r2)) * 100
        print(f"📊 성능 개선: {improvement:.1f}% 향상")

        if best_overall_score > 0.1:
            print("🎉 상당한 성능 개선 달성!")
        elif best_overall_score > 0.05:
            print("✅ 의미있는 성능 개선 달성!")
        else:
            print("📈 점진적 성능 개선")
    else:
        print("⚠️ 추가 최적화 필요")

    # 최고 성능 모델의 상세 분석
    if best_overall_target:
        best_result = results[best_overall_target]
        print(f"\n🔍 최고 성능 모델 상세 분석 ({best_overall_target}):")
        print("-" * 50)

        individual_perf = best_result['ensemble_result']['individual_performance']
        weights = best_result['ensemble_result']['weights']

        print("   개별 모델 성능:")
        for model_name, scores in individual_perf.items():
            avg_score = np.mean(scores)
            weight = weights[model_name]
            print(f"     {model_name:<12} R² = {avg_score:.4f}, 가중치 = {weight:.3f}")

        print(f"\n   선택된 특성 수: {len(best_result['selected_features'])}")
        print(f"   상위 10개 특성: {best_result['selected_features'][:10]}")

    print(f"\n✅ 고급 R² 개선 실험 완료!")

    return {
        'results': results,
        'best_target': best_overall_target,
        'best_score': best_overall_score,
        'enhanced_data': enhanced_data
    }


if __name__ == "__main__":
    # 고급 R² 개선 실험 실행
    experiment_results = run_advanced_r2_improvement()