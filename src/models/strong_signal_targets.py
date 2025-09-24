"""
예측 가능한 강한 신호 타겟 생성

금융에서 실제로 예측 가능성이 높은 패턴들을 활용한 타겟 생성:
1. 평균 회귀 신호
2. 모멘텀 지속 신호
3. 변동성 예측 신호
4. 극값 반전 신호
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.model_selection import TimeSeriesSplit
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def create_mean_reversion_data():
    """평균 회귀 패턴이 강한 시뮬레이션 데이터"""
    np.random.seed(42)
    n_samples = 800

    # 강한 평균 회귀 성분
    mean_rev_strength = 0.3  # 평균 회귀 강도
    equilibrium = 0.0

    returns = np.zeros(n_samples)
    cumulative_deviation = 0

    for i in range(1, n_samples):
        # 평균에서 멀어질수록 강한 복원력
        reversion_force = -mean_rev_strength * cumulative_deviation
        noise = np.random.normal(0, 0.01)

        returns[i] = reversion_force + noise
        cumulative_deviation += returns[i]

    prices = 100 * np.exp(np.cumsum(returns))

    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')

    return pd.DataFrame({
        'price': prices,
        'log_returns': returns,
        'cumulative_deviation': np.cumsum(returns)
    }, index=dates)


def create_momentum_data():
    """모멘텀 효과가 강한 데이터"""
    np.random.seed(42)
    n_samples = 800

    returns = np.zeros(n_samples)
    momentum_factor = 0.2

    for i in range(1, n_samples):
        # 과거 5일 평균 수익률에 비례한 모멘텀
        if i >= 5:
            past_momentum = np.mean(returns[i-5:i])
            momentum_effect = momentum_factor * past_momentum
        else:
            momentum_effect = 0

        noise = np.random.normal(0, 0.015)
        returns[i] = momentum_effect + noise

    prices = 100 * np.exp(np.cumsum(returns))
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')

    return pd.DataFrame({
        'price': prices,
        'log_returns': returns
    }, index=dates)


def create_volatility_clustering_data():
    """변동성 군집이 강한 데이터"""
    np.random.seed(42)
    n_samples = 800

    returns = np.zeros(n_samples)
    volatility = 0.02

    for i in range(1, n_samples):
        # GARCH(1,1) 스타일 변동성 모델
        volatility = 0.1 * 0.02 + 0.1 * returns[i-1]**2 + 0.8 * volatility
        returns[i] = np.random.normal(0, volatility)

    prices = 100 * np.exp(np.cumsum(returns))
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')

    # 변동성 시계열 생성
    vol_series = np.zeros(n_samples)
    vol_series[0] = 0.02

    for i in range(1, n_samples):
        vol_series[i] = 0.1 * 0.02 + 0.1 * returns[i-1]**2 + 0.8 * vol_series[i-1]

    return pd.DataFrame({
        'price': prices,
        'log_returns': returns,
        'volatility': vol_series
    }, index=dates)


def create_strong_signal_features(data, data_type='mean_reversion'):
    """강한 신호를 위한 특성 생성"""
    enhanced = data.copy()
    returns = data['log_returns']
    prices = data['price']

    if data_type == 'mean_reversion':
        # 평균 회귀 관련 특성
        for window in [5, 10, 20]:
            ma = returns.rolling(window).mean()
            enhanced[f'deviation_from_mean_{window}'] = returns - ma
            enhanced[f'cumulative_deviation_{window}'] = (returns - ma).rolling(window).sum()

        # Z-score (평균회귀의 핵심 신호)
        for window in [10, 20, 50]:
            mean = returns.rolling(window).mean()
            std = returns.rolling(window).std()
            enhanced[f'zscore_{window}'] = (returns - mean) / (std + 1e-8)

        # 가격의 평균회귀
        for window in [10, 20]:
            price_ma = prices.rolling(window).mean()
            enhanced[f'price_deviation_{window}'] = (prices - price_ma) / price_ma

    elif data_type == 'momentum':
        # 모멘텀 관련 특성
        for window in [3, 5, 10, 20]:
            enhanced[f'momentum_{window}'] = returns.rolling(window).sum()
            enhanced[f'momentum_avg_{window}'] = returns.rolling(window).mean()

        # 가속도 (모멘텀의 변화)
        enhanced['momentum_acceleration'] = enhanced['momentum_5'].diff()

        # 연속 상승/하락
        enhanced['consecutive_positive'] = (returns > 0).astype(int).groupby(
            (returns <= 0).cumsum()).cumsum()
        enhanced['consecutive_negative'] = (returns < 0).astype(int).groupby(
            (returns >= 0).cumsum()).cumsum()

    elif data_type == 'volatility':
        # 변동성 관련 특성
        for window in [5, 10, 20]:
            vol = returns.rolling(window).std()
            enhanced[f'volatility_{window}'] = vol
            enhanced[f'vol_ratio_{window}'] = vol / (returns.rolling(window*2).std() + 1e-8)

        # 제곱 수익률 (변동성 프록시)
        enhanced['squared_returns'] = returns ** 2
        enhanced['abs_returns'] = np.abs(returns)

        # 변동성의 지속성
        vol_5 = returns.rolling(5).std()
        enhanced['vol_persistence'] = vol_5 / (vol_5.shift(5) + 1e-8)

    # 공통 특성
    for lag in [1, 2, 3]:
        enhanced[f'returns_lag_{lag}'] = returns.shift(lag)

    return enhanced.dropna()


def create_predictable_targets(data, data_type='mean_reversion'):
    """예측 가능한 타겟 생성"""
    enhanced = data.copy()
    returns = data['log_returns']

    if data_type == 'mean_reversion':
        # 평균 회귀 타겟 - 과도한 편차 후 반전
        for window in [5, 10]:
            mean = returns.rolling(window).mean()
            deviation = returns - mean
            # 현재 편차가 클수록 다음 기간 반전 가능성 높음
            enhanced[f'target_reversion_{window}d'] = -deviation.shift(-1)  # 반전 예측

        # 누적 편차 기반 타겟
        cumulative_dev = (returns - returns.rolling(20).mean()).rolling(10).sum()
        enhanced['target_mean_revert'] = -cumulative_dev.shift(-1)

    elif data_type == 'momentum':
        # 모멘텀 지속 타겟
        for window in [3, 5]:
            past_momentum = returns.rolling(window).sum()
            enhanced[f'target_momentum_continue_{window}d'] = past_momentum.shift(-1)

        # 트렌드 지속 타겟
        trend_5d = returns.rolling(5).mean()
        enhanced['target_trend_continue'] = trend_5d.shift(-1)

    elif data_type == 'volatility':
        # 변동성 지속 타겟
        current_vol = returns.rolling(5).std()
        enhanced['target_vol_next'] = current_vol.shift(-1)

        # 변동성 군집 타겟
        vol_change = current_vol.diff()
        enhanced['target_vol_cluster'] = vol_change.shift(-1)

    return enhanced


def test_strong_signal_performance():
    """강한 신호 타겟 성능 테스트"""
    print("🎯 예측 가능한 강한 신호 타겟 테스트")
    print("=" * 60)

    data_types = ['mean_reversion', 'momentum', 'volatility']
    data_generators = {
        'mean_reversion': create_mean_reversion_data,
        'momentum': create_momentum_data,
        'volatility': create_volatility_clustering_data
    }

    all_results = {}

    for data_type in data_types:
        print(f"\n📊 {data_type.upper()} 패턴 테스트")
        print("-" * 40)

        # 1. 데이터 생성
        raw_data = data_generators[data_type]()
        print(f"   원본 데이터: {len(raw_data)}개 관측치")

        # 2. 특성 엔지니어링
        enhanced_data = create_strong_signal_features(raw_data, data_type)
        print(f"   특성 생성: {len(enhanced_data.columns)}개 특성")

        # 3. 타겟 생성
        final_data = create_predictable_targets(enhanced_data, data_type)
        final_data = final_data.dropna()
        print(f"   최종 데이터: {len(final_data)}개 관측치")

        # 4. 특성과 타겟 분리
        target_columns = [col for col in final_data.columns if 'target_' in col]
        feature_columns = [col for col in final_data.columns
                          if 'target_' not in col and col not in ['price', 'cumulative_deviation', 'volatility']]

        X = final_data[feature_columns]

        print(f"   테스트할 타겟: {len(target_columns)}개")

        data_results = {}

        # 5. 각 타겟별 테스트
        for target_col in target_columns:
            y = final_data[target_col].dropna()
            X_target = X.loc[y.index]

            # 간단한 모델들로 테스트
            models = {
                'Linear': LinearRegression(),
                'Ridge': Ridge(alpha=1.0),
                'ElasticNet': ElasticNet(alpha=0.1, random_state=42),
                'RandomForest': RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
            }

            # 시계열 교차검증
            tscv = TimeSeriesSplit(n_splits=3)
            target_results = {}

            for model_name, model in models.items():
                cv_scores = []

                for train_idx, val_idx in tscv.split(X_target):
                    X_train, X_val = X_target.iloc[train_idx], X_target.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    score = r2_score(y_val, y_pred)
                    cv_scores.append(score)

                avg_score = np.mean(cv_scores)
                target_results[model_name] = avg_score

            best_score = max(target_results.values())
            best_model = max(target_results.keys(), key=lambda k: target_results[k])

            print(f"     {target_col:<30} {best_model:<12} R² = {best_score:.4f}")

            data_results[target_col] = {
                'scores': target_results,
                'best_score': best_score,
                'best_model': best_model
            }

        all_results[data_type] = data_results

    # 6. 전체 결과 요약
    print("\n🏆 전체 결과 요약")
    print("=" * 60)

    best_overall_score = -np.inf
    best_overall_type = None
    best_overall_target = None

    for data_type, data_results in all_results.items():
        type_best_score = -np.inf
        type_best_target = None

        for target, result in data_results.items():
            if result['best_score'] > type_best_score:
                type_best_score = result['best_score']
                type_best_target = target

        print(f"\n{data_type.upper()}:")
        print(f"   최고 성능: {type_best_target}")
        print(f"   R² 점수: {type_best_score:.4f}")

        if type_best_score > best_overall_score:
            best_overall_score = type_best_score
            best_overall_type = data_type
            best_overall_target = type_best_target

    print(f"\n🥇 전체 최고 성능:")
    print(f"   패턴 타입: {best_overall_type.upper()}")
    print(f"   타겟: {best_overall_target}")
    print(f"   R² 점수: {best_overall_score:.4f}")

    # 성능 평가
    if best_overall_score > 0.3:
        print("🎉 우수한 예측 성능! 실용성 높음")
    elif best_overall_score > 0.1:
        print("✅ 양호한 예측 성능! 의미있는 신호 감지")
    elif best_overall_score > 0.05:
        print("📈 적정한 예측 성능! 약한 신호 감지")
    elif best_overall_score > 0:
        print("📊 기본적인 예측 성능! 미약한 신호")
    else:
        print("⚠️ 예측 성능 미흡, 추가 최적화 필요")

    # 실제 금융 데이터에서의 기대치
    print(f"\n💡 실제 금융 데이터 적용 시 예상 성능:")
    realistic_expectation = best_overall_score * 0.3  # 실제 데이터는 노이즈가 더 많음
    print(f"   예상 R² 범위: {realistic_expectation:.4f} ~ {realistic_expectation * 1.5:.4f}")

    if realistic_expectation > 0.02:
        print("   ✅ 실제 적용 가능성 높음")
    else:
        print("   ⚠️ 실제 적용을 위해 추가 개선 필요")

    return all_results, best_overall_score


def test_with_current_data():
    """현재 시스템 데이터로 강한 신호 타겟 테스트"""
    print("\n🔄 현재 시스템 데이터로 강한 신호 적용 테스트")
    print("-" * 50)

    # 현재 시스템과 유사한 데이터 생성
    np.random.seed(42)
    n_samples = 500

    # 약한 평균회귀 + 노이즈 (현실적)
    returns = np.zeros(n_samples)
    cumulative = 0

    for i in range(1, n_samples):
        # 약한 평균회귀 (현실적 수준)
        reversion = -0.05 * cumulative
        noise = np.random.normal(0, 0.02)
        returns[i] = reversion + noise
        cumulative += returns[i]

    prices = 100 * np.exp(np.cumsum(returns))
    dates = pd.date_range('2022-01-01', periods=n_samples, freq='D')

    data = pd.DataFrame({
        'price': prices,
        'log_returns': returns
    }, index=dates)

    print(f"   데이터 크기: {len(data)}개 관측치")

    # 강한 신호 특성 및 타겟 적용
    enhanced_data = create_strong_signal_features(data, 'mean_reversion')
    final_data = create_predictable_targets(enhanced_data, 'mean_reversion')
    final_data = final_data.dropna()

    print(f"   처리 후 데이터: {len(final_data)}개 관측치")

    # 타겟별 테스트
    target_columns = [col for col in final_data.columns if 'target_' in col]
    feature_columns = [col for col in final_data.columns
                      if 'target_' not in col and col != 'price']

    X = final_data[feature_columns]

    best_score = -np.inf
    best_target = None

    for target_col in target_columns:
        y = final_data[target_col].dropna()
        X_target = X.loc[y.index]

        # Ridge 회귀로 빠른 테스트
        model = Ridge(alpha=1.0)
        tscv = TimeSeriesSplit(n_splits=3)

        cv_scores = []
        for train_idx, val_idx in tscv.split(X_target):
            X_train, X_val = X_target.iloc[train_idx], X_target.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            score = r2_score(y_val, y_pred)
            cv_scores.append(score)

        avg_score = np.mean(cv_scores)
        print(f"   {target_col:<30} R² = {avg_score:.4f}")

        if avg_score > best_score:
            best_score = avg_score
            best_target = target_col

    print(f"\n✅ 현실적 데이터에서 최고 성능:")
    print(f"   타겟: {best_target}")
    print(f"   R² 점수: {best_score:.4f}")

    return best_score


if __name__ == "__main__":
    # 강한 신호 타겟 테스트
    results, best_score = test_strong_signal_performance()

    # 현실적 데이터 테스트
    realistic_score = test_with_current_data()

    print(f"\n📋 최종 요약:")
    print(f"   이상적 조건 최고 R²: {best_score:.4f}")
    print(f"   현실적 조건 R²: {realistic_score:.4f}")
    print(f"   성능 차이: {((best_score - realistic_score) / abs(realistic_score) * 100) if realistic_score != 0 else 'N/A':.1f}%")