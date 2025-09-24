"""
실제 데이터로 R² 성능 검증

강한 신호 타겟들을 실제 SPY 데이터에 적용하여 성능 검증
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("⚠️ yfinance not available, using simulated data")

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.feature_selection import SelectKBest, f_regression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def get_real_spy_data():
    """실제 SPY 데이터 수집"""
    if YFINANCE_AVAILABLE:
        print("📊 실제 SPY 데이터 수집 중...")
        spy = yf.Ticker("SPY")
        data = spy.history(start="2020-01-01", end="2024-12-31")

        if not data.empty:
            prices = data['Close']
            returns = np.log(prices / prices.shift(1)).dropna()

            result = pd.DataFrame({
                'price': prices.loc[returns.index],
                'log_returns': returns
            })

            print(f"✅ 실제 SPY 데이터 수집 완료: {len(result)}개 관측치")
            return result

    # yfinance 없거나 실패시 현실적 시뮬레이션
    print("📊 현실적 SPY 시뮬레이션 데이터 생성...")
    np.random.seed(42)
    n_samples = 1200  # 약 5년

    # SPY의 실제 특성을 반영한 시뮬레이션
    returns = np.zeros(n_samples)

    # 연간 8% 상승 트렌드 (SPY 평균)
    annual_trend = 0.08 / 252

    # 변동성 군집과 약한 평균회귀
    volatility = 0.015
    cumulative_deviation = 0

    for i in range(1, n_samples):
        # 1. 장기 상승 트렌드
        trend_component = annual_trend

        # 2. 약한 평균 회귀 (과도한 상승/하락 후 조정)
        reversion_component = -0.02 * cumulative_deviation

        # 3. 변동성 군집 (GARCH 효과)
        volatility = 0.95 * volatility + 0.05 * abs(returns[i-1])
        volatility = max(0.005, min(0.05, volatility))  # 변동성 범위 제한

        # 4. 노이즈
        noise = np.random.normal(0, volatility)

        returns[i] = trend_component + reversion_component + noise
        cumulative_deviation += returns[i] - annual_trend

    prices = 300 * np.exp(np.cumsum(returns))  # SPY 시작 가격 근사
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')

    result = pd.DataFrame({
        'price': prices,
        'log_returns': returns
    }, index=dates)

    print(f"✅ 현실적 시뮬레이션 완료: {len(result)}개 관측치")
    return result


def create_real_world_features(data):
    """실제 데이터용 특성 생성"""
    enhanced = data.copy()
    returns = data['log_returns']
    prices = data['price']

    print("🔧 실제 데이터 특성 엔지니어링...")

    # 1. 기본 기술적 지표
    for window in [5, 10, 20, 50]:
        enhanced[f'ma_{window}'] = prices.rolling(window).mean()
        enhanced[f'price_ma_ratio_{window}'] = prices / enhanced[f'ma_{window}']

    # 2. 평균 회귀 특성 (가장 유망했던 패턴)
    for window in [5, 10, 20]:
        ma_returns = returns.rolling(window).mean()
        enhanced[f'deviation_from_mean_{window}'] = returns - ma_returns
        enhanced[f'cumulative_deviation_{window}'] = (returns - ma_returns).rolling(window).sum()

        # Z-score
        std_returns = returns.rolling(window).std()
        enhanced[f'zscore_{window}'] = (returns - ma_returns) / (std_returns + 1e-8)

    # 3. 모멘텀 특성
    for window in [3, 5, 10]:
        enhanced[f'momentum_{window}'] = returns.rolling(window).sum()
        enhanced[f'momentum_avg_{window}'] = returns.rolling(window).mean()

    # 4. 변동성 특성
    for window in [5, 10, 20]:
        enhanced[f'volatility_{window}'] = returns.rolling(window).std()
        enhanced[f'realized_vol_{window}'] = enhanced[f'volatility_{window}'] * np.sqrt(252)

    # 5. 래그 특성
    for lag in [1, 2, 3, 5]:
        enhanced[f'returns_lag_{lag}'] = returns.shift(lag)
        enhanced[f'vol_lag_{lag}'] = enhanced['volatility_5'].shift(lag)

    # 6. 가격 기반 특성
    for window in [10, 20]:
        price_ma = prices.rolling(window).mean()
        enhanced[f'price_deviation_{window}'] = (prices - price_ma) / price_ma

    # 7. 고급 통계 특성
    enhanced['skewness_20'] = returns.rolling(20).skew()
    enhanced['kurtosis_20'] = returns.rolling(20).kurt()

    print(f"✅ 특성 생성 완료: {len(enhanced.columns)}개 특성")
    return enhanced.dropna()


def create_real_world_targets(data):
    """실제 데이터용 타겟 생성"""
    enhanced = data.copy()
    returns = data['log_returns']

    print("🎯 실제 데이터 타겟 생성...")

    # 1. 평균 회귀 타겟 (가장 성공적이었음)
    for window in [5, 10, 20]:
        mean_returns = returns.rolling(window).mean()
        deviation = returns - mean_returns
        enhanced[f'target_mean_revert_{window}d'] = -deviation.shift(-1)

    # 2. 모멘텀 지속 타겟
    for window in [3, 5]:
        momentum = returns.rolling(window).mean()
        enhanced[f'target_momentum_{window}d'] = momentum.shift(-1)

    # 3. 변동성 예측 타겟
    vol_5d = returns.rolling(5).std()
    enhanced['target_volatility_next'] = vol_5d.shift(-1)

    # 4. 복합 타겟 (평균회귀 + 모멘텀)
    deviation_10d = returns - returns.rolling(10).mean()
    momentum_3d = returns.rolling(3).mean()
    enhanced['target_hybrid'] = (-0.5 * deviation_10d + 0.5 * momentum_3d).shift(-1)

    # 5. 스무딩된 타겟 (노이즈 감소)
    enhanced['target_smoothed_3d'] = returns.rolling(3).mean().shift(-3)
    enhanced['target_smoothed_5d'] = returns.rolling(5).mean().shift(-5)

    print(f"✅ 타겟 생성 완료: {len([c for c in enhanced.columns if 'target_' in c])}개 타겟")
    return enhanced


def validate_on_real_data():
    """실제 데이터로 R² 성능 검증"""
    print("🚀 실제 데이터 R² 성능 검증")
    print("=" * 60)

    # 1. 실제 데이터 수집
    data = get_real_spy_data()

    # 2. 특성 및 타겟 생성
    enhanced_data = create_real_world_features(data)
    final_data = create_real_world_targets(enhanced_data)
    final_data = final_data.dropna()

    print(f"💾 최종 데이터: {len(final_data)}개 관측치, {len(final_data.columns)}개 컬럼")

    # 3. 특성과 타겟 분리
    target_columns = [col for col in final_data.columns if 'target_' in col]
    feature_columns = [col for col in final_data.columns
                      if 'target_' not in col and col not in ['price']]

    X = final_data[feature_columns]

    print(f"📊 테스트 설정: {len(feature_columns)}개 특성, {len(target_columns)}개 타겟")

    # 4. 모델 정의
    models = {
        'Linear': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    }

    # 5. 각 타겟별 성능 테스트
    print(f"\n🤖 모델 성능 테스트")
    print("-" * 60)

    all_results = {}
    best_overall_score = -np.inf
    best_overall_config = None

    for target_col in target_columns:
        print(f"\n📈 {target_col} 타겟 테스트:")

        y = final_data[target_col].dropna()
        X_target = X.loc[y.index]

        # 특성 선택 (상위 20개)
        if len(X_target.columns) > 20:
            selector = SelectKBest(score_func=f_regression, k=20)
            X_selected = selector.fit_transform(X_target, y)
            selected_features = X_target.columns[selector.get_support()]
            X_selected = pd.DataFrame(X_selected, columns=selected_features, index=X_target.index)
        else:
            X_selected = X_target
            selected_features = X_target.columns

        # 시계열 교차검증
        tscv = TimeSeriesSplit(n_splits=5)
        target_results = {}

        for model_name, model in models.items():
            cv_scores = []

            for train_idx, val_idx in tscv.split(X_selected):
                X_train, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # 데이터 정규화
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)

                # 모델 훈련 및 예측
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_val_scaled)

                score = r2_score(y_val, y_pred)
                cv_scores.append(score)

            avg_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)
            target_results[model_name] = {
                'mean': avg_score,
                'std': std_score
            }

            print(f"   {model_name:<15} R² = {avg_score:.4f} (±{std_score:.4f})")

            # 전체 최고 성능 추적
            if avg_score > best_overall_score:
                best_overall_score = avg_score
                best_overall_config = {
                    'target': target_col,
                    'model': model_name,
                    'features': len(selected_features)
                }

        # 앙상블 테스트
        print(f"   {'Ensemble':<15} 앙상블 테스트 중...")
        ensemble_scores = []

        for train_idx, val_idx in tscv.split(X_selected):
            X_train, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # 앙상블 예측
            ensemble_pred = np.zeros(len(y_val))
            weights = {'Ridge': 0.3, 'RandomForest': 0.4, 'GradientBoosting': 0.3}

            for model_name, weight in weights.items():
                model = models[model_name]
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_val_scaled)
                ensemble_pred += weight * y_pred

            ensemble_score = r2_score(y_val, ensemble_pred)
            ensemble_scores.append(ensemble_score)

        avg_ensemble_score = np.mean(ensemble_scores)
        std_ensemble_score = np.std(ensemble_scores)
        print(f"   {'Ensemble':<15} R² = {avg_ensemble_score:.4f} (±{std_ensemble_score:.4f})")

        if avg_ensemble_score > best_overall_score:
            best_overall_score = avg_ensemble_score
            best_overall_config = {
                'target': target_col,
                'model': 'Ensemble',
                'features': len(selected_features)
            }

        all_results[target_col] = target_results
        all_results[target_col]['Ensemble'] = {
            'mean': avg_ensemble_score,
            'std': std_ensemble_score
        }

    # 6. 결과 요약
    print(f"\n🏆 실제 데이터 검증 결과 요약")
    print("=" * 60)

    print(f"\n🥇 최고 성능:")
    if best_overall_config:
        print(f"   타겟: {best_overall_config['target']}")
        print(f"   모델: {best_overall_config['model']}")
        print(f"   R² 점수: {best_overall_score:.4f}")
        print(f"   사용 특성: {best_overall_config['features']}개")

    # 타겟별 최고 성능
    print(f"\n📊 타겟별 최고 성능:")
    for target_col, results in all_results.items():
        best_model = max(results.keys(), key=lambda k: results[k]['mean'])
        best_score = results[best_model]['mean']
        print(f"   {target_col:<30} {best_model:<15} R² = {best_score:.4f}")

    # 성능 분석
    print(f"\n📈 성능 분석:")
    baseline_r2 = -0.01  # 기존 시스템 평균

    if best_overall_score > 0.1:
        print("🎉 우수한 실제 성능! 상용화 가능")
    elif best_overall_score > 0.05:
        print("✅ 양호한 실제 성능! 실용적 가치 있음")
    elif best_overall_score > 0.02:
        print("📈 적정한 실제 성능! 의미있는 개선")
    elif best_overall_score > 0:
        print("📊 기본적인 실제 성능! 약한 신호 감지")
    else:
        print("⚠️ 실제 성능 미흡, 추가 연구 필요")

    if best_overall_score > baseline_r2:
        improvement = ((best_overall_score - baseline_r2) / abs(baseline_r2)) * 100
        print(f"📊 기존 시스템 대비 개선: {improvement:.1f}%")

    return all_results, best_overall_score, best_overall_config


if __name__ == "__main__":
    results, best_score, best_config = validate_on_real_data()

    print(f"\n✅ 실제 데이터 검증 완료!")
    print(f"   최종 R² 성능: {best_score:.4f}")
    if best_config:
        print(f"   최적 설정: {best_config['target']} + {best_config['model']}")