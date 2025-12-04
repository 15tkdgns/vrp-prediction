#!/usr/bin/env python3
"""
GARCH Enhanced Volatility Model - Phase 2
실제 GARCH 모델링을 통한 조건부 이분산성 특성 추가
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import warnings
import os
import json
from datetime import datetime

# GARCH 모델링을 위한 라이브러리
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
    print("✅ ARCH 라이브러리 사용 가능")
except ImportError:
    ARCH_AVAILABLE = False
    print("⚠️ ARCH 라이브러리 없음, 근사 GARCH 사용")

warnings.filterwarnings('ignore')

def load_data_for_garch():
    """GARCH 모델링용 데이터 로드"""
    print("📊 GARCH 모델링용 데이터 로드 중...")

    # SPY 데이터
    spy = yf.download('SPY', start='2010-01-01', end='2024-12-31', progress=False)
    spy['returns'] = spy['Close'].pct_change()

    # VIX 데이터
    vix = yf.download('^VIX', start='2010-01-01', end='2024-12-31', progress=False)
    spy['vix'] = vix['Close'].reindex(spy.index, method='ffill')

    # 10년 국채 금리
    try:
        treasury = yf.download('^TNX', start='2010-01-01', end='2024-12-31', progress=False)
        spy['treasury_10y'] = treasury['Close'].reindex(spy.index, method='ffill')
    except:
        spy['treasury_10y'] = 2.5

    spy = spy.dropna()
    print(f"✅ 데이터 로드 완료: {len(spy)} 관측치")
    return spy

def create_garch_features(returns, window_size=252):
    """GARCH 모델 특성 생성"""
    print("🔧 GARCH 특성 생성 중...")

    garch_features = pd.DataFrame(index=returns.index)

    if ARCH_AVAILABLE:
        # 실제 GARCH 모델 사용
        print("  📈 실제 GARCH(1,1) 모델 적용 중...")

        # 롤링 윈도우로 GARCH 모델 적합
        garch_vol = []
        garch_residuals = []

        # 백분율 단위로 변환 (GARCH 모델링을 위해)
        returns_pct = returns * 100

        for i in range(window_size, len(returns)):
            try:
                # 과거 252일 데이터로 GARCH 적합
                window_data = returns_pct.iloc[i-window_size:i]

                # GARCH(1,1) 모델
                model = arch_model(window_data, vol='GARCH', p=1, q=1, rescale=False)
                fitted_model = model.fit(disp='off', show_warning=False)

                # 조건부 분산 (변동성)
                conditional_vol = fitted_model.conditional_volatility.iloc[-1] / 100  # 다시 소수로 변환
                garch_vol.append(conditional_vol)

                # 표준화 잔차
                std_residual = fitted_model.std_resid.iloc[-1]
                garch_residuals.append(std_residual)

            except:
                # 실패 시 이동평균 변동성 사용
                rolling_vol = window_data.std() / 100
                garch_vol.append(rolling_vol)
                garch_residuals.append(0)

        # NaN으로 초기화 후 값 할당
        garch_features['garch_vol'] = np.nan
        garch_features['garch_residuals'] = np.nan

        garch_features.iloc[window_size:, garch_features.columns.get_loc('garch_vol')] = garch_vol
        garch_features.iloc[window_size:, garch_features.columns.get_loc('garch_residuals')] = garch_residuals

    else:
        # ARCH 라이브러리가 없을 때 근사 GARCH 구현
        print("  📈 근사 GARCH 모델 적용 중...")

        # 간단한 GARCH(1,1) 근사
        returns_squared = returns ** 2

        # 초기값
        omega = returns_squared.var() * 0.1
        alpha = 0.1
        beta = 0.8

        garch_vol_approx = []
        current_var = returns_squared.iloc[0]

        for i, ret_sq in enumerate(returns_squared):
            if i == 0:
                garch_vol_approx.append(np.sqrt(current_var))
            else:
                # GARCH(1,1): σ²(t) = ω + α*ε²(t-1) + β*σ²(t-1)
                prev_ret_sq = returns_squared.iloc[i-1]
                current_var = omega + alpha * prev_ret_sq + beta * current_var
                garch_vol_approx.append(np.sqrt(current_var))

        garch_features['garch_vol'] = garch_vol_approx
        garch_features['garch_residuals'] = returns / pd.Series(garch_vol_approx, index=returns.index)

    # GARCH 기반 추가 특성
    garch_features['garch_vol_ma_5'] = garch_features['garch_vol'].rolling(5).mean()
    garch_features['garch_vol_ma_20'] = garch_features['garch_vol'].rolling(20).mean()
    garch_features['garch_vol_ratio'] = garch_features['garch_vol'] / garch_features['garch_vol_ma_20']

    # GARCH 변동성 vs 실현 변동성 비교
    realized_vol = returns.rolling(20).std()
    garch_features['garch_vs_realized'] = garch_features['garch_vol'] / (realized_vol + 1e-8)

    # GARCH 잔차 기반 특성
    garch_features['garch_resid_abs'] = garch_features['garch_residuals'].abs()
    garch_features['garch_resid_sq'] = garch_features['garch_residuals'] ** 2

    print(f"✅ GARCH 특성 생성 완료: {len(garch_features.columns)}개")
    return garch_features

def create_all_features_with_garch(data):
    """GARCH 포함 모든 특성 생성"""
    print("🔧 GARCH 포함 전체 특성 생성 중...")

    features = pd.DataFrame(index=data.index)
    returns = data['returns']
    prices = data['Close']
    high = data['High']
    low = data['Low']
    volume = data['Volume']

    # 1. 기존 핵심 특성들
    for window in [5, 10, 20, 50]:
        features[f'volatility_{window}'] = returns.rolling(window).std()
        features[f'realized_vol_{window}'] = features[f'volatility_{window}'] * np.sqrt(252)

    # 2. VIX 특성
    if 'vix' in data.columns:
        vix = data['vix']
        features['vix_level'] = vix
        features['vix_change'] = vix.pct_change()
        for window in [5, 10, 20]:
            features[f'vix_ma_{window}'] = vix.rolling(window).mean()
        features['vix_term_structure'] = vix / features['vix_ma_20']

    # 3. 경제 지표
    if 'treasury_10y' in data.columns:
        treasury = data['treasury_10y']
        features['treasury_10y'] = treasury
        features['treasury_change'] = treasury.diff()
        features['vix_treasury_spread'] = features['vix_level'] - treasury

    # 4. GARCH 특성 추가
    garch_features = create_garch_features(returns)
    features = pd.concat([features, garch_features], axis=1)

    # 5. 고급 변동성 측정
    for window in [5, 10, 20]:
        gk_vol = np.log(high / low) ** 2
        features[f'garman_klass_{window}'] = gk_vol.rolling(window).mean()

        intraday_range = (high - low) / prices
        features[f'intraday_vol_{window}'] = intraday_range.rolling(window).mean()

    # 6. 지수 가중 변동성
    for span in [10, 20]:
        features[f'ewm_vol_{span}'] = returns.ewm(span=span).std()

    # 7. 래그 특성
    for lag in [1, 2, 3, 5]:
        features[f'return_lag_{lag}'] = returns.shift(lag)
        features[f'vol_lag_{lag}'] = features['volatility_5'].shift(lag)

    # 8. GARCH와 다른 특성들의 상호작용
    if 'garch_vol' in features.columns:
        features['garch_vix_ratio'] = features['garch_vol'] / (features['vix_level'] / 100 / np.sqrt(252) + 1e-8)
        features['garch_realized_spread'] = features['garch_vol'] - features['volatility_20']

    print(f"✅ GARCH 포함 전체 특성 생성 완료: {len(features.columns)}개")
    return features

def create_future_volatility_targets(data):
    """미래 변동성 타겟 생성"""
    print("🎯 변동성 타겟 생성 중...")

    targets = pd.DataFrame(index=data.index)
    returns = data['returns']

    for window in [5]:  # 5일 집중
        vol_values = []
        for i in range(len(returns)):
            if i + window < len(returns):
                future_window = returns.iloc[i+1:i+1+window]
                vol_values.append(future_window.std())
            else:
                vol_values.append(np.nan)
        targets[f'target_vol_{window}d'] = vol_values

    return targets

def test_garch_models(X, y):
    """GARCH 포함 모델 테스트"""
    print(f"\n🤖 GARCH Enhanced 모델 테스트")
    print("=" * 60)

    combined_data = pd.concat([X, y], axis=1).dropna()
    print(f"유효 샘플 수: {len(combined_data)}")

    if len(combined_data) < 300:
        print("⚠️ 샘플 수 부족")
        return {}

    X_clean = combined_data[X.columns]
    y_clean = combined_data[y.name]

    # 교차 검증
    tscv = TimeSeriesSplit(n_splits=3)
    results = {}

    models = {
        'Lasso (α=0.0001)': Lasso(alpha=0.0001, max_iter=3000),
        'Lasso (α=0.0005)': Lasso(alpha=0.0005, max_iter=3000),
        'ElasticNet (α=0.0005)': ElasticNet(alpha=0.0005, l1_ratio=0.7, max_iter=3000),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=6),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=8)
    }

    for name, model in models.items():
        scores = []

        for train_idx, test_idx in tscv.split(X_clean):
            X_train, X_test = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
            y_train, y_test = y_clean.iloc[train_idx], y_clean.iloc[test_idx]

            if 'Forest' not in name and 'Boosting' not in name:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
            else:
                X_train_scaled = X_train.values
                X_test_scaled = X_test.values

            try:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                score = r2_score(y_test, y_pred)
                scores.append(score)
            except Exception as e:
                print(f"  모델 {name} 오류: {e}")
                scores.append(-999)

        avg_score = np.mean(scores)
        std_score = np.std(scores)

        results[name] = {
            'mean_r2': avg_score,
            'std_r2': std_score
        }

        print(f"{name:25}: R² = {avg_score:7.4f} ± {std_score:.4f}")

    return results

def main():
    """메인 GARCH 모델 함수"""
    print("🚀 GARCH Enhanced Volatility Model")
    print("=" * 60)

    # 1. 데이터 로드
    spy_data = load_data_for_garch()

    # 2. GARCH 포함 특성 생성
    all_features = create_all_features_with_garch(spy_data)

    # 3. 타겟 생성
    targets = create_future_volatility_targets(spy_data)

    # 4. 상관관계 분석
    if 'target_vol_5d' in targets.columns:
        combined = pd.concat([all_features, targets[['target_vol_5d']]], axis=1).dropna()

        if len(combined) > 300:
            print(f"\n📊 GARCH 포함 상관관계 분석 (샘플 수: {len(combined)})")

            correlations = combined[all_features.columns].corrwith(
                combined['target_vol_5d']
            ).abs().sort_values(ascending=False)

            print("상위 15개 특성:")
            for i, (feature, corr) in enumerate(correlations.head(15).items()):
                print(f"  {i+1:2d}. {feature:30}: {corr:.4f}")

            # 5. 상위 특성 선별
            top_15_features = correlations.head(15).index
            final_features = all_features[top_15_features]

            print(f"\n📊 GARCH 최종 특성 수: {len(final_features.columns)}")

            # 6. 모델 테스트
            results = test_garch_models(final_features, targets['target_vol_5d'])

            if results:
                valid_results = {k: v for k, v in results.items() if v['mean_r2'] > -900}
                if valid_results:
                    best_model = max(valid_results.items(), key=lambda x: x[1]['mean_r2'])
                    print(f"\n🏆 GARCH 최고 성능: {best_model[0]}")
                    print(f"   R² = {best_model[1]['mean_r2']:.4f} ± {best_model[1]['std_r2']:.4f}")

                    # V2 Lite와 비교
                    v2_lite_r2 = 0.4556  # 이전 결과
                    improvement = (best_model[1]['mean_r2'] - v2_lite_r2) / v2_lite_r2 * 100
                    print(f"\n📈 V2 Lite 대비 개선:")
                    print(f"   V2 Lite R²: {v2_lite_r2:.4f}")
                    print(f"   GARCH R²:   {best_model[1]['mean_r2']:.4f}")
                    print(f"   개선:       {improvement:+.1f}%")

                    # 결과 저장
                    os.makedirs('results', exist_ok=True)

                    garch_results = {
                        'version': 'GARCH_Enhanced',
                        'timestamp': datetime.now().isoformat(),
                        'garch_library': 'arch' if ARCH_AVAILABLE else 'approximation',
                        'samples': len(combined),
                        'features': len(final_features.columns),
                        'best_model': {
                            'name': best_model[0],
                            'r2_mean': best_model[1]['mean_r2'],
                            'r2_std': best_model[1]['std_r2']
                        },
                        'improvement_vs_v2_lite': improvement,
                        'top_features': top_15_features.tolist(),
                        'all_results': results
                    }

                    with open('results/garch_enhanced_model.json', 'w') as f:
                        json.dump(garch_results, f, indent=2, default=str)

                    print(f"\n💾 GARCH 결과 저장: results/garch_enhanced_model.json")

    print("=" * 60)

if __name__ == "__main__":
    main()