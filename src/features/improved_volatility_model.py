#!/usr/bin/env python3
"""
개선된 변동성 예측 모델
고급 특성 엔지니어링과 다양한 알고리즘을 통한 성능 개선
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import warnings
import os
import json
from datetime import datetime

warnings.filterwarnings('ignore')

def load_enhanced_spy_data():
    """확장된 SPY 데이터 로드"""
    print("📊 확장된 SPY 데이터 로드 중...")

    # SPY 데이터
    spy = yf.download('SPY', start='2015-01-01', end='2024-12-31', progress=False)
    spy['returns'] = spy['Close'].pct_change()

    # VIX 데이터 (변동성 지수)
    try:
        vix = yf.download('^VIX', start='2015-01-01', end='2024-12-31', progress=False)
        vix_close = vix['Close'].reindex(spy.index, method='ffill')
        spy['vix'] = vix_close
        print("✅ VIX 데이터 추가")
    except:
        spy['vix'] = 20.0
        print("⚠️ VIX 데이터 로드 실패, 기본값 사용")

    spy = spy.dropna()
    print(f"✅ 확장된 데이터 로드 완료: {len(spy)} 관측치")
    return spy

def create_comprehensive_features(data):
    """포괄적인 특성 생성"""
    print("🔧 포괄적인 특성 생성 중...")

    features = pd.DataFrame(index=data.index)
    returns = data['returns']
    prices = data['Close']
    high = data['High']
    low = data['Low']
    volume = data['Volume']

    # 1. 기본 변동성 특성
    for window in [5, 10, 20, 50]:
        features[f'volatility_{window}'] = returns.rolling(window).std()
        features[f'realized_vol_{window}'] = features[f'volatility_{window}'] * np.sqrt(252)

    # 2. 지수 가중 변동성 (GARCH 스타일)
    for span in [5, 10, 20]:
        features[f'ewm_vol_{span}'] = returns.ewm(span=span).std()

    # 3. 래그 특성
    for lag in [1, 2, 3, 5, 10]:
        features[f'return_lag_{lag}'] = returns.shift(lag)
        features[f'vol_lag_{lag}'] = features['volatility_5'].shift(lag)

    # 4. 고-저 기반 변동성 (Garman-Klass)
    for window in [5, 10, 20]:
        gk_vol = np.log(high / low) ** 2
        features[f'garman_klass_{window}'] = gk_vol.rolling(window).mean()

    # 5. 일중 변동성
    for window in [5, 10, 20]:
        intraday_range = (high - low) / prices
        features[f'intraday_vol_{window}'] = intraday_range.rolling(window).mean()

    # 6. VIX 기반 특성
    if 'vix' in data.columns:
        vix = data['vix']
        features['vix_level'] = vix
        features['vix_change'] = vix.pct_change()
        for window in [5, 10, 20]:
            features[f'vix_ma_{window}'] = vix.rolling(window).mean()
            features[f'vix_std_{window}'] = vix.rolling(window).std()

    # 7. 볼륨 기반 특성
    volume_ma_5 = volume.rolling(5).mean()
    volume_ma_10 = volume.rolling(10).mean()
    volume_ma_20 = volume.rolling(20).mean()

    features['volume_ma_5'] = volume_ma_5
    features['volume_ma_10'] = volume_ma_10
    features['volume_ma_20'] = volume_ma_20
    features['volume_ratio_5'] = volume / (volume_ma_5 + 1e-8)
    features['volume_ratio_20'] = volume / (volume_ma_20 + 1e-8)

    # 8. 수익률 통계
    for window in [5, 10, 20]:
        features[f'return_mean_{window}'] = returns.rolling(window).mean()
        features[f'return_skew_{window}'] = returns.rolling(window).skew()
        features[f'return_kurt_{window}'] = returns.rolling(window).kurt()

    # 9. 변동성 비율
    features['vol_ratio_5_20'] = features['volatility_5'] / (features['volatility_20'] + 1e-8)
    features['vol_ratio_10_50'] = features['volatility_10'] / (features['volatility_50'] + 1e-8)

    # 10. Z-score 특성
    for window in [20, 50]:
        mean_ret = returns.rolling(window).mean()
        std_ret = returns.rolling(window).std()
        features[f'return_zscore_{window}'] = (returns - mean_ret) / (std_ret + 1e-8)

        mean_vol = features['volatility_5'].rolling(window).mean()
        std_vol = features['volatility_5'].rolling(window).std()
        features[f'vol_zscore_{window}'] = (features['volatility_5'] - mean_vol) / (std_vol + 1e-8)

    # 11. 모멘텀 특성
    for window in [5, 10, 20]:
        features[f'momentum_{window}'] = returns.rolling(window).sum()
        features[f'price_momentum_{window}'] = prices / prices.shift(window) - 1

    # 12. 변동성 지속성
    vol_5 = features['volatility_5']
    for lag in [1, 2, 3, 5]:
        # 변동성 자기상관
        features[f'vol_autocorr_{lag}'] = vol_5.rolling(20).corr(vol_5.shift(lag))

    print(f"✅ 포괄적인 특성 생성 완료: {len(features.columns)}개")
    return features

def create_interaction_features(base_features, n_top=8):
    """상위 특성들의 상호작용 특성 생성"""
    print(f"🔧 상위 {n_top}개 특성 간 상호작용 생성 중...")

    # 상위 특성만 선택
    selected_features = base_features.iloc[:, :n_top]
    interactions = pd.DataFrame(index=base_features.index)

    # 곱셈 상호작용
    for i, col1 in enumerate(selected_features.columns):
        for j, col2 in enumerate(selected_features.columns[i+1:], i+1):
            if j < len(selected_features.columns):
                interactions[f'{col1}_x_{col2}'] = selected_features.iloc[:, i] * selected_features.iloc[:, j]

    # 비율 상호작용 (상위 5개만)
    for i in range(min(5, len(selected_features.columns))):
        for j in range(i+1, min(6, len(selected_features.columns))):
            col1 = selected_features.columns[i]
            col2 = selected_features.columns[j]
            interactions[f'{col1}_div_{col2}'] = selected_features.iloc[:, i] / (selected_features.iloc[:, j] + 1e-8)

    print(f"✅ 상호작용 특성 {len(interactions.columns)}개 생성")
    return interactions

def create_future_volatility_targets(data):
    """미래 변동성 타겟 생성"""
    print("🎯 미래 변동성 타겟 생성 중...")

    targets = pd.DataFrame(index=data.index)
    returns = data['returns']

    # 다양한 기간의 미래 변동성
    for window in [1, 3, 5, 10, 20]:
        vol_values = []
        for i in range(len(returns)):
            if i + window < len(returns):
                future_window = returns.iloc[i+1:i+1+window]
                vol_values.append(future_window.std())
            else:
                vol_values.append(np.nan)
        targets[f'target_vol_{window}d'] = vol_values

    print(f"✅ 타겟 생성 완료: {len(targets.columns)}개")
    return targets

def test_improved_models(X, y, model_name='5일 변동성'):
    """개선된 모델들 테스트"""
    print(f"\n🤖 {model_name} 예측 - 개선된 모델 테스트")
    print("=" * 60)

    # 완전한 데이터만 사용
    combined_data = pd.concat([X, y], axis=1).dropna()
    print(f"유효 샘플 수: {len(combined_data)}")

    if len(combined_data) < 200:
        print("⚠️ 샘플 수 부족")
        return {}

    X_clean = combined_data[X.columns]
    y_clean = combined_data[y.name]

    # 시간 순서 교차 검증
    tscv = TimeSeriesSplit(n_splits=3)
    results = {}

    models = {
        'Ridge (α=0.01)': Ridge(alpha=0.01),
        'Ridge (α=1.0)': Ridge(alpha=1.0),
        'Ridge (α=100.0)': Ridge(alpha=100.0),
        'Lasso (α=0.001)': Lasso(alpha=0.001, max_iter=3000),
        'Lasso (α=0.01)': Lasso(alpha=0.01, max_iter=3000),
        'ElasticNet (α=0.01)': ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=3000),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=8),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
    }

    for name, model in models.items():
        scores = []
        mae_scores = []

        for train_idx, test_idx in tscv.split(X_clean):
            X_train, X_test = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
            y_train, y_test = y_clean.iloc[train_idx], y_clean.iloc[test_idx]

            # 스케일링 (트리 기반 모델 제외)
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
                mae = mean_absolute_error(y_test, y_pred)

                scores.append(score)
                mae_scores.append(mae)
            except Exception as e:
                print(f"  모델 {name} 오류: {e}")
                scores.append(-999)
                mae_scores.append(999)

        avg_score = np.mean(scores)
        std_score = np.std(scores)
        avg_mae = np.mean(mae_scores)

        results[name] = {
            'mean_r2': avg_score,
            'std_r2': std_score,
            'mean_mae': avg_mae
        }

        print(f"{name:20}: R² = {avg_score:7.4f} ± {std_score:.4f}, MAE = {avg_mae:.6f}")

    return results

def main():
    """메인 개선된 모델 훈련 함수"""
    print("🚀 개선된 변동성 예측 모델 훈련 시작")
    print("=" * 60)

    # 1. 확장된 데이터 로드
    spy_data = load_enhanced_spy_data()

    # 2. 포괄적인 특성 생성
    comprehensive_features = create_comprehensive_features(spy_data)

    # 3. 타겟 생성
    targets = create_future_volatility_targets(spy_data)

    # 4. 특성-타겟 상관관계 분석 (상위 특성 선별)
    if 'target_vol_5d' in targets.columns:
        combined_for_selection = pd.concat([comprehensive_features, targets[['target_vol_5d']]], axis=1).dropna()

        if len(combined_for_selection) > 100:
            print(f"\n📊 특성-타겟 상관관계 분석 (샘플 수: {len(combined_for_selection)})")

            correlations = combined_for_selection[comprehensive_features.columns].corrwith(
                combined_for_selection['target_vol_5d']
            ).abs().sort_values(ascending=False)

            print("상위 15개 특성:")
            for i, (feature, corr) in enumerate(correlations.head(15).items()):
                print(f"  {i+1:2d}. {feature:25}: {corr:.4f}")

            # 5. 상위 특성 선별 및 상호작용 생성
            top_15_features = correlations.head(15).index
            top_features_df = comprehensive_features[top_15_features]

            # 상호작용 특성 생성 (상위 8개로 제한)
            interaction_features = create_interaction_features(top_features_df, n_top=8)

            # 최종 특성 세트
            final_features = pd.concat([top_features_df, interaction_features], axis=1)
            print(f"\n📊 최종 특성 수: {len(final_features.columns)}")

            # 6. 모델 테스트
            results = test_improved_models(final_features, targets['target_vol_5d'])

            # 최고 성능 모델 찾기
            if results:
                valid_results = {k: v for k, v in results.items() if v['mean_r2'] > -900}
                if valid_results:
                    best_model = max(valid_results.items(), key=lambda x: x[1]['mean_r2'])
                    print(f"\n🏆 최고 성능: {best_model[0]}")
                    print(f"   R² = {best_model[1]['mean_r2']:.4f} ± {best_model[1]['std_r2']:.4f}")
                    print(f"   MAE = {best_model[1]['mean_mae']:.6f}")

            # 7. 결과 저장
            os.makedirs('results', exist_ok=True)

            improved_results = {
                'timestamp': datetime.now().isoformat(),
                'data_source': 'Enhanced SPY + VIX (2015-2024)',
                'feature_counts': {
                    'comprehensive': len(comprehensive_features.columns),
                    'top_selected': len(top_features_df.columns),
                    'interactions': len(interaction_features.columns),
                    'final_total': len(final_features.columns)
                },
                'model_results': results,
                'top_features': top_15_features.tolist(),
                'best_model': {
                    'name': best_model[0] if 'best_model' in locals() else None,
                    'performance': best_model[1] if 'best_model' in locals() else None
                }
            }

            with open('results/improved_volatility_model.json', 'w') as f:
                json.dump(improved_results, f, indent=2, default=str)

            print(f"\n💾 개선된 모델 결과 저장: results/improved_volatility_model.json")

    print("=" * 60)

if __name__ == "__main__":
    main()