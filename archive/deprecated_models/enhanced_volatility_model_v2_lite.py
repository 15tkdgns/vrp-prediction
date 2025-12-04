#!/usr/bin/env python3
"""
Enhanced Volatility Model V2 Lite - 효율적인 버전
Phase 1: 데이터 확장 및 핵심 경제 지표만 추가
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, ElasticNet, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import warnings
import os
import json
from datetime import datetime

warnings.filterwarnings('ignore')

def load_extended_data_lite():
    """효율적인 확장 데이터 로드"""
    print("📊 확장된 데이터 로드 중 (2010-2024)...")

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
        print("✅ 10년 국채 금리 추가")
    except:
        spy['treasury_10y'] = 2.5
        print("⚠️ 국채 금리 기본값 사용")

    spy = spy.dropna()
    print(f"✅ 데이터 로드 완료: {len(spy)} 관측치")
    return spy

def create_core_features_v2(data):
    """핵심 특성만 생성 (효율성 우선)"""
    print("🔧 핵심 특성 생성 중...")

    features = pd.DataFrame(index=data.index)
    returns = data['returns']
    prices = data['Close']
    high = data['High']
    low = data['Low']
    volume = data['Volume']

    # 1. 핵심 변동성 특성
    for window in [5, 10, 20, 50]:
        features[f'volatility_{window}'] = returns.rolling(window).std()
        features[f'realized_vol_{window}'] = features[f'volatility_{window}'] * np.sqrt(252)

    # 2. VIX 특성 (기존 + 핵심만)
    if 'vix' in data.columns:
        vix = data['vix']
        features['vix_level'] = vix
        features['vix_change'] = vix.pct_change()
        for window in [5, 10, 20]:
            features[f'vix_ma_{window}'] = vix.rolling(window).mean()
            features[f'vix_std_{window}'] = vix.rolling(window).std()

        # VIX 기간구조 프록시
        features['vix_term_structure'] = vix / features['vix_ma_20']

    # 3. 경제 지표 (신규 - 핵심만)
    if 'treasury_10y' in data.columns:
        treasury = data['treasury_10y']
        features['treasury_10y'] = treasury
        features['treasury_change'] = treasury.diff()
        features['treasury_ma_20'] = treasury.rolling(20).mean()
        features['treasury_vol_20'] = treasury.rolling(20).std()

        # VIX-금리 스프레드
        features['vix_treasury_spread'] = features['vix_level'] - treasury

    # 4. 지수 가중 변동성 (핵심만)
    for span in [10, 20]:
        features[f'ewm_vol_{span}'] = returns.ewm(span=span).std()

    # 5. 고급 변동성 (핵심만)
    for window in [5, 10, 20]:
        # Garman-Klass
        gk_vol = np.log(high / low) ** 2
        features[f'garman_klass_{window}'] = gk_vol.rolling(window).mean()

        # 일중 변동성
        intraday_range = (high - low) / prices
        features[f'intraday_vol_{window}'] = intraday_range.rolling(window).mean()

    # 6. 래그 특성 (핵심만)
    for lag in [1, 2, 3, 5]:
        features[f'return_lag_{lag}'] = returns.shift(lag)
        features[f'vol_lag_{lag}'] = features['volatility_5'].shift(lag)

    # 7. 볼륨 특성 (핵심만)
    volume_ma_20 = volume.rolling(20).mean()
    features['volume_ratio'] = volume / (volume_ma_20 + 1e-8)

    # 8. 변동성 비율 (핵심만)
    features['vol_ratio_5_20'] = features['volatility_5'] / (features['volatility_20'] + 1e-8)
    features['vol_ratio_10_50'] = features['volatility_10'] / (features['volatility_50'] + 1e-8)

    # 9. Z-score (핵심만)
    mean_ret_20 = returns.rolling(20).mean()
    std_ret_20 = returns.rolling(20).std()
    features['return_zscore_20'] = (returns - mean_ret_20) / (std_ret_20 + 1e-8)

    mean_vol_50 = features['volatility_5'].rolling(50).mean()
    std_vol_50 = features['volatility_5'].rolling(50).std()
    features['vol_zscore_50'] = (features['volatility_5'] - mean_vol_50) / (std_vol_50 + 1e-8)

    print(f"✅ 핵심 특성 생성 완료: {len(features.columns)}개")
    return features

def create_simple_interactions(base_features, n_top=12):
    """간단한 상호작용 (상위 특성만)"""
    print(f"🔧 간단한 상호작용 생성 중...")

    selected_features = base_features.iloc[:, :n_top]
    interactions = pd.DataFrame(index=base_features.index)

    # 핵심 상호작용만 (계산 효율성)
    important_pairs = [
        ('vix_level', 'intraday_vol_5'),
        ('vix_level', 'treasury_10y'),
        ('vix_level', 'ewm_vol_10'),
        ('treasury_10y', 'volatility_10'),
        ('vix_term_structure', 'volatility_5')
    ]

    for col1, col2 in important_pairs:
        if col1 in selected_features.columns and col2 in selected_features.columns:
            interactions[f'{col1}_x_{col2}'] = selected_features[col1] * selected_features[col2]
            interactions[f'{col1}_div_{col2}'] = selected_features[col1] / (selected_features[col2] + 1e-8)

    print(f"✅ 상호작용 특성 {len(interactions.columns)}개 생성")
    return interactions

def create_future_volatility_targets(data):
    """미래 변동성 타겟 생성"""
    print("🎯 변동성 타겟 생성 중...")

    targets = pd.DataFrame(index=data.index)
    returns = data['returns']

    for window in [5]:  # 5일만 집중
        vol_values = []
        for i in range(len(returns)):
            if i + window < len(returns):
                future_window = returns.iloc[i+1:i+1+window]
                vol_values.append(future_window.std())
            else:
                vol_values.append(np.nan)
        targets[f'target_vol_{window}d'] = vol_values

    print(f"✅ 타겟 생성 완료")
    return targets

def test_lite_models(X, y):
    """라이트 모델 테스트"""
    print(f"\n🤖 V2 Lite 모델 테스트")
    print("=" * 50)

    # 데이터 정리
    combined_data = pd.concat([X, y], axis=1).dropna()
    print(f"유효 샘플 수: {len(combined_data)}")

    if len(combined_data) < 200:
        print("⚠️ 샘플 수 부족")
        return {}

    X_clean = combined_data[X.columns]
    y_clean = combined_data[y.name]

    # 교차 검증
    tscv = TimeSeriesSplit(n_splits=3)
    results = {}

    models = {
        'Lasso (α=0.0001)': Lasso(alpha=0.0001, max_iter=2000),
        'Lasso (α=0.0005)': Lasso(alpha=0.0005, max_iter=2000),
        'Lasso (α=0.001)': Lasso(alpha=0.001, max_iter=2000),
        'ElasticNet (α=0.0005)': ElasticNet(alpha=0.0005, l1_ratio=0.7, max_iter=2000),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=8)
    }

    for name, model in models.items():
        scores = []

        for train_idx, test_idx in tscv.split(X_clean):
            X_train, X_test = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
            y_train, y_test = y_clean.iloc[train_idx], y_clean.iloc[test_idx]

            if 'Forest' not in name:
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
            except:
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
    """메인 함수"""
    print("🚀 Enhanced Volatility Model V2 Lite")
    print("=" * 50)

    # 1. 확장된 데이터 로드
    spy_data = load_extended_data_lite()

    # 2. 핵심 특성 생성
    core_features = create_core_features_v2(spy_data)

    # 3. 타겟 생성
    targets = create_future_volatility_targets(spy_data)

    # 4. 상관관계 분석
    if 'target_vol_5d' in targets.columns:
        combined = pd.concat([core_features, targets[['target_vol_5d']]], axis=1).dropna()

        if len(combined) > 200:
            print(f"\n📊 상관관계 분석 (샘플 수: {len(combined)})")

            correlations = combined[core_features.columns].corrwith(
                combined['target_vol_5d']
            ).abs().sort_values(ascending=False)

            print("상위 15개 특성:")
            for i, (feature, corr) in enumerate(correlations.head(15).items()):
                print(f"  {i+1:2d}. {feature:25}: {corr:.4f}")

            # 5. 상위 특성 + 상호작용
            top_12_features = correlations.head(12).index
            top_features_df = core_features[top_12_features]

            interaction_features = create_simple_interactions(top_features_df)
            final_features = pd.concat([top_features_df, interaction_features], axis=1)

            print(f"\n📊 최종 특성 수: {len(final_features.columns)}")

            # 6. 모델 테스트
            results = test_lite_models(final_features, targets['target_vol_5d'])

            if results:
                valid_results = {k: v for k, v in results.items() if v['mean_r2'] > -900}
                if valid_results:
                    best_model = max(valid_results.items(), key=lambda x: x[1]['mean_r2'])
                    print(f"\n🏆 최고 성능: {best_model[0]}")
                    print(f"   R² = {best_model[1]['mean_r2']:.4f} ± {best_model[1]['std_r2']:.4f}")

                    # 개선 분석
                    baseline_r2 = 0.0988
                    improvement = (best_model[1]['mean_r2'] - baseline_r2) / abs(baseline_r2) * 100
                    print(f"\n📈 성능 개선:")
                    print(f"   기존 R²: {baseline_r2:.4f}")
                    print(f"   V2 R²:   {best_model[1]['mean_r2']:.4f}")
                    print(f"   개선:    {improvement:+.1f}%")

                    # 결과 저장
                    os.makedirs('results', exist_ok=True)

                    v2_lite_results = {
                        'version': 'V2_Lite',
                        'timestamp': datetime.now().isoformat(),
                        'data_period': '2010-2024',
                        'samples': len(combined),
                        'features': len(final_features.columns),
                        'best_model': {
                            'name': best_model[0],
                            'r2_mean': best_model[1]['mean_r2'],
                            'r2_std': best_model[1]['std_r2']
                        },
                        'improvement_vs_baseline': improvement,
                        'top_features': top_12_features.tolist(),
                        'all_results': results
                    }

                    with open('results/enhanced_model_v2_lite.json', 'w') as f:
                        json.dump(v2_lite_results, f, indent=2, default=str)

                    print(f"\n💾 결과 저장: results/enhanced_model_v2_lite.json")

    print("=" * 50)

if __name__ == "__main__":
    main()