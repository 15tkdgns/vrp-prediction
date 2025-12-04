#!/usr/bin/env python3
"""
Enhanced Volatility Model V2 - 성능 향상 버전
Phase 1: 데이터 확장 및 경제 지표 추가를 통한 R² 성능 개선
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, ElasticNet, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import warnings
import os
import json
from datetime import datetime
import pickle

warnings.filterwarnings('ignore')

# 이전 모듈에서 함수 가져오기
import sys
sys.path.append('/root/workspace/src/features')

def load_extended_data(start_date='2010-01-01', end_date='2024-12-31'):
    """확장된 기간의 데이터 로드 (2010-2024)"""
    print(f"📊 확장된 데이터 로드: {start_date} ~ {end_date}")

    # SPY 데이터
    spy = yf.download('SPY', start=start_date, end=end_date, progress=False)
    spy['returns'] = spy['Close'].pct_change()

    # VIX 데이터
    vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)
    spy['vix'] = vix['Close'].reindex(spy.index, method='ffill')

    # 10년 국채 금리 (^TNX)
    try:
        treasury_10y = yf.download('^TNX', start=start_date, end=end_date, progress=False)
        spy['treasury_10y'] = treasury_10y['Close'].reindex(spy.index, method='ffill')
        print("✅ 10년 국채 금리 데이터 추가")
    except:
        spy['treasury_10y'] = 2.5  # 기본값
        print("⚠️ 10년 국채 금리 데이터 로드 실패, 기본값 사용")

    # 2년 국채 금리 (^IRX를 2년 프록시로 사용)
    try:
        treasury_2y = yf.download('^TNX', start=start_date, end=end_date, progress=False)  # 임시로 TNX 사용
        spy['treasury_2y'] = (treasury_10y['Close'] * 0.7).reindex(spy.index, method='ffill')  # 근사값
        print("✅ 2년 국채 금리 프록시 추가")
    except:
        spy['treasury_2y'] = 1.5  # 기본값
        print("⚠️ 2년 국채 금리 데이터 로드 실패, 기본값 사용")

    # 수익률 곡선 기울기
    spy['yield_curve_slope'] = spy['treasury_10y'] - spy['treasury_2y']

    spy = spy.dropna()
    print(f"✅ 확장된 데이터 로드 완료: {len(spy)} 관측치 ({start_date} ~ {end_date})")
    return spy

def create_enhanced_features_v2(data):
    """V2 향상된 특성 생성 - 경제 지표 포함"""
    print("🔧 V2 향상된 특성 생성 중...")

    features = pd.DataFrame(index=data.index)
    returns = data['returns']
    prices = data['Close']
    high = data['High']
    low = data['Low']
    volume = data['Volume']

    # 1. 기본 변동성 특성 (기존)
    for window in [5, 10, 20, 50, 100]:  # 100일 추가
        features[f'volatility_{window}'] = returns.rolling(window).std()
        features[f'realized_vol_{window}'] = features[f'volatility_{window}'] * np.sqrt(252)

    # 2. 지수 가중 변동성 (기존 + 추가)
    for span in [5, 10, 20, 50]:  # 50일 추가
        features[f'ewm_vol_{span}'] = returns.ewm(span=span).std()

    # 3. VIX 기반 특성 (기존 + 강화)
    if 'vix' in data.columns:
        vix = data['vix']
        features['vix_level'] = vix
        features['vix_change'] = vix.pct_change()
        for window in [5, 10, 20, 50]:  # 50일 추가
            features[f'vix_ma_{window}'] = vix.rolling(window).mean()
            features[f'vix_std_{window}'] = vix.rolling(window).std()

        # VIX 기간구조 프록시 (VIX vs 이동평균 비교)
        features['vix_term_structure'] = vix / features['vix_ma_20']
        features['vix_contango'] = features['vix_ma_5'] / features['vix_ma_20']
        features['vix_backwardation'] = np.where(features['vix_contango'] < 1, 1, 0)

    # 4. 경제 지표 기반 특성 (신규)
    if 'treasury_10y' in data.columns:
        treasury_10y = data['treasury_10y']
        features['treasury_10y_level'] = treasury_10y
        features['treasury_10y_change'] = treasury_10y.diff()
        for window in [5, 10, 20]:
            features[f'treasury_10y_ma_{window}'] = treasury_10y.rolling(window).mean()
            features[f'treasury_10y_vol_{window}'] = treasury_10y.rolling(window).std()

    if 'yield_curve_slope' in data.columns:
        yield_slope = data['yield_curve_slope']
        features['yield_curve_slope'] = yield_slope
        features['yield_slope_change'] = yield_slope.diff()
        for window in [5, 10, 20]:
            features[f'yield_slope_ma_{window}'] = yield_slope.rolling(window).mean()
            features[f'yield_slope_vol_{window}'] = yield_slope.rolling(window).std()

    # 5. 고급 변동성 측정 (기존 + 추가)
    for window in [5, 10, 20, 50]:  # 50일 추가
        # Garman-Klass 변동성
        gk_vol = np.log(high / low) ** 2
        features[f'garman_klass_{window}'] = gk_vol.rolling(window).mean()

        # 일중 변동성
        intraday_range = (high - low) / prices
        features[f'intraday_vol_{window}'] = intraday_range.rolling(window).mean()

    # 6. 래그 특성 (기존 + 추가)
    for lag in [1, 2, 3, 5, 10]:  # 10일 래그 추가
        features[f'return_lag_{lag}'] = returns.shift(lag)
        features[f'vol_lag_{lag}'] = features['volatility_5'].shift(lag)
        if 'vix_level' in features.columns:
            features[f'vix_lag_{lag}'] = features['vix_level'].shift(lag)

    # 7. 볼륨 기반 특성 (기존)
    volume_ma_5 = volume.rolling(5).mean()
    volume_ma_20 = volume.rolling(20).mean()
    volume_ma_50 = volume.rolling(50).mean()  # 추가

    features['volume_ma_5'] = volume_ma_5
    features['volume_ma_20'] = volume_ma_20
    features['volume_ma_50'] = volume_ma_50
    features['volume_ratio_5'] = volume / (volume_ma_5 + 1e-8)
    features['volume_ratio_20'] = volume / (volume_ma_20 + 1e-8)
    features['volume_ratio_50'] = volume / (volume_ma_50 + 1e-8)

    # 8. 수익률 통계 (기존 + 추가)
    for window in [5, 10, 20, 50]:  # 50일 추가
        features[f'return_mean_{window}'] = returns.rolling(window).mean()
        features[f'return_skew_{window}'] = returns.rolling(window).skew()
        features[f'return_kurt_{window}'] = returns.rolling(window).kurt()

    # 9. 변동성 비율 (기존 + 추가)
    features['vol_ratio_5_20'] = features['volatility_5'] / (features['volatility_20'] + 1e-8)
    features['vol_ratio_10_50'] = features['volatility_10'] / (features['volatility_50'] + 1e-8)
    features['vol_ratio_20_100'] = features['volatility_20'] / (features['volatility_100'] + 1e-8)

    # 10. Z-score 특성 (기존 + 추가)
    for window in [20, 50, 100]:  # 100일 추가
        # 수익률 Z-score
        mean_ret = returns.rolling(window).mean()
        std_ret = returns.rolling(window).std()
        features[f'return_zscore_{window}'] = (returns - mean_ret) / (std_ret + 1e-8)

        # 변동성 Z-score
        mean_vol = features['volatility_5'].rolling(window).mean()
        std_vol = features['volatility_5'].rolling(window).std()
        features[f'vol_zscore_{window}'] = (features['volatility_5'] - mean_vol) / (std_vol + 1e-8)

    # 11. 모멘텀 특성 (기존 + 추가)
    for window in [5, 10, 20, 50]:  # 50일 추가
        features[f'momentum_{window}'] = returns.rolling(window).sum()
        features[f'price_momentum_{window}'] = prices / prices.shift(window) - 1

    # 12. 변동성 지속성 (기존 + 추가)
    vol_5 = features['volatility_5']
    for lag in [1, 2, 3, 5, 10]:  # 10일 추가
        features[f'vol_autocorr_{lag}'] = vol_5.rolling(50).corr(vol_5.shift(lag))

    # 13. 교차 상관관계 (신규)
    if 'vix_level' in features.columns and 'treasury_10y_level' in features.columns:
        # VIX-금리 상관관계
        for window in [10, 20, 50]:
            features[f'vix_treasury_corr_{window}'] = features['vix_level'].rolling(window).corr(
                features['treasury_10y_level']
            )

    print(f"✅ V2 향상된 특성 생성 완료: {len(features.columns)}개")
    return features

def create_enhanced_interaction_features(base_features, n_top=10):
    """향상된 상호작용 특성 생성"""
    print(f"🔧 향상된 상호작용 특성 생성 중 (상위 {n_top}개)...")

    selected_features = base_features.iloc[:, :n_top]
    interactions = pd.DataFrame(index=base_features.index)

    # 곱셈 상호작용
    for i, col1 in enumerate(selected_features.columns):
        for j, col2 in enumerate(selected_features.columns[i+1:], i+1):
            if j < len(selected_features.columns):
                interactions[f'{col1}_x_{col2}'] = selected_features.iloc[:, i] * selected_features.iloc[:, j]

    # 비율 상호작용 (상위 6개만)
    for i in range(min(6, len(selected_features.columns))):
        for j in range(i+1, min(7, len(selected_features.columns))):
            col1 = selected_features.columns[i]
            col2 = selected_features.columns[j]
            interactions[f'{col1}_div_{col2}'] = selected_features.iloc[:, i] / (selected_features.iloc[:, j] + 1e-8)

    # 제곱 특성 (상위 5개만)
    for i in range(min(5, len(selected_features.columns))):
        col = selected_features.columns[i]
        interactions[f'{col}_squared'] = selected_features.iloc[:, i] ** 2

    print(f"✅ 향상된 상호작용 특성 {len(interactions.columns)}개 생성")
    return interactions

def create_future_volatility_targets(data):
    """미래 변동성 타겟 생성 (기존과 동일)"""
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

def test_enhanced_models_v2(X, y, model_name='5일 변동성 V2'):
    """향상된 모델들 테스트 V2"""
    print(f"\n🤖 {model_name} 예측 - 향상된 모델 테스트")
    print("=" * 60)

    # 완전한 데이터만 사용
    combined_data = pd.concat([X, y], axis=1).dropna()
    print(f"유효 샘플 수: {len(combined_data)}")

    if len(combined_data) < 300:  # 최소 샘플 수 증가
        print("⚠️ 샘플 수 부족")
        return {}

    X_clean = combined_data[X.columns]
    y_clean = combined_data[y.name]

    # 시간 순서 교차 검증 (더 많은 폴드)
    tscv = TimeSeriesSplit(n_splits=5)  # 3 -> 5로 증가
    results = {}

    models = {
        'Ridge (α=0.01)': Ridge(alpha=0.01),
        'Ridge (α=1.0)': Ridge(alpha=1.0),
        'Ridge (α=10.0)': Ridge(alpha=10.0),
        'Lasso (α=0.0001)': Lasso(alpha=0.0001, max_iter=3000),
        'Lasso (α=0.0005)': Lasso(alpha=0.0005, max_iter=3000),
        'Lasso (α=0.001)': Lasso(alpha=0.001, max_iter=3000),
        'Lasso (α=0.005)': Lasso(alpha=0.005, max_iter=3000),
        'ElasticNet (α=0.0005, l1=0.5)': ElasticNet(alpha=0.0005, l1_ratio=0.5, max_iter=3000),
        'ElasticNet (α=0.0005, l1=0.7)': ElasticNet(alpha=0.0005, l1_ratio=0.7, max_iter=3000),
        'RandomForest': RandomForestRegressor(n_estimators=200, random_state=42, max_depth=10),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=200, random_state=42, max_depth=6)
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

        print(f"{name:30}: R² = {avg_score:7.4f} ± {std_score:.4f}, MAE = {avg_mae:.6f}")

    return results

def main():
    """메인 향상된 모델 훈련 함수"""
    print("🚀 Enhanced Volatility Model V2 - 성능 향상 버전")
    print("=" * 70)

    # 1. 확장된 데이터 로드 (2010-2024)
    spy_data = load_extended_data('2010-01-01', '2024-12-31')

    # 2. V2 향상된 특성 생성
    enhanced_features = create_enhanced_features_v2(spy_data)

    # 3. 타겟 생성
    targets = create_future_volatility_targets(spy_data)

    # 4. 특성-타겟 상관관계 분석 (더 많은 특성 고려)
    if 'target_vol_5d' in targets.columns:
        combined_for_selection = pd.concat([enhanced_features, targets[['target_vol_5d']]], axis=1).dropna()

        if len(combined_for_selection) > 200:
            print(f"\n📊 V2 특성-타겟 상관관계 분석 (샘플 수: {len(combined_for_selection)})")

            correlations = combined_for_selection[enhanced_features.columns].corrwith(
                combined_for_selection['target_vol_5d']
            ).abs().sort_values(ascending=False)

            print("상위 20개 특성:")
            for i, (feature, corr) in enumerate(correlations.head(20).items()):
                print(f"  {i+1:2d}. {feature:30}: {corr:.4f}")

            # 5. 상위 특성 선별 및 상호작용 생성 (더 많은 특성)
            top_20_features = correlations.head(20).index
            top_features_df = enhanced_features[top_20_features]

            # 상호작용 특성 생성 (상위 10개로 제한)
            interaction_features = create_enhanced_interaction_features(top_features_df, n_top=10)

            # 최종 특성 세트
            final_features = pd.concat([top_features_df, interaction_features], axis=1)
            print(f"\n📊 V2 최종 특성 수: {len(final_features.columns)}")

            # 6. 모델 테스트
            results = test_enhanced_models_v2(final_features, targets['target_vol_5d'])

            # 최고 성능 모델 찾기
            if results:
                valid_results = {k: v for k, v in results.items() if v['mean_r2'] > -900}
                if valid_results:
                    best_model = max(valid_results.items(), key=lambda x: x[1]['mean_r2'])
                    print(f"\n🏆 V2 최고 성능: {best_model[0]}")
                    print(f"   R² = {best_model[1]['mean_r2']:.4f} ± {best_model[1]['std_r2']:.4f}")
                    print(f"   MAE = {best_model[1]['mean_mae']:.6f}")

                    # 기존 모델과 비교
                    print(f"\n📈 기존 모델 대비 개선:")
                    baseline_r2 = 0.0988  # 기존 최종 검증 성능
                    improvement = (best_model[1]['mean_r2'] - baseline_r2) / abs(baseline_r2) * 100
                    print(f"   기존 R²: {baseline_r2:.4f}")
                    print(f"   V2 R²:   {best_model[1]['mean_r2']:.4f}")
                    print(f"   개선:    {improvement:+.1f}%")

            # 7. 결과 저장
            os.makedirs('results', exist_ok=True)

            v2_results = {
                'version': 'V2',
                'timestamp': datetime.now().isoformat(),
                'data_period': '2010-2024 (확장)',
                'data_source': 'SPY + VIX + Treasury rates',
                'feature_counts': {
                    'enhanced_total': len(enhanced_features.columns),
                    'top_selected': len(top_features_df.columns),
                    'interactions': len(interaction_features.columns),
                    'final_total': len(final_features.columns)
                },
                'model_results': results,
                'top_features': top_20_features.tolist(),
                'best_model': {
                    'name': best_model[0] if 'best_model' in locals() else None,
                    'performance': best_model[1] if 'best_model' in locals() else None
                }
            }

            with open('results/enhanced_volatility_model_v2.json', 'w') as f:
                json.dump(v2_results, f, indent=2, default=str)

            print(f"\n💾 V2 모델 결과 저장: results/enhanced_volatility_model_v2.json")

    print("=" * 70)

if __name__ == "__main__":
    main()