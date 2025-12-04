#!/usr/bin/env python3
"""
포괄적 모델 검증: 모든 모델을 동일한 데이터와 검증 방법으로 테스트
- 목적: Paper figures의 하드코딩 제거
- 방법: Purged K-Fold CV로 공정한 비교
- 데이터 누출 방지: 완전한 시간적 분리
"""
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import json
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

# Purged K-Fold CV import
import sys
sys.path.append('/root/workspace/src')
from validation.purged_cross_validation import PurgedKFold

def load_spy_data():
    """SPY 데이터 로드 (2015-2024)"""
    print("📊 SPY 데이터 로드 중...")
    spy = yf.download('SPY', start='2015-01-01', end='2024-12-31', progress=False)
    spy['returns'] = spy['Close'].pct_change()

    # VIX 데이터 추가
    try:
        vix = yf.download('^VIX', start='2015-01-01', end='2024-12-31', progress=False)
        spy['vix'] = vix['Close'].reindex(spy.index, method='ffill')
    except:
        spy['vix'] = 20.0

    spy = spy.dropna()
    print(f"✅ 데이터 로드 완료: {len(spy)} 관측치")
    return spy

def create_volatility_features(data):
    """변동성 특성 생성 (시간적 분리 보장)"""
    print("🔧 변동성 특성 생성 중...")

    features = pd.DataFrame(index=data.index)
    returns = data['returns']
    high = data['High']
    low = data['Low']
    prices = data['Close']

    # 1. 기본 변동성 (현재 시점 이전만 사용)
    for window in [5, 10, 20]:
        features[f'volatility_{window}'] = returns.rolling(window).std()
        features[f'realized_vol_{window}'] = features[f'volatility_{window}'] * np.sqrt(252)

    # 2. 지수 가중 변동성
    for span in [5, 10, 20]:
        features[f'ewm_vol_{span}'] = returns.ewm(span=span).std()

    # 3. 래그 특성
    for lag in [1, 2, 3, 5]:
        features[f'vol_lag_{lag}'] = features['volatility_5'].shift(lag)

    # 4. Garman-Klass 변동성
    for window in [5, 10]:
        gk_vol = np.log(high / low) ** 2
        features[f'garman_klass_{window}'] = gk_vol.rolling(window).mean()

    # 5. 일중 변동성
    for window in [5, 10]:
        intraday_range = (high - low) / prices
        features[f'intraday_vol_{window}'] = intraday_range.rolling(window).mean()

    # 6. VIX 특성
    if 'vix' in data.columns:
        vix = data['vix']
        features['vix_level'] = vix
        for window in [5, 20]:
            features[f'vix_ma_{window}'] = vix.rolling(window).mean()
            features[f'vix_std_{window}'] = vix.rolling(window).std()

    # 7. HAR 특성 (realized volatility)
    features['rv_daily'] = features['volatility_5']
    features['rv_weekly'] = returns.rolling(5).std()  # 5일 평균
    features['rv_monthly'] = returns.rolling(22).std()  # 22일 평균

    print(f"✅ 특성 생성 완료: {len(features.columns)}개")
    return features

def create_target_volatility(data, horizon=5):
    """타겟 변동성 생성 (미래 t+1 ~ t+horizon)"""
    print(f"🎯 타겟 변동성 생성 중 (horizon={horizon})...")

    returns = data['returns']
    target = []

    for i in range(len(returns)):
        if i + horizon < len(returns):
            # 미래 수익률로만 계산 (t+1부터 시작)
            future_returns = returns.iloc[i+1:i+1+horizon]
            target.append(future_returns.std())
        else:
            target.append(np.nan)

    print(f"✅ 타겟 생성 완료")
    return pd.Series(target, index=data.index, name='target_vol_5d')

def purged_cv_evaluation(model, X, y, model_name, n_splits=5, pct_embargo=0.01):
    """Purged K-Fold CV로 모델 평가"""
    print(f"\n{'='*60}")
    print(f"평가 중: {model_name}")
    print(f"{'='*60}")

    # NaN 제거
    combined = pd.concat([X, y], axis=1).dropna()
    X_clean = combined[X.columns]
    y_clean = combined[y.name]

    print(f"유효 샘플: {len(X_clean)}")

    if len(X_clean) < 100:
        print("⚠️ 샘플 수 부족")
        return None

    # Purged K-Fold CV
    cv = PurgedKFold(n_splits=n_splits, pct_embargo=pct_embargo)

    cv_scores = []
    fold_num = 1

    for train_idx, test_idx in cv.split(X_clean, y_clean):
        # indices를 정수 위치로 변환
        if hasattr(X_clean.index, 'get_indexer'):
            train_pos = X_clean.index.get_indexer(train_idx)
            test_pos = X_clean.index.get_indexer(test_idx)
        else:
            train_pos = train_idx
            test_pos = test_idx

        X_train, X_test = X_clean.iloc[train_pos], X_clean.iloc[test_pos]
        y_train, y_test = y_clean.iloc[train_pos], y_clean.iloc[test_pos]

        # 스케일링 (트리 기반 모델은 선택적)
        if isinstance(model, (Ridge, Lasso, ElasticNet)):
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train.values
            X_test_scaled = X_test.values

        # 학습 및 평가
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        score = r2_score(y_test, y_pred)
        cv_scores.append(score)
        print(f"  Fold {fold_num}: R² = {score:.4f}")
        fold_num += 1

    mean_r2 = np.mean(cv_scores)
    std_r2 = np.std(cv_scores)

    print(f"\n평균 CV R² = {mean_r2:.4f} ± {std_r2:.4f}")

    return {
        'model_name': model_name,
        'cv_r2_mean': mean_r2,
        'cv_r2_std': std_r2,
        'cv_fold_scores': cv_scores,
        'n_samples': len(X_clean),
        'n_features': len(X_clean.columns)
    }

def walk_forward_test(model, X, y, model_name, train_ratio=0.8):
    """Walk-Forward Test (Out-of-sample)"""
    print(f"\nWalk-Forward Test: {model_name}")

    # NaN 제거
    combined = pd.concat([X, y], axis=1).dropna()
    X_clean = combined[X.columns]
    y_clean = combined[y.name]

    split_idx = int(len(X_clean) * train_ratio)
    X_train, X_test = X_clean.iloc[:split_idx], X_clean.iloc[split_idx:]
    y_train, y_test = y_clean.iloc[:split_idx], y_clean.iloc[split_idx:]

    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # 스케일링
    if isinstance(model, (Ridge, Lasso, ElasticNet)):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train.values
        X_test_scaled = X_test.values

    # 학습 및 평가
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"  Test R² = {r2:.4f}, MAE = {mae:.6f}")

    return {
        'test_r2': r2,
        'test_mse': mse,
        'test_mae': mae,
        'test_rmse': np.sqrt(mse)
    }

def main():
    """메인 검증 함수"""
    print("="*80)
    print("🔍 포괄적 모델 검증 시작")
    print("="*80)
    print("\n목적: Paper figures의 하드코딩 제거")
    print("방법: 동일 데이터 + 동일 검증 방법으로 공정한 비교")
    print("="*80)

    # 1. 데이터 로드
    spy_data = load_spy_data()

    # 2. 특성 생성
    features = create_volatility_features(spy_data)

    # 3. 타겟 생성
    target = create_target_volatility(spy_data, horizon=5)

    # 4. 상위 31개 특성 선택 (상관관계 기반)
    print("\n📊 특성 선택 중...")
    combined_for_selection = pd.concat([features, target], axis=1).dropna()
    correlations = combined_for_selection[features.columns].corrwith(
        combined_for_selection['target_vol_5d']
    ).abs().sort_values(ascending=False)

    top_features = correlations.head(31).index
    X = features[top_features]

    print(f"✅ 선택된 특성: {len(top_features)}개")
    print("상위 10개 특성:")
    for i, (feat, corr) in enumerate(correlations.head(10).items()):
        print(f"  {i+1:2d}. {feat:25}: {corr:.4f}")

    # 5. 모델 정의
    models = {
        'HAR Benchmark': Ridge(alpha=0.01),  # HAR는 3개 특성만 사용
        'Ridge Volatility': Ridge(alpha=1.0),
        'Lasso 0.001': Lasso(alpha=0.001, max_iter=3000, random_state=42),
        'ElasticNet': ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=3000, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    }

    # 6. 모든 모델 평가
    results = {}

    for model_name, model in models.items():
        # HAR는 3개 특성만 사용
        if model_name == 'HAR Benchmark':
            har_features = ['rv_daily', 'rv_weekly', 'rv_monthly']
            X_model = X[har_features] if all(f in X.columns for f in har_features) else X.iloc[:, :3]
        else:
            X_model = X

        # Purged K-Fold CV
        cv_result = purged_cv_evaluation(model, X_model, target, model_name)

        if cv_result is not None:
            # Walk-Forward Test
            wf_result = walk_forward_test(model, X_model, target, model_name)

            # 결과 병합
            results[model_name] = {
                **cv_result,
                **wf_result
            }

    # 7. 결과 저장
    output_dir = Path('/root/workspace/data/validation')
    output_dir.mkdir(exist_ok=True)

    validation_summary = {
        'timestamp': datetime.now().isoformat(),
        'data_source': 'SPY (2015-2024)',
        'validation_method': 'Purged K-Fold CV (5-fold, embargo=1%)',
        'target': 'target_vol_5d (5-day future volatility)',
        'n_features_total': len(features.columns),
        'n_features_selected': len(top_features),
        'models': results
    }

    output_file = output_dir / 'comprehensive_model_validation.json'
    with open(output_file, 'w') as f:
        json.dump(validation_summary, f, indent=2, default=str)

    print("\n" + "="*80)
    print("✅ 검증 완료 - 결과 요약")
    print("="*80)
    print(f"\n{'Model':<20} {'CV R²':<15} {'Test R²':<15}")
    print("-"*50)

    for model_name, result in results.items():
        cv_r2 = result['cv_r2_mean']
        test_r2 = result.get('test_r2', np.nan)
        print(f"{model_name:<20} {cv_r2:>7.4f} ± {result['cv_r2_std']:.4f}  {test_r2:>7.4f}")

    print("\n" + "="*80)
    print(f"💾 결과 저장: {output_file}")
    print("="*80)

    return validation_summary

if __name__ == "__main__":
    results = main()
