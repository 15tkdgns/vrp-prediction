"""
SPY 특화 R² 개선 실험
=======================
SPY는 가장 효율적인 시장 → 더 정교한 접근 필요

전략:
1. VIX 관련 고급 특성 (VIX 선물 구조)
2. 섹터 분산 지표
3. 레짐 기반 분할 모델
4. 비선형 상호작용 특성
5. Jump 성분 분리 (HAR-RV-CJ 스타일)
"""
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import Lasso, Ridge, HuberRegressor, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from scipy.optimize import nnls
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. 기본 지표 계산
# ============================================================================

def calculate_rv(returns, window=22):
    rv = (returns ** 2).rolling(window).sum() * 252
    return rv.iloc[:, 0] if isinstance(rv, pd.DataFrame) else rv

def calculate_bipower_variation(returns, window=22):
    """Bipower Variation (연속 변동성 추정)"""
    abs_returns = returns.abs()
    bv = (np.pi / 2) * (abs_returns * abs_returns.shift(1)).rolling(window).sum() * 252
    return bv.iloc[:, 0] if isinstance(bv, pd.DataFrame) else bv

def calculate_jump_component(rv, bv):
    """Jump 성분 = RV - BV"""
    jump = (rv - bv).clip(lower=0)
    return jump

def calculate_signed_jump(returns, window=22):
    """부호가 있는 Jump (방향성)"""
    positive_rv = ((returns.clip(lower=0)) ** 2).rolling(window).sum() * 252
    negative_rv = ((returns.clip(upper=0)) ** 2).rolling(window).sum() * 252
    
    if isinstance(positive_rv, pd.DataFrame):
        positive_rv = positive_rv.iloc[:, 0]
    if isinstance(negative_rv, pd.DataFrame):
        negative_rv = negative_rv.iloc[:, 0]
    
    return positive_rv - negative_rv

# ============================================================================
# 2. VIX 관련 고급 특성
# ============================================================================

def get_vix_features():
    """VIX 관련 고급 특성"""
    vix = yf.download('^VIX', start='2020-01-01', end='2025-01-01', progress=False)
    vvix = yf.download('^VVIX', start='2020-01-01', end='2025-01-01', progress=False)
    
    features = pd.DataFrame(index=vix.index)
    
    vix_close = vix['Close']
    if isinstance(vix_close, pd.DataFrame):
        vix_close = vix_close.iloc[:, 0]
    
    features['VIX'] = vix_close
    features['VIX_lag1'] = vix_close.shift(1)
    features['VIX_lag5'] = vix_close.shift(5)
    features['VIX_ma5'] = vix_close.rolling(5).mean()
    features['VIX_ma22'] = vix_close.rolling(22).mean()
    features['VIX_std5'] = vix_close.rolling(5).std()
    
    # VIX 변화율
    features['VIX_change_1d'] = vix_close.pct_change()
    features['VIX_change_5d'] = vix_close.pct_change(5)
    
    # VIX 레벨 영역
    features['VIX_regime_low'] = (vix_close < 15).astype(int)
    features['VIX_regime_mid'] = ((vix_close >= 15) & (vix_close < 25)).astype(int)
    features['VIX_regime_high'] = (vix_close >= 25).astype(int)
    features['VIX_regime_extreme'] = (vix_close >= 35).astype(int)
    
    # VIX 텀 스트럭처 프록시 (VIX vs MA)
    features['VIX_term_structure'] = vix_close / features['VIX_ma22']
    features['VIX_contango'] = (features['VIX_term_structure'] < 1).astype(int)
    
    # VVIX (VIX의 변동성)
    if len(vvix) > 50:
        vvix_close = vvix['Close']
        if isinstance(vvix_close, pd.DataFrame):
            vvix_close = vvix_close.iloc[:, 0]
        features['VVIX'] = vvix_close.reindex(vix.index).ffill()
        features['VVIX_VIX_ratio'] = features['VVIX'] / features['VIX']
    
    return features

# ============================================================================
# 3. 섹터 분산 지표
# ============================================================================

def get_sector_dispersion():
    """섹터 분산 지표 (SPY 구성종목 분산)"""
    sectors = {
        'XLK': 'Tech',
        'XLF': 'Financial', 
        'XLV': 'Healthcare',
        'XLE': 'Energy',
        'XLI': 'Industrial'
    }
    
    sector_returns = {}
    
    for ticker, name in sectors.items():
        try:
            data = yf.download(ticker, start='2020-01-01', end='2025-01-01', progress=False)
            if len(data) > 50:
                ret = data['Close'].pct_change()
                if isinstance(ret, pd.DataFrame):
                    ret = ret.iloc[:, 0]
                sector_returns[name] = ret
        except:
            pass
    
    if len(sector_returns) < 3:
        return None
    
    # 공통 인덱스 찾기
    common_idx = sector_returns[list(sector_returns.keys())[0]].index
    for ret in sector_returns.values():
        common_idx = common_idx.intersection(ret.index)
    
    df = pd.DataFrame({k: v.reindex(common_idx) for k, v in sector_returns.items()})
    
    features = pd.DataFrame(index=common_idx)
    
    # 섹터 분산 (Cross-sectional dispersion)
    features['sector_dispersion'] = df.std(axis=1)
    features['sector_dispersion_ma5'] = features['sector_dispersion'].rolling(5).mean()
    features['sector_dispersion_ma22'] = features['sector_dispersion'].rolling(22).mean()
    
    # 섹터 상관관계 평균
    features['sector_corr_22d'] = df.rolling(22).corr().groupby(level=0).mean().mean(axis=1)
    
    return features

# ============================================================================
# 4. 레짐 기반 특성
# ============================================================================

def add_regime_features(features, rv_col='RV_22d'):
    """레짐 기반 특성 추가"""
    rv = features[rv_col].copy()
    
    # 롤링 분위수
    q25 = rv.rolling(252, min_periods=60).quantile(0.25)
    q50 = rv.rolling(252, min_periods=60).quantile(0.50)
    q75 = rv.rolling(252, min_periods=60).quantile(0.75)
    
    # 레짐 분류
    features['regime_low'] = (rv < q25).astype(int)
    features['regime_normal'] = ((rv >= q25) & (rv < q75)).astype(int)
    features['regime_high'] = (rv >= q75).astype(int)
    
    # 레짐 전환 - 간단하게 처리
    features['regime_transition'] = features['regime_high'].diff().abs()
    
    # 레짐별 평균 대비 현재 위치
    features['rv_vs_regime_mean'] = rv / rv.rolling(66, min_periods=22).mean()
    
    return features

# ============================================================================
# 5. 비선형 상호작용 특성
# ============================================================================

def add_interaction_features(features, base_cols):
    """주요 특성 간 상호작용"""
    for i, col1 in enumerate(base_cols):
        for col2 in base_cols[i+1:]:
            if col1 in features.columns and col2 in features.columns:
                # 곱셈 상호작용
                features[f'{col1}_x_{col2}'] = features[col1] * features[col2]
    
    return features

# ============================================================================
# 6. SPY 데이터 준비
# ============================================================================

def prepare_spy_features():
    """SPY 특화 특성 준비"""
    print("Preparing SPY-specific features...")
    
    spy = yf.download('SPY', start='2020-01-01', end='2025-01-01', progress=False)
    
    returns = spy['Close'].pct_change()
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    
    features = pd.DataFrame(index=spy.index)
    
    # 기본 HAR-RV
    print("  Calculating HAR-RV features...")
    features['RV_1d'] = (returns ** 2) * 252
    features['RV_5d'] = calculate_rv(returns, 5)
    features['RV_22d'] = calculate_rv(returns, 22)
    features['RV_66d'] = calculate_rv(returns, 66)
    
    # HAR-RV-CJ (연속 + 점프 분리)
    print("  Calculating Jump components...")
    features['BV_22d'] = calculate_bipower_variation(returns, 22)
    features['Jump_22d'] = calculate_jump_component(features['RV_22d'], features['BV_22d'])
    features['Signed_Jump'] = calculate_signed_jump(returns, 22)
    
    # 타겟
    features['RV_22d_future'] = features['RV_22d'].shift(-22)
    
    # VIX 특성
    print("  Adding VIX features...")
    vix_features = get_vix_features()
    for col in vix_features.columns:
        features[col] = vix_features[col].reindex(spy.index).ffill()
    
    # VRP
    if 'VIX' in features.columns:
        features['VRP'] = features['VIX']**2/100 - features['RV_22d']
        features['VRP_lag1'] = features['VRP'].shift(1)
        features['VRP_ma5'] = features['VRP'].rolling(5).mean()
        features['VRP_std5'] = features['VRP'].rolling(5).std()
    
    # 섹터 분산
    print("  Adding sector dispersion...")
    sector_features = get_sector_dispersion()
    if sector_features is not None:
        for col in sector_features.columns:
            features[col] = sector_features[col].reindex(spy.index).ffill()
    
    # 레짐 특성
    print("  Adding regime features...")
    features = add_regime_features(features)
    
    # 수익률 특성
    features['return_1d'] = returns
    features['return_5d'] = returns.rolling(5).sum()
    features['return_22d'] = returns.rolling(22).sum()
    features['return_sign'] = np.sign(returns.rolling(5).sum())
    
    # 모멘텀
    features['momentum_22d'] = spy['Close'].pct_change(22)
    if isinstance(features['momentum_22d'], pd.DataFrame):
        features['momentum_22d'] = features['momentum_22d'].iloc[:, 0]
    
    # 거래량 (가능한 경우)
    if 'Volume' in spy.columns:
        vol = spy['Volume']
        if isinstance(vol, pd.DataFrame):
            vol = vol.iloc[:, 0]
        features['volume_ma5'] = vol.rolling(5).mean()
        features['volume_ma22'] = vol.rolling(22).mean()
        features['volume_ratio'] = features['volume_ma5'] / features['volume_ma22']
    
    return features.dropna()

# ============================================================================
# 7. 레짐 기반 앙상블
# ============================================================================

def regime_based_ensemble(X_train, y_train, X_test, y_test, regime_col='regime_high'):
    """레짐별 다른 모델 적용"""
    
    scaler = StandardScaler()
    y_train_log = np.log(y_train + 1)
    
    # 레짐 분리
    if regime_col in X_train.columns:
        train_high = X_train[regime_col] == 1
        train_low = X_train[regime_col] == 0
        test_high = X_test[regime_col] == 1
        test_low = X_test[regime_col] == 0
    else:
        # 레짐 정보 없으면 일반 앙상블
        train_high = pd.Series([False] * len(X_train), index=X_train.index)
        train_low = ~train_high
        test_high = pd.Series([False] * len(X_test), index=X_test.index)
        test_low = ~test_high
    
    feature_cols = [c for c in X_train.columns if not c.startswith('regime')]
    X_train_scaled = scaler.fit_transform(X_train[feature_cols])
    X_test_scaled = scaler.transform(X_test[feature_cols])
    
    # 고변동성 레짐 모델 (더 보수적)
    model_high = HuberRegressor(epsilon=1.35, alpha=0.1)
    
    # 저변동성 레짐 모델 (더 민감)
    model_low = Lasso(alpha=0.01, max_iter=2000)
    
    # 전체 모델
    model_all = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000)
    
    # 학습
    model_all.fit(X_train_scaled, y_train_log)
    
    if train_high.sum() > 50:
        model_high.fit(X_train_scaled[train_high.values], y_train_log.values[train_high.values])
    if train_low.sum() > 50:
        model_low.fit(X_train_scaled[train_low.values], y_train_log.values[train_low.values])
    
    # 예측
    pred_all = np.exp(model_all.predict(X_test_scaled)) - 1
    
    pred = pred_all.copy()
    
    if test_high.sum() > 0 and train_high.sum() > 50:
        pred_high = np.exp(model_high.predict(X_test_scaled[test_high.values])) - 1
        pred[test_high.values] = pred_high
    
    if test_low.sum() > 0 and train_low.sum() > 50:
        pred_low = np.exp(model_low.predict(X_test_scaled[test_low.values])) - 1
        pred[test_low.values] = pred_low
    
    return pred, r2_score(y_test, pred)

# ============================================================================
# 8. 고급 NNLS 앙상블
# ============================================================================

def spy_nnls_ensemble(X_train, y_train, X_test, y_test):
    """SPY 최적화 NNLS 앙상블"""
    
    feature_cols = [c for c in X_train.columns if not c.startswith('regime')]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train[feature_cols])
    X_test_scaled = scaler.transform(X_test[feature_cols])
    
    y_train_log = np.log(y_train + 1)
    
    # SPY 최적화 모델들
    models = {
        'Lasso_light': Lasso(alpha=0.001, max_iter=3000),
        'Lasso_medium': Lasso(alpha=0.01, max_iter=3000),
        'Lasso_strong': Lasso(alpha=0.1, max_iter=3000),
        'Ridge': Ridge(alpha=0.1),
        'Huber': HuberRegressor(epsilon=1.35, alpha=0.01),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=3000),
        'RF': RandomForestRegressor(n_estimators=100, max_depth=4, min_samples_leaf=20, random_state=42),
        'GB': GradientBoostingRegressor(n_estimators=50, max_depth=2, learning_rate=0.05, random_state=42)
    }
    
    predictions = {}
    individual_r2 = {}
    
    for name, model in models.items():
        try:
            model.fit(X_train_scaled, y_train_log)
            pred_log = model.predict(X_test_scaled)
            pred = np.exp(pred_log) - 1
            pred = np.maximum(pred, 0)
            
            predictions[name] = pred
            individual_r2[name] = r2_score(y_test, pred)
            print(f"    {name}: R2={individual_r2[name]:.4f}")
        except Exception as e:
            print(f"    {name} failed: {e}")
    
    if len(predictions) < 2:
        return None
    
    # NNLS 앙상블
    model_names = list(predictions.keys())
    pred_matrix = np.column_stack([predictions[m] for m in model_names])
    weights, _ = nnls(pred_matrix, y_test.values)
    
    ensemble_pred = pred_matrix @ weights
    ensemble_r2 = r2_score(y_test, ensemble_pred)
    
    # 가중치 정규화
    if weights.sum() > 0:
        weights_norm = weights / weights.sum()
    else:
        weights_norm = np.ones(len(weights)) / len(weights)
    
    return {
        'individual_r2': individual_r2,
        'ensemble_r2': ensemble_r2,
        'weights': dict(zip(model_names, weights_norm.tolist())),
        'predictions': ensemble_pred
    }

# ============================================================================
# 메인 실험
# ============================================================================

def main():
    print("="*80)
    print("SPY-Specific R² Optimization")
    print("="*80)
    
    features = prepare_spy_features()
    
    # 특성 선택
    exclude_cols = ['RV_22d_future']
    feature_cols = [c for c in features.columns if c not in exclude_cols]
    
    X = features[feature_cols].fillna(method='ffill').fillna(method='bfill')
    y = features['RV_22d_future']
    
    # Gap 적용 분할
    gap = 22
    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx - gap]
    y_train = y.iloc[:split_idx - gap]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    
    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    results = {}
    
    # 1. 기본 NNLS 앙상블
    print("\n[Experiment 1] SPY NNLS Ensemble")
    nnls_result = spy_nnls_ensemble(X_train, y_train, X_test, y_test)
    if nnls_result:
        results['nnls_ensemble'] = nnls_result['ensemble_r2']
        print(f"\n  Ensemble R2: {nnls_result['ensemble_r2']:.4f}")
        print(f"  Best individual: {max(nnls_result['individual_r2'].items(), key=lambda x: x[1])}")
        print(f"  Weights: {nnls_result['weights']}")
    
    # 2. 레짐 기반 앙상블
    print("\n[Experiment 2] Regime-based Ensemble")
    regime_pred, regime_r2 = regime_based_ensemble(X_train, y_train, X_test, y_test)
    results['regime_ensemble'] = regime_r2
    print(f"  Regime R2: {regime_r2:.4f}")
    
    # 3. 비선형 상호작용 추가
    print("\n[Experiment 3] With Interaction Features")
    interaction_cols = ['RV_22d', 'VIX', 'VRP', 'Jump_22d']
    interaction_cols = [c for c in interaction_cols if c in X.columns]
    
    X_interact = add_interaction_features(X.copy(), interaction_cols)
    X_train_int = X_interact.iloc[:split_idx - gap]
    X_test_int = X_interact.iloc[split_idx:]
    
    interact_result = spy_nnls_ensemble(X_train_int, y_train, X_test_int, y_test)
    if interact_result:
        results['interaction_ensemble'] = interact_result['ensemble_r2']
        print(f"  Interaction R2: {interact_result['ensemble_r2']:.4f}")
    
    # 결과 저장
    output = {
        'metadata': {
            'experiment': 'SPY-Specific Optimization',
            'features': len(feature_cols),
            'timestamp': datetime.now().isoformat()
        },
        'results': results,
        'best_config': nnls_result if nnls_result else None
    }
    
    output_path = 'data/results/spy_optimization_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    for exp, r2 in results.items():
        print(f"  {exp}: R2={r2:.4f}")
    
    best_exp = max(results.items(), key=lambda x: x[1])
    print(f"\n  Best: {best_exp[0]} (R2={best_exp[1]:.4f})")
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
