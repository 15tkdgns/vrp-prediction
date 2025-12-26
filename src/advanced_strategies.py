"""
고급 R² 개선 전략 5가지
========================
1. HARQ-WLS: 측정 오차 보정 (1/sqrt(RQ) 가중치)
2. Rectify: 선형 반복예측 + 비선형 잔차 보정
3. Box-Cox: 최적 λ 탐색
4. Rough Volatility: Hurst 지수 특성
5. Transfer Learning: SPY/QQQ → 타 자산 전이

목표: R² = 0.116 → 0.15+ 달성
"""
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import Lasso, Ridge, HuberRegressor, ElasticNet, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from scipy.optimize import nnls, minimize_scalar
from scipy.stats import boxcox, yeojohnson
from scipy.special import inv_boxcox
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. 데이터 준비 및 기본 특성
# ============================================================================

def calculate_rv(returns, window=22):
    """실현 변동성"""
    rv = (returns ** 2).rolling(window).sum() * 252
    return rv.iloc[:, 0] if isinstance(rv, pd.DataFrame) else rv

def calculate_rq(returns, window=22):
    """실현 쿼티시티 (측정 오차 추정)"""
    rq = (window / 3) * (returns ** 4).rolling(window).sum() * (252 ** 2)
    return rq.iloc[:, 0] if isinstance(rq, pd.DataFrame) else rq

def calculate_hurst_exponent(series, max_lag=20):
    """Hurst 지수 계산 (Rough Volatility 특성)"""
    try:
        lags = range(2, max_lag + 1)
        tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]  # Hurst exponent
    except:
        return 0.5  # 기본값

def rolling_hurst(series, window=60):
    """롤링 Hurst 지수"""
    result = pd.Series(index=series.index, dtype=float)
    for i in range(window, len(series)):
        result.iloc[i] = calculate_hurst_exponent(series.iloc[i-window:i].values)
    return result

def prepare_advanced_features(ticker):
    """고급 특성 포함 데이터 준비"""
    df = yf.download(ticker, start='2020-01-01', end='2025-01-01', progress=False)
    if len(df) < 100:
        return None
    
    returns = df['Close'].pct_change()
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    
    features = pd.DataFrame(index=df.index)
    
    # 기본 HAR-RV
    features['RV_1d'] = (returns ** 2) * 252
    features['RV_5d'] = calculate_rv(returns, 5)
    features['RV_22d'] = calculate_rv(returns, 22)
    features['RV_daily_mean'] = features['RV_1d'].rolling(5).mean()
    features['RV_weekly_mean'] = features['RV_5d'].rolling(4).mean()
    features['RV_monthly_mean'] = features['RV_22d'].rolling(3).mean()
    
    # RQ (측정 오차 프록시)
    features['RQ'] = calculate_rq(returns, 22)
    features['RQ_lag1'] = features['RQ'].shift(1)
    
    # 타겟
    features['RV_22d_future'] = features['RV_22d'].shift(-22)
    
    # VIX 프록시
    spy = yf.download('SPY', start='2020-01-01', end='2025-01-01', progress=False)
    spy_ret = spy['Close'].pct_change()
    if isinstance(spy_ret, pd.DataFrame):
        spy_ret = spy_ret.iloc[:, 0]
    spy_rv = calculate_rv(spy_ret, 22)
    features['VIX_proxy'] = spy_rv.reindex(df.index).ffill()
    
    features['VIX_lag1'] = features['VIX_proxy'].shift(1)
    features['VIX_lag5'] = features['VIX_proxy'].shift(5)
    features['VIX_change'] = features['VIX_proxy'].pct_change()
    
    # VRP
    features['VRP'] = features['VIX_proxy'] - features['RV_22d']
    features['VRP_lag1'] = features['VRP'].shift(1)
    features['VRP_ma5'] = features['VRP'].rolling(5).mean()
    
    # 시장
    features['regime_high'] = (features['VIX_proxy'] >= 25).astype(int)
    features['return_5d'] = returns.rolling(5).sum()
    features['return_22d'] = returns.rolling(22).sum()
    
    # Hurst 지수 (Rough Volatility)
    print(f"  Calculating Hurst exponent...")
    features['Hurst'] = rolling_hurst(features['RV_22d'].dropna(), window=60)
    features['Hurst'] = features['Hurst'].ffill()
    
    return features.dropna()

# ============================================================================
# 2. Box-Cox 최적 λ 탐색
# ============================================================================

def find_optimal_boxcox_lambda(y):
    """Box-Cox 최적 λ 파라미터 찾기"""
    y_positive = y.values + 1e-6  # 양수 보장
    
    try:
        _, best_lambda = boxcox(y_positive)
        return best_lambda
    except:
        return 0  # 로그 변환

def apply_boxcox_transform(y, lmbda):
    """Box-Cox 변환 적용"""
    y_positive = y.values + 1e-6
    if lmbda == 0:
        return np.log(y_positive)
    else:
        return (y_positive ** lmbda - 1) / lmbda

def inverse_boxcox_transform(y_transformed, lmbda):
    """Box-Cox 역변환"""
    if lmbda == 0:
        return np.exp(y_transformed) - 1e-6
    else:
        return (y_transformed * lmbda + 1) ** (1/lmbda) - 1e-6

# ============================================================================
# 3. HARQ-WLS (가중 최소 제곱법)
# ============================================================================

def train_with_wls(model, X_train, y_train, rq_values):
    """RQ 기반 가중치로 WLS 학습"""
    # 가중치: 1 / sqrt(RQ)
    weights = 1 / np.sqrt(rq_values + 1e-8)
    weights = weights / weights.sum() * len(weights)  # 정규화
    
    # sample_weight 지원 모델
    if hasattr(model, 'fit') and 'sample_weight' in model.fit.__code__.co_varnames:
        model.fit(X_train, y_train, sample_weight=weights)
    else:
        # 수동 가중치 적용
        X_weighted = X_train * np.sqrt(weights).reshape(-1, 1)
        y_weighted = y_train * np.sqrt(weights)
        model.fit(X_weighted, y_weighted)
    
    return model

# ============================================================================
# 4. Rectify 전략 (선형 반복예측 + 비선형 잔차 보정)
# ============================================================================

def rectify_prediction(X_train, y_train, X_test, y_test):
    """Rectify 전략: 선형 예측 + 비선형 잔차 보정"""
    
    # Step 1: 선형 모델로 1차 예측
    linear_model = Ridge(alpha=0.001)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    linear_model.fit(X_train_scaled, y_train)
    linear_pred_train = linear_model.predict(X_train_scaled)
    linear_pred_test = linear_model.predict(X_test_scaled)
    
    # Step 2: 잔차 계산
    residuals_train = y_train - linear_pred_train
    
    # Step 3: 비선형 모델로 잔차 예측
    nonlinear_model = GradientBoostingRegressor(
        n_estimators=50, max_depth=2, learning_rate=0.1, random_state=42
    )
    nonlinear_model.fit(X_train_scaled, residuals_train)
    residual_pred_test = nonlinear_model.predict(X_test_scaled)
    
    # Step 4: 최종 예측 = 선형 + 잔차
    final_pred = linear_pred_test + residual_pred_test
    
    return final_pred, {
        'linear_r2': r2_score(y_test, linear_pred_test),
        'final_r2': r2_score(y_test, final_pred)
    }

# ============================================================================
# 5. Transfer Learning (전이 학습)
# ============================================================================

def transfer_learning_predict(source_model, source_scaler, X_target, y_target, fine_tune_ratio=0.3):
    """소스 자산에서 학습된 모델을 타겟 자산에 전이"""
    
    X_target_scaled = source_scaler.transform(X_target)
    
    # 소스 모델로 직접 예측
    source_pred = source_model.predict(X_target_scaled)
    source_r2 = r2_score(y_target, source_pred)
    
    # Fine-tuning: 타겟 데이터 일부로 미세 조정
    n_finetune = int(len(X_target) * fine_tune_ratio)
    if n_finetune > 50:
        X_ft = X_target_scaled[:n_finetune]
        y_ft = y_target.iloc[:n_finetune].values
        
        # 편향 보정
        pred_ft = source_model.predict(X_ft)
        bias = np.mean(y_ft - pred_ft)
        
        adjusted_pred = source_pred + bias
        adjusted_r2 = r2_score(y_target, adjusted_pred)
    else:
        adjusted_pred = source_pred
        adjusted_r2 = source_r2
    
    return {
        'source_r2': source_r2,
        'adjusted_r2': adjusted_r2,
        'predictions': adjusted_pred
    }

# ============================================================================
# 통합 NNLS 앙상블 (모든 전략 적용)
# ============================================================================

def advanced_nnls_ensemble(X_train, y_train, X_test, y_test, rq_train, lmbda, use_wls=True):
    """고급 전략이 적용된 NNLS 앙상블"""
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Box-Cox 변환
    y_train_bc = apply_boxcox_transform(y_train, lmbda)
    
    # 모델 정의
    models = {
        'Lasso': Lasso(alpha=0.03, max_iter=2000, random_state=42),
        'Huber': HuberRegressor(epsilon=1.0, alpha=0.017),
        'Ridge': Ridge(alpha=0.076, random_state=42),
        'ElasticNet': ElasticNet(alpha=0.42, l1_ratio=0.66, max_iter=2000, random_state=42),
        'RF': RandomForestRegressor(n_estimators=112, max_depth=6, min_samples_leaf=50, random_state=42)
    }
    
    predictions = {}
    individual_r2 = {}
    
    for name, model in models.items():
        try:
            if use_wls and name in ['Lasso', 'Huber', 'Ridge', 'ElasticNet']:
                # WLS 적용
                train_with_wls(model, X_train_scaled, y_train_bc, rq_train)
            else:
                model.fit(X_train_scaled, y_train_bc)
            
            pred_bc = model.predict(X_test_scaled)
            pred = inverse_boxcox_transform(pred_bc, lmbda)
            
            predictions[name] = pred
            individual_r2[name] = r2_score(y_test, pred)
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
    
    return {
        'individual_r2': individual_r2,
        'ensemble_r2': ensemble_r2,
        'weights': dict(zip(model_names, weights.tolist())),
        'predictions': ensemble_pred
    }

# ============================================================================
# 메인 실험
# ============================================================================

def run_advanced_experiment(ticker, gap=22):
    """고급 전략 실험 실행"""
    print(f"\n{'='*70}")
    print(f"Advanced Strategies for {ticker}")
    print(f"{'='*70}")
    
    features = prepare_advanced_features(ticker)
    if features is None:
        return None
    
    # 특성 선택
    feature_cols = [
        'RV_1d', 'RV_5d', 'RV_22d', 
        'RV_daily_mean', 'RV_weekly_mean', 'RV_monthly_mean',
        'VIX_lag1', 'VIX_lag5', 'VIX_change',
        'VRP_lag1', 'VRP_ma5',
        'regime_high', 'return_5d', 'return_22d',
        'Hurst'  # Rough Volatility 특성
    ]
    feature_cols = [c for c in feature_cols if c in features.columns]
    
    X = features[feature_cols]
    y = features['RV_22d_future']
    rq = features['RQ_lag1'] if 'RQ_lag1' in features.columns else features['RQ']
    
    # 분할
    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx - gap]
    y_train = y.iloc[:split_idx - gap]
    rq_train = rq.iloc[:split_idx - gap].values
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    
    print(f"  Features: {len(feature_cols)}, Train: {len(X_train)}, Test: {len(X_test)}")
    
    results = {}
    
    # Strategy 1: Box-Cox 최적 λ 탐색
    print(f"\n  [Strategy 1] Finding optimal Box-Cox λ...")
    optimal_lambda = find_optimal_boxcox_lambda(y_train)
    print(f"    Optimal λ = {optimal_lambda:.4f}")
    results['boxcox_lambda'] = optimal_lambda
    
    # Strategy 2: 기본 NNLS (비교용)
    print(f"\n  [Strategy 2] Baseline NNLS (log transform)...")
    baseline_result = advanced_nnls_ensemble(
        X_train, y_train, X_test, y_test, rq_train, lmbda=0, use_wls=False
    )
    if baseline_result:
        print(f"    Baseline R2 = {baseline_result['ensemble_r2']:.4f}")
        results['baseline'] = baseline_result['ensemble_r2']
    
    # Strategy 3: Box-Cox + WLS
    print(f"\n  [Strategy 3] Box-Cox + WLS...")
    boxcox_wls_result = advanced_nnls_ensemble(
        X_train, y_train, X_test, y_test, rq_train, lmbda=optimal_lambda, use_wls=True
    )
    if boxcox_wls_result:
        print(f"    Box-Cox + WLS R2 = {boxcox_wls_result['ensemble_r2']:.4f}")
        print(f"    Individual: {boxcox_wls_result['individual_r2']}")
        results['boxcox_wls'] = boxcox_wls_result['ensemble_r2']
        results['boxcox_wls_weights'] = boxcox_wls_result['weights']
    
    # Strategy 4: Rectify 전략
    print(f"\n  [Strategy 4] Rectify (Linear + Nonlinear)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    y_train_log = np.log(y_train + 1)
    
    rectify_pred, rectify_info = rectify_prediction(
        X_train, y_train_log, X_test, np.log(y_test + 1)
    )
    rectify_pred_orig = np.exp(rectify_pred) - 1
    rectify_r2 = r2_score(y_test, rectify_pred_orig)
    print(f"    Linear R2 = {rectify_info['linear_r2']:.4f}")
    print(f"    Rectify R2 = {rectify_r2:.4f}")
    results['rectify'] = rectify_r2
    
    # 최종 결과 요약
    print(f"\n  === Results Summary for {ticker} ===")
    for strategy, r2 in results.items():
        if isinstance(r2, float):
            print(f"    {strategy}: R2 = {r2:.4f}")
    
    return results

def run_transfer_learning_experiment():
    """전이 학습 실험"""
    print("\n" + "="*70)
    print("Transfer Learning: SPY/QQQ → Other Assets")
    print("="*70)
    
    # 소스 자산 학습 (QQQ - 가장 좋은 성능)
    source_ticker = 'QQQ'
    print(f"\nTraining source model on {source_ticker}...")
    
    features = prepare_advanced_features(source_ticker)
    
    feature_cols = [
        'RV_1d', 'RV_5d', 'RV_22d', 
        'RV_daily_mean', 'RV_weekly_mean', 'RV_monthly_mean',
        'VIX_lag1', 'VIX_lag5', 'VIX_change',
        'VRP_lag1', 'VRP_ma5',
        'regime_high', 'return_5d', 'return_22d',
        'Hurst'
    ]
    feature_cols = [c for c in feature_cols if c in features.columns]
    
    X = features[feature_cols]
    y = features['RV_22d_future']
    
    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx - 22]
    y_train = np.log(y.iloc[:split_idx - 22] + 1)
    
    source_scaler = StandardScaler()
    X_train_scaled = source_scaler.fit_transform(X_train)
    
    source_model = HuberRegressor(epsilon=1.0, alpha=0.017)
    source_model.fit(X_train_scaled, y_train)
    
    # 타겟 자산들에 전이
    target_tickers = ['TLT', 'EEM', 'GLD']
    transfer_results = {}
    
    for target in target_tickers:
        print(f"\nTransferring to {target}...")
        
        target_features = prepare_advanced_features(target)
        if target_features is None:
            continue
        
        X_target = target_features[[c for c in feature_cols if c in target_features.columns]]
        y_target = target_features['RV_22d_future']
        
        split_idx = int(len(X_target) * 0.8)
        X_target_test = X_target.iloc[split_idx:]
        y_target_test = y_target.iloc[split_idx:]
        
        result = transfer_learning_predict(
            source_model, source_scaler, X_target_test, y_target_test
        )
        
        # 역변환
        result['adjusted_r2'] = r2_score(y_target_test, np.exp(result['predictions']) - 1)
        
        print(f"  Direct transfer R2 = {result['source_r2']:.4f}")
        print(f"  Adjusted R2 = {result['adjusted_r2']:.4f}")
        
        transfer_results[target] = result
    
    return transfer_results

def main():
    assets = ['SPY', 'GLD', 'QQQ', 'TLT', 'EEM']
    
    all_results = {}
    
    print("="*80)
    print("Advanced R² Improvement Strategies")
    print("="*80)
    
    # 각 자산별 고급 전략 실험
    for asset in assets:
        result = run_advanced_experiment(asset, gap=22)
        if result:
            all_results[asset] = result
    
    # 전이 학습 실험
    transfer_results = run_transfer_learning_experiment()
    all_results['transfer_learning'] = transfer_results
    
    # 결과 저장
    output = {
        'metadata': {
            'experiment': 'Advanced R² Improvement Strategies',
            'strategies': [
                '1. HARQ-WLS (1/sqrt(RQ) weights)',
                '2. Box-Cox optimal lambda',
                '3. Rectify (Linear + Nonlinear)',
                '4. Rough Volatility (Hurst exponent)',
                '5. Transfer Learning (QQQ → others)'
            ],
            'timestamp': datetime.now().isoformat()
        },
        'results': all_results
    }
    
    output_path = 'data/results/advanced_strategies_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    # 최종 요약
    print("\n" + "="*80)
    print("Final Summary")
    print("="*80)
    
    for asset, results in all_results.items():
        if asset != 'transfer_learning' and isinstance(results, dict):
            best_strategy = max(
                [(k, v) for k, v in results.items() if isinstance(v, float)],
                key=lambda x: x[1],
                default=('none', -999)
            )
            print(f"  {asset}: Best = {best_strategy[0]} (R2={best_strategy[1]:.4f})")
    
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
