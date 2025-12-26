"""
SCI 논문 퀄리티 향상 실험
=========================
1. Robustness Check (시장 조건별)
2. 경제적 유의성 (트레이딩 시뮬레이션)
3. HAR-RV 모델 비교
4. Subsample 분석 (Pre/Post COVID)
5. 예측 호라이즌 비교 (5/10/22일)
"""
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import Ridge, Lasso, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def calculate_rv(returns, window):
    rv = (returns ** 2).rolling(window).sum() * 252
    return rv.iloc[:, 0] if isinstance(rv, pd.DataFrame) else rv

def prepare_features(ticker, start='2015-01-01', end='2025-01-01'):
    """하이브리드 특성"""
    data = yf.download(ticker, start=start, end=end, progress=False)
    if len(data) < 300:
        return None
    
    returns = data['Close'].pct_change()
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    
    features = pd.DataFrame(index=data.index)
    
    rv_5d = calculate_rv(returns, 5)
    rv_22d = calculate_rv(returns, 22)
    
    features['RV_5d_lag1'] = rv_5d.shift(1)
    features['RV_22d_lag1'] = rv_22d.shift(1)
    features['RV_ratio_lag1'] = (rv_5d / rv_22d.clip(lower=1e-8)).shift(1)
    
    vix = yf.download('^VIX', start=start, end=end, progress=False)
    vix_close = vix['Close'].iloc[:, 0] if isinstance(vix['Close'], pd.DataFrame) else vix['Close']
    features['VIX_lag1'] = vix_close.reindex(data.index).ffill().shift(1)
    features['VIX_change_lag1'] = features['VIX_lag1'].pct_change()
    features['direction_5d_lag1'] = returns.rolling(5).apply(lambda x: np.mean(x > 0)).shift(1)
    
    features['RV_5d_future'] = rv_5d.shift(-5)
    features['returns'] = returns
    
    return features.dropna()

# ============================================================================
# 1. Robustness Check (시장 조건별)
# ============================================================================

def robustness_check(ticker):
    """시장 조건별 Robustness Check"""
    print(f"\n[1] Robustness Check: {ticker}")
    
    features = prepare_features(ticker)
    if features is None:
        return None
    
    feature_cols = ['RV_5d_lag1', 'RV_22d_lag1', 'RV_ratio_lag1', 
                    'VIX_lag1', 'VIX_change_lag1', 'direction_5d_lag1']
    X = features[feature_cols]
    y = features['RV_5d_future']
    
    results = {}
    
    # 전체 기간
    gap = 5
    n = len(X)
    train_end = int(n * 0.7) - gap
    
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_test = X.iloc[train_end + gap:]
    y_test = y.iloc[train_end + gap:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    model = Ridge(alpha=100.0)
    model.fit(X_train_s, np.sqrt(y_train))
    pred = np.maximum(model.predict(X_test_s) ** 2, 0)
    
    results['full_period'] = r2_score(y_test, pred)
    
    # VIX 레벨별
    vix = features['VIX_lag1'].iloc[train_end + gap:]
    
    low_vix = vix < 15
    mid_vix = (vix >= 15) & (vix < 25)
    high_vix = vix >= 25
    
    if low_vix.sum() > 30:
        results['low_vix'] = r2_score(y_test[low_vix], pred[low_vix])
    if mid_vix.sum() > 30:
        results['mid_vix'] = r2_score(y_test[mid_vix], pred[mid_vix])
    if high_vix.sum() > 30:
        results['high_vix'] = r2_score(y_test[high_vix], pred[high_vix])
    
    print(f"  전체: R² = {results['full_period']:.4f}")
    print(f"  Low VIX (<15): R² = {results.get('low_vix', 'N/A')}")
    print(f"  Mid VIX (15-25): R² = {results.get('mid_vix', 'N/A')}")
    print(f"  High VIX (>25): R² = {results.get('high_vix', 'N/A')}")
    
    return results

# ============================================================================
# 2. 경제적 유의성 (트레이딩 시뮬레이션)
# ============================================================================

def trading_simulation(ticker):
    """변동성 예측 기반 트레이딩 시뮬레이션"""
    print(f"\n[2] Trading Simulation: {ticker}")
    
    features = prepare_features(ticker)
    if features is None:
        return None
    
    feature_cols = ['RV_5d_lag1', 'RV_22d_lag1', 'RV_ratio_lag1', 
                    'VIX_lag1', 'VIX_change_lag1', 'direction_5d_lag1']
    X = features[feature_cols]
    y = features['RV_5d_future']
    returns = features['returns'].iloc[5:]  # shift로 인한 조정
    
    gap = 5
    n = len(X)
    train_end = int(n * 0.7) - gap
    
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_test = X.iloc[train_end + gap:]
    y_test = y.iloc[train_end + gap:]
    test_returns = returns.iloc[train_end + gap:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    model = Ridge(alpha=100.0)
    model.fit(X_train_s, np.sqrt(y_train))
    pred = np.maximum(model.predict(X_test_s) ** 2, 0)
    
    # 전략: 고변동성 예측 시 포지션 축소
    current_rv = X_test['RV_5d_lag1'].values
    pred_change = (pred - current_rv) / (current_rv + 1e-8)
    
    # 변동성 상승 예측 시 50% 포지션, 하락 시 100%
    position = np.where(pred_change > 0.1, 0.5, 1.0)
    
    # 수익률 계산 (5일 홀딩)
    strategy_returns = []
    buy_hold_returns = []
    
    for i in range(0, len(test_returns) - 5, 5):
        period_return = test_returns.iloc[i:i+5].sum()
        strategy_returns.append(position[i] * period_return)
        buy_hold_returns.append(period_return)
    
    strategy_returns = np.array(strategy_returns)
    buy_hold_returns = np.array(buy_hold_returns)
    
    # 성과 지표
    strategy_total = np.prod(1 + strategy_returns) - 1
    buy_hold_total = np.prod(1 + buy_hold_returns) - 1
    
    strategy_sharpe = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8) * np.sqrt(52)
    buy_hold_sharpe = np.mean(buy_hold_returns) / (np.std(buy_hold_returns) + 1e-8) * np.sqrt(52)
    
    strategy_mdd = np.min(np.cumprod(1 + strategy_returns) / np.maximum.accumulate(np.cumprod(1 + strategy_returns)) - 1)
    buy_hold_mdd = np.min(np.cumprod(1 + buy_hold_returns) / np.maximum.accumulate(np.cumprod(1 + buy_hold_returns)) - 1)
    
    results = {
        'strategy_return': strategy_total,
        'buy_hold_return': buy_hold_total,
        'strategy_sharpe': strategy_sharpe,
        'buy_hold_sharpe': buy_hold_sharpe,
        'strategy_mdd': strategy_mdd,
        'buy_hold_mdd': buy_hold_mdd,
        'excess_return': strategy_total - buy_hold_total,
        'excess_sharpe': strategy_sharpe - buy_hold_sharpe
    }
    
    print(f"  전략 수익률: {strategy_total:.2%} (Sharpe: {strategy_sharpe:.2f})")
    print(f"  Buy&Hold 수익률: {buy_hold_total:.2%} (Sharpe: {buy_hold_sharpe:.2f})")
    print(f"  초과 수익률: {strategy_total - buy_hold_total:.2%}")
    print(f"  MDD 개선: {buy_hold_mdd:.2%} → {strategy_mdd:.2%}")
    
    return results

# ============================================================================
# 3. HAR-RV 모델 비교
# ============================================================================

def har_rv_comparison(ticker):
    """HAR-RV 모델과 비교"""
    print(f"\n[3] HAR-RV Model Comparison: {ticker}")
    
    features = prepare_features(ticker)
    if features is None:
        return None
    
    # HAR-RV 특성 추가
    returns = yf.download(ticker, start='2015-01-01', end='2025-01-01', progress=False)['Close'].pct_change()
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    
    rv_1d = (returns ** 2) * 252
    rv_5d = rv_1d.rolling(5).mean()
    rv_22d = rv_1d.rolling(22).mean()
    
    har_features = pd.DataFrame(index=features.index)
    har_features['RV_1d_lag1'] = rv_1d.shift(1).reindex(features.index)
    har_features['RV_5d_lag1'] = rv_5d.shift(1).reindex(features.index)
    har_features['RV_22d_lag1'] = rv_22d.shift(1).reindex(features.index)
    har_features = har_features.dropna()
    
    y = features['RV_5d_future'].reindex(har_features.index)
    
    gap = 5
    n = len(har_features)
    train_end = int(n * 0.7) - gap
    
    # HAR-RV 모델
    X_har_train = har_features.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_har_test = har_features.iloc[train_end + gap:]
    y_test = y.iloc[train_end + gap:]
    
    scaler_har = StandardScaler()
    X_har_train_s = scaler_har.fit_transform(X_har_train)
    X_har_test_s = scaler_har.transform(X_har_test)
    
    model_har = Ridge(alpha=100.0)
    model_har.fit(X_har_train_s, y_train)
    pred_har = np.maximum(model_har.predict(X_har_test_s), 0)
    
    r2_har = r2_score(y_test, pred_har)
    
    # 우리 모델 (Hybrid)
    feature_cols = ['RV_5d_lag1', 'RV_22d_lag1', 'RV_ratio_lag1', 
                    'VIX_lag1', 'VIX_change_lag1', 'direction_5d_lag1']
    X_hybrid = features[feature_cols].reindex(har_features.index)
    
    X_hybrid_train = X_hybrid.iloc[:train_end]
    X_hybrid_test = X_hybrid.iloc[train_end + gap:]
    
    scaler_hybrid = StandardScaler()
    X_hybrid_train_s = scaler_hybrid.fit_transform(X_hybrid_train)
    X_hybrid_test_s = scaler_hybrid.transform(X_hybrid_test)
    
    model_hybrid = Ridge(alpha=100.0)
    model_hybrid.fit(X_hybrid_train_s, np.sqrt(y_train))
    pred_hybrid = np.maximum(model_hybrid.predict(X_hybrid_test_s) ** 2, 0)
    
    r2_hybrid = r2_score(y_test, pred_hybrid)
    
    # Persistence
    r2_persist = r2_score(y_test, X_har_test['RV_5d_lag1'].values)
    
    results = {
        'HAR_RV': r2_har,
        'Hybrid': r2_hybrid,
        'Persistence': r2_persist,
        'Improvement_vs_HAR': r2_hybrid - r2_har
    }
    
    print(f"  HAR-RV R²: {r2_har:.4f}")
    print(f"  Hybrid R²: {r2_hybrid:.4f}")
    print(f"  Persistence R²: {r2_persist:.4f}")
    print(f"  HAR-RV 대비 개선: {r2_hybrid - r2_har:.4f}")
    
    return results

# ============================================================================
# 4. Subsample 분석 (Pre/Post COVID)
# ============================================================================

def subsample_analysis(ticker):
    """Subsample 분석"""
    print(f"\n[4] Subsample Analysis: {ticker}")
    
    results = {}
    
    # Pre-COVID (2015-2019)
    features_pre = prepare_features(ticker, start='2015-01-01', end='2020-01-01')
    if features_pre is not None and len(features_pre) > 500:
        feature_cols = ['RV_5d_lag1', 'RV_22d_lag1', 'RV_ratio_lag1', 
                        'VIX_lag1', 'VIX_change_lag1', 'direction_5d_lag1']
        X = features_pre[feature_cols]
        y = features_pre['RV_5d_future']
        
        gap = 5
        n = len(X)
        train_end = int(n * 0.7) - gap
        
        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_test = X.iloc[train_end + gap:]
        y_test = y.iloc[train_end + gap:]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        model = Ridge(alpha=100.0)
        model.fit(X_train_s, np.sqrt(y_train))
        pred = np.maximum(model.predict(X_test_s) ** 2, 0)
        
        results['pre_covid'] = r2_score(y_test, pred)
    
    # Post-COVID (2021-2024)
    features_post = prepare_features(ticker, start='2021-01-01', end='2025-01-01')
    if features_post is not None and len(features_post) > 500:
        feature_cols = ['RV_5d_lag1', 'RV_22d_lag1', 'RV_ratio_lag1', 
                        'VIX_lag1', 'VIX_change_lag1', 'direction_5d_lag1']
        X = features_post[feature_cols]
        y = features_post['RV_5d_future']
        
        gap = 5
        n = len(X)
        train_end = int(n * 0.7) - gap
        
        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_test = X.iloc[train_end + gap:]
        y_test = y.iloc[train_end + gap:]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        model = Ridge(alpha=100.0)
        model.fit(X_train_s, np.sqrt(y_train))
        pred = np.maximum(model.predict(X_test_s) ** 2, 0)
        
        results['post_covid'] = r2_score(y_test, pred)
    
    print(f"  Pre-COVID (2015-2019): R² = {results.get('pre_covid', 'N/A')}")
    print(f"  Post-COVID (2021-2024): R² = {results.get('post_covid', 'N/A')}")
    
    return results

# ============================================================================
# 5. 예측 호라이즌 비교
# ============================================================================

def horizon_comparison(ticker):
    """예측 호라이즌 비교 (5/10/22일)"""
    print(f"\n[5] Horizon Comparison: {ticker}")
    
    data = yf.download(ticker, start='2015-01-01', end='2025-01-01', progress=False)
    returns = data['Close'].pct_change()
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    
    results = {}
    
    for horizon in [5, 10, 22]:
        rv = calculate_rv(returns, horizon)
        
        features = pd.DataFrame(index=data.index)
        features['RV_lag1'] = rv.shift(1)
        features['RV_22d_lag1'] = calculate_rv(returns, 22).shift(1)
        
        vix = yf.download('^VIX', start='2015-01-01', end='2025-01-01', progress=False)
        vix_close = vix['Close'].iloc[:, 0] if isinstance(vix['Close'], pd.DataFrame) else vix['Close']
        features['VIX_lag1'] = vix_close.reindex(data.index).ffill().shift(1)
        
        features['RV_future'] = rv.shift(-horizon)
        features = features.dropna()
        
        feature_cols = ['RV_lag1', 'RV_22d_lag1', 'VIX_lag1']
        X = features[feature_cols]
        y = features['RV_future']
        
        gap = horizon
        n = len(X)
        train_end = int(n * 0.7) - gap
        
        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_test = X.iloc[train_end + gap:]
        y_test = y.iloc[train_end + gap:]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        model = Ridge(alpha=100.0)
        model.fit(X_train_s, np.sqrt(y_train))
        pred = np.maximum(model.predict(X_test_s) ** 2, 0)
        
        results[f'{horizon}d'] = r2_score(y_test, pred)
        print(f"  {horizon}일 예측: R² = {results[f'{horizon}d']:.4f}")
    
    return results

# ============================================================================
# 메인
# ============================================================================

def main():
    print("="*80)
    print("SCI 논문 퀄리티 향상 실험")
    print("="*80)
    
    assets = ['SPY', 'QQQ', 'XLK', 'XLF']
    
    all_results = {}
    
    for ticker in assets:
        print(f"\n{'='*60}")
        print(f"Asset: {ticker}")
        print(f"{'='*60}")
        
        all_results[ticker] = {}
        
        # 1. Robustness Check
        r = robustness_check(ticker)
        if r:
            all_results[ticker]['robustness'] = r
        
        # 2. Trading Simulation
        r = trading_simulation(ticker)
        if r:
            all_results[ticker]['trading'] = r
        
        # 3. HAR-RV 비교
        r = har_rv_comparison(ticker)
        if r:
            all_results[ticker]['har_comparison'] = r
        
        # 4. Subsample 분석
        r = subsample_analysis(ticker)
        if r:
            all_results[ticker]['subsample'] = r
        
        # 5. 호라이즌 비교
        r = horizon_comparison(ticker)
        if r:
            all_results[ticker]['horizon'] = r
    
    # 결과 저장
    output = {
        'metadata': {
            'experiment': 'SCI Quality Enhancement',
            'timestamp': datetime.now().isoformat()
        },
        'results': all_results
    }
    
    output_path = 'data/results/sci_quality_experiments.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    # 요약
    print("\n" + "="*80)
    print("요약")
    print("="*80)
    
    # HAR-RV 비교 요약
    print("\n[HAR-RV vs Hybrid]")
    for ticker in assets:
        if ticker in all_results and 'har_comparison' in all_results[ticker]:
            h = all_results[ticker]['har_comparison']
            print(f"  {ticker}: HAR={h['HAR_RV']:.4f}, Hybrid={h['Hybrid']:.4f}, 개선={h['Improvement_vs_HAR']:.4f}")
    
    # 트레이딩 요약
    print("\n[Trading Strategy]")
    for ticker in assets:
        if ticker in all_results and 'trading' in all_results[ticker]:
            t = all_results[ticker]['trading']
            print(f"  {ticker}: 초과수익={t['excess_return']:.2%}, Sharpe개선={t['excess_sharpe']:.2f}")
    
    print(f"\n결과 저장: {output_path}")

if __name__ == "__main__":
    main()
