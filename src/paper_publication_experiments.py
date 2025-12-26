"""
논문 게재용 추가 실험 및 검증
============================
1. Bootstrap 신뢰구간 (R² 안정성)
2. Rolling Window 분석 (시간적 안정성)
3. 하이퍼파라미터 민감도 분석
4. 자산간 상관관계 분석
5. 예측 오차 분포 분석
6. 모델 앙상블 효과
"""
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import Ridge, Lasso, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
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
    
    return features.dropna()

# ============================================================================
# 1. Bootstrap 신뢰구간
# ============================================================================

def bootstrap_confidence_interval(ticker, n_bootstrap=100):
    """Bootstrap으로 R² 신뢰구간 계산"""
    print(f"\n[1] Bootstrap 신뢰구간: {ticker}")
    
    features = prepare_features(ticker)
    if features is None:
        return None
    
    feature_cols = ['RV_5d_lag1', 'RV_22d_lag1', 'RV_ratio_lag1', 
                    'VIX_lag1', 'VIX_change_lag1', 'direction_5d_lag1']
    X = features[feature_cols]
    y = features['RV_5d_future']
    
    gap = 5
    n = len(X)
    train_end = int(n * 0.7) - gap
    
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_test = X.iloc[train_end + gap:]
    y_test = y.iloc[train_end + gap:]
    
    bootstrap_r2s = []
    
    for i in range(n_bootstrap):
        # Bootstrap 샘플링 (train set)
        np.random.seed(i)
        boot_idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
        
        X_boot = X_train.iloc[boot_idx]
        y_boot = y_train.iloc[boot_idx]
        
        scaler = StandardScaler()
        X_boot_s = scaler.fit_transform(X_boot)
        X_test_s = scaler.transform(X_test)
        
        model = Ridge(alpha=100.0)
        model.fit(X_boot_s, np.sqrt(y_boot))
        pred = np.maximum(model.predict(X_test_s) ** 2, 0)
        
        r2 = r2_score(y_test, pred)
        bootstrap_r2s.append(r2)
    
    bootstrap_r2s = np.array(bootstrap_r2s)
    
    ci_lower = np.percentile(bootstrap_r2s, 2.5)
    ci_upper = np.percentile(bootstrap_r2s, 97.5)
    mean_r2 = np.mean(bootstrap_r2s)
    
    print(f"  Mean R²: {mean_r2:.4f}")
    print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  Std: {np.std(bootstrap_r2s):.4f}")
    
    return {
        'mean': mean_r2,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'std': np.std(bootstrap_r2s)
    }

# ============================================================================
# 2. Rolling Window 분석
# ============================================================================

def rolling_window_analysis(ticker, window_size=500, step=50):
    """Rolling Window로 시간 안정성 분석"""
    print(f"\n[2] Rolling Window 분석: {ticker}")
    
    features = prepare_features(ticker)
    if features is None:
        return None
    
    feature_cols = ['RV_5d_lag1', 'RV_22d_lag1', 'RV_ratio_lag1', 
                    'VIX_lag1', 'VIX_change_lag1', 'direction_5d_lag1']
    X = features[feature_cols]
    y = features['RV_5d_future']
    
    gap = 5
    results = []
    
    for start in range(0, len(X) - window_size, step):
        end = start + window_size
        
        train_end = int(window_size * 0.7) - gap
        
        X_window = X.iloc[start:end]
        y_window = y.iloc[start:end]
        
        X_train = X_window.iloc[:train_end]
        y_train = y_window.iloc[:train_end]
        X_test = X_window.iloc[train_end + gap:]
        y_test = y_window.iloc[train_end + gap:]
        
        if len(X_test) < 50:
            continue
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        model = Ridge(alpha=100.0)
        model.fit(X_train_s, np.sqrt(y_train))
        pred = np.maximum(model.predict(X_test_s) ** 2, 0)
        
        r2 = r2_score(y_test, pred)
        
        results.append({
            'start_date': str(X_window.index[0].date()),
            'end_date': str(X_window.index[-1].date()),
            'r2': r2
        })
    
    r2_values = [r['r2'] for r in results]
    
    print(f"  Window Count: {len(results)}")
    print(f"  R² Range: [{min(r2_values):.4f}, {max(r2_values):.4f}]")
    print(f"  R² Mean: {np.mean(r2_values):.4f}")
    print(f"  R² Std: {np.std(r2_values):.4f}")
    
    return {
        'windows': results,
        'mean': np.mean(r2_values),
        'std': np.std(r2_values),
        'min': min(r2_values),
        'max': max(r2_values)
    }

# ============================================================================
# 3. 하이퍼파라미터 민감도 분석
# ============================================================================

def hyperparameter_sensitivity(ticker):
    """Alpha 민감도 분석"""
    print(f"\n[3] 하이퍼파라미터 민감도: {ticker}")
    
    features = prepare_features(ticker)
    if features is None:
        return None
    
    feature_cols = ['RV_5d_lag1', 'RV_22d_lag1', 'RV_ratio_lag1', 
                    'VIX_lag1', 'VIX_change_lag1', 'direction_5d_lag1']
    X = features[feature_cols]
    y = features['RV_5d_future']
    
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
    
    alphas = [0.01, 0.1, 1, 10, 50, 100, 500, 1000, 5000, 10000]
    results = []
    
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(X_train_s, np.sqrt(y_train))
        pred = np.maximum(model.predict(X_test_s) ** 2, 0)
        r2 = r2_score(y_test, pred)
        
        results.append({'alpha': alpha, 'r2': r2})
        print(f"  Alpha={alpha:>6}: R²={r2:.4f}")
    
    best = max(results, key=lambda x: x['r2'])
    print(f"\n  Best: Alpha={best['alpha']}, R²={best['r2']:.4f}")
    
    return results

# ============================================================================
# 4. 자산간 상관관계 분석
# ============================================================================

def cross_asset_correlation():
    """자산간 예측 오차 상관관계"""
    print("\n[4] 자산간 상관관계 분석")
    
    assets = ['SPY', 'QQQ', 'XLK', 'XLF']
    predictions = {}
    
    for ticker in assets:
        features = prepare_features(ticker)
        if features is None:
            continue
        
        feature_cols = ['RV_5d_lag1', 'RV_22d_lag1', 'RV_ratio_lag1', 
                        'VIX_lag1', 'VIX_change_lag1', 'direction_5d_lag1']
        X = features[feature_cols]
        y = features['RV_5d_future']
        
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
        
        # 오차 계산
        error = y_test.values - pred
        predictions[ticker] = pd.Series(error, index=y_test.index)
    
    # 상관관계 행렬
    error_df = pd.DataFrame(predictions)
    correlation = error_df.corr()
    
    print("\n  예측 오차 상관관계:")
    print(correlation.to_string())
    
    return correlation.to_dict()

# ============================================================================
# 5. 예측 오차 분포 분석
# ============================================================================

def error_distribution_analysis(ticker):
    """예측 오차 분포 분석"""
    print(f"\n[5] 예측 오차 분포: {ticker}")
    
    features = prepare_features(ticker)
    if features is None:
        return None
    
    feature_cols = ['RV_5d_lag1', 'RV_22d_lag1', 'RV_ratio_lag1', 
                    'VIX_lag1', 'VIX_change_lag1', 'direction_5d_lag1']
    X = features[feature_cols]
    y = features['RV_5d_future']
    
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
    
    error = y_test.values - pred
    
    # 정규성 검정 (Jarque-Bera)
    jb_stat, jb_pvalue = stats.jarque_bera(error)
    
    # 기타 통계
    skewness = stats.skew(error)
    kurtosis = stats.kurtosis(error)
    
    print(f"  Mean Error: {np.mean(error):.4f}")
    print(f"  Std Error: {np.std(error):.4f}")
    print(f"  Skewness: {skewness:.4f}")
    print(f"  Kurtosis: {kurtosis:.4f}")
    print(f"  Jarque-Bera: stat={jb_stat:.2f}, p={jb_pvalue:.4f}")
    
    return {
        'mean': np.mean(error),
        'std': np.std(error),
        'skewness': skewness,
        'kurtosis': kurtosis,
        'jb_stat': jb_stat,
        'jb_pvalue': jb_pvalue
    }

# ============================================================================
# 6. 앙상블 효과 분석
# ============================================================================

def ensemble_analysis(ticker):
    """모델 앙상블 효과"""
    print(f"\n[6] 앙상블 효과: {ticker}")
    
    features = prepare_features(ticker)
    if features is None:
        return None
    
    feature_cols = ['RV_5d_lag1', 'RV_22d_lag1', 'RV_ratio_lag1', 
                    'VIX_lag1', 'VIX_change_lag1', 'direction_5d_lag1']
    X = features[feature_cols]
    y = features['RV_5d_future']
    
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
    
    # 개별 모델
    models = {
        'Ridge_10': Ridge(alpha=10.0),
        'Ridge_100': Ridge(alpha=100.0),
        'Ridge_1000': Ridge(alpha=1000.0),
        'Lasso': Lasso(alpha=0.01, max_iter=3000),
        'Huber': HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=500)
    }
    
    predictions = {}
    individual_r2 = {}
    
    for name, model in models.items():
        model.fit(X_train_s, np.sqrt(y_train))
        pred = np.maximum(model.predict(X_test_s) ** 2, 0)
        predictions[name] = pred
        individual_r2[name] = r2_score(y_test, pred)
        print(f"  {name}: R²={individual_r2[name]:.4f}")
    
    # 단순 평균 앙상블
    ensemble_pred = np.mean(list(predictions.values()), axis=0)
    ensemble_r2 = r2_score(y_test, ensemble_pred)
    
    print(f"\n  Ensemble (Simple Avg): R²={ensemble_r2:.4f}")
    
    # 최고 개별 모델 대비 개선
    best_individual = max(individual_r2.values())
    improvement = ensemble_r2 - best_individual
    print(f"  vs Best Individual: {improvement:+.4f}")
    
    return {
        'individual': individual_r2,
        'ensemble': ensemble_r2,
        'improvement': improvement
    }

# ============================================================================
# 메인
# ============================================================================

def main():
    print("="*80)
    print("논문 게재용 추가 실험 및 검증")
    print("="*80)
    
    assets = ['SPY', 'QQQ', 'XLK', 'XLF']
    
    all_results = {}
    
    # 1. Bootstrap 신뢰구간
    all_results['bootstrap'] = {}
    for ticker in assets:
        result = bootstrap_confidence_interval(ticker, n_bootstrap=100)
        if result:
            all_results['bootstrap'][ticker] = result
    
    # 2. Rolling Window
    all_results['rolling_window'] = {}
    for ticker in assets:
        result = rolling_window_analysis(ticker)
        if result:
            all_results['rolling_window'][ticker] = {
                'mean': result['mean'],
                'std': result['std'],
                'min': result['min'],
                'max': result['max']
            }
    
    # 3. 민감도 분석 (SPY만)
    all_results['sensitivity'] = hyperparameter_sensitivity('SPY')
    
    # 4. 상관관계
    all_results['correlation'] = cross_asset_correlation()
    
    # 5. 오차 분포
    all_results['error_dist'] = {}
    for ticker in assets:
        result = error_distribution_analysis(ticker)
        if result:
            all_results['error_dist'][ticker] = result
    
    # 6. 앙상블
    all_results['ensemble'] = {}
    for ticker in assets:
        result = ensemble_analysis(ticker)
        if result:
            all_results['ensemble'][ticker] = result
    
    # 결과 저장
    output = {
        'metadata': {
            'experiment': 'Paper Publication Experiments',
            'timestamp': datetime.now().isoformat()
        },
        'results': all_results
    }
    
    output_path = 'data/results/paper_publication_experiments.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    # 요약
    print("\n" + "="*80)
    print("요약")
    print("="*80)
    
    print("\n[Bootstrap 95% CI]")
    for ticker in assets:
        if ticker in all_results['bootstrap']:
            b = all_results['bootstrap'][ticker]
            print(f"  {ticker}: {b['mean']:.4f} [{b['ci_lower']:.4f}, {b['ci_upper']:.4f}]")
    
    print("\n[앙상블 효과]")
    for ticker in assets:
        if ticker in all_results['ensemble']:
            e = all_results['ensemble'][ticker]
            print(f"  {ticker}: 앙상블 R²={e['ensemble']:.4f}, 개선={e['improvement']:+.4f}")
    
    print(f"\n결과 저장: {output_path}")

if __name__ == "__main__":
    main()
