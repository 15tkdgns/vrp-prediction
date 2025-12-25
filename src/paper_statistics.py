"""
논문용 통계 검정 및 테이블 데이터 생성
=====================================
1. 기술 통계량 (Descriptive Statistics)
2. 모델 성능 비교 테이블
3. Diebold-Mariano 검정
4. Walk-Forward CV 결과 테이블
5. 특성 중요도 테이블
"""
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import Ridge, Lasso, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def calculate_rv(returns, window):
    rv = (returns ** 2).rolling(window).sum() * 252
    return rv.iloc[:, 0] if isinstance(rv, pd.DataFrame) else rv

def prepare_features(ticker):
    """하이브리드 특성"""
    data = yf.download(ticker, start='2015-01-01', end='2025-01-01', progress=False)
    if len(data) < 500:
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
    
    vix = yf.download('^VIX', start='2015-01-01', end='2025-01-01', progress=False)
    vix_close = vix['Close'].iloc[:, 0] if isinstance(vix['Close'], pd.DataFrame) else vix['Close']
    features['VIX_lag1'] = vix_close.reindex(data.index).ffill().shift(1)
    features['VIX_change_lag1'] = features['VIX_lag1'].pct_change()
    features['direction_5d_lag1'] = returns.rolling(5).apply(lambda x: np.mean(x > 0)).shift(1)
    
    features['RV_5d_future'] = rv_5d.shift(-5)
    
    return features.dropna()

# ============================================================================
# 1. 기술 통계량 (Descriptive Statistics)
# ============================================================================

def generate_descriptive_stats():
    """기술 통계량 생성"""
    print("\n" + "="*60)
    print("[1] 기술 통계량 (Descriptive Statistics)")
    print("="*60)
    
    assets = ['SPY', 'QQQ', 'EEM', 'GLD', 'TLT', 'IWM', 'XLE', 'XLK', 'XLF']
    
    stats_data = []
    
    for ticker in assets:
        features = prepare_features(ticker)
        if features is None:
            continue
        
        rv = features['RV_5d_future']
        
        stats_data.append({
            'Asset': ticker,
            'N': len(rv),
            'Mean': rv.mean(),
            'Std': rv.std(),
            'Min': rv.min(),
            'Q1': rv.quantile(0.25),
            'Median': rv.quantile(0.50),
            'Q3': rv.quantile(0.75),
            'Max': rv.max(),
            'Skewness': rv.skew(),
            'Kurtosis': rv.kurtosis()
        })
    
    df = pd.DataFrame(stats_data)
    
    print("\nTable 1: Descriptive Statistics of 5-Day Realized Volatility")
    print(df.to_string(index=False))
    
    return df

# ============================================================================
# 2. 모델 성능 비교 테이블
# ============================================================================

def generate_model_comparison():
    """모델 성능 비교 테이블"""
    print("\n" + "="*60)
    print("[2] 모델 성능 비교 (Model Comparison)")
    print("="*60)
    
    # 최적 설정
    configs = {
        'SPY': ('ridge', 'sqrt', 100.0),
        'QQQ': ('lasso', 'log', 0.01),
        'XLK': ('ridge', 'raw', 1000.0),
        'XLF': ('huber', 'log', 1.0)
    }
    
    results = []
    
    for ticker, (model_type, transform, alpha) in configs.items():
        features = prepare_features(ticker)
        if features is None:
            continue
        
        feature_cols = [c for c in features.columns if c != 'RV_5d_future']
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
        
        if transform == 'sqrt':
            y_train_t = np.sqrt(y_train)
        elif transform == 'log':
            y_train_t = np.log(y_train + 1)
        else:
            y_train_t = y_train.values
        
        if model_type == 'ridge':
            model = Ridge(alpha=alpha)
        elif model_type == 'lasso':
            model = Lasso(alpha=alpha, max_iter=3000)
        else:
            model = HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=500)
        
        model.fit(X_train_s, y_train_t)
        
        if transform == 'sqrt':
            pred = model.predict(X_test_s) ** 2
        elif transform == 'log':
            pred = np.exp(model.predict(X_test_s)) - 1
        else:
            pred = model.predict(X_test_s)
        
        pred = np.maximum(pred, 0)
        
        # Persistence 벤치마크
        persist_pred = X_test['RV_5d_lag1'].values
        
        # 성능 지표
        r2 = r2_score(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mae = mean_absolute_error(y_test, pred)
        
        r2_persist = r2_score(y_test, persist_pred)
        rmse_persist = np.sqrt(mean_squared_error(y_test, persist_pred))
        
        # 방향 정확도
        actual_dir = (y_test.values > persist_pred).astype(int)
        pred_dir = (pred > persist_pred).astype(int)
        direction_acc = (actual_dir == pred_dir).mean()
        
        results.append({
            'Asset': ticker,
            'Model': f'{model_type.capitalize()}({transform})',
            'R2': r2,
            'R2_Persist': r2_persist,
            'R2_Improvement': r2 - r2_persist,
            'RMSE': rmse,
            'RMSE_Persist': rmse_persist,
            'MAE': mae,
            'Direction_Acc': direction_acc
        })
    
    df = pd.DataFrame(results)
    
    print("\nTable 2: Model Performance Comparison")
    print(df.to_string(index=False))
    
    return df

# ============================================================================
# 3. Diebold-Mariano 검정
# ============================================================================

def diebold_mariano_test(e1, e2, h=1):
    """
    Diebold-Mariano 검정
    H0: 두 모델의 예측력이 동일
    """
    d = e1**2 - e2**2
    
    n = len(d)
    d_mean = np.mean(d)
    
    # HAC 표준오차 (Newey-West)
    gamma_0 = np.var(d)
    gamma_sum = 0
    for k in range(1, h):
        weight = 1 - k / h
        gamma_k = np.cov(d[:-k], d[k:])[0, 1]
        gamma_sum += 2 * weight * gamma_k
    
    var_d = (gamma_0 + gamma_sum) / n
    
    if var_d > 0:
        dm_stat = d_mean / np.sqrt(var_d)
        p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    else:
        dm_stat = np.nan
        p_value = np.nan
    
    return dm_stat, p_value

def generate_dm_tests():
    """Diebold-Mariano 검정 결과"""
    print("\n" + "="*60)
    print("[3] Diebold-Mariano Test (Model vs Persistence)")
    print("="*60)
    
    configs = {
        'SPY': ('ridge', 'sqrt', 100.0),
        'QQQ': ('lasso', 'log', 0.01),
        'XLK': ('ridge', 'raw', 1000.0),
        'XLF': ('huber', 'log', 1.0)
    }
    
    results = []
    
    for ticker, (model_type, transform, alpha) in configs.items():
        features = prepare_features(ticker)
        if features is None:
            continue
        
        feature_cols = [c for c in features.columns if c != 'RV_5d_future']
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
        
        if transform == 'sqrt':
            y_train_t = np.sqrt(y_train)
        elif transform == 'log':
            y_train_t = np.log(y_train + 1)
        else:
            y_train_t = y_train.values
        
        if model_type == 'ridge':
            model = Ridge(alpha=alpha)
        elif model_type == 'lasso':
            model = Lasso(alpha=alpha, max_iter=3000)
        else:
            model = HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=500)
        
        model.fit(X_train_s, y_train_t)
        
        if transform == 'sqrt':
            pred = model.predict(X_test_s) ** 2
        elif transform == 'log':
            pred = np.exp(model.predict(X_test_s)) - 1
        else:
            pred = model.predict(X_test_s)
        
        pred = np.maximum(pred, 0)
        
        # 에러 계산
        e_model = y_test.values - pred
        e_persist = y_test.values - X_test['RV_5d_lag1'].values
        
        # DM 검정
        dm_stat, p_value = diebold_mariano_test(e_model, e_persist, h=5)
        
        results.append({
            'Asset': ticker,
            'DM_Statistic': dm_stat,
            'P_Value': p_value,
            'Significant_5%': '***' if p_value < 0.01 else ('**' if p_value < 0.05 else ('*' if p_value < 0.10 else ''))
        })
    
    df = pd.DataFrame(results)
    
    print("\nTable 3: Diebold-Mariano Test Results")
    print("H0: Model and Persistence have equal predictive accuracy")
    print(df.to_string(index=False))
    print("\n*** p<0.01, ** p<0.05, * p<0.10")
    
    return df

# ============================================================================
# 4. Walk-Forward CV 결과 테이블
# ============================================================================

def generate_wf_cv_results():
    """Walk-Forward CV 결과"""
    print("\n" + "="*60)
    print("[4] Walk-Forward Cross-Validation Results")
    print("="*60)
    
    configs = {
        'SPY': ('ridge', 'sqrt', 100.0),
        'QQQ': ('lasso', 'log', 0.01),
        'XLK': ('ridge', 'raw', 1000.0),
        'XLF': ('huber', 'log', 1.0)
    }
    
    all_results = []
    
    for ticker, (model_type, transform, alpha) in configs.items():
        features = prepare_features(ticker)
        if features is None:
            continue
        
        feature_cols = [c for c in features.columns if c != 'RV_5d_future']
        X = features[feature_cols]
        y = features['RV_5d_future']
        
        tscv = TimeSeriesSplit(n_splits=5)
        gap = 5
        
        fold_r2s = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            train_idx = train_idx[:-gap] if len(train_idx) > gap else train_idx
            
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]
            
            if len(X_train) < 200 or len(X_test) < 50:
                continue
            
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            
            if transform == 'sqrt':
                y_train_t = np.sqrt(y_train)
            elif transform == 'log':
                y_train_t = np.log(y_train + 1)
            else:
                y_train_t = y_train.values
            
            if model_type == 'ridge':
                model = Ridge(alpha=alpha)
            elif model_type == 'lasso':
                model = Lasso(alpha=alpha, max_iter=3000)
            else:
                model = HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=500)
            
            model.fit(X_train_s, y_train_t)
            
            if transform == 'sqrt':
                pred = model.predict(X_test_s) ** 2
            elif transform == 'log':
                pred = np.exp(model.predict(X_test_s)) - 1
            else:
                pred = model.predict(X_test_s)
            
            pred = np.maximum(pred, 0)
            r2 = r2_score(y_test, pred)
            fold_r2s.append(r2)
        
        all_results.append({
            'Asset': ticker,
            'Model': f'{model_type.capitalize()}({transform})',
            'Fold_1': fold_r2s[0] if len(fold_r2s) > 0 else np.nan,
            'Fold_2': fold_r2s[1] if len(fold_r2s) > 1 else np.nan,
            'Fold_3': fold_r2s[2] if len(fold_r2s) > 2 else np.nan,
            'Fold_4': fold_r2s[3] if len(fold_r2s) > 3 else np.nan,
            'Fold_5': fold_r2s[4] if len(fold_r2s) > 4 else np.nan,
            'Mean': np.mean(fold_r2s),
            'Std': np.std(fold_r2s)
        })
    
    df = pd.DataFrame(all_results)
    
    print("\nTable 4: Walk-Forward Cross-Validation R² by Fold")
    print(df.to_string(index=False))
    
    return df

# ============================================================================
# 5. 특성 중요도 테이블
# ============================================================================

def generate_feature_importance_table():
    """특성 중요도 테이블"""
    print("\n" + "="*60)
    print("[5] Feature Importance (Permutation Importance)")
    print("="*60)
    
    configs = {
        'SPY': ('ridge', 'sqrt', 100.0),
        'QQQ': ('lasso', 'log', 0.01),
        'XLK': ('ridge', 'raw', 1000.0),
        'XLF': ('huber', 'log', 1.0)
    }
    
    feature_names = ['RV_5d_lag1', 'RV_22d_lag1', 'RV_ratio_lag1', 
                     'VIX_lag1', 'VIX_change_lag1', 'direction_5d_lag1']
    
    importance_data = {f: {} for f in feature_names}
    
    for ticker, (model_type, transform, alpha) in configs.items():
        features = prepare_features(ticker)
        if features is None:
            continue
        
        feature_cols = [c for c in features.columns if c != 'RV_5d_future']
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
        
        if transform == 'sqrt':
            y_train_t = np.sqrt(y_train)
        elif transform == 'log':
            y_train_t = np.log(y_train + 1)
        else:
            y_train_t = y_train.values
        
        if model_type == 'ridge':
            model = Ridge(alpha=alpha)
        elif model_type == 'lasso':
            model = Lasso(alpha=alpha, max_iter=3000)
        else:
            model = HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=500)
        
        model.fit(X_train_s, y_train_t)
        
        # 기본 예측
        if transform == 'sqrt':
            pred = model.predict(X_test_s) ** 2
        elif transform == 'log':
            pred = np.exp(model.predict(X_test_s)) - 1
        else:
            pred = model.predict(X_test_s)
        pred = np.maximum(pred, 0)
        base_r2 = r2_score(y_test, pred)
        
        # 순열 중요도
        for i, col in enumerate(feature_cols):
            X_test_perm = X_test_s.copy()
            np.random.seed(42)
            X_test_perm[:, i] = np.random.permutation(X_test_perm[:, i])
            
            if transform == 'sqrt':
                pred_perm = model.predict(X_test_perm) ** 2
            elif transform == 'log':
                pred_perm = np.exp(model.predict(X_test_perm)) - 1
            else:
                pred_perm = model.predict(X_test_perm)
            pred_perm = np.maximum(pred_perm, 0)
            
            perm_r2 = r2_score(y_test, pred_perm)
            importance_data[col][ticker] = base_r2 - perm_r2
    
    df = pd.DataFrame(importance_data).T
    df['Mean'] = df.mean(axis=1)
    df = df.sort_values('Mean', ascending=False)
    
    print("\nTable 5: Permutation Feature Importance (R² Decrease)")
    print(df.to_string())
    
    return df

# ============================================================================
# 메인
# ============================================================================

def main():
    print("="*80)
    print("논문용 통계 검정 및 테이블 데이터 생성")
    print("="*80)
    
    all_tables = {}
    
    # 1. 기술 통계량
    all_tables['descriptive_stats'] = generate_descriptive_stats()
    
    # 2. 모델 성능 비교
    all_tables['model_comparison'] = generate_model_comparison()
    
    # 3. DM 검정
    all_tables['dm_test'] = generate_dm_tests()
    
    # 4. Walk-Forward CV
    all_tables['wf_cv'] = generate_wf_cv_results()
    
    # 5. 특성 중요도
    all_tables['feature_importance'] = generate_feature_importance_table()
    
    # JSON 저장
    output = {
        'metadata': {
            'experiment': 'Paper Statistics and Tables',
            'timestamp': datetime.now().isoformat()
        },
        'tables': {
            k: v.to_dict() for k, v in all_tables.items()
        }
    }
    
    output_path = 'data/results/paper_statistics.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    # CSV 저장
    for name, df in all_tables.items():
        csv_path = f'data/results/table_{name}.csv'
        df.to_csv(csv_path, index=True)
        print(f"\n저장됨: {csv_path}")
    
    print(f"\n전체 결과 저장: {output_path}")

if __name__ == "__main__":
    main()
