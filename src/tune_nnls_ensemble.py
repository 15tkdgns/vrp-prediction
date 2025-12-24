"""
NNLS 앙상블 구성 모델 튜닝
===========================
성공적인 NNLS 앙상블의 핵심 모델들 최적화:
- Lasso: QQQ, SPY에서 96-97% 가중치
- Huber: GLD, EEM에서 92-95% 가중치
- RF: 보조 모델로 활용
"""
import pandas as pd
import numpy as np
import json
import optuna
from optuna.samplers import TPESampler
from sklearn.linear_model import Lasso, ElasticNet, HuberRegressor, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from scipy.optimize import nnls
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 데이터 준비
# ============================================================================

def calculate_rv(returns, window=22):
    rv = (returns ** 2).rolling(window).sum() * 252
    return rv.iloc[:, 0] if isinstance(rv, pd.DataFrame) else rv

def prepare_data(ticker):
    df = yf.download(ticker, start='2020-01-01', end='2025-01-01', progress=False)
    if len(df) < 100:
        return None, None
    
    returns = df['Close'].pct_change()
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    
    features = pd.DataFrame(index=df.index)
    
    features['RV_1d'] = (returns ** 2) * 252
    features['RV_5d'] = calculate_rv(returns, 5)
    features['RV_22d'] = calculate_rv(returns, 22)
    features['RV_daily_mean'] = features['RV_1d'].rolling(5).mean()
    features['RV_weekly_mean'] = features['RV_5d'].rolling(4).mean()
    features['RV_monthly_mean'] = features['RV_22d'].rolling(3).mean()
    
    features['RV_22d_future'] = features['RV_22d'].shift(-22)
    
    spy = yf.download('SPY', start='2020-01-01', end='2025-01-01', progress=False)
    spy_ret = spy['Close'].pct_change()
    if isinstance(spy_ret, pd.DataFrame):
        spy_ret = spy_ret.iloc[:, 0]
    spy_rv = calculate_rv(spy_ret, 22)
    features['VIX_proxy'] = spy_rv.reindex(df.index).ffill()
    
    features['VIX_lag1'] = features['VIX_proxy'].shift(1)
    features['VIX_lag5'] = features['VIX_proxy'].shift(5)
    features['VIX_change'] = features['VIX_proxy'].pct_change()
    
    features['VRP'] = features['VIX_proxy'] - features['RV_22d']
    features['VRP_lag1'] = features['VRP'].shift(1)
    features['VRP_ma5'] = features['VRP'].rolling(5).mean()
    
    features['regime_high'] = (features['VIX_proxy'] >= 25).astype(int)
    features['return_5d'] = returns.rolling(5).sum()
    features['return_22d'] = returns.rolling(22).sum()
    
    features = features.dropna()
    
    feature_cols = [c for c in features.columns if c != 'RV_22d_future']
    X = features[feature_cols]
    y = features['RV_22d_future']
    
    return X, y

# ============================================================================
# Cross-Validation 기반 모델 평가
# ============================================================================

def cv_evaluate_model(model, X, y, n_splits=3, gap=22):
    """TimeSeriesSplit CV로 모델 평가"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    
    for train_idx, test_idx in tscv.split(X):
        if len(train_idx) < gap + 100:
            continue
        
        train_idx = train_idx[:-gap]
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        y_train_log = np.log(y_train + 1)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model.fit(X_train_scaled, y_train_log)
        y_pred_log = model.predict(X_test_scaled)
        y_pred = np.exp(y_pred_log) - 1
        
        scores.append(r2_score(y_test, y_pred))
    
    return np.mean(scores) if scores else -999

# ============================================================================
# Lasso 튜닝
# ============================================================================

def tune_lasso(X, y, n_trials=100):
    """Lasso 하이퍼파라미터 튜닝"""
    
    def objective(trial):
        alpha = trial.suggest_float('alpha', 1e-6, 1.0, log=True)
        max_iter = trial.suggest_int('max_iter', 1000, 5000)
        
        model = Lasso(alpha=alpha, max_iter=max_iter, random_state=42)
        return cv_evaluate_model(model, X, y)
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    return study.best_params, study.best_value

# ============================================================================
# ElasticNet 튜닝
# ============================================================================

def tune_elasticnet(X, y, n_trials=100):
    """ElasticNet 하이퍼파라미터 튜닝"""
    
    def objective(trial):
        alpha = trial.suggest_float('alpha', 1e-6, 1.0, log=True)
        l1_ratio = trial.suggest_float('l1_ratio', 0.0, 1.0)
        max_iter = trial.suggest_int('max_iter', 1000, 5000)
        
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, random_state=42)
        return cv_evaluate_model(model, X, y)
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    return study.best_params, study.best_value

# ============================================================================
# Huber 튜닝
# ============================================================================

def tune_huber(X, y, n_trials=100):
    """Huber 하이퍼파라미터 튜닝"""
    
    def objective(trial):
        epsilon = trial.suggest_float('epsilon', 1.0, 3.0)
        alpha = trial.suggest_float('alpha', 1e-6, 1.0, log=True)
        max_iter = trial.suggest_int('max_iter', 100, 500)
        
        model = HuberRegressor(epsilon=epsilon, alpha=alpha, max_iter=max_iter)
        return cv_evaluate_model(model, X, y)
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    return study.best_params, study.best_value

# ============================================================================
# Ridge 튜닝
# ============================================================================

def tune_ridge(X, y, n_trials=100):
    """Ridge 하이퍼파라미터 튜닝"""
    
    def objective(trial):
        alpha = trial.suggest_float('alpha', 1e-6, 100.0, log=True)
        
        model = Ridge(alpha=alpha, random_state=42)
        return cv_evaluate_model(model, X, y)
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    return study.best_params, study.best_value

# ============================================================================
# RF 튜닝
# ============================================================================

def tune_rf(X, y, n_trials=50):
    """Random Forest 하이퍼파라미터 튜닝"""
    
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 300)
        max_depth = trial.suggest_int('max_depth', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 5, 50)
        
        model = RandomForestRegressor(
            n_estimators=n_estimators, 
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
        return cv_evaluate_model(model, X, y)
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    return study.best_params, study.best_value

# ============================================================================
# 튜닝된 모델로 NNLS 앙상블
# ============================================================================

def run_tuned_nnls_ensemble(ticker, best_params, gap=22):
    """튜닝된 모델들로 NNLS 앙상블 실행"""
    X, y = prepare_data(ticker)
    if X is None:
        return None
    
    y_log = np.log(y + 1)
    
    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx - gap]
    y_train = y_log.iloc[:split_idx - gap]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    
    # 튜닝된 모델들 생성
    models = {}
    
    if 'Lasso' in best_params:
        p = best_params['Lasso']
        models['Lasso'] = Lasso(alpha=p['alpha'], max_iter=p.get('max_iter', 2000), random_state=42)
    
    if 'ElasticNet' in best_params:
        p = best_params['ElasticNet']
        models['ElasticNet'] = ElasticNet(alpha=p['alpha'], l1_ratio=p['l1_ratio'], 
                                           max_iter=p.get('max_iter', 2000), random_state=42)
    
    if 'Huber' in best_params:
        p = best_params['Huber']
        models['Huber'] = HuberRegressor(epsilon=p['epsilon'], alpha=p['alpha'],
                                          max_iter=p.get('max_iter', 200))
    
    if 'Ridge' in best_params:
        p = best_params['Ridge']
        models['Ridge'] = Ridge(alpha=p['alpha'], random_state=42)
    
    if 'RF' in best_params:
        p = best_params['RF']
        models['RF'] = RandomForestRegressor(n_estimators=p['n_estimators'],
                                              max_depth=p['max_depth'],
                                              min_samples_leaf=p.get('min_samples_leaf', 10),
                                              random_state=42)
    
    # 학습 및 예측
    predictions = {}
    individual_r2 = {}
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    for name, model in models.items():
        try:
            model.fit(X_train_scaled, y_train)
            pred_log = model.predict(X_test_scaled)
            pred = np.exp(pred_log) - 1
            
            predictions[name] = pred
            individual_r2[name] = r2_score(y_test, pred)
        except Exception as e:
            print(f"  {name} failed: {e}")
    
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
        'weights': dict(zip(model_names, weights.tolist()))
    }

# ============================================================================
# 메인 실행
# ============================================================================

def main():
    assets = ['SPY', 'GLD', 'QQQ', 'TLT', 'EEM']
    n_trials = 100  # 각 모델당 trials
    
    all_results = {}
    
    print("="*80)
    print("NNLS Ensemble Component Tuning")
    print("="*80)
    
    for asset in assets:
        print(f"\n{'#'*80}")
        print(f"# Tuning models for {asset}")
        print(f"{'#'*80}")
        
        X, y = prepare_data(asset)
        if X is None:
            continue
        
        asset_results = {'best_params': {}}
        
        # Lasso 튜닝
        print(f"\n[Lasso Tuning] {n_trials} trials")
        best_params, best_score = tune_lasso(X, y, n_trials)
        asset_results['best_params']['Lasso'] = best_params
        asset_results['Lasso_cv_r2'] = round(best_score, 6)
        print(f"  Best params: {best_params}")
        print(f"  Best CV R2: {best_score:.4f}")
        
        # ElasticNet 튜닝
        print(f"\n[ElasticNet Tuning] {n_trials} trials")
        best_params, best_score = tune_elasticnet(X, y, n_trials)
        asset_results['best_params']['ElasticNet'] = best_params
        asset_results['ElasticNet_cv_r2'] = round(best_score, 6)
        print(f"  Best params: {best_params}")
        print(f"  Best CV R2: {best_score:.4f}")
        
        # Huber 튜닝
        print(f"\n[Huber Tuning] {n_trials} trials")
        best_params, best_score = tune_huber(X, y, n_trials)
        asset_results['best_params']['Huber'] = best_params
        asset_results['Huber_cv_r2'] = round(best_score, 6)
        print(f"  Best params: {best_params}")
        print(f"  Best CV R2: {best_score:.4f}")
        
        # Ridge 튜닝
        print(f"\n[Ridge Tuning] {n_trials} trials")
        best_params, best_score = tune_ridge(X, y, n_trials)
        asset_results['best_params']['Ridge'] = best_params
        asset_results['Ridge_cv_r2'] = round(best_score, 6)
        print(f"  Best params: {best_params}")
        print(f"  Best CV R2: {best_score:.4f}")
        
        # RF 튜닝 (적은 trials)
        print(f"\n[RF Tuning] 50 trials")
        best_params, best_score = tune_rf(X, y, 50)
        asset_results['best_params']['RF'] = best_params
        asset_results['RF_cv_r2'] = round(best_score, 6)
        print(f"  Best params: {best_params}")
        print(f"  Best CV R2: {best_score:.4f}")
        
        # 튜닝된 모델로 NNLS 앙상블
        print(f"\n[Tuned NNLS Ensemble]")
        ens_result = run_tuned_nnls_ensemble(asset, asset_results['best_params'], gap=22)
        if ens_result:
            asset_results['tuned_ensemble'] = ens_result
            print(f"  Individual R2: {ens_result['individual_r2']}")
            print(f"  Ensemble R2: {ens_result['ensemble_r2']:.4f}")
            print(f"  Weights: {ens_result['weights']}")
        
        all_results[asset] = asset_results
    
    # 결과 저장
    output = {
        'metadata': {
            'experiment': 'NNLS Ensemble Component Tuning',
            'n_trials': n_trials,
            'timestamp': datetime.now().isoformat()
        },
        'results': all_results
    }
    
    output_path = 'data/results/nnls_tuned_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    print("\n" + "="*80)
    print("Results Summary:")
    print("="*80)
    
    for asset, results in all_results.items():
        if 'tuned_ensemble' in results:
            print(f"  {asset}: Tuned Ensemble R2={results['tuned_ensemble']['ensemble_r2']:.4f}")
    
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
