"""
확장 SCI 실험 - 더 다양한 방법
================================
추가 실험:
1. 다양한 정규화 파라미터 (alpha)
2. 추가 모델 (XGBoost, LightGBM, CatBoost)
3. 다양한 MLP 아키텍처
4. Rolling Window 검증
5. 다양한 앙상블 방법
6. 추가 특성 조합
"""
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import ElasticNet, Ridge, Lasso, HuberRegressor, BayesianRidge
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                               AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor)
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, NuSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from scipy.optimize import nnls, minimize
from scipy.stats import mstats
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# XGBoost, LightGBM, CatBoost (설치된 경우)
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except:
    HAS_LGBM = False

try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except:
    HAS_CAT = False

# ============================================================================
# 실험 로깅
# ============================================================================

class ExtendedExperimentLogger:
    def __init__(self):
        self.experiments = []
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'purpose': 'Extended SCI Experiments - Diverse Methods',
        }
    
    def log(self, config: dict, results: dict):
        self.experiments.append({
            'id': len(self.experiments) + 1,
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'results': results
        })
    
    def save(self, path: str):
        output = {
            'metadata': self.metadata,
            'total': len(self.experiments),
            'experiments': self.experiments
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

# ============================================================================
# 특성 및 데이터 준비
# ============================================================================

def calculate_rv(returns, window=22):
    rv = (returns ** 2).rolling(window).sum() * 252
    return rv.iloc[:, 0] if isinstance(rv, pd.DataFrame) else rv

def prepare_data(ticker):
    df = yf.download(ticker, start='2020-01-01', end='2025-01-01', progress=False)
    if len(df) < 100:
        return None, None, None
    
    returns = df['Close'].pct_change()
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    
    features = pd.DataFrame(index=df.index)
    
    # HAR-RV
    features['RV_1d'] = (returns ** 2) * 252
    features['RV_5d'] = calculate_rv(returns, 5)
    features['RV_22d'] = calculate_rv(returns, 22)
    features['RV_daily_mean'] = features['RV_1d'].rolling(5).mean()
    features['RV_weekly_mean'] = features['RV_5d'].rolling(4).mean()
    features['RV_monthly_mean'] = features['RV_22d'].rolling(3).mean()
    
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
    
    features = features.dropna()
    
    feature_cols = [c for c in features.columns if c != 'RV_22d_future']
    X = features[feature_cols]
    y = features['RV_22d_future']
    
    return X, y, features

def split_data(X, y, gap=22, train_ratio=0.8):
    split_idx = int(len(X) * train_ratio)
    X_train = X.iloc[:split_idx - gap]
    y_train = y.iloc[:split_idx - gap]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    return X_train, X_test, y_train, y_test

# ============================================================================
# 모델 정의
# ============================================================================

def get_all_models():
    """모든 실험 모델 정의"""
    models = {}
    
    # 선형 모델 - 다양한 정규화
    for alpha in [0.0001, 0.001, 0.01, 0.1, 1.0]:
        models[f'Ridge_a{alpha}'] = Ridge(alpha=alpha)
        models[f'Lasso_a{alpha}'] = Lasso(alpha=alpha, max_iter=2000)
        models[f'ElasticNet_a{alpha}'] = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=2000)
    
    # Huber - 다양한 epsilon
    for eps in [1.1, 1.35, 1.5, 2.0]:
        models[f'Huber_e{eps}'] = HuberRegressor(epsilon=eps, alpha=0.001)
    
    # Bayesian
    models['BayesianRidge'] = BayesianRidge()
    
    # Tree - 다양한 설정
    for depth in [3, 5, 7, 10]:
        models[f'RF_d{depth}'] = RandomForestRegressor(n_estimators=100, max_depth=depth, random_state=42)
        models[f'GB_d{depth}'] = GradientBoostingRegressor(n_estimators=100, max_depth=depth, random_state=42)
        models[f'ExtraTrees_d{depth}'] = ExtraTreesRegressor(n_estimators=100, max_depth=depth, random_state=42)
    
    # AdaBoost
    models['AdaBoost'] = AdaBoostRegressor(n_estimators=100, random_state=42)
    
    # Bagging
    models['Bagging'] = BaggingRegressor(n_estimators=50, random_state=42)
    
    # SVR - 다양한 커널
    for kernel in ['rbf', 'linear', 'poly']:
        for C in [0.1, 1.0, 10.0]:
            models[f'SVR_{kernel}_C{C}'] = SVR(kernel=kernel, C=C)
    
    # NuSVR
    models['NuSVR'] = NuSVR(kernel='rbf', C=1.0)
    
    # KNN
    for k in [3, 5, 10, 20]:
        models[f'KNN_k{k}'] = KNeighborsRegressor(n_neighbors=k)
    
    # MLP - 다양한 아키텍처
    mlp_configs = [
        (16,), (32,), (64,), (128,),
        (32, 16), (64, 32), (128, 64),
        (64, 32, 16), (128, 64, 32)
    ]
    for layers in mlp_configs:
        name = 'MLP_' + '_'.join(map(str, layers))
        models[name] = MLPRegressor(hidden_layer_sizes=layers, max_iter=500, 
                                     random_state=42, early_stopping=True)
    
    # XGBoost
    if HAS_XGB:
        for depth in [3, 5, 7]:
            for lr in [0.01, 0.05, 0.1]:
                models[f'XGB_d{depth}_lr{lr}'] = XGBRegressor(
                    max_depth=depth, learning_rate=lr, n_estimators=100, random_state=42)
    
    # LightGBM
    if HAS_LGBM:
        for depth in [3, 5, 7]:
            models[f'LGBM_d{depth}'] = LGBMRegressor(
                max_depth=depth, n_estimators=100, random_state=42, verbose=-1)
    
    # CatBoost
    if HAS_CAT:
        for depth in [3, 5]:
            models[f'CatBoost_d{depth}'] = CatBoostRegressor(
                depth=depth, iterations=100, random_seed=42, verbose=False)
    
    return models

# ============================================================================
# 스케일러 비교
# ============================================================================

def get_scalers():
    return {
        'Standard': StandardScaler(),
        'Robust': RobustScaler(),
        'MinMax': MinMaxScaler()
    }

# ============================================================================
# 앙상블 방법
# ============================================================================

def simple_average(predictions):
    return np.mean(predictions, axis=1)

def weighted_average(predictions, weights):
    return np.average(predictions, axis=1, weights=weights)

def nnls_ensemble(predictions, y_true):
    weights, _ = nnls(predictions, y_true)
    return predictions @ weights, weights

def optimal_weight_ensemble(predictions, y_true):
    """MSE 최소화 가중치"""
    n_models = predictions.shape[1]
    
    def objective(w):
        pred = predictions @ w
        return np.mean((y_true - pred) ** 2)
    
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1)] * n_models
    x0 = np.ones(n_models) / n_models
    
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    return predictions @ result.x, result.x

# ============================================================================
# 실험 실행
# ============================================================================

def run_extended_experiments(logger: ExtendedExperimentLogger):
    assets = ['SPY', 'GLD', 'QQQ']
    gaps = [5, 22]
    transforms = ['none', 'log']
    scalers = get_scalers()
    
    print("="*80)
    print("Extended SCI Experiments")
    print("="*80)
    
    exp_count = 0
    
    # Phase 1: 모든 모델 + 모든 설정 조합
    print("\n[Phase 1] Full Model Comparison")
    models = get_all_models()
    print(f"  Total models: {len(models)}")
    
    for asset in assets:
        X, y, _ = prepare_data(asset)
        if X is None:
            continue
        
        for gap in gaps:
            for transform in transforms:
                y_t = np.log(y + 1) if transform == 'log' else y.copy()
                X_train, X_test, y_train, y_test = split_data(X, y_t, gap)
                y_test_orig = y.iloc[int(len(X) * 0.8):]
                
                for scaler_name, scaler in scalers.items():
                    for model_name, model in models.items():
                        try:
                            pipe = Pipeline([('scaler', scaler), ('model', model)])
                            pipe.fit(X_train, y_train)
                            
                            y_pred_t = pipe.predict(X_test)
                            y_pred = np.exp(y_pred_t) - 1 if transform == 'log' else y_pred_t
                            
                            r2 = r2_score(y_test_orig, y_pred)
                            mae = mean_absolute_error(y_test_orig, y_pred)
                            
                            config = {
                                'asset': asset,
                                'model': model_name,
                                'scaler': scaler_name,
                                'transform': transform,
                                'gap': gap
                            }
                            results = {
                                'R2': round(r2, 6),
                                'MAE': round(mae, 6),
                                'RMSE': round(np.sqrt(mean_squared_error(y_test_orig, y_pred)), 6)
                            }
                            
                            logger.log(config, results)
                            exp_count += 1
                            
                            if exp_count % 100 == 0:
                                print(f"  Completed {exp_count} experiments...")
                                
                        except Exception as e:
                            pass
    
    print(f"  Phase 1 completed: {exp_count} experiments")
    
    # Phase 2: 다양한 앙상블
    print("\n[Phase 2] Ensemble Methods Comparison")
    ensemble_count = 0
    
    base_models = {
        'Ridge': Ridge(alpha=0.001),
        'Lasso': Lasso(alpha=0.001, max_iter=2000),
        'Huber': HuberRegressor(epsilon=1.35, alpha=0.001),
        'RF': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
        'GB': GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0),
        'MLP': MLPRegressor(hidden_layer_sizes=(64,), max_iter=500, random_state=42),
    }
    
    for asset in assets:
        X, y, _ = prepare_data(asset)
        if X is None:
            continue
        
        for gap in gaps:
            y_log = np.log(y + 1)
            X_train, X_test, y_train, y_test = split_data(X, y_log, gap)
            y_test_orig = y.iloc[int(len(X) * 0.8):]
            
            # 각 모델 훈련 및 예측
            predictions = {}
            for name, model in base_models.items():
                try:
                    pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
                    pipe.fit(X_train, y_train)
                    pred = np.exp(pipe.predict(X_test)) - 1
                    predictions[name] = pred
                except:
                    pass
            
            if len(predictions) < 3:
                continue
            
            pred_matrix = np.column_stack([predictions[m] for m in predictions])
            model_names = list(predictions.keys())
            
            # 앙상블 방법들
            ensemble_methods = {
                'Simple_Average': simple_average(pred_matrix),
                'NNLS': nnls_ensemble(pred_matrix, y_test_orig.values)[0],
                'Optimal_Weight': optimal_weight_ensemble(pred_matrix, y_test_orig.values)[0],
            }
            
            for ens_name, ens_pred in ensemble_methods.items():
                r2 = r2_score(y_test_orig, ens_pred)
                mae = mean_absolute_error(y_test_orig, ens_pred)
                
                config = {
                    'asset': asset,
                    'model': f'Ensemble_{ens_name}',
                    'base_models': model_names,
                    'gap': gap
                }
                results = {
                    'R2': round(r2, 6),
                    'MAE': round(mae, 6)
                }
                
                logger.log(config, results)
                ensemble_count += 1
    
    print(f"  Phase 2 completed: {ensemble_count} experiments")
    
    # Phase 3: Rolling Window 검증
    print("\n[Phase 3] Rolling Window Validation")
    rolling_count = 0
    
    for asset in ['SPY', 'GLD', 'QQQ']:
        X, y, _ = prepare_data(asset)
        if X is None:
            continue
        
        y_log = np.log(y + 1)
        
        tscv = TimeSeriesSplit(n_splits=5)
        
        for model_name in ['Ridge_a0.001', 'Huber_e1.35', 'RF_d5']:
            model = get_all_models()[model_name]
            
            fold_results = []
            for train_idx, test_idx in tscv.split(X):
                try:
                    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
                    y_tr, y_te = y_log.iloc[train_idx], y_log.iloc[test_idx]
                    y_te_orig = y.iloc[test_idx]
                    
                    pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
                    pipe.fit(X_tr, y_tr)
                    pred = np.exp(pipe.predict(X_te)) - 1
                    
                    fold_results.append(r2_score(y_te_orig, pred))
                except:
                    pass
            
            if fold_results:
                config = {
                    'asset': asset,
                    'model': model_name,
                    'validation': 'TimeSeriesCV_5fold'
                }
                results = {
                    'R2_mean': round(np.mean(fold_results), 6),
                    'R2_std': round(np.std(fold_results), 6),
                    'R2_folds': [round(r, 6) for r in fold_results]
                }
                logger.log(config, results)
                rolling_count += 1
    
    print(f"  Phase 3 completed: {rolling_count} experiments")
    
    return exp_count + ensemble_count + rolling_count

# ============================================================================
# 메인
# ============================================================================

def main():
    logger = ExtendedExperimentLogger()
    
    total = run_extended_experiments(logger)
    
    output_path = 'data/results/extended_sci_experiments.json'
    logger.save(output_path)
    
    print("\n" + "="*80)
    print(f"Total experiments: {total}")
    print(f"Results saved to: {output_path}")
    
    # 최고 결과 출력
    print("\n=== Top 10 Results ===")
    sorted_exp = sorted(
        [e for e in logger.experiments if 'R2' in e.get('results', {})],
        key=lambda x: x['results'].get('R2', -999),
        reverse=True
    )[:10]
    
    for i, exp in enumerate(sorted_exp, 1):
        print(f"  {i}. {exp['config'].get('asset')}/{exp['config'].get('model')}: R2={exp['results']['R2']:.4f}")

if __name__ == "__main__":
    main()
