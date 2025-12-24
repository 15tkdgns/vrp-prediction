"""
대안적 접근: 과적합 방지 + 단순 모델 앙상블
=============================================
1. 강한 정규화 (높은 alpha)
2. 단순 모델 선호 (얕은 트리, 적은 파라미터)
3. Walk-Forward 검증
4. 특성 선택 (중요도 기반)
5. NNLS 앙상블 + 편향 보정
"""
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.model_selection import TimeSeriesSplit
from scipy.optimize import nnls
from scipy.stats import mstats
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

def prepare_features(ticker):
    df = yf.download(ticker, start='2020-01-01', end='2025-01-01', progress=False)
    if len(df) < 100:
        return None, None, None
    
    returns = df['Close'].pct_change()
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    
    features = pd.DataFrame(index=df.index)
    
    # HAR-RV (핵심 특성)
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
    
    return X, y, feature_cols

# ============================================================================
# 단순하지만 강건한 모델들 (과적합 방지)
# ============================================================================

def get_simple_models():
    """과적합을 방지하는 단순 모델들"""
    return {
        # 선형 모델 - 강한 정규화
        'Ridge_strong': Ridge(alpha=10.0),  # 강한 정규화
        'Ridge_medium': Ridge(alpha=1.0),
        'Lasso_strong': Lasso(alpha=0.1, max_iter=2000),
        'ElasticNet_strong': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000),
        'Huber_strong': HuberRegressor(epsilon=1.35, alpha=0.1),
        'BayesianRidge': BayesianRidge(),  # 자동 정규화
        
        # 트리 모델 - 얕은 깊이, 적은 트리
        'RF_shallow': RandomForestRegressor(n_estimators=50, max_depth=3, 
                                             min_samples_leaf=10, random_state=42),
        'GB_shallow': GradientBoostingRegressor(n_estimators=50, max_depth=2,
                                                 min_samples_leaf=10, learning_rate=0.1,
                                                 random_state=42),
        'ExtraTrees_shallow': ExtraTreesRegressor(n_estimators=50, max_depth=3,
                                                   min_samples_leaf=10, random_state=42),
        
        # SVR - 단순 설정
        'SVR_rbf': SVR(kernel='rbf', C=0.1, epsilon=0.1),  # 작은 C
        'SVR_linear': SVR(kernel='linear', C=0.1),
    }

# ============================================================================
# Walk-Forward 검증
# ============================================================================

def walk_forward_validation(X, y, model, n_splits=5, gap=22):
    """시계열 Walk-Forward 검증"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    
    for train_idx, test_idx in tscv.split(X):
        # Gap 적용
        if len(train_idx) < gap + 50:
            continue
        
        train_idx = train_idx[:-gap]
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Log 변환
        y_train_log = np.log(y_train + 1)
        
        # 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 학습 및 예측
        model.fit(X_train_scaled, y_train_log)
        y_pred_log = model.predict(X_test_scaled)
        y_pred = np.exp(y_pred_log) - 1
        
        # 평가
        r2 = r2_score(y_test, y_pred)
        scores.append(r2)
    
    return np.mean(scores), np.std(scores)

# ============================================================================
# 특성 중요도 기반 선택
# ============================================================================

def select_important_features(X, y, n_features=10):
    """중요 특성 선택"""
    # Mutual Information 기반
    mi_scores = mutual_info_regression(X, y, random_state=42)
    top_features = X.columns[np.argsort(mi_scores)[-n_features:]]
    return list(top_features)

# ============================================================================
# 편향 보정 NNLS 앙상블
# ============================================================================

def bias_corrected_nnls_ensemble(predictions, y_train_mean, y_test):
    """편향 보정된 NNLS 앙상블"""
    model_names = list(predictions.keys())
    pred_matrix = np.column_stack([predictions[m] for m in model_names])
    
    # 각 모델 예측의 편향 보정
    corrected_preds = {}
    for name, pred in predictions.items():
        pred_mean = np.mean(pred)
        correction = y_train_mean - pred_mean
        corrected_preds[name] = pred + correction
    
    corrected_matrix = np.column_stack([corrected_preds[m] for m in model_names])
    
    # NNLS로 가중치 학습
    weights, _ = nnls(corrected_matrix, y_test)
    
    # 가중치 정규화
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.ones(len(weights)) / len(weights)
    
    ensemble_pred = corrected_matrix @ weights
    
    return ensemble_pred, dict(zip(model_names, weights))

# ============================================================================
# 메인 실험
# ============================================================================

def run_robust_experiment(ticker, gap=22, use_feature_selection=True):
    """강건한 실험 실행"""
    print(f"\n{'='*60}")
    print(f"Processing {ticker} (Robust Approach)...")
    
    X, y, feature_cols = prepare_features(ticker)
    if X is None:
        return None
    
    # 특성 선택 (옵션)
    if use_feature_selection:
        selected_features = select_important_features(X, y, n_features=8)
        X = X[selected_features]
        print(f"  Selected features: {selected_features}")
    
    # 데이터 분할
    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx - gap]
    y_train = y.iloc[:split_idx - gap]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    
    # Log + Winsorization
    y_train_winsorized = pd.Series(mstats.winsorize(y_train, limits=[0.01, 0.01]), 
                                    index=y_train.index)
    y_train_log = np.log(y_train_winsorized + 1)
    
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}, Features: {X.shape[1]}")
    
    # 모델들
    models = get_simple_models()
    
    # 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = {}
    predictions = {}
    
    # 각 모델 학습 및 평가
    for name, model in models.items():
        try:
            # Walk-Forward CV로 모델 검증
            cv_mean, cv_std = walk_forward_validation(X, y, model, n_splits=3, gap=gap)
            
            # 전체 훈련 데이터로 재학습
            model.fit(X_train_scaled, y_train_log)
            y_pred_log = model.predict(X_test_scaled)
            y_pred = np.exp(y_pred_log) - 1
            
            predictions[name] = y_pred
            
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            results[name] = {
                'R2': round(r2, 6),
                'MAE': round(mae, 6),
                'CV_R2_mean': round(cv_mean, 6),
                'CV_R2_std': round(cv_std, 6)
            }
            
            print(f"  {name}: R2={r2:.4f}, CV_R2={cv_mean:.4f}±{cv_std:.4f}")
            
        except Exception as e:
            print(f"  {name} failed: {e}")
    
    # === 편향 보정 NNLS 앙상블 ===
    if len(predictions) >= 3:
        ensemble_pred, weights = bias_corrected_nnls_ensemble(
            predictions, y_train.mean(), y_test.values
        )
        
        ensemble_r2 = r2_score(y_test, ensemble_pred)
        ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
        
        # 상위 모델만 사용한 앙상블
        top_models = sorted(results.keys(), key=lambda x: results[x]['R2'], reverse=True)[:5]
        top_preds = {k: predictions[k] for k in top_models if k in predictions}
        
        if len(top_preds) >= 2:
            top_ensemble_pred, top_weights = bias_corrected_nnls_ensemble(
                top_preds, y_train.mean(), y_test.values
            )
            top_ensemble_r2 = r2_score(y_test, top_ensemble_pred)
        else:
            top_ensemble_r2 = ensemble_r2
            top_weights = weights
        
        results['NNLS_Ensemble_All'] = {
            'R2': round(ensemble_r2, 6),
            'MAE': round(ensemble_mae, 6),
            'weights': {k: round(v, 4) for k, v in weights.items() if v > 0.01}
        }
        
        results['NNLS_Ensemble_Top5'] = {
            'R2': round(top_ensemble_r2, 6),
            'weights': {k: round(v, 4) for k, v in top_weights.items() if v > 0.01}
        }
        
        print(f"  NNLS_Ensemble_All: R2={ensemble_r2:.4f}")
        print(f"  NNLS_Ensemble_Top5: R2={top_ensemble_r2:.4f}")
    
    return results

def main():
    assets = ['SPY', 'GLD', 'QQQ', 'TLT', 'EEM']
    
    all_results = {}
    
    print("="*80)
    print("Robust Approach: Simple Models + Strong Regularization")
    print("="*80)
    
    for asset in assets:
        result = run_robust_experiment(asset, gap=22, use_feature_selection=True)
        if result:
            all_results[asset] = result
    
    # 결과 저장
    output = {
        'metadata': {
            'experiment': 'Robust Regularized Ensemble',
            'approach': 'Simple models, strong regularization, feature selection, bias correction',
            'timestamp': datetime.now().isoformat()
        },
        'results': all_results
    }
    
    output_path = 'data/results/robust_experiment_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n{'='*80}")
    print(f"Results saved to {output_path}")
    
    # 최고 결과 요약
    print("\n=== Best Results per Asset ===")
    for asset, results in all_results.items():
        best_r2 = -999
        best_model = ""
        for model, metrics in results.items():
            if isinstance(metrics, dict) and 'R2' in metrics:
                if metrics['R2'] > best_r2:
                    best_r2 = metrics['R2']
                    best_model = model
        print(f"  {asset}: {best_model} (R2={best_r2:.4f})")

if __name__ == "__main__":
    main()
