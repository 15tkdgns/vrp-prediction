"""
R² 개선 실험 스크립트
======================
4가지 핵심 전략 적용:
1. 타겟 변수 변환 (Log, Winsorization)
2. 피처 엔지니어링 (HAR-RV, 상호작용)
3. 모델 구조 (Huber Loss, Rolling CV)
4. 다중 예측 기간 앙상블
"""
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import ElasticNet, Ridge, Lasso, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import mstats
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def calculate_rv(returns, window=22):
    """실현 변동성 계산"""
    return returns.rolling(window=window).std() * np.sqrt(252) * 100

def prepare_enhanced_features(df):
    """향상된 특성 생성 (HAR-RV 구조, 상호작용 변수)"""
    features = pd.DataFrame(index=df.index)
    returns = df['Close'].pct_change()
    
    # === 1. HAR-RV 구조 (일간, 주간, 월간) ===
    features['RV_1d'] = returns.abs() * np.sqrt(252) * 100
    features['RV_5d'] = calculate_rv(returns, 5)    # 주간
    features['RV_22d'] = calculate_rv(returns, 22)  # 월간
    
    # HAR-RV: 일/주/월 변동성 평균
    features['RV_daily_mean'] = features['RV_1d'].rolling(5).mean()
    features['RV_weekly_mean'] = features['RV_5d'].rolling(4).mean()  # 4주 평균
    features['RV_monthly_mean'] = features['RV_22d'].rolling(3).mean()  # 3개월 평균
    
    # === 2. 다중 예측 기간 타겟 ===
    features['RV_5d_future'] = features['RV_5d'].shift(-5)
    features['RV_10d_future'] = features['RV_5d'].shift(-10)
    features['RV_22d_future'] = features['RV_22d'].shift(-22)
    
    # === 3. VIX 프록시 ===
    spy = yf.download('SPY', start='2020-01-01', end='2025-01-01', progress=False)
    spy_returns = spy['Close'].pct_change()
    spy_vix = calculate_rv(spy_returns, 22)
    features['VIX'] = spy_vix.reindex(df.index).ffill()
    
    # VIX lag 특성
    features['VIX_lag1'] = features['VIX'].shift(1)
    features['VIX_lag5'] = features['VIX'].shift(5)
    features['VIX_change'] = features['VIX'].pct_change()
    features['VIX_change_5d'] = features['VIX'].pct_change(5)
    
    # === 4. VRP 관련 ===
    features['VRP'] = features['VIX'] - features['RV_22d']
    features['VRP_lag1'] = features['VRP'].shift(1)
    features['VRP_lag5'] = features['VRP'].shift(5)
    features['VRP_ma5'] = features['VRP'].rolling(5).mean()
    
    # === 5. 시장 상태 ===
    features['regime_high'] = (features['VIX'] >= 25).astype(int)
    features['regime_crisis'] = (features['VIX'] >= 35).astype(int)
    features['return_5d'] = returns.rolling(5).sum()
    features['return_22d'] = returns.rolling(22).sum()
    
    # === 6. 상호작용 변수 (비선형) ===
    features['VIX_x_return'] = features['VIX'] * features['return_5d']
    features['VIX_x_VRP'] = features['VIX'] * features['VRP_lag1']
    features['RV_ratio'] = features['RV_5d'] / (features['RV_22d'] + 0.01)  # 단기/장기 비율
    
    # === 7. 변동성 변화율 ===
    features['RV_momentum'] = features['RV_22d'].pct_change(5)
    features['VIX_momentum'] = features['VIX'].pct_change(5)
    
    return features.dropna()

def apply_target_engineering(y, method='log'):
    """타겟 변환 (Log, Winsorization)"""
    if method == 'log':
        # 로그 변환 (음수 방지를 위해 shift)
        y_transformed = np.log1p(y)
    elif method == 'winsorize':
        # 상하위 1% Winsorization
        y_transformed = pd.Series(mstats.winsorize(y, limits=[0.01, 0.01]))
        y_transformed.index = y.index
    elif method == 'both':
        # Winsorization 후 로그 변환
        y_winsorized = pd.Series(mstats.winsorize(y, limits=[0.01, 0.01]))
        y_winsorized.index = y.index
        y_transformed = np.log1p(y_winsorized)
    else:
        y_transformed = y
    return y_transformed

def inverse_transform(y_pred, method='log'):
    """역변환"""
    if method == 'log' or method == 'both':
        return np.expm1(y_pred)
    return y_pred

def run_improved_experiment(ticker):
    """향상된 실험 실행"""
    print(f"\n{'='*60}")
    print(f"Processing {ticker} with improved strategies...")
    
    # 데이터 다운로드
    df = yf.download(ticker, start='2020-01-01', end='2025-01-01', progress=False)
    if len(df) < 100:
        return None
    
    # 향상된 특성 생성
    features = prepare_enhanced_features(df)
    
    # === 특성 목록 (향상된 버전) ===
    feature_cols = [
        # HAR-RV
        'RV_1d', 'RV_5d', 'RV_22d', 
        'RV_daily_mean', 'RV_weekly_mean', 'RV_monthly_mean',
        # VIX
        'VIX_lag1', 'VIX_lag5', 'VIX_change', 'VIX_change_5d',
        # VRP
        'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
        # 시장
        'regime_high', 'regime_crisis', 'return_5d', 'return_22d',
        # 상호작용
        'VIX_x_return', 'VIX_x_VRP', 'RV_ratio',
        # 모멘텀
        'RV_momentum', 'VIX_momentum'
    ]
    
    X = features[feature_cols]
    
    results = {}
    
    # === 전략별 실험 ===
    strategies = {
        'baseline': {'target': 'RV_22d_future', 'transform': 'none'},
        'log_transform': {'target': 'RV_22d_future', 'transform': 'log'},
        'winsorize': {'target': 'RV_22d_future', 'transform': 'winsorize'},
        'log+winsorize': {'target': 'RV_22d_future', 'transform': 'both'},
        'short_horizon_5d': {'target': 'RV_5d_future', 'transform': 'log'},
        'short_horizon_10d': {'target': 'RV_10d_future', 'transform': 'log'},
    }
    
    for strategy_name, config in strategies.items():
        target_col = config['target']
        transform = config['transform']
        
        if target_col not in features.columns:
            continue
            
        y = features[target_col].copy()
        
        # 타겟 변환
        y_transformed = apply_target_engineering(y, transform)
        
        # Gap 설정
        if '5d' in target_col:
            gap = 5
        elif '10d' in target_col:
            gap = 10
        else:
            gap = 22
        
        # Train/Test 분할
        split_idx = int(len(X) * 0.8)
        X_train = X.iloc[:split_idx - gap]
        y_train = y_transformed.iloc[:split_idx - gap]
        X_test = X.iloc[split_idx:]
        y_test_transformed = y_transformed.iloc[split_idx:]
        y_test_original = y.iloc[split_idx:]
        
        # 모델 정의 (Huber Loss 포함)
        models = {
            'ElasticNet': Pipeline([
                ('scaler', StandardScaler()),
                ('model', ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42))  # 규제 완화
            ]),
            'Huber': Pipeline([
                ('scaler', StandardScaler()),
                ('model', HuberRegressor(epsilon=1.35, alpha=0.01))
            ]),
            'MLP': Pipeline([
                ('scaler', StandardScaler()),
                ('model', MLPRegressor(hidden_layer_sizes=(64,), max_iter=500, 
                                       random_state=42, early_stopping=True))
            ]),
        }
        
        strategy_results = {}
        
        for model_name, model in models.items():
            try:
                model.fit(X_train, y_train)
                y_pred_transformed = model.predict(X_test)
                
                # 역변환
                y_pred = inverse_transform(y_pred_transformed, transform)
                
                # 평가 (원래 스케일)
                r2 = r2_score(y_test_original, y_pred)
                mae = mean_absolute_error(y_test_original, y_pred)
                
                strategy_results[model_name] = {
                    'R2': round(r2, 4),
                    'MAE': round(mae, 4)
                }
                
                print(f"  {strategy_name} + {model_name}: R2={r2:.4f}, MAE={mae:.2f}")
                
            except Exception as e:
                print(f"  {strategy_name} + {model_name} failed: {e}")
                strategy_results[model_name] = {'R2': None, 'MAE': None}
        
        results[strategy_name] = strategy_results
    
    return results

def run_stacking_experiment(ticker):
    """Stacking 앙상블 실험 (다중 기간 예측)"""
    print(f"\n{'='*60}")
    print(f"Stacking Ensemble for {ticker}...")
    
    df = yf.download(ticker, start='2020-01-01', end='2025-01-01', progress=False)
    if len(df) < 100:
        return None
    
    features = prepare_enhanced_features(df)
    
    feature_cols = [
        'RV_1d', 'RV_5d', 'RV_22d', 
        'RV_daily_mean', 'RV_weekly_mean', 'RV_monthly_mean',
        'VIX_lag1', 'VIX_lag5', 'VIX_change',
        'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
        'regime_high', 'return_5d', 'return_22d',
        'VIX_x_return', 'RV_ratio'
    ]
    
    X = features[feature_cols]
    y = features['RV_22d_future']
    y_log = np.log1p(y)
    
    gap = 22
    split_idx = int(len(X) * 0.8)
    
    X_train = X.iloc[:split_idx - gap]
    y_train = y_log.iloc[:split_idx - gap]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    
    # Stacking 앙상블
    estimators = [
        ('elasticnet', Pipeline([('scaler', StandardScaler()), 
                                  ('model', ElasticNet(alpha=0.01, l1_ratio=0.5))])),
        ('huber', Pipeline([('scaler', StandardScaler()), 
                            ('model', HuberRegressor(epsilon=1.35))])),
        ('mlp', Pipeline([('scaler', StandardScaler()), 
                          ('model', MLPRegressor(hidden_layer_sizes=(32,), max_iter=300))]))
    ]
    
    stacking = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=TimeSeriesSplit(n_splits=3)
    )
    
    try:
        stacking.fit(X_train, y_train)
        y_pred_log = stacking.predict(X_test)
        y_pred = np.expm1(y_pred_log)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"  Stacking Ensemble: R2={r2:.4f}, MAE={mae:.2f}")
        return {'R2': round(r2, 4), 'MAE': round(mae, 4)}
        
    except Exception as e:
        print(f"  Stacking failed: {e}")
        return None

def main():
    assets = ['SPY', 'GLD', 'QQQ']  # 주요 자산만
    
    all_results = {}
    stacking_results = {}
    
    for asset in assets:
        result = run_improved_experiment(asset)
        if result:
            all_results[asset] = result
        
        stacking = run_stacking_experiment(asset)
        if stacking:
            stacking_results[asset] = stacking
    
    # 결과 저장
    output = {
        'strategy_results': all_results,
        'stacking_results': stacking_results,
        'assets': assets,
        'strategies': ['baseline', 'log_transform', 'winsorize', 'log+winsorize', 
                      'short_horizon_5d', 'short_horizon_10d'],
        'timestamp': datetime.now().isoformat()
    }
    
    output_path = 'data/results/improved_model_comparison.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Results saved to {output_path}")
    
    # 요약 출력
    print("\n=== Best Results Summary ===")
    for asset in assets:
        if asset in all_results:
            print(f"\n{asset}:")
            best_r2 = -999
            best_config = ""
            for strategy, models in all_results[asset].items():
                for model, metrics in models.items():
                    if metrics.get('R2') is not None and metrics['R2'] > best_r2:
                        best_r2 = metrics['R2']
                        best_config = f"{strategy} + {model}"
            print(f"  Best: {best_config} (R2={best_r2:.4f})")
            if asset in stacking_results:
                print(f"  Stacking: R2={stacking_results[asset]['R2']:.4f}")

if __name__ == "__main__":
    main()
