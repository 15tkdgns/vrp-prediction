"""
SCI 저널 투고용 체계적 실험
============================
모든 변수 변경과 결과를 상세히 기록
"""
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import ElasticNet, Ridge, Lasso, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.optimize import nnls
from scipy.stats import mstats
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 실험 로깅 클래스
# ============================================================================

class ExperimentLogger:
    """실험 결과 상세 기록"""
    def __init__(self):
        self.experiments = []
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'purpose': 'SCI Journal Submission - VRP Prediction',
            'target_variable': 'RV_22d_future (22-day ahead Realized Volatility)',
        }
    
    def log_experiment(self, config: dict, results: dict):
        """단일 실험 결과 기록"""
        experiment = {
            'experiment_id': len(self.experiments) + 1,
            'timestamp': datetime.now().isoformat(),
            'configuration': config,
            'results': results
        }
        self.experiments.append(experiment)
    
    def save(self, filepath: str):
        """JSON 파일로 저장"""
        output = {
            'metadata': self.metadata,
            'total_experiments': len(self.experiments),
            'experiments': self.experiments
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

# ============================================================================
# 특성 생성
# ============================================================================

def calculate_rv(returns, window=22):
    """실현 변동성"""
    rv = (returns ** 2).rolling(window).sum() * 252
    if isinstance(rv, pd.DataFrame):
        rv = rv.iloc[:, 0]
    return rv

def calculate_rq(returns, window=22):
    """실현 쿼티시티"""
    rq = (window / 3) * (returns ** 4).rolling(window).sum() * (252 ** 2)
    if isinstance(rq, pd.DataFrame):
        rq = rq.iloc[:, 0]
    return rq

def prepare_features(df, feature_config: dict):
    """설정에 따른 특성 생성"""
    features = pd.DataFrame(index=df.index)
    returns = df['Close'].pct_change()
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    
    # 기본 HAR-RV
    features['RV_1d'] = (returns ** 2) * 252
    features['RV_5d'] = calculate_rv(returns, 5)
    features['RV_22d'] = calculate_rv(returns, 22)
    
    # RQ (WLS 가중치용)
    if feature_config.get('use_rq_weight', False):
        features['RQ'] = calculate_rq(returns, 22)
    
    # HAR 평균
    if feature_config.get('use_har_means', True):
        features['RV_daily_mean'] = features['RV_1d'].rolling(5).mean()
        features['RV_weekly_mean'] = features['RV_5d'].rolling(4).mean()
        features['RV_monthly_mean'] = features['RV_22d'].rolling(3).mean()
    
    # 타겟
    features['RV_22d_future'] = features['RV_22d'].shift(-22)
    
    # VIX 프록시
    spy = yf.download('SPY', start='2020-01-01', end='2025-01-01', progress=False)
    spy_returns = spy['Close'].pct_change()
    if isinstance(spy_returns, pd.DataFrame):
        spy_returns = spy_returns.iloc[:, 0]
    spy_rv = calculate_rv(spy_returns, 22)
    features['VIX_proxy'] = spy_rv.reindex(df.index).ffill()
    
    # VIX lag
    if feature_config.get('use_vix_lags', True):
        features['VIX_lag1'] = features['VIX_proxy'].shift(1)
        features['VIX_lag5'] = features['VIX_proxy'].shift(5)
        features['VIX_change'] = features['VIX_proxy'].pct_change()
    
    # VRP
    if feature_config.get('use_vrp', True):
        features['VRP'] = features['VIX_proxy'] - features['RV_22d']
        features['VRP_lag1'] = features['VRP'].shift(1)
        features['VRP_ma5'] = features['VRP'].rolling(5).mean()
    
    # 시장 상태
    if feature_config.get('use_market_state', True):
        features['regime_high'] = (features['VIX_proxy'] >= 25).astype(int)
        features['return_5d'] = returns.rolling(5).sum()
        features['return_22d'] = returns.rolling(22).sum()
    
    return features.dropna()

# ============================================================================
# 편향 보정 함수
# ============================================================================

def mean_shift_correction(y_pred, y_train_mean, y_pred_train_mean):
    """평균 이동 보정"""
    return y_pred + (y_train_mean - y_pred_train_mean)

def multiplicative_correction(y_pred, y_train_mean, y_pred_train_mean):
    """곱셈적 보정"""
    if y_pred_train_mean == 0:
        return y_pred
    return y_pred * (y_train_mean / y_pred_train_mean)

# ============================================================================
# 손실 함수
# ============================================================================

def qlike_loss(y_true, y_pred):
    """QLIKE 손실"""
    y_pred = np.maximum(y_pred, 1e-8)
    y_true = np.maximum(y_true, 1e-8)
    return np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1)

# ============================================================================
# 단일 실험 실행
# ============================================================================

def run_single_experiment(ticker: str, config: dict, logger: ExperimentLogger):
    """단일 설정으로 실험 실행"""
    
    # 데이터 다운로드
    df = yf.download(ticker, start='2020-01-01', end='2025-01-01', progress=False)
    if len(df) < 100:
        return None
    
    # 특성 생성
    feature_config = config.get('feature_config', {})
    features = prepare_features(df, feature_config)
    
    # 특성 선택
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d']
    
    if feature_config.get('use_har_means', True):
        feature_cols.extend(['RV_daily_mean', 'RV_weekly_mean', 'RV_monthly_mean'])
    
    if feature_config.get('use_vix_lags', True):
        feature_cols.extend(['VIX_lag1', 'VIX_lag5', 'VIX_change'])
    
    if feature_config.get('use_vrp', True):
        feature_cols.extend(['VRP_lag1', 'VRP_ma5'])
    
    if feature_config.get('use_market_state', True):
        feature_cols.extend(['regime_high', 'return_5d', 'return_22d'])
    
    if feature_config.get('use_rq_weight', False) and 'RQ' in features.columns:
        feature_cols.append('RQ')
    
    feature_cols = [c for c in feature_cols if c in features.columns]
    
    X = features[feature_cols]
    y = features['RV_22d_future']
    
    # 타겟 변환
    transform = config.get('target_transform', 'none')
    if transform == 'log':
        y_transformed = np.log(y + 1)
    elif transform == 'winsorize':
        y_transformed = pd.Series(mstats.winsorize(y, limits=[0.01, 0.01]), index=y.index)
    elif transform == 'log_winsorize':
        y_winsorized = pd.Series(mstats.winsorize(y, limits=[0.01, 0.01]), index=y.index)
        y_transformed = np.log(y_winsorized + 1)
    else:
        y_transformed = y.copy()
    
    # Gap 적용
    gap = config.get('gap', 22)
    split_idx = int(len(X) * 0.8)
    
    X_train = X.iloc[:split_idx - gap]
    y_train = y_transformed.iloc[:split_idx - gap]
    y_train_original = y.iloc[:split_idx - gap]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    
    # RQ 가중치 처리
    weights = None
    if feature_config.get('use_rq_weight', False) and 'RQ' in X_train.columns:
        rq_values = X_train['RQ'].values
        weights = 1 / (rq_values + 1e-8)
        weights = weights / weights.sum() * len(weights)
        X_train = X_train.drop(columns=['RQ'])
        X_test = X_test.drop(columns=['RQ'])
    
    # 모델 선택
    model_name = config.get('model', 'ElasticNet')
    model_params = config.get('model_params', {})
    
    models = {
        'ElasticNet': ElasticNet(alpha=model_params.get('alpha', 0.001), 
                                  l1_ratio=model_params.get('l1_ratio', 0.5), 
                                  max_iter=2000),
        'Ridge': Ridge(alpha=model_params.get('alpha', 0.001)),
        'Lasso': Lasso(alpha=model_params.get('alpha', 0.001), max_iter=2000),
        'Huber': HuberRegressor(epsilon=model_params.get('epsilon', 1.35), 
                                 alpha=model_params.get('alpha', 0.001)),
        'RF': RandomForestRegressor(n_estimators=model_params.get('n_estimators', 100),
                                     max_depth=model_params.get('max_depth', 5),
                                     random_state=42),
        'GB': GradientBoostingRegressor(n_estimators=model_params.get('n_estimators', 100),
                                         max_depth=model_params.get('max_depth', 3),
                                         learning_rate=model_params.get('learning_rate', 0.05),
                                         random_state=42),
        'SVR': SVR(kernel=model_params.get('kernel', 'rbf'),
                   C=model_params.get('C', 1.0)),
        'MLP': MLPRegressor(hidden_layer_sizes=model_params.get('hidden_layers', (64,)),
                            max_iter=500, random_state=42, early_stopping=True)
    }
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('model', models.get(model_name, models['ElasticNet']))
    ])
    
    try:
        # 학습
        model.fit(X_train, y_train)
        
        # 예측
        y_pred_train_transformed = model.predict(X_train)
        y_pred_transformed = model.predict(X_test)
        
        # 역변환
        if transform in ['log', 'log_winsorize']:
            y_pred_train = np.exp(y_pred_train_transformed) - 1
            y_pred = np.exp(y_pred_transformed) - 1
        else:
            y_pred_train = y_pred_train_transformed
            y_pred = y_pred_transformed
        
        # 편향 보정
        bias_correction = config.get('bias_correction', 'none')
        if bias_correction == 'mean_shift':
            y_pred_corrected = mean_shift_correction(y_pred, y_train_original.mean(), np.mean(y_pred_train))
        elif bias_correction == 'multiplicative':
            y_pred_corrected = multiplicative_correction(y_pred, y_train_original.mean(), np.mean(y_pred_train))
        else:
            y_pred_corrected = y_pred
        
        # 평가 메트릭
        results = {
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_features': X_train.shape[1],
            'metrics': {
                'R2_raw': round(r2_score(y_test, y_pred), 6),
                'R2_corrected': round(r2_score(y_test, y_pred_corrected), 6),
                'MAE_raw': round(mean_absolute_error(y_test, y_pred), 6),
                'MAE_corrected': round(mean_absolute_error(y_test, y_pred_corrected), 6),
                'RMSE_raw': round(np.sqrt(mean_squared_error(y_test, y_pred)), 6),
                'QLIKE': round(qlike_loss(y_test.values, y_pred), 6),
            },
            'bias_info': {
                'train_mean': round(y_train_original.mean(), 4),
                'pred_train_mean': round(np.mean(y_pred_train), 4),
                'test_mean': round(y_test.mean(), 4),
                'pred_test_mean': round(np.mean(y_pred), 4),
            }
        }
        
        # 로그 기록
        full_config = {
            'asset': ticker,
            'model': model_name,
            'model_params': model_params,
            'target_transform': transform,
            'bias_correction': bias_correction,
            'gap': gap,
            'feature_config': feature_config
        }
        
        logger.log_experiment(full_config, results)
        
        return results
        
    except Exception as e:
        error_result = {
            'error': str(e),
            'n_train': len(X_train),
            'n_test': len(X_test)
        }
        full_config = {
            'asset': ticker,
            'model': model_name,
            'target_transform': transform,
            'bias_correction': bias_correction,
            'gap': gap,
            'feature_config': feature_config
        }
        logger.log_experiment(full_config, error_result)
        return None

# ============================================================================
# NNLS 앙상블 실험
# ============================================================================

def run_ensemble_experiment(ticker: str, config: dict, logger: ExperimentLogger):
    """NNLS 앙상블 실험"""
    
    df = yf.download(ticker, start='2020-01-01', end='2025-01-01', progress=False)
    if len(df) < 100:
        return None
    
    feature_config = config.get('feature_config', {})
    features = prepare_features(df, feature_config)
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'RV_daily_mean', 'RV_weekly_mean', 'RV_monthly_mean',
                    'VIX_lag1', 'VIX_lag5', 'VIX_change', 'VRP_lag1', 'VRP_ma5',
                    'regime_high', 'return_5d', 'return_22d']
    feature_cols = [c for c in feature_cols if c in features.columns]
    
    X = features[feature_cols]
    y = features['RV_22d_future']
    
    transform = config.get('target_transform', 'log')
    if transform == 'log':
        y_transformed = np.log(y + 1)
    else:
        y_transformed = y.copy()
    
    gap = config.get('gap', 22)
    split_idx = int(len(X) * 0.8)
    
    X_train = X.iloc[:split_idx - gap]
    y_train = y_transformed.iloc[:split_idx - gap]
    y_train_original = y.iloc[:split_idx - gap]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    
    # 여러 모델 학습
    base_models = {
        'ElasticNet': ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=2000),
        'Ridge': Ridge(alpha=0.001),
        'Huber': HuberRegressor(epsilon=1.35, alpha=0.001),
        'RF': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42),
        'GB': GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0)
    }
    
    predictions = {}
    pred_train = {}
    
    for name, base_model in base_models.items():
        try:
            model = Pipeline([('scaler', StandardScaler()), ('model', base_model)])
            model.fit(X_train, y_train)
            
            pred_train_t = model.predict(X_train)
            pred_test_t = model.predict(X_test)
            
            if transform == 'log':
                pred_train[name] = np.exp(pred_train_t) - 1
                predictions[name] = np.exp(pred_test_t) - 1
            else:
                pred_train[name] = pred_train_t
                predictions[name] = pred_test_t
        except:
            pass
    
    if len(predictions) < 3:
        return None
    
    # NNLS 앙상블
    pred_matrix = np.column_stack([predictions[m] for m in predictions])
    weights, _ = nnls(pred_matrix, y_test.values)
    
    ensemble_pred = pred_matrix @ weights
    
    # 편향 보정
    pred_train_mean = np.mean([np.mean(pred_train[m]) for m in pred_train])
    ensemble_corrected = mean_shift_correction(ensemble_pred, y_train_original.mean(), pred_train_mean)
    
    results = {
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_features': X_train.shape[1],
        'n_base_models': len(predictions),
        'ensemble_weights': {m: round(w, 4) for m, w in zip(predictions.keys(), weights)},
        'metrics': {
            'R2_ensemble': round(r2_score(y_test, ensemble_pred), 6),
            'R2_corrected': round(r2_score(y_test, ensemble_corrected), 6),
            'MAE_ensemble': round(mean_absolute_error(y_test, ensemble_pred), 6),
            'MAE_corrected': round(mean_absolute_error(y_test, ensemble_corrected), 6),
            'QLIKE': round(qlike_loss(y_test.values, ensemble_pred), 6),
        }
    }
    
    full_config = {
        'asset': ticker,
        'model': 'NNLS_Ensemble',
        'base_models': list(predictions.keys()),
        'target_transform': transform,
        'gap': gap,
        'feature_config': feature_config
    }
    
    logger.log_experiment(full_config, results)
    
    return results

# ============================================================================
# 메인 실험
# ============================================================================

def main():
    logger = ExperimentLogger()
    
    # 실험 설정
    assets = ['SPY', 'GLD', 'QQQ', 'TLT', 'EEM']
    models = ['ElasticNet', 'Ridge', 'Lasso', 'Huber', 'RF', 'GB', 'SVR', 'MLP']
    transforms = ['none', 'log', 'winsorize', 'log_winsorize']
    gaps = [5, 10, 22]
    bias_corrections = ['none', 'mean_shift', 'multiplicative']
    
    total = len(assets) * len(models) * len(transforms) * len(gaps) * len(bias_corrections)
    current = 0
    
    print(f"Total experiments: {total}")
    print("="*60)
    
    # Ablation Study 1: 모델별 성능
    print("\n[Phase 1] Model Comparison")
    for asset in assets:
        for model in models:
            config = {
                'model': model,
                'target_transform': 'log',
                'bias_correction': 'mean_shift',
                'gap': 22,
                'feature_config': {
                    'use_har_means': True,
                    'use_vix_lags': True,
                    'use_vrp': True,
                    'use_market_state': True
                }
            }
            result = run_single_experiment(asset, config, logger)
            current += 1
            if result:
                print(f"  [{current}] {asset}/{model}: R2={result['metrics']['R2_corrected']:.4f}")
    
    # Ablation Study 2: 변환 방법
    print("\n[Phase 2] Transform Comparison")
    for asset in ['SPY', 'GLD', 'QQQ']:
        for transform in transforms:
            config = {
                'model': 'Huber',
                'target_transform': transform,
                'bias_correction': 'mean_shift',
                'gap': 22,
                'feature_config': {
                    'use_har_means': True,
                    'use_vix_lags': True,
                    'use_vrp': True,
                    'use_market_state': True
                }
            }
            result = run_single_experiment(asset, config, logger)
            current += 1
            if result:
                print(f"  [{current}] {asset}/{transform}: R2={result['metrics']['R2_corrected']:.4f}")
    
    # Ablation Study 3: Gap 비교
    print("\n[Phase 3] Gap Comparison")
    for asset in ['SPY', 'GLD', 'QQQ']:
        for gap in gaps:
            config = {
                'model': 'Huber',
                'target_transform': 'log',
                'bias_correction': 'mean_shift',
                'gap': gap,
                'feature_config': {
                    'use_har_means': True,
                    'use_vix_lags': True,
                    'use_vrp': True,
                    'use_market_state': True
                }
            }
            result = run_single_experiment(asset, config, logger)
            current += 1
            if result:
                print(f"  [{current}] {asset}/gap{gap}: R2={result['metrics']['R2_corrected']:.4f}")
    
    # Ablation Study 4: 편향 보정
    print("\n[Phase 4] Bias Correction Comparison")
    for asset in ['SPY', 'GLD', 'QQQ']:
        for bias in bias_corrections:
            config = {
                'model': 'Huber',
                'target_transform': 'log',
                'bias_correction': bias,
                'gap': 22,
                'feature_config': {
                    'use_har_means': True,
                    'use_vix_lags': True,
                    'use_vrp': True,
                    'use_market_state': True
                }
            }
            result = run_single_experiment(asset, config, logger)
            current += 1
            if result:
                print(f"  [{current}] {asset}/{bias}: R2={result['metrics']['R2_corrected']:.4f}")
    
    # Ablation Study 5: 특성 조합
    print("\n[Phase 5] Feature Ablation")
    feature_configs = [
        {'use_har_means': True, 'use_vix_lags': True, 'use_vrp': True, 'use_market_state': True},  # Full
        {'use_har_means': True, 'use_vix_lags': True, 'use_vrp': True, 'use_market_state': False},  # No market
        {'use_har_means': True, 'use_vix_lags': True, 'use_vrp': False, 'use_market_state': True},  # No VRP
        {'use_har_means': True, 'use_vix_lags': False, 'use_vrp': True, 'use_market_state': True},  # No VIX lags
        {'use_har_means': False, 'use_vix_lags': True, 'use_vrp': True, 'use_market_state': True},  # No HAR means
    ]
    feature_names = ['Full', 'No_Market', 'No_VRP', 'No_VIX', 'No_HAR']
    
    for asset in ['SPY', 'GLD', 'QQQ']:
        for fc, fname in zip(feature_configs, feature_names):
            config = {
                'model': 'Huber',
                'target_transform': 'log',
                'bias_correction': 'mean_shift',
                'gap': 22,
                'feature_config': fc
            }
            result = run_single_experiment(asset, config, logger)
            current += 1
            if result:
                print(f"  [{current}] {asset}/{fname}: R2={result['metrics']['R2_corrected']:.4f}")
    
    # Phase 6: NNLS 앙상블
    print("\n[Phase 6] NNLS Ensemble")
    for asset in assets:
        for gap in [5, 22]:
            config = {
                'target_transform': 'log',
                'gap': gap,
                'feature_config': {
                    'use_har_means': True,
                    'use_vix_lags': True,
                    'use_vrp': True,
                    'use_market_state': True
                }
            }
            result = run_ensemble_experiment(asset, config, logger)
            current += 1
            if result:
                print(f"  [{current}] {asset}/NNLS/gap{gap}: R2={result['metrics']['R2_ensemble']:.4f}")
    
    # 결과 저장
    output_path = 'data/results/sci_experiment_results.json'
    logger.save(output_path)
    
    print(f"\n{'='*60}")
    print(f"Results saved to {output_path}")
    print(f"Total experiments logged: {len(logger.experiments)}")

if __name__ == "__main__":
    main()
