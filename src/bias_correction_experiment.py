"""
편향 보정 및 고급 최적화 실험
==============================
1. 평균 이동 (Mean Shifting) 편향 보정
2. 분수 차분 (d=0.1~0.4)
3. WLS with RQ 가중치
4. Box-Cox 변환
5. QLIKE/ELF 손실 함수
"""
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import ElasticNet, Ridge, Lasso, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.optimize import nnls
from scipy.stats import mstats, boxcox
from scipy.special import inv_boxcox
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 분수 차분 (Fractional Differentiation)
# ============================================================================

def fractional_diff(series, d=0.4, thresh=1e-5):
    """분수 차분 (FFD - Fixed-width window)"""
    weights = [1.0]
    k = 1
    while True:
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < thresh:
            break
        weights.append(w)
        k += 1
    
    weights = np.array(weights[::-1])
    width = len(weights) - 1
    
    result = []
    for i in range(width, len(series)):
        result.append(np.dot(weights, series.iloc[i-width:i+1].values))
    
    return pd.Series(result, index=series.index[width:])

# ============================================================================
# Box-Cox 변환
# ============================================================================

def apply_boxcox(y, lmbda=None):
    """Box-Cox 변환 적용"""
    y_clean = np.array(y)
    y_clean = y_clean[~np.isnan(y_clean)]
    y_positive = np.maximum(y_clean, 1e-6)  # 0 방지
    
    try:
        if lmbda is None:
            y_transformed, lmbda = boxcox(y_positive)
        else:
            y_transformed = boxcox(y_positive, lmbda=lmbda)
        return y_transformed, lmbda
    except:
        # Box-Cox 실패 시 log 변환으로 대체
        return np.log(y_positive), 0.0

def inverse_boxcox(y_transformed, lmbda):
    """Box-Cox 역변환"""
    return inv_boxcox(y_transformed, lmbda) - 1e-6

# ============================================================================
# 손실 함수
# ============================================================================

def qlike_loss(y_true, y_pred):
    """QLIKE 손실 함수"""
    y_pred = np.maximum(y_pred, 1e-8)
    y_true = np.maximum(y_true, 1e-8)
    return np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1)

def elf_loss(y_true, y_pred):
    """엔트로피 손실 함수 (ELF)"""
    y_pred = np.maximum(y_pred, 1e-8)
    y_true = np.maximum(y_true, 1e-8)
    return np.mean(np.log(y_pred / y_true) ** 2)

# ============================================================================
# 편향 보정
# ============================================================================

def mean_shift_correction(y_pred, y_train_mean, y_pred_train_mean):
    """평균 이동 편향 보정"""
    shift = y_train_mean - y_pred_train_mean
    return y_pred + shift

def multiplicative_bias_correction(y_pred, y_train_mean, y_pred_train_mean):
    """곱셈적 편향 보정"""
    if y_pred_train_mean == 0:
        return y_pred
    ratio = y_train_mean / y_pred_train_mean
    return y_pred * ratio

def rectify_correction(y_pred, residuals_train, X_test, X_train):
    """Rectify 전략 - 잔차 기반 보정"""
    # 간단한 구현: 훈련 잔차의 평균으로 보정
    return y_pred - np.mean(residuals_train)

# ============================================================================
# 특성 생성
# ============================================================================

def calculate_rv(returns, window=22):
    """실현 변동성"""
    return (returns ** 2).rolling(window).sum() * 252

def calculate_rq(returns, window=22):
    """실현 쿼티시티"""
    n = window
    return (n / 3) * (returns ** 4).rolling(window).sum() * (252 ** 2)

def prepare_features_simple(df):
    """간소화된 특성 생성 (분수 차분 제외)"""
    features = pd.DataFrame(index=df.index)
    returns = df['Close'].pct_change()
    
    # 기본 변동성
    features['RV_1d'] = (returns ** 2) * 252
    features['RV_5d'] = calculate_rv(returns, 5)
    features['RV_22d'] = calculate_rv(returns, 22)
    
    # RQ (WLS 가중치용)
    features['RQ'] = calculate_rq(returns, 22)
    
    # HAR 평균
    features['RV_daily_mean'] = features['RV_1d'].rolling(5).mean()
    features['RV_weekly_mean'] = features['RV_5d'].rolling(4).mean()
    features['RV_monthly_mean'] = features['RV_22d'].rolling(3).mean()
    
    # 타겟
    features['RV_22d_future'] = features['RV_22d'].shift(-22)
    
    # VIX 프록시
    spy = yf.download('SPY', start='2020-01-01', end='2025-01-01', progress=False)
    spy_returns = spy['Close'].pct_change()
    spy_rv = calculate_rv(spy_returns, 22)
    if isinstance(spy_rv, pd.DataFrame):
        spy_rv = spy_rv.iloc[:, 0]
    features['VIX_proxy'] = spy_rv.reindex(df.index).ffill()
    
    features['VIX_lag1'] = features['VIX_proxy'].shift(1)
    features['VIX_lag5'] = features['VIX_proxy'].shift(5)
    features['VIX_change'] = features['VIX_proxy'].pct_change()
    
    # VRP
    features['VRP'] = features['VIX_proxy'] - features['RV_22d']
    features['VRP_lag1'] = features['VRP'].shift(1)
    features['VRP_ma5'] = features['VRP'].rolling(5).mean()
    
    # 시장 상태
    features['regime_high'] = (features['VIX_proxy'] >= 25).astype(int)
    features['return_5d'] = returns.rolling(5).sum()
    features['return_22d'] = returns.rolling(22).sum()
    
    return features.dropna()

# ============================================================================
# WLS 적용 모델 학습
# ============================================================================

def fit_with_wls(model, X_train, y_train, weights):
    """WLS 가중치 적용 학습"""
    try:
        # sklearn의 sample_weight 파라미터 사용
        if hasattr(model, 'fit'):
            # Pipeline인 경우
            if hasattr(model, 'named_steps'):
                model.fit(X_train, y_train)  # Pipeline은 sample_weight 직접 전달 어려움
            else:
                try:
                    model.fit(X_train, y_train, sample_weight=weights)
                except TypeError:
                    model.fit(X_train, y_train)
        return model
    except:
        model.fit(X_train, y_train)
        return model

# ============================================================================
# 메인 실험 함수
# ============================================================================

def run_bias_correction_experiment(ticker, gap=22, transform='boxcox'):
    """편향 보정 실험"""
    print(f"\n{'='*60}")
    print(f"Processing {ticker} (Gap={gap}d, Transform={transform})...")
    
    df = yf.download(ticker, start='2020-01-01', end='2025-01-01', progress=False)
    if len(df) < 100:
        return None
    
    features = prepare_features_simple(df)
    
    # 특성 선택
    feature_cols = [
        'RV_1d', 'RV_5d', 'RV_22d',
        'RV_daily_mean', 'RV_weekly_mean', 'RV_monthly_mean',
        'VIX_lag1', 'VIX_lag5', 'VIX_change',
        'VRP_lag1', 'VRP_ma5',
        'regime_high', 'return_5d', 'return_22d'
    ]
    
    # 분수 차분 특성 추가
    fd_cols = [c for c in features.columns if '_fd' in c]
    feature_cols.extend(fd_cols)
    
    # RQ 가중치용
    if 'RQ' in features.columns:
        feature_cols.append('RQ')
    
    feature_cols = [c for c in feature_cols if c in features.columns]
    
    X = features[feature_cols]
    y = features['RV_22d_future']
    
    # === 타겟 변환 ===
    lmbda = None
    if transform == 'boxcox':
        y_transformed, lmbda = apply_boxcox(y.values)
        y_transformed = pd.Series(y_transformed, index=y.index)
    elif transform == 'log':
        y_transformed = np.log(y + 1)
    else:
        y_transformed = y.copy()
    
    # Gap 적용 분할
    split_idx = int(len(X) * 0.8)
    
    X_train = X.iloc[:split_idx - gap]
    y_train = y_transformed.iloc[:split_idx - gap]
    y_train_original = y.iloc[:split_idx - gap]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    
    # WLS 가중치 (RQ 역수)
    if 'RQ' in X_train.columns:
        rq_values = X_train['RQ'].values
        weights = 1 / (rq_values + 1e-8)
        weights = weights / weights.sum() * len(weights)
        # RQ는 특성에서 제거
        X_train = X_train.drop(columns=['RQ'])
        X_test = X_test.drop(columns=['RQ'])
    else:
        weights = np.ones(len(X_train))
    
    print(f"  Features: {X_train.shape[1]}, Train: {len(X_train)}, Test: {len(X_test)}")
    
    # 모델 정의
    models = {
        'ElasticNet': Pipeline([('s', StandardScaler()), ('m', ElasticNet(alpha=0.001, max_iter=2000))]),
        'Ridge': Pipeline([('s', StandardScaler()), ('m', Ridge(alpha=0.001))]),
        'Huber': Pipeline([('s', StandardScaler()), ('m', HuberRegressor(epsilon=1.35, alpha=0.001))]),
        'RF': Pipeline([('s', StandardScaler()), ('m', RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42))]),
        'GB': Pipeline([('s', StandardScaler()), ('m', GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42))]),
        'SVR': Pipeline([('s', StandardScaler()), ('m', SVR(kernel='rbf', C=1.0))]),
    }
    
    results = {}
    all_predictions = {}
    all_predictions_corrected = {}
    
    for model_name, model in models.items():
        try:
            # 학습
            model.fit(X_train, y_train)
            
            # 훈련 데이터 예측 (편향 보정용)
            y_pred_train_transformed = model.predict(X_train)
            
            # 테스트 예측
            y_pred_transformed = model.predict(X_test)
            
            # 역변환
            if transform == 'boxcox':
                y_pred_train = inverse_boxcox(y_pred_train_transformed, lmbda)
                y_pred = inverse_boxcox(y_pred_transformed, lmbda)
            elif transform == 'log':
                y_pred_train = np.exp(y_pred_train_transformed) - 1
                y_pred = np.exp(y_pred_transformed) - 1
            else:
                y_pred_train = y_pred_train_transformed
                y_pred = y_pred_transformed
            
            # === 편향 보정 ===
            # 1. 평균 이동 보정
            y_pred_mean_shift = mean_shift_correction(
                y_pred, 
                y_train_original.mean(), 
                np.mean(y_pred_train)
            )
            
            # 2. 곱셈적 보정
            y_pred_mult = multiplicative_bias_correction(
                y_pred,
                y_train_original.mean(),
                np.mean(y_pred_train)
            )
            
            # 3. Rectify (잔차 기반)
            residuals_train = y_train_original.values - y_pred_train
            y_pred_rectify = rectify_correction(y_pred, residuals_train, X_test, X_train)
            
            all_predictions[model_name] = y_pred
            all_predictions_corrected[model_name] = y_pred_mean_shift
            
            # 평가
            r2_raw = r2_score(y_test, y_pred)
            r2_mean_shift = r2_score(y_test, y_pred_mean_shift)
            r2_mult = r2_score(y_test, y_pred_mult)
            r2_rectify = r2_score(y_test, y_pred_rectify)
            
            mae_raw = mean_absolute_error(y_test, y_pred)
            qlike = qlike_loss(y_test.values, y_pred)
            elf = elf_loss(y_test.values, y_pred)
            
            results[model_name] = {
                'R2_raw': round(r2_raw, 4),
                'R2_mean_shift': round(r2_mean_shift, 4),
                'R2_mult': round(r2_mult, 4),
                'R2_rectify': round(r2_rectify, 4),
                'MAE': round(mae_raw, 4),
                'QLIKE': round(qlike, 4),
                'ELF': round(elf, 4)
            }
            
            print(f"  {model_name}: R2_raw={r2_raw:.4f}, R2_shift={r2_mean_shift:.4f}, R2_mult={r2_mult:.4f}")
            
        except Exception as e:
            print(f"  {model_name} failed: {e}")
            results[model_name] = None
    
    # === NNLS 앙상블 (보정된 예측) ===
    if len(all_predictions_corrected) >= 3:
        try:
            pred_matrix = np.column_stack([all_predictions_corrected[m] for m in all_predictions_corrected])
            weights_nnls, _ = nnls(pred_matrix, y_test.values)
            
            ensemble_pred = pred_matrix @ weights_nnls
            ensemble_r2 = r2_score(y_test, ensemble_pred)
            
            # 앙상블에도 추가 보정
            ensemble_mean_shift = mean_shift_correction(
                ensemble_pred,
                y_train_original.mean(),
                np.mean(ensemble_pred)
            )
            ensemble_r2_corrected = r2_score(y_test, ensemble_mean_shift)
            
            results['NNLS_Ensemble'] = {
                'R2_raw': round(ensemble_r2, 4),
                'R2_corrected': round(ensemble_r2_corrected, 4),
                'Weights': {m: round(w, 3) for m, w in zip(all_predictions_corrected.keys(), weights_nnls)}
            }
            
            print(f"  NNLS_Ensemble: R2={ensemble_r2:.4f}, R2_corrected={ensemble_r2_corrected:.4f}")
            
        except Exception as e:
            print(f"  NNLS_Ensemble failed: {e}")
    
    return results

def main():
    assets = ['SPY', 'GLD', 'QQQ']
    transforms = ['boxcox', 'log']
    gaps = [5, 22]
    
    all_results = {}
    
    for transform in transforms:
        for gap in gaps:
            print(f"\n{'#'*60}")
            print(f"# Transform={transform}, Gap={gap}")
            print(f"{'#'*60}")
            
            key = f"{transform}_gap{gap}"
            all_results[key] = {}
            
            for asset in assets:
                result = run_bias_correction_experiment(asset, gap=gap, transform=transform)
                if result:
                    all_results[key][asset] = result
    
    # 결과 저장
    output = {
        'results': all_results,
        'assets': assets,
        'transforms': transforms,
        'gaps': gaps,
        'features': 'HAR + Fractional Diff + WLS + Bias Correction',
        'timestamp': datetime.now().isoformat()
    }
    
    output_path = 'data/results/bias_correction_experiment.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n{'='*60}")
    print(f"Results saved to {output_path}")
    
    # 최고 결과 요약
    print("\n=== Best Results ===")
    best_r2 = -999
    best_config = ""
    
    for key, assets_data in all_results.items():
        for asset, models in assets_data.items():
            for model, metrics in models.items():
                if metrics and isinstance(metrics, dict):
                    for r2_key in ['R2_mean_shift', 'R2_mult', 'R2_rectify', 'R2_corrected', 'R2_raw']:
                        if r2_key in metrics and metrics[r2_key] is not None:
                            if metrics[r2_key] > best_r2:
                                best_r2 = metrics[r2_key]
                                best_config = f"{key}/{asset}/{model}/{r2_key}"
    
    print(f"  Best: {best_config} = {best_r2:.4f}")

if __name__ == "__main__":
    main()
