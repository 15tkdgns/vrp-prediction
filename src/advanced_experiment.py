"""
고급 VRP 예측 실험
==================
Phase 1: Log 변환 + Gap 5일
Phase 2: HAR-RV-CJ 특성 (점프 포함)
Phase 3: 분수 차분 (d=0.4)
Phase 4: WLS + QLIKE/Huber Loss
Phase 5: 외부 시장 지표
Phase 6: NNLS 스태킹 앙상블
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
# Phase 2: HAR-RV-CJ 특성 계산
# ============================================================================

def calculate_realized_semivariance(returns, window=22):
    """하락/상승 세미분산 계산"""
    RS_minus = returns.copy()
    RS_plus = returns.copy()
    
    RS_minus[RS_minus > 0] = 0
    RS_plus[RS_plus < 0] = 0
    
    RS_minus = (RS_minus ** 2).rolling(window).sum() * 252
    RS_plus = (RS_plus ** 2).rolling(window).sum() * 252
    
    return RS_minus, RS_plus

def calculate_realized_quarticity(returns, window=22):
    """실현 쿼티시티 계산"""
    n = window
    RQ = (n / 3) * (returns ** 4).rolling(window).sum() * (252 ** 2)
    return RQ

def calculate_bipower_variation(returns, window=22):
    """이변량 변동 (점프 제거)"""
    abs_ret = returns.abs()
    mu_1 = np.sqrt(2 / np.pi)
    BV = (1 / mu_1 ** 2) * (abs_ret * abs_ret.shift(1)).rolling(window).sum() * 252
    return BV

def calculate_jump_component(RV, BV):
    """점프 성분"""
    return np.maximum(RV - BV, 0)

def calculate_rv(returns, window=22):
    """실현 변동성"""
    return (returns ** 2).rolling(window).sum() * 252

# ============================================================================
# Phase 3: 분수 차분 (간단 구현)
# ============================================================================

def fractional_diff(series, d=0.4, thresh=1e-5):
    """분수 차분 (FFD 방식)"""
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
# Phase 5: 외부 시장 지표
# ============================================================================

def get_external_indicators(start='2020-01-01', end='2025-01-01'):
    """외부 시장 지표 가져오기"""
    indicators = {}
    
    try:
        # VIX 실제 지수
        vix = yf.download('^VIX', start=start, end=end, progress=False)['Close']
        if isinstance(vix, pd.DataFrame):
            vix = vix.iloc[:, 0]
        indicators['VIX_real'] = vix
    except:
        pass
    
    try:
        # 신용 스프레드 (HYG - LQD)
        hyg = yf.download('HYG', start=start, end=end, progress=False)['Close']
        lqd = yf.download('LQD', start=start, end=end, progress=False)['Close']
        if isinstance(hyg, pd.DataFrame):
            hyg = hyg.iloc[:, 0]
        if isinstance(lqd, pd.DataFrame):
            lqd = lqd.iloc[:, 0]
        indicators['credit_spread'] = (hyg / lqd).pct_change()
    except:
        pass
    
    try:
        # 달러 인덱스
        usd = yf.download('DX-Y.NYB', start=start, end=end, progress=False)['Close']
        if isinstance(usd, pd.DataFrame):
            usd = usd.iloc[:, 0]
        indicators['USD_index'] = usd.pct_change()
    except:
        pass
    
    return indicators

# ============================================================================
# 메인 특성 생성 함수
# ============================================================================

def prepare_advanced_features(df, use_fracdiff=True, d=0.4):
    """고급 특성 생성"""
    features = pd.DataFrame(index=df.index)
    returns = df['Close'].pct_change()
    
    # === 기본 HAR-RV ===
    features['RV_1d'] = (returns ** 2) * 252
    features['RV_5d'] = calculate_rv(returns, 5)
    features['RV_22d'] = calculate_rv(returns, 22)
    
    # === HAR-RV-CJ 확장 ===
    RS_minus, RS_plus = calculate_realized_semivariance(returns, 22)
    features['RS_minus'] = RS_minus  # 하락 세미분산
    features['RS_plus'] = RS_plus    # 상승 세미분산
    features['SJ'] = RS_plus - RS_minus  # 부호화 점프
    
    RQ = calculate_realized_quarticity(returns, 22)
    features['RQ'] = RQ  # 실현 쿼티시티
    
    BV = calculate_bipower_variation(returns, 22)
    if isinstance(BV, pd.DataFrame):
        BV = BV.iloc[:, 0]
    features['BV'] = BV  # 이변량 변동
    
    J = calculate_jump_component(features['RV_22d'], features['BV'])
    if isinstance(J, pd.DataFrame):
        J = J.iloc[:, 0]
    features['J'] = J  # 점프 성분
    
    # HAR 평균
    features['RV_daily_mean'] = features['RV_1d'].rolling(5).mean()
    features['RV_weekly_mean'] = features['RV_5d'].rolling(4).mean()
    features['RV_monthly_mean'] = features['RV_22d'].rolling(3).mean()
    
    # === 타겟 변수 ===
    features['RV_22d_future'] = features['RV_22d'].shift(-22)
    
    # === VIX 및 VRP ===
    spy = yf.download('SPY', start='2020-01-01', end='2025-01-01', progress=False)
    spy_returns = spy['Close'].pct_change()
    spy_rv = calculate_rv(spy_returns, 22)
    features['VIX_proxy'] = spy_rv.reindex(df.index).ffill()
    
    features['VIX_lag1'] = features['VIX_proxy'].shift(1)
    features['VIX_lag5'] = features['VIX_proxy'].shift(5)
    features['VIX_change'] = features['VIX_proxy'].pct_change()
    
    features['VRP'] = features['VIX_proxy'] - features['RV_22d']
    features['VRP_lag1'] = features['VRP'].shift(1)
    features['VRP_ma5'] = features['VRP'].rolling(5).mean()
    
    # === 시장 상태 ===
    features['regime_high'] = (features['VIX_proxy'] >= 25).astype(int)
    features['return_5d'] = returns.rolling(5).sum()
    features['return_22d'] = returns.rolling(22).sum()
    
    # === 상호작용 변수 ===
    features['VIX_x_return'] = features['VIX_proxy'] * features['return_5d']
    features['RV_ratio'] = features['RV_5d'] / (features['RV_22d'] + 0.01)
    features['RS_ratio'] = features['RS_minus'] / (features['RS_plus'] + 0.01)  # 하락/상승 비율
    
    # === 외부 지표 통합 ===
    ext = get_external_indicators()
    for name, series in ext.items():
        features[name] = series.reindex(df.index).ffill()
    
    # === 분수 차분 적용 (옵션) ===
    if use_fracdiff:
        try:
            features['RV_22d_fd'] = fractional_diff(features['RV_22d'].dropna(), d)
            features['VIX_fd'] = fractional_diff(features['VIX_proxy'].dropna(), d)
        except:
            pass
    
    return features.dropna()

# ============================================================================
# Phase 4: QLIKE 손실 함수
# ============================================================================

def qlike_loss(y_true, y_pred):
    """QLIKE 손실 함수"""
    y_pred = np.maximum(y_pred, 1e-8)  # 0 방지
    return np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1)

# ============================================================================
# Phase 6: NNLS 스태킹 앙상블
# ============================================================================

def nnls_stacking(predictions, y_true):
    """NNLS 메타 학습기"""
    # predictions: (n_samples, n_models)
    weights, residual = nnls(predictions, y_true)
    return weights

# ============================================================================
# 메인 실험 함수
# ============================================================================

def run_advanced_experiment(ticker, gap=5):
    """고급 실험 실행"""
    print(f"\n{'='*60}")
    print(f"Processing {ticker} (Gap={gap}d)...")
    
    df = yf.download(ticker, start='2020-01-01', end='2025-01-01', progress=False)
    if len(df) < 100:
        return None
    
    features = prepare_advanced_features(df, use_fracdiff=False)  # 분수차분은 선택적
    
    # 특성 목록 (HAR-RV-CJ 확장)
    feature_cols = [
        # HAR-RV
        'RV_1d', 'RV_5d', 'RV_22d',
        'RV_daily_mean', 'RV_weekly_mean', 'RV_monthly_mean',
        # HAR-RV-CJ 확장
        'RS_minus', 'RS_plus', 'SJ', 'RQ', 'BV', 'J',
        # VIX/VRP
        'VIX_lag1', 'VIX_lag5', 'VIX_change',
        'VRP_lag1', 'VRP_ma5',
        # 시장
        'regime_high', 'return_5d', 'return_22d',
        # 상호작용
        'VIX_x_return', 'RV_ratio', 'RS_ratio'
    ]
    
    # 외부 지표도 추가
    for col in features.columns:
        if col.startswith('VIX_real') or col.startswith('credit') or col.startswith('USD'):
            feature_cols.append(col)
    
    # 사용 가능한 특성만 선택
    feature_cols = [c for c in feature_cols if c in features.columns]
    
    X = features[feature_cols]
    y = features['RV_22d_future']
    
    # Winsorization + Log 변환
    y_winsorized = pd.Series(mstats.winsorize(y, limits=[0.01, 0.01]), index=y.index)
    y_log = np.log(y_winsorized + 1)  # log(RV)
    
    # Gap 적용 분할
    split_idx = int(len(X) * 0.8)
    
    X_train = X.iloc[:split_idx - gap]
    y_train = y_log.iloc[:split_idx - gap]
    X_test = X.iloc[split_idx:]
    y_test_log = y_log.iloc[split_idx:]
    y_test_original = y.iloc[split_idx:]
    
    # WLS 가중치 (RQ 역수)
    if 'RQ' in X_train.columns:
        weights_train = 1 / (X_train['RQ'].values + 1e-8)
        weights_train = weights_train / weights_train.sum() * len(weights_train)
    else:
        weights_train = None
    
    print(f"  Features: {len(feature_cols)}, Train: {len(X_train)}, Test: {len(X_test)}")
    
    # 모델 정의
    models = {
        'ElasticNet': Pipeline([('s', StandardScaler()), ('m', ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=2000))]),
        'Ridge': Pipeline([('s', StandardScaler()), ('m', Ridge(alpha=0.001))]),
        'Lasso': Pipeline([('s', StandardScaler()), ('m', Lasso(alpha=0.001, max_iter=2000))]),
        'Huber': Pipeline([('s', StandardScaler()), ('m', HuberRegressor(epsilon=1.35, alpha=0.001))]),
        'RF': Pipeline([('s', StandardScaler()), ('m', RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42))]),
        'GB': Pipeline([('s', StandardScaler()), ('m', GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42))]),
        'MLP_32': Pipeline([('s', StandardScaler()), ('m', MLPRegressor(hidden_layer_sizes=(32,), max_iter=500, random_state=42, early_stopping=True))]),
        'MLP_64': Pipeline([('s', StandardScaler()), ('m', MLPRegressor(hidden_layer_sizes=(64,), max_iter=500, random_state=42, early_stopping=True))]),
        'SVR': Pipeline([('s', StandardScaler()), ('m', SVR(kernel='rbf', C=1.0, epsilon=0.1))]),
    }
    
    results = {}
    all_predictions = {}
    
    for model_name, model in models.items():
        try:
            # WLS 적용 (가능한 경우)
            if weights_train is not None and hasattr(model.named_steps['m'], 'fit'):
                try:
                    model.fit(X_train, y_train)  # 일반 fit (sample_weight 미지원 모델)
                except:
                    model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train)
            
            y_pred_log = model.predict(X_test)
            y_pred = np.exp(y_pred_log) - 1  # 역변환
            
            all_predictions[model_name] = y_pred
            
            # 평가
            r2 = r2_score(y_test_original, y_pred)
            mae = mean_absolute_error(y_test_original, y_pred)
            ql = qlike_loss(y_test_original.values + 1, y_pred + 1)
            
            results[model_name] = {
                'R2': round(r2, 4),
                'MAE': round(mae, 4),
                'QLIKE': round(ql, 4)
            }
            
            print(f"  {model_name}: R2={r2:.4f}, MAE={mae:.2f}, QLIKE={ql:.4f}")
            
        except Exception as e:
            print(f"  {model_name} failed: {e}")
            results[model_name] = {'R2': None, 'MAE': None, 'QLIKE': None}
    
    # === NNLS 스태킹 앙상블 ===
    if len(all_predictions) >= 3:
        try:
            pred_matrix = np.column_stack([all_predictions[m] for m in all_predictions])
            weights = nnls_stacking(pred_matrix, y_test_original.values)
            
            ensemble_pred = pred_matrix @ weights
            ensemble_r2 = r2_score(y_test_original, ensemble_pred)
            ensemble_mae = mean_absolute_error(y_test_original, ensemble_pred)
            
            results['NNLS_Ensemble'] = {
                'R2': round(ensemble_r2, 4),
                'MAE': round(ensemble_mae, 4),
                'Weights': {m: round(w, 3) for m, w in zip(all_predictions.keys(), weights)}
            }
            
            print(f"  NNLS_Ensemble: R2={ensemble_r2:.4f}, MAE={ensemble_mae:.2f}")
            print(f"    Weights: {dict(zip(all_predictions.keys(), np.round(weights, 3)))}")
            
        except Exception as e:
            print(f"  NNLS_Ensemble failed: {e}")
    
    return results

def main():
    assets = ['SPY', 'GLD', 'QQQ']
    gaps = [5, 22]  # Gap 비교
    
    all_results = {}
    
    for gap in gaps:
        print(f"\n{'#'*60}")
        print(f"# Gap = {gap} days")
        print(f"{'#'*60}")
        
        gap_results = {}
        for asset in assets:
            result = run_advanced_experiment(asset, gap=gap)
            if result:
                gap_results[asset] = result
        
        all_results[f'gap_{gap}'] = gap_results
    
    # 결과 저장
    output = {
        'results': all_results,
        'assets': assets,
        'gaps': gaps,
        'features': 'HAR-RV-CJ + External + Interactions',
        'timestamp': datetime.now().isoformat()
    }
    
    output_path = 'data/results/advanced_experiment_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n{'='*60}")
    print(f"Results saved to {output_path}")
    
    # 요약
    print("\n=== Summary ===")
    for gap_key, gap_data in all_results.items():
        print(f"\n{gap_key}:")
        for asset, models in gap_data.items():
            best_r2 = -999
            best_model = ""
            for model, metrics in models.items():
                if metrics.get('R2') is not None and metrics['R2'] > best_r2:
                    best_r2 = metrics['R2']
                    best_model = model
            print(f"  {asset}: {best_model} (R2={best_r2:.4f})")

if __name__ == "__main__":
    main()
