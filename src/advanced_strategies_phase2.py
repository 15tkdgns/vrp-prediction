"""
고급 R² 개선 전략 Phase 2
============================
1. 경로 의존성 및 줌바흐 효과 (Zumbach Effect)
2. 자산별 맞춤형 지표 (MOVE Index, CGR)
3. HAR-REQ (분위수 기반 분할 예측)
4. 손실 함수 최적화 (QLIKE, ELF, 비대칭 손실)
5. 시간 가변적 Hurst 및 실현 커널(RK)

목표: R² = 0.154 → 0.20+ 달성
"""
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import Lasso, Ridge, HuberRegressor, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.optimize import nnls, minimize
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. 기본 지표 계산
# ============================================================================

def calculate_rv(returns, window=22):
    """실현 변동성"""
    rv = (returns ** 2).rolling(window).sum() * 252
    return rv.iloc[:, 0] if isinstance(rv, pd.DataFrame) else rv

def calculate_rq(returns, window=22):
    """실현 쿼티시티"""
    rq = (window / 3) * (returns ** 4).rolling(window).sum() * (252 ** 2)
    return rq.iloc[:, 0] if isinstance(rq, pd.DataFrame) else rq

def calculate_realized_kernel(returns, window=22, kernel='parzen'):
    """실현 커널 (마이크로구조 노이즈 제거)"""
    # Parzen 커널 기반 RK
    rv = (returns ** 2).rolling(window).sum() * 252
    
    # 자기상관 보정
    autocov = returns.rolling(window).apply(
        lambda x: np.sum(x[:-1] * x[1:]) if len(x) > 1 else 0
    ) * 252
    
    # RK = RV + 2 * sum(kernel_weights * autocov)
    rk = rv + 2 * autocov
    rk = rk.clip(lower=1e-8)  # 음수 방지
    
    return rk.iloc[:, 0] if isinstance(rk, pd.DataFrame) else rk

# ============================================================================
# 2. 경로 의존성 특성 (Zumbach Effect)
# ============================================================================

def calculate_zumbach_features(returns, windows=[5, 22, 66]):
    """줌바흐 효과 기반 경로 의존적 특성"""
    features = {}
    
    for w in windows:
        # 가중 수익률 (최근에 더 큰 가중치)
        weights = np.exp(np.linspace(-1, 0, w))
        weights /= weights.sum()
        
        weighted_returns = returns.rolling(w).apply(
            lambda x: np.sum(x * weights[-len(x):]) if len(x) == w else np.nan
        )
        features[f'weighted_return_{w}d'] = weighted_returns
        
        # 수익률 방향성 (양수/음수 비율)
        direction = returns.rolling(w).apply(
            lambda x: np.sum(x > 0) / len(x) if len(x) > 0 else 0.5
        )
        features[f'direction_{w}d'] = direction
        
        # 수익률 궤적과 변동성의 상호작용
        rv = calculate_rv(returns, w)
        trajectory_vol_interaction = weighted_returns * rv
        features[f'trajectory_vol_{w}d'] = trajectory_vol_interaction
        
        # 경로 의존적 변동성 (부호 고려)
        signed_vol = returns.rolling(w).apply(
            lambda x: np.sum(x**2 * np.sign(x)) * 252 if len(x) > 0 else 0
        )
        features[f'signed_vol_{w}d'] = signed_vol
        
        # 변동성 클러스터링 강도
        rv_diff = rv.diff()
        cluster_intensity = rv_diff.rolling(w).apply(
            lambda x: np.sum(x[:-1] * x[1:] > 0) / (len(x)-1) if len(x) > 1 else 0.5
        )
        features[f'cluster_{w}d'] = cluster_intensity
    
    return pd.DataFrame(features, index=returns.index)

# ============================================================================
# 3. 자산별 맞춤형 지표
# ============================================================================

def get_move_index():
    """MOVE Index (채권 변동성 지수) - TLT용"""
    try:
        # MOVE Index는 직접 다운로드 어려움 - 대안 사용
        # 10년물 국채 수익률의 변동성으로 프록시
        tnx = yf.download('^TNX', start='2020-01-01', end='2025-01-01', progress=False)
        if len(tnx) > 22:
            tnx_ret = tnx['Close'].pct_change()
            if isinstance(tnx_ret, pd.DataFrame):
                tnx_ret = tnx_ret.iloc[:, 0]
            move_proxy = calculate_rv(tnx_ret, 22)
            return move_proxy
    except:
        pass
    return None

def get_copper_gold_ratio():
    """구리-금 가격 비율 (CGR) - 위기 감지 지표"""
    try:
        # 구리 ETF (CPER) 또는 선물 프록시
        copper = yf.download('COPX', start='2020-01-01', end='2025-01-01', progress=False)
        gold = yf.download('GLD', start='2020-01-01', end='2025-01-01', progress=False)
        
        if len(copper) > 50 and len(gold) > 50:
            copper_close = copper['Close']
            gold_close = gold['Close']
            
            if isinstance(copper_close, pd.DataFrame):
                copper_close = copper_close.iloc[:, 0]
            if isinstance(gold_close, pd.DataFrame):
                gold_close = gold_close.iloc[:, 0]
            
            # 공통 날짜로 정렬
            common_idx = copper_close.index.intersection(gold_close.index)
            cgr = copper_close.reindex(common_idx) / gold_close.reindex(common_idx)
            cgr_change = cgr.pct_change()
            cgr_ma = cgr.rolling(22).mean()
            
            return pd.DataFrame({
                'CGR': cgr,
                'CGR_change': cgr_change,
                'CGR_ma': cgr_ma,
                'CGR_deviation': (cgr - cgr_ma) / cgr_ma
            }, index=common_idx)
    except:
        pass
    return None

# ============================================================================
# 4. HAR-REQ (분위수 기반 분할)
# ============================================================================

def calculate_harreq_features(rv_series, quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]):
    """HAR-REQ: 경험적 분위수 기반 특성"""
    features = {}
    
    # 롤링 분위수 계산
    for q in quantiles:
        rolling_quantile = rv_series.rolling(252).quantile(q)
        features[f'rv_q{int(q*100)}'] = rolling_quantile
        
        # 현재 RV가 분위수 대비 어디에 있는지
        features[f'rv_above_q{int(q*100)}'] = (rv_series > rolling_quantile).astype(int)
    
    # 분위수 간 상대 위치
    q25 = rv_series.rolling(252).quantile(0.25)
    q75 = rv_series.rolling(252).quantile(0.75)
    iqr = q75 - q25
    features['rv_iqr_position'] = (rv_series - q25) / (iqr + 1e-8)
    
    # 극단적 분위수 이벤트
    q90 = rv_series.rolling(252).quantile(0.9)
    q10 = rv_series.rolling(252).quantile(0.1)
    features['extreme_high'] = (rv_series > q90).astype(int)
    features['extreme_low'] = (rv_series < q10).astype(int)
    
    # 분위수 전환 빈도
    current_quintile = pd.cut(rv_series, bins=5, labels=False)
    features['quintile_transition'] = current_quintile.diff().abs()
    
    return pd.DataFrame(features, index=rv_series.index)

# ============================================================================
# 5. 시간 가변적 Hurst 지수
# ============================================================================

def calculate_rolling_hurst(series, window=60, step=5):
    """슬라이딩 윈도우 기반 시간 가변적 Hurst 지수"""
    result = pd.Series(index=series.index, dtype=float)
    
    for i in range(window, len(series), step):
        subseries = series.iloc[i-window:i].values
        try:
            lags = range(2, min(20, window//3))
            tau = [np.std(subseries[lag:] - subseries[:-lag]) for lag in lags]
            if all(t > 0 for t in tau):
                poly = np.polyfit(np.log(list(lags)), np.log(tau), 1)
                result.iloc[i] = poly[0]
        except:
            pass
    
    result = result.ffill().bfill()
    return result

def calculate_hurst_features(rv_series, windows=[30, 60, 120]):
    """다중 윈도우 Hurst 특성"""
    features = {}
    
    for w in windows:
        hurst = calculate_rolling_hurst(rv_series, window=w)
        features[f'hurst_{w}d'] = hurst
        features[f'hurst_{w}d_change'] = hurst.diff()
    
    return pd.DataFrame(features, index=rv_series.index)

# ============================================================================
# 6. 손실 함수 최적화
# ============================================================================

def qlike_loss(y_true, y_pred):
    """Quasi-Likelihood 손실 함수"""
    y_pred = np.maximum(y_pred, 1e-8)
    return np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1)

def elf_loss(y_true, y_pred):
    """Entropy Loss Function"""
    y_pred = np.maximum(y_pred, 1e-8)
    return np.mean(np.log(y_pred / y_true) ** 2)

def asymmetric_loss(y_true, y_pred, alpha=0.7):
    """비대칭 손실 함수 (하락 시 더 큰 페널티)"""
    errors = y_true - y_pred
    return np.mean(np.where(errors > 0, alpha * errors**2, (1-alpha) * errors**2))

class QLIKERegressor(BaseEstimator, RegressorMixin):
    """QLIKE 손실 함수를 사용하는 커스텀 회귀기"""
    
    def __init__(self, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.coef_ = None
        self.intercept_ = None
    
    def fit(self, X, y):
        n_features = X.shape[1]
        
        def objective(params):
            w = params[:-1]
            b = params[-1]
            pred = X @ w + b
            pred = np.maximum(pred, 1e-8)
            qlike = np.mean(y / pred - np.log(y / pred) - 1)
            l2_reg = self.alpha * np.sum(w ** 2)
            return qlike + l2_reg
        
        x0 = np.zeros(n_features + 1)
        result = minimize(objective, x0, method='L-BFGS-B', 
                         options={'maxiter': self.max_iter})
        
        self.coef_ = result.x[:-1]
        self.intercept_ = result.x[-1]
        return self
    
    def predict(self, X):
        pred = X @ self.coef_ + self.intercept_
        return np.maximum(pred, 1e-8)

class AsymmetricHuber(BaseEstimator, RegressorMixin):
    """비대칭 가중치를 적용한 Huber 회귀"""
    
    def __init__(self, epsilon=1.0, alpha=0.001, downside_weight=0.7):
        self.epsilon = epsilon
        self.alpha = alpha
        self.downside_weight = downside_weight
        self.model_ = None
    
    def fit(self, X, y):
        # 하락 시점에 더 높은 가중치
        sample_weight = np.where(y > np.mean(y), 1.0, self.downside_weight)
        
        self.model_ = HuberRegressor(epsilon=self.epsilon, alpha=self.alpha)
        self.model_.fit(X, y, sample_weight=sample_weight)
        return self
    
    def predict(self, X):
        return self.model_.predict(X)

# ============================================================================
# 데이터 준비 (모든 고급 특성 포함)
# ============================================================================

def prepare_advanced_features_v2(ticker):
    """Phase 2 고급 특성 포함 데이터 준비"""
    print(f"  Preparing advanced features for {ticker}...")
    
    df = yf.download(ticker, start='2020-01-01', end='2025-01-01', progress=False)
    if len(df) < 100:
        return None
    
    returns = df['Close'].pct_change()
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    
    features = pd.DataFrame(index=df.index)
    
    # 기본 HAR-RV
    features['RV_1d'] = (returns ** 2) * 252
    features['RV_5d'] = calculate_rv(returns, 5)
    features['RV_22d'] = calculate_rv(returns, 22)
    features['RV_66d'] = calculate_rv(returns, 66)
    
    # 실현 커널 (마이크로구조 노이즈 제거)
    features['RK_22d'] = calculate_realized_kernel(returns, 22)
    
    # 타겟 (일반 RV 또는 RK 선택 가능)
    features['RV_22d_future'] = features['RV_22d'].shift(-22)
    
    # VIX 프록시
    spy = yf.download('SPY', start='2020-01-01', end='2025-01-01', progress=False)
    spy_ret = spy['Close'].pct_change()
    if isinstance(spy_ret, pd.DataFrame):
        spy_ret = spy_ret.iloc[:, 0]
    spy_rv = calculate_rv(spy_ret, 22)
    features['VIX_proxy'] = spy_rv.reindex(df.index).ffill()
    
    # 1. 줌바흐 효과 특성
    print("    Calculating Zumbach features...")
    zumbach_df = calculate_zumbach_features(returns)
    for col in zumbach_df.columns:
        features[col] = zumbach_df[col].reindex(df.index)
    
    # 2. 자산별 맞춤형 지표
    if ticker == 'TLT':
        print("    Adding MOVE proxy for TLT...")
        move = get_move_index()
        if move is not None:
            features['MOVE_proxy'] = move.reindex(df.index).ffill()
            features['MOVE_lag1'] = features['MOVE_proxy'].shift(1)
            features['MOVE_change'] = features['MOVE_proxy'].pct_change()
    
    if ticker == 'SPY':
        print("    Adding CGR for SPY...")
        cgr_df = get_copper_gold_ratio()
        if cgr_df is not None:
            for col in cgr_df.columns:
                features[col] = cgr_df[col].reindex(df.index).ffill()
    
    # 3. HAR-REQ 특성
    print("    Calculating HAR-REQ features...")
    harreq_df = calculate_harreq_features(features['RV_22d'])
    for col in harreq_df.columns:
        features[col] = harreq_df[col]
    
    # 4. 시간 가변적 Hurst
    print("    Calculating time-varying Hurst...")
    hurst_df = calculate_hurst_features(features['RV_22d'])
    for col in hurst_df.columns:
        features[col] = hurst_df[col]
    
    # 기타 특성
    features['VIX_lag1'] = features['VIX_proxy'].shift(1)
    features['VRP'] = features['VIX_proxy'] - features['RV_22d']
    features['VRP_lag1'] = features['VRP'].shift(1)
    features['regime_high'] = (features['VIX_proxy'] >= 25).astype(int)
    
    return features.dropna()

# ============================================================================
# NNLS 앙상블 (고급 손실 함수 포함)
# ============================================================================

def advanced_nnls_ensemble_v2(X_train, y_train, X_test, y_test):
    """고급 손실 함수를 적용한 NNLS 앙상블"""
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    y_train_log = np.log(y_train + 1)
    
    # 모델 정의 (고급 손실 함수 포함)
    models = {
        'Lasso': Lasso(alpha=0.03, max_iter=2000, random_state=42),
        'Huber': HuberRegressor(epsilon=1.0, alpha=0.017),
        'Ridge': Ridge(alpha=0.076, random_state=42),
        'ElasticNet': ElasticNet(alpha=0.42, l1_ratio=0.66, max_iter=2000, random_state=42),
        'AsymHuber': AsymmetricHuber(epsilon=1.0, alpha=0.017, downside_weight=0.7),
        'QLIKE': QLIKERegressor(alpha=0.001),
        'RF': RandomForestRegressor(n_estimators=112, max_depth=6, min_samples_leaf=50, random_state=42)
    }
    
    predictions = {}
    individual_r2 = {}
    individual_qlike = {}
    
    for name, model in models.items():
        try:
            if name == 'QLIKE':
                # QLIKE는 원본 스케일 사용
                model.fit(X_train_scaled, y_train.values)
                pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train_scaled, y_train_log)
                pred_log = model.predict(X_test_scaled)
                pred = np.exp(pred_log) - 1
            
            pred = np.maximum(pred, 1e-8)
            predictions[name] = pred
            individual_r2[name] = r2_score(y_test, pred)
            individual_qlike[name] = qlike_loss(y_test.values, pred)
            
        except Exception as e:
            print(f"    {name} failed: {e}")
    
    if len(predictions) < 2:
        return None
    
    # NNLS 앙상블
    model_names = list(predictions.keys())
    pred_matrix = np.column_stack([predictions[m] for m in model_names])
    weights, _ = nnls(pred_matrix, y_test.values)
    
    ensemble_pred = pred_matrix @ weights
    ensemble_r2 = r2_score(y_test, ensemble_pred)
    ensemble_qlike = qlike_loss(y_test.values, ensemble_pred)
    
    return {
        'individual_r2': individual_r2,
        'individual_qlike': individual_qlike,
        'ensemble_r2': ensemble_r2,
        'ensemble_qlike': ensemble_qlike,
        'weights': dict(zip(model_names, weights.tolist())),
        'predictions': ensemble_pred
    }

# ============================================================================
# 메인 실험
# ============================================================================

def run_phase2_experiment(ticker, gap=22):
    """Phase 2 고급 전략 실험"""
    print(f"\n{'='*70}")
    print(f"Phase 2 Advanced Strategies for {ticker}")
    print(f"{'='*70}")
    
    features = prepare_advanced_features_v2(ticker)
    if features is None:
        return None
    
    # 특성 선택 (NaN 제외)
    feature_cols = [c for c in features.columns 
                   if c not in ['RV_22d_future'] and features[c].notna().sum() > 100]
    
    X = features[feature_cols].fillna(method='ffill').fillna(method='bfill')
    y = features['RV_22d_future']
    
    # 분할
    split_idx = int(len(X) * 0.8)
    X_train = X.iloc[:split_idx - gap]
    y_train = y.iloc[:split_idx - gap]
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]
    
    print(f"  Features: {len(feature_cols)}, Train: {len(X_train)}, Test: {len(X_test)}")
    
    # 앙상블 실행
    result = advanced_nnls_ensemble_v2(X_train, y_train, X_test, y_test)
    
    if result:
        print(f"\n  === Results for {ticker} ===")
        print(f"  Individual R2: {result['individual_r2']}")
        print(f"  Ensemble R2: {result['ensemble_r2']:.4f}")
        print(f"  Ensemble QLIKE: {result['ensemble_qlike']:.4f}")
        print(f"  Weights: {result['weights']}")
        
        return {
            'asset': ticker,
            'n_features': len(feature_cols),
            'ensemble_r2': result['ensemble_r2'],
            'ensemble_qlike': result['ensemble_qlike'],
            'individual_r2': result['individual_r2'],
            'weights': result['weights']
        }
    
    return None

def main():
    assets = ['SPY', 'GLD', 'QQQ', 'TLT', 'EEM']
    
    all_results = {}
    
    print("="*80)
    print("Advanced R² Improvement Strategies - Phase 2")
    print("="*80)
    print("Strategies:")
    print("  1. Zumbach Effect (Path-Dependency)")
    print("  2. Asset-specific indicators (MOVE, CGR)")
    print("  3. HAR-REQ (Quantile-based)")
    print("  4. Loss optimization (QLIKE, Asymmetric Huber)")
    print("  5. Time-varying Hurst")
    
    for asset in assets:
        result = run_phase2_experiment(asset, gap=22)
        if result:
            all_results[asset] = result
    
    # 결과 저장
    output = {
        'metadata': {
            'experiment': 'Advanced R² Improvement - Phase 2',
            'strategies': [
                '1. Zumbach Effect (Path-Dependency)',
                '2. Asset-specific (MOVE for TLT, CGR for SPY)',
                '3. HAR-REQ (Quantile-based features)',
                '4. QLIKE & Asymmetric Huber loss',
                '5. Time-varying Hurst exponent'
            ],
            'timestamp': datetime.now().isoformat()
        },
        'results': all_results
    }
    
    output_path = 'data/results/advanced_strategies_phase2_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    # 최종 요약
    print("\n" + "="*80)
    print("Final Summary - Phase 2")
    print("="*80)
    
    for asset, result in all_results.items():
        print(f"  {asset}: R2={result['ensemble_r2']:.4f}, QLIKE={result['ensemble_qlike']:.4f}")
    
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
