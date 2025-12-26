"""
5일 예측 모델 튜닝 최적화
==========================
검증된 R²를 기반으로 하이퍼파라미터 최적화

목표: Walk-Forward CV R² 개선
- SPY: 0.18 → ?
- QQQ: 0.15 → ?
- XLF: 0.11 → ?
- XLK: 0.09 → ?
"""
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
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
# Walk-Forward CV 튜닝
# ============================================================================

def tune_with_wf_cv(ticker, n_splits=5):
    """Walk-Forward CV로 하이퍼파라미터 튜닝"""
    print(f"\n{'='*60}")
    print(f"Tuning: {ticker}")
    print(f"{'='*60}")
    
    features = prepare_features(ticker)
    if features is None:
        return None
    
    feature_cols = [c for c in features.columns if c != 'RV_5d_future']
    X = features[feature_cols]
    y = features['RV_5d_future']
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    gap = 5
    
    # 튜닝할 조합
    configs = [
        # (모델 이름, 모델, 타겟 변환)
        ('Ridge_a10_raw', Ridge(alpha=10.0), 'raw'),
        ('Ridge_a100_raw', Ridge(alpha=100.0), 'raw'),
        ('Ridge_a1000_raw', Ridge(alpha=1000.0), 'raw'),
        ('Ridge_a10_log', Ridge(alpha=10.0), 'log'),
        ('Ridge_a100_log', Ridge(alpha=100.0), 'log'),
        ('Ridge_a1000_log', Ridge(alpha=1000.0), 'log'),
        ('Ridge_a10_sqrt', Ridge(alpha=10.0), 'sqrt'),
        ('Ridge_a100_sqrt', Ridge(alpha=100.0), 'sqrt'),
        ('Lasso_a0.01_log', Lasso(alpha=0.01, max_iter=3000), 'log'),
        ('Lasso_a0.1_log', Lasso(alpha=0.1, max_iter=3000), 'log'),
        ('ElasticNet_log', ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=3000), 'log'),
        ('Huber_raw', HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=500), 'raw'),
        ('Huber_log', HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=500), 'log'),
    ]
    
    results = {}
    
    for name, model_template, transform in configs:
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
            
            # 타겟 변환
            if transform == 'log':
                y_train_t = np.log(y_train + 1)
            elif transform == 'sqrt':
                y_train_t = np.sqrt(y_train)
            else:
                y_train_t = y_train.values
            
            try:
                # 모델 복사 및 학습
                from sklearn.base import clone
                model = clone(model_template)
                model.fit(X_train_s, y_train_t)
                
                pred_t = model.predict(X_test_s)
                
                # 역변환
                if transform == 'log':
                    pred = np.exp(pred_t) - 1
                elif transform == 'sqrt':
                    pred = pred_t ** 2
                else:
                    pred = pred_t
                
                pred = np.maximum(pred, 0)
                r2 = r2_score(y_test, pred)
                fold_r2s.append(r2)
            except Exception as e:
                pass
        
        if fold_r2s:
            avg_r2 = np.mean(fold_r2s)
            std_r2 = np.std(fold_r2s)
            results[name] = {'avg_r2': avg_r2, 'std_r2': std_r2, 'folds': fold_r2s}
    
    # 최고 결과
    if results:
        best = max(results.items(), key=lambda x: x[1]['avg_r2'])
        print(f"\n  Best: {best[0]}")
        print(f"    CV Avg R²: {best[1]['avg_r2']:.4f} (±{best[1]['std_r2']:.4f})")
        
        # 상위 5개 출력
        print("\n  Top 5 Configurations:")
        sorted_results = sorted(results.items(), key=lambda x: x[1]['avg_r2'], reverse=True)[:5]
        for name, r in sorted_results:
            print(f"    {name}: {r['avg_r2']:.4f} (±{r['std_r2']:.4f})")
        
        return {
            'asset': ticker,
            'best_config': best[0],
            'best_avg_r2': best[1]['avg_r2'],
            'best_std_r2': best[1]['std_r2'],
            'all_results': {k: v for k, v in sorted_results}
        }
    
    return None

# ============================================================================
# 최적 모델로 최종 테스트
# ============================================================================

def final_test(ticker, best_config):
    """최적 설정으로 최종 테스트"""
    features = prepare_features(ticker)
    if features is None:
        return None
    
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
    
    # 설정 파싱
    parts = best_config.split('_')
    model_type = parts[0]
    transform = parts[-1]
    
    # alpha 추출
    alpha = 100.0
    for p in parts:
        if p.startswith('a'):
            try:
                alpha = float(p[1:])
            except:
                pass
    
    # 모델 생성
    if model_type == 'Ridge':
        model = Ridge(alpha=alpha)
    elif model_type == 'Lasso':
        model = Lasso(alpha=alpha, max_iter=3000)
    elif model_type == 'ElasticNet':
        model = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=3000)
    elif model_type == 'Huber':
        model = HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=500)
    else:
        model = Ridge(alpha=100.0)
    
    # 타겟 변환
    if transform == 'log':
        y_train_t = np.log(y_train + 1)
    elif transform == 'sqrt':
        y_train_t = np.sqrt(y_train)
    else:
        y_train_t = y_train.values
    
    model.fit(X_train_s, y_train_t)
    pred_t = model.predict(X_test_s)
    
    if transform == 'log':
        pred = np.exp(pred_t) - 1
    elif transform == 'sqrt':
        pred = pred_t ** 2
    else:
        pred = pred_t
    
    pred = np.maximum(pred, 0)
    r2 = r2_score(y_test, pred)
    
    # 방향 정확도
    actual_direction = (y_test.values > X_test['RV_5d_lag1'].values).astype(int)
    pred_direction = (pred > X_test['RV_5d_lag1'].values).astype(int)
    direction_acc = (actual_direction == pred_direction).mean()
    
    return {
        'test_r2': r2,
        'direction_accuracy': direction_acc
    }

# ============================================================================
# 메인
# ============================================================================

def main():
    print("="*80)
    print("5-Day Model Tuning Optimization")
    print("="*80)
    print("Goal: Improve Walk-Forward CV R²")
    
    assets = ['SPY', 'QQQ', 'XLF', 'XLK']
    
    all_results = {}
    
    for ticker in assets:
        result = tune_with_wf_cv(ticker)
        if result:
            all_results[ticker] = result
            
            # 최종 테스트
            final = final_test(ticker, result['best_config'])
            if final:
                result['final_test'] = final
                print(f"    Final Test R²: {final['test_r2']:.4f}")
                print(f"    Direction Accuracy: {final['direction_accuracy']:.2%}")
    
    # 결과 저장
    output = {
        'metadata': {
            'experiment': '5-Day Model Tuning',
            'timestamp': datetime.now().isoformat()
        },
        'results': all_results
    }
    
    output_path = 'data/results/5day_tuning_optimized.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    # 요약
    print("\n" + "="*80)
    print("Summary: Optimized Results")
    print("="*80)
    
    print(f"{'Asset':<8} {'Best Config':<25} {'CV R²':>10} {'Test R²':>10} {'Dir Acc':>10}")
    print("-"*70)
    
    for ticker, result in all_results.items():
        best = result['best_config']
        cv_r2 = result['best_avg_r2']
        test_r2 = result.get('final_test', {}).get('test_r2', 0)
        dir_acc = result.get('final_test', {}).get('direction_accuracy', 0)
        print(f"{ticker:<8} {best:<25} {cv_r2:>10.4f} {test_r2:>10.4f} {dir_acc:>10.2%}")
    
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
