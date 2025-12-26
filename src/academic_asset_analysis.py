"""
학술 연구 기반 자산 분석
=======================
변동성 예측 관련 주요 논문에서 자주 사용하는 자산으로 구성

참고 문헌:
- Corsi (2009) HAR-RV: S&P 500
- Bollerslev et al. (2009): S&P 500, VIX
- Andersen et al. (2007): S&P 500, Bonds, FX
- Christoffersen et al. (2008): 주요 지수, 섹터
- Bekaert & Hoerova (2014): S&P 500, VRP
- Liu et al. (2015): Commodity, Gold, Oil

핵심 자산:
1. 주식 지수 (Stock Index): SPY, QQQ
2. 채권 (Bonds): TLT
3. 금 (Gold): GLD
4. 원유 (Oil): USO
5. 신흥시장 (Emerging): EEM
6. 섹터 (Sector): XLF (금융)
"""
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import Ridge, Lasso, HuberRegressor, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def calculate_rv(returns, window):
    rv = (returns ** 2).rolling(window).sum() * 252
    return rv.iloc[:, 0] if isinstance(rv, pd.DataFrame) else rv

def prepare_features(ticker):
    """특성 준비"""
    try:
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
    except Exception as e:
        return None

def get_models():
    """모델 정의"""
    return {
        'Ridge_10': Ridge(alpha=10.0),
        'Ridge_100': Ridge(alpha=100.0),
        'Lasso_0.01': Lasso(alpha=0.01, max_iter=3000),
        'Huber': HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=500),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=3000)
    }

def evaluate_asset_model(ticker, model_name, model, features, gap=5):
    """자산-모델 평가"""
    feature_cols = ['RV_5d_lag1', 'RV_22d_lag1', 'RV_ratio_lag1', 
                    'VIX_lag1', 'VIX_change_lag1', 'direction_5d_lag1']
    X = features[feature_cols]
    y = features['RV_5d_future']
    
    n = len(X)
    train_end = int(n * 0.7) - gap
    
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_test = X.iloc[train_end + gap:]
    y_test = y.iloc[train_end + gap:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    try:
        from sklearn.base import clone
        model_copy = clone(model)
        model_copy.fit(X_train_s, np.sqrt(y_train))
        pred_t = model_copy.predict(X_test_s)
        pred = np.maximum(pred_t ** 2, 0)
        
        r2 = r2_score(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mae = mean_absolute_error(y_test, pred)
        
        persist_pred = X_test['RV_5d_lag1'].values
        actual_dir = (y_test.values > persist_pred).astype(int)
        pred_dir = (pred > persist_pred).astype(int)
        direction_acc = (actual_dir == pred_dir).mean()
        
        return {
            'r2': round(r2, 4),
            'rmse': round(rmse, 4),
            'mae': round(mae, 4),
            'direction_acc': round(direction_acc, 4)
        }
    except:
        return None

def main():
    print("="*80)
    print("학술 연구 기반 자산 분석")
    print("="*80)
    
    # 핵심 자산 (학술 연구 기반)
    assets = [
        # 주식 지수 (가장 많이 연구됨)
        ('SPY', 'Stock Index', 'S&P 500 ETF'),
        ('QQQ', 'Stock Index', 'NASDAQ 100'),
        
        # 금 (안전자산, commodity 연구)
        ('GLD', 'Commodity', 'Gold'),
        
        # 원유 (에너지, commodity 연구)
        ('USO', 'Commodity', 'Oil'),
        
        # 채권 (multi-asset 연구)
        ('TLT', 'Bond', '20+ Year Treasury'),
        
        # 신흥시장 (글로벌 연구)
        ('EEM', 'Emerging', 'Emerging Markets'),
        
        # 섹터 (금융 - 위기 연구)
        ('XLF', 'Sector', 'Financials'),
        
        # 기술 섹터
        ('XLK', 'Sector', 'Technology'),
    ]
    
    models = get_models()
    
    results = {
        'assets': {},
        'matrix': {},
        'best_by_asset': {},
        'best_by_model': {}
    }
    
    for ticker, asset_class, asset_name in assets:
        print(f"\n[{ticker}] {asset_name} ({asset_class})")
        
        features = prepare_features(ticker)
        if features is None:
            print("  SKIP - insufficient data")
            continue
        
        results['assets'][ticker] = {
            'name': asset_name,
            'class': asset_class,
            'n_samples': len(features)
        }
        
        results['matrix'][ticker] = {}
        
        for model_name, model in models.items():
            result = evaluate_asset_model(ticker, model_name, model, features)
            if result:
                results['matrix'][ticker][model_name] = result
                print(f"  {model_name}: R²={result['r2']:.3f}, Dir={result['direction_acc']:.1%}")
        
        # 자산별 최고 모델
        if results['matrix'][ticker]:
            best = max(results['matrix'][ticker].items(), key=lambda x: x[1]['r2'])
            results['best_by_asset'][ticker] = {
                'model': best[0],
                'metrics': best[1]
            }
    
    # 모델별 평균 성능
    for model_name in models.keys():
        r2_values = []
        for ticker in results['matrix'].keys():
            if model_name in results['matrix'][ticker]:
                r2_values.append(results['matrix'][ticker][model_name]['r2'])
        if r2_values:
            results['best_by_model'][model_name] = {
                'avg_r2': round(np.mean(r2_values), 4),
                'std_r2': round(np.std(r2_values), 4)
            }
    
    # JSON 저장
    output = {
        'metadata': {
            'experiment': 'Academic Research Based Asset Analysis',
            'timestamp': datetime.now().isoformat(),
            'reference': 'Corsi (2009), Bollerslev et al. (2009), Liu et al. (2015)',
            'assets': [a[0] for a in assets]
        },
        'results': results
    }
    
    output_path = 'data/results/academic_asset_analysis.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    # 요약
    print("\n" + "="*80)
    print("자산별 최적 모델")
    print("="*80)
    
    for ticker, info in results['best_by_asset'].items():
        print(f"  {ticker}: {info['model']} (R²={info['metrics']['r2']:.4f})")
    
    print("\n" + "="*80)
    print("모델별 평균 성능")
    print("="*80)
    
    sorted_models = sorted(results['best_by_model'].items(), 
                          key=lambda x: x[1]['avg_r2'], reverse=True)
    for model, stats in sorted_models:
        print(f"  {model}: avg R²={stats['avg_r2']:.4f} (±{stats['std_r2']:.4f})")
    
    print(f"\n결과 저장: {output_path}")

if __name__ == "__main__":
    main()
