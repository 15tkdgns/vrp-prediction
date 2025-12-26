"""
확장 자산 분석
=============
다양한 자산군에 대한 5일 예측 모델 성능 분석

자산군:
- 주식 지수: SPY, QQQ, IWM
- 섹터 ETF: XLK, XLF, XLE
- 신흥 시장: EEM, FXI, EWZ
- 원자재: GLD, SLV, USO
- 채권: TLT, HYG, LQD
- 부동산: VNQ
"""
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import Ridge
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
        print(f"  Error for {ticker}: {e}")
        return None

def analyze_asset(ticker, asset_class, asset_name):
    """개별 자산 분석"""
    print(f"\n  {ticker} ({asset_name})...", end="")
    
    features = prepare_features(ticker)
    if features is None:
        print(" SKIP (insufficient data)")
        return None
    
    feature_cols = ['RV_5d_lag1', 'RV_22d_lag1', 'RV_ratio_lag1', 
                    'VIX_lag1', 'VIX_change_lag1', 'direction_5d_lag1']
    X = features[feature_cols]
    y = features['RV_5d_future']
    
    # Walk-Forward CV
    tscv = TimeSeriesSplit(n_splits=5)
    gap = 5
    cv_r2s = []
    
    for train_idx, test_idx in tscv.split(X):
        train_idx = train_idx[:-gap] if len(train_idx) > gap else train_idx
        
        if len(train_idx) < 200 or len(test_idx) < 50:
            continue
        
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        model = Ridge(alpha=100.0)
        model.fit(X_train_s, np.sqrt(y_train))
        pred = np.maximum(model.predict(X_test_s) ** 2, 0)
        
        cv_r2s.append(r2_score(y_test, pred))
    
    if not cv_r2s:
        print(" SKIP (insufficient CV)")
        return None
    
    # Test R²
    n = len(X)
    train_end = int(n * 0.7) - gap
    
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_test = X.iloc[train_end + gap:]
    y_test = y.iloc[train_end + gap:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    model = Ridge(alpha=100.0)
    model.fit(X_train_s, np.sqrt(y_train))
    pred = np.maximum(model.predict(X_test_s) ** 2, 0)
    
    test_r2 = r2_score(y_test, pred)
    
    # 방향 정확도
    persist_pred = X_test['RV_5d_lag1'].values
    actual_dir = (y_test.values > persist_pred).astype(int)
    pred_dir = (pred > persist_pred).astype(int)
    dir_acc = (actual_dir == pred_dir).mean()
    
    cv_mean = np.mean(cv_r2s)
    cv_std = np.std(cv_r2s)
    
    print(f" CV={cv_mean:.3f}, Test={test_r2:.3f}, Dir={dir_acc:.1%}")
    
    return {
        'ticker': ticker,
        'name': asset_name,
        'class': asset_class,
        'n_samples': len(X),
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'test_r2': test_r2,
        'direction_acc': dir_acc
    }

def main():
    print("="*80)
    print("확장 자산 분석 - 5일 예측")
    print("="*80)
    
    # 자산 리스트
    assets = [
        # 주식 지수
        ('SPY', 'Equity Index', 'S&P 500'),
        ('QQQ', 'Equity Index', 'NASDAQ 100'),
        ('IWM', 'Equity Index', 'Russell 2000'),
        ('DIA', 'Equity Index', 'Dow Jones'),
        
        # 섹터 ETF
        ('XLK', 'Sector', 'Technology'),
        ('XLF', 'Sector', 'Financials'),
        ('XLE', 'Sector', 'Energy'),
        ('XLV', 'Sector', 'Healthcare'),
        ('XLI', 'Sector', 'Industrials'),
        ('XLY', 'Sector', 'Consumer Disc'),
        ('XLP', 'Sector', 'Consumer Staples'),
        ('XLU', 'Sector', 'Utilities'),
        
        # 신흥 시장
        ('EEM', 'Emerging', 'Emerging Markets'),
        ('FXI', 'Emerging', 'China'),
        ('EWZ', 'Emerging', 'Brazil'),
        ('EWJ', 'Emerging', 'Japan'),
        
        # 원자재
        ('GLD', 'Commodity', 'Gold'),
        ('SLV', 'Commodity', 'Silver'),
        ('USO', 'Commodity', 'Oil'),
        
        # 채권
        ('TLT', 'Bond', '20+ Year Treasury'),
        ('IEF', 'Bond', '7-10 Year Treasury'),
        ('HYG', 'Bond', 'High Yield'),
        ('LQD', 'Bond', 'Investment Grade'),
        
        # 부동산
        ('VNQ', 'Real Estate', 'REIT'),
    ]
    
    results = []
    
    for ticker, asset_class, asset_name in assets:
        result = analyze_asset(ticker, asset_class, asset_name)
        if result:
            results.append(result)
    
    # 결과 정리
    df = pd.DataFrame(results)
    df = df.sort_values('cv_mean', ascending=False)
    
    # 결과 저장
    output = {
        'metadata': {
            'experiment': 'Extended Asset Analysis',
            'timestamp': datetime.now().isoformat(),
            'n_assets': len(results)
        },
        'results': results
    }
    
    output_path = 'data/results/extended_asset_analysis.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    # 요약
    print("\n" + "="*80)
    print("요약 - CV R² 순위")
    print("="*80)
    
    print(f"\n{'Rank':<5} {'Ticker':<6} {'Class':<15} {'Name':<20} {'CV R²':>8} {'Test R²':>8} {'Dir':>8}")
    print("-"*80)
    
    for i, row in df.iterrows():
        rank = list(df.index).index(i) + 1
        print(f"{rank:<5} {row['ticker']:<6} {row['class']:<15} {row['name']:<20} {row['cv_mean']:>8.4f} {row['test_r2']:>8.4f} {row['direction_acc']:>7.1%}")
    
    # 자산군별 요약
    print("\n" + "="*80)
    print("자산군별 평균")
    print("="*80)
    
    class_summary = df.groupby('class').agg({
        'cv_mean': 'mean',
        'test_r2': 'mean',
        'direction_acc': 'mean'
    }).sort_values('cv_mean', ascending=False)
    
    print(f"\n{'Class':<20} {'CV R²':>10} {'Test R²':>10} {'Dir Acc':>10}")
    print("-"*55)
    for cls, row in class_summary.iterrows():
        print(f"{cls:<20} {row['cv_mean']:>10.4f} {row['test_r2']:>10.4f} {row['direction_acc']:>9.1%}")
    
    print(f"\n결과 저장: {output_path}")

if __name__ == "__main__":
    main()
