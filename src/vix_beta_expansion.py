#!/usr/bin/env python3
"""
VIX-Beta 이론 확장 분석
======================

다양한 자산에 대한 VIX-RV 상관관계와 VRP 예측력 분석:
- QQQ (기술주)
- TLT (채권)
- USO (원유)
- EEM (신흥국)
- IWM (소형주)
- GLD (금)

실행: python src/vix_beta_expansion.py
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import yfinance as yf
from pathlib import Path
import json
from datetime import datetime

SEED = 42
np.random.seed(SEED)


def download_asset_data(ticker, start='2020-01-01', end='2025-01-01'):
    """자산 데이터 다운로드"""
    data = yf.download(ticker, start=start, end=end, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data


def prepare_asset_data(asset_data, vix_data):
    """자산 데이터 전처리"""
    df = pd.DataFrame(index=asset_data.index)
    df['Close'] = asset_data['Close']
    df['VIX'] = vix_data['Close'].reindex(df.index).ffill()
    df['returns'] = df['Close'].pct_change()
    
    # 변동성 계산
    df['RV_1d'] = df['returns'].abs() * np.sqrt(252) * 100
    df['RV_5d'] = df['returns'].rolling(5).std() * np.sqrt(252) * 100
    df['RV_22d'] = df['returns'].rolling(22).std() * np.sqrt(252) * 100
    
    df['VRP'] = df['VIX'] - df['RV_22d']
    df['RV_future'] = df['RV_22d'].shift(-22)
    df['VRP_true'] = df['VIX'] - df['RV_future']
    
    # 특성 생성
    df['VIX_lag1'] = df['VIX'].shift(1)
    df['VIX_lag5'] = df['VIX'].shift(5)
    df['VIX_change'] = df['VIX'].pct_change()
    df['VRP_lag1'] = df['VRP'].shift(1)
    df['VRP_lag5'] = df['VRP'].shift(5)
    df['VRP_ma5'] = df['VRP'].rolling(5).mean()
    df['regime_high'] = (df['VIX'] >= 25).astype(int)
    df['return_5d'] = df['returns'].rolling(5).sum()
    df['return_22d'] = df['returns'].rolling(22).sum()
    
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    return df


def analyze_vix_beta(df):
    """VIX-Beta 분석"""
    # VIX-RV 상관관계
    vix_rv_corr = df['VIX'].corr(df['RV_22d'])
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5', 
                   'VIX_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                   'regime_high', 'return_5d', 'return_22d']
    
    # 데이터 분할
    X = df[feature_cols].values
    y_rv = df['RV_future'].values
    y_vrp = df['VRP_true'].values
    vix = df['VIX'].values
    
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_rv_train = y_rv[:split_idx]
    vix_test = vix[split_idx:]
    y_vrp_test = y_vrp[split_idx:]
    
    if len(X_test) < 10:
        return None
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # 간접 예측 (RV 예측 후 VRP 계산)
    model = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=SEED, max_iter=10000)
    model.fit(X_train_s, y_rv_train)
    
    rv_pred = model.predict(X_test_s)
    vrp_pred = vix_test - rv_pred
    
    # 성능 지표
    r2_indirect = r2_score(y_vrp_test, vrp_pred)
    direction_acc = ((vrp_pred > vrp_pred.mean()) == (y_vrp_test > y_vrp_test.mean())).mean()
    
    # 트레이딩 성과
    signals = (vrp_pred > vrp_pred.mean()).astype(int)
    trade_returns = y_vrp_test[signals == 1]
    win_rate = (trade_returns > 0).mean() if len(trade_returns) > 0 else 0
    
    return {
        'vix_rv_correlation': float(vix_rv_corr),
        'r2_indirect': float(r2_indirect),
        'direction_accuracy': float(direction_acc),
        'win_rate': float(win_rate),
        'n_samples': len(df),
        'n_test': len(X_test),
        'avg_vix': float(df['VIX'].mean()),
        'avg_rv': float(df['RV_22d'].mean()),
        'avg_vrp': float(df['VRP'].mean())
    }


def main():
    print("\n" + "=" * 60)
    print("📊 VIX-Beta 이론 확장 분석")
    print("=" * 60)
    
    # VIX 데이터 다운로드
    print("\n[1/3] 데이터 다운로드...")
    vix_data = download_asset_data('^VIX')
    
    assets = {
        'SPY': 'S&P 500 ETF',
        'QQQ': 'NASDAQ 100 (기술주)',
        'TLT': '장기 국채',
        'GLD': '금',
        'USO': '원유',
        'EEM': '신흥국',
        'IWM': 'Russell 2000 (소형주)',
        'XLF': '금융 섹터',
        'XLE': '에너지 섹터'
    }
    
    results = {}
    
    print(f"\n[2/3] 자산별 VIX-Beta 분석...")
    print("-" * 75)
    print(f"{'자산':<8} {'설명':<20} {'VIX-RV 상관':>12} {'R² (간접)':>12} {'방향정확도':>10} {'승률':>8}")
    print("-" * 75)
    
    for ticker, desc in assets.items():
        try:
            asset_data = download_asset_data(ticker)
            df = prepare_asset_data(asset_data, vix_data)
            
            result = analyze_vix_beta(df)
            
            if result is None:
                print(f"{ticker:<8} {desc:<20} {'데이터 부족':>12}")
                continue
            
            results[ticker] = {**result, 'description': desc}
            
            print(f"{ticker:<8} {desc:<20} {result['vix_rv_correlation']:>12.3f} "
                  f"{result['r2_indirect']:>12.4f} {result['direction_accuracy']*100:>9.1f}% "
                  f"{result['win_rate']*100:>7.1f}%")
        except Exception as e:
            print(f"{ticker:<8} {desc:<20} {'오류: ' + str(e)[:20]:>12}")
    
    # VIX-Beta 분석
    print("\n[3/3] VIX-Beta 이론 검증...")
    
    if len(results) > 1:
        correlations = [r['vix_rv_correlation'] for r in results.values()]
        r2_values = [r['r2_indirect'] for r in results.values()]
        
        # VIX-RV 상관 vs R² 상관관계
        vix_beta_corr = np.corrcoef(correlations, r2_values)[0, 1]
        
        # 상관 낮음 vs 높음 그룹
        median_corr = np.median(correlations)
        low_corr_assets = [k for k, v in results.items() if v['vix_rv_correlation'] < median_corr]
        high_corr_assets = [k for k, v in results.items() if v['vix_rv_correlation'] >= median_corr]
        
        low_corr_r2 = np.mean([results[k]['r2_indirect'] for k in low_corr_assets])
        high_corr_r2 = np.mean([results[k]['r2_indirect'] for k in high_corr_assets])
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📊 VIX-Beta 이론 검증 결과")
    print("=" * 60)
    
    if len(results) > 1:
        best_asset = max(results.items(), key=lambda x: x[1]['r2_indirect'])
        worst_asset = min(results.items(), key=lambda x: x[1]['r2_indirect'])
        
        print(f"""
    🔹 VIX-Beta 상관관계:
       • VIX-RV 상관 vs R² 상관: {vix_beta_corr:.3f}
       • 이론 지지 여부: {'✅ 지지' if vix_beta_corr < -0.3 else '⚠️ 약한 지지' if vix_beta_corr < 0 else '❌ 미지지'}
    
    🔹 그룹별 평균 R²:
       • 저상관 그룹 ({', '.join(low_corr_assets)}): {low_corr_r2:.4f}
       • 고상관 그룹 ({', '.join(high_corr_assets)}): {high_corr_r2:.4f}
       • 차이: {low_corr_r2 - high_corr_r2:+.4f}
    
    🔹 최고/최저 예측력:
       • 최고: {best_asset[0]} ({best_asset[1]['description']}) - R² = {best_asset[1]['r2_indirect']:.4f}
       • 최저: {worst_asset[0]} ({worst_asset[1]['description']}) - R² = {worst_asset[1]['r2_indirect']:.4f}
    
    💡 VIX-Beta 이론 결론:
       VIX와 상관관계가 {'낮은' if vix_beta_corr < 0 else '높은'} 자산에서 VRP 예측력이 {'높음' if vix_beta_corr < 0 else '낮음'}
    """)
    
    # 결과 저장
    output = {
        'assets': results,
        'vix_beta_analysis': {
            'vix_rv_vs_r2_correlation': float(vix_beta_corr) if len(results) > 1 else None,
            'low_corr_assets': low_corr_assets if len(results) > 1 else [],
            'high_corr_assets': high_corr_assets if len(results) > 1 else [],
            'low_corr_avg_r2': float(low_corr_r2) if len(results) > 1 else None,
            'high_corr_avg_r2': float(high_corr_r2) if len(results) > 1 else None,
            'theory_supported': bool(vix_beta_corr < 0) if len(results) > 1 else None
        },
        'timestamp': datetime.now().isoformat()
    }
    
    output_path = Path('data/results/vix_beta_expansion.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"💾 결과 저장: {output_path}")
    print("\n✅ VIX-Beta 확장 분석 완료!")


if __name__ == '__main__':
    main()
