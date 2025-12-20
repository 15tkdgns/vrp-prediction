#!/usr/bin/env python3
"""
VRP 예측 모델 검증 및 추가 분석
================================

1. Bootstrap 신뢰구간
2. Regime별 성능 분석
3. 연도별 안정성
4. 특성 중요도
5. 방향 예측 정밀도/재현율
6. 트레이딩 시뮬레이션
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, accuracy_score, precision_score, recall_score, f1_score
import yfinance as yf
from pathlib import Path
import json
from datetime import datetime

SEED = 42
np.random.seed(SEED)


def load_data_and_train():
    """데이터 로드 및 모델 학습"""
    csv_path = Path('data/raw/spy_data_2020_2025.csv')
    spy = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    
    vix = yf.download('^VIX', start='2020-01-01', end='2025-01-01', progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    
    spy['VIX'] = vix['Close'].reindex(spy.index).ffill().bfill()
    spy['returns'] = spy['Close'].pct_change()
    
    # 특성 생성
    spy['RV_1d'] = spy['returns'].abs() * np.sqrt(252) * 100
    spy['RV_5d'] = spy['returns'].rolling(5).std() * np.sqrt(252) * 100
    spy['RV_22d'] = spy['returns'].rolling(22).std() * np.sqrt(252) * 100
    spy['VRP'] = spy['VIX'] - spy['RV_22d']
    spy['RV_future'] = spy['RV_22d'].shift(-22)
    spy['VRP_true'] = spy['VIX'] - spy['RV_future']
    
    spy['VIX_lag1'] = spy['VIX'].shift(1)
    spy['VIX_lag5'] = spy['VIX'].shift(5)
    spy['VIX_change'] = spy['VIX'].pct_change()
    spy['VRP_lag1'] = spy['VRP'].shift(1)
    spy['VRP_lag5'] = spy['VRP'].shift(5)
    spy['VRP_ma5'] = spy['VRP'].rolling(5).mean()
    spy['regime_high'] = (spy['VIX'] >= 25).astype(int)
    spy['return_5d'] = spy['returns'].rolling(5).sum()
    spy['return_22d'] = spy['returns'].rolling(22).sum()
    
    spy = spy.replace([np.inf, -np.inf], np.nan)
    spy = spy.dropna()
    
    feature_cols = ['RV_1d', 'RV_5d', 'RV_22d', 'VIX_lag1', 'VIX_lag5', 
                   'VIX_change', 'VRP_lag1', 'VRP_lag5', 'VRP_ma5',
                   'regime_high', 'return_5d', 'return_22d']
    
    return spy, feature_cols


def validation_1_bootstrap(spy, feature_cols):
    """검증 1: Bootstrap 신뢰구간"""
    print("\n" + "=" * 60)
    print("[1/6] Bootstrap 신뢰구간")
    print("=" * 60)
    
    X = spy[feature_cols].values
    y_rv = spy['RV_future'].values
    y_vrp = spy['VRP_true'].values
    vix = spy['VIX'].values
    
    split_idx = int(len(spy) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_rv_train, y_rv_test = y_rv[:split_idx], y_rv[split_idx:]
    y_vrp_test = y_vrp[split_idx:]
    vix_test = vix[split_idx:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # 모델 학습
    en = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=SEED, max_iter=10000)
    en.fit(X_train_s, y_rv_train)
    rv_pred = en.predict(X_test_s)
    vrp_pred = vix_test - rv_pred
    
    # Bootstrap
    n_bootstrap = 1000
    r2_scores = []
    
    for i in range(n_bootstrap):
        idx = np.random.choice(len(y_vrp_test), size=len(y_vrp_test), replace=True)
        r2 = r2_score(y_vrp_test[idx], vrp_pred[idx])
        r2_scores.append(r2)
    
    r2_mean = np.mean(r2_scores)
    r2_std = np.std(r2_scores)
    ci_lower = np.percentile(r2_scores, 2.5)
    ci_upper = np.percentile(r2_scores, 97.5)
    
    print(f"\n  📊 VRP R² 통계:")
    print(f"     점추정: {r2_score(y_vrp_test, vrp_pred):.4f}")
    print(f"     Bootstrap 평균: {r2_mean:.4f} ± {r2_std:.4f}")
    print(f"     95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    
    return {
        'point_estimate': float(r2_score(y_vrp_test, vrp_pred)),
        'bootstrap_mean': float(r2_mean),
        'bootstrap_std': float(r2_std),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper)
    }, vrp_pred, y_vrp_test, spy.index[split_idx:]


def validation_2_regime(spy, feature_cols, vrp_pred, y_vrp_test, test_dates):
    """검증 2: Regime별 성능"""
    print("\n" + "=" * 60)
    print("[2/6] Regime별 성능")
    print("=" * 60)
    
    split_idx = int(len(spy) * 0.8)
    vix_test = spy['VIX'].values[split_idx:]
    
    regimes = {
        'Low Vol (VIX<20)': vix_test < 20,
        'Normal (20≤VIX<25)': (vix_test >= 20) & (vix_test < 25),
        'High Vol (25≤VIX<35)': (vix_test >= 25) & (vix_test < 35),
        'Crisis (VIX≥35)': vix_test >= 35
    }
    
    results = {}
    print(f"\n  {'Regime':25s} | {'샘플':6s} | {'R²':8s} | {'방향정확도':10s}")
    print("  " + "-" * 60)
    
    for regime, mask in regimes.items():
        if mask.sum() >= 10:
            r2 = r2_score(y_vrp_test[mask], vrp_pred[mask])
            
            vrp_mean = y_vrp_test.mean()
            dir_acc = ((y_vrp_test[mask] > vrp_mean) == (vrp_pred[mask] > vrp_mean)).mean()
            
            print(f"  {regime:25s} | {mask.sum():6d} | {r2:8.4f} | {dir_acc*100:8.1f}%")
            results[regime] = {'r2': float(r2), 'n_samples': int(mask.sum()), 'direction_acc': float(dir_acc)}
        else:
            print(f"  {regime:25s} | {mask.sum():6d} | 샘플 부족")
    
    return results


def validation_3_yearly(spy, feature_cols, vrp_pred, y_vrp_test, test_dates):
    """검증 3: 연도별 안정성"""
    print("\n" + "=" * 60)
    print("[3/6] 연도별 안정성")
    print("=" * 60)
    
    test_df = pd.DataFrame({
        'actual': y_vrp_test,
        'pred': vrp_pred
    }, index=test_dates)
    
    test_df['year'] = test_df.index.year
    
    results = {}
    print(f"\n  {'연도':6s} | {'샘플':6s} | {'R²':8s} | {'방향정확도':10s}")
    print("  " + "-" * 40)
    
    for year in sorted(test_df['year'].unique()):
        mask = test_df['year'] == year
        if mask.sum() >= 10:
            r2 = r2_score(test_df.loc[mask, 'actual'], test_df.loc[mask, 'pred'])
            
            vrp_mean = y_vrp_test.mean()
            dir_acc = ((test_df.loc[mask, 'actual'] > vrp_mean) == (test_df.loc[mask, 'pred'] > vrp_mean)).mean()
            
            print(f"  {year:6d} | {mask.sum():6d} | {r2:8.4f} | {dir_acc*100:8.1f}%")
            results[year] = {'r2': float(r2), 'n_samples': int(mask.sum()), 'direction_acc': float(dir_acc)}
    
    return results


def validation_4_feature_importance(spy, feature_cols):
    """검증 4: 특성 중요도"""
    print("\n" + "=" * 60)
    print("[4/6] 특성 중요도")
    print("=" * 60)
    
    X = spy[feature_cols].values
    y_rv = spy['RV_future'].values
    
    split_idx = int(len(spy) * 0.8)
    X_train = X[:split_idx]
    y_train = y_rv[:split_idx]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    
    en = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=SEED, max_iter=10000)
    en.fit(X_train_s, y_train)
    
    importance = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': np.abs(en.coef_)
    }).sort_values('coefficient', ascending=False)
    
    print(f"\n  📊 특성 중요도 (절대 계수):")
    for _, row in importance.iterrows():
        bar = "█" * int(row['coefficient'] * 10)
        print(f"     {row['feature']:15s}: {row['coefficient']:.4f} {bar}")
    
    return importance.to_dict('records')


def validation_5_direction_metrics(vrp_pred, y_vrp_test):
    """검증 5: 방향 예측 정밀도/재현율"""
    print("\n" + "=" * 60)
    print("[5/6] 방향 예측 메트릭")
    print("=" * 60)
    
    vrp_mean = y_vrp_test.mean()
    
    # 이진 분류: VRP > 평균
    y_actual = (y_vrp_test > vrp_mean).astype(int)
    y_pred = (vrp_pred > vrp_mean).astype(int)
    
    accuracy = accuracy_score(y_actual, y_pred)
    precision = precision_score(y_actual, y_pred)
    recall = recall_score(y_actual, y_pred)
    f1 = f1_score(y_actual, y_pred)
    
    print(f"\n  📊 이진 분류 (VRP > {vrp_mean:.2f}%):")
    print(f"     정확도:    {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"     정밀도:    {precision:.4f}")
    print(f"     재현율:    {recall:.4f}")
    print(f"     F1 Score:  {f1:.4f}")
    
    # 3분위 분류
    q33 = np.percentile(y_vrp_test, 33)
    q67 = np.percentile(y_vrp_test, 67)
    
    y_actual_3 = np.where(y_vrp_test < q33, 0, np.where(y_vrp_test < q67, 1, 2))
    y_pred_3 = np.where(vrp_pred < q33, 0, np.where(vrp_pred < q67, 1, 2))
    
    acc_3 = accuracy_score(y_actual_3, y_pred_3)
    print(f"\n  📊 3분위 분류:")
    print(f"     정확도:    {acc_3:.4f} ({acc_3*100:.1f}%)")
    
    return {
        'binary': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        },
        'tertile': {
            'accuracy': float(acc_3)
        }
    }


def validation_6_trading_simulation(vrp_pred, y_vrp_test):
    """검증 6: 트레이딩 시뮬레이션"""
    print("\n" + "=" * 60)
    print("[6/6] 트레이딩 시뮬레이션")
    print("=" * 60)
    
    vrp_mean = y_vrp_test.mean()
    
    # 전략: 예측 VRP > 평균이면 변동성 매도 (VRP 수취)
    positions = (vrp_pred > vrp_mean).astype(int)
    
    # 수익 = 포지션 * 실제 VRP
    returns = positions * y_vrp_test
    
    # 통계
    n_trades = positions.sum()
    total_return = returns.sum()
    avg_return = returns[positions == 1].mean() if n_trades > 0 else 0
    win_rate = (returns[positions == 1] > 0).mean() if n_trades > 0 else 0
    
    # Buy & Hold (항상 매도)
    bh_total = y_vrp_test.sum()
    bh_avg = y_vrp_test.mean()
    
    print(f"\n  📊 VRP 매도 전략 (예측 VRP > 평균시):")
    print(f"     거래 횟수:     {n_trades}/{len(positions)} ({n_trades/len(positions)*100:.1f}%)")
    print(f"     총 VRP 수취:   {total_return:.2f}%")
    print(f"     평균 VRP 수취: {avg_return:.2f}%")
    print(f"     승률:          {win_rate*100:.1f}%")
    
    print(f"\n  📊 Buy & Hold (항상 매도):")
    print(f"     총 VRP 수취:   {bh_total:.2f}%")
    print(f"     평균 VRP 수취: {bh_avg:.2f}%")
    
    outperformance = avg_return - bh_avg
    print(f"\n  📊 전략 vs Buy & Hold:")
    print(f"     초과 수익: {outperformance:+.2f}%/거래")
    
    return {
        'strategy': {
            'n_trades': int(n_trades),
            'total_return': float(total_return),
            'avg_return': float(avg_return),
            'win_rate': float(win_rate)
        },
        'buy_hold': {
            'total_return': float(bh_total),
            'avg_return': float(bh_avg)
        },
        'outperformance': float(outperformance)
    }


def main():
    print("\n" + "🔬" * 30)
    print("VRP 예측 모델 검증 및 추가 분석")
    print("🔬" * 30)
    
    # 데이터 준비 및 예측
    print("\n데이터 준비...")
    spy, feature_cols = load_data_and_train()
    print(f"  ✓ 데이터: {len(spy)} 행")
    
    # 검증 1: Bootstrap
    bootstrap_results, vrp_pred, y_vrp_test, test_dates = validation_1_bootstrap(spy, feature_cols)
    
    # 검증 2: Regime
    regime_results = validation_2_regime(spy, feature_cols, vrp_pred, y_vrp_test, test_dates)
    
    # 검증 3: 연도별
    yearly_results = validation_3_yearly(spy, feature_cols, vrp_pred, y_vrp_test, test_dates)
    
    # 검증 4: 특성 중요도
    importance_results = validation_4_feature_importance(spy, feature_cols)
    
    # 검증 5: 방향 메트릭
    direction_results = validation_5_direction_metrics(vrp_pred, y_vrp_test)
    
    # 검증 6: 트레이딩
    trading_results = validation_6_trading_simulation(vrp_pred, y_vrp_test)
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📊 최종 요약")
    print("=" * 60)
    
    print(f"""
    🎯 VRP 예측 성능:
       R² = {bootstrap_results['point_estimate']:.4f}
       95% CI: [{bootstrap_results['ci_lower']:.4f}, {bootstrap_results['ci_upper']:.4f}]
    
    📊 방향 예측:
       정확도 = {direction_results['binary']['accuracy']*100:.1f}%
       F1 Score = {direction_results['binary']['f1']:.4f}
    
    💰 트레이딩 성과:
       전략 평균 수익: {trading_results['strategy']['avg_return']:.2f}%
       Buy & Hold 대비: {trading_results['outperformance']:+.2f}%/거래
       승률: {trading_results['strategy']['win_rate']*100:.1f}%
    """)
    
    # 저장
    output = {
        'bootstrap': bootstrap_results,
        'regime': regime_results,
        'yearly': {str(k): v for k, v in yearly_results.items()},
        'feature_importance': importance_results,
        'direction_metrics': direction_results,
        'trading': trading_results,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('paper/vrp_validation_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"💾 결과 저장: paper/vrp_validation_results.json")


if __name__ == '__main__':
    main()
