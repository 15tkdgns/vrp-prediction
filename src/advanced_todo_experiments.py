"""
심화 실험 - todo_temp.txt 기반
==============================
1. 자산별 특화 VIX 적용 (GLD->GVZ, TLT->TYVIX, EEM->VXEEM)
2. VIX 변동성 기간구조 활용 (VIX9D, VIX, VIX3M)
3. 레짐 스위칭 가중치 전략
4. Fleming et al. (2001) 효용 함수 기반 성능료
"""
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def calculate_rv(returns, window):
    rv = (returns ** 2).rolling(window).sum() * 252
    return rv.iloc[:, 0] if isinstance(rv, pd.DataFrame) else rv

# ============================================================================
# 1. 자산별 특화 VIX 적용
# ============================================================================

def asset_specific_vix_experiment():
    """자산별 특화 VIX 지수 활용"""
    print("\n" + "="*60)
    print("[1] 자산별 특화 VIX 실험")
    print("="*60)
    
    # 자산-VIX 매핑
    asset_vix_mapping = {
        'SPY': '^VIX',      # S&P 500 VIX
        'QQQ': '^VIX',      # QQQ도 VIX 사용
        'GLD': '^GVZ',      # Gold VIX
        'USO': '^OVX',      # Oil VIX
        'TLT': '^VIX',      # TYVIX 대용 (데이터 제한)
        'EEM': '^VIX',      # VXEEM 대용
        'XLF': '^VIX',
        'XLK': '^VIX',
    }
    
    results = {}
    
    for ticker, vix_ticker in asset_vix_mapping.items():
        print(f"\n  {ticker} (VIX: {vix_ticker})...")
        
        try:
            # 자산 데이터
            data = yf.download(ticker, start='2015-01-01', end='2025-01-01', progress=False)
            if len(data) < 500:
                print("    SKIP - insufficient data")
                continue
            
            returns = data['Close'].pct_change()
            if isinstance(returns, pd.DataFrame):
                returns = returns.iloc[:, 0]
            
            rv_5d = calculate_rv(returns, 5)
            rv_22d = calculate_rv(returns, 22)
            
            # 특화 VIX 다운로드
            vix_data = yf.download(vix_ticker, start='2015-01-01', end='2025-01-01', progress=False)
            vix_close = vix_data['Close'].iloc[:, 0] if isinstance(vix_data['Close'], pd.DataFrame) else vix_data['Close']
            
            # 일반 VIX도 다운로드 (비교용)
            general_vix = yf.download('^VIX', start='2015-01-01', end='2025-01-01', progress=False)
            general_vix_close = general_vix['Close'].iloc[:, 0] if isinstance(general_vix['Close'], pd.DataFrame) else general_vix['Close']
            
            features = pd.DataFrame(index=data.index)
            features['RV_5d_lag1'] = rv_5d.shift(1)
            features['RV_22d_lag1'] = rv_22d.shift(1)
            features['VIX_specific_lag1'] = vix_close.reindex(data.index).ffill().shift(1)
            features['VIX_general_lag1'] = general_vix_close.reindex(data.index).ffill().shift(1)
            features['RV_5d_future'] = rv_5d.shift(-5)
            features = features.dropna()
            
            if len(features) < 500:
                print("    SKIP - insufficient aligned data")
                continue
            
            gap = 5
            n = len(features)
            train_end = int(n * 0.7) - gap
            
            # 특화 VIX 모델
            X_specific = features[['RV_5d_lag1', 'RV_22d_lag1', 'VIX_specific_lag1']].iloc[:train_end]
            X_specific_test = features[['RV_5d_lag1', 'RV_22d_lag1', 'VIX_specific_lag1']].iloc[train_end+gap:]
            
            # 일반 VIX 모델
            X_general = features[['RV_5d_lag1', 'RV_22d_lag1', 'VIX_general_lag1']].iloc[:train_end]
            X_general_test = features[['RV_5d_lag1', 'RV_22d_lag1', 'VIX_general_lag1']].iloc[train_end+gap:]
            
            y_train = features['RV_5d_future'].iloc[:train_end]
            y_test = features['RV_5d_future'].iloc[train_end+gap:]
            
            scaler_s = StandardScaler()
            scaler_g = StandardScaler()
            
            X_s_train = scaler_s.fit_transform(X_specific)
            X_s_test = scaler_s.transform(X_specific_test)
            X_g_train = scaler_g.fit_transform(X_general)
            X_g_test = scaler_g.transform(X_general_test)
            
            model_s = Ridge(alpha=100.0)
            model_g = Ridge(alpha=100.0)
            
            model_s.fit(X_s_train, np.sqrt(y_train))
            model_g.fit(X_g_train, np.sqrt(y_train))
            
            pred_s = np.maximum(model_s.predict(X_s_test) ** 2, 0)
            pred_g = np.maximum(model_g.predict(X_g_test) ** 2, 0)
            
            r2_specific = r2_score(y_test, pred_s)
            r2_general = r2_score(y_test, pred_g)
            
            results[ticker] = {
                'r2_specific_vix': r2_specific,
                'r2_general_vix': r2_general,
                'improvement': r2_specific - r2_general,
                'vix_used': vix_ticker
            }
            
            print(f"    Specific VIX R2: {r2_specific:.4f}")
            print(f"    General VIX R2: {r2_general:.4f}")
            print(f"    Improvement: {r2_specific - r2_general:+.4f}")
            
        except Exception as e:
            print(f"    Error: {e}")
    
    return results

# ============================================================================
# 2. VIX 변동성 기간구조
# ============================================================================

def vix_term_structure_experiment():
    """VIX 기간구조 활용 (단기/중기/장기)"""
    print("\n" + "="*60)
    print("[2] VIX 기간구조 실험")
    print("="*60)
    
    # VIX 기간구조 지수
    vix_indices = {
        'VIX9D': '^VIX9D',   # 9일
        'VIX': '^VIX',       # 30일
        'VIX3M': '^VIX3M',   # 3개월
    }
    
    results = {}
    
    # SPY로 테스트
    ticker = 'SPY'
    print(f"\n  {ticker}...")
    
    try:
        data = yf.download(ticker, start='2015-01-01', end='2025-01-01', progress=False)
        returns = data['Close'].pct_change()
        if isinstance(returns, pd.DataFrame):
            returns = returns.iloc[:, 0]
        
        rv_5d = calculate_rv(returns, 5)
        rv_22d = calculate_rv(returns, 22)
        
        features = pd.DataFrame(index=data.index)
        features['RV_5d_lag1'] = rv_5d.shift(1)
        features['RV_22d_lag1'] = rv_22d.shift(1)
        
        # VIX 기간구조 데이터
        for name, symbol in vix_indices.items():
            vix = yf.download(symbol, start='2015-01-01', end='2025-01-01', progress=False)
            if len(vix) > 0:
                vix_close = vix['Close'].iloc[:, 0] if isinstance(vix['Close'], pd.DataFrame) else vix['Close']
                features[f'{name}_lag1'] = vix_close.reindex(data.index).ffill().shift(1)
        
        # 기간구조 파생 특성
        if 'VIX9D_lag1' in features.columns and 'VIX_lag1' in features.columns:
            features['VIX_slope_lag1'] = features['VIX_lag1'] - features['VIX9D_lag1']  # 기울기
        if 'VIX_lag1' in features.columns and 'VIX3M_lag1' in features.columns:
            features['VIX_curve_lag1'] = features['VIX3M_lag1'] - features['VIX_lag1']  # 장기 기울기
        
        features['RV_5d_future'] = rv_5d.shift(-5)
        features = features.dropna()
        
        gap = 5
        n = len(features)
        train_end = int(n * 0.7) - gap
        
        y_train = features['RV_5d_future'].iloc[:train_end]
        y_test = features['RV_5d_future'].iloc[train_end+gap:]
        
        # 기본 모델 (VIX만)
        X_basic = features[['RV_5d_lag1', 'RV_22d_lag1', 'VIX_lag1']].iloc[:train_end]
        X_basic_test = features[['RV_5d_lag1', 'RV_22d_lag1', 'VIX_lag1']].iloc[train_end+gap:]
        
        scaler_b = StandardScaler()
        X_b_train = scaler_b.fit_transform(X_basic)
        X_b_test = scaler_b.transform(X_basic_test)
        
        model_b = Ridge(alpha=100.0)
        model_b.fit(X_b_train, np.sqrt(y_train))
        pred_b = np.maximum(model_b.predict(X_b_test) ** 2, 0)
        r2_basic = r2_score(y_test, pred_b)
        
        # 기간구조 모델
        term_cols = ['RV_5d_lag1', 'RV_22d_lag1']
        for col in ['VIX9D_lag1', 'VIX_lag1', 'VIX3M_lag1', 'VIX_slope_lag1', 'VIX_curve_lag1']:
            if col in features.columns:
                term_cols.append(col)
        
        X_term = features[term_cols].iloc[:train_end]
        X_term_test = features[term_cols].iloc[train_end+gap:]
        
        scaler_t = StandardScaler()
        X_t_train = scaler_t.fit_transform(X_term)
        X_t_test = scaler_t.transform(X_term_test)
        
        model_t = Ridge(alpha=100.0)
        model_t.fit(X_t_train, np.sqrt(y_train))
        pred_t = np.maximum(model_t.predict(X_t_test) ** 2, 0)
        r2_term = r2_score(y_test, pred_t)
        
        results[ticker] = {
            'r2_basic': r2_basic,
            'r2_term_structure': r2_term,
            'improvement': r2_term - r2_basic,
            'features_used': term_cols
        }
        
        print(f"    Basic (VIX only) R2: {r2_basic:.4f}")
        print(f"    Term Structure R2: {r2_term:.4f}")
        print(f"    Improvement: {r2_term - r2_basic:+.4f}")
        
    except Exception as e:
        print(f"    Error: {e}")
    
    return results

# ============================================================================
# 3. 레짐 스위칭 가중치 전략
# ============================================================================

def regime_switching_strategy():
    """VIX 레짐별 포지션 사이징"""
    print("\n" + "="*60)
    print("[3] 레짐 스위칭 가중치 전략")
    print("="*60)
    
    ticker = 'SPY'
    
    data = yf.download(ticker, start='2015-01-01', end='2025-01-01', progress=False)
    returns = data['Close'].pct_change()
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    
    rv_5d = calculate_rv(returns, 5)
    
    vix = yf.download('^VIX', start='2015-01-01', end='2025-01-01', progress=False)
    vix_close = vix['Close'].iloc[:, 0] if isinstance(vix['Close'], pd.DataFrame) else vix['Close']
    vix_aligned = vix_close.reindex(data.index).ffill()
    
    features = pd.DataFrame(index=data.index)
    features['RV_5d_lag1'] = rv_5d.shift(1)
    features['VIX_lag1'] = vix_aligned.shift(1)
    features['RV_5d_future'] = rv_5d.shift(-5)
    features['returns'] = returns
    features = features.dropna()
    
    gap = 5
    n = len(features)
    train_end = int(n * 0.7) - gap
    test_start = train_end + gap
    
    test_data = features.iloc[test_start:]
    
    # 레짐 정의
    def get_regime(vix):
        if vix < 15:
            return 'low'
        elif vix < 25:
            return 'mid'
        else:
            return 'high'
    
    test_data = test_data.copy()
    test_data['regime'] = test_data['VIX_lag1'].apply(get_regime)
    
    # 레짐별 포지션 전략
    strategies = {
        'Buy & Hold': {'low': 1.0, 'mid': 1.0, 'high': 1.0},
        'Conservative': {'low': 1.0, 'mid': 0.7, 'high': 0.3},
        'Aggressive': {'low': 1.2, 'mid': 1.0, 'high': 0.5},
        'Inverse': {'low': 0.5, 'mid': 1.0, 'high': 1.5},
    }
    
    results = {}
    
    # 5일 수익률 계산
    returns_5d = []
    for i in range(0, len(test_data) - 5, 5):
        period_return = test_data['returns'].iloc[i:i+5].sum()
        returns_5d.append({
            'return': period_return,
            'regime': test_data['regime'].iloc[i]
        })
    
    returns_df = pd.DataFrame(returns_5d)
    
    for strategy_name, weights in strategies.items():
        strategy_returns = []
        for _, row in returns_df.iterrows():
            weight = weights[row['regime']]
            strategy_returns.append(row['return'] * weight)
        
        strategy_returns = np.array(strategy_returns)
        
        total_return = np.prod(1 + strategy_returns) - 1
        sharpe = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-8) * np.sqrt(52)
        max_dd = np.min(np.cumprod(1 + strategy_returns) / np.maximum.accumulate(np.cumprod(1 + strategy_returns)) - 1)
        
        results[strategy_name] = {
            'total_return': total_return,
            'sharpe': sharpe,
            'max_drawdown': max_dd
        }
        
        print(f"\n  {strategy_name}:")
        print(f"    Total Return: {total_return:.2%}")
        print(f"    Sharpe Ratio: {sharpe:.2f}")
        print(f"    Max Drawdown: {max_dd:.2%}")
    
    return results

# ============================================================================
# 4. Fleming et al. (2001) 효용 함수 기반 성능료
# ============================================================================

def utility_based_performance_fee():
    """효용 함수 기반 성능료 계산"""
    print("\n" + "="*60)
    print("[4] 효용 함수 기반 성능료 (Fleming et al., 2001)")
    print("="*60)
    
    ticker = 'SPY'
    
    data = yf.download(ticker, start='2015-01-01', end='2025-01-01', progress=False)
    returns = data['Close'].pct_change()
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    
    rv_5d = calculate_rv(returns, 5)
    
    vix = yf.download('^VIX', start='2015-01-01', end='2025-01-01', progress=False)
    vix_close = vix['Close'].iloc[:, 0] if isinstance(vix['Close'], pd.DataFrame) else vix['Close']
    vix_aligned = vix_close.reindex(data.index).ffill()
    
    features = pd.DataFrame(index=data.index)
    features['RV_5d_lag1'] = rv_5d.shift(1)
    features['VIX_lag1'] = vix_aligned.shift(1)
    features['RV_5d_future'] = rv_5d.shift(-5)
    features['returns'] = returns
    features = features.dropna()
    
    gap = 5
    n = len(features)
    train_end = int(n * 0.7) - gap
    test_start = train_end + gap
    
    test_data = features.iloc[test_start:]
    
    # 5일 수익률
    returns_5d = []
    for i in range(0, len(test_data) - 5, 5):
        period_return = test_data['returns'].iloc[i:i+5].sum()
        vix_level = test_data['VIX_lag1'].iloc[i]
        returns_5d.append({
            'return': period_return,
            'vix': vix_level
        })
    
    returns_df = pd.DataFrame(returns_5d)
    
    # Buy & Hold 수익률
    bh_returns = returns_df['return'].values
    
    # ML 전략 수익률 (VIX 기반 포지션 조절)
    ml_returns = []
    for _, row in returns_df.iterrows():
        if row['vix'] > 25:  # 고변동성: 50% 포지션
            weight = 0.5
        elif row['vix'] < 15:  # 저변동성: 100% 포지션
            weight = 1.0
        else:  # 중간: 80% 포지션
            weight = 0.8
        ml_returns.append(row['return'] * weight)
    
    ml_returns = np.array(ml_returns)
    
    # 위험 회피 계수별 효용 계산
    gammas = [2, 6, 10]  # 위험 회피 계수
    
    results = {}
    
    for gamma in gammas:
        # 2차 효용 함수: U = E[r] - (gamma/2) * Var[r]
        utility_bh = np.mean(bh_returns) - (gamma / 2) * np.var(bh_returns)
        utility_ml = np.mean(ml_returns) - (gamma / 2) * np.var(ml_returns)
        
        # 효용 증분 (연간화)
        utility_gain = (utility_ml - utility_bh) * 52  # 주간 → 연간
        
        # 성능료 (basis points)
        performance_fee_bps = utility_gain * 10000
        
        results[f'gamma_{gamma}'] = {
            'utility_bh': utility_bh,
            'utility_ml': utility_ml,
            'utility_gain_annual': utility_gain,
            'performance_fee_bps': performance_fee_bps
        }
        
        print(f"\n  Gamma = {gamma}:")
        print(f"    Utility (B&H): {utility_bh:.6f}")
        print(f"    Utility (ML): {utility_ml:.6f}")
        print(f"    Annual Utility Gain: {utility_gain:.4f}")
        print(f"    Performance Fee: {performance_fee_bps:.1f} bps")
    
    return results

# ============================================================================
# 메인
# ============================================================================

def main():
    print("="*80)
    print("심화 실험 - todo_temp.txt 기반")
    print("="*80)
    
    all_results = {}
    
    # 1. 자산별 특화 VIX
    all_results['asset_specific_vix'] = asset_specific_vix_experiment()
    
    # 2. VIX 기간구조
    all_results['vix_term_structure'] = vix_term_structure_experiment()
    
    # 3. 레짐 스위칭 전략
    all_results['regime_switching'] = regime_switching_strategy()
    
    # 4. 효용 함수 성능료
    all_results['utility_performance_fee'] = utility_based_performance_fee()
    
    # 저장
    output = {
        'metadata': {
            'experiment': 'Advanced Experiments from todo_temp.txt',
            'timestamp': datetime.now().isoformat()
        },
        'results': all_results
    }
    
    output_path = 'data/results/advanced_todo_experiments.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    
    print("\n" + "="*80)
    print("요약")
    print("="*80)
    
    print(f"\n결과 저장: {output_path}")

if __name__ == "__main__":
    main()
