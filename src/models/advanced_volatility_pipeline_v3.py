#!/usr/bin/env python3
"""
고급 변동성 예측 파이프라인 v3.0
=================================

구현 내용:
1. HAR-RV 피처 (1일/5일/22일 변동성)
2. GARCH(1,1) 필터링 및 잔차 추출
3. GARCH-LSTM 하이브리드 모델
4. 확률적 예측 (분포 추정)

예상 실행 시간: 10-15분
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 선택적 임포트
try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False
    print("⚠️ arch 패키지 없음. GARCH 기능 비활성화.")

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠️ PyTorch 없음. LSTM 기능 비활성화.")

SEED = 42
np.random.seed(SEED)
if HAS_TORCH:
    torch.manual_seed(SEED)


# =============================================================================
# 1. HAR-RV 피처 엔지니어링
# =============================================================================

class HARFeatureEngineer:
    """HAR-RV (Heterogeneous Autoregressive Realized Volatility) 피처"""
    
    def __init__(self):
        self.windows = {
            'daily': 1,      # 일별
            'weekly': 5,     # 주간
            'monthly': 22    # 월간
        }
    
    def create_har_features(self, df):
        """HAR 피처 생성"""
        print("  → HAR 피처 생성...")
        
        # Realized Volatility 계산 (일별 수익률 제곱의 합)
        df['returns'] = df['Close'].pct_change()
        df['returns_sq'] = df['returns'] ** 2
        
        # 다양한 윈도우의 RV
        for name, window in self.windows.items():
            if window == 1:
                df[f'rv_{name}'] = df['returns_sq']
            else:
                df[f'rv_{name}'] = df['returns_sq'].rolling(window).mean()
            
            # 래그된 RV (t-1)
            df[f'rv_{name}_lag1'] = df[f'rv_{name}'].shift(1)
        
        # HAR 모델의 핵심 피처: RV_d(t-1), RV_w(t-1), RV_m(t-1)
        df['har_rv_d'] = df['rv_daily'].shift(1)       # 어제
        df['har_rv_w'] = df['rv_weekly'].shift(1)      # 지난 주 평균
        df['har_rv_m'] = df['rv_monthly'].shift(1)     # 지난 달 평균
        
        # HAR 비율 (상대적 변동성 수준)
        df['har_ratio_w_d'] = df['har_rv_w'] / (df['har_rv_d'] + 1e-10)
        df['har_ratio_m_d'] = df['har_rv_m'] / (df['har_rv_d'] + 1e-10)
        df['har_ratio_m_w'] = df['har_rv_m'] / (df['har_rv_w'] + 1e-10)
        
        # 변동성 변화
        df['har_rv_d_change'] = df['har_rv_d'].pct_change()
        df['har_rv_w_change'] = df['har_rv_w'].pct_change()
        
        # Jump 성분 (급격한 변동성 변화)
        df['har_jump'] = np.maximum(df['har_rv_d'] - df['har_rv_w'], 0)
        
        print(f"    - HAR 피처 10개 생성 완료")
        return df
    
    def create_realized_variance(self, df):
        """다양한 Realized Variance 추정기"""
        print("  → Realized Variance 추정...")
        
        # Parkinson (High-Low 기반)
        df['rv_parkinson'] = (1 / (4 * np.log(2))) * (
            np.log(df['High'] / df['Low']) ** 2
        )
        
        # Garman-Klass
        log_hl = np.log(df['High'] / df['Low'])
        log_co = np.log(df['Close'] / df['Open'])
        df['rv_garman_klass'] = 0.5 * log_hl**2 - (2*np.log(2) - 1) * log_co**2
        
        # Rogers-Satchell (drift 고려)
        log_ho = np.log(df['High'] / df['Open'])
        log_lo = np.log(df['Low'] / df['Open'])
        log_co = np.log(df['Close'] / df['Open'])
        df['rv_rogers_satchell'] = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
        
        # 롤링 평균
        for rv_col in ['rv_parkinson', 'rv_garman_klass', 'rv_rogers_satchell']:
            for w in [5, 10, 20]:
                df[f'{rv_col}_{w}d'] = df[rv_col].rolling(w).mean()
        
        print(f"    - RV 추정기 12개 생성 완료")
        return df


# =============================================================================
# 2. GARCH 필터링
# =============================================================================

class GARCHFilter:
    """GARCH(1,1) 필터링 및 잔차 추출"""
    
    def __init__(self, p=1, q=1):
        self.p = p
        self.q = q
        self.model = None
        self.result = None
    
    def fit_filter(self, returns):
        """GARCH 모델 피팅 및 조건부 변동성 추출"""
        if not HAS_ARCH:
            return None, None, None
        
        print("  → GARCH(1,1) 필터링...")
        
        # 수익률을 퍼센트로 변환
        returns_pct = returns.dropna() * 100
        
        try:
            # GARCH(1,1) 모델
            model = arch_model(returns_pct, vol='Garch', p=self.p, q=self.q,
                              mean='Constant', rescale=False)
            result = model.fit(disp='off', show_warning=False)
            
            # 조건부 변동성 (연간화)
            cond_vol = result.conditional_volatility / 100
            
            # 표준화 잔차 (GARCH로 설명되지 않는 부분)
            std_residuals = result.std_resid
            
            # 모델 파라미터
            params = {
                'omega': result.params['omega'],
                'alpha': result.params['alpha[1]'],
                'beta': result.params['beta[1]']
            }
            
            print(f"    - GARCH 파라미터: α={params['alpha']:.4f}, β={params['beta']:.4f}")
            print(f"    - 지속성 (α+β): {params['alpha'] + params['beta']:.4f}")
            
            self.result = result
            return cond_vol, std_residuals, params
            
        except Exception as e:
            print(f"    ⚠️ GARCH 피팅 오류: {e}")
            return None, None, None
    
    def create_garch_features(self, df):
        """GARCH 기반 피처 생성"""
        if not HAS_ARCH:
            return df
        
        print("  → GARCH 피처 생성...")
        
        cond_vol, std_resid, params = self.fit_filter(df['returns'])
        
        if cond_vol is not None:
            # 조건부 변동성
            df['garch_vol'] = np.nan
            df.loc[cond_vol.index, 'garch_vol'] = cond_vol.values
            df['garch_vol'] = df['garch_vol'].ffill()
            
            # GARCH 변동성 래그
            df['garch_vol_lag1'] = df['garch_vol'].shift(1)
            df['garch_vol_lag5'] = df['garch_vol'].shift(5)
            
            # 표준화 잔차 (LSTM이 학습할 비선형 패턴)
            if std_resid is not None:
                df['garch_residual'] = np.nan
                df.loc[std_resid.index, 'garch_residual'] = std_resid.values
                df['garch_residual'] = df['garch_residual'].ffill()
                
                # 잔차의 절대값 및 제곱
                df['garch_resid_abs'] = np.abs(df['garch_residual'])
                df['garch_resid_sq'] = df['garch_residual'] ** 2
            
            # GARCH vs Realized Vol 비율
            df['garch_rv_ratio'] = df['garch_vol'] / (df['rv_weekly'] + 1e-10)
            
            print(f"    - GARCH 피처 7개 생성 완료")
        
        return df


# =============================================================================
# 3. LSTM 모델
# =============================================================================

class LSTMVolatilityModel(nn.Module):
    """LSTM 변동성 예측 모델"""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # 마지막 타임스텝의 출력 사용
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output)


class GARCHLSTMHybrid:
    """GARCH-LSTM 하이브리드 모델"""
    
    def __init__(self, seq_length=20, hidden_size=64, epochs=50, lr=0.001):
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr
        self.model = None
        self.scaler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def prepare_sequences(self, X, y):
        """시계열 시퀀스 생성"""
        X_seq, y_seq = [], []
        for i in range(len(X) - self.seq_length):
            X_seq.append(X[i:i+self.seq_length])
            y_seq.append(y[i+self.seq_length])
        return np.array(X_seq), np.array(y_seq)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """모델 학습"""
        if not HAS_TORCH:
            print("  ⚠️ PyTorch 없음. LSTM 학습 불가.")
            return
        
        print(f"  → LSTM 학습 시작 (epochs={self.epochs})...")
        
        # 스케일링
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # 시퀀스 생성
        X_seq, y_seq = self.prepare_sequences(X_train_scaled, y_train.values)
        
        # 텐서 변환
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).unsqueeze(1).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # 모델 초기화
        input_size = X_train.shape[1]
        self.model = LSTMVolatilityModel(input_size, self.hidden_size).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        
        # 학습
        best_loss = float('inf')
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(loader)
            scheduler.step(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
        
        print(f"    - 최종 Loss: {best_loss:.6f}")
    
    def predict(self, X_test):
        """예측"""
        if not HAS_TORCH or self.model is None:
            return None
        
        self.model.eval()
        X_test_scaled = self.scaler.transform(X_test)
        X_seq, _ = self.prepare_sequences(X_test_scaled, pd.Series(np.zeros(len(X_test))))
        
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy().flatten()
        
        return predictions


# =============================================================================
# 4. 확률적 예측 (분포 추정)
# =============================================================================

class ProbabilisticVolatility:
    """확률적 변동성 예측 (평균 + 분산 추정)"""
    
    def __init__(self):
        self.mean_model = None
        self.var_model = None
    
    def fit(self, X_train, y_train):
        """평균과 분산 모델 학습"""
        print("  → 확률적 예측 모델 학습...")
        
        # 평균 예측 모델
        self.mean_model = GradientBoostingRegressor(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            random_state=SEED
        )
        self.mean_model.fit(X_train, y_train)
        
        # 잔차 계산
        y_pred_mean = self.mean_model.predict(X_train)
        residuals = np.abs(y_train - y_pred_mean)
        
        # 분산 예측 모델 (잔차의 절대값 예측)
        self.var_model = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            random_state=SEED
        )
        self.var_model.fit(X_train, residuals)
        
        print("    - 평균/분산 모델 학습 완료")
    
    def predict(self, X_test, confidence=0.95):
        """확률적 예측 (평균, 하한, 상한)"""
        from scipy import stats
        
        y_mean = self.mean_model.predict(X_test)
        y_std = self.var_model.predict(X_test)
        
        # 신뢰 구간 (정규 분포 가정)
        z = stats.norm.ppf((1 + confidence) / 2)
        y_lower = y_mean - z * y_std
        y_upper = y_mean + z * y_std
        
        return y_mean, y_lower, y_upper, y_std


# =============================================================================
# 5. 통합 파이프라인
# =============================================================================

class AdvancedVolatilityPipeline:
    """고급 변동성 예측 파이프라인 v3.0"""
    
    def __init__(self, start_date='2015-01-01', end_date='2024-12-31'):
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.feature_cols = []
        self.results = {}
        
    def load_data(self):
        """데이터 로드"""
        print("\n" + "="*60)
        print("[1/7] 데이터 로드...")
        print("="*60)
        
        tickers = {
            'SPY': 'SPY',
            'VIX': '^VIX',
        }
        
        all_data = {}
        for name, ticker in tickers.items():
            df = yf.download(ticker, start=self.start_date, end=self.end_date,
                           progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            all_data[name] = df
            print(f"  ✓ {name}: {len(df)} 행")
        
        self.data = all_data['SPY'].copy()
        self.data['VIX'] = all_data['VIX']['Close']
        self.data = self.data.ffill().dropna()
        
        print(f"\n  ✓ 최종: {len(self.data)} 행")
        return self.data
    
    def engineer_features(self):
        """피처 엔지니어링"""
        print("\n" + "="*60)
        print("[2/7] 피처 엔지니어링...")
        print("="*60)
        
        df = self.data.copy()
        
        # 1. HAR 피처
        har = HARFeatureEngineer()
        df = har.create_har_features(df)
        df = har.create_realized_variance(df)
        
        # 2. GARCH 피처
        garch = GARCHFilter()
        df = garch.create_garch_features(df)
        
        # 3. VIX 피처 (기존)
        print("  → VIX 피처 생성...")
        df['vix_lag1'] = df['VIX'].shift(1)
        df['vix_lag5'] = df['VIX'].shift(5)
        df['vix_change'] = df['VIX'].pct_change()
        df['vix_zscore'] = (df['VIX'] - df['VIX'].rolling(20).mean()) / (df['VIX'].rolling(20).std() + 1e-10)
        
        # 4. Regime 피처 (기존)
        print("  → Regime 피처 생성...")
        vix_lag = df['VIX'].shift(1)
        df['regime_high_vol'] = (vix_lag >= 25).astype(int)
        df['regime_crisis'] = (vix_lag >= 35).astype(int)
        df['vol_in_high_regime'] = df['regime_high_vol'] * df['rv_weekly']
        df['vix_excess_25'] = np.maximum(vix_lag - 25, 0)
        
        self.data = df
        print(f"\n  ✓ 총 {len(df.columns)} 컬럼 생성")
        return df
    
    def create_target(self, horizon=5):
        """타겟 생성"""
        print("\n" + "="*60)
        print(f"[3/7] 타겟 생성 (horizon={horizon})...")
        print("="*60)
        
        df = self.data.copy()
        
        # 미래 Realized Variance (t+1 ~ t+horizon)
        future_rv = []
        returns_sq = df['returns_sq'].values
        
        for i in range(len(returns_sq)):
            if i + horizon < len(returns_sq):
                future_rv.append(np.mean(returns_sq[i+1:i+1+horizon]))
            else:
                future_rv.append(np.nan)
        
        df['target_rv'] = future_rv
        df['target_vol'] = np.sqrt(df['target_rv'])  # 변동성으로 변환
        
        self.data = df
        print(f"  ✓ 타겟 생성 완료 (평균: {np.nanmean(df['target_vol']):.6f})")
        return df
    
    def select_features(self):
        """피처 선택"""
        print("\n" + "="*60)
        print("[4/7] 피처 선택...")
        print("="*60)
        
        exclude = ['Open', 'High', 'Low', 'Close', 'Volume', 'VIX',
                   'returns', 'returns_sq', 'target_rv', 'target_vol']
        
        self.feature_cols = [c for c in self.data.columns 
                            if c not in exclude and not c.startswith('rv_daily')]
        
        # 타겟과의 상관관계 분석
        df_clean = self.data.dropna()
        correlations = df_clean[self.feature_cols].corrwith(df_clean['target_vol']).abs()
        correlations = correlations.sort_values(ascending=False)
        
        print("\n  📊 상위 15 피처 (타겟 상관관계):")
        for i, (feat, corr) in enumerate(correlations.head(15).items()):
            print(f"    {i+1}. {feat}: {corr:.4f}")
        
        # 상위 40개만 선택 (과적합 방지)
        self.feature_cols = correlations.head(40).index.tolist()
        
        print(f"\n  ✓ {len(self.feature_cols)}개 피처 선택됨")
        return self.feature_cols
    
    def prepare_data(self, test_ratio=0.2):
        """데이터 분할"""
        print("\n" + "="*60)
        print("[5/7] 데이터 분할...")
        print("="*60)
        
        df = self.data.dropna().copy()
        
        # 무한대 값 처리
        for col in self.feature_cols:
            if col in df.columns:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                df[col] = df[col].fillna(df[col].median())
        
        # 이상치 클리핑 (99.9 백분위수)
        for col in self.feature_cols:
            if col in df.columns:
                lower = df[col].quantile(0.001)
                upper = df[col].quantile(0.999)
                df[col] = df[col].clip(lower, upper)
        
        df = df.dropna()
        
        split_idx = int(len(df) * (1 - test_ratio))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        X_train = train_df[self.feature_cols]
        y_train = train_df['target_vol']
        X_test = test_df[self.feature_cols]
        y_test = test_df['target_vol']
        
        print(f"  ✓ Train: {len(X_train)}, Test: {len(X_test)}")
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.test_df = test_df
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self):
        """모델 학습"""
        print("\n" + "="*60)
        print("[6/7] 모델 학습...")
        print("="*60)
        
        # 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        X_test_scaled = scaler.transform(self.X_test)
        
        results = {}
        
        # 1. ElasticNet (베이스라인)
        print("\n  [1] ElasticNet (베이스라인)...")
        en = ElasticNet(alpha=0.0005, l1_ratio=0.5, random_state=SEED, max_iter=10000)
        en.fit(X_train_scaled, self.y_train)
        y_pred_en = en.predict(X_test_scaled)
        results['ElasticNet'] = {
            'r2': r2_score(self.y_test, y_pred_en),
            'rmse': np.sqrt(mean_squared_error(self.y_test, y_pred_en)),
            'predictions': y_pred_en
        }
        print(f"      R²: {results['ElasticNet']['r2']:.4f}")
        
        # 2. GradientBoosting
        print("  [2] GradientBoosting...")
        gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                                       random_state=SEED)
        gb.fit(X_train_scaled, self.y_train)
        y_pred_gb = gb.predict(X_test_scaled)
        results['GradientBoosting'] = {
            'r2': r2_score(self.y_test, y_pred_gb),
            'rmse': np.sqrt(mean_squared_error(self.y_test, y_pred_gb)),
            'predictions': y_pred_gb
        }
        print(f"      R²: {results['GradientBoosting']['r2']:.4f}")
        
        # 3. GARCH-LSTM 하이브리드
        if HAS_TORCH:
            print("  [3] GARCH-LSTM 하이브리드...")
            lstm = GARCHLSTMHybrid(seq_length=20, hidden_size=64, epochs=30)
            lstm.fit(self.X_train, self.y_train)
            y_pred_lstm = lstm.predict(self.X_test)
            if y_pred_lstm is not None and len(y_pred_lstm) > 0:
                # 시퀀스로 인한 길이 차이 조정
                y_test_lstm = self.y_test.values[20:]
                y_pred_lstm = y_pred_lstm[:len(y_test_lstm)]
                results['GARCH-LSTM'] = {
                    'r2': r2_score(y_test_lstm, y_pred_lstm),
                    'rmse': np.sqrt(mean_squared_error(y_test_lstm, y_pred_lstm)),
                    'predictions': y_pred_lstm
                }
                print(f"      R²: {results['GARCH-LSTM']['r2']:.4f}")
        
        # 4. 확률적 예측
        print("  [4] 확률적 예측...")
        prob = ProbabilisticVolatility()
        prob.fit(X_train_scaled, self.y_train)
        y_mean, y_lower, y_upper, y_std = prob.predict(X_test_scaled)
        results['Probabilistic'] = {
            'r2': r2_score(self.y_test, y_mean),
            'rmse': np.sqrt(mean_squared_error(self.y_test, y_mean)),
            'predictions': y_mean,
            'lower': y_lower,
            'upper': y_upper,
            'std': y_std
        }
        print(f"      R²: {results['Probabilistic']['r2']:.4f}")
        
        # 결과 저장
        self.results = results
        self.scaler = scaler
        
        return results
    
    def evaluate(self):
        """결과 평가"""
        print("\n" + "="*60)
        print("[7/7] 결과 평가...")
        print("="*60)
        
        print("\n" + "-"*60)
        print("📊 모델 성능 비교:")
        print("-"*60)
        print(f"{'모델':<20} {'R²':>12} {'RMSE':>12}")
        print("-"*60)
        
        for name, res in sorted(self.results.items(), 
                                key=lambda x: x[1]['r2'], reverse=True):
            print(f"{name:<20} {res['r2']:>12.4f} {res['rmse']:>12.6f}")
        
        # 최고 모델
        best_name = max(self.results, key=lambda x: self.results[x]['r2'])
        best_r2 = self.results[best_name]['r2']
        
        print(f"\n  🏆 최고 모델: {best_name} (R² = {best_r2:.4f})")
        
        return best_name, best_r2
    
    def save_results(self, best_name):
        """결과 저장"""
        print("\n" + "="*60)
        print("결과 저장...")
        print("="*60)
        
        model_dir = Path('data/models')
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 메트릭 저장
        metrics = {
            'model_name': f'Advanced {best_name}',
            'test_r2': float(self.results[best_name]['r2']),
            'test_rmse': float(self.results[best_name]['rmse']),
            'n_features': len(self.feature_cols),
            'methods_used': ['HAR-RV', 'GARCH', 'LSTM', 'Probabilistic'],
            'all_results': {k: {'r2': float(v['r2'])} for k, v in self.results.items()},
            'timestamp': datetime.now().isoformat()
        }
        
        with open('data/raw/advanced_model_performance.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"  ✓ 메트릭 저장됨")
        return metrics
    
    def run(self):
        """전체 파이프라인 실행"""
        start = datetime.now()
        print("\n" + "🚀"*30)
        print("고급 변동성 예측 파이프라인 v3.0")
        print("🚀"*30)
        
        self.load_data()
        self.engineer_features()
        self.create_target()
        self.select_features()
        self.prepare_data()
        self.train_models()
        best_name, best_r2 = self.evaluate()
        metrics = self.save_results(best_name)
        
        elapsed = datetime.now() - start
        print("\n" + "="*60)
        print("✅ 완료!")
        print("="*60)
        print(f"  ⏱️ 소요 시간: {elapsed}")
        print(f"  🏆 최고 모델: {best_name}")
        print(f"  📊 Test R²: {best_r2:.4f}")
        
        return metrics


def main():
    pipeline = AdvancedVolatilityPipeline(
        start_date='2015-01-01',
        end_date='2024-12-31'
    )
    metrics = pipeline.run()
    return metrics


if __name__ == '__main__':
    metrics = main()
