#!/usr/bin/env python3
"""
통합 변동성 예측 파이프라인 v2.0
- VIX + 크로스에셋 특성
- GARCH 기반 특성
- 고급 Realized Volatility (Yang-Zhang)
- XGBoost/LightGBM 비선형 모델
- Stacking 앙상블
- Optuna 하이퍼파라미터 최적화

예상 실행 시간: 15-20분
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
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
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠️ XGBoost not installed. Using GradientBoosting instead.")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("⚠️ LightGBM not installed. Skipping.")

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("⚠️ Optuna not installed. Using default hyperparameters.")

try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False
    print("⚠️ arch package not installed. Skipping GARCH features.")

# 재현성 보장
SEED = 42
np.random.seed(SEED)


class EnhancedVolatilityPipeline:
    """통합 변동성 예측 파이프라인"""
    
    def __init__(self, start_date='2015-01-01', end_date='2024-12-31'):
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.features = None
        self.target = None
        self.feature_cols = []
        self.scaler = None
        self.best_model = None
        self.results = {}
        
    def load_multi_asset_data(self):
        """다중 자산 데이터 로드 (SPY, VIX, TLT, GLD, DXY 대용)"""
        print("\n" + "="*60)
        print("[1/7] 다중 자산 데이터 로드 중...")
        print("="*60)
        
        tickers = {
            'SPY': 'SPY',      # S&P 500 ETF
            'VIX': '^VIX',     # 변동성 지수
            'TLT': 'TLT',      # 20년 국채 ETF
            'GLD': 'GLD',      # 금 ETF
            'UUP': 'UUP',      # 달러 인덱스 ETF (DXY 대용)
            'HYG': 'HYG',      # 하이일드 채권 (리스크 지표)
        }
        
        all_data = {}
        
        for name, ticker in tickers.items():
            try:
                df = yf.download(ticker, start=self.start_date, end=self.end_date, 
                               progress=False, auto_adjust=True)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                all_data[name] = df
                print(f"  ✓ {name}: {len(df)} 행")
            except Exception as e:
                print(f"  ⚠️ {name} 로드 실패: {e}")
        
        # 기본 데이터프레임 (SPY)
        self.data = all_data['SPY'].copy()
        self.data.columns = [f'SPY_{col}' for col in self.data.columns]
        
        # 다른 자산 병합
        for name, df in all_data.items():
            if name != 'SPY':
                for col in ['Close', 'Volume']:
                    if col in df.columns:
                        self.data[f'{name}_{col}'] = df[col]
        
        # 결측치 처리 (전방 채움)
        self.data = self.data.ffill().dropna()
        
        print(f"\n  ✓ 최종 데이터: {len(self.data)} 행, {len(self.data.columns)} 열")
        return self.data
    
    def create_enhanced_features(self):
        """고급 특성 엔지니어링"""
        print("\n" + "="*60)
        print("[2/7] 고급 특성 엔지니어링...")
        print("="*60)
        
        df = self.data.copy()
        
        # 1. 기본 수익률 및 변동성
        print("  → 기본 수익률/변동성...")
        df['returns'] = df['SPY_Close'].pct_change()
        df['log_returns'] = np.log(df['SPY_Close'] / df['SPY_Close'].shift(1))
        
        # 2. 고급 Realized Volatility (Yang-Zhang)
        print("  → Yang-Zhang Realized Volatility...")
        df = self._add_yang_zhang_volatility(df)
        
        # 3. 다중 기간 변동성
        print("  → 다중 기간 변동성...")
        for window in [5, 10, 20, 50]:
            df[f'rv_{window}'] = df['returns'].rolling(window).std() * np.sqrt(252)
            df[f'rv_rank_{window}'] = df[f'rv_{window}'].rolling(60).rank(pct=True)
        
        # 변동성 비율
        df['rv_ratio_5_20'] = df['rv_5'] / (df['rv_20'] + 1e-8)
        df['rv_ratio_10_50'] = df['rv_10'] / (df['rv_50'] + 1e-8)
        
        # 4. VIX 관련 특성 (핵심!)
        print("  → VIX 특성...")
        if 'VIX_Close' in df.columns:
            df['vix'] = df['VIX_Close']
            df['vix_change'] = df['vix'].pct_change()
            df['vix_ma_5'] = df['vix'].rolling(5).mean()
            df['vix_ma_20'] = df['vix'].rolling(20).mean()
            df['vix_zscore'] = (df['vix'] - df['vix'].rolling(50).mean()) / (df['vix'].rolling(50).std() + 1e-8)
            
            # VIX 기간 구조 (단기 vs 장기)
            df['vix_term_structure'] = df['vix_ma_5'] / (df['vix_ma_20'] + 1e-8)
            
            # VIX vs Realized Vol (Variance Risk Premium proxy)
            df['vrp_proxy'] = df['vix'] / 100 - df['rv_20']
        
        # 5. 크로스에셋 특성
        print("  → 크로스에셋 특성...")
        cross_assets = ['TLT', 'GLD', 'UUP', 'HYG']
        for asset in cross_assets:
            col = f'{asset}_Close'
            if col in df.columns:
                df[f'{asset}_returns'] = df[col].pct_change()
                df[f'{asset}_vol_20'] = df[f'{asset}_returns'].rolling(20).std() * np.sqrt(252)
                # SPY와의 상관관계 (롤링)
                df[f'{asset}_corr_20'] = df['returns'].rolling(20).corr(df[f'{asset}_returns'])
        
        # 6. 수익률 통계
        print("  → 수익률 통계...")
        for window in [5, 10, 20]:
            df[f'return_mean_{window}'] = df['returns'].rolling(window).mean()
            df[f'return_skew_{window}'] = df['returns'].rolling(window).skew()
            df[f'return_kurt_{window}'] = df['returns'].rolling(window).kurt()
        
        # 7. 래그 특성
        print("  → 래그 특성...")
        for lag in [1, 2, 3, 5, 10]:
            df[f'return_lag_{lag}'] = df['returns'].shift(lag)
            df[f'rv_5_lag_{lag}'] = df['rv_5'].shift(lag)
            if 'vix' in df.columns:
                df[f'vix_lag_{lag}'] = df['vix'].shift(lag)
        
        # 8. 모멘텀 특성
        print("  → 모멘텀 특성...")
        for window in [5, 10, 20]:
            df[f'momentum_{window}'] = df['returns'].rolling(window).sum()
            df[f'abs_momentum_{window}'] = df['returns'].abs().rolling(window).sum()
        
        # 9. GARCH 특성 (선택적)
        if HAS_ARCH:
            print("  → GARCH(1,1) 특성...")
            df = self._add_garch_features(df)
        
        # 10. 기술적 지표
        print("  → 기술적 지표...")
        # RSI
        delta = df['SPY_Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        sma_20 = df['SPY_Close'].rolling(20).mean()
        std_20 = df['SPY_Close'].rolling(20).std()
        df['bb_upper'] = sma_20 + 2 * std_20
        df['bb_lower'] = sma_20 - 2 * std_20
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (sma_20 + 1e-8)
        df['bb_position'] = (df['SPY_Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
        
        # ATR-like
        df['high_low_range'] = (df['SPY_High'] - df['SPY_Low']) / (df['SPY_Close'] + 1e-8)
        df['atr_proxy'] = df['high_low_range'].rolling(14).mean()
        
        self.features = df
        print(f"\n  ✓ 총 {len(df.columns)} 특성 생성 완료")
        return df
    
    def _add_yang_zhang_volatility(self, df):
        """Yang-Zhang 변동성 추정기 (더 정확한 RV)"""
        for window in [5, 10, 20]:
            # 구성 요소
            log_ho = np.log(df['SPY_High'] / df['SPY_Open'])
            log_lo = np.log(df['SPY_Low'] / df['SPY_Open'])
            log_co = np.log(df['SPY_Close'] / df['SPY_Open'])
            log_oc = np.log(df['SPY_Open'] / df['SPY_Close'].shift(1))
            log_cc = np.log(df['SPY_Close'] / df['SPY_Close'].shift(1))
            
            # Rogers-Satchell
            rs = (log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)).rolling(window).mean()
            
            # Overnight variance
            overnight_var = log_oc.rolling(window).var()
            
            # Open-Close variance
            oc_var = log_co.rolling(window).var()
            
            # Yang-Zhang
            k = 0.34 / (1.34 + (window + 1) / (window - 1))
            yz_var = overnight_var + k * oc_var + (1 - k) * rs
            df[f'yz_vol_{window}'] = np.sqrt(yz_var * 252)
        
        return df
    
    def _add_garch_features(self, df):
        """GARCH(1,1) 조건부 변동성"""
        try:
            returns = df['returns'].dropna() * 100  # 퍼센트로 변환
            
            # GARCH 모델 피팅 (최근 500일)
            model = arch_model(returns[-500:], vol='Garch', p=1, q=1, rescale=False)
            result = model.fit(disp='off')
            
            # 전체 기간에 대해 조건부 분산 계산
            full_model = arch_model(returns, vol='Garch', p=1, q=1, rescale=False)
            full_result = full_model.fit(disp='off')
            
            cond_vol = full_result.conditional_volatility / 100 * np.sqrt(252)
            
            # 인덱스 맞춤
            df['garch_vol'] = np.nan
            df.loc[returns.index, 'garch_vol'] = cond_vol.values
            df['garch_vol'] = df['garch_vol'].ffill()
            
            # GARCH vs Realized Vol
            df['garch_rv_ratio'] = df['garch_vol'] / (df['rv_20'] + 1e-8)
            
        except Exception as e:
            print(f"    ⚠️ GARCH 피팅 오류: {e}")
        
        return df
    
    def create_target(self, horizon=5):
        """미래 변동성 타겟 생성 (데이터 누출 방지)"""
        print("\n" + "="*60)
        print(f"[3/7] 타겟 생성 (horizon={horizon}일)...")
        print("="*60)
        
        df = self.features.copy()
        
        # 미래 변동성 (t+1 ~ t+horizon)
        future_vol = []
        returns = df['returns'].values
        
        for i in range(len(returns)):
            if i + horizon < len(returns):
                future_returns = returns[i+1:i+1+horizon]
                vol = np.std(future_returns) * np.sqrt(252)
                future_vol.append(vol)
            else:
                future_vol.append(np.nan)
        
        df['target_vol'] = future_vol
        
        # Log 변환 (선택적 - 정규성 개선)
        df['target_vol_log'] = np.log(df['target_vol'] + 1e-8)
        
        self.features = df
        print(f"  ✓ 타겟 생성 완료")
        return df
    
    def prepare_train_test(self, test_ratio=0.2):
        """학습/테스트 데이터 준비"""
        print("\n" + "="*60)
        print("[4/7] 데이터 분할 및 전처리...")
        print("="*60)
        
        df = self.features.copy()
        
        # 특성 컬럼 선택 (타겟 및 원본 가격 제외)
        exclude_cols = ['target_vol', 'target_vol_log', 'SPY_Open', 'SPY_High', 
                       'SPY_Low', 'SPY_Close', 'SPY_Volume', 'VIX_Close', 'VIX_Volume',
                       'TLT_Close', 'TLT_Volume', 'GLD_Close', 'GLD_Volume',
                       'UUP_Close', 'UUP_Volume', 'HYG_Close', 'HYG_Volume',
                       'returns', 'log_returns']
        
        self.feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # 결측치 제거
        df = df.dropna()
        
        # 분할
        split_idx = int(len(df) * (1 - test_ratio))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        X_train = train_df[self.feature_cols]
        y_train = train_df['target_vol']
        X_test = test_df[self.feature_cols]
        y_test = test_df['target_vol']
        
        # 스케일링 (RobustScaler - 이상치에 강건)
        self.scaler = RobustScaler()
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=self.feature_cols,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=self.feature_cols,
            index=X_test.index
        )
        
        print(f"  ✓ Train: {len(X_train)} 샘플")
        print(f"  ✓ Test: {len(X_test)} 샘플")
        print(f"  ✓ 특성: {len(self.feature_cols)}개")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, test_df
    
    def train_multiple_models(self, X_train, X_test, y_train, y_test):
        """여러 모델 학습 및 비교"""
        print("\n" + "="*60)
        print("[5/7] 다중 모델 학습...")
        print("="*60)
        
        models = {}
        results = {}
        
        # TimeSeriesSplit for CV
        tscv = TimeSeriesSplit(n_splits=5)
        
        # 1. ElasticNet (기존 모델)
        print("\n  [1] ElasticNet...")
        en = ElasticNet(alpha=0.001, l1_ratio=0.3, random_state=SEED, max_iter=10000)
        en.fit(X_train, y_train)
        models['ElasticNet'] = en
        results['ElasticNet'] = self._evaluate_model(en, X_train, X_test, y_train, y_test, tscv)
        
        # 2. Ridge
        print("  [2] Ridge...")
        ridge = Ridge(alpha=1.0, random_state=SEED)
        ridge.fit(X_train, y_train)
        models['Ridge'] = ridge
        results['Ridge'] = self._evaluate_model(ridge, X_train, X_test, y_train, y_test, tscv)
        
        # 3. Random Forest
        print("  [3] Random Forest...")
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5,
                                   random_state=SEED, n_jobs=-1)
        rf.fit(X_train, y_train)
        models['RandomForest'] = rf
        results['RandomForest'] = self._evaluate_model(rf, X_train, X_test, y_train, y_test, tscv)
        
        # 4. Gradient Boosting
        print("  [4] Gradient Boosting...")
        gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                                       random_state=SEED)
        gb.fit(X_train, y_train)
        models['GradientBoosting'] = gb
        results['GradientBoosting'] = self._evaluate_model(gb, X_train, X_test, y_train, y_test, tscv)
        
        # 5. XGBoost
        if HAS_XGB:
            print("  [5] XGBoost...")
            xgb_model = xgb.XGBRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                random_state=SEED, n_jobs=-1, verbosity=0
            )
            xgb_model.fit(X_train, y_train)
            models['XGBoost'] = xgb_model
            results['XGBoost'] = self._evaluate_model(xgb_model, X_train, X_test, y_train, y_test, tscv)
        
        # 6. LightGBM
        if HAS_LGB:
            print("  [6] LightGBM...")
            lgb_model = lgb.LGBMRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                random_state=SEED, n_jobs=-1, verbose=-1
            )
            lgb_model.fit(X_train, y_train)
            models['LightGBM'] = lgb_model
            results['LightGBM'] = self._evaluate_model(lgb_model, X_train, X_test, y_train, y_test, tscv)
        
        # 결과 출력
        print("\n" + "-"*60)
        print("📊 모델별 성능 비교:")
        print("-"*60)
        print(f"{'모델':<20} {'CV R²':>12} {'Test R²':>12} {'Test RMSE':>12}")
        print("-"*60)
        for name, res in sorted(results.items(), key=lambda x: x[1]['test_r2'], reverse=True):
            print(f"{name:<20} {res['cv_r2_mean']:>12.4f} {res['test_r2']:>12.4f} {res['test_rmse']:>12.6f}")
        
        self.models = models
        self.results['individual'] = results
        
        return models, results
    
    def _evaluate_model(self, model, X_train, X_test, y_train, y_test, tscv):
        """모델 평가"""
        # CV 점수
        cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='r2')
        
        # 테스트 예측
        y_pred = model.predict(X_test)
        
        return {
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'test_r2': r2_score(y_test, y_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'test_mae': mean_absolute_error(y_test, y_pred)
        }
    
    def build_stacking_ensemble(self, X_train, X_test, y_train, y_test):
        """Stacking 앙상블 구축"""
        print("\n" + "="*60)
        print("[6/7] Stacking 앙상블 구축...")
        print("="*60)
        
        from sklearn.model_selection import KFold
        
        # 베이스 모델 선정
        base_estimators = [
            ('ridge', Ridge(alpha=1.0, random_state=SEED)),
            ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, 
                                         random_state=SEED, n_jobs=-1)),
            ('gb', GradientBoostingRegressor(n_estimators=100, max_depth=5, 
                                             learning_rate=0.1, random_state=SEED))
        ]
        
        if HAS_XGB:
            base_estimators.append(
                ('xgb', xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                                         random_state=SEED, n_jobs=-1, verbosity=0))
            )
        
        if HAS_LGB:
            base_estimators.append(
                ('lgb', lgb.LGBMRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                                          random_state=SEED, n_jobs=-1, verbose=-1))
            )
        
        # Stacking (메타 모델: Ridge) - KFold 사용 (TimeSeriesSplit 대신)
        stacking = StackingRegressor(
            estimators=base_estimators,
            final_estimator=Ridge(alpha=1.0),
            cv=KFold(n_splits=5, shuffle=False),  # KFold로 변경
            n_jobs=-1
        )
        
        print("  → Stacking 학습 중 (2-3분 소요)...")
        stacking.fit(X_train, y_train)
        
        # 평가
        y_pred = stacking.predict(X_test)
        test_r2 = r2_score(y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_mae = mean_absolute_error(y_test, y_pred)
        
        # CV 점수
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(stacking, X_train, y_train, cv=tscv, scoring='r2')
        
        print(f"\n  ✓ Stacking 앙상블 성능:")
        print(f"    - CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"    - Test R²: {test_r2:.4f}")
        print(f"    - Test RMSE: {test_rmse:.6f}")
        
        self.models['Stacking'] = stacking
        self.results['stacking'] = {
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae
        }
        
        return stacking
    
    def optimize_with_optuna(self, X_train, X_test, y_train, y_test, n_trials=50):
        """Optuna로 하이퍼파라미터 최적화"""
        if not HAS_OPTUNA:
            print("\n⚠️ Optuna를 사용할 수 없습니다. 기본 모델 사용.")
            return None
        
        print("\n" + "="*60)
        print(f"[6.5/7] Optuna 하이퍼파라미터 최적화 ({n_trials} trials)...")
        print("="*60)
        
        tscv = TimeSeriesSplit(n_splits=5)
        
        def objective(trial):
            # 모델 타입 선택
            model_type = trial.suggest_categorical('model_type', ['xgb', 'lgb', 'gb'])
            
            if model_type == 'xgb' and HAS_XGB:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                }
                model = xgb.XGBRegressor(**params, random_state=SEED, n_jobs=-1, verbosity=0)
                
            elif model_type == 'lgb' and HAS_LGB:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                }
                model = lgb.LGBMRegressor(**params, random_state=SEED, n_jobs=-1, verbose=-1)
                
            else:  # GradientBoosting
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                }
                model = GradientBoostingRegressor(**params, random_state=SEED)
            
            # CV 평가
            cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='r2')
            return cv_scores.mean()
        
        # Optuna 스터디 생성
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"\n  ✓ 최적 Trial: #{study.best_trial.number}")
        print(f"  ✓ 최적 CV R²: {study.best_value:.4f}")
        print(f"  ✓ 최적 파라미터:")
        for key, value in study.best_params.items():
            print(f"    - {key}: {value}")
        
        # 최적 모델 학습
        best_params = study.best_params
        model_type = best_params.pop('model_type')
        
        if model_type == 'xgb':
            best_model = xgb.XGBRegressor(**best_params, random_state=SEED, n_jobs=-1, verbosity=0)
        elif model_type == 'lgb':
            best_model = lgb.LGBMRegressor(**best_params, random_state=SEED, n_jobs=-1, verbose=-1)
        else:
            best_model = GradientBoostingRegressor(**best_params, random_state=SEED)
        
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        
        self.models['Optuna_Best'] = best_model
        self.results['optuna'] = {
            'best_params': study.best_params,
            'model_type': model_type,
            'cv_r2': study.best_value,
            'test_r2': r2_score(y_test, y_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'test_mae': mean_absolute_error(y_test, y_pred)
        }
        
        return best_model
    
    def select_best_model(self, X_test, y_test):
        """최고 성능 모델 선택"""
        print("\n" + "="*60)
        print("[7/7] 최종 모델 선택...")
        print("="*60)
        
        best_r2 = -np.inf
        best_name = None
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            if r2 > best_r2:
                best_r2 = r2
                best_name = name
                self.best_model = model
        
        print(f"\n  🏆 최고 성능 모델: {best_name}")
        print(f"  📊 Test R²: {best_r2:.4f}")
        
        return self.best_model, best_name
    
    def save_results(self, X_test, y_test, test_df, best_name):
        """결과 저장"""
        print("\n" + "="*60)
        print("결과 저장 중...")
        print("="*60)
        
        # 디렉토리 생성
        model_dir = Path('data/models')
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 모델 저장
        joblib.dump(self.best_model, model_dir / 'enhanced_best_model.pkl')
        joblib.dump(self.scaler, model_dir / 'enhanced_scaler.pkl')
        print(f"  ✓ 모델 저장됨: {model_dir / 'enhanced_best_model.pkl'}")
        
        # 예측 결과 저장
        y_pred = self.best_model.predict(X_test)
        predictions = pd.DataFrame({
            'Date': test_df.index,
            'actual_volatility': y_test.values,
            'predicted_volatility': y_pred
        })
        predictions.to_csv('data/raw/enhanced_test_predictions.csv', index=False)
        print(f"  ✓ 예측 결과 저장됨: data/raw/enhanced_test_predictions.csv")
        
        # 메트릭 저장
        final_metrics = {
            'model_name': f'Enhanced {best_name}',
            'test_r2': float(r2_score(y_test, y_pred)),
            'test_rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'test_mae': float(mean_absolute_error(y_test, y_pred)),
            'n_features': len(self.feature_cols),
            'n_samples_test': len(y_test),
            'feature_list': self.feature_cols[:20],  # 상위 20개만
            'timestamp': datetime.now().isoformat()
        }
        
        with open('data/raw/enhanced_model_performance.json', 'w') as f:
            json.dump(final_metrics, f, indent=2)
        print(f"  ✓ 메트릭 저장됨: data/raw/enhanced_model_performance.json")
        
        return final_metrics
    
    def run_full_pipeline(self):
        """전체 파이프라인 실행"""
        start_time = datetime.now()
        print("\n" + "🚀"*30)
        print("통합 변동성 예측 파이프라인 v2.0 시작")
        print("🚀"*30)
        
        # 1. 데이터 로드
        self.load_multi_asset_data()
        
        # 2. 특성 엔지니어링
        self.create_enhanced_features()
        
        # 3. 타겟 생성
        self.create_target(horizon=5)
        
        # 4. 데이터 분할
        X_train, X_test, y_train, y_test, test_df = self.prepare_train_test()
        
        # 5. 다중 모델 학습
        self.train_multiple_models(X_train, X_test, y_train, y_test)
        
        # 6. Stacking 앙상블
        self.build_stacking_ensemble(X_train, X_test, y_train, y_test)
        
        # 6.5. Optuna 최적화 (선택적)
        if HAS_OPTUNA:
            self.optimize_with_optuna(X_train, X_test, y_train, y_test, n_trials=30)
        
        # 7. 최고 모델 선택
        best_model, best_name = self.select_best_model(X_test, y_test)
        
        # 8. 결과 저장
        final_metrics = self.save_results(X_test, y_test, test_df, best_name)
        
        # 완료
        elapsed = datetime.now() - start_time
        print("\n" + "="*60)
        print("✅ 파이프라인 완료!")
        print("="*60)
        print(f"  ⏱️ 총 소요 시간: {elapsed}")
        print(f"  🏆 최고 모델: {best_name}")
        print(f"  📊 Test R²: {final_metrics['test_r2']:.4f}")
        print(f"  📉 Test RMSE: {final_metrics['test_rmse']:.6f}")
        
        return final_metrics


def main():
    """메인 실행"""
    pipeline = EnhancedVolatilityPipeline(
        start_date='2015-01-01',
        end_date='2024-12-31'
    )
    
    metrics = pipeline.run_full_pipeline()
    return metrics


if __name__ == '__main__':
    metrics = main()
