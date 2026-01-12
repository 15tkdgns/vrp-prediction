#!/usr/bin/env python3
"""
통합 변동성 예측 파이프라인 v2.1
- 핵심 개선: 타겟 정규화, 특성 선택, 단순화된 특성
- 기존 R² 0.22 → 목표 R² 0.30+

주요 변경점:
1. 타겟: 원시 5일 std (연간화 X)
2. 핵심 특성만 사용 (VIX 중심)
3. 특성 상관관계 기반 선택
4. 더 보수적인 모델 설정
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
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

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from arch import arch_model
    HAS_ARCH = True
except ImportError:
    HAS_ARCH = False

SEED = 42
np.random.seed(SEED)


class ImprovedVolatilityPipeline:
    """개선된 변동성 예측 파이프라인 v2.1"""
    
    def __init__(self, start_date='2015-01-01', end_date='2024-12-31'):
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.feature_cols = []
        self.scaler = None
        self.best_model = None
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """다중 자산 데이터 로드"""
        print("\n" + "="*60)
        print("[1/8] 데이터 로드...")
        print("="*60)
        
        tickers = {
            'SPY': 'SPY',
            'VIX': '^VIX',
            'TLT': 'TLT',
            'GLD': 'GLD',
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
        
        # SPY 기본 + 다른 자산 Close 병합
        self.data = all_data['SPY'].copy()
        for name in ['VIX', 'TLT', 'GLD']:
            if name in all_data:
                self.data[f'{name}_Close'] = all_data[name]['Close']
        
        self.data = self.data.ffill().dropna()
        print(f"\n  ✓ 최종: {len(self.data)} 행")
        return self.data
    
    def create_features(self):
        """핵심 특성만 생성 (과적합 방지)"""
        print("\n" + "="*60)
        print("[2/8] 핵심 특성 생성...")
        print("="*60)
        
        df = self.data.copy()
        
        # 기본 수익률
        df['returns'] = df['Close'].pct_change()
        
        # === 1. 과거 변동성 (핵심) ===
        print("  → 변동성 특성...")
        for w in [5, 10, 20, 50]:
            df[f'rv_{w}'] = df['returns'].rolling(w).std()  # 원시 std
        
        # 변동성 래그 (HAR 스타일)
        for lag in [1, 5, 10, 22]:
            df[f'rv_5_lag_{lag}'] = df['rv_5'].shift(lag)
        
        # 변동성 변화
        df['rv_change_1'] = df['rv_5'].pct_change()
        df['rv_change_5'] = df['rv_5'].pct_change(5)
        
        # 변동성 비율
        df['rv_ratio_5_20'] = df['rv_5'] / (df['rv_20'] + 1e-10)
        df['rv_ratio_5_50'] = df['rv_5'] / (df['rv_50'] + 1e-10)
        
        # === 2. VIX (변동성 예측의 핵심) ===
        print("  → VIX 특성...")
        if 'VIX_Close' in df.columns:
            df['vix'] = df['VIX_Close'] / 100  # 정규화
            df['vix_lag_1'] = df['vix'].shift(1)
            df['vix_lag_5'] = df['vix'].shift(5)
            df['vix_ma_5'] = df['vix'].rolling(5).mean()
            df['vix_ma_20'] = df['vix'].rolling(20).mean()
            df['vix_change'] = df['vix'].pct_change()
            
            # VIX 상대 위치 (0-1 사이)
            df['vix_percentile'] = df['vix'].rolling(252).rank(pct=True)
            
            # VIX vs Realized Vol (VRP proxy)
            df['vix_rv_ratio'] = df['vix'] / (df['rv_20'] + 1e-10)
        
        # === 3. 수익률 특성 (간소화) ===
        print("  → 수익률 특성...")
        for w in [5, 10, 20]:
            df[f'return_mean_{w}'] = df['returns'].rolling(w).mean()
            df[f'abs_return_sum_{w}'] = df['returns'].abs().rolling(w).sum()
        
        # 수익률 래그
        for lag in [1, 2, 3, 5]:
            df[f'return_lag_{lag}'] = df['returns'].shift(lag)
        
        # === 4. 크로스에셋 (간소화) ===
        print("  → 크로스에셋...")
        for asset in ['TLT', 'GLD']:
            col = f'{asset}_Close'
            if col in df.columns:
                df[f'{asset}_return'] = df[col].pct_change()
                df[f'{asset}_return_lag_1'] = df[f'{asset}_return'].shift(1)
                df[f'spy_{asset}_corr'] = df['returns'].rolling(20).corr(df[f'{asset}_return'])
        
        # === 5. 기술적 지표 (최소한) ===
        print("  → 기술적 지표...")
        # ATR proxy
        df['range'] = (df['High'] - df['Low']) / df['Close']
        df['atr_5'] = df['range'].rolling(5).mean()
        df['atr_20'] = df['range'].rolling(20).mean()
        
        self.data = df
        print(f"\n  ✓ 총 {len(df.columns)} 컬럼")
        return df
    
    def create_target(self, horizon=5):
        """미래 변동성 타겟"""
        print("\n" + "="*60)
        print(f"[3/8] 타겟 생성 (horizon={horizon})...")
        print("="*60)
        
        df = self.data.copy()
        
        # 미래 변동성: t+1 ~ t+horizon의 수익률 표준편차
        target = []
        returns = df['returns'].values
        
        for i in range(len(returns)):
            if i + horizon < len(returns):
                future_ret = returns[i+1:i+1+horizon]
                target.append(np.std(future_ret))
            else:
                target.append(np.nan)
        
        df['target'] = target
        
        # 타겟 통계
        print(f"  ✓ 타겟 평균: {np.nanmean(target):.6f}")
        print(f"  ✓ 타겟 표준편차: {np.nanstd(target):.6f}")
        
        self.data = df
        return df
    
    def select_features(self, method='correlation', top_k=30):
        """특성 선택"""
        print("\n" + "="*60)
        print(f"[4/8] 특성 선택 (method={method}, k={top_k})...")
        print("="*60)
        
        df = self.data.dropna().copy()
        
        # 특성 후보
        exclude = ['Open', 'High', 'Low', 'Close', 'Volume', 
                   'VIX_Close', 'TLT_Close', 'GLD_Close',
                   'returns', 'target']
        candidates = [c for c in df.columns if c not in exclude]
        
        X = df[candidates]
        y = df['target']
        
        if method == 'correlation':
            # 타겟과의 상관관계로 선택
            correlations = X.corrwith(y).abs().sort_values(ascending=False)
            selected = correlations.head(top_k).index.tolist()
            
            print("\n  📊 상위 특성 (타겟 상관관계):")
            for i, feat in enumerate(selected[:10]):
                print(f"    {i+1}. {feat}: {correlations[feat]:.4f}")
        
        elif method == 'mutual_info':
            # 상호정보량으로 선택
            selector = SelectKBest(mutual_info_regression, k=top_k)
            selector.fit(X, y)
            mask = selector.get_support()
            selected = [c for c, m in zip(candidates, mask) if m]
        
        else:
            selected = candidates[:top_k]
        
        self.feature_cols = selected
        print(f"\n  ✓ {len(selected)}개 특성 선택됨")
        return selected
    
    def prepare_data(self, test_ratio=0.2):
        """학습/테스트 분할"""
        print("\n" + "="*60)
        print("[5/8] 데이터 분할...")
        print("="*60)
        
        df = self.data.dropna().copy()
        
        split_idx = int(len(df) * (1 - test_ratio))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        X_train = train_df[self.feature_cols]
        y_train = train_df['target']
        X_test = test_df[self.feature_cols]
        y_test = test_df['target']
        
        # 스케일링
        self.scaler = StandardScaler()
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
        
        print(f"  ✓ Train: {len(X_train)}, Test: {len(X_test)}")
        print(f"  ✓ 특성: {len(self.feature_cols)}")
        
        # 클래스 변수로 저장
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        self.test_df = test_df
        
        return X_train_scaled, X_test_scaled, y_train, y_test, test_df
    
    def train_baseline_models(self):
        """기준 모델들 학습"""
        print("\n" + "="*60)
        print("[6/8] 기준 모델 학습...")
        print("="*60)
        
        X_train, X_test = self.X_train, self.X_test
        y_train, y_test = self.y_train, self.y_test
        
        tscv = TimeSeriesSplit(n_splits=5)
        results = {}
        
        # 1. Ridge (강한 정규화)
        print("\n  [1] Ridge Regression...")
        ridge_params = {'alpha': [0.1, 1.0, 10.0, 100.0]}
        ridge = GridSearchCV(Ridge(), ridge_params, cv=tscv, scoring='r2')
        ridge.fit(X_train, y_train)
        self.models['Ridge'] = ridge.best_estimator_
        results['Ridge'] = self._evaluate(ridge.best_estimator_, X_test, y_test)
        print(f"      Best alpha: {ridge.best_params_['alpha']}, Test R²: {results['Ridge']['test_r2']:.4f}")
        
        # 2. ElasticNet
        print("  [2] ElasticNet...")
        en_params = {'alpha': [0.001, 0.01, 0.1, 1.0], 'l1_ratio': [0.1, 0.5, 0.9]}
        en = GridSearchCV(ElasticNet(max_iter=10000), en_params, cv=tscv, scoring='r2')
        en.fit(X_train, y_train)
        self.models['ElasticNet'] = en.best_estimator_
        results['ElasticNet'] = self._evaluate(en.best_estimator_, X_test, y_test)
        print(f"      Best params: {en.best_params_}, Test R²: {results['ElasticNet']['test_r2']:.4f}")
        
        # 3. Random Forest (보수적 설정)
        print("  [3] Random Forest...")
        rf = RandomForestRegressor(
            n_estimators=100, max_depth=5, min_samples_leaf=10,
            random_state=SEED, n_jobs=-1
        )
        rf.fit(X_train, y_train)
        self.models['RandomForest'] = rf
        results['RandomForest'] = self._evaluate(rf, X_test, y_test)
        print(f"      Test R²: {results['RandomForest']['test_r2']:.4f}")
        
        # 4. Gradient Boosting (보수적 설정)
        print("  [4] Gradient Boosting...")
        gb = GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            min_samples_leaf=10, random_state=SEED
        )
        gb.fit(X_train, y_train)
        self.models['GradientBoosting'] = gb
        results['GradientBoosting'] = self._evaluate(gb, X_test, y_test)
        print(f"      Test R²: {results['GradientBoosting']['test_r2']:.4f}")
        
        # 5. XGBoost
        if HAS_XGB:
            print("  [5] XGBoost...")
            xgb_model = xgb.XGBRegressor(
                n_estimators=100, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=1.0, reg_lambda=1.0,
                random_state=SEED, n_jobs=-1, verbosity=0
            )
            xgb_model.fit(X_train, y_train)
            self.models['XGBoost'] = xgb_model
            results['XGBoost'] = self._evaluate(xgb_model, X_test, y_test)
            print(f"      Test R²: {results['XGBoost']['test_r2']:.4f}")
        
        # 6. LightGBM
        if HAS_LGB:
            print("  [6] LightGBM...")
            lgb_model = lgb.LGBMRegressor(
                n_estimators=100, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=1.0, reg_lambda=1.0,
                random_state=SEED, n_jobs=-1, verbose=-1
            )
            lgb_model.fit(X_train, y_train)
            self.models['LightGBM'] = lgb_model
            results['LightGBM'] = self._evaluate(lgb_model, X_test, y_test)
            print(f"      Test R²: {results['LightGBM']['test_r2']:.4f}")
        
        self.results = results
        self._print_comparison(results)
        return results
    
    def train_har_model(self):
        """HAR 모델 (Heterogeneous Autoregressive)"""
        print("\n" + "="*60)
        print("[6.5/8] HAR 모델 (벤치마크)...")
        print("="*60)
        
        # HAR 특성만 사용
        har_features = ['rv_5', 'rv_5_lag_1', 'rv_5_lag_5', 'rv_5_lag_22']
        har_features = [f for f in har_features if f in self.data.columns]
        
        df = self.data.dropna().copy()
        split_idx = int(len(df) * 0.8)
        
        X_train = df.iloc[:split_idx][har_features]
        y_train = df.iloc[:split_idx]['target']
        X_test = df.iloc[split_idx:][har_features]
        y_test = df.iloc[split_idx:]['target']
        
        # 스케일링
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        # HAR은 단순 OLS
        har = Ridge(alpha=0.1)
        har.fit(X_train_s, y_train)
        
        y_pred = har.predict(X_test_s)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"\n  ✓ HAR 모델 성능:")
        print(f"    - Test R²: {r2:.4f}")
        print(f"    - Test RMSE: {rmse:.6f}")
        print(f"    - 특성: {har_features}")
        
        self.models['HAR'] = har
        self.results['HAR'] = {'test_r2': r2, 'test_rmse': rmse}
        
        return har
    
    def train_simple_vix_model(self):
        """VIX 기반 단순 모델 (벤치마크)"""
        print("\n" + "="*60)
        print("[6.6/8] VIX 기반 모델 (벤치마크)...")
        print("="*60)
        
        # VIX만 사용
        vix_features = ['vix', 'vix_lag_1', 'vix_change', 'vix_rv_ratio']
        vix_features = [f for f in vix_features if f in self.data.columns]
        
        if not vix_features:
            print("  ⚠️ VIX 특성 없음")
            return None
        
        df = self.data.dropna().copy()
        split_idx = int(len(df) * 0.8)
        
        X_train = df.iloc[:split_idx][vix_features]
        y_train = df.iloc[:split_idx]['target']
        X_test = df.iloc[split_idx:][vix_features]
        y_test = df.iloc[split_idx:]['target']
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        vix_model = Ridge(alpha=1.0)
        vix_model.fit(X_train_s, y_train)
        
        y_pred = vix_model.predict(X_test_s)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"\n  ✓ VIX 모델 성능:")
        print(f"    - Test R²: {r2:.4f}")
        print(f"    - Test RMSE: {rmse:.6f}")
        print(f"    - 특성: {vix_features}")
        
        self.models['VIX_Only'] = vix_model
        self.results['VIX_Only'] = {'test_r2': r2, 'test_rmse': rmse}
        
        return vix_model
    
    def _evaluate(self, model, X_test, y_test):
        """모델 평가"""
        y_pred = model.predict(X_test)
        return {
            'test_r2': r2_score(y_test, y_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'test_mae': mean_absolute_error(y_test, y_pred)
        }
    
    def _print_comparison(self, results):
        """결과 비교 출력"""
        print("\n" + "-"*60)
        print("📊 모델 성능 비교:")
        print("-"*60)
        print(f"{'모델':<20} {'Test R²':>12} {'Test RMSE':>12}")
        print("-"*60)
        for name, res in sorted(results.items(), key=lambda x: x[1]['test_r2'], reverse=True):
            print(f"{name:<20} {res['test_r2']:>12.4f} {res['test_rmse']:>12.6f}")
    
    def select_best_model(self):
        """최고 모델 선택"""
        print("\n" + "="*60)
        print("[7/8] 최종 모델 선택...")
        print("="*60)
        
        best_name = max(self.results, key=lambda x: self.results[x]['test_r2'])
        self.best_model = self.models[best_name]
        
        print(f"\n  🏆 최고 모델: {best_name}")
        print(f"  📊 Test R²: {self.results[best_name]['test_r2']:.4f}")
        
        return self.best_model, best_name
    
    def save_results(self, best_name):
        """결과 저장"""
        print("\n" + "="*60)
        print("[8/8] 결과 저장...")
        print("="*60)
        
        model_dir = Path('data/models')
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 모델 저장
        joblib.dump(self.best_model, model_dir / 'improved_best_model.pkl')
        joblib.dump(self.scaler, model_dir / 'improved_scaler.pkl')
        
        # 예측 저장
        y_pred = self.best_model.predict(self.X_test)
        predictions = pd.DataFrame({
            'Date': self.test_df.index,
            'actual': self.y_test.values,
            'predicted': y_pred
        })
        predictions.to_csv('data/raw/improved_predictions.csv', index=False)
        
        # 메트릭 저장
        metrics = {
            'model_name': f'Improved {best_name}',
            'test_r2': float(self.results[best_name]['test_r2']),
            'test_rmse': float(self.results[best_name]['test_rmse']),
            'n_features': len(self.feature_cols),
            'all_results': {k: {'test_r2': float(v['test_r2'])} for k, v in self.results.items()},
            'timestamp': datetime.now().isoformat()
        }
        
        with open('data/raw/improved_model_performance.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"  ✓ 모델 저장됨")
        print(f"  ✓ 예측 저장됨")
        print(f"  ✓ 메트릭 저장됨")
        
        return metrics
    
    def analyze_feature_importance(self):
        """특성 중요도 분석"""
        print("\n" + "="*60)
        print("특성 중요도 분석...")
        print("="*60)
        
        # Ridge 계수
        if 'Ridge' in self.models:
            ridge = self.models['Ridge']
            importance = pd.Series(
                np.abs(ridge.coef_),
                index=self.feature_cols
            ).sort_values(ascending=False)
            
            print("\n  📊 Ridge 계수 (상위 10):")
            for i, (feat, imp) in enumerate(importance.head(10).items()):
                print(f"    {i+1}. {feat}: {imp:.4f}")
        
        # Random Forest 중요도
        if 'RandomForest' in self.models:
            rf = self.models['RandomForest']
            importance = pd.Series(
                rf.feature_importances_,
                index=self.feature_cols
            ).sort_values(ascending=False)
            
            print("\n  📊 Random Forest 중요도 (상위 10):")
            for i, (feat, imp) in enumerate(importance.head(10).items()):
                print(f"    {i+1}. {feat}: {imp:.4f}")
    
    def run(self):
        """전체 파이프라인 실행"""
        start = datetime.now()
        print("\n" + "🚀"*30)
        print("개선된 변동성 예측 파이프라인 v2.1")
        print("🚀"*30)
        
        # 1. 데이터 로드
        self.load_data()
        
        # 2. 특성 생성
        self.create_features()
        
        # 3. 타겟 생성
        self.create_target(horizon=5)
        
        # 4. 특성 선택
        self.select_features(method='correlation', top_k=25)
        
        # 5. 데이터 분할
        self.prepare_data()
        
        # 6. 기준 모델 학습
        self.train_baseline_models()
        
        # 6.5. HAR 벤치마크
        self.train_har_model()
        
        # 6.6. VIX 벤치마크
        self.train_simple_vix_model()
        
        # 7. 최고 모델 선택
        best_model, best_name = self.select_best_model()
        
        # 8. 결과 저장
        metrics = self.save_results(best_name)
        
        # 특성 중요도
        self.analyze_feature_importance()
        
        # 최종 비교
        self._print_comparison(self.results)
        
        elapsed = datetime.now() - start
        print("\n" + "="*60)
        print("✅ 완료!")
        print("="*60)
        print(f"  ⏱️ 소요 시간: {elapsed}")
        print(f"  🏆 최고 모델: {best_name}")
        print(f"  📊 Test R²: {metrics['test_r2']:.4f}")
        
        return metrics


def main():
    pipeline = ImprovedVolatilityPipeline(
        start_date='2015-01-01',
        end_date='2024-12-31'
    )
    metrics = pipeline.run()
    return metrics


if __name__ == '__main__':
    metrics = main()
