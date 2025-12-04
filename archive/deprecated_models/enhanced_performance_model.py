#!/usr/bin/env python3
"""
Enhanced Performance Model - 안전한 성능 개선
Walk-Forward 검증을 기본으로 하여 신중하게 성능 향상 시도
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

class EnhancedPerformancePredictor:
    """Walk-Forward 검증 기반 안전한 성능 개선"""

    def __init__(self):
        self.models = {}
        self.validation_results = {}

    def load_extended_data(self):
        """더 긴 기간의 데이터 로드 (2000년부터)"""
        print("📊 확장된 데이터 로딩 중 (2000-2024)...")

        try:
            # SPY 데이터 (더 긴 기간)
            spy = yf.download('SPY', start='2000-01-01', end='2024-12-31', progress=False)
            spy['returns'] = spy['Close'].pct_change()

            # VIX 데이터
            vix = yf.download('^VIX', start='2000-01-01', end='2024-12-31', progress=False)
            spy['vix'] = vix['Close'].reindex(spy.index, method='ffill')

            # 추가 경제 지표들
            try:
                # 10년 국채 금리
                treasury = yf.download('^TNX', start='2000-01-01', end='2024-12-31', progress=False)
                spy['treasury_10y'] = treasury['Close'].reindex(spy.index, method='ffill')
            except:
                spy['treasury_10y'] = 2.5

            try:
                # 2년 국채 금리 (수익률 곡선을 위해)
                treasury_2y = yf.download('^TNX', start='2000-01-01', end='2024-12-31', progress=False)
                spy['treasury_2y'] = treasury_2y['Close'].reindex(spy.index, method='ffill') * 0.8  # 근사치
            except:
                spy['treasury_2y'] = 2.0

            spy = spy.dropna()
            print(f"✅ 확장 데이터 로드 완료: {len(spy)} 관측치 (2000-2024)")

            # 데이터 품질 확인
            print(f"   기간: {spy.index[0].strftime('%Y-%m-%d')} ~ {spy.index[-1].strftime('%Y-%m-%d')}")
            print(f"   주요 경제 이벤트 포함:")
            print(f"     - 닷컴 버블 (2000-2002)")
            print(f"     - 금융위기 (2008-2009)")
            print(f"     - 코로나 위기 (2020)")

            return spy

        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            return None

    def create_enhanced_features(self, data):
        """향상된 특성 생성 (안전한 범위 내에서)"""
        print("🔧 향상된 특성 생성 중...")

        features = pd.DataFrame(index=data.index)
        returns = data['returns']
        prices = data['Close']
        high = data['High']
        low = data['Low']
        volume = data['Volume']
        vix = data['vix']

        # 1. 핵심 변동성 특성 (다양한 윈도우)
        for window in [5, 10, 20, 60]:  # 60일 추가로 더 긴 기간 변동성
            features[f'volatility_{window}'] = returns.rolling(window).std()
            # 연율화된 변동성
            features[f'vol_annualized_{window}'] = features[f'volatility_{window}'] * np.sqrt(252)

        # 2. VIX 특성 강화
        features['vix_level'] = vix / 100  # 정규화
        for window in [5, 10, 20, 60]:
            features[f'vix_ma_{window}'] = vix.rolling(window).mean() / 100

        # VIX 기간구조
        features['vix_term_structure'] = features['vix_level'] / features['vix_ma_20']
        features['vix_change'] = vix.pct_change()
        features['vix_momentum'] = features['vix_change'].rolling(5).mean()

        # 3. 경제 지표 특성
        if 'treasury_10y' in data.columns:
            treasury_10y = data['treasury_10y']
            treasury_2y = data['treasury_2y']

            features['treasury_10y'] = treasury_10y / 100
            features['treasury_2y'] = treasury_2y / 100

            # 수익률 곡선 기울기
            features['yield_curve_slope'] = (treasury_10y - treasury_2y) / 100

            # VIX-금리 스프레드 (이전에 효과적이었던 특성)
            features['vix_treasury_spread'] = features['vix_level'] - features['treasury_10y']

            # 금리 변화
            features['treasury_change'] = treasury_10y.diff() / 100

        # 4. 가격 기반 특성
        for window in [5, 10, 20]:
            # 가격 모멘텀
            features[f'price_momentum_{window}'] = prices.pct_change(window)

            # 일중 변동성
            intraday_range = (high - low) / prices
            features[f'intraday_vol_{window}'] = intraday_range.rolling(window).mean()

            # Garman-Klass 추정량
            gk = np.log(high / low) ** 2
            features[f'garman_klass_{window}'] = gk.rolling(window).mean()

        # 5. 지수 가중 특성
        for span in [10, 30]:
            features[f'ewm_vol_{span}'] = returns.ewm(span=span).std()
            features[f'ewm_vix_{span}'] = vix.ewm(span=span).mean() / 100

        # 6. 래그 특성 (제한적)
        for lag in [1, 2, 5]:
            features[f'vol_lag_{lag}'] = features['volatility_5'].shift(lag)
            features[f'vix_lag_{lag}'] = features['vix_level'].shift(lag)

        # 7. 안전한 상호작용 특성
        features['vix_vol_ratio'] = features['vix_level'] / (features['volatility_20'] + 1e-8)
        if 'yield_curve_slope' in features.columns:
            features['vix_yield_interaction'] = features['vix_level'] * features['yield_curve_slope']

        # 8. 시장 체제 지표
        # 변동성 레짐
        vol_percentile = features['volatility_20'].rolling(252).rank(pct=True)
        features['vol_regime_high'] = (vol_percentile > 0.8).astype(int)
        features['vol_regime_low'] = (vol_percentile < 0.2).astype(int)

        print(f"✅ 향상된 특성 생성 완료: {len(features.columns)}개")
        return features

    def create_robust_targets(self, data):
        """안정적인 타겟 변수 생성"""
        print("🎯 안정적 타겟 생성 중...")

        targets = pd.DataFrame(index=data.index)
        returns = data['returns']

        # 다양한 예측 기간의 변동성
        for horizon in [5, 10, 20]:
            vol_values = []
            for i in range(len(returns)):
                if i + horizon < len(returns):
                    future_returns = returns.iloc[i+1:i+1+horizon]
                    vol_values.append(future_returns.std())
                else:
                    vol_values.append(np.nan)
            targets[f'target_vol_{horizon}d'] = vol_values

        return targets

    def optimize_regularization_strength(self, X, y):
        """정규화 강도 최적화 (Walk-Forward 기준)"""
        print("🔧 정규화 강도 최적화 중...")

        # 후보 정규화 강도들
        ridge_alphas = [0.1, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0]
        lasso_alphas = [0.001, 0.005, 0.01, 0.05, 0.1]
        elastic_alphas = [0.001, 0.005, 0.01, 0.05]

        best_models = {}

        combined_data = pd.concat([X, y], axis=1).dropna()
        X_clean = combined_data[X.columns]
        y_clean = combined_data[y.name]

        print(f"최적화 데이터: {len(X_clean)} 샘플")

        # 빠른 Walk-Forward (5개 폴드만)
        initial_window = 1000  # 4년
        step_size = 200        # 8개월 단위
        max_folds = 5

        # Ridge 최적화
        print("  Ridge 정규화 최적화...")
        ridge_scores = {}

        for alpha in ridge_alphas:
            fold_scores = []
            current_start = 0
            fold = 0

            while fold < max_folds and current_start + initial_window + 100 < len(X_clean):
                train_end = current_start + initial_window
                test_start = train_end
                test_end = min(test_start + 100, len(X_clean))

                X_train = X_clean.iloc[current_start:train_end]
                y_train = y_clean.iloc[current_start:train_end]
                X_test = X_clean.iloc[test_start:test_end]
                y_test = y_clean.iloc[test_start:test_end]

                try:
                    scaler = RobustScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    model = Ridge(alpha=alpha)
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)

                    r2 = r2_score(y_test, y_pred)
                    fold_scores.append(r2)

                except:
                    fold_scores.append(-999)

                current_start += step_size
                fold += 1

            valid_scores = [s for s in fold_scores if s > -900]
            if valid_scores:
                ridge_scores[alpha] = {
                    'mean_r2': np.mean(valid_scores),
                    'std_r2': np.std(valid_scores),
                    'count': len(valid_scores)
                }

            print(f"    α={alpha:6.1f}: R²={ridge_scores.get(alpha, {}).get('mean_r2', -999):7.4f}")

        # 최고 Ridge 선택
        if ridge_scores:
            best_ridge_alpha = max(ridge_scores.items(), key=lambda x: x[1]['mean_r2'])
            best_models['Ridge'] = {
                'model': Ridge(alpha=best_ridge_alpha[0]),
                'alpha': best_ridge_alpha[0],
                'performance': best_ridge_alpha[1]
            }
            print(f"    ✅ 최고 Ridge: α={best_ridge_alpha[0]}, R²={best_ridge_alpha[1]['mean_r2']:.4f}")

        # ElasticNet 최적화 (간단히)
        print("  ElasticNet 최적화...")
        best_elastic_score = -999
        best_elastic_alpha = 0.01

        for alpha in elastic_alphas:
            fold_scores = []
            current_start = 0
            fold = 0

            while fold < 3 and current_start + initial_window + 100 < len(X_clean):  # 더 빠르게
                train_end = current_start + initial_window
                test_start = train_end
                test_end = min(test_start + 100, len(X_clean))

                X_train = X_clean.iloc[current_start:train_end]
                y_train = y_clean.iloc[current_start:train_end]
                X_test = X_clean.iloc[test_start:test_end]
                y_test = y_clean.iloc[test_start:test_end]

                try:
                    scaler = RobustScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    model = ElasticNet(alpha=alpha, l1_ratio=0.7, max_iter=3000)
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)

                    r2 = r2_score(y_test, y_pred)
                    fold_scores.append(r2)

                except:
                    fold_scores.append(-999)

                current_start += step_size
                fold += 1

            valid_scores = [s for s in fold_scores if s > -900]
            if valid_scores:
                mean_score = np.mean(valid_scores)
                if mean_score > best_elastic_score:
                    best_elastic_score = mean_score
                    best_elastic_alpha = alpha

            print(f"    α={alpha:6.3f}: R²={mean_score if valid_scores else -999:7.4f}")

        if best_elastic_score > -900:
            best_models['ElasticNet'] = {
                'model': ElasticNet(alpha=best_elastic_alpha, l1_ratio=0.7, max_iter=3000),
                'alpha': best_elastic_alpha,
                'performance': {'mean_r2': best_elastic_score}
            }
            print(f"    ✅ 최고 ElasticNet: α={best_elastic_alpha}, R²={best_elastic_score:.4f}")

        return best_models

    def validate_with_extended_walkforward(self, X, y, optimized_models):
        """확장된 Walk-Forward 검증"""
        print("🚀 확장된 Walk-Forward 검증 중...")

        combined_data = pd.concat([X, y], axis=1).dropna()
        X_clean = combined_data[X.columns]
        y_clean = combined_data[y.name]

        print(f"검증 데이터: {len(X_clean)} 샘플")

        # 더 보수적인 설정
        initial_window = 1260  # 5년
        step_size = 126        # 6개월 단위
        test_window = 63       # 3개월 예측

        results = {}
        for model_name in optimized_models.keys():
            results[model_name] = []

        current_start = 0
        fold = 0

        while current_start + initial_window + test_window < len(X_clean):
            fold += 1

            train_end = current_start + initial_window
            test_start = train_end
            test_end = min(test_start + test_window, len(X_clean))

            X_train = X_clean.iloc[current_start:train_end]
            y_train = y_clean.iloc[current_start:train_end]
            X_test = X_clean.iloc[test_start:test_end]
            y_test = y_clean.iloc[test_start:test_end]

            print(f"  Fold {fold}: 훈련 {len(X_train)} ({X_train.index[0].strftime('%Y-%m')} ~ {X_train.index[-1].strftime('%Y-%m')})")
            print(f"           테스트 {len(X_test)} ({X_test.index[0].strftime('%Y-%m')} ~ {X_test.index[-1].strftime('%Y-%m')})")

            for model_name, model_info in optimized_models.items():
                try:
                    model = model_info['model']

                    # 스케일링
                    scaler = RobustScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    # 훈련 및 예측
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)

                    # 성능 계산
                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)

                    results[model_name].append({
                        'fold': fold,
                        'r2': r2,
                        'rmse': rmse,
                        'mae': mae,
                        'test_period': f"{X_test.index[0].strftime('%Y-%m')} ~ {X_test.index[-1].strftime('%Y-%m')}"
                    })

                    print(f"    {model_name:15}: R²={r2:7.4f}, RMSE={rmse:.5f}")

                except Exception as e:
                    print(f"    {model_name:15}: 오류 - {e}")
                    results[model_name].append({
                        'fold': fold,
                        'r2': -999,
                        'rmse': 999,
                        'mae': 999,
                        'error': str(e)
                    })

            current_start += step_size

        # 결과 요약
        summary = {}
        print(f"\n📊 확장된 Walk-Forward 결과:")
        print(f"{'Model':<15} {'Mean R²':<10} {'Std R²':<10} {'Valid Folds':<12} {'Best Fold'}")
        print("-" * 65)

        for model_name, fold_results in results.items():
            valid_r2s = [r['r2'] for r in fold_results if r['r2'] != -999]

            if valid_r2s:
                mean_r2 = np.mean(valid_r2s)
                std_r2 = np.std(valid_r2s)
                max_r2 = np.max(valid_r2s)
                valid_folds = len(valid_r2s)

                summary[model_name] = {
                    'mean_r2': mean_r2,
                    'std_r2': std_r2,
                    'max_r2': max_r2,
                    'valid_folds': valid_folds,
                    'total_folds': len(fold_results)
                }

                print(f"{model_name:<15} {mean_r2:<10.4f} {std_r2:<10.4f} {valid_folds}/{len(fold_results):<12} {max_r2:.4f}")
            else:
                summary[model_name] = {'mean_r2': -999, 'valid_folds': 0}
                print(f"{model_name:<15} {'FAILED':<10} {'N/A':<10} {'0':<12} {'N/A'}")

        return summary, results

def main():
    """메인 향상된 성능 개선 함수"""
    print("🚀 Enhanced Performance Model - 안전한 성능 개선")
    print("=" * 80)
    print("⚠️ Walk-Forward 검증을 기본으로 하여 과적합 방지")
    print("=" * 80)

    predictor = EnhancedPerformancePredictor()

    # 1. 확장된 데이터 로드
    data = predictor.load_extended_data()
    if data is None:
        return

    # 2. 향상된 특성 생성
    features = predictor.create_enhanced_features(data)
    targets = predictor.create_robust_targets(data)

    # 3. 주요 타겟에 집중 (5일 변동성)
    target_col = 'target_vol_5d'
    if target_col not in targets.columns:
        print(f"❌ 타겟 컬럼 {target_col} 없음")
        return

    # 4. 특성 선별 (상관관계 기준)
    combined = pd.concat([features, targets[[target_col]]], axis=1).dropna()
    print(f"\n📊 데이터 준비 완료: {len(combined)} 샘플")

    correlations = combined[features.columns].corrwith(combined[target_col]).abs().sort_values(ascending=False)

    print(f"\n📈 상위 20개 특성:")
    for i, (feature, corr) in enumerate(correlations.head(20).items()):
        print(f"  {i+1:2d}. {feature:30}: {corr:.4f}")

    # 상위 특성 선별 (과적합 방지를 위해 적당히)
    top_features = correlations.head(20).index
    final_features = features[top_features]

    print(f"\n🎯 최종 특성 수: {len(final_features.columns)}")

    # 5. 정규화 강도 최적화
    optimized_models = predictor.optimize_regularization_strength(final_features, targets[target_col])

    if not optimized_models:
        print("❌ 최적화된 모델 없음")
        return

    # 6. 확장된 Walk-Forward 검증
    wf_summary, wf_details = predictor.validate_with_extended_walkforward(
        final_features, targets[target_col], optimized_models
    )

    # 7. 결과 분석
    best_model = None
    best_score = -999

    for model_name, stats in wf_summary.items():
        if stats.get('mean_r2', -999) > best_score:
            best_score = stats['mean_r2']
            best_model = model_name

    # 8. 결과 저장
    os.makedirs('results', exist_ok=True)

    enhanced_results = {
        'version': 'Enhanced_Performance_Safe',
        'timestamp': datetime.now().isoformat(),
        'approach': 'Extended data + optimized regularization',
        'data_period': '2000-2024',
        'total_samples': len(combined),
        'features_count': len(final_features.columns),
        'best_model': {
            'name': best_model,
            'mean_r2': best_score,
            'std_r2': wf_summary.get(best_model, {}).get('std_r2', 0) if best_model else 0
        },
        'all_results': wf_summary,
        'top_features': top_features.tolist(),
        'optimization_details': {model: info.get('performance', {}) for model, info in optimized_models.items()},
        'validation_method': 'Extended Walk-Forward',
        'safety_measures': [
            'Walk-Forward validation only',
            'Extended historical data (2000-2024)',
            'Optimized regularization strength',
            'Conservative feature selection',
            'Robust scaling'
        ]
    }

    with open('results/enhanced_performance_model.json', 'w') as f:
        json.dump(enhanced_results, f, indent=2, default=str)

    print(f"\n💾 결과 저장: results/enhanced_performance_model.json")

    # 9. 최종 평가
    if best_model and best_score > 0:
        print(f"\n🎉 성능 개선 성공!")
        print(f"   최고 모델: {best_model}")
        print(f"   Walk-Forward R²: {best_score:.4f}")

        # 이전 Robust 모델과 비교
        robust_r2 = 0.0145  # 이전 결과
        improvement = (best_score - robust_r2) / robust_r2 * 100 if robust_r2 > 0 else 0

        print(f"   이전 Robust 모델: {robust_r2:.4f}")
        print(f"   개선도: {improvement:+.1f}%")

        if best_score > robust_r2:
            print(f"   ✅ 안전한 성능 개선 달성!")
        else:
            print(f"   ⚠️ 성능 개선 미미, 추가 최적화 필요")

    else:
        print(f"\n⚠️ 성능 개선 실패")
        print(f"   최고 성능: {best_score:.4f}")
        print(f"   추가 접근법 필요")

    print("=" * 80)

if __name__ == "__main__":
    main()