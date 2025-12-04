#!/usr/bin/env python3
"""
Fine-Tuned Performance Model - 정밀 조정된 성능 개선
이전 실패를 바탕으로 더 정밀하고 보수적인 접근
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

class FineTunedPredictor:
    """정밀 조정된 성능 개선 모델"""

    def __init__(self):
        self.results = {}

    def load_optimal_data_period(self):
        """최적 데이터 기간 실험"""
        print("📊 최적 데이터 기간 탐색 중...")

        # 여러 시작 기간 테스트
        periods = {
            '2008_start': '2008-01-01',  # 금융위기부터 (더 안정적일 수 있음)
            '2010_start': '2010-01-01',  # 위기 이후부터
            '2015_start': '2015-01-01'   # 최근 기간만
        }

        best_period = None
        best_samples = 0

        for period_name, start_date in periods.items():
            try:
                spy = yf.download('SPY', start=start_date, end='2024-12-31', progress=False)
                spy['returns'] = spy['Close'].pct_change()

                vix = yf.download('^VIX', start=start_date, end='2024-12-31', progress=False)
                spy['vix'] = vix['Close'].reindex(spy.index, method='ffill')

                spy = spy.dropna()

                print(f"  {period_name}: {len(spy)} 샘플 ({spy.index[0].strftime('%Y-%m')} ~ {spy.index[-1].strftime('%Y-%m')})")

                # 충분한 샘플 수 확인
                if len(spy) > 2000:  # 최소 8년 데이터
                    if len(spy) > best_samples:
                        best_samples = len(spy)
                        best_period = (period_name, start_date, spy)

            except Exception as e:
                print(f"  {period_name}: 로드 실패 - {e}")

        if best_period:
            period_name, start_date, data = best_period
            print(f"✅ 최적 기간 선택: {period_name} ({len(data)} 샘플)")
            return data
        else:
            print("❌ 적절한 데이터 기간 없음")
            return None

    def create_minimal_robust_features(self, data):
        """최소한의 강건한 특성만 생성"""
        print("🔧 최소 특성 생성 중...")

        features = pd.DataFrame(index=data.index)
        returns = data['returns']
        prices = data['Close']
        high = data['High']
        low = data['Low']
        vix = data['vix']

        # 1. 핵심 변동성만 (이전에 가장 효과적이었던 것들)
        features['volatility_5'] = returns.rolling(5).std()
        features['volatility_10'] = returns.rolling(10).std()
        features['volatility_20'] = returns.rolling(20).std()

        # 2. VIX 핵심 특성만
        features['vix_level'] = vix / 100
        features['vix_ma_5'] = vix.rolling(5).mean() / 100
        features['vix_ma_10'] = vix.rolling(10).mean() / 100

        # 3. 일중 변동성 (가장 강력했던 특성)
        intraday_range = (high - low) / prices
        features['intraday_vol_5'] = intraday_range.rolling(5).mean()
        features['intraday_vol_10'] = intraday_range.rolling(10).mean()

        # 4. 최소한의 래그
        features['vol_lag_1'] = features['volatility_5'].shift(1)

        # 5. 가장 강력했던 상호작용 하나만
        features['vix_vol_ratio'] = features['vix_level'] / (features['volatility_10'] + 1e-8)

        print(f"✅ 최소 특성 생성 완료: {len(features.columns)}개")
        return features

    def fine_tune_regularization(self, X, y):
        """정밀한 정규화 강도 조정"""
        print("🎯 정밀 정규화 조정 중...")

        # 이전 성공 사례 주변에서 정밀 탐색
        ridge_alphas = [1.0, 2.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0]

        combined_data = pd.concat([X, y], axis=1).dropna()
        X_clean = combined_data[X.columns]
        y_clean = combined_data[y.name]

        print(f"조정 데이터: {len(X_clean)} 샘플")

        # 더 신중한 Walk-Forward (3개 폴드만으로 빠르게)
        initial_window = 800   # 약 3년
        step_size = 150        # 6개월
        test_window = 60       # 3개월

        best_alpha = 10.0
        best_score = -999
        alpha_scores = {}

        for alpha in ridge_alphas:
            fold_scores = []
            current_start = 0
            fold = 0

            while fold < 3 and current_start + initial_window + test_window < len(X_clean):
                train_end = current_start + initial_window
                test_start = train_end
                test_end = min(test_start + test_window, len(X_clean))

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
                mean_score = np.mean(valid_scores)
                alpha_scores[alpha] = {
                    'mean_r2': mean_score,
                    'std_r2': np.std(valid_scores),
                    'count': len(valid_scores)
                }

                if mean_score > best_score:
                    best_score = mean_score
                    best_alpha = alpha

                print(f"  α={alpha:6.1f}: R²={mean_score:7.4f} ± {np.std(valid_scores):.4f}")

        print(f"✅ 최적 정규화: α={best_alpha}, R²={best_score:.4f}")
        return best_alpha, alpha_scores

    def create_ensemble_models(self, best_alpha):
        """앙상블 모델 생성"""
        print("🤖 앙상블 모델 생성 중...")

        models = {
            'Ridge_Optimal': Ridge(alpha=best_alpha),
            'Ridge_Conservative': Ridge(alpha=best_alpha * 1.5),  # 더 보수적
            'Ridge_Aggressive': Ridge(alpha=best_alpha * 0.7),   # 덜 보수적
        }

        # 간단한 평균 앙상블도 준비
        models['Ensemble_Simple'] = 'ensemble'

        print(f"✅ 앙상블 모델 준비: {len(models)}개")
        return models

    def comprehensive_walkforward_test(self, X, y, models, best_alpha):
        """종합적인 Walk-Forward 테스트"""
        print("🚀 종합 Walk-Forward 테스트...")

        combined_data = pd.concat([X, y], axis=1).dropna()
        X_clean = combined_data[X.columns]
        y_clean = combined_data[y.name]

        print(f"테스트 데이터: {len(X_clean)} 샘플")

        # 보수적 설정
        initial_window = 1000  # 4년
        step_size = 100        # 4개월
        test_window = 50       # 2개월

        results = {}
        for model_name in models.keys():
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

            print(f"  Fold {fold}: 훈련 {len(X_train)}, 테스트 {len(X_test)} ({X_test.index[0].strftime('%Y-%m')})")

            # 개별 모델 예측 저장 (앙상블용)
            individual_predictions = {}

            for model_name, model in models.items():
                if model_name == 'Ensemble_Simple':
                    continue  # 나중에 처리

                try:
                    scaler = RobustScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)

                    # 개별 모델 성능
                    r2 = r2_score(y_test, y_pred)
                    results[model_name].append({
                        'fold': fold,
                        'r2': r2,
                        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
                    })

                    # 앙상블용 예측 저장
                    individual_predictions[model_name] = y_pred

                    print(f"    {model_name:20}: R²={r2:7.4f}")

                except Exception as e:
                    print(f"    {model_name:20}: 오류 - {e}")
                    results[model_name].append({'fold': fold, 'r2': -999, 'rmse': 999})

            # 앙상블 예측 (평균)
            if len(individual_predictions) >= 2:
                try:
                    # 개별 모델들의 평균
                    ensemble_pred = np.mean(list(individual_predictions.values()), axis=0)
                    ensemble_r2 = r2_score(y_test, ensemble_pred)

                    results['Ensemble_Simple'].append({
                        'fold': fold,
                        'r2': ensemble_r2,
                        'rmse': np.sqrt(mean_squared_error(y_test, ensemble_pred))
                    })

                    print(f"    {'Ensemble_Simple':20}: R²={ensemble_r2:7.4f}")

                except:
                    results['Ensemble_Simple'].append({'fold': fold, 'r2': -999, 'rmse': 999})

            current_start += step_size

        # 결과 요약
        summary = {}
        print(f"\n📊 종합 테스트 결과:")
        print(f"{'Model':<20} {'Mean R²':<10} {'Std R²':<10} {'Max R²':<10} {'Valid'}")
        print("-" * 70)

        for model_name, fold_results in results.items():
            valid_r2s = [r['r2'] for r in fold_results if r['r2'] != -999]

            if valid_r2s:
                mean_r2 = np.mean(valid_r2s)
                std_r2 = np.std(valid_r2s)
                max_r2 = np.max(valid_r2s)
                valid_count = len(valid_r2s)

                summary[model_name] = {
                    'mean_r2': mean_r2,
                    'std_r2': std_r2,
                    'max_r2': max_r2,
                    'valid_folds': valid_count,
                    'total_folds': len(fold_results)
                }

                print(f"{model_name:<20} {mean_r2:<10.4f} {std_r2:<10.4f} {max_r2:<10.4f} {valid_count}/{len(fold_results)}")
            else:
                summary[model_name] = {'mean_r2': -999}
                print(f"{model_name:<20} {'FAILED':<10}")

        return summary, results

def main():
    """메인 정밀 조정 함수"""
    print("🎯 Fine-Tuned Performance Model")
    print("=" * 70)
    print("정밀한 조정을 통한 안전한 성능 개선")
    print("=" * 70)

    predictor = FineTunedPredictor()

    # 1. 최적 데이터 기간 선택
    data = predictor.load_optimal_data_period()
    if data is None:
        return

    # 2. 최소한의 강건한 특성
    features = predictor.create_minimal_robust_features(data)

    # 3. 타겟 생성
    targets = pd.DataFrame(index=data.index)
    returns = data['returns']
    vol_values = []
    for i in range(len(returns)):
        if i + 5 < len(returns):
            future_returns = returns.iloc[i+1:i+6]
            vol_values.append(future_returns.std())
        else:
            vol_values.append(np.nan)
    targets['target_vol_5d'] = vol_values

    # 4. 정밀한 정규화 조정
    best_alpha, alpha_scores = predictor.fine_tune_regularization(features, targets['target_vol_5d'])

    # 5. 앙상블 모델 생성
    models = predictor.create_ensemble_models(best_alpha)

    # 6. 종합 테스트
    summary, detailed_results = predictor.comprehensive_walkforward_test(
        features, targets['target_vol_5d'], models, best_alpha
    )

    # 7. 최고 모델 선택
    best_model = None
    best_score = -999

    for model_name, stats in summary.items():
        if stats.get('mean_r2', -999) > best_score:
            best_score = stats['mean_r2']
            best_model = model_name

    # 8. 결과 저장
    os.makedirs('results', exist_ok=True)

    fine_tuned_results = {
        'version': 'Fine_Tuned_Performance',
        'timestamp': datetime.now().isoformat(),
        'approach': 'Minimal features + fine-tuned regularization + ensemble',
        'data_samples': len(pd.concat([features, targets], axis=1).dropna()),
        'features_count': len(features.columns),
        'best_model': {
            'name': best_model,
            'mean_r2': best_score,
            'std_r2': summary.get(best_model, {}).get('std_r2', 0) if best_model else 0
        },
        'optimization_results': {
            'best_alpha': best_alpha,
            'alpha_scores': alpha_scores
        },
        'all_results': summary,
        'feature_names': features.columns.tolist()
    }

    with open('results/fine_tuned_performance_model.json', 'w') as f:
        json.dump(fine_tuned_results, f, indent=2, default=str)

    print(f"\n💾 결과 저장: results/fine_tuned_performance_model.json")

    # 9. 최종 평가
    if best_model and best_score > 0:
        print(f"\n🎉 정밀 조정 성공!")
        print(f"   최고 모델: {best_model}")
        print(f"   Walk-Forward R²: {best_score:.4f}")

        # 이전 결과들과 비교
        robust_r2 = 0.0145
        improvement = (best_score - robust_r2) / robust_r2 * 100 if robust_r2 > 0 else 0

        print(f"   이전 Robust 모델: {robust_r2:.4f}")
        print(f"   개선도: {improvement:+.1f}%")

        if improvement > 50:
            print(f"   ✅ 유의미한 성능 개선!")
        elif improvement > 0:
            print(f"   ✅ 소폭 성능 개선")
        else:
            print(f"   ⚠️ 성능 유지 또는 하락")

    else:
        print(f"\n⚠️ 정밀 조정에도 성능 개선 실패")
        print(f"   최고 성능: {best_score:.4f}")
        print(f"   근본적인 접근법 재검토 필요")

    print("=" * 70)

if __name__ == "__main__":
    main()