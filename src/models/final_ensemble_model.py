#!/usr/bin/env python3
"""
Final Ensemble Model - 최종 앙상블 접근법
다양한 전략을 조합한 마지막 성능 개선 시도
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
import warnings
import os
import json
from datetime import datetime

warnings.filterwarnings('ignore')

class FinalEnsemblePredictor:
    """최종 앙상블 예측기"""

    def __init__(self):
        self.base_models = {}
        self.ensemble_weights = {}

    def load_stable_data(self):
        """안정적 데이터 로드"""
        print("📊 안정적 데이터 로드 중...")

        # 이전에 가장 좋았던 기간 사용
        spy = yf.download('SPY', start='2008-01-01', end='2024-12-31', progress=False)
        spy['returns'] = spy['Close'].pct_change()

        vix = yf.download('^VIX', start='2008-01-01', end='2024-12-31', progress=False)
        spy['vix'] = vix['Close'].reindex(spy.index, method='ffill')

        spy = spy.dropna()
        print(f"✅ 데이터 로드: {len(spy)} 샘플 (2008-2024)")
        return spy

    def create_ultra_simple_features(self, data):
        """극도로 단순한 특성만 (최고 성능 특성들만)"""
        print("🔧 극단적 단순화 특성 생성...")

        features = pd.DataFrame(index=data.index)
        returns = data['returns']
        prices = data['Close']
        high = data['High']
        low = data['Low']
        vix = data['vix']

        # 이전 분석에서 가장 강력했던 특성들만
        features['volatility_5'] = returns.rolling(5).std()
        features['volatility_10'] = returns.rolling(10).std()
        features['vix_level'] = vix / 100
        features['vix_ma_5'] = vix.rolling(5).mean() / 100

        # 일중 변동성 (가장 강력했던 특성)
        intraday_range = (high - low) / prices
        features['intraday_vol_5'] = intraday_range.rolling(5).mean()
        features['intraday_vol_10'] = intraday_range.rolling(10).mean()

        print(f"✅ 극단적 단순 특성: {len(features.columns)}개")
        return features

    def create_diverse_models(self):
        """다양한 특성을 가진 기본 모델들"""
        print("🤖 다양한 기본 모델 생성...")

        models = {
            'Ultra_Conservative': Ridge(alpha=100.0),   # 매우 보수적
            'Conservative': Ridge(alpha=50.0),          # 보수적
            'Moderate': Ridge(alpha=20.0),              # 중간
            'Aggressive': Ridge(alpha=10.0),            # 공격적
            'Ultra_Aggressive': Ridge(alpha=5.0)        # 매우 공격적
        }

        print(f"✅ 기본 모델 {len(models)}개 준비")
        return models

    def adaptive_ensemble_walkforward(self, X, y):
        """적응형 앙상블 Walk-Forward"""
        print("🚀 적응형 앙상블 Walk-Forward 테스트...")

        combined_data = pd.concat([X, y], axis=1).dropna()
        X_clean = combined_data[X.columns]
        y_clean = combined_data[y.name]

        print(f"테스트 데이터: {len(X_clean)} 샘플")

        # 매우 보수적 설정
        initial_window = 800   # 약 3년
        step_size = 60         # 3개월
        test_window = 20       # 1개월 (더 짧은 예측)
        lookback_window = 5    # 최근 5개 성능으로 가중치 계산

        models = self.create_diverse_models()

        # 결과 저장
        results = {model_name: [] for model_name in models.keys()}
        results['Ensemble_Equal'] = []
        results['Ensemble_Adaptive'] = []

        # 각 모델의 최근 성능 추적
        recent_performance = {model_name: [] for model_name in models.keys()}

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

            # 개별 모델 예측
            individual_predictions = {}
            individual_r2s = {}

            for model_name, model in models.items():
                try:
                    scaler = RobustScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)

                    r2 = r2_score(y_test, y_pred)
                    individual_predictions[model_name] = y_pred
                    individual_r2s[model_name] = r2

                    # 최근 성능 추적 업데이트
                    recent_performance[model_name].append(r2)
                    if len(recent_performance[model_name]) > lookback_window:
                        recent_performance[model_name].pop(0)

                    results[model_name].append({
                        'fold': fold,
                        'r2': r2,
                        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
                    })

                    print(f"    {model_name:20}: R²={r2:7.4f}")

                except Exception as e:
                    print(f"    {model_name:20}: 오류 - {e}")
                    results[model_name].append({'fold': fold, 'r2': -999, 'rmse': 999})
                    individual_r2s[model_name] = -999

            # 앙상블 예측들
            if len(individual_predictions) >= 2:
                valid_predictions = {k: v for k, v in individual_predictions.items()
                                   if individual_r2s[k] > -900}

                if len(valid_predictions) >= 2:
                    # 1. 동등 가중치 앙상블
                    equal_pred = np.mean(list(valid_predictions.values()), axis=0)
                    equal_r2 = r2_score(y_test, equal_pred)

                    results['Ensemble_Equal'].append({
                        'fold': fold,
                        'r2': equal_r2,
                        'rmse': np.sqrt(mean_squared_error(y_test, equal_pred))
                    })

                    print(f"    {'Ensemble_Equal':20}: R²={equal_r2:7.4f}")

                    # 2. 적응형 가중치 앙상블
                    if fold > 3:  # 충분한 이력이 있을 때만
                        weights = {}
                        total_weight = 0

                        for model_name in valid_predictions.keys():
                            recent_perf = recent_performance.get(model_name, [])
                            if recent_perf:
                                # 최근 성능의 평균 (음수는 0으로)
                                avg_perf = max(0, np.mean(recent_perf))
                                weights[model_name] = avg_perf
                                total_weight += avg_perf

                        # 가중치 정규화
                        if total_weight > 0:
                            for model_name in weights:
                                weights[model_name] /= total_weight
                        else:
                            # 모든 모델이 음수 성능이면 동등 가중치
                            for model_name in weights:
                                weights[model_name] = 1.0 / len(weights)

                        # 가중 평균 예측
                        adaptive_pred = np.zeros_like(equal_pred)
                        for model_name, pred in valid_predictions.items():
                            weight = weights.get(model_name, 0)
                            adaptive_pred += weight * pred

                        adaptive_r2 = r2_score(y_test, adaptive_pred)

                        results['Ensemble_Adaptive'].append({
                            'fold': fold,
                            'r2': adaptive_r2,
                            'rmse': np.sqrt(mean_squared_error(y_test, adaptive_pred)),
                            'weights': weights.copy()
                        })

                        print(f"    {'Ensemble_Adaptive':20}: R²={adaptive_r2:7.4f}")

                        # 가중치 출력 (상위 3개)
                        top_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:3]
                        weight_str = ", ".join([f"{name}:{weight:.2f}" for name, weight in top_weights])
                        print(f"      가중치: {weight_str}")

            current_start += step_size

        # 결과 요약
        summary = {}
        print(f"\n📊 적응형 앙상블 결과:")
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

    def analyze_ensemble_effectiveness(self, summary):
        """앙상블 효과 분석"""
        print(f"\n🔍 앙상블 효과 분석:")

        # 개별 모델들의 최고 성능
        individual_best = -999
        individual_best_name = None

        ensemble_performances = {}

        for model_name, stats in summary.items():
            mean_r2 = stats.get('mean_r2', -999)

            if 'Ensemble' not in model_name:
                if mean_r2 > individual_best:
                    individual_best = mean_r2
                    individual_best_name = model_name
            else:
                ensemble_performances[model_name] = mean_r2

        print(f"  개별 모델 최고: {individual_best_name} (R² = {individual_best:.4f})")

        for ensemble_name, ensemble_r2 in ensemble_performances.items():
            improvement = (ensemble_r2 - individual_best) / abs(individual_best) * 100 if individual_best != 0 else 0
            print(f"  {ensemble_name}: R² = {ensemble_r2:.4f} (개선: {improvement:+.1f}%)")

        # 최고 성능 모델 선택
        all_performances = [(name, stats.get('mean_r2', -999)) for name, stats in summary.items()]
        best_model, best_score = max(all_performances, key=lambda x: x[1])

        print(f"\n🏆 최고 성능: {best_model} (R² = {best_score:.4f})")

        return best_model, best_score

def main():
    """메인 최종 앙상블 함수"""
    print("🎯 Final Ensemble Model - 최종 성능 개선 시도")
    print("=" * 80)
    print("적응형 앙상블 + 극단적 단순화 + 짧은 예측 기간")
    print("=" * 80)

    predictor = FinalEnsemblePredictor()

    # 1. 안정적 데이터 로드
    data = predictor.load_stable_data()

    # 2. 극도로 단순한 특성
    features = predictor.create_ultra_simple_features(data)

    # 3. 타겟 생성 (더 짧은 기간)
    targets = pd.DataFrame(index=data.index)
    returns = data['returns']

    # 3일 예측으로 단축 (더 안정적일 수 있음)
    vol_values = []
    for i in range(len(returns)):
        if i + 3 < len(returns):
            future_returns = returns.iloc[i+1:i+4]
            vol_values.append(future_returns.std())
        else:
            vol_values.append(np.nan)
    targets['target_vol_3d'] = vol_values

    # 4. 적응형 앙상블 Walk-Forward
    summary, detailed_results = predictor.adaptive_ensemble_walkforward(
        features, targets['target_vol_3d']
    )

    # 5. 앙상블 효과 분석
    best_model, best_score = predictor.analyze_ensemble_effectiveness(summary)

    # 6. 결과 저장
    os.makedirs('results', exist_ok=True)

    final_results = {
        'version': 'Final_Ensemble_Ultimate',
        'timestamp': datetime.now().isoformat(),
        'approach': 'Adaptive ensemble + ultra-simple features + short prediction',
        'data_samples': len(pd.concat([features, targets], axis=1).dropna()),
        'features_count': len(features.columns),
        'prediction_horizon': '3 days',
        'best_model': {
            'name': best_model,
            'mean_r2': best_score,
            'std_r2': summary.get(best_model, {}).get('std_r2', 0) if best_model else 0
        },
        'all_results': summary,
        'feature_names': features.columns.tolist(),
        'ensemble_strategy': 'Adaptive weighting based on recent performance'
    }

    with open('results/final_ensemble_model.json', 'w') as f:
        json.dump(final_results, f, indent=2, default=str)

    print(f"\n💾 최종 결과 저장: results/final_ensemble_model.json")

    # 7. 최종 평가 및 결론
    print(f"\n🏁 최종 평가:")

    if best_score > 0:
        print(f"✅ 성공: 양의 R² 달성!")
        print(f"   최고 모델: {best_model}")
        print(f"   성능: R² = {best_score:.4f}")

        # 이전 모델들과 비교
        robust_r2 = 0.0145
        improvement = (best_score - robust_r2) / robust_r2 * 100 if robust_r2 > 0 else 0

        print(f"   이전 Robust 모델: {robust_r2:.4f}")
        print(f"   개선도: {improvement:+.1f}%")

        if best_score > robust_r2 * 2:
            print(f"   🎉 대폭적인 성능 개선!")
        elif best_score > robust_r2:
            print(f"   ✅ 유의미한 성능 개선")
        else:
            print(f"   📈 성능 유지")

    else:
        print(f"❌ 실패: 여전히 음의 성능")
        print(f"   최고 성능: {best_score:.4f}")
        print(f"   결론: 변동성 예측은 근본적으로 어려운 문제")

    print(f"\n💡 최종 결론:")
    if best_score > 0:
        print(f"   변동성 예측에서 작은 양의 성능도 가치 있음")
        print(f"   실제 거래에서는 리스크 관리 목적으로 활용 가능")
    else:
        print(f"   복잡한 모델보다 단순한 규칙 기반 접근이 더 나을 수 있음")
        print(f"   변동성 예측의 한계를 인정하고 다른 접근법 고려 필요")

    print("=" * 80)

if __name__ == "__main__":
    main()