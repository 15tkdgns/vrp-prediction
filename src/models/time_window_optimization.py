#!/usr/bin/env python3
"""
Time Window Optimization - 시간 윈도우 최적화
다양한 예측 기간을 비교하여 최적 시간 윈도우 찾기
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
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

class TimeWindowOptimizer:
    """시간 윈도우 최적화기"""

    def __init__(self):
        self.results = {}

    def load_data(self):
        """데이터 로드"""
        print("📊 데이터 로드 중...")

        spy = yf.download('SPY', start='2015-01-01', end='2024-12-31', progress=False)  # 더 최근 데이터만
        spy['returns'] = spy['Close'].pct_change()

        vix = yf.download('^VIX', start='2015-01-01', end='2024-12-31', progress=False)
        spy['vix'] = vix['Close'].reindex(spy.index, method='ffill')

        spy = spy.dropna()
        print(f"✅ 데이터 로드: {len(spy)} 샘플 (2015-2024)")
        return spy

    def create_minimal_features(self, data):
        """최소한의 특성"""
        print("🔧 최소 특성 생성...")

        features = pd.DataFrame(index=data.index)
        returns = data['returns']
        prices = data['Close']
        high = data['High']
        low = data['Low']
        vix = data['vix']

        # 가장 핵심적인 특성들만
        features['volatility_5'] = returns.rolling(5).std()
        features['vix_level'] = vix / 100
        features['intraday_vol'] = ((high - low) / prices).rolling(5).mean()

        print(f"✅ 최소 특성: {len(features.columns)}개")
        return features

    def create_multiple_targets(self, data):
        """다양한 예측 기간의 타겟 생성"""
        print("🎯 다양한 타겟 생성...")

        targets = pd.DataFrame(index=data.index)
        returns = data['returns']

        # 1일, 2일, 3일, 5일, 10일 예측
        horizons = [1, 2, 3, 5, 10]

        for horizon in horizons:
            vol_values = []
            for i in range(len(returns)):
                if i + horizon < len(returns):
                    if horizon == 1:
                        # 1일 예측: 단순히 다음 날의 절댓값 수익률
                        vol_values.append(abs(returns.iloc[i+1]))
                    else:
                        # 다일 예측: 해당 기간의 표준편차
                        future_returns = returns.iloc[i+1:i+1+horizon]
                        vol_values.append(future_returns.std())
                else:
                    vol_values.append(np.nan)

            targets[f'target_vol_{horizon}d'] = vol_values

        print(f"✅ 타겟 생성: {len(horizons)}개 기간")
        return targets, horizons

    def quick_walkforward_test(self, X, y, horizon):
        """빠른 Walk-Forward 테스트"""
        combined_data = pd.concat([X, y], axis=1).dropna()
        X_clean = combined_data[X.columns]
        y_clean = combined_data[y.name]

        if len(X_clean) < 500:
            return {'mean_r2': -999, 'valid_folds': 0}

        # 매우 빠른 설정
        initial_window = 400  # 1.5년
        step_size = 50        # 2개월
        test_window = 10      # 2주
        max_folds = 10        # 최대 10개 폴드만

        model = Ridge(alpha=20.0)  # 보수적 정규화
        r2_scores = []

        current_start = 0
        fold = 0

        while fold < max_folds and current_start + initial_window + test_window < len(X_clean):
            fold += 1

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

                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)

                r2 = r2_score(y_test, y_pred)
                r2_scores.append(r2)

            except:
                r2_scores.append(-999)

            current_start += step_size

        valid_scores = [s for s in r2_scores if s > -900]

        if valid_scores:
            return {
                'mean_r2': np.mean(valid_scores),
                'std_r2': np.std(valid_scores),
                'max_r2': np.max(valid_scores),
                'valid_folds': len(valid_scores),
                'total_folds': len(r2_scores)
            }
        else:
            return {'mean_r2': -999, 'valid_folds': 0}

    def compare_time_windows(self, features, targets, horizons):
        """시간 윈도우 비교"""
        print("⏱️ 시간 윈도우 비교 중...")

        results = {}

        for horizon in horizons:
            print(f"\n  {horizon}일 예측 테스트 중...")

            target_col = f'target_vol_{horizon}d'
            if target_col not in targets.columns:
                continue

            result = self.quick_walkforward_test(features, targets[target_col], horizon)
            results[horizon] = result

            if result['valid_folds'] > 0:
                print(f"    R² = {result['mean_r2']:7.4f} ± {result.get('std_r2', 0):.4f} "
                      f"(최고: {result.get('max_r2', 0):.4f}, 성공: {result['valid_folds']}/{result.get('total_folds', 0)})")
            else:
                print(f"    실패")

        return results

    def find_optimal_window(self, results):
        """최적 시간 윈도우 찾기"""
        print(f"\n🎯 최적 시간 윈도우 분석:")

        valid_results = {h: r for h, r in results.items() if r.get('valid_folds', 0) > 0}

        if not valid_results:
            print("❌ 모든 시간 윈도우에서 실패")
            return None, None

        # 평균 R² 기준으로 최고 성능
        best_horizon = max(valid_results.items(), key=lambda x: x[1]['mean_r2'])

        print(f"\n📊 시간 윈도우별 성능:")
        print(f"{'기간':<8} {'평균 R²':<12} {'표준편차':<12} {'최고 R²':<12} {'성공률'}")
        print("-" * 60)

        for horizon, stats in valid_results.items():
            success_rate = stats['valid_folds'] / stats.get('total_folds', 1) * 100
            marker = "🏆" if horizon == best_horizon[0] else "  "

            print(f"{marker} {horizon}일{'':<4} {stats['mean_r2']:<12.4f} "
                  f"{stats.get('std_r2', 0):<12.4f} {stats.get('max_r2', 0):<12.4f} "
                  f"{success_rate:>6.1f}%")

        print(f"\n🏆 최적 시간 윈도우: {best_horizon[0]}일")
        print(f"   평균 R²: {best_horizon[1]['mean_r2']:.4f}")
        print(f"   표준편차: {best_horizon[1].get('std_r2', 0):.4f}")
        print(f"   최고 성능: {best_horizon[1].get('max_r2', 0):.4f}")

        return best_horizon[0], best_horizon[1]

    def detailed_test_optimal_window(self, features, targets, optimal_horizon):
        """최적 윈도우로 상세 테스트"""
        print(f"\n🔍 {optimal_horizon}일 예측 상세 테스트...")

        target_col = f'target_vol_{optimal_horizon}d'
        combined_data = pd.concat([features, targets[[target_col]]], axis=1).dropna()
        X_clean = combined_data[features.columns]
        y_clean = combined_data[target_col]

        # 더 엄격한 설정
        initial_window = 600  # 2.5년
        step_size = 30        # 1개월
        test_window = 15      # 3주

        model = Ridge(alpha=20.0)
        results = []

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

            try:
                scaler = RobustScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)

                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                results.append({
                    'fold': fold,
                    'r2': r2,
                    'rmse': rmse,
                    'test_period': f"{X_test.index[0].strftime('%Y-%m')} ~ {X_test.index[-1].strftime('%Y-%m')}"
                })

                print(f"  Fold {fold:2d}: R²={r2:7.4f}, RMSE={rmse:.5f} ({X_test.index[0].strftime('%Y-%m')})")

            except Exception as e:
                print(f"  Fold {fold:2d}: 오류 - {e}")
                results.append({'fold': fold, 'r2': -999, 'rmse': 999})

            current_start += step_size

        # 상세 분석
        valid_r2s = [r['r2'] for r in results if r['r2'] != -999]

        if valid_r2s:
            detailed_stats = {
                'mean_r2': np.mean(valid_r2s),
                'std_r2': np.std(valid_r2s),
                'min_r2': np.min(valid_r2s),
                'max_r2': np.max(valid_r2s),
                'positive_folds': sum(1 for r2 in valid_r2s if r2 > 0),
                'total_valid_folds': len(valid_r2s),
                'success_rate': sum(1 for r2 in valid_r2s if r2 > 0) / len(valid_r2s) * 100
            }

            print(f"\n📊 {optimal_horizon}일 예측 상세 결과:")
            print(f"   평균 R²: {detailed_stats['mean_r2']:.4f} ± {detailed_stats['std_r2']:.4f}")
            print(f"   범위: {detailed_stats['min_r2']:.4f} ~ {detailed_stats['max_r2']:.4f}")
            print(f"   양의 성능: {detailed_stats['positive_folds']}/{detailed_stats['total_valid_folds']} "
                  f"({detailed_stats['success_rate']:.1f}%)")

            return detailed_stats, results
        else:
            print(f"❌ {optimal_horizon}일 예측 상세 테스트 실패")
            return None, results

def main():
    """메인 시간 윈도우 최적화 함수"""
    print("⏱️ Time Window Optimization - 시간 윈도우 최적화")
    print("=" * 70)

    optimizer = TimeWindowOptimizer()

    # 1. 데이터 로드
    data = optimizer.load_data()

    # 2. 최소 특성 생성
    features = optimizer.create_minimal_features(data)

    # 3. 다양한 타겟 생성
    targets, horizons = optimizer.create_multiple_targets(data)

    # 4. 시간 윈도우 비교
    comparison_results = optimizer.compare_time_windows(features, targets, horizons)

    # 5. 최적 윈도우 찾기
    optimal_horizon, optimal_stats = optimizer.find_optimal_window(comparison_results)

    if optimal_horizon is None:
        print("\n❌ 최적 시간 윈도우를 찾을 수 없음")
        print("   모든 예측 기간에서 음의 성능")
        return

    # 6. 최적 윈도우 상세 테스트
    detailed_stats, detailed_results = optimizer.detailed_test_optimal_window(
        features, targets, optimal_horizon
    )

    # 7. 결과 저장
    os.makedirs('results', exist_ok=True)

    time_window_results = {
        'version': 'Time_Window_Optimization',
        'timestamp': datetime.now().isoformat(),
        'data_period': '2015-2024',
        'horizons_tested': horizons,
        'comparison_results': comparison_results,
        'optimal_horizon': optimal_horizon,
        'optimal_stats': optimal_stats,
        'detailed_stats': detailed_stats,
        'feature_names': features.columns.tolist()
    }

    with open('results/time_window_optimization.json', 'w') as f:
        json.dump(time_window_results, f, indent=2, default=str)

    print(f"\n💾 결과 저장: results/time_window_optimization.json")

    # 8. 최종 평가
    if detailed_stats and detailed_stats['mean_r2'] > 0:
        print(f"\n🎉 시간 윈도우 최적화 성공!")
        print(f"   최적 예측 기간: {optimal_horizon}일")
        print(f"   평균 성능: R² = {detailed_stats['mean_r2']:.4f}")
        print(f"   양의 성능 비율: {detailed_stats['success_rate']:.1f}%")

        # 이전 최고 결과와 비교
        previous_best = 0.0145  # 이전 Robust 모델
        if detailed_stats['mean_r2'] > previous_best:
            improvement = (detailed_stats['mean_r2'] - previous_best) / previous_best * 100
            print(f"   이전 최고 대비: +{improvement:.1f}% 개선")
        else:
            print(f"   이전 최고 (R²={previous_best:.4f})에는 미치지 못함")

    elif detailed_stats:
        print(f"\n⚠️ 최적화 완료, 하지만 여전히 음의 성능")
        print(f"   최적 예측 기간: {optimal_horizon}일")
        print(f"   평균 성능: R² = {detailed_stats['mean_r2']:.4f}")
        print(f"   양의 성능 폴드: {detailed_stats['positive_folds']}/{detailed_stats['total_valid_folds']}")

    else:
        print(f"\n❌ 시간 윈도우 최적화 실패")
        print(f"   모든 예측 기간에서 신뢰할 수 있는 성능을 얻지 못함")

    print("=" * 70)

if __name__ == "__main__":
    main()