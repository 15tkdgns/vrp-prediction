#!/usr/bin/env python3
"""
Robust Volatility Model - 과적합 해결 방안
심각한 과적합 문제를 해결하기 위한 보수적이고 안정적인 접근법
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import warnings
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

class RobustVolatilityPredictor:
    """과적합 방지를 위한 보수적 변동성 예측기"""

    def __init__(self):
        self.scaler = None
        self.model = None
        self.feature_names = None
        self.validation_results = {}

    def load_conservative_data(self):
        """보수적 데이터 로딩 - 과적합 방지"""
        print("📊 보수적 데이터 로딩 중...")

        # 더 긴 기간의 안정적인 데이터
        spy = yf.download('SPY', start='2008-01-01', end='2024-12-31', progress=False)
        spy['returns'] = spy['Close'].pct_change()

        # VIX (필수 경제 지표)
        vix = yf.download('^VIX', start='2008-01-01', end='2024-12-31', progress=False)
        spy['vix'] = vix['Close'].reindex(spy.index, method='ffill')

        spy = spy.dropna()
        print(f"✅ 안정적 데이터 로드: {len(spy)} 관측치 (2008-2024)")
        return spy

    def create_simple_robust_features(self, data):
        """단순하고 안정적인 특성만 생성"""
        print("🔧 보수적 특성 생성 중...")

        features = pd.DataFrame(index=data.index)
        returns = data['returns']
        prices = data['Close']
        high = data['High']
        low = data['Low']
        vix = data['vix']

        # 1. 핵심 변동성 특성만 (과도한 엔지니어링 방지)
        for window in [5, 10, 20]:
            features[f'volatility_{window}'] = returns.rolling(window).std()

        # 2. VIX 기본 특성 (가장 안정적)
        features['vix_level'] = vix / 100  # 정규화
        features['vix_ma_10'] = vix.rolling(10).mean() / 100
        features['vix_change'] = vix.pct_change()

        # 3. 단순한 가격 특성
        for window in [5, 10]:
            intraday_range = (high - low) / prices
            features[f'intraday_range_{window}'] = intraday_range.rolling(window).mean()

        # 4. 제한된 래그 특성 (과적합 방지)
        features['vol_lag_1'] = features['volatility_5'].shift(1)
        features['vol_lag_2'] = features['volatility_5'].shift(2)

        # 5. 안정적인 상호작용 (최소한만)
        features['vix_vol_ratio'] = features['vix_level'] / (features['volatility_10'] + 1e-8)

        print(f"✅ 보수적 특성 생성 완료: {len(features.columns)}개")
        return features

    def create_stable_targets(self, data):
        """안정적인 타겟 변수 생성"""
        print("🎯 안정적 타겟 생성 중...")

        targets = pd.DataFrame(index=data.index)
        returns = data['returns']

        # 5일 미래 변동성 (단순하고 안정적)
        vol_values = []
        for i in range(len(returns)):
            if i + 5 < len(returns):
                future_returns = returns.iloc[i+1:i+6]
                vol_values.append(future_returns.std())
            else:
                vol_values.append(np.nan)

        targets['target_vol_5d'] = vol_values
        return targets

    def apply_robust_preprocessing(self, X, y, test_size=0.3):
        """강건한 전처리 및 분할"""
        print("🛡️ 강건한 전처리 적용 중...")

        # 데이터 결합 및 정리
        combined_data = pd.concat([X, y], axis=1).dropna()
        print(f"전처리 후 샘플 수: {len(combined_data)}")

        if len(combined_data) < 1000:
            print("⚠️ 샘플 수 부족 - 최소 1000개 필요")
            return None, None, None, None

        # 시간순 분할 (매우 보수적)
        split_idx = int(len(combined_data) * (1 - test_size))

        train_data = combined_data.iloc[:split_idx]
        test_data = combined_data.iloc[split_idx:]

        X_train = train_data[X.columns]
        y_train = train_data[y.name]
        X_test = test_data[X.columns]
        y_test = test_data[y.name]

        print(f"훈련 세트: {len(X_train)} ({X_train.index[0].strftime('%Y-%m')} ~ {X_train.index[-1].strftime('%Y-%m')})")
        print(f"테스트 세트: {len(X_test)} ({X_test.index[0].strftime('%Y-%m')} ~ {X_test.index[-1].strftime('%Y-%m')})")

        return X_train, X_test, y_train, y_test

    def train_conservative_models(self, X_train, y_train):
        """보수적 모델 훈련 - 강한 정규화"""
        print("🤖 보수적 모델 훈련 중...")

        models = {
            'Ridge_Strong': Ridge(alpha=10.0),  # 강한 정규화
            'Ridge_VeryStrong': Ridge(alpha=50.0),  # 매우 강한 정규화
            'Lasso_Conservative': Lasso(alpha=0.01, max_iter=3000),  # 보수적 Lasso
            'RF_Simple': RandomForestRegressor(
                n_estimators=20,  # 적은 트리
                max_depth=3,      # 얕은 깊이
                min_samples_split=50,  # 보수적 분할
                min_samples_leaf=20,   # 보수적 리프
                random_state=42
            )
        }

        trained_models = {}

        for name, model in models.items():
            print(f"  훈련 중: {name}")

            if 'RF' not in name:
                # 스케일링 (RF 제외)
                scaler = RobustScaler()  # 이상치에 강건한 스케일러
                X_scaled = scaler.fit_transform(X_train)
                model.fit(X_scaled, y_train)
                trained_models[name] = {'model': model, 'scaler': scaler}
            else:
                model.fit(X_train, y_train)
                trained_models[name] = {'model': model, 'scaler': None}

        print(f"✅ {len(trained_models)}개 보수적 모델 훈련 완료")
        return trained_models

    def validate_with_walk_forward(self, X, y, trained_models, n_splits=5):
        """Walk-Forward 검증 (핵심 검증)"""
        print("🚀 Walk-Forward 검증 중...")

        combined_data = pd.concat([X, y], axis=1).dropna()
        X_clean = combined_data[X.columns]
        y_clean = combined_data[y.name]

        # 보수적 Walk-Forward 설정
        initial_window = 756  # 3년
        step_size = 126       # 6개월 단위

        results = {}
        for model_name in trained_models.keys():
            results[model_name] = []

        current_start = 0
        fold = 0

        while current_start + initial_window + 63 < len(X_clean):
            fold += 1

            # 훈련/테스트 분할
            train_end = current_start + initial_window
            test_start = train_end
            test_end = min(test_start + 63, len(X_clean))  # 3개월 예측

            X_train_fold = X_clean.iloc[current_start:train_end]
            y_train_fold = y_clean.iloc[current_start:train_end]
            X_test_fold = X_clean.iloc[test_start:test_end]
            y_test_fold = y_clean.iloc[test_start:test_end]

            print(f"  Fold {fold}: 훈련 {len(X_train_fold)}, 테스트 {len(X_test_fold)}")

            # 각 모델 테스트
            for model_name, model_info in trained_models.items():
                try:
                    model = model_info['model']
                    scaler = model_info['scaler']

                    if scaler is not None:
                        # 스케일링 적용
                        X_train_scaled = scaler.fit_transform(X_train_fold)
                        X_test_scaled = scaler.transform(X_test_fold)

                        # 모델 재훈련
                        model.fit(X_train_scaled, y_train_fold)
                        y_pred = model.predict(X_test_scaled)
                    else:
                        # RF - 스케일링 없음
                        model.fit(X_train_fold, y_train_fold)
                        y_pred = model.predict(X_test_fold)

                    # 성능 계산
                    r2 = r2_score(y_test_fold, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test_fold, y_pred))

                    results[model_name].append({
                        'fold': fold,
                        'r2': r2,
                        'rmse': rmse,
                        'test_period': f"{X_test_fold.index[0].strftime('%Y-%m')} ~ {X_test_fold.index[-1].strftime('%Y-%m')}"
                    })

                except Exception as e:
                    print(f"    {model_name} 오류: {e}")
                    results[model_name].append({
                        'fold': fold,
                        'r2': -999,
                        'rmse': 999,
                        'error': str(e)
                    })

            current_start += step_size

        # 결과 요약
        summary = {}
        print(f"\n📊 Walk-Forward 검증 결과:")
        print(f"{'Model':<20} {'Mean R²':<10} {'Std R²':<10} {'Valid Folds'}")
        print("-" * 55)

        for model_name, fold_results in results.items():
            valid_r2s = [r['r2'] for r in fold_results if r['r2'] != -999]

            if valid_r2s:
                mean_r2 = np.mean(valid_r2s)
                std_r2 = np.std(valid_r2s)
                valid_folds = len(valid_r2s)

                summary[model_name] = {
                    'mean_r2': mean_r2,
                    'std_r2': std_r2,
                    'valid_folds': valid_folds,
                    'total_folds': len(fold_results)
                }

                print(f"{model_name:<20} {mean_r2:<10.4f} {std_r2:<10.4f} {valid_folds}/{len(fold_results)}")
            else:
                summary[model_name] = {
                    'mean_r2': -999,
                    'std_r2': 0,
                    'valid_folds': 0,
                    'total_folds': len(fold_results)
                }
                print(f"{model_name:<20} {'FAILED':<10} {'N/A':<10} 0/{len(fold_results)}")

        return summary, results

    def evaluate_stability(self, wf_summary):
        """모델 안정성 평가"""
        print(f"\n🔍 모델 안정성 평가:")

        stable_models = []

        for model_name, stats in wf_summary.items():
            if stats['mean_r2'] > -900:
                r2_mean = stats['mean_r2']
                r2_std = stats['std_r2']
                success_rate = stats['valid_folds'] / stats['total_folds']

                # 안정성 기준
                is_positive = r2_mean > 0
                is_stable = r2_std < 0.5
                is_reliable = success_rate > 0.8

                stability_score = 0
                if is_positive:
                    stability_score += 3
                if is_stable:
                    stability_score += 2
                if is_reliable:
                    stability_score += 1

                stable_models.append({
                    'name': model_name,
                    'r2_mean': r2_mean,
                    'r2_std': r2_std,
                    'success_rate': success_rate,
                    'stability_score': stability_score,
                    'is_positive': is_positive,
                    'is_stable': is_stable,
                    'is_reliable': is_reliable
                })

                status = "✅" if stability_score >= 4 else "⚠️" if stability_score >= 2 else "❌"
                print(f"  {status} {model_name}: R²={r2_mean:.3f}±{r2_std:.3f}, 성공률={success_rate:.1%}, 점수={stability_score}/6")

        # 최고 안정 모델
        if stable_models:
            best_stable = max(stable_models, key=lambda x: x['stability_score'])
            print(f"\n🏆 최고 안정 모델: {best_stable['name']}")
            print(f"   안정성 점수: {best_stable['stability_score']}/6")
            print(f"   성능: R² = {best_stable['r2_mean']:.4f} ± {best_stable['r2_std']:.4f}")
            return best_stable
        else:
            print("❌ 안정적인 모델 없음")
            return None

def main():
    """메인 실행 함수"""
    print("🛡️ Robust Volatility Model - 과적합 해결")
    print("=" * 70)

    predictor = RobustVolatilityPredictor()

    # 1. 보수적 데이터 로드
    data = predictor.load_conservative_data()

    # 2. 단순하고 안정적인 특성 생성
    features = predictor.create_simple_robust_features(data)
    targets = predictor.create_stable_targets(data)

    # 3. 강건한 전처리
    X_train, X_test, y_train, y_test = predictor.apply_robust_preprocessing(
        features, targets['target_vol_5d']
    )

    if X_train is None:
        print("❌ 전처리 실패")
        return

    # 4. 보수적 모델 훈련
    trained_models = predictor.train_conservative_models(X_train, y_train)

    # 5. Walk-Forward 검증 (핵심)
    wf_summary, wf_details = predictor.validate_with_walk_forward(
        features, targets['target_vol_5d'], trained_models
    )

    # 6. 안정성 평가
    best_stable_model = predictor.evaluate_stability(wf_summary)

    # 7. 결과 저장
    os.makedirs('results', exist_ok=True)

    robust_results = {
        'version': 'Robust_Anti_Overfitting',
        'timestamp': datetime.now().isoformat(),
        'approach': 'Conservative and Stable',
        'data_period': '2008-2024',
        'samples': len(pd.concat([features, targets], axis=1).dropna()),
        'features': len(features.columns),
        'walk_forward_summary': wf_summary,
        'best_stable_model': best_stable_model,
        'feature_names': features.columns.tolist(),
        'validation_approach': 'Walk-Forward Only',
        'overfitting_prevention': [
            'Strong regularization (alpha=10-50)',
            'Conservative feature engineering',
            'Robust scaling',
            'Simple model architectures',
            'Walk-Forward validation only'
        ]
    }

    with open('results/robust_volatility_model.json', 'w') as f:
        json.dump(robust_results, f, indent=2, default=str)

    print(f"\n💾 보수적 모델 결과 저장: results/robust_volatility_model.json")

    # 8. 최종 평가
    if best_stable_model and best_stable_model['r2_mean'] > 0:
        print(f"\n🎉 성공: 안정적인 모델 발견")
        print(f"   모델: {best_stable_model['name']}")
        print(f"   Walk-Forward R²: {best_stable_model['r2_mean']:.4f}")
        print(f"   실제 거래 적용 가능성: {'높음' if best_stable_model['stability_score'] >= 4 else '보통'}")
    else:
        print(f"\n⚠️ 모든 모델이 과적합 또는 불안정")
        print(f"   추가 단순화 또는 다른 접근법 필요")

    print("=" * 70)

if __name__ == "__main__":
    main()