#!/usr/bin/env python3
"""
Walk-Forward Validation - 실제 거래 환경 시뮬레이션
시간적 안정성과 실제 운용 가능성 검증
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
import os
import json
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

def load_data_for_walkforward():
    """Walk-Forward용 데이터 로드"""
    print("📊 Walk-Forward 검증용 데이터 로드 중...")

    # SPY 데이터
    spy = yf.download('SPY', start='2015-01-01', end='2024-12-31', progress=False)
    spy['returns'] = spy['Close'].pct_change()

    # VIX 데이터
    vix = yf.download('^VIX', start='2015-01-01', end='2024-12-31', progress=False)
    spy['vix'] = vix['Close'].reindex(spy.index, method='ffill')

    # 10년 국채 금리
    try:
        treasury = yf.download('^TNX', start='2015-01-01', end='2024-12-31', progress=False)
        spy['treasury_10y'] = treasury['Close'].reindex(spy.index, method='ffill')
    except:
        spy['treasury_10y'] = 2.5

    spy = spy.dropna()
    print(f"✅ 데이터 로드 완료: {len(spy)} 관측치")
    return spy

def create_core_features_for_wf(data):
    """Walk-Forward용 핵심 특성 생성"""
    print("🔧 핵심 특성 생성 중...")

    features = pd.DataFrame(index=data.index)
    returns = data['returns']
    prices = data['Close']
    high = data['High']
    low = data['Low']

    # 1. 핵심 변동성 특성
    for window in [5, 10, 20]:
        features[f'volatility_{window}'] = returns.rolling(window).std()

    # 2. VIX 특성
    if 'vix' in data.columns:
        vix = data['vix']
        features['vix_level'] = vix
        features['vix_ma_5'] = vix.rolling(5).mean()
        features['vix_ma_20'] = vix.rolling(20).mean()
        features['vix_term_structure'] = vix / features['vix_ma_20']

    # 3. 경제 지표
    if 'treasury_10y' in data.columns:
        treasury = data['treasury_10y']
        features['treasury_10y'] = treasury
        features['vix_treasury_spread'] = features['vix_level'] - treasury

    # 4. 고급 변동성
    for window in [5, 10]:
        intraday_range = (high - low) / prices
        features[f'intraday_vol_{window}'] = intraday_range.rolling(window).mean()

    # 5. 지수 가중 변동성
    features['ewm_vol_10'] = returns.ewm(span=10).std()

    # 6. 래그 특성
    for lag in [1, 2, 3]:
        features[f'vol_lag_{lag}'] = features['volatility_5'].shift(lag)

    print(f"✅ 특성 생성 완료: {len(features.columns)}개")
    return features

def create_targets_for_wf(data):
    """Walk-Forward용 타겟 생성"""
    targets = pd.DataFrame(index=data.index)
    returns = data['returns']

    vol_values = []
    for i in range(len(returns)):
        if i + 5 < len(returns):
            future_window = returns.iloc[i+1:i+6]
            vol_values.append(future_window.std())
        else:
            vol_values.append(np.nan)
    targets['target_vol_5d'] = vol_values

    return targets

class WalkForwardValidator:
    """Walk-Forward 검증 클래스"""

    def __init__(self, initial_window=756, refit_frequency=63, prediction_horizon=21):
        """
        초기화
        initial_window: 초기 훈련 윈도우 (3년 = 756일)
        refit_frequency: 재훈련 빈도 (분기별 = 63일)
        prediction_horizon: 예측 기간 (1개월 = 21일)
        """
        self.initial_window = initial_window
        self.refit_frequency = refit_frequency
        self.prediction_horizon = prediction_horizon
        self.results = []

    def run_walk_forward(self, X, y, model_configs):
        """Walk-Forward 검증 실행"""
        print(f"🚀 Walk-Forward 검증 시작")
        print(f"  초기 훈련 윈도우: {self.initial_window}일")
        print(f"  재훈련 빈도: {self.refit_frequency}일")
        print(f"  예측 기간: {self.prediction_horizon}일")

        # 데이터 정리
        combined_data = pd.concat([X, y], axis=1).dropna()
        X_clean = combined_data[X.columns]
        y_clean = combined_data[y.name]

        print(f"  총 샘플 수: {len(X_clean)}")

        # Walk-Forward 루프
        current_start = 0
        fold_count = 0

        while current_start + self.initial_window + self.prediction_horizon < len(X_clean):
            fold_count += 1

            # 훈련 데이터
            train_end = current_start + self.initial_window
            X_train = X_clean.iloc[current_start:train_end]
            y_train = y_clean.iloc[current_start:train_end]

            # 테스트 데이터
            test_start = train_end
            test_end = min(test_start + self.prediction_horizon, len(X_clean))
            X_test = X_clean.iloc[test_start:test_end]
            y_test = y_clean.iloc[test_start:test_end]

            print(f"\n  Fold {fold_count}: 훈련 {len(X_train)}, 테스트 {len(X_test)}")
            print(f"    훈련 기간: {X_train.index[0].strftime('%Y-%m-%d')} ~ {X_train.index[-1].strftime('%Y-%m-%d')}")
            print(f"    테스트 기간: {X_test.index[0].strftime('%Y-%m-%d')} ~ {X_test.index[-1].strftime('%Y-%m-%d')}")

            # 각 모델 테스트
            fold_results = {
                'fold': fold_count,
                'train_start': X_train.index[0],
                'train_end': X_train.index[-1],
                'test_start': X_test.index[0],
                'test_end': X_test.index[-1],
                'models': {}
            }

            for model_name, model_config in model_configs.items():
                try:
                    # 모델 훈련
                    model = model_config['model']
                    use_scaling = model_config.get('scaling', True)

                    if use_scaling:
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                    else:
                        X_train_scaled = X_train.values
                        X_test_scaled = X_test.values

                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)

                    # 성능 계산
                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                    fold_results['models'][model_name] = {
                        'r2': r2,
                        'mae': mae,
                        'rmse': rmse,
                        'predictions': y_pred.tolist(),
                        'actuals': y_test.tolist()
                    }

                    print(f"      {model_name:20}: R² = {r2:6.3f}, MAE = {mae:.5f}")

                except Exception as e:
                    print(f"      {model_name:20}: 오류 - {e}")
                    fold_results['models'][model_name] = {
                        'r2': -999,
                        'mae': 999,
                        'rmse': 999,
                        'error': str(e)
                    }

            self.results.append(fold_results)

            # 다음 윈도우로 이동
            current_start += self.refit_frequency

        print(f"\n✅ Walk-Forward 검증 완료: {fold_count}개 폴드")
        return self.results

    def analyze_results(self):
        """결과 분석"""
        print(f"\n📊 Walk-Forward 결과 분석")
        print("=" * 60)

        if not self.results:
            print("❌ 분석할 결과가 없습니다")
            return {}

        # 모델별 성능 집계
        model_names = list(self.results[0]['models'].keys())
        summary = {}

        for model_name in model_names:
            r2_scores = []
            mae_scores = []

            for fold_result in self.results:
                model_result = fold_result['models'][model_name]
                if model_result['r2'] != -999:  # 오류가 아닌 경우만
                    r2_scores.append(model_result['r2'])
                    mae_scores.append(model_result['mae'])

            if r2_scores:
                summary[model_name] = {
                    'mean_r2': np.mean(r2_scores),
                    'std_r2': np.std(r2_scores),
                    'min_r2': np.min(r2_scores),
                    'max_r2': np.max(r2_scores),
                    'mean_mae': np.mean(mae_scores),
                    'successful_folds': len(r2_scores),
                    'total_folds': len(self.results)
                }

        # 결과 출력
        print(f"{'Model':<25} {'Mean R²':<10} {'Std R²':<10} {'Min R²':<10} {'Max R²':<10} {'Success Rate'}")
        print("-" * 80)

        for model_name, stats in summary.items():
            success_rate = stats['successful_folds'] / stats['total_folds'] * 100
            print(f"{model_name:<25} {stats['mean_r2']:<10.4f} {stats['std_r2']:<10.4f} "
                  f"{stats['min_r2']:<10.4f} {stats['max_r2']:<10.4f} {success_rate:>10.1f}%")

        return summary

    def create_time_series_plot(self, model_name='best'):
        """시계열 성능 플롯 생성"""
        print(f"📊 {model_name} 모델 시계열 성능 플롯 생성 중...")

        if not self.results:
            print("❌ 플롯할 데이터가 없습니다")
            return

        # 최고 성능 모델 선택
        if model_name == 'best':
            summary = self.analyze_results()
            if summary:
                model_name = max(summary.items(), key=lambda x: x[1]['mean_r2'])[0]
            else:
                return

        # 시계열 데이터 추출
        dates = []
        r2_scores = []

        for fold_result in self.results:
            if model_name in fold_result['models']:
                model_result = fold_result['models'][model_name]
                if model_result['r2'] != -999:
                    dates.append(fold_result['test_start'])
                    r2_scores.append(model_result['r2'])

        if not dates:
            print(f"❌ {model_name} 모델의 유효한 결과가 없습니다")
            return

        # 플롯 생성
        plt.figure(figsize=(12, 6))
        plt.plot(dates, r2_scores, marker='o', linewidth=2, markersize=6)
        plt.title(f'Walk-Forward R² Performance: {model_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Test Period Start Date')
        plt.ylabel('R² Score')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)

        # 평균선 추가
        mean_r2 = np.mean(r2_scores)
        plt.axhline(y=mean_r2, color='red', linestyle='--', alpha=0.7,
                   label=f'Mean R² = {mean_r2:.4f}')
        plt.legend()

        plt.tight_layout()

        # 저장
        os.makedirs('figures', exist_ok=True)
        plt.savefig('figures/walk_forward_performance.png', dpi=300, bbox_inches='tight')
        print(f"✅ 저장: figures/walk_forward_performance.png")
        plt.close()

def main():
    """메인 Walk-Forward 검증 함수"""
    print("🚀 Walk-Forward Validation - 실제 거래 환경 시뮬레이션")
    print("=" * 70)

    # 1. 데이터 로드
    spy_data = load_data_for_walkforward()

    # 2. 특성 및 타겟 생성
    features = create_core_features_for_wf(spy_data)
    targets = create_targets_for_wf(spy_data)

    # 3. 상위 특성 선별 (빠른 실행을 위해)
    combined = pd.concat([features, targets], axis=1).dropna()
    correlations = combined[features.columns].corrwith(combined['target_vol_5d']).abs().sort_values(ascending=False)

    top_10_features = correlations.head(10).index
    final_features = features[top_10_features]

    print(f"📊 선별된 특성: {len(final_features.columns)}개")
    print("상위 10개 특성:")
    for i, (feature, corr) in enumerate(correlations.head(10).items()):
        print(f"  {i+1:2d}. {feature:25}: {corr:.4f}")

    # 4. 모델 구성
    model_configs = {
        'Lasso_0001': {
            'model': Lasso(alpha=0.0001, max_iter=2000),
            'scaling': True
        },
        'Lasso_0005': {
            'model': Lasso(alpha=0.0005, max_iter=2000),
            'scaling': True
        },
        'ElasticNet': {
            'model': ElasticNet(alpha=0.0005, l1_ratio=0.7, max_iter=2000),
            'scaling': True
        },
        'RandomForest': {
            'model': RandomForestRegressor(n_estimators=50, random_state=42, max_depth=6),
            'scaling': False
        }
    }

    # 5. Walk-Forward 검증 실행
    validator = WalkForwardValidator(
        initial_window=504,  # 2년
        refit_frequency=63,  # 분기별
        prediction_horizon=21  # 1개월
    )

    results = validator.run_walk_forward(final_features, targets['target_vol_5d'], model_configs)

    # 6. 결과 분석
    summary = validator.analyze_results()

    # 7. 시각화
    validator.create_time_series_plot()

    # 8. 결과 저장
    os.makedirs('results', exist_ok=True)

    walk_forward_results = {
        'validation_type': 'Walk-Forward',
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'initial_window': validator.initial_window,
            'refit_frequency': validator.refit_frequency,
            'prediction_horizon': validator.prediction_horizon
        },
        'data_period': '2015-2024',
        'features_used': top_10_features.tolist(),
        'total_folds': len(results),
        'summary': summary,
        'detailed_results': results
    }

    with open('results/walk_forward_validation.json', 'w') as f:
        json.dump(walk_forward_results, f, indent=2, default=str)

    print(f"\n💾 Walk-Forward 결과 저장: results/walk_forward_validation.json")

    # 최고 성능 모델
    if summary:
        best_model = max(summary.items(), key=lambda x: x[1]['mean_r2'])
        print(f"\n🏆 Walk-Forward 최고 성능: {best_model[0]}")
        print(f"   평균 R²: {best_model[1]['mean_r2']:.4f} ± {best_model[1]['std_r2']:.4f}")
        print(f"   성공률: {best_model[1]['successful_folds']}/{best_model[1]['total_folds']} "
              f"({best_model[1]['successful_folds']/best_model[1]['total_folds']*100:.1f}%)")

    print("=" * 70)

if __name__ == "__main__":
    main()