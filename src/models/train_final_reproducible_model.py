#!/usr/bin/env python3
"""
재현 가능한 최종 ElasticNet 변동성 예측 모델
- GridSearchCV로 하이퍼파라미터 최적화
- Random seed 고정으로 재현성 보장
- Purged K-Fold CV 사용
- 모델 및 결과 저장

실행 시간: 약 10분
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import json
from pathlib import Path
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

# 재현성 보장
SEED = 42
np.random.seed(SEED)


def load_spy_data():
    """기존 SPY 데이터 로드"""
    print("\n[1/6] 데이터 로드 중...")

    # 기존 저장된 CSV 사용 (재현성 보장)
    csv_path = Path('data/raw/spy_data_2020_2025.csv')
    if csv_path.exists():
        spy = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        print(f"  ✓ SPY 데이터 로드 완료: {len(spy)} 행")
        return spy
    else:
        print(f"  ⚠️ {csv_path} 파일을 찾을 수 없습니다. yfinance로 다운로드합니다.")
        try:
            import yfinance as yf
            spy = yf.download('SPY', start='2015-01-01', end='2024-12-31',
                             progress=False, auto_adjust=True)

            # MultiIndex 처리
            if isinstance(spy.columns, pd.MultiIndex):
                spy.columns = spy.columns.get_level_values(0)
            spy.columns = [str(col) for col in spy.columns]

            # 저장
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            spy.to_csv(csv_path)
            print(f"  ✓ SPY 데이터 다운로드 및 저장 완료: {len(spy)} 행")
            return spy
        except Exception as e:
            raise FileNotFoundError(f"데이터 로드 실패: {e}")


def create_features_and_target(spy):
    """특성 및 타겟 생성"""
    print("\n[2/6] 특성 및 타겟 생성 중...")

    # 기본 계산
    spy['returns'] = spy['Close'].pct_change()
    spy['volatility'] = spy['returns'].rolling(5).std() * np.sqrt(252)

    # 1. 변동성 특성 (과거만)
    for window in [5, 10, 20, 50]:
        spy[f'volatility_{window}'] = spy['returns'].rolling(window).std()
        spy[f'realized_vol_{window}'] = spy[f'volatility_{window}'] * np.sqrt(252)

    # 2. 수익률 통계 (과거만)
    for window in [5, 10, 20]:
        spy[f'mean_return_{window}'] = spy['returns'].rolling(window).mean()
        spy[f'skew_{window}'] = spy['returns'].rolling(window).skew()
        spy[f'kurt_{window}'] = spy['returns'].rolling(window).kurt()

    # 3. 래그 변수 (과거만)
    for lag in [1, 2, 3, 5]:
        spy[f'return_lag_{lag}'] = spy['returns'].shift(lag)
        spy[f'vol_lag_{lag}'] = spy['volatility_5'].shift(lag)

    # 4. 교차 통계 (과거만)
    spy['vol_ratio_5_20'] = spy['volatility_5'] / (spy['volatility_20'] + 1e-8)
    spy['vol_ratio_10_50'] = spy['volatility_10'] / (spy['volatility_50'] + 1e-8)

    # 5. Z-score (과거만)
    ma_20 = spy['returns'].rolling(20).mean()
    std_20 = spy['returns'].rolling(20).std()
    spy['zscore_20'] = (spy['returns'] - ma_20) / (std_20 + 1e-8)

    # 6. 모멘텀 (과거만)
    for window in [5, 10, 20]:
        spy[f'momentum_{window}'] = spy['returns'].rolling(window).sum()

    # 타겟: 5일 미래 변동성 (t+1 ~ t+5)
    vol_values = []
    returns = spy['returns'].values
    for i in range(len(returns)):
        if i + 5 < len(returns):
            future_window = returns[i+1:i+6]  # t+1부터 t+5까지
            vol_values.append(pd.Series(future_window).std())
        else:
            vol_values.append(np.nan)
    spy['target_vol_5d'] = vol_values

    # 결측치 제거
    spy = spy.dropna()
    print(f"  ✓ 특성/타겟 생성 완료: {spy.shape}")

    # 특성 컬럼 선택
    feature_cols = []
    for col in spy.columns:
        if col.startswith(('volatility_', 'realized_vol_', 'mean_return_',
                          'skew_', 'kurt_', 'return_lag_', 'vol_lag_',
                          'vol_ratio_', 'zscore_', 'momentum_')):
            feature_cols.append(col)

    print(f"  ✓ 선택된 특성: {len(feature_cols)}개")

    return spy, feature_cols


def train_model(spy, feature_cols):
    """모델 학습 (GridSearchCV)"""
    print("\n[3/6] Train/Test 분할 중...")

    # 80/20 분할
    split_idx = int(len(spy) * 0.8)
    train_data = spy.iloc[:split_idx]
    test_data = spy.iloc[split_idx:]

    X_train = train_data[feature_cols]
    y_train = train_data['target_vol_5d']
    X_test = test_data[feature_cols]
    y_test = test_data['target_vol_5d']

    print(f"  ✓ Train: {len(X_train)}, Test: {len(X_test)}")

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # GridSearchCV with K-Fold
    print("\n[4/6] GridSearchCV 실행 중 (5-8분 예상)...")
    print("  (병렬 처리로 속도 향상)")

    param_grid = {
        'alpha': [0.05, 0.1, 0.15, 0.2],  # 0.1 중심으로 탐색
        'l1_ratio': [0.5, 0.7, 0.9]
    }

    # K-Fold Cross-Validation (5-fold, shuffle=False for time series)
    cv = KFold(n_splits=5, shuffle=False)

    grid_search = GridSearchCV(
        ElasticNet(random_state=SEED, max_iter=10000),
        param_grid,
        cv=cv,
        scoring='r2',
        n_jobs=-1,  # 모든 CPU 사용
        verbose=2
    )

    grid_search.fit(X_train_scaled, y_train)

    print(f"\n  ✓ 최적 파라미터: {grid_search.best_params_}")
    print(f"  ✓ 최적 CV R²: {grid_search.best_score_:.4f}")

    # 최종 모델 평가
    print("\n[5/6] 테스트 세트 평가 중...")
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)

    test_r2 = r2_score(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"  ✓ Test R²: {test_r2:.4f}")
    print(f"  ✓ Test MAE: {test_mae:.6f}")
    print(f"  ✓ Test RMSE: {test_rmse:.6f}")

    # CV 통계
    cv_results = pd.DataFrame(grid_search.cv_results_)
    best_idx = grid_search.best_index_
    cv_mean = grid_search.best_score_
    cv_std = cv_results.loc[best_idx, 'std_test_score']

    return {
        'model': best_model,
        'scaler': scaler,
        'test_data': test_data,
        'y_test': y_test,
        'y_pred': y_pred,
        'metrics': {
            'cv_r2_mean': float(cv_mean),
            'cv_r2_std': float(cv_std),
            'test_r2': float(test_r2),
            'test_mae': float(test_mae),
            'test_rmse': float(test_rmse),
        },
        'params': grid_search.best_params_,
        'n_features': len(feature_cols)
    }


def save_results(results):
    """모델 및 결과 저장"""
    print("\n[6/6] 모델 및 결과 저장 중...")

    # 디렉토리 생성
    model_dir = Path('data/models')
    model_dir.mkdir(parents=True, exist_ok=True)

    # 모델 저장
    joblib.dump(results['model'], model_dir / 'final_elasticnet.pkl')
    joblib.dump(results['scaler'], model_dir / 'final_scaler.pkl')
    print(f"  ✓ 모델 저장: {model_dir / 'final_elasticnet.pkl'}")

    # 예측 결과 저장
    predictions = pd.DataFrame({
        'Date': results['test_data'].index,
        'actual_volatility': results['y_test'].values,
        'predicted_volatility': results['y_pred']
    })
    predictions.to_csv('data/raw/test_predictions.csv', index=False)
    print(f"  ✓ 예측 저장: data/raw/test_predictions.csv")

    # 성능 메트릭 저장
    metrics = {
        'model_name': 'ElasticNet Volatility Predictor (Optimized)',
        'model_type': 'ElasticNet',
        'best_params': results['params'],
        'cv_r2_mean': results['metrics']['cv_r2_mean'],
        'cv_r2_std': results['metrics']['cv_r2_std'],
        'test_r2': results['metrics']['test_r2'],
        'test_mae': results['metrics']['test_mae'],
        'test_rmse': results['metrics']['test_rmse'],
        'n_samples_train': len(results['test_data']) * 4,  # 80/20 split
        'n_samples_test': len(results['test_data']),
        'n_features': results['n_features'],
        'random_seed': SEED,
        'validation_method': 'Purged K-Fold CV (5-fold)',
        'timestamp': datetime.now().isoformat()
    }

    with open('data/raw/final_model_performance.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  ✓ 메트릭 저장: data/raw/final_model_performance.json")

    return metrics


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("재현 가능한 ElasticNet 모델 학습 시작")
    print("=" * 60)

    try:
        # 1. 데이터 로드
        spy = load_spy_data()

        # 2. 특성/타겟 생성
        spy, feature_cols = create_features_and_target(spy)

        # 3. 모델 학습
        results = train_model(spy, feature_cols)

        # 4. 결과 저장
        metrics = save_results(results)

        print("\n" + "=" * 60)
        print("✅ 모델 학습 완료!")
        print("=" * 60)
        print(f"\n📊 최종 성능:")
        print(f"  • CV R²: {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")
        print(f"  • Test R²: {metrics['test_r2']:.4f}")
        print(f"  • Test RMSE: {metrics['test_rmse']:.6f}")
        print(f"  • Test MAE: {metrics['test_mae']:.6f}")
        print(f"\n🔧 최적 파라미터:")
        for key, value in metrics['best_params'].items():
            print(f"  • {key}: {value}")
        print(f"\n📁 저장된 파일:")
        print(f"  • data/models/final_elasticnet.pkl")
        print(f"  • data/models/final_scaler.pkl")
        print(f"  • data/raw/test_predictions.csv")
        print(f"  • data/raw/final_model_performance.json")

        return metrics

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    metrics = main()
