#!/usr/bin/env python3
"""
Enhanced Time Aware Blending V9 vs V8 비교 테스트
목표: MDD < 0.6, Log Loss < 0.7
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import RobustScaler
import sys
sys.path.append('/root/workspace')

from src.models.enhanced_time_aware_blending_v9 import create_enhanced_time_aware_blending_v9
from src.models.enhanced_time_aware_blending_v10 import create_enhanced_time_aware_blending_v10

class EnhancedTimeAwareComparison:
    def __init__(self):
        self.data_path = Path("/root/workspace/data")
        self.results_path = Path("/root/workspace/results/analysis")
        self.results_path.mkdir(parents=True, exist_ok=True)

    def load_training_data(self):
        """훈련 데이터 로드"""
        try:
            # SPY 훈련 데이터 로드
            data_file = self.data_path / "training" / "sp500_2020_2024_enhanced.csv"
            if data_file.exists():
                df = pd.read_csv(data_file)
                print(f"✅ 훈련 데이터 로드: {len(df)} 행")
                return df
            else:
                print("❌ 훈련 데이터를 찾을 수 없습니다. 합성 데이터를 생성합니다.")
                return self.generate_synthetic_data()
        except Exception as e:
            print(f"데이터 로드 오류: {e}")
            return self.generate_synthetic_data()

    def generate_synthetic_data(self):
        """합성 데이터 생성 (실제 시장 특성 반영)"""
        np.random.seed(42)
        n_samples = 1000

        # 시장 패턴을 반영한 특징 생성
        dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')

        # 가격 데이터 (누적 수익률)
        returns = np.random.normal(0.0005, 0.015, n_samples)  # 연 12.6% 수익, 23.7% 변동성
        prices = 100 * np.cumprod(1 + returns)

        # 기술적 지표들
        df = pd.DataFrame({
            'date': dates,
            'close': prices,
            'volume': np.random.lognormal(15, 0.5, n_samples),
            'returns': returns
        })

        # 기술적 지표 계산
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['rsi'] = self.calculate_rsi(df['close'], 14)
        df['volatility'] = df['returns'].rolling(20).std()
        df['momentum'] = df['close'].pct_change(5)

        # 지연 특징들 (데이터 누출 방지)
        for lag in [1, 2, 3]:
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            df[f'volatility_lag_{lag}'] = df['volatility'].shift(lag)

        # 타겟: 다음날 수익률
        df['target'] = df['returns'].shift(-1)

        # 결측값 제거
        df = df.dropna()

        print(f"✅ 합성 데이터 생성: {len(df)} 행, {df.columns.tolist()}")
        return df

    def calculate_rsi(self, prices, window=14):
        """RSI 계산"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def prepare_features_and_targets(self, df):
        """특징과 타겟 준비"""
        # 실제 데이터 컬럼에 맞춰 특징 선택 (누출 방지)
        feature_columns = [
            'MA_20', 'MA_50', 'RSI', 'Volume', 'Volatility', 'ATR',
            'Returns_lag_1', 'Returns_lag_2', 'Returns_lag_3', 'Returns_lag_4', 'Returns_lag_5',
            'RSI_lag_1', 'RSI_lag_2', 'RSI_lag_3', 'RSI_lag_4', 'RSI_lag_5',
            'Volatility_lag_1', 'Volatility_lag_2', 'Volatility_lag_3', 'Volatility_lag_4', 'Volatility_lag_5',
            'BB_position', 'volume_ratio', 'vol_regime', 'trend_strength'
        ]

        # 존재하는 컬럼만 선택
        available_features = [col for col in feature_columns if col in df.columns]
        print(f"🔧 전체 컬럼: {len(df.columns)}")
        print(f"🔧 요청된 특징: {len(feature_columns)}")
        print(f"🔧 사용 가능한 특징: {len(available_features)}")

        if len(available_features) == 0:
            print("❌ 사용 가능한 특징이 없습니다. 기본 특징을 사용합니다.")
            # 기본 특징들 선택
            basic_features = ['Close', 'Volume', 'Returns']
            available_features = [col for col in basic_features if col in df.columns]
            if 'Returns' in available_features:
                available_features.remove('Returns')  # 타겟에서 제외

        X = df[available_features].values

        # 타겟: 현재 Returns 사용 (미래 예측이 아닌 현재 움직임 예측)
        if 'Returns' in df.columns:
            y = df['Returns'].values
        else:
            # Returns가 없으면 Close 가격 변화율로 계산
            y = df['Close'].pct_change().values

        # 결측값 처리
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[mask]
        y = y[mask]

        print(f"✅ 특징 준비: {X.shape}, 타겟: {y.shape}")
        print(f"🔧 사용된 특징: {available_features}")

        return X, y, available_features

    def calculate_advanced_metrics(self, y_true, y_pred, y_pred_proba=None):
        """고급 성능 지표 계산"""
        metrics = {}

        # 기본 회귀 지표
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])

        # 방향 정확도
        direction_true = (y_true > 0).astype(int)
        direction_pred = (y_pred > 0).astype(int)
        metrics['direction_accuracy'] = np.mean(direction_true == direction_pred) * 100

        # Log Loss (확률 예측이 있는 경우)
        if y_pred_proba is not None:
            try:
                metrics['log_loss'] = log_loss(direction_true, y_pred_proba)
            except:
                metrics['log_loss'] = None
        else:
            metrics['log_loss'] = None

        # MDD 계산
        cumulative_returns = np.cumprod(1 + y_pred)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        metrics['max_drawdown'] = abs(np.min(drawdown))

        # 샤프 비율
        if np.std(y_pred) > 0:
            metrics['sharpe_ratio'] = np.mean(y_pred) / np.std(y_pred) * np.sqrt(252)
        else:
            metrics['sharpe_ratio'] = 0

        # 소르티노 비율
        negative_returns = y_pred[y_pred < 0]
        if len(negative_returns) > 0:
            downside_std = np.std(negative_returns) * np.sqrt(252)
            if downside_std > 0:
                metrics['sortino_ratio'] = np.mean(y_pred) * 252 / downside_std
            else:
                metrics['sortino_ratio'] = 0
        else:
            metrics['sortino_ratio'] = float('inf')

        return metrics

    def run_time_series_validation(self, X, y, n_splits=5):
        """시계열 교차 검증"""
        print(f"🔍 시계열 교차 검증 시작 (분할: {n_splits})")

        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_results = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            print(f"  📊 Fold {fold + 1}/{n_splits} 처리 중...")

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # V9 모델 훈련
            model_v9 = create_enhanced_time_aware_blending_v9()
            model_v9.fit(X_train, y_train)

            # 예측 (이력 정보 포함)
            y_pred_v9 = model_v9.predict(X_test, y_history=y_train)
            y_pred_proba_v9 = model_v9.predict_proba(X_test, y_history=y_train)

            # 성능 계산
            metrics_v9 = self.calculate_advanced_metrics(y_test, y_pred_v9, y_pred_proba_v9)
            metrics_v9['model'] = 'enhanced_time_aware_v9'
            metrics_v9['fold'] = fold + 1

            # 모델 상태 정보
            summary_v9 = model_v9.get_performance_summary()
            metrics_v9.update(summary_v9)

            fold_results.append(metrics_v9)

            print(f"    ✅ V9 - MAE: {metrics_v9['mae']:.4f}, "
                  f"MDD: {metrics_v9['max_drawdown']:.4f}, "
                  f"Log Loss: {metrics_v9['log_loss']:.4f}")

            # V10 모델 훈련
            model_v10 = create_enhanced_time_aware_blending_v10()
            model_v10.fit(X_train, y_train)

            # 예측 (이력 정보 포함)
            y_pred_v10 = model_v10.predict(X_test, y_history=y_train)
            y_pred_proba_v10 = model_v10.predict_proba(X_test, y_history=y_train)

            # 성능 계산
            metrics_v10 = self.calculate_advanced_metrics(y_test, y_pred_v10, y_pred_proba_v10)
            metrics_v10['model'] = 'enhanced_time_aware_v10'
            metrics_v10['fold'] = fold + 1

            # 모델 상태 정보
            summary_v10 = model_v10.get_performance_summary()
            metrics_v10.update(summary_v10)

            fold_results.append(metrics_v10)

            print(f"    ✅ V10 - MAE: {metrics_v10['mae']:.4f}, "
                  f"MDD: {metrics_v10['max_drawdown']:.4f}, "
                  f"Log Loss: {metrics_v10['log_loss']:.4f}")

        return fold_results

    def analyze_results(self, fold_results):
        """결과 분석"""
        print("\n📊 결과 분석 중...")

        df_results = pd.DataFrame(fold_results)

        # 평균 성능 계산
        avg_metrics = {}
        for metric in ['mae', 'max_drawdown', 'log_loss', 'direction_accuracy', 'sharpe_ratio', 'sortino_ratio']:
            if metric in df_results.columns:
                values = df_results[metric].dropna()
                if len(values) > 0:
                    avg_metrics[f'{metric}_mean'] = np.mean(values)
                    avg_metrics[f'{metric}_std'] = np.std(values)
                    avg_metrics[f'{metric}_min'] = np.min(values)
                    avg_metrics[f'{metric}_max'] = np.max(values)

        return avg_metrics, df_results

    def save_results(self, avg_metrics, fold_results, comparison_df):
        """결과 저장"""
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')

        # JSON 결과 저장
        results = {
            'timestamp': timestamp,
            'model_name': 'enhanced_time_aware_blending_v9',
            'optimization_target': 'MDD_minimization_log_loss_reduction',
            'average_metrics': avg_metrics,
            'fold_results': fold_results,
            'goal_achievement': {
                'mdd_target': 0.6,
                'mdd_achieved': avg_metrics.get('max_drawdown_mean', 1.0),
                'mdd_success': avg_metrics.get('max_drawdown_mean', 1.0) < 0.6,
                'log_loss_target': 0.7,
                'log_loss_achieved': avg_metrics.get('log_loss_mean', 1.0),
                'log_loss_success': avg_metrics.get('log_loss_mean', 1.0) < 0.7
            }
        }

        # 파일 저장
        results_file = self.results_path / f"enhanced_time_aware_v9_results_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"📊 결과 저장: {results_file}")

        # CSV 저장
        csv_file = self.results_path / f"enhanced_time_aware_v9_comparison_{timestamp}.csv"
        comparison_df.to_csv(csv_file, index=False)
        print(f"📊 비교 결과 저장: {csv_file}")

        return results

    def run_comprehensive_test(self):
        """종합 테스트 실행"""
        print("🚀 Enhanced Time Aware Blending V9 종합 테스트 시작\n")
        print("🎯 목표: MDD < 0.6, Log Loss < 0.7\n")

        # 데이터 로드
        df = self.load_training_data()
        X, y, features = self.prepare_features_and_targets(df)

        print(f"📊 데이터 준비 완료: {X.shape[0]} 샘플, {X.shape[1]} 특징")

        # 교차 검증 실행
        fold_results = self.run_time_series_validation(X, y, n_splits=5)

        # 결과 분석
        avg_metrics, comparison_df = self.analyze_results(fold_results)

        # 결과 저장
        final_results = self.save_results(avg_metrics, fold_results, comparison_df)

        # 결과 출력
        self.print_final_results(avg_metrics, final_results)

        return final_results

    def print_final_results(self, avg_metrics, results):
        """최종 결과 출력"""
        print("\n" + "="*80)
        print("🏆 Enhanced Time Aware Blending V9 vs V10 최종 결과")
        print("="*80)

        # V9와 V10 결과 분리
        df_results = pd.DataFrame(results['fold_results'])
        v9_results = df_results[df_results['model'] == 'enhanced_time_aware_v9']
        v10_results = df_results[df_results['model'] == 'enhanced_time_aware_v10']

        print(f"\n🎯 모델별 성능 비교:")

        if len(v9_results) > 0:
            v9_log_loss = v9_results['log_loss'].mean()
            v9_mdd = v9_results['max_drawdown'].mean()
            v9_mae = v9_results['mae'].mean()

            print(f"\n📊 V9 성능:")
            print(f"  Log Loss: {v9_log_loss:.4f} ({'✅' if v9_log_loss < 0.7 else '❌'} < 0.7)")
            print(f"  MDD: {v9_mdd:.4f} ({'✅' if v9_mdd < 0.6 else '❌'} < 0.6)")
            print(f"  MAE: {v9_mae:.4f}")
            print(f"  방향 정확도: {v9_results['direction_accuracy'].mean():.2f}%")

        if len(v10_results) > 0:
            v10_log_loss = v10_results['log_loss'].mean()
            v10_mdd = v10_results['max_drawdown'].mean()
            v10_mae = v10_results['mae'].mean()

            print(f"\n📊 V10 성능:")
            print(f"  Log Loss: {v10_log_loss:.4f} ({'✅' if v10_log_loss < 0.7 else '❌'} < 0.7)")
            print(f"  MDD: {v10_mdd:.4f} ({'✅' if v10_mdd < 0.6 else '❌'} < 0.6)")
            print(f"  MAE: {v10_mae:.4f}")
            print(f"  방향 정확도: {v10_results['direction_accuracy'].mean():.2f}%")

        # 승자 결정
        if len(v9_results) > 0 and len(v10_results) > 0:
            v9_score = (1 if v9_log_loss < 0.7 else 0) + (1 if v9_mdd < 0.6 else 0)
            v10_score = (1 if v10_log_loss < 0.7 else 0) + (1 if v10_mdd < 0.6 else 0)

            print(f"\n🏆 목표 달성 점수:")
            print(f"  V9: {v9_score}/2 (Log Loss: {'✅' if v9_log_loss < 0.7 else '❌'}, MDD: {'✅' if v9_mdd < 0.6 else '❌'})")
            print(f"  V10: {v10_score}/2 (Log Loss: {'✅' if v10_log_loss < 0.7 else '❌'}, MDD: {'✅' if v10_mdd < 0.6 else '❌'})")

            if v10_score > v9_score:
                print(f"\n🎉 승자: Enhanced Time Aware Blending V10!")
            elif v9_score > v10_score:
                print(f"\n🎉 승자: Enhanced Time Aware Blending V9!")
            else:
                if v10_log_loss < v9_log_loss:
                    print(f"\n🎉 승자: Enhanced Time Aware Blending V10 (더 낮은 Log Loss)")
                else:
                    print(f"\n🎉 승자: Enhanced Time Aware Blending V9 (더 낮은 Log Loss)")

        print(f"\n📊 종합 평가:")
        goal_achievement = results['goal_achievement']
        print(f"  MDD 목표 < 0.6: {goal_achievement['mdd_achieved']:.4f} {'✅' if goal_achievement['mdd_success'] else '❌'}")
        print(f"  Log Loss 목표 < 0.7: {goal_achievement['log_loss_achieved']:.4f} {'✅' if goal_achievement['log_loss_success'] else '❌'}")

if __name__ == "__main__":
    tester = EnhancedTimeAwareComparison()
    results = tester.run_comprehensive_test()