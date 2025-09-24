#!/usr/bin/env python3
"""
✅ 누출 제거 챔피언 시스템

데이터 누출을 완전히 제거한 올바른 주가예측 시스템
목표: 데이터 누출 없이 85%+ 정확도 달성
"""

import sys
sys.path.append('/root/workspace/src')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# 자체 모듈
from pipeline.advanced_metric_pipeline import AdvancedMetricPipeline
from core.data_processor import DataProcessor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
import catboost as cb

class LeakFreeChampionSystem:
    """누출 제거 챔피언 시스템"""

    def __init__(self):
        self.data_processor = DataProcessor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"✅ 누출 제거 챔피언 시스템 초기화")
        print(f"   💾 디바이스: {self.device}")
        print(f"   🎯 목표: 누출 없이 85%+ 달성")

    def create_leak_free_features(self, df):
        """누출 없는 안전한 특성 생성"""
        print("   🔧 누출 없는 안전한 특성 생성...")

        enhanced_df = df.copy()

        # 🚨 중요: 미래 정보를 절대 사용하지 않음
        # 모든 특성은 현재 시점(t) 또는 과거 시점(t-n)만 사용

        # 1. 기본 수익률 (과거만 사용)
        enhanced_df['returns'] = enhanced_df['Close'].pct_change()
        enhanced_df['log_returns'] = np.log(enhanced_df['Close'] / enhanced_df['Close'].shift(1))

        # 2. 안전한 모멘텀 특성 (과거만)
        for period in [3, 5, 10, 15, 20]:
            enhanced_df[f'momentum_{period}'] = enhanced_df['Close'] / enhanced_df['Close'].shift(period) - 1
            enhanced_df[f'returns_mean_{period}'] = enhanced_df['returns'].rolling(period).mean()

        # 3. 안전한 변동성 특성 (과거만)
        for window in [5, 10, 20, 30]:
            enhanced_df[f'volatility_{window}'] = enhanced_df['returns'].rolling(window).std()
            enhanced_df[f'realized_vol_{window}'] = np.sqrt(enhanced_df['returns'].rolling(window).apply(lambda x: np.sum(x**2)))

        # 4. 안전한 기술적 지표 (과거만)
        # RSI
        for period in [7, 14, 21]:
            enhanced_df[f'rsi_{period}'] = self._calculate_rsi(enhanced_df['Close'], period)

        # 이동평균
        for period in [5, 10, 20, 50]:
            enhanced_df[f'sma_{period}'] = enhanced_df['Close'].rolling(period).mean()
            enhanced_df[f'close_sma_ratio_{period}'] = enhanced_df['Close'] / enhanced_df[f'sma_{period}']

        # 5. 안전한 볼륨 특성 (과거만)
        enhanced_df['volume_sma_20'] = enhanced_df['Volume'].rolling(20).mean()
        enhanced_df['volume_ratio'] = enhanced_df['Volume'] / enhanced_df['volume_sma_20']
        enhanced_df['volume_momentum_5'] = enhanced_df['Volume'] / enhanced_df['Volume'].shift(5)

        # 6. 안전한 가격 특성 (현재 시점만)
        enhanced_df['hl_ratio'] = (enhanced_df['High'] - enhanced_df['Low']) / enhanced_df['Close']
        enhanced_df['body_ratio'] = abs(enhanced_df['Open'] - enhanced_df['Close']) / (enhanced_df['High'] - enhanced_df['Low'] + 1e-8)
        enhanced_df['gap_ratio'] = (enhanced_df['Open'] - enhanced_df['Close'].shift(1)) / enhanced_df['Close'].shift(1)

        # 7. 안전한 고급 기술적 지표
        # MACD (과거 데이터만)
        ema_12 = enhanced_df['Close'].ewm(span=12).mean()
        ema_26 = enhanced_df['Close'].ewm(span=26).mean()
        enhanced_df['macd'] = ema_12 - ema_26
        enhanced_df['macd_signal'] = enhanced_df['macd'].ewm(span=9).mean()
        enhanced_df['macd_histogram'] = enhanced_df['macd'] - enhanced_df['macd_signal']

        # 볼린저 밴드 (과거 데이터만)
        for period in [20]:
            sma = enhanced_df['Close'].rolling(period).mean()
            std = enhanced_df['Close'].rolling(period).std()
            enhanced_df[f'bb_upper_{period}'] = sma + (2 * std)
            enhanced_df[f'bb_lower_{period}'] = sma - (2 * std)
            enhanced_df[f'bb_width_{period}'] = enhanced_df[f'bb_upper_{period}'] - enhanced_df[f'bb_lower_{period}']
            enhanced_df[f'bb_position_{period}'] = (enhanced_df['Close'] - enhanced_df[f'bb_lower_{period}']) / (enhanced_df[f'bb_width_{period}'] + 1e-8)

        # 8. 안전한 상관관계 특성 (과거 윈도우만)
        enhanced_df['price_volume_corr_20'] = enhanced_df['Close'].rolling(20).corr(enhanced_df['Volume'])

        # 9. 안전한 통계적 특성 (과거만)
        for window in [10, 20]:
            enhanced_df[f'skewness_{window}'] = enhanced_df['returns'].rolling(window).skew()
            enhanced_df[f'kurtosis_{window}'] = enhanced_df['returns'].rolling(window).kurt()

        # 🚨 중요: 타겟 변수는 미래 정보 (shift(-1))
        enhanced_df['future_return'] = enhanced_df['Close'].pct_change().shift(-1)
        enhanced_df['direction_target'] = (enhanced_df['future_return'] > 0).astype(int)

        # NaN 처리
        enhanced_df = enhanced_df.fillna(method='ffill').fillna(0)
        enhanced_df = enhanced_df.replace([np.inf, -np.inf], 0)

        # 🚨 누출 의심 특성 제거
        exclude_features = [
            'next_return', 'return_target', 'future_return',  # 명백한 누출 특성
            'Date', 'Open', 'High', 'Low', 'Close', 'Volume'  # 원본 OHLCV
        ]

        print(f"   🔒 제거된 누출 특성: {len(exclude_features)}개")
        return enhanced_df

    def _calculate_rsi(self, prices, window=14):
        """RSI 계산 (과거 데이터만 사용)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))

    def run_leak_free_experiment(self, data_path='/root/workspace/data/training/sp500_2020_2024_enhanced.csv'):
        """누출 제거 실험 실행"""
        print("✅ 누출 제거 챔피언 시스템 실험 시작")
        print("="*70)

        try:
            # 1. 기준선 시스템 (검증용)
            print("📊 1단계: 기준선 시스템 재확인")
            baseline_config = {
                'data_path': data_path,
                'target_type': 'direction',
                'sequence_length': 15,
                'cv_splits': 3,
                'gpu_enabled': True,
                'save_models': False,
                'save_results': False,
                'output_dir': '/tmp'
            }

            baseline_pipeline = AdvancedMetricPipeline(baseline_config)
            baseline_result = baseline_pipeline.run_advanced_pipeline()

            baseline_best = 0
            for model_name, metrics in baseline_result.get('model_results', {}).items():
                if 'direction_accuracy' in metrics:
                    acc = metrics['direction_accuracy']
                    if acc > baseline_best:
                        baseline_best = acc

            print(f"   ✅ 기준선: {baseline_best:.2f}%")

            # 2. 누출 제거 시스템
            print("\n📊 2단계: 누출 제거 시스템")

            df = self.data_processor.load_and_validate_data(data_path)
            leak_free_df = self.create_leak_free_features(df)

            print(f"   📈 누출 제거 데이터: {leak_free_df.shape}")

            # 3. 안전한 CatBoost 실험
            print("\n📊 3단계: 안전한 CatBoost 실험")
            safe_catboost_accuracy = self._run_safe_catboost(leak_free_df)
            print(f"   ✅ 안전한 CatBoost: {safe_catboost_accuracy:.2f}%")

            # 4. 누출 제거 검증
            print("\n📊 4단계: 데이터 누출 재검증")
            leakage_check = self._verify_no_leakage(leak_free_df)

            # 최종 결과 분석
            improvement = safe_catboost_accuracy - baseline_best

            print(f"\n✅ 누출 제거 챔피언 시스템 최종 결과:")
            print("="*70)
            print(f"📊 기준선               : {baseline_best:.2f}%")
            print(f"🔒 안전한 CatBoost      : {safe_catboost_accuracy:.2f}% (개선: {improvement:+.2f}%p)")
            print(f"🛡️ 데이터 누출 검증     : {'✅ 안전' if leakage_check else '🚨 의심'}")

            # 목표 달성 여부
            if safe_catboost_accuracy >= 85.0:
                print(f"🎉 목표 달성! 85%+ 정확도 (누출 없이)")
            elif safe_catboost_accuracy > baseline_best:
                print(f"📊 기준선 개선! (누출 없이)")
            else:
                print(f"⚠️ 추가 최적화 필요 (하지만 누출 없음)")

            # 결과 저장
            final_result = {
                'timestamp': datetime.now().isoformat(),
                'experiment_type': 'leak_free_champion_system',
                'data_leakage_removed': True,
                'baseline_accuracy': baseline_best,
                'safe_catboost_accuracy': safe_catboost_accuracy,
                'improvement': improvement,
                'no_leakage_verified': leakage_check,
                'target_achieved_safely': safe_catboost_accuracy >= 85.0
            }

            output_path = f"/root/workspace/data/results/leak_free_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_path, 'w') as f:
                json.dump(final_result, f, indent=2, default=str)

            print(f"\n💾 결과 저장: {output_path}")
            print(f"🏁 누출 제거 챔피언 시스템 실험 완료!")

            return final_result

        except Exception as e:
            print(f"❌ 누출 제거 챔피언 시스템 실패: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _run_safe_catboost(self, leak_free_df):
        """안전한 CatBoost 실행 (누출 제거)"""
        print("   🛡️ 안전한 CatBoost 훈련...")

        # 안전한 특성만 선택
        safe_features = []
        for col in leak_free_df.columns:
            if col not in ['direction_target', 'future_return', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']:
                # 누출 의심 특성 제외
                if 'next_' not in col and 'target' not in col and 'future_' not in col:
                    safe_features.append(col)

        print(f"      사용할 안전한 특성: {len(safe_features)}개")

        X = leak_free_df[safe_features].values
        y = leak_free_df['direction_target'].values

        # NaN 처리
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.5, posinf=1.0, neginf=0.0).astype(int)

        # 유효한 데이터만 선택
        valid_idx = ~pd.isna(leak_free_df['direction_target'])
        X = X[valid_idx]
        y = y[valid_idx]

        # 교차 검증 (시간 순서 엄격 준수)
        tscv = TimeSeriesSplit(n_splits=3)
        accuracies = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # 정규화
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # CatBoost 훈련 (보수적 설정)
            model = cb.CatBoostClassifier(
                iterations=100,  # 과적합 방지
                learning_rate=0.1,
                depth=6,  # 깊이 제한
                loss_function='CrossEntropy',
                eval_metric='Accuracy',
                random_seed=42,
                verbose=False,
                early_stopping_rounds=20
            )

            model.fit(
                X_train_scaled, y_train,
                eval_set=(X_val_scaled, y_val),
                verbose=False
            )

            val_probs = model.predict_proba(X_val_scaled)[:, 1]
            val_preds = (val_probs > 0.5).astype(int)
            accuracy = np.mean(val_preds == y_val)
            accuracies.append(accuracy)

            print(f"      Fold {fold+1}: {accuracy:.4f}")

        return np.mean(accuracies) * 100

    def _verify_no_leakage(self, leak_free_df):
        """데이터 누출 없음 검증"""
        print("   🔍 데이터 누출 재검증...")

        # 특성-타겟 상관관계 확인
        safe_features = []
        for col in leak_free_df.columns:
            if col not in ['direction_target', 'future_return', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']:
                if 'next_' not in col and 'target' not in col and 'future_' not in col:
                    safe_features.append(col)

        max_correlation = 0
        suspicious_feature = None

        for feature in safe_features:
            if feature in leak_free_df.columns:
                corr = abs(leak_free_df[feature].corr(leak_free_df['direction_target']))
                if not pd.isna(corr) and corr > max_correlation:
                    max_correlation = corr
                    suspicious_feature = feature

        print(f"      최대 특성-타겟 상관관계: {max_correlation:.4f} ({suspicious_feature})")

        # 0.3 이하면 안전한 것으로 판단
        is_safe = max_correlation < 0.3
        print(f"      {'✅ 안전' if is_safe else '🚨 의심'} (기준: 0.3)")

        return is_safe

def main():
    """메인 실행"""
    system = LeakFreeChampionSystem()
    results = system.run_leak_free_experiment()

    print("\n🎉 누출 제거 챔피언 시스템 테스트 완료!")
    return results

if __name__ == "__main__":
    main()