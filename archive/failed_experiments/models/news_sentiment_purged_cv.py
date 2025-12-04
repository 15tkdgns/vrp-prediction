#!/usr/bin/env python3
"""
Purged K-Fold CV를 이용한 뉴스 감성 지표 주가 예측 검증
데이터 누출 방지 및 시간적 분리 보장
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from validation.purged_cross_validation import PurgedKFold
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class NewsSentimentPurgedCV:
    """Purged K-Fold CV 기반 검증"""

    def __init__(self, dataset_path="data/training/spy_news_sentiment_dataset.csv"):
        self.dataset_path = dataset_path
        self.data = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.cv_results = []
        self.results = {}

    def load_and_prepare_data(self):
        """데이터 로드 및 준비"""
        print("📂 데이터셋 로드 중...")

        try:
            self.data = pd.read_csv(self.dataset_path, index_col=0, parse_dates=True)
            self.data = self.data.dropna()

            print(f"✅ 데이터: {self.data.shape}")
            print(f"   기간: {self.data.index.min()} ~ {self.data.index.max()}")

            # 감성 특성
            sentiment_features = [
                'sentiment_mean', 'sentiment_std', 'sentiment_min', 'sentiment_max',
                'news_count', 'sentiment_range', 'sentiment_ma_5', 'sentiment_ma_20',
                'sentiment_momentum', 'news_volume_ma_10', 'news_volume_ratio'
            ]

            # 가격 특성
            price_features = ['returns', 'volatility_5d', 'volatility_20d']

            # 사용 가능한 특성
            available_sentiment = [f for f in sentiment_features if f in self.data.columns]
            available_price = [f for f in price_features if f in self.data.columns]

            self.feature_names = available_sentiment + available_price

            # X, y 분리
            self.X = self.data[self.feature_names]
            self.y = self.data['target_return_1d']

            print(f"   특성 수: {len(self.feature_names)}")
            print(f"   타겟: target_return_1d")

            return True

        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            return False

    def run_purged_kfold_cv(self, n_splits=5, pct_embargo=0.01):
        """Purged K-Fold 교차 검증 실행"""
        print(f"\n🔄 Purged K-Fold CV 실행 (n_splits={n_splits}, embargo={pct_embargo*100:.1f}%)...")

        try:
            # Purged K-Fold 설정
            cv = PurgedKFold(n_splits=n_splits, pct_embargo=pct_embargo)

            fold_idx = 1

            for train_indices, test_indices in cv.split(self.X, self.y):
                print(f"\n   Fold {fold_idx}/{n_splits}")

                # 훈련/테스트 데이터 (인덱스 기반 선택)
                X_train = self.X.loc[train_indices]
                X_test = self.X.loc[test_indices]
                y_train = self.y.loc[train_indices]
                y_test = self.y.loc[test_indices]

                print(f"     학습: {len(train_indices)} 샘플 ({X_train.index.min()} ~ {X_train.index.max()})")
                print(f"     테스트: {len(test_indices)} 샘플 ({X_test.index.min()} ~ {X_test.index.max()})")

                # 스케일링
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # 모델 학습
                model = Ridge(alpha=1.0, random_state=42)
                model.fit(X_train_scaled, y_train)

                # 예측
                y_pred = model.predict(X_test_scaled)

                # 성능 지표
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)

                print(f"     R²: {r2:.4f}, RMSE: {np.sqrt(mse):.6f}, MAE: {mae:.6f}")

                # 결과 저장
                self.cv_results.append({
                    'fold': fold_idx,
                    'train_size': len(train_indices),
                    'test_size': len(test_indices),
                    'train_start': str(X_train.index.min()),
                    'train_end': str(X_train.index.max()),
                    'test_start': str(X_test.index.min()),
                    'test_end': str(X_test.index.max()),
                    'r2_score': float(r2),
                    'mse': float(mse),
                    'rmse': float(np.sqrt(mse)),
                    'mae': float(mae)
                })

                fold_idx += 1

            # 평균 성능 계산
            mean_r2 = np.mean([r['r2_score'] for r in self.cv_results])
            std_r2 = np.std([r['r2_score'] for r in self.cv_results])
            mean_rmse = np.mean([r['rmse'] for r in self.cv_results])
            mean_mae = np.mean([r['mae'] for r in self.cv_results])

            print(f"\n📊 Purged K-Fold CV 평균 성능:")
            print(f"   R² = {mean_r2:.4f} ± {std_r2:.4f}")
            print(f"   RMSE = {mean_rmse:.6f}")
            print(f"   MAE = {mean_mae:.6f}")

            self.results['purged_cv'] = {
                'n_splits': n_splits,
                'pct_embargo': pct_embargo,
                'mean_r2': float(mean_r2),
                'std_r2': float(std_r2),
                'mean_rmse': float(mean_rmse),
                'mean_mae': float(mean_mae),
                'folds': self.cv_results
            }

            return True

        except Exception as e:
            print(f"❌ Purged K-Fold CV 실패: {e}")
            import traceback
            traceback.print_exc()
            return False

    def compare_with_standard_cv(self):
        """Standard KFold와 비교 (시간 순서 무시)"""
        print(f"\n🔍 Standard K-Fold CV와 비교 (시간 순서 무시)...")

        try:
            from sklearn.model_selection import KFold

            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            standard_cv_results = []

            for fold_idx, (train_indices, test_indices) in enumerate(cv.split(self.X), 1):
                X_train = self.X.iloc[train_indices]
                X_test = self.X.iloc[test_indices]
                y_train = self.y.iloc[train_indices]
                y_test = self.y.iloc[test_indices]

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                model = Ridge(alpha=1.0, random_state=42)
                model.fit(X_train_scaled, y_train)

                y_pred = model.predict(X_test_scaled)
                r2 = r2_score(y_test, y_pred)

                standard_cv_results.append(r2)

            mean_standard_r2 = np.mean(standard_cv_results)
            mean_purged_r2 = self.results['purged_cv']['mean_r2']

            print(f"   Standard K-Fold R²: {mean_standard_r2:.4f}")
            print(f"   Purged K-Fold R²: {mean_purged_r2:.4f}")
            print(f"   차이: {mean_standard_r2 - mean_purged_r2:.4f}")

            if mean_standard_r2 > mean_purged_r2 + 0.05:
                print(f"   ⚠️  Standard CV가 더 높음 → 시간적 누출 의심")
            else:
                print(f"   ✅ 두 방법 결과 유사 → 누출 없음")

            self.results['standard_cv'] = {
                'mean_r2': float(mean_standard_r2),
                'comparison': 'No significant leakage' if mean_standard_r2 <= mean_purged_r2 + 0.05 else 'Potential leakage'
            }

            return True

        except Exception as e:
            print(f"❌ Standard CV 비교 실패: {e}")
            return False

    def save_results(self):
        """결과 저장"""
        try:
            self.results['metadata'] = {
                'experiment': 'news_sentiment_purged_cv',
                'dataset': self.dataset_path,
                'n_samples': len(self.X),
                'n_features': len(self.feature_names),
                'features': self.feature_names,
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            output_path = "data/raw/news_sentiment_purged_cv_results.json"
            with open(output_path, 'w') as f:
                json.dump(self.results, f, indent=2)

            print(f"\n💾 결과 저장: {output_path}")
            return True

        except Exception as e:
            print(f"❌ 결과 저장 실패: {e}")
            return False

    def run_validation(self):
        """전체 검증 실행"""
        print("="*60)
        print("🔬 Purged K-Fold CV 검증 실험")
        print("="*60)
        print("목표: 데이터 누출 없는 엄격한 교차 검증\n")

        if not self.load_and_prepare_data():
            return False

        if not self.run_purged_kfold_cv(n_splits=5, pct_embargo=0.01):
            return False

        if not self.compare_with_standard_cv():
            return False

        if not self.save_results():
            return False

        print("\n" + "="*60)
        print("✅ 검증 완료!")
        print("="*60)

        # 최종 결론
        mean_r2 = self.results['purged_cv']['mean_r2']

        print(f"\n📋 최종 결론:")
        print(f"   Purged K-Fold CV R² = {mean_r2:.4f}")

        if abs(mean_r2) < 0.05:
            print(f"   ✅ LLM 감성 지표로 주가 예측 불가능 확인 (R² ≈ 0)")
        elif mean_r2 < 0.1:
            print(f"   ⚠️  매우 낮은 예측력 (R² < 0.1)")
        else:
            print(f"   ⚠️  예상보다 높은 R² - 추가 검증 필요")

        return True

if __name__ == "__main__":
    validator = NewsSentimentPurgedCV(
        dataset_path="data/training/spy_news_sentiment_dataset.csv"
    )

    validator.run_validation()
