#!/usr/bin/env python3
"""
고급 패턴 탐지: XGBoost + 특성 중요도 분석
목표: LLM 감성 지표에서 미세한 예측 패턴 발견
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score
import xgboost as xgb
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AdvancedPatternDetection:
    """XGBoost 기반 고급 패턴 탐지"""

    def __init__(self, dataset_path="data/training/advanced_news_twitter_dataset.csv"):
        self.dataset_path = dataset_path
        self.data = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.model = None
        self.results = {}

    def load_and_prepare_data(self):
        """데이터 로드 및 준비"""
        print("📂 고급 데이터셋 로드 중...")

        try:
            self.data = pd.read_csv(self.dataset_path, index_col=0, parse_dates=True)
            self.data = self.data.dropna()

            print(f"✅ 데이터: {self.data.shape}")
            print(f"   기간: {self.data.index.min()} ~ {self.data.index.max()}")

            # 고급 NLP 특성만 선택 (타겟 변수 제외)
            exclude_targets = ['target_return', 'target_direction', 'target_extreme']
            nlp_features = [col for col in self.data.columns if
                           ('sentiment' in col or 'virality' in col or 'news_count' in col or 'extreme' in col) and
                           not any(exc in col for exc in exclude_targets)]

            # 가격 특성 추가 (보조)
            price_features = ['returns', 'intraday_volatility', 'volume_surge']

            self.feature_names = nlp_features + [f for f in price_features if f in self.data.columns]

            print(f"\n📊 선택된 특성 ({len(self.feature_names)}개):")
            print(f"   NLP 특성: {len(nlp_features)}개")
            print(f"   가격 특성: {len([f for f in price_features if f in self.data.columns])}개")

            # X, y 분리
            self.X = self.data[self.feature_names]
            self.y = self.data['target_return_1d']

            return True

        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            return False

    def train_xgboost_model(self, target='return'):
        """XGBoost 모델 학습 (시계열 교차 검증)"""
        print(f"\n🌳 XGBoost 모델 학습 중 (타겟: {target})...")

        try:
            # 타겟 선택
            if target == 'return':
                y = self.data['target_return_1d']
                task = 'regression'
            elif target == 'direction':
                y = self.data['target_direction_1d']
                task = 'classification'
            elif target == 'extreme':
                # -1, 0, 1 → 0, 1, 2로 변환 (XGBoost 분류 요구사항)
                y = self.data['target_extreme_move'] + 1
                task = 'multiclass'
            else:
                raise ValueError(f"Unknown target: {target}")

            # 시계열 분할 (5-fold)
            tscv = TimeSeriesSplit(n_splits=5)

            cv_scores = []
            feature_importances = []

            for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(self.X), 1):
                X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                print(f"\n   Fold {fold_idx}/5:")
                print(f"     학습: {len(train_idx)} 샘플")
                print(f"     테스트: {len(test_idx)} 샘플")

                # XGBoost 설정
                if task == 'regression':
                    model = xgb.XGBRegressor(
                        n_estimators=200,
                        max_depth=5,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42 + fold_idx,
                        objective='reg:squarederror'
                    )
                elif task == 'multiclass':
                    model = xgb.XGBClassifier(
                        n_estimators=200,
                        max_depth=5,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42 + fold_idx,
                        objective='multi:softmax',
                        num_class=3
                    )
                else:
                    model = xgb.XGBClassifier(
                        n_estimators=200,
                        max_depth=5,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42 + fold_idx,
                        objective='binary:logistic'
                    )

                # 학습
                model.fit(X_train, y_train, verbose=False)

                # 예측
                y_pred = model.predict(X_test)

                # 성능 평가
                if task == 'regression':
                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)
                    print(f"     R²: {r2:.4f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}")
                    cv_scores.append({'r2': r2, 'rmse': rmse, 'mae': mae})
                else:
                    acc = accuracy_score(y_test, y_pred)
                    print(f"     Accuracy: {acc:.4f}")
                    cv_scores.append({'accuracy': acc})

                # 특성 중요도
                feature_importances.append(model.feature_importances_)

            # 평균 성능
            if task == 'regression':
                mean_r2 = np.mean([s['r2'] for s in cv_scores])
                std_r2 = np.std([s['r2'] for s in cv_scores])
                mean_rmse = np.mean([s['rmse'] for s in cv_scores])
                mean_mae = np.mean([s['mae'] for s in cv_scores])

                print(f"\n📊 TimeSeriesSplit CV 평균 성능:")
                print(f"   R² = {mean_r2:.4f} ± {std_r2:.4f}")
                print(f"   RMSE = {mean_rmse:.6f}")
                print(f"   MAE = {mean_mae:.6f}")

                self.results[f'{target}_regression'] = {
                    'mean_r2': float(mean_r2),
                    'std_r2': float(std_r2),
                    'mean_rmse': float(mean_rmse),
                    'mean_mae': float(mean_mae),
                    'cv_scores': cv_scores
                }
            else:
                mean_acc = np.mean([s['accuracy'] for s in cv_scores])
                std_acc = np.std([s['accuracy'] for s in cv_scores])

                print(f"\n📊 TimeSeriesSplit CV 평균 성능:")
                print(f"   Accuracy = {mean_acc:.4f} ± {std_acc:.4f}")

                self.results[f'{target}_classification'] = {
                    'mean_accuracy': float(mean_acc),
                    'std_accuracy': float(std_acc),
                    'cv_scores': cv_scores
                }

            # 평균 특성 중요도
            mean_importance = np.mean(feature_importances, axis=0)
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': mean_importance
            }).sort_values('importance', ascending=False)

            print(f"\n🔍 Top 10 중요 특성:")
            for idx, row in feature_importance_df.head(10).iterrows():
                print(f"   {row['feature']:35s}: {row['importance']:.4f}")

            self.results[f'{target}_feature_importance'] = feature_importance_df.to_dict('records')

            return True

        except Exception as e:
            print(f"❌ XGBoost 학습 실패: {e}")
            import traceback
            traceback.print_exc()
            return False

    def detect_microstructure_patterns(self):
        """시장 미시구조 패턴 탐지"""
        print(f"\n🔬 시장 미시구조 패턴 탐지 중...")

        try:
            # 패턴 1: 장 시작 전 감성 vs 당일 수익률
            premarket_corr = self.data[['sentiment_premarket', 'returns']].corr().iloc[0, 1]
            print(f"\n   패턴 1: 장 시작 전 감성 ↔ 당일 수익률")
            print(f"     상관계수: {premarket_corr:.4f}")

            # 패턴 2: 트위터 vs 뉴스 감성 차이
            twitter_news_diff = self.data['sentiment_twitter'] - self.data['sentiment_news']
            diff_return_corr = twitter_news_diff.corr(self.data['target_return_1d'])
            print(f"\n   패턴 2: 트위터-뉴스 감성 차이 ↔ 미래 수익률")
            print(f"     상관계수: {diff_return_corr:.4f}")

            # 패턴 3: 확산 속도 vs 극단 움직임
            virality_extreme = self.data[self.data['target_extreme_move'] != 0]['virality_max'].mean()
            virality_normal = self.data[self.data['target_extreme_move'] == 0]['virality_max'].mean()
            print(f"\n   패턴 3: 확산 속도 vs 극단 움직임")
            print(f"     극단 움직임 시 확산 속도: {virality_extreme:.2f}")
            print(f"     정상 시 확산 속도: {virality_normal:.2f}")
            print(f"     차이: {virality_extreme - virality_normal:.2f}")

            # 패턴 4: 감성 강도 vs 방향성 정확도
            high_strength = self.data[self.data['sentiment_strength_mean'] > 0.7]
            low_strength = self.data[self.data['sentiment_strength_mean'] <= 0.3]

            if len(high_strength) > 0:
                high_acc = ((high_strength['sentiment_mean'] > 0) == (high_strength['target_return_1d'] > 0)).mean()
                print(f"\n   패턴 4: 감성 강도 vs 방향성 정확도")
                print(f"     고강도 감성(>0.7) 방향성 정확도: {high_acc:.4f}")

            if len(low_strength) > 0:
                low_acc = ((low_strength['sentiment_mean'] > 0) == (low_strength['target_return_1d'] > 0)).mean()
                print(f"     저강도 감성(≤0.3) 방향성 정확도: {low_acc:.4f}")

            # 패턴 5: 극단 감성 비율 vs 반전 확률
            extreme_positive = self.data[self.data['extreme_positive_ratio'] > 0.3]
            if len(extreme_positive) > 0:
                reversal_prob = (extreme_positive['target_return_1d'] < 0).mean()
                print(f"\n   패턴 5: 극단 긍정 감성 vs 반전 확률")
                print(f"     극단 긍정(>30%) 후 하락 확률: {reversal_prob:.4f}")

            extreme_negative = self.data[self.data['extreme_negative_ratio'] > 0.3]
            if len(extreme_negative) > 0:
                bounce_prob = (extreme_negative['target_return_1d'] > 0).mean()
                print(f"     극단 부정(>30%) 후 상승 확률: {bounce_prob:.4f}")

            # 결과 저장
            self.results['microstructure_patterns'] = {
                'premarket_correlation': float(premarket_corr),
                'twitter_news_diff_correlation': float(diff_return_corr),
                'virality_extreme_vs_normal': {
                    'extreme': float(virality_extreme),
                    'normal': float(virality_normal),
                    'difference': float(virality_extreme - virality_normal)
                }
            }

            return True

        except Exception as e:
            print(f"❌ 패턴 탐지 실패: {e}")
            import traceback
            traceback.print_exc()
            return False

    def save_results(self):
        """결과 저장"""
        try:
            self.results['metadata'] = {
                'experiment': 'advanced_pattern_detection',
                'dataset': self.dataset_path,
                'n_samples': len(self.X),
                'n_features': len(self.feature_names),
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            output_path = "data/raw/advanced_pattern_detection_results.json"
            with open(output_path, 'w') as f:
                json.dump(self.results, f, indent=2)

            print(f"\n💾 결과 저장: {output_path}")
            return True

        except Exception as e:
            print(f"❌ 결과 저장 실패: {e}")
            return False

    def run_analysis(self):
        """전체 분석 실행"""
        print("="*60)
        print("🔬 고급 패턴 탐지 분석")
        print("="*60)
        print("목표: XGBoost로 LLM 감성 지표의 미세 패턴 발견\n")

        if not self.load_and_prepare_data():
            return False

        # 1. 수익률 예측 (회귀)
        if not self.train_xgboost_model(target='return'):
            return False

        # 2. 방향성 예측 (분류)
        if not self.train_xgboost_model(target='direction'):
            return False

        # 3. 극단 움직임 예측 (분류)
        if not self.train_xgboost_model(target='extreme'):
            return False

        # 4. 미시구조 패턴 탐지
        if not self.detect_microstructure_patterns():
            return False

        # 5. 결과 저장
        if not self.save_results():
            return False

        print("\n" + "="*60)
        print("✅ 분석 완료!")
        print("="*60)

        # 최종 요약
        print(f"\n📋 최종 결과 요약:")

        if 'return_regression' in self.results:
            r2 = self.results['return_regression']['mean_r2']
            print(f"   수익률 예측 (회귀): R² = {r2:.4f}")

        if 'direction_classification' in self.results:
            acc = self.results['direction_classification']['mean_accuracy']
            print(f"   방향성 예측 (분류): Accuracy = {acc:.4f}")

        if 'extreme_classification' in self.results:
            acc = self.results['extreme_classification']['mean_accuracy']
            print(f"   극단 움직임 예측: Accuracy = {acc:.4f}")

        print(f"\n💡 핵심 인사이트:")
        if 'return_regression' in self.results:
            r2 = self.results['return_regression']['mean_r2']
            if r2 > 0.1:
                print(f"   ✅ XGBoost로 패턴 발견 (R² = {r2:.4f})")
            elif r2 > 0.05:
                print(f"   ⚠️  약한 패턴 존재 (R² = {r2:.4f})")
            else:
                print(f"   ❌ 유의미한 패턴 없음 (R² = {r2:.4f})")

        return True

if __name__ == "__main__":
    detector = AdvancedPatternDetection(
        dataset_path="data/training/advanced_news_twitter_dataset.csv"
    )

    detector.run_analysis()
