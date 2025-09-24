"""
V2 Ridge 모델 과적합 종합 검증 시스템
V2 모델 (alpha=19.5029, R²=0.3256)의 과적합 여부를 다각도로 분석

과적합 검증 항목:
1. 훈련 vs 검증 성능 비교
2. 학습 곡선 분석 (Learning Curve)
3. 교차검증 일관성 분석
4. Walk-Forward Validation
5. 잔차 분석 (Residual Analysis)
6. 특성 중요도 안정성
7. 정규화 효과 검증
8. 시간적 안정성 분석
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import learning_curve, validation_curve
import logging

warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/root/workspace/data/raw/v2_overfitting_check.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PurgedKFoldSklearn:
    """sklearn 호환 Purged K-Fold Cross-Validation"""

    def __init__(self, n_splits=5, purge_length=5, embargo_length=5):
        self.n_splits = n_splits
        self.purge_length = purge_length
        self.embargo_length = embargo_length

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        indices = np.arange(n_samples)
        test_size = n_samples // self.n_splits
        splits = []

        for i in range(self.n_splits):
            test_start = i * test_size
            test_end = min((i + 1) * test_size, n_samples)
            test_indices = indices[test_start:test_end]

            purge_start = test_end
            purge_end = min(test_end + self.purge_length, n_samples)
            embargo_end = min(purge_end + self.embargo_length, n_samples)

            train_indices = np.concatenate([
                indices[:test_start],
                indices[embargo_end:]
            ])

            if len(train_indices) > 0 and len(test_indices) > 0:
                splits.append((train_indices, test_indices))

        return splits

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

class V2OverfittingChecker:
    """V2 모델 과적합 종합 검증기"""

    def __init__(self):
        self.best_alpha = 19.5029  # V2 최적값
        self.cv = PurgedKFoldSklearn()
        self.scaler = StandardScaler()
        self.results = {}

    def load_spy_data(self):
        """SPY 데이터 로드"""
        logger.info("📊 SPY 데이터 로딩...")

        spy = yf.Ticker("SPY")
        data = spy.history(start="2015-01-01", end="2024-12-31")
        data['returns'] = np.log(data['Close'] / data['Close'].shift(1))
        data = data.dropna()

        logger.info(f"✅ 데이터 로딩 완료: {len(data)}개 관측치")
        return data

    def create_v2_features(self, data):
        """V2와 동일한 특성 생성"""
        logger.info("🔧 V2 특성 엔지니어링...")

        returns = data['returns']
        prices = data['Close']
        features = pd.DataFrame(index=data.index)

        # 1. 변동성 특성 (6개)
        for window in [3, 5, 10, 15, 20, 30]:
            features[f'vol_{window}'] = returns.rolling(window).std()

        # 2. 통계적 모멘트 (6개)
        for window in [5, 10, 20]:
            features[f'skew_{window}'] = returns.rolling(window).skew()
            features[f'kurt_{window}'] = returns.rolling(window).kurt()

        # 3. 래그 특성 (6개)
        for lag in [1, 2, 3]:
            features[f'return_lag_{lag}'] = returns.shift(lag)
            features[f'vol_lag_{lag}'] = features['vol_5'].shift(lag)

        # 4. 변동성 체제 (4개)
        short_vol = features['vol_5']
        medium_vol = features['vol_20']
        long_vol = features['vol_30']

        features['vol_regime_short'] = (short_vol > medium_vol).astype(float)
        features['vol_regime_medium'] = (medium_vol > long_vol).astype(float)
        features['vol_expansion'] = short_vol / (long_vol + 1e-8)
        features['vol_contraction'] = long_vol / (short_vol + 1e-8)

        # 5. 통계적 지표 (5개)
        for window in [10, 20, 30]:
            ma = returns.rolling(window).mean()
            std = returns.rolling(window).std()
            features[f'zscore_{window}'] = (returns - ma) / (std + 1e-8)

        for window in [10, 20]:
            ma = returns.rolling(window).mean()
            std = returns.rolling(window).std()
            features[f'sharpe_{window}'] = (ma * np.sqrt(252)) / (std + 1e-8)

        # 6. 상호작용 특성 (3개)
        features['vol_5_20_ratio'] = features['vol_5'] / (features['vol_20'] + 1e-8)
        features['vol_10_30_ratio'] = features['vol_10'] / (features['vol_30'] + 1e-8)
        features['vol_price_interaction'] = features['vol_20'] * returns

        # 타겟: 5일 후 변동성
        target = []
        for i in range(len(returns)):
            if i + 5 < len(returns):
                future_vol = returns.iloc[i+1:i+6].std()
                target.append(future_vol)
            else:
                target.append(np.nan)

        features['target_vol_5d'] = target
        features = features.dropna()

        X = features.drop('target_vol_5d', axis=1)
        y = features['target_vol_5d']

        logger.info(f"✅ V2 특성 생성 완료: {X.shape[1]}개 특성, {len(X)}개 샘플")
        return X, y

    def test_1_train_vs_validation(self):
        """1. 훈련 vs 검증 성능 비교"""
        logger.info("🔍 1. 훈련 vs 검증 성능 비교...")

        data = self.load_spy_data()
        X, y = self.create_v2_features(data)
        X_scaled = self.scaler.fit_transform(X)

        model = Ridge(alpha=self.best_alpha)
        splits = list(self.cv.split(X_scaled))

        train_scores = []
        val_scores = []

        for train_idx, val_idx in splits:
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # 모델 훈련
            model.fit(X_train, y_train)

            # 훈련 성능
            train_pred = model.predict(X_train)
            train_r2 = r2_score(y_train, train_pred)
            train_scores.append(train_r2)

            # 검증 성능
            val_pred = model.predict(X_val)
            val_r2 = r2_score(y_val, val_pred)
            val_scores.append(val_r2)

        train_mean = np.mean(train_scores)
        val_mean = np.mean(val_scores)
        performance_gap = train_mean - val_mean

        result = {
            'train_r2_mean': float(train_mean),
            'train_r2_std': float(np.std(train_scores)),
            'val_r2_mean': float(val_mean),
            'val_r2_std': float(np.std(val_scores)),
            'performance_gap': float(performance_gap),
            'overfitting_indicator': bool(performance_gap > 0.05),  # 5% 이상 차이면 과적합 의심
            'train_scores': [float(s) for s in train_scores],
            'val_scores': [float(s) for s in val_scores]
        }

        logger.info(f"   훈련 R²: {train_mean:.4f} ± {np.std(train_scores):.4f}")
        logger.info(f"   검증 R²: {val_mean:.4f} ± {np.std(val_scores):.4f}")
        logger.info(f"   성능 차이: {performance_gap:.4f}")
        logger.info(f"   과적합 의심: {'YES' if result['overfitting_indicator'] else 'NO'}")

        self.results['train_vs_validation'] = result
        return result

    def test_2_learning_curve_analysis(self):
        """2. 학습 곡선 분석"""
        logger.info("🔍 2. 학습 곡선 분석...")

        data = self.load_spy_data()
        X, y = self.create_v2_features(data)
        X_scaled = self.scaler.fit_transform(X)

        model = Ridge(alpha=self.best_alpha)

        # 다양한 훈련 크기로 학습 곡선 생성
        train_sizes = np.linspace(0.3, 1.0, 8)
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X_scaled, y,
            cv=self.cv,
            train_sizes=train_sizes,
            scoring='r2',
            n_jobs=-1
        )

        # 평균과 표준편차 계산
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)

        # 수렴 분석
        final_gap = train_mean[-1] - val_mean[-1]
        convergence_trend = np.corrcoef(train_sizes_abs, val_mean)[0, 1]  # 검증 성능과 데이터 크기 상관관계

        result = {
            'train_sizes': [int(s) for s in train_sizes_abs],
            'train_scores_mean': [float(s) for s in train_mean],
            'train_scores_std': [float(s) for s in train_std],
            'val_scores_mean': [float(s) for s in val_mean],
            'val_scores_std': [float(s) for s in val_std],
            'final_performance_gap': float(final_gap),
            'convergence_trend': float(convergence_trend),
            'overfitting_indicator': bool(final_gap > 0.05 or convergence_trend < 0)
        }

        logger.info(f"   최종 성능 차이: {final_gap:.4f}")
        logger.info(f"   수렴 트렌드: {convergence_trend:.4f}")
        logger.info(f"   학습 곡선 과적합: {'YES' if result['overfitting_indicator'] else 'NO'}")

        self.results['learning_curve'] = result
        return result

    def test_3_cross_validation_consistency(self):
        """3. 교차검증 일관성 분석"""
        logger.info("🔍 3. 교차검증 일관성 분석...")

        data = self.load_spy_data()
        X, y = self.create_v2_features(data)
        X_scaled = self.scaler.fit_transform(X)

        model = Ridge(alpha=self.best_alpha)
        splits = list(self.cv.split(X_scaled))

        fold_performances = []
        fold_mse = []
        fold_mae = []

        for i, (train_idx, val_idx) in enumerate(splits):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)

            r2 = r2_score(y_val, val_pred)
            mse = mean_squared_error(y_val, val_pred)
            mae = mean_absolute_error(y_val, val_pred)

            fold_performances.append(r2)
            fold_mse.append(mse)
            fold_mae.append(mae)

            logger.info(f"   Fold {i+1}: R²={r2:.4f}, MSE={mse:.6f}, MAE={mae:.6f}")

        # 일관성 지표
        r2_mean = np.mean(fold_performances)
        r2_std = np.std(fold_performances)
        cv_stability = r2_std / abs(r2_mean)  # 변동계수

        result = {
            'fold_r2_scores': [float(s) for s in fold_performances],
            'fold_mse_scores': [float(s) for s in fold_mse],
            'fold_mae_scores': [float(s) for s in fold_mae],
            'r2_mean': float(r2_mean),
            'r2_std': float(r2_std),
            'r2_min': float(np.min(fold_performances)),
            'r2_max': float(np.max(fold_performances)),
            'cv_stability': float(cv_stability),
            'consistency_indicator': bool(cv_stability < 0.1)  # 변동계수 10% 미만이면 일관성 있음
        }

        logger.info(f"   R² 평균: {r2_mean:.4f} ± {r2_std:.4f}")
        logger.info(f"   변동계수: {cv_stability:.4f}")
        logger.info(f"   일관성: {'GOOD' if result['consistency_indicator'] else 'POOR'}")

        self.results['cross_validation_consistency'] = result
        return result

    def test_4_walk_forward_validation(self):
        """4. Walk-Forward Validation (시간적 안정성)"""
        logger.info("🔍 4. Walk-Forward Validation...")

        data = self.load_spy_data()
        X, y = self.create_v2_features(data)
        X_scaled = self.scaler.fit_transform(X)

        model = Ridge(alpha=self.best_alpha)

        # 시간 순서대로 walk-forward 검증
        n_samples = len(X_scaled)
        min_train_size = n_samples // 3  # 최소 1/3은 훈련용
        test_size = 200  # 고정 테스트 크기

        walk_forward_scores = []
        time_periods = []

        for start_idx in range(min_train_size, n_samples - test_size, test_size // 2):
            end_idx = min(start_idx + test_size, n_samples)

            # 훈련: 처음부터 start_idx까지
            # 테스트: start_idx부터 end_idx까지
            X_train = X_scaled[:start_idx]
            y_train = y.iloc[:start_idx]
            X_test = X_scaled[start_idx:end_idx]
            y_test = y.iloc[start_idx:end_idx]

            model.fit(X_train, y_train)
            test_pred = model.predict(X_test)
            test_r2 = r2_score(y_test, test_pred)

            walk_forward_scores.append(test_r2)
            time_periods.append(f"{start_idx}-{end_idx}")

            logger.info(f"   Period {start_idx:4d}-{end_idx:4d}: R²={test_r2:.4f}")

        # 시간적 안정성 분석
        wf_mean = np.mean(walk_forward_scores)
        wf_std = np.std(walk_forward_scores)
        time_trend = np.corrcoef(range(len(walk_forward_scores)), walk_forward_scores)[0, 1]

        result = {
            'walk_forward_scores': [float(s) for s in walk_forward_scores],
            'time_periods': time_periods,
            'wf_mean': float(wf_mean),
            'wf_std': float(wf_std),
            'time_trend': float(time_trend),
            'temporal_stability': bool(abs(time_trend) < 0.3)  # 시간 트렌드가 약하면 안정적
        }

        logger.info(f"   Walk-Forward R²: {wf_mean:.4f} ± {wf_std:.4f}")
        logger.info(f"   시간 트렌드: {time_trend:.4f}")
        logger.info(f"   시간적 안정성: {'STABLE' if result['temporal_stability'] else 'UNSTABLE'}")

        self.results['walk_forward_validation'] = result
        return result

    def test_5_regularization_effect(self):
        """5. 정규화 효과 검증"""
        logger.info("🔍 5. 정규화 효과 검증...")

        data = self.load_spy_data()
        X, y = self.create_v2_features(data)
        X_scaled = self.scaler.fit_transform(X)

        # 다양한 alpha 값으로 검증 곡선 생성
        alpha_range = np.logspace(-2, 2, 20)  # 0.01 ~ 100

        train_scores, val_scores = validation_curve(
            Ridge(), X_scaled, y,
            param_name='alpha',
            param_range=alpha_range,
            cv=self.cv,
            scoring='r2',
            n_jobs=-1
        )

        train_mean = np.mean(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)

        # V2 alpha 위치 찾기
        best_idx = np.argmin(np.abs(alpha_range - self.best_alpha))
        v2_train_score = train_mean[best_idx]
        v2_val_score = val_mean[best_idx]

        # 과적합 구간 찾기
        performance_gaps = train_mean - val_mean
        overfitting_zone = performance_gaps > 0.05

        result = {
            'alpha_range': [float(a) for a in alpha_range],
            'train_scores_mean': [float(s) for s in train_mean],
            'val_scores_mean': [float(s) for s in val_mean],
            'performance_gaps': [float(g) for g in performance_gaps],
            'v2_alpha': float(self.best_alpha),
            'v2_train_score': float(v2_train_score),
            'v2_val_score': float(v2_val_score),
            'v2_performance_gap': float(performance_gaps[best_idx]),
            'optimal_regularization': bool(performance_gaps[best_idx] < 0.05),
            'overfitting_zone_count': int(np.sum(overfitting_zone))
        }

        logger.info(f"   V2 Alpha {self.best_alpha:.4f}: 훈련={v2_train_score:.4f}, 검증={v2_val_score:.4f}")
        logger.info(f"   V2 성능 차이: {performance_gaps[best_idx]:.4f}")
        logger.info(f"   최적 정규화: {'YES' if result['optimal_regularization'] else 'NO'}")

        self.results['regularization_effect'] = result
        return result

    def test_6_feature_stability(self):
        """6. 특성 중요도 안정성"""
        logger.info("🔍 6. 특성 중요도 안정성...")

        data = self.load_spy_data()
        X, y = self.create_v2_features(data)
        X_scaled = self.scaler.fit_transform(X)

        model = Ridge(alpha=self.best_alpha)
        splits = list(self.cv.split(X_scaled))

        feature_names = X.columns.tolist()
        feature_importance_matrix = []

        for train_idx, val_idx in splits:
            X_train = X_scaled[train_idx]
            y_train = y.iloc[train_idx]

            model.fit(X_train, y_train)

            # Ridge 계수를 중요도로 사용
            importance = np.abs(model.coef_)
            feature_importance_matrix.append(importance)

        feature_importance_matrix = np.array(feature_importance_matrix)

        # 안정성 지표
        importance_mean = np.mean(feature_importance_matrix, axis=0)
        importance_std = np.std(feature_importance_matrix, axis=0)
        stability_scores = importance_std / (importance_mean + 1e-8)  # 변동계수

        # 중요한 특성 (상위 10개)
        top_features_idx = np.argsort(importance_mean)[-10:]

        result = {
            'feature_names': feature_names,
            'importance_mean': [float(s) for s in importance_mean],
            'importance_std': [float(s) for s in importance_std],
            'stability_scores': [float(s) for s in stability_scores],
            'top_features': [feature_names[i] for i in top_features_idx],
            'top_features_stability': [float(stability_scores[i]) for i in top_features_idx],
            'overall_stability': float(np.mean(stability_scores)),
            'stable_features': bool(np.mean(stability_scores) < 0.5)  # 변동계수 50% 미만이면 안정적
        }

        logger.info(f"   전체 안정성: {result['overall_stability']:.4f}")
        logger.info(f"   상위 특성: {', '.join(result['top_features'][:5])}")
        logger.info(f"   특성 안정성: {'STABLE' if result['stable_features'] else 'UNSTABLE'}")

        self.results['feature_stability'] = result
        return result

    def generate_comprehensive_report(self):
        """종합 과적합 분석 보고서 생성"""
        logger.info("📋 종합 과적합 분석 보고서 생성...")

        # 각 테스트 실행
        test1 = self.test_1_train_vs_validation()
        test2 = self.test_2_learning_curve_analysis()
        test3 = self.test_3_cross_validation_consistency()
        test4 = self.test_4_walk_forward_validation()
        test5 = self.test_5_regularization_effect()
        test6 = self.test_6_feature_stability()

        # 과적합 지표 종합
        overfitting_indicators = [
            test1['overfitting_indicator'],
            test2['overfitting_indicator'],
            not test3['consistency_indicator'],
            not test4['temporal_stability'],
            not test5['optimal_regularization'],
            not test6['stable_features']
        ]

        overfitting_count = sum(overfitting_indicators)
        overfitting_risk = overfitting_count / len(overfitting_indicators)

        # 최종 판정
        if overfitting_risk <= 0.2:
            final_verdict = "LOW_RISK"
            verdict_desc = "과적합 위험 낮음"
        elif overfitting_risk <= 0.5:
            final_verdict = "MEDIUM_RISK"
            verdict_desc = "과적합 위험 보통"
        else:
            final_verdict = "HIGH_RISK"
            verdict_desc = "과적합 위험 높음"

        comprehensive_result = {
            'model_info': {
                'alpha': self.best_alpha,
                'reported_r2': 0.3256,
                'analysis_date': datetime.now().isoformat()
            },
            'test_results': {
                'train_vs_validation': test1,
                'learning_curve': test2,
                'cross_validation_consistency': test3,
                'walk_forward_validation': test4,
                'regularization_effect': test5,
                'feature_stability': test6
            },
            'overfitting_assessment': {
                'overfitting_indicators': overfitting_indicators,
                'overfitting_count': overfitting_count,
                'overfitting_risk': float(overfitting_risk),
                'final_verdict': final_verdict,
                'verdict_description': verdict_desc
            },
            'recommendations': self.generate_recommendations(overfitting_risk, overfitting_indicators)
        }

        # 결과 저장
        save_path = '/root/workspace/data/raw/v2_comprehensive_overfitting_analysis.json'
        with open(save_path, 'w') as f:
            json.dump(comprehensive_result, f, indent=2)

        logger.info(f"📊 종합 분석 결과:")
        logger.info(f"   과적합 지표: {overfitting_count}/6개 양성")
        logger.info(f"   과적합 위험도: {overfitting_risk:.1%}")
        logger.info(f"   최종 판정: {verdict_desc}")
        logger.info(f"💾 결과 저장: {save_path}")

        return comprehensive_result

    def generate_recommendations(self, overfitting_risk, indicators):
        """과적합 분석 기반 권장사항 생성"""
        recommendations = []

        if overfitting_risk <= 0.2:
            recommendations.extend([
                "✅ V2 모델은 과적합 위험이 낮음",
                "✅ 현재 설정 (alpha=19.5029)을 프로덕션에서 사용 권장",
                "✅ 정기적인 성능 모니터링만으로 충분"
            ])
        elif overfitting_risk <= 0.5:
            recommendations.extend([
                "⚠️ 중간 수준의 과적합 위험 존재",
                "⚠️ 추가 정규화 고려 (alpha 증가)",
                "⚠️ 특성 선택 재검토 필요",
                "⚠️ 더 엄격한 검증 체계 도입"
            ])
        else:
            recommendations.extend([
                "❌ 높은 과적합 위험 - 즉시 개선 필요",
                "❌ 강한 정규화 적용 (alpha >> 19.5)",
                "❌ 특성 수 대폭 감소",
                "❌ 앙상블 방법 고려",
                "❌ 프로덕션 배포 보류"
            ])

        return recommendations

def main():
    """메인 실행 함수"""
    logger.info("🔍 V2 모델 과적합 종합 검증 시작")

    checker = V2OverfittingChecker()

    try:
        result = checker.generate_comprehensive_report()

        verdict = result['overfitting_assessment']['final_verdict']
        risk = result['overfitting_assessment']['overfitting_risk']

        logger.info("="*60)
        logger.info("🎯 V2 모델 과적합 검증 완료")
        logger.info(f"📊 최종 판정: {result['overfitting_assessment']['verdict_description']}")
        logger.info(f"📊 위험도: {risk:.1%}")
        logger.info("="*60)

        # 권장사항 출력
        logger.info("💡 권장사항:")
        for rec in result['recommendations']:
            logger.info(f"   {rec}")

        return result

    except Exception as e:
        logger.error(f"❌ 과적합 검증 실패: {str(e)}")
        raise

if __name__ == "__main__":
    main()