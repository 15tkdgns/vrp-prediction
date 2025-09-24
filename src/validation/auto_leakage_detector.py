#!/usr/bin/env python3
"""
자동 데이터 누출 검증 파이프라인
실시간 모니터링 및 즉시 차단 시스템
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutoLeakageDetector:
    """자동 데이터 누출 검증 시스템"""

    def __init__(self):
        # CLAUDE.md 준수 기준
        self.CRITICAL_R2_THRESHOLD = 0.95     # 95% 이상 절대 금지
        self.SUSPICIOUS_R2_THRESHOLD = 0.15   # 15% 이상 의심
        self.WARNING_R2_THRESHOLD = 0.10      # 10% 이상 경고

        self.CRITICAL_ACC_THRESHOLD = 95.0    # 95% 이상 절대 금지
        self.SUSPICIOUS_ACC_THRESHOLD = 65.0  # 65% 이상 의심
        self.WARNING_ACC_THRESHOLD = 60.0     # 60% 이상 경고

        self.MAX_CORRELATION = 0.30           # 최대 허용 상관관계
        self.MAX_IC = 0.08                   # 최대 정보 계수

        self.validation_history = []
        self.alert_count = 0

        logger.info("자동 누출 검증 시스템 초기화")
        logger.info(f"CLAUDE.md 기준 - 금지: >95%, 의심: >15% R² / >65% 정확도")

    def validate_data_before_training(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> bool:
        """훈련 전 데이터 검증"""
        logger.info("=== 훈련 전 데이터 검증 ===")

        validation_result = {
            'stage': 'pre_training',
            'timestamp': pd.Timestamp.now(),
            'passed': True,
            'issues': []
        }

        # 1. 기본 데이터 품질 검사
        issues = self._check_data_quality(X, y, feature_names)
        validation_result['issues'].extend(issues)

        # 2. 특징-타겟 상관관계 검사
        correlation_issues = self._check_feature_target_correlation(X, y, feature_names)
        validation_result['issues'].extend(correlation_issues)

        # 3. 정보 계수 검사
        ic_issues = self._check_information_coefficient(X, y)
        validation_result['issues'].extend(ic_issues)

        # 4. 기준선 성능 측정
        baseline_metrics = self._measure_baseline_performance(X, y)
        validation_result['baseline_metrics'] = baseline_metrics

        # 5. 검증 결과 평가
        validation_result['passed'] = len(validation_result['issues']) == 0

        # 기준선 성능 검사
        if baseline_metrics.get('r2', 0) > self.SUSPICIOUS_R2_THRESHOLD:
            validation_result['issues'].append(f"기준선 R² 과도: {baseline_metrics['r2']:.3f}")
            validation_result['passed'] = False

        if baseline_metrics.get('direction_acc', 0) > self.SUSPICIOUS_ACC_THRESHOLD:
            validation_result['issues'].append(f"기준선 정확도 과도: {baseline_metrics['direction_acc']:.1f}%")
            validation_result['passed'] = False

        # 결과 기록
        self.validation_history.append(validation_result)

        # 결과 출력
        if validation_result['passed']:
            logger.info("✅ 훈련 전 검증 통과")
        else:
            logger.error("❌ 훈련 전 검증 실패")
            for issue in validation_result['issues']:
                logger.error(f"   - {issue}")

        return validation_result['passed']

    def validate_during_training(self, fold: int, model_name: str, metrics: Dict[str, float]) -> bool:
        """훈련 중 실시간 검증"""
        logger.info(f"=== 훈련 중 검증 (Fold {fold}, {model_name}) ===")

        validation_result = {
            'stage': 'during_training',
            'fold': fold,
            'model_name': model_name,
            'timestamp': pd.Timestamp.now(),
            'metrics': metrics,
            'status': 'PASS',
            'issues': []
        }

        # 1. R² 검사 (회귀 모델)
        r2 = metrics.get('r2', 0)
        if r2 > self.CRITICAL_R2_THRESHOLD:
            validation_result['status'] = 'CRITICAL_STOP'
            validation_result['issues'].append(f"CLAUDE.md 위반: R² {r2:.3f} > {self.CRITICAL_R2_THRESHOLD}")
            self.alert_count += 1

        elif r2 > self.SUSPICIOUS_R2_THRESHOLD:
            validation_result['status'] = 'SUSPICIOUS'
            validation_result['issues'].append(f"의심스러운 R²: {r2:.3f} > {self.SUSPICIOUS_R2_THRESHOLD}")

        elif r2 > self.WARNING_R2_THRESHOLD:
            validation_result['status'] = 'WARNING'
            validation_result['issues'].append(f"경고 R²: {r2:.3f} > {self.WARNING_R2_THRESHOLD}")

        # 2. 정확도 검사 (분류 모델)
        accuracy = metrics.get('accuracy', 0) * 100  # 백분율 변환
        direction_acc = metrics.get('direction_accuracy', 0)

        max_acc = max(accuracy, direction_acc)

        if max_acc > self.CRITICAL_ACC_THRESHOLD:
            validation_result['status'] = 'CRITICAL_STOP'
            validation_result['issues'].append(f"CLAUDE.md 위반: 정확도 {max_acc:.1f}% > {self.CRITICAL_ACC_THRESHOLD}%")
            self.alert_count += 1

        elif max_acc > self.SUSPICIOUS_ACC_THRESHOLD:
            validation_result['status'] = 'SUSPICIOUS'
            validation_result['issues'].append(f"의심스러운 정확도: {max_acc:.1f}% > {self.SUSPICIOUS_ACC_THRESHOLD}%")

        elif max_acc > self.WARNING_ACC_THRESHOLD:
            validation_result['status'] = 'WARNING'
            validation_result['issues'].append(f"경고 정확도: {max_acc:.1f}% > {self.WARNING_ACC_THRESHOLD}%")

        # 3. 결과 기록
        self.validation_history.append(validation_result)

        # 4. 결과 출력
        status_icon = {
            'PASS': '✅',
            'WARNING': '⚠️',
            'SUSPICIOUS': '🚨',
            'CRITICAL_STOP': '❌'
        }

        logger.info(f"{status_icon[validation_result['status']]} {validation_result['status']}")

        if validation_result['issues']:
            for issue in validation_result['issues']:
                logger.warning(f"   - {issue}")

        # 5. 중단 여부 결정
        should_stop = validation_result['status'] == 'CRITICAL_STOP'

        if should_stop:
            logger.error(f"🛑 훈련 중단 필요: {model_name} (Fold {fold})")

        return not should_stop  # False면 중단

    def validate_after_training(self, all_results: Dict[str, Dict]) -> bool:
        """훈련 후 종합 검증"""
        logger.info("=== 훈련 후 종합 검증 ===")

        validation_result = {
            'stage': 'post_training',
            'timestamp': pd.Timestamp.now(),
            'model_count': len(all_results),
            'passed_models': [],
            'failed_models': [],
            'overall_status': 'PASS'
        }

        for model_name, results in all_results.items():
            model_status = self._validate_single_model_results(model_name, results)

            if model_status['passed']:
                validation_result['passed_models'].append(model_name)
            else:
                validation_result['failed_models'].append({
                    'name': model_name,
                    'issues': model_status['issues']
                })

        # 전체 평가
        if validation_result['failed_models']:
            validation_result['overall_status'] = 'FAILED'

        # 결과 기록
        self.validation_history.append(validation_result)

        # 결과 출력
        logger.info(f"통과 모델: {len(validation_result['passed_models'])}개")
        logger.info(f"실패 모델: {len(validation_result['failed_models'])}개")

        if validation_result['failed_models']:
            logger.error("실패한 모델들:")
            for failed in validation_result['failed_models']:
                logger.error(f"   - {failed['name']}: {failed['issues']}")

        return validation_result['overall_status'] == 'PASS'

    def _check_data_quality(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> List[str]:
        """기본 데이터 품질 검사"""
        issues = []

        # 1. 무한값/NaN 검사
        if np.any(np.isinf(X)) or np.any(np.isnan(X)):
            issues.append("특징에 무한값 또는 NaN 포함")

        if np.any(np.isinf(y)) or np.any(np.isnan(y)):
            issues.append("타겟에 무한값 또는 NaN 포함")

        # 2. 데이터 크기 검사
        if len(X) < 200:
            issues.append(f"데이터 크기 부족: {len(X)}개 (최소 200개 필요)")

        # 3. 특징 개수 검사
        if X.shape[1] > 100:
            issues.append(f"특징 개수 과다: {X.shape[1]}개 (권장 최대 50개)")

        # 4. 타겟 분산 검사
        if np.std(y) < 1e-6:
            issues.append("타겟 분산이 거의 0 (상수 타겟 의심)")

        return issues

    def _check_feature_target_correlation(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> List[str]:
        """특징-타겟 상관관계 검사"""
        issues = []

        max_corr = 0
        suspicious_features = []

        for i in range(X.shape[1]):
            corr = abs(np.corrcoef(X[:, i], y)[0, 1])

            if not np.isnan(corr):
                max_corr = max(max_corr, corr)

                if corr > self.MAX_CORRELATION:
                    feature_name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
                    suspicious_features.append((feature_name, corr))

        if suspicious_features:
            issues.append(f"과도한 상관관계 특징: {len(suspicious_features)}개")
            for name, corr in suspicious_features[:5]:
                issues.append(f"   {name}: {corr:.3f}")

        logger.info(f"최대 특징-타겟 상관관계: {max_corr:.3f}")

        return issues

    def _check_information_coefficient(self, X: np.ndarray, y: np.ndarray) -> List[str]:
        """정보 계수 검사"""
        issues = []

        from scipy.stats import spearmanr

        ic_values = []
        for i in range(X.shape[1]):
            try:
                ic, _ = spearmanr(X[:, i], y)
                if not np.isnan(ic):
                    ic_values.append(abs(ic))
            except:
                continue

        if ic_values:
            max_ic = max(ic_values)
            avg_ic = np.mean(ic_values)

            logger.info(f"정보 계수 - 최대: {max_ic:.3f}, 평균: {avg_ic:.3f}")

            if max_ic > self.MAX_IC:
                issues.append(f"과도한 정보 계수: {max_ic:.3f} > {self.MAX_IC}")

        return issues

    def _measure_baseline_performance(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """기준선 성능 측정"""
        logger.info("기준선 성능 측정")

        # 간단한 시간 분할
        split_point = int(len(X) * 0.8)
        X_train, X_test = X[:split_point], X[split_point:]
        y_train, y_test = y[:split_point], y[split_point:]

        # 데이터 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 선형 회귀 기준선
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)

        # 회귀 지표
        r2 = r2_score(y_test, y_pred)

        # 방향 정확도 (회귀에서도 측정)
        direction_correct = np.sum(np.sign(y_test) == np.sign(y_pred))
        direction_acc = direction_correct / len(y_test) * 100

        metrics = {
            'r2': r2,
            'direction_acc': direction_acc,
            'samples': len(y_test)
        }

        logger.info(f"기준선 성능 - R²: {r2:.3f}, 방향정확도: {direction_acc:.1f}%")

        return metrics

    def _validate_single_model_results(self, model_name: str, results: Dict) -> Dict:
        """개별 모델 결과 검증"""
        validation = {
            'model_name': model_name,
            'passed': True,
            'issues': []
        }

        # 평균 성능 추출
        avg_r2 = results.get('mean_r2', 0)
        avg_acc = results.get('mean_accuracy', 0) * 100
        avg_direction = results.get('mean_direction_accuracy', 0)

        # R² 검증
        if avg_r2 > self.CRITICAL_R2_THRESHOLD:
            validation['passed'] = False
            validation['issues'].append(f"CLAUDE.md 위반 R²: {avg_r2:.3f}")

        elif avg_r2 > self.SUSPICIOUS_R2_THRESHOLD:
            validation['passed'] = False
            validation['issues'].append(f"의심스러운 R²: {avg_r2:.3f}")

        # 정확도 검증
        max_acc = max(avg_acc, avg_direction)
        if max_acc > self.CRITICAL_ACC_THRESHOLD:
            validation['passed'] = False
            validation['issues'].append(f"CLAUDE.md 위반 정확도: {max_acc:.1f}%")

        elif max_acc > self.SUSPICIOUS_ACC_THRESHOLD:
            validation['passed'] = False
            validation['issues'].append(f"의심스러운 정확도: {max_acc:.1f}%")

        return validation

    def generate_validation_report(self) -> Dict:
        """검증 보고서 생성"""
        logger.info("=== 검증 보고서 생성 ===")

        report = {
            'summary': {
                'total_validations': len(self.validation_history),
                'alert_count': self.alert_count,
                'last_validation': self.validation_history[-1]['timestamp'] if self.validation_history else None
            },
            'stage_summary': {},
            'detailed_history': self.validation_history
        }

        # 단계별 요약
        stages = ['pre_training', 'during_training', 'post_training']
        for stage in stages:
            stage_validations = [v for v in self.validation_history if v.get('stage') == stage]
            report['stage_summary'][stage] = {
                'count': len(stage_validations),
                'passed': sum(1 for v in stage_validations if v.get('passed', True) or v.get('status') in ['PASS', 'WARNING']),
                'failed': sum(1 for v in stage_validations if not v.get('passed', True) or v.get('status') in ['SUSPICIOUS', 'CRITICAL_STOP'])
            }

        # 보고서 출력
        print("\n" + "=" * 80)
        print("📋 자동 데이터 누출 검증 보고서")
        print("=" * 80)

        print(f"\n📊 검증 요약:")
        print(f"   총 검증 횟수: {report['summary']['total_validations']}")
        print(f"   경고 발생: {report['summary']['alert_count']}")

        for stage, summary in report['stage_summary'].items():
            print(f"\n{stage.replace('_', ' ').title()}:")
            print(f"   검증 수행: {summary['count']}")
            print(f"   통과: {summary['passed']}")
            print(f"   실패: {summary['failed']}")

        # 최근 이슈들
        recent_issues = []
        for validation in self.validation_history[-5:]:  # 최근 5개
            if validation.get('issues'):
                recent_issues.extend(validation['issues'])

        if recent_issues:
            print(f"\n⚠️ 최근 발견된 이슈들:")
            for issue in recent_issues[:10]:  # 최대 10개
                print(f"   - {issue}")

        print("\n" + "=" * 80)

        return report

def main():
    """테스트 실행"""
    detector = AutoLeakageDetector()

    # 더미 데이터로 테스트
    X = np.random.randn(1000, 10)
    y = np.random.randn(1000)
    feature_names = [f"feature_{i}" for i in range(10)]

    # 1. 훈련 전 검증
    pre_valid = detector.validate_data_before_training(X, y, feature_names)
    print(f"훈련 전 검증: {'통과' if pre_valid else '실패'}")

    # 2. 훈련 중 검증 시뮬레이션
    test_metrics = [
        {'r2': 0.05, 'accuracy': 0.55, 'direction_accuracy': 55.0},  # 정상
        {'r2': 0.12, 'accuracy': 0.62, 'direction_accuracy': 62.0},  # 경고
        {'r2': 0.20, 'accuracy': 0.70, 'direction_accuracy': 70.0},  # 의심
        {'r2': 0.96, 'accuracy': 0.96, 'direction_accuracy': 96.0},  # 중단
    ]

    for i, metrics in enumerate(test_metrics):
        result = detector.validate_during_training(i, f"test_model_{i}", metrics)
        print(f"모델 {i} 검증: {'계속' if result else '중단'}")

    # 3. 검증 보고서
    report = detector.generate_validation_report()

    return detector

if __name__ == "__main__":
    detector = main()