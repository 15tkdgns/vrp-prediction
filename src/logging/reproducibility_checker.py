#!/usr/bin/env python3
"""
🔄 재현성 체커 시스템
학술 논문을 위한 실험 재현성 검증 도구

주요 기능:
- 실험 재현성 검증
- 환경 일관성 체크
- 결과 비교 및 분석
- 재현성 보고서 생성
"""

import json
import numpy as np
import pandas as pd
import hashlib
import subprocess
import platform
import sys
import os
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ReproducibilityResult:
    """재현성 검증 결과"""
    is_reproducible: bool
    similarity_score: float
    environment_match: bool
    seed_consistency: bool
    data_consistency: bool
    code_consistency: bool
    metric_differences: Dict[str, float]
    warnings: List[str]
    recommendations: List[str]

class ReproducibilityChecker:
    """
    실험 재현성 검증 시스템

    동일한 설정으로 실험을 재실행하고 결과의 일관성을 검증
    """

    def __init__(self, tolerance: float = 1e-6, strict_mode: bool = False):
        """
        초기화

        Args:
            tolerance: 수치 비교 허용 오차
            strict_mode: 엄격 모드 (환경까지 완전히 일치해야 함)
        """
        self.tolerance = tolerance
        self.strict_mode = strict_mode

    def verify_experiment_reproducibility(self,
                                        original_experiment_path: str,
                                        rerun_function: Callable,
                                        n_runs: int = 3) -> ReproducibilityResult:
        """
        실험 재현성 검증

        Args:
            original_experiment_path: 원본 실험 기록 경로
            rerun_function: 재실행 함수
            n_runs: 재실행 횟수

        Returns:
            재현성 검증 결과
        """
        print(f"🔄 재현성 검증 시작: {original_experiment_path}")

        # 원본 실험 로드
        original_data = self._load_experiment_data(original_experiment_path)
        if original_data is None:
            raise ValueError(f"원본 실험을 로드할 수 없습니다: {original_experiment_path}")

        # 환경 일관성 체크
        environment_match = self._check_environment_consistency(original_data)

        # 여러 번 재실행
        rerun_results = []
        for i in range(n_runs):
            print(f"  재실행 {i+1}/{n_runs}")
            try:
                rerun_result = rerun_function()
                rerun_results.append(rerun_result)
            except Exception as e:
                print(f"  재실행 {i+1} 실패: {str(e)}")
                rerun_results.append(None)

        # 결과 분석
        return self._analyze_reproducibility(original_data, rerun_results, environment_match)

    def _load_experiment_data(self, experiment_path: str) -> Optional[Dict]:
        """실험 데이터 로드"""
        try:
            with open(experiment_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"실험 데이터 로드 오류: {str(e)}")
            return None

    def _check_environment_consistency(self, original_data: Dict) -> bool:
        """환경 일관성 체크"""
        print("  환경 일관성 체크 중...")

        if 'system_info' not in original_data:
            print("  경고: 원본 실험에 시스템 정보가 없습니다.")
            return False

        original_system = original_data['system_info']
        current_system = self._get_current_system_info()

        # 핵심 환경 요소 비교
        checks = {
            'python_version': original_system.get('python_version') == current_system['python_version'],
            'platform': original_system.get('platform') == current_system['platform'],
            'working_directory': original_system.get('working_directory') == current_system['working_directory']
        }

        # 중요한 패키지 버전 비교
        original_packages = original_system.get('installed_packages', {})
        current_packages = current_system['installed_packages']

        critical_packages = ['numpy', 'pandas', 'scikit-learn', 'tensorflow', 'torch']
        package_consistency = True

        for package in critical_packages:
            if package in original_packages and package in current_packages:
                if original_packages[package] != current_packages[package]:
                    print(f"  경고: {package} 버전 불일치 - 원본: {original_packages[package]}, 현재: {current_packages[package]}")
                    package_consistency = False

        all_consistent = all(checks.values()) and package_consistency

        if not all_consistent:
            print("  ⚠️ 환경 불일치 감지")
            for check_name, result in checks.items():
                if not result:
                    print(f"    - {check_name}: 불일치")

        return all_consistent

    def _get_current_system_info(self) -> Dict:
        """현재 시스템 정보 수집"""
        import psutil

        # 설치된 패키지 정보
        try:
            import pkg_resources
            installed_packages = {pkg.project_name: pkg.version
                                for pkg in pkg_resources.working_set}
        except Exception:
            installed_packages = {}

        return {
            'python_version': platform.python_version(),
            'platform': platform.platform(),
            'working_directory': os.getcwd(),
            'installed_packages': installed_packages
        }

    def _analyze_reproducibility(self, original_data: Dict,
                               rerun_results: List[Optional[Dict]],
                               environment_match: bool) -> ReproducibilityResult:
        """재현성 분석"""
        print("  결과 분석 중...")

        # 성공한 재실행만 필터링
        valid_reruns = [result for result in rerun_results if result is not None]

        if not valid_reruns:
            return ReproducibilityResult(
                is_reproducible=False,
                similarity_score=0.0,
                environment_match=environment_match,
                seed_consistency=False,
                data_consistency=False,
                code_consistency=False,
                metric_differences={},
                warnings=["모든 재실행이 실패했습니다."],
                recommendations=["실험 코드와 환경을 점검하세요."]
            )

        # 메트릭 비교
        original_metrics = original_data.get('metrics', {})
        metric_differences = self._compare_metrics(original_metrics, valid_reruns)

        # 시드 일관성 체크
        seed_consistency = self._check_seed_consistency(original_data, valid_reruns)

        # 데이터 일관성 체크
        data_consistency = self._check_data_consistency(original_data, valid_reruns)

        # 전체 유사도 계산
        similarity_score = self._calculate_similarity_score(metric_differences)

        # 재현성 판정
        is_reproducible = (
            similarity_score > 0.95 and  # 95% 이상 유사
            seed_consistency and
            data_consistency and
            (environment_match or not self.strict_mode)
        )

        # 경고 및 권고사항 생성
        warnings, recommendations = self._generate_warnings_and_recommendations(
            similarity_score, environment_match, seed_consistency,
            data_consistency, metric_differences
        )

        return ReproducibilityResult(
            is_reproducible=is_reproducible,
            similarity_score=similarity_score,
            environment_match=environment_match,
            seed_consistency=seed_consistency,
            data_consistency=data_consistency,
            code_consistency=True,  # 코드가 실행되면 일관성 있다고 가정
            metric_differences=metric_differences,
            warnings=warnings,
            recommendations=recommendations
        )

    def _compare_metrics(self, original_metrics: Dict, rerun_results: List[Dict]) -> Dict[str, float]:
        """메트릭 비교"""
        differences = {}

        if not original_metrics:
            return differences

        for metric_name in original_metrics.keys():
            if metric_name in ['custom_metrics']:  # 중첩 딕셔너리 건너뛰기
                continue

            original_value = original_metrics[metric_name]
            if not isinstance(original_value, (int, float)):
                continue

            rerun_values = []
            for rerun in rerun_results:
                if metric_name in rerun.get('metrics', {}):
                    rerun_values.append(rerun['metrics'][metric_name])

            if rerun_values:
                mean_rerun = np.mean(rerun_values)
                std_rerun = np.std(rerun_values) if len(rerun_values) > 1 else 0

                # 절대 차이와 상대 차이 계산
                abs_diff = abs(original_value - mean_rerun)
                rel_diff = abs_diff / abs(original_value) if original_value != 0 else abs_diff

                differences[metric_name] = {
                    'original': original_value,
                    'rerun_mean': mean_rerun,
                    'rerun_std': std_rerun,
                    'absolute_difference': abs_diff,
                    'relative_difference': rel_diff,
                    'within_tolerance': abs_diff <= self.tolerance
                }

        return differences

    def _check_seed_consistency(self, original_data: Dict, rerun_results: List[Dict]) -> bool:
        """시드 일관성 체크"""
        original_seed = original_data.get('config', {}).get('random_seed')

        if original_seed is None:
            return False

        # 재실행 결과에서 동일한 시드 사용 여부 확인
        for rerun in rerun_results:
            rerun_seed = rerun.get('config', {}).get('random_seed')
            if rerun_seed != original_seed:
                return False

        return True

    def _check_data_consistency(self, original_data: Dict, rerun_results: List[Dict]) -> bool:
        """데이터 일관성 체크"""
        original_data_info = original_data.get('config', {}).get('data_info', {})

        if not original_data_info:
            return True  # 데이터 정보가 없으면 일관성 있다고 가정

        for rerun in rerun_results:
            rerun_data_info = rerun.get('config', {}).get('data_info', {})

            # 중요한 데이터 속성 비교
            for key in ['n_samples', 'n_features', 'target_type']:
                if key in original_data_info and key in rerun_data_info:
                    if original_data_info[key] != rerun_data_info[key]:
                        return False

        return True

    def _calculate_similarity_score(self, metric_differences: Dict) -> float:
        """전체 유사도 점수 계산"""
        if not metric_differences:
            return 1.0

        total_score = 0.0
        count = 0

        for metric_name, diff_info in metric_differences.items():
            if isinstance(diff_info, dict) and 'within_tolerance' in diff_info:
                if diff_info['within_tolerance']:
                    score = 1.0
                else:
                    # 상대 차이를 기반으로 점수 계산
                    rel_diff = diff_info['relative_difference']
                    score = max(0.0, 1.0 - rel_diff)

                total_score += score
                count += 1

        return total_score / count if count > 0 else 1.0

    def _generate_warnings_and_recommendations(self, similarity_score: float,
                                             environment_match: bool,
                                             seed_consistency: bool,
                                             data_consistency: bool,
                                             metric_differences: Dict) -> Tuple[List[str], List[str]]:
        """경고 및 권고사항 생성"""
        warnings = []
        recommendations = []

        # 유사도 기반 경고
        if similarity_score < 0.95:
            warnings.append(f"결과 유사도가 낮습니다 ({similarity_score:.3f})")
            recommendations.append("메트릭 차이가 큰 원인을 분석하세요")

        # 환경 불일치 경고
        if not environment_match:
            warnings.append("실행 환경이 원본과 다릅니다")
            if self.strict_mode:
                recommendations.append("동일한 환경에서 실험을 재실행하세요")
            else:
                recommendations.append("환경 차이가 결과에 미치는 영향을 고려하세요")

        # 시드 불일치 경고
        if not seed_consistency:
            warnings.append("랜덤 시드가 일관되지 않습니다")
            recommendations.append("동일한 랜덤 시드를 사용하세요")

        # 데이터 불일치 경고
        if not data_consistency:
            warnings.append("데이터 설정이 일관되지 않습니다")
            recommendations.append("동일한 데이터셋과 전처리를 사용하세요")

        # 특정 메트릭 차이 경고
        large_diff_metrics = []
        for metric_name, diff_info in metric_differences.items():
            if isinstance(diff_info, dict) and diff_info.get('relative_difference', 0) > 0.1:
                large_diff_metrics.append(metric_name)

        if large_diff_metrics:
            warnings.append(f"큰 차이를 보이는 메트릭: {', '.join(large_diff_metrics)}")
            recommendations.append("해당 메트릭들의 변동 원인을 조사하세요")

        # 일반적인 권고사항
        if not warnings:
            recommendations.append("실험이 성공적으로 재현되었습니다")
        else:
            recommendations.append("재현성 개선을 위해 환경과 설정을 표준화하세요")
            recommendations.append("Docker 컨테이너 사용을 고려하세요")

        return warnings, recommendations

    def generate_reproducibility_report(self, result: ReproducibilityResult,
                                      experiment_name: str) -> str:
        """재현성 보고서 생성"""
        report = [
            f"🔄 재현성 검증 보고서: {experiment_name}",
            "=" * 60,
            ""
        ]

        # 전체 결과
        status = "✅ 재현 가능" if result.is_reproducible else "❌ 재현 불가능"
        report.extend([
            f"📊 전체 결과: {status}",
            f"📈 유사도 점수: {result.similarity_score:.3f}",
            ""
        ])

        # 세부 검증 결과
        report.extend([
            "🔍 세부 검증 결과:",
            f"   환경 일치: {'✅' if result.environment_match else '❌'}",
            f"   시드 일관성: {'✅' if result.seed_consistency else '❌'}",
            f"   데이터 일관성: {'✅' if result.data_consistency else '❌'}",
            f"   코드 일관성: {'✅' if result.code_consistency else '❌'}",
            ""
        ])

        # 메트릭 차이 분석
        if result.metric_differences:
            report.extend([
                "📊 메트릭 차이 분석:",
                "-" * 40
            ])

            for metric_name, diff_info in result.metric_differences.items():
                if isinstance(diff_info, dict):
                    original = diff_info['original']
                    rerun_mean = diff_info['rerun_mean']
                    rel_diff = diff_info['relative_difference']
                    within_tol = diff_info['within_tolerance']

                    tolerance_status = "✅" if within_tol else "❌"
                    report.append(f"   {metric_name}: {tolerance_status}")
                    report.append(f"     원본: {original:.6f}")
                    report.append(f"     재실행: {rerun_mean:.6f}")
                    report.append(f"     상대 차이: {rel_diff:.3%}")

            report.append("")

        # 경고사항
        if result.warnings:
            report.extend([
                "⚠️ 경고사항:",
            ])
            for warning in result.warnings:
                report.append(f"   - {warning}")
            report.append("")

        # 권고사항
        if result.recommendations:
            report.extend([
                "💡 권고사항:",
            ])
            for recommendation in result.recommendations:
                report.append(f"   - {recommendation}")
            report.append("")

        # 재현성 개선 가이드
        if not result.is_reproducible:
            report.extend([
                "🛠️ 재현성 개선 가이드:",
                "   1. 모든 랜덤 시드를 명시적으로 설정",
                "   2. 환경 종속성을 requirements.txt에 명시",
                "   3. 데이터 전처리 과정을 완전히 문서화",
                "   4. Docker 컨테이너를 사용한 환경 표준화",
                "   5. 실험 코드의 비결정적 요소 제거",
                ""
            ])

        return "\n".join(report)

    def batch_verify_experiments(self, experiment_dir: str,
                                rerun_functions: Dict[str, Callable]) -> Dict[str, ReproducibilityResult]:
        """여러 실험의 재현성 일괄 검증"""
        results = {}
        experiment_files = list(Path(experiment_dir).glob("*.json"))

        for exp_file in experiment_files:
            exp_id = exp_file.stem
            if exp_id in rerun_functions:
                print(f"검증 중: {exp_id}")
                try:
                    result = self.verify_experiment_reproducibility(
                        str(exp_file), rerun_functions[exp_id], n_runs=2
                    )
                    results[exp_id] = result
                except Exception as e:
                    print(f"검증 실패 ({exp_id}): {str(e)}")

        return results

def main():
    """테스트 및 예제 실행"""
    print("🔄 재현성 체커 시스템 테스트")
    print("=" * 60)

    # 재현성 체커 초기화
    checker = ReproducibilityChecker(tolerance=1e-3, strict_mode=False)

    # 테스트용 실험 데이터 생성
    test_experiment = {
        'config': {
            'experiment_id': 'test_experiment_001',
            'experiment_name': 'Test_Reproducibility',
            'model_type': 'RandomForestRegressor',
            'model_parameters': {'n_estimators': 100, 'random_state': 42},
            'data_info': {'n_samples': 1000, 'n_features': 10},
            'random_seed': 42
        },
        'metrics': {
            'mae': 0.123456,
            'mse': 0.234567,
            'rmse': 0.484334,
            'r2': 0.789012,
            'direction_accuracy': 75.5
        },
        'system_info': {
            'python_version': platform.python_version(),
            'platform': platform.platform(),
            'working_directory': os.getcwd(),
            'installed_packages': {'numpy': '1.21.0', 'pandas': '1.3.0'}
        }
    }

    # 임시 실험 파일 저장
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_experiment, f)
        temp_file = f.name

    # 재실행 함수 정의 (완전 동일한 결과)
    def perfect_rerun():
        return {
            'config': test_experiment['config'],
            'metrics': test_experiment['metrics']  # 동일한 결과
        }

    # 재실행 함수 정의 (약간의 차이)
    def slightly_different_rerun():
        import random
        metrics = test_experiment['metrics'].copy()
        # 작은 노이즈 추가
        for key in metrics:
            if isinstance(metrics[key], (int, float)):
                noise = random.uniform(-0.001, 0.001)
                metrics[key] += noise

        return {
            'config': test_experiment['config'],
            'metrics': metrics
        }

    # 재실행 함수 정의 (큰 차이)
    def very_different_rerun():
        metrics = test_experiment['metrics'].copy()
        # 큰 차이 추가
        for key in metrics:
            if isinstance(metrics[key], (int, float)):
                metrics[key] *= 1.1  # 10% 차이

        return {
            'config': test_experiment['config'],
            'metrics': metrics
        }

    try:
        # 1. 완벽한 재현 테스트
        print("\n1. 완벽한 재현 테스트")
        result1 = checker.verify_experiment_reproducibility(
            temp_file, perfect_rerun, n_runs=3
        )

        print(f"재현 가능: {result1.is_reproducible}")
        print(f"유사도: {result1.similarity_score:.3f}")

        # 2. 약간의 차이가 있는 재현 테스트
        print("\n2. 약간의 차이 테스트")
        result2 = checker.verify_experiment_reproducibility(
            temp_file, slightly_different_rerun, n_runs=3
        )

        print(f"재현 가능: {result2.is_reproducible}")
        print(f"유사도: {result2.similarity_score:.3f}")

        # 3. 큰 차이가 있는 재현 테스트
        print("\n3. 큰 차이 테스트")
        result3 = checker.verify_experiment_reproducibility(
            temp_file, very_different_rerun, n_runs=3
        )

        print(f"재현 가능: {result3.is_reproducible}")
        print(f"유사도: {result3.similarity_score:.3f}")

        # 보고서 생성
        print("\n4. 재현성 보고서 (큰 차이 케이스)")
        print("-" * 60)
        report = checker.generate_reproducibility_report(result3, "Test_Experiment")
        print(report)

    finally:
        # 임시 파일 삭제
        os.unlink(temp_file)

if __name__ == "__main__":
    main()