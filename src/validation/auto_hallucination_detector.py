#!/usr/bin/env python3
"""
🔍 자동 할루시네이션 탐지 시스템

실험 결과의 진위를 자동으로 검증하고 할루시네이션을 탐지하는 시스템
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

warnings.filterwarnings('ignore')

@dataclass
class ValidationResult:
    """검증 결과 데이터 클래스"""
    is_valid: bool
    confidence: float
    issues: List[str]
    recommendations: List[str]
    risk_level: str

class HallucinationDetector:
    """할루시네이션 자동 탐지 시스템"""

    def __init__(self):
        self.red_flags = []
        self.performance_thresholds = {
            'volatility_r2_max': 0.20,
            'direction_accuracy_max': 0.70,
            'classification_accuracy_max': 0.95,
            'minimum_sample_size': 500
        }

    def detect_file_hallucination(self, file_path: str) -> ValidationResult:
        """파일 기반 할루시네이션 탐지"""
        issues = []
        recommendations = []
        confidence = 1.0

        file_path = Path(file_path)

        # 1. 파일명 검사
        if 'simulated' in file_path.name.lower():
            issues.append(f"🚨 파일명에 'simulated' 포함: {file_path.name}")
            confidence -= 0.5

        if 'fake' in file_path.name.lower():
            issues.append(f"🚨 파일명에 'fake' 포함: {file_path.name}")
            confidence -= 0.5

        # 2. 파일 내용 검사
        try:
            if file_path.suffix == '.json':
                validation = self._validate_json_file(file_path)
            elif file_path.suffix == '.txt':
                validation = self._validate_text_file(file_path)
            else:
                validation = ValidationResult(
                    is_valid=True, confidence=0.5,
                    issues=["지원하지 않는 파일 형식"],
                    recommendations=["수동 검증 필요"],
                    risk_level="MEDIUM"
                )

            # 결과 통합
            issues.extend(validation.issues)
            recommendations.extend(validation.recommendations)
            confidence = min(confidence, validation.confidence)

        except Exception as e:
            issues.append(f"❌ 파일 읽기 오류: {str(e)}")
            confidence = 0.0

        # 3. 위험도 평가
        if confidence < 0.3:
            risk_level = "HIGH"
        elif confidence < 0.7:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        is_valid = len(issues) == 0 and confidence > 0.7

        return ValidationResult(
            is_valid=is_valid,
            confidence=confidence,
            issues=issues,
            recommendations=recommendations,
            risk_level=risk_level
        )

    def _validate_json_file(self, file_path: Path) -> ValidationResult:
        """JSON 파일 내용 검증"""
        issues = []
        recommendations = []
        confidence = 1.0

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 할루시네이션 경고 헤더 확인
        if "HALLUCINATION WARNING" in str(data):
            issues.append("🚨 파일에 할루시네이션 경고 존재")
            confidence = 0.0
            recommendations.append("이 파일은 사용 금지됨")

        # 성능 지표 검사
        for key, value in data.items():
            if isinstance(value, (int, float)):
                if 'r2' in key.lower() and value > self.performance_thresholds['volatility_r2_max']:
                    issues.append(f"🚨 비현실적 R²: {key} = {value:.4f} (최대 {self.performance_thresholds['volatility_r2_max']})")
                    confidence -= 0.3

                if 'accuracy' in key.lower() and value > self.performance_thresholds['direction_accuracy_max']:
                    issues.append(f"🚨 비현실적 정확도: {key} = {value:.4f}")
                    confidence -= 0.3

        # 샘플 수 검사
        if 'sample_count' in data:
            if data['sample_count'] < self.performance_thresholds['minimum_sample_size']:
                issues.append(f"⚠️ 샘플 수 부족: {data['sample_count']} < {self.performance_thresholds['minimum_sample_size']}")
                confidence -= 0.2

        # 모든 값이 양수인지 확인 (의심스러운 패턴)
        if self._check_all_positive_pattern(data):
            issues.append("🚨 모든 결과가 양수 (의심스러운 패턴)")
            confidence -= 0.4

        return ValidationResult(
            is_valid=len(issues) == 0,
            confidence=max(0, confidence),
            issues=issues,
            recommendations=recommendations,
            risk_level="HIGH" if confidence < 0.3 else "MEDIUM" if confidence < 0.7 else "LOW"
        )

    def _validate_text_file(self, file_path: Path) -> ValidationResult:
        """텍스트 파일 내용 검증"""
        issues = []
        recommendations = []
        confidence = 1.0

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 할루시네이션 경고 확인
        if "HALLUCINATION WARNING" in content:
            issues.append("🚨 파일에 할루시네이션 경고 존재")
            confidence = 0.0

        # 과장된 성과 표현 탐지
        exaggerated_terms = [
            "완벽", "혁신적", "획기적", "최고", "최적",
            "완전", "절대", "100%", "완성",
            "breakthrough", "perfect", "optimal"
        ]

        for term in exaggerated_terms:
            if term in content.lower():
                issues.append(f"⚠️ 과장된 표현 발견: '{term}'")
                confidence -= 0.1

        # R² 값 추출 및 검사
        import re
        r2_pattern = r'R²[=\s]*([0-9.]+)'
        r2_matches = re.findall(r2_pattern, content)

        for r2_str in r2_matches:
            try:
                r2_value = float(r2_str)
                if r2_value > self.performance_thresholds['volatility_r2_max']:
                    issues.append(f"🚨 비현실적 R²: {r2_value:.4f}")
                    confidence -= 0.3
            except ValueError:
                continue

        return ValidationResult(
            is_valid=len(issues) == 0,
            confidence=max(0, confidence),
            issues=issues,
            recommendations=recommendations,
            risk_level="HIGH" if confidence < 0.3 else "MEDIUM" if confidence < 0.7 else "LOW"
        )

    def _check_all_positive_pattern(self, data: dict) -> bool:
        """모든 값이 양수인 의심스러운 패턴 확인"""
        numeric_values = []

        def extract_numbers(obj):
            if isinstance(obj, (int, float)):
                numeric_values.append(obj)
            elif isinstance(obj, dict):
                for value in obj.values():
                    extract_numbers(value)
            elif isinstance(obj, list):
                for item in obj:
                    extract_numbers(item)

        extract_numbers(data)

        # R² 값들만 확인
        r2_values = []
        for key, value in data.items():
            if 'r2' in key.lower() and isinstance(value, (int, float)):
                r2_values.append(value)
            elif isinstance(value, list) and 'r2' in key.lower():
                r2_values.extend([v for v in value if isinstance(v, (int, float))])

        # R² 값이 10개 이상이고 모두 양수면 의심
        if len(r2_values) >= 10 and all(v > 0 for v in r2_values):
            return True

        return False

    def validate_experiment_consistency(self, experiment_files: List[str]) -> ValidationResult:
        """여러 실험 파일 간 일관성 검증"""
        issues = []
        recommendations = []
        confidence = 1.0

        sample_counts = {}
        r2_values = {}

        # 각 파일에서 메타데이터 추출
        for file_path in experiment_files:
            try:
                if not Path(file_path).exists():
                    continue

                if file_path.endswith('.json'):
                    with open(file_path, 'r') as f:
                        data = json.load(f)

                    # 샘플 수 추출
                    for key, value in data.items():
                        if 'sample' in key.lower() and isinstance(value, int):
                            sample_counts[file_path] = value
                            break

                    # R² 값 추출
                    for key, value in data.items():
                        if 'r2' in key.lower() and isinstance(value, (int, float)):
                            if file_path not in r2_values:
                                r2_values[file_path] = []
                            r2_values[file_path].append(value)

            except Exception as e:
                issues.append(f"❌ 파일 읽기 오류 {file_path}: {str(e)}")
                confidence -= 0.1

        # 샘플 수 일관성 확인
        if len(set(sample_counts.values())) > 1:
            issues.append(f"⚠️ 샘플 수 불일치: {sample_counts}")
            confidence -= 0.2

        # R² 값 일관성 확인 (동일한 실험에서 너무 다른 결과)
        all_r2 = []
        for values in r2_values.values():
            all_r2.extend(values)

        if len(all_r2) > 1:
            r2_std = np.std(all_r2)
            r2_mean = np.mean(all_r2)

            if r2_std > 0.1 and abs(r2_mean) > 0.01:  # 표준편차가 크고 평균이 0이 아님
                issues.append(f"⚠️ R² 값 편차 큼: 평균 {r2_mean:.4f}, 표준편차 {r2_std:.4f}")
                confidence -= 0.1

        return ValidationResult(
            is_valid=len(issues) == 0,
            confidence=max(0, confidence),
            issues=issues,
            recommendations=recommendations,
            risk_level="HIGH" if confidence < 0.3 else "MEDIUM" if confidence < 0.7 else "LOW"
        )

    def generate_validation_report(self, target_dir: str = "/root/workspace") -> Dict:
        """전체 프로젝트 검증 보고서 생성"""
        results = {
            'validation_timestamp': datetime.now().isoformat(),
            'target_directory': target_dir,
            'files_validated': 0,
            'hallucination_files': [],
            'suspicious_files': [],
            'validated_files': [],
            'summary': {}
        }

        # 검증 대상 파일 찾기
        target_path = Path(target_dir)
        json_files = list(target_path.rglob("*.json"))
        txt_files = list(target_path.rglob("*.txt"))

        all_files = json_files + txt_files

        for file_path in all_files:
            # 특정 디렉토리 제외
            if any(exclude in str(file_path) for exclude in ['node_modules', '.git', '__pycache__']):
                continue

            validation = self.detect_file_hallucination(str(file_path))
            results['files_validated'] += 1

            file_info = {
                'path': str(file_path),
                'confidence': validation.confidence,
                'risk_level': validation.risk_level,
                'issues': validation.issues,
                'recommendations': validation.recommendations
            }

            if validation.risk_level == "HIGH":
                results['hallucination_files'].append(file_info)
            elif validation.risk_level == "MEDIUM":
                results['suspicious_files'].append(file_info)
            else:
                results['validated_files'].append(file_info)

        # 요약 통계
        results['summary'] = {
            'total_files': results['files_validated'],
            'hallucination_count': len(results['hallucination_files']),
            'suspicious_count': len(results['suspicious_files']),
            'validated_count': len(results['validated_files']),
            'integrity_percentage': (len(results['validated_files']) / max(1, results['files_validated'])) * 100
        }

        return results

def main():
    """메인 검증 실행"""
    print("🔍 자동 할루시네이션 탐지 시스템 시작")
    print("="*60)

    detector = HallucinationDetector()

    # 전체 프로젝트 검증
    report = detector.generate_validation_report()

    # 결과 출력
    print(f"\n📊 검증 결과 요약:")
    print(f"   전체 파일: {report['summary']['total_files']}개")
    print(f"   할루시네이션: {report['summary']['hallucination_count']}개")
    print(f"   의심스러운 파일: {report['summary']['suspicious_count']}개")
    print(f"   검증된 파일: {report['summary']['validated_count']}개")
    print(f"   무결성: {report['summary']['integrity_percentage']:.1f}%")

    # 할루시네이션 파일 상세
    if report['hallucination_files']:
        print(f"\n🚨 할루시네이션 파일:")
        for file_info in report['hallucination_files']:
            print(f"   ❌ {file_info['path']}")
            print(f"      신뢰도: {file_info['confidence']:.2f}")
            for issue in file_info['issues'][:3]:  # 최대 3개만 표시
                print(f"      - {issue}")

    # 보고서 저장
    report_path = "/root/workspace/data/validation_report_auto.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n💾 상세 보고서 저장: {report_path}")

    return report

if __name__ == "__main__":
    main()