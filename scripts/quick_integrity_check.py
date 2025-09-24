#!/usr/bin/env python3
"""
🚀 빠른 무결성 검증 스크립트

일상적인 작업 전후로 실행하여 데이터 무결성을 확인하는 간단한 도구
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

try:
    from src.validation.auto_hallucination_detector import HallucinationDetector
except ImportError:
    print("❌ 검증 모듈을 찾을 수 없습니다. PYTHONPATH를 확인하세요.")
    sys.exit(1)

def quick_check():
    """빠른 무결성 검증"""
    print("🚀 빠른 무결성 검증 시작...")
    print("="*40)

    detector = HallucinationDetector()

    # 핵심 파일들만 빠르게 체크
    critical_files = [
        "/root/workspace/data/raw/simulated_successful_validation.json",
        "/root/workspace/현재_시스템_상태2.txt",
        "/root/workspace/data/raw/model_performance.json"
    ]

    results = {
        'safe_files': [],
        'hallucination_files': [],
        'missing_files': []
    }

    for file_path in critical_files:
        if not Path(file_path).exists():
            results['missing_files'].append(file_path)
            continue

        validation = detector.detect_file_hallucination(file_path)

        if validation.risk_level == "HIGH":
            results['hallucination_files'].append({
                'path': file_path,
                'issues': validation.issues[:2]  # 상위 2개만
            })
        else:
            results['safe_files'].append(file_path)

    # 결과 출력
    print(f"✅ 안전 파일: {len(results['safe_files'])}개")
    for file_path in results['safe_files']:
        print(f"   • {Path(file_path).name}")

    if results['hallucination_files']:
        print(f"\n🚨 할루시네이션 파일: {len(results['hallucination_files'])}개")
        for file_info in results['hallucination_files']:
            print(f"   ❌ {Path(file_info['path']).name}")
            for issue in file_info['issues']:
                print(f"      - {issue}")

    if results['missing_files']:
        print(f"\n⚠️ 누락 파일: {len(results['missing_files'])}개")
        for file_path in results['missing_files']:
            print(f"   ? {Path(file_path).name}")

    # 종합 판정
    if len(results['hallucination_files']) == 0:
        print(f"\n🎉 무결성 검증 통과!")
        return True
    else:
        print(f"\n⚠️ 할루시네이션 파일 발견. 주의 필요!")
        return False

def check_experiment_registry():
    """실험 등록부 상태 확인"""
    registry_path = "/root/workspace/data/experiment_registry.json"

    if not Path(registry_path).exists():
        print("❌ 실험 등록부가 없습니다.")
        return False

    try:
        with open(registry_path, 'r') as f:
            registry = json.load(f)

        stats = registry.get('experiment_registry', {})
        print(f"\n📊 실험 등록부 상태:")
        print(f"   총 실험: {stats.get('total_experiments', 0)}개")
        print(f"   검증된 실험: {stats.get('validated_experiments', 0)}개")
        print(f"   할루시네이션: {stats.get('hallucination_experiments', 0)}개")

        integrity = stats.get('validated_experiments', 0) / max(1, stats.get('total_experiments', 1)) * 100
        print(f"   무결성: {integrity:.1f}%")

        return integrity > 70

    except Exception as e:
        print(f"❌ 등록부 읽기 오류: {e}")
        return False

def main():
    """메인 실행 함수"""
    print("🔍 프로젝트 무결성 빠른 검증")
    print("=" * 50)

    # 1. 핵심 파일 검증
    integrity_ok = quick_check()

    # 2. 실험 등록부 확인
    registry_ok = check_experiment_registry()

    # 3. 종합 판정
    print("\n" + "=" * 50)
    if integrity_ok and registry_ok:
        print("🎉 전체 시스템 무결성: 양호")
        print("✅ 안전하게 작업을 진행할 수 있습니다.")
        exit_code = 0
    else:
        print("⚠️ 전체 시스템 무결성: 주의 필요")
        print("🔧 상세 검증을 실행하거나 문제를 해결하세요.")
        print("\n상세 검증 명령어:")
        print("   python3 src/validation/auto_hallucination_detector.py")
        exit_code = 1

    return exit_code

if __name__ == "__main__":
    sys.exit(main())