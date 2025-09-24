#!/usr/bin/env python3
"""
🏆 최종 성능 검증 및 비교 시스템

모든 실험 결과를 종합하여 최종 성능 평가 및 검증
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import glob

class FinalPerformanceValidator:
    """최종 성능 검증 시스템"""

    def __init__(self):
        self.results_dir = Path("/root/workspace/data/results")
        self.baseline_accuracy = 0.8482  # 검증된 기준선
        self.target_min = 0.87
        self.target_max = 0.90

    def collect_all_results(self):
        """모든 실험 결과 수집"""
        print("📊 모든 실험 결과 수집 중...")

        all_results = []

        # 다양한 실험 결과 파일들 찾기
        result_patterns = [
            "advanced_metric_results_*.json",
            "fast_ensemble_results_*.json",
            "improved_lstm_results_*.json",
            "enhanced_performance_results_*.json",
            "conservative_tuning_results_*.json",
            "fast_test_results_*.json"
        ]

        for pattern in result_patterns:
            files = glob.glob(str(self.results_dir / pattern))
            for file_path in files:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)

                    # 결과 정규화
                    result = self._normalize_result(data, file_path)
                    if result:
                        all_results.append(result)

                except Exception as e:
                    print(f"   ⚠️ 파일 읽기 오류 {file_path}: {e}")

        print(f"   ✅ 총 {len(all_results)}개 실험 결과 수집")
        return all_results

    def _normalize_result(self, data, file_path):
        """실험 결과 정규화"""
        try:
            # 파일명에서 실험 타입 추출
            filename = Path(file_path).name
            if "advanced_metric" in filename:
                return self._normalize_advanced_metric(data)
            elif "fast_ensemble" in filename:
                return self._normalize_ensemble(data)
            elif "improved_lstm" in filename:
                return self._normalize_lstm(data)
            elif "enhanced_performance" in filename:
                return self._normalize_enhanced(data)
            elif "conservative_tuning" in filename:
                return self._normalize_tuning(data)
            elif "fast_test" in filename:
                return self._normalize_fast_test(data)

        except Exception as e:
            print(f"   ⚠️ 정규화 오류 {file_path}: {e}")
            return None

    def _normalize_advanced_metric(self, data):
        """Advanced Metric 결과 정규화"""
        if 'model_results' not in data:
            return None

        best_acc = 0
        best_model = None

        for model_name, metrics in data['model_results'].items():
            if 'direction_accuracy' in metrics:
                acc = metrics['direction_accuracy'] / 100.0
                if acc > best_acc:
                    best_acc = acc
                    best_model = model_name

        return {
            'experiment_type': 'advanced_metric_pipeline',
            'timestamp': data.get('timestamp', ''),
            'accuracy': best_acc,
            'best_model': best_model,
            'sample_count': data.get('data_info', {}).get('sample_count', 0),
            'feature_count': data.get('data_info', {}).get('feature_count', 0),
            'status': 'completed'
        }

    def _normalize_ensemble(self, data):
        """앙상블 결과 정규화"""
        return {
            'experiment_type': 'fast_ensemble',
            'timestamp': data.get('timestamp', ''),
            'accuracy': data.get('achieved_accuracy', 0),
            'improvement': data.get('improvement', 0),
            'sample_count': data.get('sample_count', 0),
            'feature_count': data.get('feature_count', 0),
            'status': data.get('status', 'completed')
        }

    def _normalize_lstm(self, data):
        """LSTM 결과 정규화"""
        return {
            'experiment_type': 'improved_lstm',
            'timestamp': data.get('timestamp', ''),
            'accuracy': data.get('achieved_accuracy', 0),
            'improvement': data.get('improvement', 0),
            'sample_count': data.get('sample_count', 0),
            'feature_count': data.get('feature_count', 0),
            'status': data.get('status', 'completed')
        }

    def _normalize_enhanced(self, data):
        """Enhanced 결과 정규화"""
        return {
            'experiment_type': 'enhanced_performance',
            'timestamp': data.get('timestamp', ''),
            'accuracy': data.get('achieved_accuracy', 0),
            'improvement': data.get('improvement', 0),
            'target_achieved': data.get('target_achieved', False),
            'sample_count': data.get('sample_count', 0),
            'feature_count': data.get('feature_count', 0),
            'status': data.get('status', 'completed')
        }

    def _normalize_tuning(self, data):
        """Tuning 결과 정규화"""
        return {
            'experiment_type': 'conservative_tuning',
            'timestamp': data.get('timestamp', ''),
            'accuracy': data.get('achieved_accuracy', 0),
            'improvement': data.get('improvement', 0),
            'best_configuration': data.get('best_configuration', ''),
            'target_achieved': data.get('target_achieved', False),
            'status': data.get('status', 'completed')
        }

    def _normalize_fast_test(self, data):
        """Fast Test 결과 정규화"""
        return {
            'experiment_type': 'fast_test_10min',
            'timestamp': data.get('timestamp', ''),
            'accuracy': data.get('best_accuracy', 0),
            'improvement': data.get('improvement', 0),
            'total_time': data.get('total_time_seconds', 0),
            'completed_tests': data.get('completed_tests', 0),
            'status': 'completed' if data.get('completed_tests', 0) > 0 else 'failed'
        }

    def analyze_performance(self, all_results):
        """성능 분석"""
        print("\n🔍 성능 분석 수행...")

        if not all_results:
            print("   ❌ 분석할 결과가 없습니다.")
            return None

        # 성능 순위
        valid_results = [r for r in all_results if r['accuracy'] > 0]
        if not valid_results:
            print("   ❌ 유효한 결과가 없습니다.")
            return None

        sorted_results = sorted(valid_results, key=lambda x: x['accuracy'], reverse=True)

        print(f"\n📊 실험 성능 순위:")
        print("-" * 80)

        for i, result in enumerate(sorted_results[:10], 1):
            improvement = (result['accuracy'] - self.baseline_accuracy) * 100
            status_icon = "🎯" if result['accuracy'] >= self.target_min else "📊"

            print(f"{i:2d}. {status_icon} {result['experiment_type']:25s} "
                  f"{result['accuracy']:.4f} ({improvement:+.2f}%p)")

        # 통계 분석
        accuracies = [r['accuracy'] for r in valid_results]
        improvements = [(r['accuracy'] - self.baseline_accuracy) * 100 for r in valid_results]

        stats = {
            'total_experiments': len(all_results),
            'valid_experiments': len(valid_results),
            'best_accuracy': max(accuracies),
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'best_improvement': max(improvements),
            'mean_improvement': np.mean(improvements),
            'target_achieved_count': sum(1 for acc in accuracies if acc >= self.target_min),
            'target_achievement_rate': sum(1 for acc in accuracies if acc >= self.target_min) / len(accuracies) * 100
        }

        print(f"\n📈 통계 분석:")
        print(f"   총 실험 수: {stats['total_experiments']}개")
        print(f"   유효 실험 수: {stats['valid_experiments']}개")
        print(f"   최고 성능: {stats['best_accuracy']:.4f}")
        print(f"   평균 성능: {stats['mean_accuracy']:.4f} ± {stats['std_accuracy']:.4f}")
        print(f"   최고 개선: {stats['best_improvement']:+.2f}%p")
        print(f"   평균 개선: {stats['mean_improvement']:+.2f}%p")
        print(f"   목표 달성: {stats['target_achieved_count']}/{stats['valid_experiments']}개 "
              f"({stats['target_achievement_rate']:.1f}%)")

        return {
            'results_ranking': sorted_results,
            'statistics': stats,
            'analysis_timestamp': datetime.now().isoformat()
        }

    def validate_integrity(self, all_results):
        """결과 무결성 검증"""
        print("\n🔍 결과 무결성 검증...")

        integrity_issues = []

        for result in all_results:
            # 비현실적 성능 체크
            if result['accuracy'] > 0.98:
                integrity_issues.append(f"의심스러운 고성능: {result['experiment_type']} = {result['accuracy']:.4f}")

            # 음수 개선 체크
            if 'improvement' in result and result['improvement'] < -50:
                integrity_issues.append(f"과도한 성능 저하: {result['experiment_type']} = {result['improvement']:.2f}%p")

            # 데이터 일관성 체크
            if result.get('sample_count', 0) > 0 and result['sample_count'] < 1000:
                integrity_issues.append(f"적은 샘플 수: {result['experiment_type']} = {result['sample_count']}개")

        if integrity_issues:
            print(f"   ⚠️ 발견된 문제:")
            for issue in integrity_issues:
                print(f"      - {issue}")
        else:
            print(f"   ✅ 무결성 검증 통과")

        return len(integrity_issues) == 0

    def generate_final_report(self):
        """최종 보고서 생성"""
        print("🏆 최종 성능 검증 및 보고서 생성")
        print("="*70)

        # 결과 수집
        all_results = self.collect_all_results()

        # 성능 분석
        analysis = self.analyze_performance(all_results)

        # 무결성 검증
        integrity_ok = self.validate_integrity(all_results)

        # 최종 평가
        if analysis:
            best_accuracy = analysis['statistics']['best_accuracy']
            target_achieved = best_accuracy >= self.target_min

            print(f"\n🎯 최종 평가:")
            print(f"   기준선: {self.baseline_accuracy:.4f}")
            print(f"   목표 범위: {self.target_min:.2f}-{self.target_max:.2f}")
            print(f"   달성 성능: {best_accuracy:.4f}")

            if target_achieved:
                if best_accuracy <= self.target_max:
                    print(f"   🎉 목표 달성! (범위 내)")
                else:
                    print(f"   🚀 목표 초과 달성!")
            else:
                needed = (self.target_min - best_accuracy) * 100
                print(f"   📊 목표 미달 (추가 {needed:.2f}%p 필요)")

        # 보고서 저장
        final_report = {
            'validation_timestamp': datetime.now().isoformat(),
            'baseline_accuracy': self.baseline_accuracy,
            'target_range': [self.target_min, self.target_max],
            'all_results': all_results,
            'analysis': analysis,
            'integrity_passed': integrity_ok,
            'final_assessment': {
                'target_achieved': target_achieved if analysis else False,
                'best_accuracy': analysis['statistics']['best_accuracy'] if analysis else 0,
                'recommendation': self._generate_recommendation(analysis, integrity_ok)
            }
        }

        output_path = f"/root/workspace/data/final_performance_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, 'w') as f:
            json.dump(final_report, f, indent=2)

        print(f"\n💾 최종 보고서 저장: {output_path}")

        return final_report

    def _generate_recommendation(self, analysis, integrity_ok):
        """추천사항 생성"""
        if not analysis or not integrity_ok:
            return "추가 검증 필요 - 결과 무결성 문제"

        best_acc = analysis['statistics']['best_accuracy']

        if best_acc >= 0.90:
            return "실전 적용 권장 - 목표 초과 달성"
        elif best_acc >= 0.87:
            return "실전 적용 가능 - 목표 달성"
        elif best_acc >= 0.86:
            return "추가 최적화 후 적용 - 목표 근접"
        else:
            return "추가 연구 필요 - 목표 미달"

def main():
    """메인 실행"""
    validator = FinalPerformanceValidator()
    report = validator.generate_final_report()

    return report

if __name__ == "__main__":
    main()