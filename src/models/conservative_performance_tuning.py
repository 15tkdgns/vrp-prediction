#!/usr/bin/env python3
"""
🎯 보수적 성능 튜닝 시스템

기준선: 84.82% (검증된 원본 성과)
목표: 86-88% (안전한 개선)

원본 파이프라인 기반 하이퍼파라미터 미세조정만 수행
"""

import sys
sys.path.append('/root/workspace/src')

from pipeline.advanced_metric_pipeline import AdvancedMetricPipeline
import json
from datetime import datetime
import numpy as np

class ConservativePerformanceTuner:
    """보수적 성능 튜닝 시스템"""

    def __init__(self):
        self.baseline_config = {
            'data_path': '/root/workspace/data/training/sp500_2020_2024_enhanced.csv',
            'target_type': 'direction',
            'sequence_length': 20,
            'cv_splits': 3,
            'gpu_enabled': True,
            'save_models': False,  # 속도 향상
            'save_results': False,
            'output_dir': '/tmp'
        }

    def run_tuning_experiment(self, config_name, modified_params=None):
        """튜닝 실험 실행"""
        print(f"🔧 {config_name} 실험 중...")

        # 기본 설정에 수정사항 적용
        config = self.baseline_config.copy()
        if modified_params:
            config.update(modified_params)

        # 파이프라인 실행
        pipeline = AdvancedMetricPipeline(config)
        results = pipeline.run_advanced_pipeline()

        # 최고 성능 추출
        model_results = results.get('model_results', {})
        best_acc = 0
        best_model = None

        for model_name, metrics in model_results.items():
            if 'direction_accuracy' in metrics:
                acc = metrics['direction_accuracy'] / 100.0  # % to decimal
                if acc > best_acc:
                    best_acc = acc
                    best_model = model_name

        print(f"   ✅ {config_name}: {best_acc:.4f} ({best_model})")

        return {
            'config_name': config_name,
            'best_accuracy': best_acc,
            'best_model': best_model,
            'config': config
        }

    def run_conservative_optimization(self):
        """보수적 최적화 실행"""
        print("🎯 보수적 성능 튜닝 시작")
        print("="*60)

        results = []

        # 1. 베이스라인 (원본 설정)
        baseline_result = self.run_tuning_experiment("베이스라인")
        results.append(baseline_result)

        # 2. CV 분할 수 증가 (더 안정적 검증)
        cv_result = self.run_tuning_experiment(
            "CV 분할 증가",
            {'cv_splits': 5}
        )
        results.append(cv_result)

        # 3. 시퀀스 길이 조정
        seq_result = self.run_tuning_experiment(
            "시퀀스 길이 조정",
            {'sequence_length': 25}
        )
        results.append(seq_result)

        # 4. CV + 시퀀스 조합
        combo_result = self.run_tuning_experiment(
            "CV + 시퀀스 조합",
            {'cv_splits': 5, 'sequence_length': 25}
        )
        results.append(combo_result)

        # 결과 분석
        print(f"\n📊 튜닝 결과 요약:")
        print("-" * 60)

        baseline_acc = results[0]['best_accuracy']
        best_result = max(results, key=lambda x: x['best_accuracy'])

        for result in sorted(results, key=lambda x: x['best_accuracy'], reverse=True):
            improvement = (result['best_accuracy'] - baseline_acc) * 100
            print(f"   {result['config_name']:15s}: {result['best_accuracy']:.4f} ({improvement:+.2f}%p)")

        print(f"\n🏆 최고 성능: {best_result['config_name']} = {best_result['best_accuracy']:.4f}")

        # 목표 달성 여부
        target_min = 0.86
        target_max = 0.88

        if best_result['best_accuracy'] >= target_min:
            if best_result['best_accuracy'] <= target_max:
                print(f"🎯 목표 달성! ({target_min:.2f}-{target_max:.2f} 범위)")
            else:
                print(f"🎯 목표 초과! (목표: {target_max:.2f} 이하)")
        else:
            needed = (target_min - best_result['best_accuracy']) * 100
            print(f"📊 목표 미달 (추가 {needed:.2f}%p 필요)")

        # 최종 결과
        final_result = {
            'baseline_accuracy': baseline_acc,
            'best_accuracy': best_result['best_accuracy'],
            'improvement': (best_result['best_accuracy'] - baseline_acc) * 100,
            'best_config': best_result['config_name'],
            'target_achieved': best_result['best_accuracy'] >= target_min,
            'all_results': results
        }

        return final_result

def main():
    """메인 실행"""
    tuner = ConservativePerformanceTuner()

    results = tuner.run_conservative_optimization()

    # 결과 저장
    experiment_results = {
        'timestamp': datetime.now().isoformat(),
        'experiment_type': 'conservative_performance_tuning',
        'approach': 'Original pipeline with hyperparameter fine-tuning',
        'baseline_accuracy': results['baseline_accuracy'],
        'achieved_accuracy': results['best_accuracy'],
        'improvement': results['improvement'],
        'best_configuration': results['best_config'],
        'target_range': '86-88%',
        'target_achieved': results['target_achieved'],
        'validation_method': 'Original AdvancedMetricPipeline',
        'status': 'completed'
    }

    output_path = f"/root/workspace/data/results/conservative_tuning_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, 'w') as f:
        json.dump(experiment_results, f, indent=2)

    print(f"\n💾 결과 저장: {output_path}")

    return results

if __name__ == "__main__":
    main()