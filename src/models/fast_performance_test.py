#!/usr/bin/env python3
"""
⚡ 빠른 성능 테스트 (10분 제한)

기준선 성능 재현 및 간단한 개선 테스트
백그라운드 실행용 최적화
"""

import sys
sys.path.append('/root/workspace/src')

from pipeline.advanced_metric_pipeline import AdvancedMetricPipeline
import json
from datetime import datetime
import time

def quick_performance_test():
    """빠른 성능 테스트 실행"""
    print("⚡ 10분 제한 빠른 성능 테스트 시작")
    print(f"시작 시간: {datetime.now().strftime('%H:%M:%S')}")
    print("="*60)

    start_time = time.time()
    results = []

    # 1. 기본 설정 (빠른 실행)
    base_config = {
        'data_path': '/root/workspace/data/training/sp500_2020_2024_enhanced.csv',
        'target_type': 'direction',
        'sequence_length': 20,
        'cv_splits': 2,  # 속도 향상을 위해 2-fold
        'gpu_enabled': True,
        'save_models': False,
        'save_results': False,
        'output_dir': '/tmp'
    }

    try:
        print("🔧 테스트 1: 기본 설정 (2-fold)")
        pipeline1 = AdvancedMetricPipeline(base_config)
        result1 = pipeline1.run_advanced_pipeline()

        # 최고 성능 추출
        best_acc1 = 0
        for model_name, metrics in result1.get('model_results', {}).items():
            if 'direction_accuracy' in metrics:
                acc = metrics['direction_accuracy'] / 100.0
                if acc > best_acc1:
                    best_acc1 = acc

        results.append({
            'config': '기본설정_2fold',
            'accuracy': best_acc1,
            'elapsed': time.time() - start_time
        })

        print(f"   ✅ 결과: {best_acc1:.4f}")
        print(f"   ⏱️ 소요시간: {time.time() - start_time:.1f}초")

        # 시간 체크
        if time.time() - start_time > 480:  # 8분 경과
            print("⏰ 시간 제한 근접, 남은 테스트 건너뛰기")
        else:
            print("🔧 테스트 2: 시퀀스 길이 조정")

            # 2. 시퀀스 길이 조정
            config2 = base_config.copy()
            config2['sequence_length'] = 15  # 더 빠른 처리

            pipeline2 = AdvancedMetricPipeline(config2)
            result2 = pipeline2.run_advanced_pipeline()

            best_acc2 = 0
            for model_name, metrics in result2.get('model_results', {}).items():
                if 'direction_accuracy' in metrics:
                    acc = metrics['direction_accuracy'] / 100.0
                    if acc > best_acc2:
                        best_acc2 = acc

            results.append({
                'config': '시퀀스15_2fold',
                'accuracy': best_acc2,
                'elapsed': time.time() - start_time
            })

            print(f"   ✅ 결과: {best_acc2:.4f}")
            print(f"   ⏱️ 누적시간: {time.time() - start_time:.1f}초")

    except Exception as e:
        print(f"❌ 테스트 중 오류: {str(e)}")

    # 결과 요약
    total_time = time.time() - start_time
    print(f"\n📊 빠른 테스트 결과 요약:")
    print(f"   총 소요시간: {total_time:.1f}초")

    if results:
        best_result = max(results, key=lambda x: x['accuracy'])
        print(f"   최고 성능: {best_result['accuracy']:.4f} ({best_result['config']})")

        # 기준선 대비
        baseline = 0.8482
        improvement = (best_result['accuracy'] - baseline) * 100
        print(f"   기준선 대비: {improvement:+.2f}%p")

    # 결과 저장
    final_result = {
        'timestamp': datetime.now().isoformat(),
        'experiment_type': 'fast_performance_test_10min',
        'total_time_seconds': total_time,
        'time_limit_seconds': 600,
        'completed_tests': len(results),
        'results': results,
        'best_accuracy': best_result['accuracy'] if results else 0,
        'baseline_accuracy': 0.8482,
        'improvement': improvement if results else 0
    }

    output_path = f"/root/workspace/data/results/fast_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, 'w') as f:
        json.dump(final_result, f, indent=2)

    print(f"💾 결과 저장: {output_path}")
    print(f"🏁 빠른 테스트 완료 ({total_time:.1f}초)")

    return final_result

if __name__ == "__main__":
    quick_performance_test()