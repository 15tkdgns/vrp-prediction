#!/usr/bin/env python3
"""
🏆 종합 성능 비교 및 최적 모델 선정 시스템

모든 실험 결과를 수집하고 종합적으로 분석하여 최적 모델 선정
"""

import sys
sys.path.append('/root/workspace/src')

import pandas as pd
import numpy as np
import json
import glob
from datetime import datetime
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ComprehensivePerformanceComparator:
    """종합 성능 비교 시스템"""

    def __init__(self):
        self.results_dir = '/root/workspace/data/results'
        self.experiments_summary = {}
        self.safety_criteria = {
            'max_r2': 0.25,           # R² 상한선
            'max_direction_acc': 70.0, # 방향정확도 상한선 (%)
            'max_correlation': 0.3     # 특성-타겟 상관관계 상한선
        }

        print(f"🏆 종합 성능 비교 및 최적 모델 선정 시스템")
        print(f"   📊 안전 기준: R²<{self.safety_criteria['max_r2']}, 방향정확도<{self.safety_criteria['max_direction_acc']}%")

    def load_experiment_results(self):
        """모든 실험 결과 파일 로딩"""
        print("📂 실험 결과 파일 로딩...")

        experiment_files = []

        # JSON 결과 파일들 찾기
        json_patterns = [
            'ultra_safe_no_leak_results_*.json',
            'ultra_safe_hyperparameter_tuning_*.json',
            'safe_neural_networks_*.json',
            'safe_ensemble_methods_*.json',
            'advanced_leak_free_regression_*.json',
            'comprehensive_regression_results_*.json'
        ]

        for pattern in json_patterns:
            files = glob.glob(os.path.join(self.results_dir, pattern))
            experiment_files.extend(files)

        print(f"   발견된 실험 파일: {len(experiment_files)}개")

        # 파일별로 로딩
        all_results = {}

        for file_path in experiment_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                    # 파일명에서 실험 타입 추출
                    file_name = os.path.basename(file_path)
                    experiment_type = data.get('experiment_type', file_name.split('_')[0])

                    all_results[file_name] = {
                        'experiment_type': experiment_type,
                        'timestamp': data.get('timestamp', 'unknown'),
                        'results': data.get('results', {}),
                        'file_path': file_path
                    }

                print(f"   ✅ {file_name}: {len(data.get('results', {}))}개 모델")

            except Exception as e:
                print(f"   ❌ {file_path} 로딩 실패: {e}")

        # 기존 성능 데이터도 로딩
        self._load_existing_performance_data(all_results)

        return all_results

    def _load_existing_performance_data(self, all_results):
        """기존 성능 데이터 로딩"""

        # model_performance.json 로딩
        performance_file = '/root/workspace/data/raw/model_performance.json'
        if os.path.exists(performance_file):
            try:
                with open(performance_file, 'r') as f:
                    data = json.load(f)
                    all_results['model_performance.json'] = {
                        'experiment_type': 'existing_models',
                        'timestamp': data.get('timestamp', 'existing'),
                        'results': data.get('models', {}),
                        'file_path': performance_file
                    }
                print(f"   ✅ 기존 성능 데이터: {len(data.get('models', {}))}개 모델")
            except Exception as e:
                print(f"   ❌ 기존 성능 데이터 로딩 실패: {e}")

        # validation_report.json 로딩
        validation_file = '/root/workspace/data/raw/validation_report.json'
        if os.path.exists(validation_file):
            try:
                with open(validation_file, 'r') as f:
                    data = json.load(f)
                    # validation_report의 구조에 맞게 파싱
                    validation_results = {}
                    if 'results' in data:
                        for model_name, metrics in data['results'].items():
                            validation_results[f"{model_name}_validated"] = metrics

                    all_results['validation_report.json'] = {
                        'experiment_type': 'validation_results',
                        'timestamp': data.get('timestamp', 'validation'),
                        'results': validation_results,
                        'file_path': validation_file
                    }
                print(f"   ✅ 검증 데이터: {len(validation_results)}개 모델")
            except Exception as e:
                print(f"   ❌ 검증 데이터 로딩 실패: {e}")

    def normalize_model_metrics(self, results_dict):
        """모델 메트릭 정규화"""
        print("🔧 모델 메트릭 정규화...")

        normalized_models = {}

        for file_name, file_data in results_dict.items():
            experiment_type = file_data['experiment_type']
            timestamp = file_data['timestamp']

            for model_name, metrics in file_data['results'].items():

                # 정규화된 모델명 생성
                normalized_name = f"{experiment_type}_{model_name}"

                # 메트릭 정규화
                normalized_metrics = {
                    'experiment_type': experiment_type,
                    'timestamp': timestamp,
                    'file_source': file_name,
                    'direction_accuracy': None,
                    'mae': None,
                    'r2': None,
                    'mse': None,
                    'fold_accuracies': None,
                    'safety_status': 'unknown'
                }

                # 다양한 메트릭 형식 처리
                if isinstance(metrics, dict):
                    # 방향 정확도 추출
                    for key in ['direction_accuracy', 'avg_direction_accuracy', 'accuracy', 'best_direction_accuracy']:
                        if key in metrics and metrics[key] is not None:
                            normalized_metrics['direction_accuracy'] = float(metrics[key])
                            if normalized_metrics['direction_accuracy'] > 1:  # 백분율을 소수로 변환
                                normalized_metrics['direction_accuracy'] /= 100
                            break

                    # MAE 추출
                    for key in ['mae', 'avg_mae', 'mean_absolute_error']:
                        if key in metrics and metrics[key] is not None:
                            normalized_metrics['mae'] = float(metrics[key])
                            break

                    # R² 추출
                    for key in ['r2', 'avg_r2', 'r2_score', 'r_squared']:
                        if key in metrics and metrics[key] is not None:
                            normalized_metrics['r2'] = float(metrics[key])
                            break

                    # MSE 추출
                    for key in ['mse', 'avg_mse', 'mean_squared_error']:
                        if key in metrics and metrics[key] is not None:
                            normalized_metrics['mse'] = float(metrics[key])
                            break

                    # Fold accuracies 추출
                    for key in ['fold_accuracies', 'cv_accuracies']:
                        if key in metrics and metrics[key] is not None:
                            normalized_metrics['fold_accuracies'] = metrics[key]
                            break

                    # 안전성 상태 평가
                    normalized_metrics['safety_status'] = self._evaluate_safety(normalized_metrics)

                normalized_models[normalized_name] = normalized_metrics

        print(f"   ✅ 정규화된 모델: {len(normalized_models)}개")
        return normalized_models

    def _evaluate_safety(self, metrics):
        """모델 안전성 평가"""
        direction_acc = metrics.get('direction_accuracy')
        r2 = metrics.get('r2')

        if direction_acc is None and r2 is None:
            return 'insufficient_data'

        unsafe_reasons = []

        # 방향 정확도 검사
        if direction_acc is not None:
            if direction_acc > 0.95:
                unsafe_reasons.append('extreme_accuracy')
            elif direction_acc > (self.safety_criteria['max_direction_acc'] / 100):
                unsafe_reasons.append('high_accuracy')

        # R² 검사
        if r2 is not None:
            if r2 > 0.9:
                unsafe_reasons.append('extreme_r2')
            elif r2 > self.safety_criteria['max_r2']:
                unsafe_reasons.append('high_r2')

        if unsafe_reasons:
            return f"unsafe_{','.join(unsafe_reasons)}"
        else:
            return 'safe'

    def create_comprehensive_ranking(self, normalized_models):
        """종합 순위 생성"""
        print("🏆 종합 순위 생성...")

        # 안전한 모델들만 필터링
        safe_models = {k: v for k, v in normalized_models.items()
                      if v['safety_status'] == 'safe'}

        print(f"   안전한 모델: {len(safe_models)}개")
        print(f"   위험한 모델: {len(normalized_models) - len(safe_models)}개")

        if not safe_models:
            print("   ⚠️ 안전한 모델이 없습니다!")
            return []

        # 성능 점수 계산
        ranked_models = []

        for model_name, metrics in safe_models.items():
            score = self._calculate_composite_score(metrics)

            ranked_models.append({
                'model_name': model_name,
                'composite_score': score,
                **metrics
            })

        # 점수 기준 정렬
        ranked_models.sort(key=lambda x: x['composite_score'], reverse=True)

        return ranked_models

    def _calculate_composite_score(self, metrics):
        """종합 점수 계산"""
        score = 0.0
        weight_sum = 0.0

        # 방향 정확도 점수 (가중치: 0.5)
        direction_acc = metrics.get('direction_accuracy')
        if direction_acc is not None:
            # 50%를 기준으로 점수 계산 (50% = 0점, 70% = 100점)
            direction_score = max(0, (direction_acc - 0.5) / 0.2 * 100)
            score += direction_score * 0.5
            weight_sum += 0.5

        # R² 점수 (가중치: 0.3)
        r2 = metrics.get('r2')
        if r2 is not None:
            # R²: 음수면 0점, 0.25면 100점
            r2_score = max(0, min(100, (r2 + 0.1) / 0.35 * 100))
            score += r2_score * 0.3
            weight_sum += 0.3

        # MAE 점수 (가중치: 0.2) - 낮을수록 좋음
        mae = metrics.get('mae')
        if mae is not None:
            # MAE: 1.0이면 0점, 0.3이면 100점
            mae_score = max(0, min(100, (1.0 - mae) / 0.7 * 100))
            score += mae_score * 0.2
            weight_sum += 0.2

        # 정규화
        if weight_sum > 0:
            score /= weight_sum

        return score

    def generate_comprehensive_report(self, ranked_models, all_results):
        """종합 보고서 생성"""
        print("📋 종합 보고서 생성...")

        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_models_analyzed': len(all_results),
                'safe_models': len([m for m in ranked_models if m['safety_status'] == 'safe']),
                'unsafe_models': len(all_results) - len([m for m in ranked_models if m['safety_status'] == 'safe']),
                'best_model': ranked_models[0]['model_name'] if ranked_models else None,
                'best_score': ranked_models[0]['composite_score'] if ranked_models else None
            },
            'safety_criteria': self.safety_criteria,
            'ranking': ranked_models[:10],  # 상위 10개 모델
            'experiment_summary': self._create_experiment_summary(ranked_models)
        }

        return report

    def _create_experiment_summary(self, ranked_models):
        """실험별 요약 생성"""
        experiment_summary = {}

        for model in ranked_models:
            exp_type = model['experiment_type']
            if exp_type not in experiment_summary:
                experiment_summary[exp_type] = {
                    'total_models': 0,
                    'best_model': None,
                    'best_score': 0,
                    'avg_direction_accuracy': 0,
                    'avg_r2': 0
                }

            experiment_summary[exp_type]['total_models'] += 1

            if model['composite_score'] > experiment_summary[exp_type]['best_score']:
                experiment_summary[exp_type]['best_model'] = model['model_name']
                experiment_summary[exp_type]['best_score'] = model['composite_score']

        # 평균 계산
        for exp_type in experiment_summary:
            exp_models = [m for m in ranked_models if m['experiment_type'] == exp_type]

            direction_accs = [m['direction_accuracy'] for m in exp_models if m['direction_accuracy'] is not None]
            if direction_accs:
                experiment_summary[exp_type]['avg_direction_accuracy'] = np.mean(direction_accs)

            r2s = [m['r2'] for m in exp_models if m['r2'] is not None]
            if r2s:
                experiment_summary[exp_type]['avg_r2'] = np.mean(r2s)

        return experiment_summary

    def run_comprehensive_analysis(self):
        """종합 분석 실행"""
        print("🔍 종합 성능 분석 시작")
        print("="*70)

        try:
            # 1. 모든 실험 결과 로딩
            all_results = self.load_experiment_results()

            if not all_results:
                print("❌ 분석할 실험 결과가 없습니다!")
                return None

            # 2. 메트릭 정규화
            normalized_models = self.normalize_model_metrics(all_results)

            # 3. 종합 순위 생성
            ranked_models = self.create_comprehensive_ranking(normalized_models)

            # 4. 보고서 생성
            report = self.generate_comprehensive_report(ranked_models, normalized_models)

            # 5. 결과 출력
            self._print_analysis_results(report)

            # 6. 결과 저장
            self._save_analysis_results(report)

            return report

        except Exception as e:
            print(f"❌ 종합 분석 실패: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def _print_analysis_results(self, report):
        """분석 결과 출력"""
        print(f"\n🏆 종합 성능 분석 결과")
        print("="*70)

        summary = report['summary']
        print(f"📊 분석 요약:")
        print(f"   총 분석 모델: {summary['total_models_analyzed']}개")
        print(f"   안전한 모델: {summary['safe_models']}개")
        print(f"   위험한 모델: {summary['unsafe_models']}개")

        if summary['best_model']:
            print(f"   🥇 최고 모델: {summary['best_model']}")
            print(f"   🎯 최고 점수: {summary['best_score']:.2f}")

        print(f"\n🏅 상위 모델 순위:")
        print("-" * 70)

        for i, model in enumerate(report['ranking'][:5], 1):
            direction_acc = model.get('direction_accuracy', 0)
            r2 = model.get('r2', 0)
            mae = model.get('mae', 0)

            print(f"{i:2d}. {model['model_name'][:40]:<40}")
            print(f"     종합점수: {model['composite_score']:.2f} | "
                  f"방향정확도: {direction_acc:.1%} | "
                  f"R²: {r2:.4f} | "
                  f"MAE: {mae:.4f}")

        print(f"\n📈 실험별 요약:")
        print("-" * 70)

        for exp_type, summary in report['experiment_summary'].items():
            print(f"{exp_type}:")
            print(f"   모델 수: {summary['total_models']}개")
            print(f"   최고 모델: {summary['best_model']}")
            print(f"   평균 방향정확도: {summary['avg_direction_accuracy']:.1%}")
            print(f"   평균 R²: {summary['avg_r2']:.4f}")

    def _save_analysis_results(self, report):
        """분석 결과 저장"""
        output_path = f"/root/workspace/data/results/comprehensive_performance_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\n💾 종합 분석 결과 저장: {output_path}")
        except Exception as e:
            print(f"❌ 결과 저장 실패: {e}")

        # 마크다운 보고서도 생성
        self._generate_markdown_report(report)

    def _generate_markdown_report(self, report):
        """마크다운 보고서 생성"""
        output_path = f"/root/workspace/comprehensive_performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("# 🏆 종합 모델 성능 분석 보고서\n\n")

                f.write("## 📊 분석 요약\n\n")
                summary = report['summary']
                f.write(f"- **총 분석 모델**: {summary['total_models_analyzed']}개\n")
                f.write(f"- **안전한 모델**: {summary['safe_models']}개\n")
                f.write(f"- **위험한 모델**: {summary['unsafe_models']}개\n")

                if summary['best_model']:
                    f.write(f"- **🥇 최고 모델**: {summary['best_model']}\n")
                    f.write(f"- **🎯 최고 점수**: {summary['best_score']:.2f}\n")

                f.write("\n## 🏅 상위 모델 순위\n\n")
                f.write("| 순위 | 모델명 | 종합점수 | 방향정확도 | R² | MAE | 실험타입 |\n")
                f.write("|------|--------|----------|-----------|----|----|----------|\n")

                for i, model in enumerate(report['ranking'][:10], 1):
                    direction_acc = model.get('direction_accuracy', 0)
                    r2 = model.get('r2', 0)
                    mae = model.get('mae', 0)
                    exp_type = model.get('experiment_type', 'unknown')

                    f.write(f"| {i} | {model['model_name'][:30]} | {model['composite_score']:.2f} | "
                           f"{direction_acc:.1%} | {r2:.4f} | {mae:.4f} | {exp_type} |\n")

                f.write("\n## 📈 실험별 요약\n\n")

                for exp_type, exp_summary in report['experiment_summary'].items():
                    f.write(f"### {exp_type}\n\n")
                    f.write(f"- **모델 수**: {exp_summary['total_models']}개\n")
                    f.write(f"- **최고 모델**: {exp_summary['best_model']}\n")
                    f.write(f"- **평균 방향정확도**: {exp_summary['avg_direction_accuracy']:.1%}\n")
                    f.write(f"- **평균 R²**: {exp_summary['avg_r2']:.4f}\n\n")

                f.write(f"\n## 📋 분석 시간\n\n")
                f.write(f"**생성 시간**: {report['timestamp']}\n")

            print(f"📄 마크다운 보고서 저장: {output_path}")

        except Exception as e:
            print(f"❌ 마크다운 보고서 저장 실패: {e}")

def main():
    """메인 실행"""
    comparator = ComprehensivePerformanceComparator()
    report = comparator.run_comprehensive_analysis()

    if report:
        print("\n🎉 종합 성능 분석 완료!")
        print("✅ 모든 모델이 안전성 검증을 거쳐 순위가 매겨졌습니다.")
    else:
        print("\n❌ 종합 성능 분석 실패!")

    return report

if __name__ == "__main__":
    main()