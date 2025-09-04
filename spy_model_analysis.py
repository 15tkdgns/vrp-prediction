#!/usr/bin/env python3
"""
SPY 모델 결과 분석 및 시각화
학습된 모델의 성능을 종합적으로 분석하고 시각화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix)

plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)

class SPYModelAnalyzer:
    """SPY 모델 분석 클래스"""
    
    def __init__(self, model_files_dict, log_file):
        """
        Args:
            model_files_dict: 모델 파일 경로 딕셔너리
            log_file: 학습 로그 파일 경로
        """
        self.model_files = model_files_dict
        self.log_file = log_file
        
        # 데이터 로드
        self.load_training_log()
        self.load_models_and_pipeline()
        
        # 결과 저장 폴더
        self.results_dir = "analysis_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def load_training_log(self):
        """학습 로그 로드"""
        with open(self.log_file, 'r') as f:
            self.training_log = json.load(f)
        
        print("=== 학습 로그 정보 ===")
        print(f"학습 기간: {self.training_log.get('start_time')} ~ {self.training_log.get('end_time')}")
        print(f"최종 테스트 성능: {self.training_log['final_test_performance']['metrics']}")
        
    def load_models_and_pipeline(self):
        """모델 및 파이프라인 로드"""
        # 모델들 로드
        with open(self.model_files['models'], 'rb') as f:
            self.best_models = pickle.load(f)
            
        # 앙상블 모델 로드
        if self.model_files.get('ensemble'):
            with open(self.model_files['ensemble'], 'rb') as f:
                self.ensemble_model = pickle.load(f)
        else:
            self.ensemble_model = None
            
        # 파이프라인 객체 로드
        with open(self.model_files['pipeline'], 'rb') as f:
            self.pipeline_objects = pickle.load(f)
            
        print("=== 로드된 모델 정보 ===")
        print(f"개별 모델 수: {len(self.best_models)}")
        print(f"모델 목록: {list(self.best_models.keys())}")
        print(f"앙상블 모델: {'있음' if self.ensemble_model else '없음'}")
        
    def analyze_cross_validation_results(self):
        """교차 검증 결과 분석"""
        print("\n=== 교차 검증 결과 분석 ===")
        
        cv_results = self.training_log['cross_validation_results']
        
        # 성능 비교 테이블
        performance_data = []
        for model_name, results in cv_results.items():
            if 'error' not in results:
                performance_data.append({
                    'Model': model_name,
                    'CV_Score': results['best_score'],
                    'CV_Std': results['cv_std'],
                    'Score_Range': f"{results['best_score'] - results['cv_std']:.4f} - {results['best_score'] + results['cv_std']:.4f}"
                })
        
        cv_df = pd.DataFrame(performance_data).sort_values('CV_Score', ascending=False)
        print(cv_df.to_string(index=False))
        
        # 시각화
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # CV 점수 비교
        models = cv_df['Model']
        scores = cv_df['CV_Score']
        errors = cv_df['CV_Std']
        
        bars = ax1.bar(models, scores, yerr=errors, capsize=5, alpha=0.7)
        ax1.set_title('교차 검증 ROC AUC 성능 비교', fontsize=14, pad=20)
        ax1.set_ylabel('ROC AUC Score')
        ax1.tick_params(axis='x', rotation=45)
        ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Random Baseline')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 색상 그라데이션
        colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # 모델별 파라미터 복잡도
        model_complexity = []
        for model_name in models:
            if model_name in self.training_log['cross_validation_results']:
                best_params = self.training_log['cross_validation_results'][model_name]['best_params']
                # 파라미터 수를 복잡도 지표로 사용
                complexity = len(best_params)
                model_complexity.append(complexity)
        
        if model_complexity:
            scatter = ax2.scatter(model_complexity, scores, s=100, alpha=0.7, c=colors)
            ax2.set_xlabel('Model Complexity (# parameters)')
            ax2.set_ylabel('CV ROC AUC Score')
            ax2.set_title('모델 복잡도 vs 성능', fontsize=14, pad=20)
            
            # 모델 이름 라벨
            for i, model in enumerate(models):
                ax2.annotate(model, (model_complexity[i], scores.iloc[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/cross_validation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return cv_df
    
    def analyze_validation_performance(self):
        """검증 세트 성능 분석"""
        print("\n=== 검증 세트 성능 분석 ===")
        
        val_results = self.training_log['validation_performance']
        
        # 성능 메트릭 테이블
        metrics_data = []
        for model_name, metrics in val_results.items():
            if 'error' not in metrics:
                metrics_data.append({
                    'Model': model_name,
                    'Accuracy': metrics.get('accuracy', 0),
                    'Precision': metrics.get('precision', 0),
                    'Recall': metrics.get('recall', 0),
                    'F1': metrics.get('f1', 0),
                    'ROC_AUC': metrics.get('roc_auc', 0)
                })
        
        metrics_df = pd.DataFrame(metrics_data)
        print(metrics_df.round(4).to_string(index=False))
        
        # 히트맵으로 성능 비교
        plt.figure(figsize=(12, 8))
        
        # 메트릭 데이터 준비 (모델별로 정규화)
        heatmap_data = metrics_df.set_index('Model').T
        
        # 히트맵 생성
        sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', center=0.5, 
                   fmt='.3f', linewidths=0.5, cbar_kws={'label': 'Score'})
        plt.title('검증 세트 성능 히트맵', fontsize=16, pad=20)
        plt.ylabel('성능 지표')
        plt.xlabel('모델')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/validation_performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return metrics_df
    
    def analyze_feature_importance(self):
        """특성 중요도 분석"""
        print("\n=== 특성 중요도 분석 ===")
        
        importance_data = self.training_log['feature_importance']
        
        # 앙상블 평균 중요도 시각화
        if 'ensemble_mean' in importance_data:
            ensemble_features = importance_data['ensemble_mean']['top_features']
            
            # 데이터 준비
            features = [f[0] for f in ensemble_features]
            importances = [f[1] for f in ensemble_features]
            
            # 상위 15개 특성만 시각화
            top_n = min(15, len(features))
            features = features[:top_n]
            importances = importances[:top_n]
            
            # 수평 막대 그래프
            plt.figure(figsize=(12, 8))
            colors = plt.cm.plasma(np.linspace(0, 1, len(features)))
            bars = plt.barh(range(len(features)), importances, color=colors)
            
            plt.yticks(range(len(features)), features)
            plt.xlabel('Feature Importance')
            plt.title('앙상블 모델 특성 중요도 (상위 15개)', fontsize=16, pad=20)
            plt.gca().invert_yaxis()  # 가장 중요한 특성을 위쪽에
            
            # 값 표시
            for i, (bar, imp) in enumerate(zip(bars, importances)):
                plt.text(imp + max(importances)*0.01, i, f'{imp:.4f}', 
                        va='center', fontsize=9)
            
            plt.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            plt.savefig(f'{self.results_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # 특성 카테고리별 중요도 분석
            self._analyze_feature_categories(features, importances)
    
    def _analyze_feature_categories(self, features, importances):
        """특성 카테고리별 중요도 분석"""
        
        # 특성 카테고리 정의
        categories = {
            'Volume': ['Volume', 'volume'],
            'Returns': ['Return', 'return'],
            'Price_Ratios': ['Ratio', 'ratio'],
            'Technical_Indicators': ['RSI', 'MACD', 'BB_', 'ATR', 'Stoch', 'Williams'],
            'Moving_Averages': ['SMA', 'sma'],
            'Lags': ['Lag'],
            'Time': ['Day', 'Month', 'Quarter', 'Is']
        }
        
        # 각 특성을 카테고리로 분류
        feature_categories = {}
        for feature, importance in zip(features, importances):
            categorized = False
            for category, keywords in categories.items():
                if any(keyword in feature for keyword in keywords):
                    if category not in feature_categories:
                        feature_categories[category] = []
                    feature_categories[category].append(importance)
                    categorized = True
                    break
            
            if not categorized:
                if 'Others' not in feature_categories:
                    feature_categories['Others'] = []
                feature_categories['Others'].append(importance)
        
        # 카테고리별 평균 중요도
        category_importance = {cat: np.mean(imps) for cat, imps in feature_categories.items()}
        
        # 파이 차트로 시각화
        plt.figure(figsize=(10, 8))
        categories = list(category_importance.keys())
        sizes = list(category_importance.values())
        colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
        
        wedges, texts, autotexts = plt.pie(sizes, labels=categories, autopct='%1.1f%%', 
                                         colors=colors, startangle=90)
        
        plt.title('특성 카테고리별 평균 중요도', fontsize=16, pad=20)
        plt.axis('equal')
        
        # 범례 추가
        plt.legend(wedges, [f'{cat}: {imp:.4f}' for cat, imp in category_importance.items()],
                  title="Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/feature_categories.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_final_test_performance(self):
        """최종 테스트 성능 분석"""
        print("\n=== 최종 테스트 성능 분석 ===")
        
        test_results = self.training_log['final_test_performance']
        
        # 기본 정보
        print(f"테스트 기간: {test_results['test_period']}")
        print(f"테스트 샘플 수: {test_results['test_samples']}")
        print(f"모델 유형: {test_results['model_type']}")
        
        # 성능 지표
        metrics = test_results['metrics']
        print("\n성능 지표:")
        for metric, value in metrics.items():
            print(f"  {metric.upper()}: {value:.4f}")
        
        # 혼동 행렬 시각화
        conf_matrix = np.array(test_results['confusion_matrix'])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Predicted: Down', 'Predicted: Up'],
                   yticklabels=['Actual: Down', 'Actual: Up'])
        plt.title('테스트 세트 혼동 행렬', fontsize=16, pad=20)
        plt.ylabel('실제 레이블')
        plt.xlabel('예측 레이블')
        
        # 정확도 정보 추가
        total = conf_matrix.sum()
        accuracy = np.trace(conf_matrix) / total
        plt.text(0.5, -0.1, f'정확도: {accuracy:.4f} ({np.trace(conf_matrix)}/{total})',
                transform=plt.gca().transAxes, ha='center', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 분류 리포트 시각화
        class_report = test_results['classification_report']
        if '0' in class_report and '1' in class_report:
            
            report_df = pd.DataFrame({
                'Down (0)': class_report['0'],
                'Up (1)': class_report['1']
            }).T
            
            plt.figure(figsize=(12, 6))
            sns.heatmap(report_df[['precision', 'recall', 'f1-score']], 
                       annot=True, cmap='RdYlBu', center=0.5, fmt='.3f')
            plt.title('클래스별 성능 지표', fontsize=16, pad=20)
            plt.ylabel('클래스')
            plt.xlabel('지표')
            plt.tight_layout()
            plt.savefig(f'{self.results_dir}/classification_report.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def create_comprehensive_report(self):
        """종합 보고서 생성"""
        print("\n=== 종합 보고서 생성 ===")
        
        # 마크다운 보고서 생성
        report = f"""# SPY 머신러닝 모델 분석 보고서

## 실험 개요
- **학습 시작**: {self.training_log['start_time']}
- **학습 완료**: {self.training_log['end_time']}
- **총 특성 수**: {self.training_log.get('feature_engineering', {}).get('total_features', 'N/A')}
- **선택된 특성 수**: {self.training_log.get('feature_selection', {}).get('selected_features', 'N/A')}
- **모델 수**: {len(self.training_log['cross_validation_results'])}

## 데이터 정보
- **총 샘플 수**: {self.training_log.get('data_split', {}).get('total_samples', 'N/A')}
- **훈련 샘플**: {self.training_log.get('data_split', {}).get('train_samples', 'N/A')} ({self.training_log.get('data_split', {}).get('train_period', 'N/A')})
- **검증 샘플**: {self.training_log.get('data_split', {}).get('val_samples', 'N/A')} ({self.training_log.get('data_split', {}).get('val_period', 'N/A')})
- **테스트 샘플**: {self.training_log.get('data_split', {}).get('test_samples', 'N/A')} ({self.training_log.get('data_split', {}).get('test_period', 'N/A')})

## 교차 검증 결과
"""
        
        # CV 결과 추가
        cv_results = self.training_log['cross_validation_results']
        sorted_models = sorted([(name, results['best_score']) for name, results in cv_results.items() 
                               if 'error' not in results], key=lambda x: x[1], reverse=True)
        
        for i, (model_name, score) in enumerate(sorted_models, 1):
            std = cv_results[model_name]['cv_std']
            report += f"{i}. **{model_name}**: {score:.4f} (±{std:.4f})\\n"
        
        # 최종 성능 추가
        final_metrics = self.training_log['final_test_performance']['metrics']
        report += f"""
## 최종 테스트 성능
- **정확도**: {final_metrics['accuracy']:.4f}
- **정밀도**: {final_metrics['precision']:.4f}
- **재현율**: {final_metrics['recall']:.4f}
- **F1 스코어**: {final_metrics['f1']:.4f}
- **ROC AUC**: {final_metrics['roc_auc']:.4f}

## 주요 특성 (상위 5개)
"""
        
        # 특성 중요도 추가
        if 'ensemble_mean' in self.training_log['feature_importance']:
            top_features = self.training_log['feature_importance']['ensemble_mean']['top_features'][:5]
            for i, (feature, importance) in enumerate(top_features, 1):
                report += f"{i}. **{feature}**: {importance:.4f}\\n"
        
        report += f"""
## 결론 및 권장사항

### 모델 성능 평가
현재 모델의 ROC AUC는 {final_metrics['roc_auc']:.4f}로, 랜덤 예측(0.5)와 비슷한 수준입니다. 
이는 주식 시장의 본질적인 불확실성을 반영하는 결과로 해석됩니다.

### 개선 방안
1. **더 많은 외부 데이터 활용**: 뉴스, 경제 지표, 감정 분석 등
2. **딥러닝 모델 적용**: LSTM, GRU, Transformer 등
3. **특성 엔지니어링 개선**: 더 복잡한 기술적 지표, 시간 기반 특성
4. **앙상블 방법 개선**: 다양한 투표 방식, 메타 학습기 활용
5. **예측 기간 조정**: 단기(1일) 대신 중장기 예측 시도

### 데이터 누출 및 오버피팅 방지
- ✅ 시간 순서 기반 데이터 분할
- ✅ 교차 검증 적용 (TimeSeriesSplit)
- ✅ 정규화 및 특성 선택
- ✅ 앙상블 모델링

보고서 생성일: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        # 보고서 저장
        with open(f'{self.results_dir}/comprehensive_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"종합 보고서 저장: {self.results_dir}/comprehensive_report.md")
        
        return report

def main():
    """메인 실행 함수"""
    
    # 최신 모델 파일들 찾기
    models_dir = "models"
    model_files = {}
    log_file = None
    
    # 파일 목록에서 최신 파일들 찾기
    all_files = os.listdir(models_dir)
    
    # 가장 최근 타임스탬프 찾기
    timestamps = []
    for file in all_files:
        if 'spy_' in file and '.pkl' in file:
            parts = file.split('_')
            if len(parts) >= 3:
                timestamp = '_'.join(parts[-2:]).replace('.pkl', '').replace('.json', '')
                timestamps.append(timestamp)
    
    if not timestamps:
        print("모델 파일을 찾을 수 없습니다.")
        return
    
    latest_timestamp = sorted(timestamps)[-1]
    print(f"분석할 모델 타임스탬프: {latest_timestamp}")
    
    # 파일 경로 설정
    model_files['models'] = f"{models_dir}/spy_best_models_{latest_timestamp}.pkl"
    model_files['ensemble'] = f"{models_dir}/spy_ensemble_model_{latest_timestamp}.pkl"
    model_files['pipeline'] = f"{models_dir}/spy_pipeline_{latest_timestamp}.pkl"
    log_file = f"{models_dir}/spy_training_log_{latest_timestamp}.json"
    
    # 파일 존재 확인
    for key, path in model_files.items():
        if not os.path.exists(path):
            print(f"파일을 찾을 수 없습니다: {path}")
            if key == 'ensemble':
                model_files[key] = None  # 앙상블 파일은 선택사항
            else:
                return
    
    if not os.path.exists(log_file):
        print(f"로그 파일을 찾을 수 없습니다: {log_file}")
        return
    
    # 분석 실행
    analyzer = SPYModelAnalyzer(model_files, log_file)
    
    # 각종 분석 수행
    cv_df = analyzer.analyze_cross_validation_results()
    metrics_df = analyzer.analyze_validation_performance()
    analyzer.analyze_feature_importance()
    analyzer.analyze_final_test_performance()
    
    # 종합 보고서 생성
    report = analyzer.create_comprehensive_report()
    
    print(f"\n=== 분석 완료 ===")
    print(f"결과 저장 폴더: {analyzer.results_dir}")
    print("생성된 파일들:")
    print("- cross_validation_analysis.png")
    print("- validation_performance_heatmap.png")
    print("- feature_importance.png")
    print("- feature_categories.png")
    print("- confusion_matrix.png")
    print("- classification_report.png")
    print("- comprehensive_report.md")

if __name__ == "__main__":
    main()