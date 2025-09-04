#!/usr/bin/env python3
"""
뉴스 감정분석 포함/미포함 모델 성능 비교
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelComparison:
    """뉴스 감정분석 포함/미포함 모델 성능 비교 클래스"""
    
    def __init__(self, data_file: str = "data/raw/integrated_spy_news_data.csv"):
        self.data_file = data_file
        self.data = None
        self.baseline_models = {}
        self.enhanced_models = {}
        self.results = {}
        
        # 특성 그룹 정의
        self.technical_features = None
        self.news_features = None
        self.all_features = None
        
    def load_data(self) -> pd.DataFrame:
        """통합 데이터 로드"""
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"Data file not found: {self.data_file}")
            
        logger.info(f"Loading integrated data from {self.data_file}")
        self.data = pd.read_csv(self.data_file)
        
        # 날짜 컬럼 처리
        self.data['date'] = pd.to_datetime(self.data['date'])
        
        # 특성 그룹 정의
        self._define_feature_groups()
        
        logger.info(f"Loaded {len(self.data)} records with {len(self.data.columns)} features")
        return self.data
    
    def _define_feature_groups(self):
        """특성 그룹 정의"""
        # 제외할 컬럼들
        exclude_cols = ['date', 'target', 'next_day_return', 'open', 'high', 'low', 'close', 'volume']
        
        # 뉴스 관련 특성
        news_keywords = ['sentiment', 'news', 'article', 'impact']
        self.news_features = [col for col in self.data.columns 
                             if any(keyword in col.lower() for keyword in news_keywords) 
                             and col not in exclude_cols]
        
        # 기술적 지표 특성 (뉴스 특성을 제외한 나머지)
        self.technical_features = [col for col in self.data.columns 
                                  if col not in exclude_cols 
                                  and col not in self.news_features]
        
        # 전체 특성
        self.all_features = self.technical_features + self.news_features
        
        logger.info(f"Technical features ({len(self.technical_features)}): {self.technical_features}")
        logger.info(f"News features ({len(self.news_features)}): {self.news_features}")
    
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """데이터 준비 및 분할"""
        if self.data is None:
            self.load_data()
        
        # 결측값 처리
        self.data = self.data.fillna(self.data.mean(numeric_only=True))
        
        X = self.data[self.all_features]
        y = self.data['target']
        
        # 시계열 데이터이므로 시간 순서 유지하여 분할
        # 처음 70%는 훈련용, 나머지 30%는 테스트용
        split_idx = int(len(self.data) * 0.7)
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        logger.info(f"Train period: {self.data['date'].iloc[0]} ~ {self.data['date'].iloc[split_idx-1]}")
        logger.info(f"Test period: {self.data['date'].iloc[split_idx]} ~ {self.data['date'].iloc[-1]}")
        
        return X_train, X_test, y_train, y_test
    
    def train_models(self):
        """베이스라인 모델과 향상된 모델 훈련"""
        logger.info("🚀 모델 훈련 시작")
        
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        # 모델 정의
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=6),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        # 1. 베이스라인 모델 (기술적 지표만)
        logger.info("📊 베이스라인 모델 훈련 (기술적 지표만)")
        X_train_technical = X_train[self.technical_features]
        X_test_technical = X_test[self.technical_features]
        
        # 스케일링
        scaler_baseline = StandardScaler()
        X_train_technical_scaled = scaler_baseline.fit_transform(X_train_technical)
        X_test_technical_scaled = scaler_baseline.transform(X_test_technical)
        
        for name, model in models.items():
            logger.info(f"Training baseline {name}...")
            model_copy = type(model)(**model.get_params())
            model_copy.fit(X_train_technical_scaled, y_train)
            self.baseline_models[name] = {
                'model': model_copy,
                'scaler': scaler_baseline,
                'features': self.technical_features
            }
        
        # 2. 향상된 모델 (기술적 지표 + 뉴스)
        logger.info("📈 향상된 모델 훈련 (기술적 지표 + 뉴스)")
        
        # 스케일링
        scaler_enhanced = StandardScaler()
        X_train_scaled = scaler_enhanced.fit_transform(X_train)
        X_test_scaled = scaler_enhanced.transform(X_test)
        
        for name, model in models.items():
            logger.info(f"Training enhanced {name}...")
            model_copy = type(model)(**model.get_params())
            model_copy.fit(X_train_scaled, y_train)
            self.enhanced_models[name] = {
                'model': model_copy,
                'scaler': scaler_enhanced,
                'features': self.all_features
            }
        
        logger.info("✅ 모든 모델 훈련 완료!")
    
    def evaluate_models(self) -> Dict[str, Any]:
        """모델 성능 평가"""
        logger.info("📊 모델 성능 평가 시작")
        
        X_train, X_test, y_train, y_test = self.prepare_data()
        results = {}
        
        # 베이스라인 모델 평가
        logger.info("베이스라인 모델 평가...")
        X_test_technical = X_test[self.technical_features]
        
        for name, model_info in self.baseline_models.items():
            model = model_info['model']
            scaler = model_info['scaler']
            
            X_test_scaled = scaler.transform(X_test_technical)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            results[f'Baseline_{name}'] = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # 향상된 모델 평가
        logger.info("향상된 모델 평가...")
        for name, model_info in self.enhanced_models.items():
            model = model_info['model']
            scaler = model_info['scaler']
            
            X_test_scaled = scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            results[f'Enhanced_{name}'] = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        self.results = results
        return results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """성능 지표 계산"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'auc': roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0.5
        }
    
    def analyze_feature_importance(self) -> Dict[str, Any]:
        """특성 중요도 분석"""
        logger.info("🔍 특성 중요도 분석")
        
        importance_analysis = {}
        
        # RandomForest 모델의 특성 중요도 분석
        for model_type in ['Baseline', 'Enhanced']:
            rf_model = self.baseline_models['RandomForest'] if model_type == 'Baseline' else self.enhanced_models['RandomForest']
            features = rf_model['features']
            importances = rf_model['model'].feature_importances_
            
            # 중요도 순으로 정렬
            feature_importance = pd.DataFrame({
                'feature': features,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            importance_analysis[model_type] = feature_importance
        
        # 뉴스 관련 특성의 기여도 분석
        enhanced_importance = importance_analysis['Enhanced']
        news_importance = enhanced_importance[enhanced_importance['feature'].isin(self.news_features)]
        technical_importance = enhanced_importance[enhanced_importance['feature'].isin(self.technical_features)]
        
        news_total_importance = news_importance['importance'].sum()
        technical_total_importance = technical_importance['importance'].sum()
        
        importance_analysis['news_contribution'] = {
            'news_total_importance': news_total_importance,
            'technical_total_importance': technical_total_importance,
            'news_percentage': news_total_importance / (news_total_importance + technical_total_importance) * 100,
            'top_news_features': news_importance.head(5).to_dict('records'),
            'top_technical_features': technical_importance.head(5).to_dict('records')
        }
        
        return importance_analysis
    
    def create_performance_comparison_chart(self, save_path: str = "results/model_comparison_chart.png"):
        """성능 비교 차트 생성"""
        logger.info("📊 성능 비교 차트 생성")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 결과 데이터 준비
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        
        baseline_data = []
        enhanced_data = []
        model_names = []
        
        for key, values in self.results.items():
            if key.startswith('Baseline_'):
                model_name = key.replace('Baseline_', '')
                model_names.append(model_name)
                baseline_data.append([values[metric] for metric in metrics])
            elif key.startswith('Enhanced_'):
                model_name = key.replace('Enhanced_', '')
                enhanced_data.append([values[metric] for metric in metrics])
        
        # 차트 생성
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle('모델 성능 비교: 기술적 지표 vs 기술적 지표 + 뉴스', fontsize=16)
        
        # 1. 전체 성능 비교
        x = np.arange(len(metrics))
        width = 0.35
        
        baseline_means = np.mean(baseline_data, axis=0)
        enhanced_means = np.mean(enhanced_data, axis=0)
        
        axes[0].bar(x - width/2, baseline_means, width, label='기술적 지표만', alpha=0.8)
        axes[0].bar(x + width/2, enhanced_means, width, label='기술적 지표 + 뉴스', alpha=0.8)
        axes[0].set_xlabel('성능 지표')
        axes[0].set_ylabel('점수')
        axes[0].set_title('평균 성능 비교')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(metrics)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. 모델별 정확도 비교
        model_baseline_acc = [self.results[f'Baseline_{name}']['accuracy'] for name in model_names]
        model_enhanced_acc = [self.results[f'Enhanced_{name}']['accuracy'] for name in model_names]
        
        x_models = np.arange(len(model_names))
        axes[1].bar(x_models - width/2, model_baseline_acc, width, label='기술적 지표만', alpha=0.8)
        axes[1].bar(x_models + width/2, model_enhanced_acc, width, label='기술적 지표 + 뉴스', alpha=0.8)
        axes[1].set_xlabel('모델')
        axes[1].set_ylabel('정확도')
        axes[1].set_title('모델별 정확도 비교')
        axes[1].set_xticks(x_models)
        axes[1].set_xticklabels(model_names)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. 성능 향상률
        improvements = [(enhanced_means[i] - baseline_means[i]) / baseline_means[i] * 100 
                       for i in range(len(metrics))]
        
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        axes[2].bar(metrics, improvements, color=colors, alpha=0.7)
        axes[2].set_xlabel('성능 지표')
        axes[2].set_ylabel('향상률 (%)')
        axes[2].set_title('뉴스 감정분석 추가시 성능 향상률')
        axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[2].grid(True, alpha=0.3)
        
        # 수치 표시
        for i, imp in enumerate(improvements):
            axes[2].text(i, imp + (1 if imp > 0 else -1), f'{imp:.1f}%', 
                        ha='center', va='bottom' if imp > 0 else 'top')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"차트 저장 완료: {save_path}")
        return save_path
    
    def save_results(self, output_file: str = "results/model_comparison_results.json"):
        """결과 저장"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 특성 중요도 분석
        importance_analysis = self.analyze_feature_importance()
        
        # JSON 직렬화를 위한 변환 함수
        def convert_to_json_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.bool_, np.int64, np.float64)):
                return obj.item()
            elif isinstance(obj, dict):
                return {key: convert_to_json_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            elif hasattr(obj, 'to_dict'):
                return convert_to_json_serializable(obj.to_dict())
            else:
                return obj
        
        # 전체 결과 구성
        full_results = {
            'timestamp': datetime.now().isoformat(),
            'data_info': {
                'total_samples': int(len(self.data)),
                'total_features': len(self.all_features),
                'technical_features': len(self.technical_features),
                'news_features': len(self.news_features),
                'target_distribution': {str(k): int(v) for k, v in self.data['target'].value_counts().to_dict().items()}
            },
            'performance_results': convert_to_json_serializable(self.results),
            'feature_importance': {
                'news_contribution_percentage': float(importance_analysis['news_contribution']['news_percentage']),
                'top_news_features': convert_to_json_serializable(importance_analysis['news_contribution']['top_news_features']),
                'top_technical_features': convert_to_json_serializable(importance_analysis['news_contribution']['top_technical_features'])
            },
            'summary': convert_to_json_serializable(self._generate_summary())
        }
        
        # JSON으로 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"결과 저장 완료: {output_file}")
        return output_file
    
    def _generate_summary(self) -> Dict[str, Any]:
        """결과 요약 생성"""
        # 평균 성능 계산
        baseline_avg = {}
        enhanced_avg = {}
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        
        for metric in metrics:
            baseline_values = [results[metric] for key, results in self.results.items() if key.startswith('Baseline_')]
            enhanced_values = [results[metric] for key, results in self.results.items() if key.startswith('Enhanced_')]
            
            baseline_avg[metric] = np.mean(baseline_values)
            enhanced_avg[metric] = np.mean(enhanced_values)
        
        # 향상률 계산
        improvements = {}
        for metric in metrics:
            improvements[metric] = (enhanced_avg[metric] - baseline_avg[metric]) / baseline_avg[metric] * 100
        
        return {
            'baseline_average': baseline_avg,
            'enhanced_average': enhanced_avg,
            'improvements': improvements,
            'best_baseline_model': max([(k, v['accuracy']) for k, v in self.results.items() if k.startswith('Baseline_')], key=lambda x: x[1])[0],
            'best_enhanced_model': max([(k, v['accuracy']) for k, v in self.results.items() if k.startswith('Enhanced_')], key=lambda x: x[1])[0],
            'news_helps': enhanced_avg['accuracy'] > baseline_avg['accuracy']
        }
    
    def print_results(self):
        """결과 출력"""
        logger.info("\n" + "="*80)
        logger.info("🏆 모델 성능 비교 결과")
        logger.info("="*80)
        
        # 개별 모델 성능
        logger.info("\n📊 개별 모델 성능:")
        for model_name, metrics in self.results.items():
            logger.info(f"\n{model_name}:")
            for metric, value in metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
        
        # 요약
        summary = self._generate_summary()
        logger.info(f"\n🔍 종합 분석:")
        logger.info(f"최고 베이스라인 모델: {summary['best_baseline_model']} ({self.results[summary['best_baseline_model']]['accuracy']:.4f})")
        logger.info(f"최고 향상 모델: {summary['best_enhanced_model']} ({self.results[summary['best_enhanced_model']]['accuracy']:.4f})")
        logger.info(f"뉴스 감정분석 효과: {'✅ 도움됨' if summary['news_helps'] else '❌ 도움안됨'}")
        
        logger.info(f"\n📈 평균 성능 향상률:")
        for metric, improvement in summary['improvements'].items():
            status = "📈" if improvement > 0 else "📉"
            logger.info(f"  {metric}: {status} {improvement:+.2f}%")


def main():
    """메인 실행 함수"""
    logger.info("🚀 뉴스 감정분석 모델 성능 비교 시작")
    
    try:
        # 비교기 초기화
        comparator = ModelComparison()
        
        # 데이터 로드
        comparator.load_data()
        
        # 모델 훈련
        comparator.train_models()
        
        # 성능 평가
        comparator.evaluate_models()
        
        # 결과 출력
        comparator.print_results()
        
        # 차트 생성
        comparator.create_performance_comparison_chart()
        
        # 결과 저장
        comparator.save_results()
        
        logger.info("✅ 모델 성능 비교 완료!")
        
    except Exception as e:
        logger.error(f"❌ 모델 성능 비교 실패: {str(e)}")
        raise


if __name__ == "__main__":
    main()