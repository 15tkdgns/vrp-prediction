#!/usr/bin/env python3
"""
SHAP (SHapley Additive exPlanations) 분석 모듈
모델의 예측 결과를 개별 특성 기여도로 분해하여 해석성 향상
"""

import pandas as pd
import numpy as np
import shap
import json
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SHAPAnalyzer:
    """SHAP를 이용한 모델 해석성 분석 클래스"""
    
    def __init__(self, data_file: str = "data/raw/integrated_spy_news_data.csv", 
                 results_dir: str = "results"):
        self.data_file = data_file
        self.results_dir = results_dir
        self.data = None
        self.models = {}
        self.shap_values = {}
        self.explainers = {}
        
        # 결과 저장 디렉토리 생성
        os.makedirs(results_dir, exist_ok=True)
        
    def load_data(self):
        """데이터 로드 및 전처리"""
        logger.info(f"Loading data from {self.data_file}")
        self.data = pd.read_csv(self.data_file)
        
        # 날짜 컬럼 처리
        if 'date' in self.data.columns:
            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data = self.data.sort_values('date')
            
        # NaN 값 제거
        self.data = self.data.dropna()
        
        logger.info(f"Data loaded: {len(self.data)} samples, {self.data.shape[1]} features")
        
    def load_trained_models(self, model_file: str = "data/models/model_comparison_models.pkl"):
        """훈련된 모델들 로드"""
        try:
            with open(model_file, 'rb') as f:
                saved_data = pickle.load(f)
                self.models = saved_data.get('models', {})
                logger.info(f"Loaded {len(self.models)} trained models")
        except FileNotFoundError:
            logger.warning(f"Model file not found: {model_file}")
            logger.info("Will train models from scratch if needed")
            
    def create_explainers(self, model_name: str, model_data: Dict):
        """모델별 SHAP Explainer 생성"""
        model = model_data['model']
        features = model_data['features']
        X = self.data[features]
        
        # 모델 타입에 따라 적절한 Explainer 선택
        if 'RandomForest' in model_name or 'GradientBoosting' in model_name or 'XGBoost' in model_name:
            explainer = shap.TreeExplainer(model)
            logger.info(f"Created TreeExplainer for {model_name}")
        elif 'LogisticRegression' in model_name:
            # 배경 데이터 샘플링 (너무 크면 메모리 문제)
            background = shap.sample(X, min(100, len(X)))
            explainer = shap.LinearExplainer(model, background)
            logger.info(f"Created LinearExplainer for {model_name}")
        else:
            # 일반적인 경우 KernelExplainer 사용 (느리지만 범용)
            background = shap.sample(X, min(50, len(X)))
            explainer = shap.KernelExplainer(model.predict_proba, background)
            logger.info(f"Created KernelExplainer for {model_name}")
            
        return explainer, X
        
    def calculate_shap_values(self, model_name: str, sample_size: Optional[int] = None):
        """특정 모델의 SHAP 값 계산"""
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return None
            
        logger.info(f"Calculating SHAP values for {model_name}")
        
        model_data = self.models[model_name]
        explainer, X = self.create_explainers(model_name, model_data)
        
        # 샘플 크기 조정 (계산 시간 단축)
        if sample_size and len(X) > sample_size:
            X_sample = X.sample(n=sample_size, random_state=42)
            logger.info(f"Using sample size: {sample_size}")
        else:
            X_sample = X
            
        # SHAP 값 계산
        shap_values = explainer.shap_values(X_sample)
        
        # 이진 분류의 경우 양성 클래스의 SHAP 값만 사용
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]  # 상승(1) 클래스
            
        self.shap_values[model_name] = {
            'values': shap_values,
            'data': X_sample,
            'features': model_data['features'],
            'explainer': explainer
        }
        
        logger.info(f"SHAP values calculated for {model_name}: {shap_values.shape}")
        return shap_values
        
    def generate_summary_plot(self, model_name: str, max_display: int = 15):
        """SHAP Summary Plot 생성"""
        if model_name not in self.shap_values:
            logger.error(f"SHAP values not found for {model_name}")
            return None
            
        shap_data = self.shap_values[model_name]
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_data['values'], 
            shap_data['data'], 
            feature_names=shap_data['features'],
            max_display=max_display,
            show=False
        )
        plt.title(f'SHAP Summary Plot - {model_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 한글 특성명으로 변환
        feature_names_kr = self._translate_feature_names(shap_data['features'])
        
        # 파일 저장
        plot_file = os.path.join(self.results_dir, f'shap_summary_{model_name.lower()}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"SHAP summary plot saved: {plot_file}")
        return plot_file
        
    def generate_waterfall_plot(self, model_name: str, sample_idx: int = 0):
        """특정 샘플의 SHAP Waterfall Plot 생성"""
        if model_name not in self.shap_values:
            logger.error(f"SHAP values not found for {model_name}")
            return None
            
        shap_data = self.shap_values[model_name]
        
        # SHAP Explanation 객체 생성
        explanation = shap.Explanation(
            values=shap_data['values'][sample_idx],
            base_values=shap_data['explainer'].expected_value if hasattr(shap_data['explainer'], 'expected_value') else 0,
            data=shap_data['data'].iloc[sample_idx].values,
            feature_names=shap_data['features']
        )
        
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(explanation, max_display=15, show=False)
        plt.title(f'SHAP Waterfall Plot - {model_name} (Sample {sample_idx})', 
                  fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plot_file = os.path.join(self.results_dir, f'shap_waterfall_{model_name.lower()}_sample_{sample_idx}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"SHAP waterfall plot saved: {plot_file}")
        return plot_file
        
    def calculate_feature_importance(self, model_name: str) -> Dict:
        """SHAP 값 기반 특성 중요도 계산"""
        if model_name not in self.shap_values:
            logger.error(f"SHAP values not found for {model_name}")
            return {}
            
        shap_data = self.shap_values[model_name]
        shap_values = shap_data['values']
        features = shap_data['features']
        
        # 절댓값 평균으로 중요도 계산
        importance_scores = np.abs(shap_values).mean(0)
        
        # 특성명과 중요도를 쌍으로 만들어 정렬
        feature_importance = list(zip(features, importance_scores))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        # 뉴스 vs 기술적 지표 분류
        news_keywords = ['sentiment', 'news', 'article', 'impact']
        news_features = []
        technical_features = []
        
        for feature, importance in feature_importance:
            feature_data = {
                'feature': feature,
                'feature_kr': self._translate_feature_name(feature),
                'importance': float(importance),
                'importance_pct': float(importance / sum(importance_scores) * 100)
            }
            
            if any(keyword in feature.lower() for keyword in news_keywords):
                news_features.append(feature_data)
            else:
                technical_features.append(feature_data)
                
        return {
            'total_features': len(features),
            'top_features': [
                {
                    'feature': f,
                    'feature_kr': self._translate_feature_name(f),
                    'importance': float(imp),
                    'importance_pct': float(imp / sum(importance_scores) * 100)
                }
                for f, imp in feature_importance[:10]
            ],
            'news_features': news_features,
            'technical_features': technical_features,
            'news_contribution_pct': sum([f['importance'] for f in news_features]) / sum(importance_scores) * 100,
            'technical_contribution_pct': sum([f['importance'] for f in technical_features]) / sum(importance_scores) * 100
        }
        
    def analyze_feature_interactions(self, model_name: str, feature1: str, feature2: str):
        """두 특성 간의 SHAP 상호작용 분석"""
        if model_name not in self.shap_values:
            logger.error(f"SHAP values not found for {model_name}")
            return None
            
        shap_data = self.shap_values[model_name]
        
        if feature1 not in shap_data['features'] or feature2 not in shap_data['features']:
            logger.error(f"Features not found: {feature1}, {feature2}")
            return None
            
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature1, 
            shap_data['values'], 
            shap_data['data'],
            feature_names=shap_data['features'],
            interaction_index=feature2,
            show=False
        )
        plt.title(f'SHAP Dependence Plot: {feature1} vs {feature2}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plot_file = os.path.join(self.results_dir, f'shap_interaction_{model_name.lower()}_{feature1}_{feature2}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"SHAP interaction plot saved: {plot_file}")
        return plot_file
        
    def generate_daily_explanations(self, model_name: str, start_date: str = None, end_date: str = None) -> List[Dict]:
        """일별 예측 설명 생성"""
        if model_name not in self.shap_values:
            logger.error(f"SHAP values not found for {model_name}")
            return []
            
        shap_data = self.shap_values[model_name]
        data = shap_data['data'].copy()
        
        # 날짜 정보가 있는 경우 필터링
        if 'date' in self.data.columns and start_date:
            date_filter = (self.data['date'] >= start_date)
            if end_date:
                date_filter = date_filter & (self.data['date'] <= end_date)
            
            filtered_indices = self.data[date_filter].index
            # SHAP 데이터에서 해당 인덱스만 선택
            common_indices = data.index.intersection(filtered_indices)
            data = data.loc[common_indices]
            
        explanations = []
        
        for idx in range(min(20, len(data))):  # 최대 20일만 분석
            row_idx = data.index[idx]
            shap_row = shap_data['values'][idx] if idx < len(shap_data['values']) else None
            
            if shap_row is None:
                continue
                
            # 상위 기여도 특성들
            abs_shap = np.abs(shap_row)
            top_indices = np.argsort(abs_shap)[-5:][::-1]  # 상위 5개
            
            top_features = []
            for i in top_indices:
                feature = shap_data['features'][i]
                contribution = float(shap_row[i])
                value = float(data.iloc[idx, i])
                
                top_features.append({
                    'feature': feature,
                    'feature_kr': self._translate_feature_name(feature),
                    'contribution': contribution,
                    'value': value,
                    'impact': 'positive' if contribution > 0 else 'negative'
                })
                
            date_str = self.data.loc[row_idx, 'date'].strftime('%Y-%m-%d') if 'date' in self.data.columns else f"Sample_{idx}"
            
            explanations.append({
                'date': date_str,
                'sample_idx': idx,
                'total_shap': float(np.sum(shap_row)),
                'top_features': top_features
            })
            
        return explanations
        
    def _translate_feature_name(self, feature: str) -> str:
        """특성명을 한글로 번역"""
        translations = {
            'sentiment_change': '감정 변화율',
            'sentiment_ma_7': '7일 평균 감정',
            'news_count_change': '뉴스 수 변화',
            'sentiment_abs': '감정 강도',
            'sentiment_volatility': '감정 변동성',
            'price_to_ma20': '20일선 대비 가격',
            'ma_10': '10일 이동평균',
            'volatility_20': '20일 변동성',
            'price_change_abs': '절대 가격 변화',
            'price_to_ma5': '5일선 대비 가격',
            'ma_5': '5일 이동평균',
            'ma_20': '20일 이동평균',
            'rsi': 'RSI 지표',
            'macd': 'MACD',
            'volume_change': '거래량 변화',
            'unusual_volume': '비정상 거래량',
            'price_change': '가격 변화율',
            'volatility_5': '5일 변동성'
        }
        return translations.get(feature, feature)
        
    def _translate_feature_names(self, features: List[str]) -> List[str]:
        """특성명 리스트를 한글로 번역"""
        return [self._translate_feature_name(f) for f in features]
        
    def save_results(self, model_name: str, output_file: str = None):
        """SHAP 분석 결과를 JSON으로 저장"""
        if model_name not in self.shap_values:
            logger.error(f"SHAP values not found for {model_name}")
            return None
            
        if not output_file:
            output_file = os.path.join(self.results_dir, f'shap_analysis_{model_name.lower()}.json')
            
        # 특성 중요도 계산
        feature_importance = self.calculate_feature_importance(model_name)
        
        # 일별 설명 생성
        daily_explanations = self.generate_daily_explanations(model_name)
        
        # 결과 통합
        results = {
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'feature_importance': feature_importance,
            'daily_explanations': daily_explanations[:10],  # 상위 10일만
            'summary': {
                'total_samples': len(self.shap_values[model_name]['data']),
                'total_features': len(self.shap_values[model_name]['features']),
                'news_features_count': len(feature_importance['news_features']),
                'technical_features_count': len(feature_importance['technical_features'])
            }
        }
        
        # JSON 직렬화 가능한 형태로 변환
        results = self._convert_to_json_serializable(results)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        logger.info(f"SHAP analysis results saved: {output_file}")
        return output_file
        
    def _convert_to_json_serializable(self, obj):
        """JSON 직렬화 가능한 형태로 변환"""
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.int_)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float_)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
            
    def run_complete_analysis(self, model_names: List[str] = None, sample_size: int = 200):
        """전체 SHAP 분석 실행"""
        logger.info("Starting complete SHAP analysis")
        
        # 데이터 로드
        self.load_data()
        
        # 모델 로드 (없으면 새로 학습)
        self.load_trained_models()
        
        if not self.models:
            logger.info("No pre-trained models found, training new models...")
            self._train_models_for_analysis()
            
        # 분석할 모델 선택
        if not model_names:
            model_names = list(self.models.keys())
            
        logger.info(f"Available models: {list(self.models.keys())}")
        logger.info(f"Target models: {model_names}")
            
        results_files = []
        
        for model_name in model_names:
            try:
                logger.info(f"Analyzing model: {model_name}")
                
                # SHAP 값 계산
                self.calculate_shap_values(model_name, sample_size=sample_size)
                
                # 시각화 생성
                self.generate_summary_plot(model_name)
                self.generate_waterfall_plot(model_name, sample_idx=0)
                
                # 결과 저장
                result_file = self.save_results(model_name)
                results_files.append(result_file)
                
                logger.info(f"Analysis completed for {model_name}")
                
            except Exception as e:
                logger.error(f"Error analyzing {model_name}: {str(e)}")
                continue
                
        logger.info(f"SHAP analysis completed. Results saved: {results_files}")
        return results_files
        
    def _train_models_for_analysis(self):
        """분석용 모델 학습 (모델이 없는 경우)"""
        logger.info("Training models for SHAP analysis...")
        
        # 모델 비교 모듈 임포트
        try:
            import sys
            sys.path.append('/root/workspace/src/utils')
            from model_comparison import ModelComparison
            
            # 모델 학습
            comparator = ModelComparison(self.data_file)
            comparator.load_data()
            X_train, X_test, y_train, y_test = comparator.prepare_data()
            comparator.train_models()  # 모델 훈련 (반환값 없음)
            
            # Enhanced 모델들만 사용
            self.models = comparator.enhanced_models
            
            logger.info(f"Trained {len(self.models)} models for analysis")
            
        except ImportError as e:
            logger.error(f"Could not import ModelComparison: {e}")
            logger.error("Please run model comparison first")


def main():
    """메인 실행 함수"""
    analyzer = SHAPAnalyzer()
    
    # 주요 모델들에 대해 SHAP 분석 실행 (ModelComparison에서 사용하는 키명)
    target_models = [
        'GradientBoosting',
        'RandomForest', 
        'LogisticRegression'
    ]
    
    results = analyzer.run_complete_analysis(
        model_names=target_models,
        sample_size=150  # 계산 시간 단축
    )
    
    print(f"✅ SHAP 분석 완료!")
    print(f"📁 결과 파일: {results}")
    

if __name__ == "__main__":
    main()