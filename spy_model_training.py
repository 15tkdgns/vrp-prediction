#!/usr/bin/env python3
"""
SPY 머신러닝 모델 학습 및 평가
다중 모델 비교, 하이퍼파라미터 최적화, 엄격한 검증
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ML libraries  
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Cross-validation and metrics
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, classification_report, confusion_matrix,
                           precision_recall_curve, roc_curve)

# Utilities
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
from datetime import datetime
import logging
import os
from spy_ml_pipeline import SPYMLPipeline

class SPYModelTrainer:
    """SPY 모델 학습 및 평가 클래스"""
    
    def __init__(self, pipeline):
        """
        Args:
            pipeline: SPYMLPipeline 인스턴스
        """
        self.pipeline = pipeline
        self.models = {}
        self.model_results = {}
        self.best_models = {}
        self.ensemble_model = None
        
        # 실험 결과 저장
        self.training_log = {
            'start_time': datetime.now().isoformat(),
            'model_configs': {},
            'cross_validation_results': {},
            'final_model_performance': {},
            'feature_importance': {}
        }
        
        self.logger = logging.getLogger(__name__)
        
    def define_model_space(self):
        """모델 및 하이퍼파라미터 공간 정의"""
        self.logger.info("=== 모델 하이퍼파라미터 공간 정의 ===")
        
        # 1. Random Forest - 오버피팅 방지 중점
        rf_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'class_weight': ['balanced', 'balanced_subsample'],
            'random_state': [42]
        }
        
        # 2. Gradient Boosting - Early stopping 적용
        gb_params = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.8, 0.9, 1.0],
            'random_state': [42]
        }
        
        # 3. Logistic Regression - 정규화 중점
        lr_params = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'solver': ['liblinear', 'saga'],
            'class_weight': ['balanced'],
            'random_state': [42],
            'max_iter': [1000]
        }
        
        # 4. SVM - RBF와 Linear 커널
        svm_params = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'class_weight': ['balanced'],
            'random_state': [42]
        }
        
        # 5. Ridge Classifier - 정규화
        ridge_params = {
            'alpha': [0.1, 1.0, 10.0, 100.0],
            'class_weight': ['balanced'],
            'random_state': [42]
        }
        
        self.model_param_grid = {
            'RandomForest': (RandomForestClassifier(), rf_params),
            'GradientBoosting': (GradientBoostingClassifier(), gb_params), 
            'LogisticRegression': (LogisticRegression(), lr_params),
            'SVM': (SVC(probability=True), svm_params),
            'Ridge': (RidgeClassifier(), ridge_params)
        }
        
        # 설정 기록
        self.training_log['model_configs'] = {
            name: {'base_model': str(model), 'param_grid_size': len(params)}
            for name, (model, params) in self.model_param_grid.items()
        }
        
        self.logger.info(f"정의된 모델 수: {len(self.model_param_grid)}")
        for name, (_, params) in self.model_param_grid.items():
            param_combinations = np.prod([len(v) for v in params.values()])
            self.logger.info(f"{name}: {param_combinations}개 조합")
            
    def perform_cross_validation(self, cv_folds=5, scoring='roc_auc', n_jobs=-1):
        """
        시계열 교차 검증 수행
        
        Args:
            cv_folds: CV 폴드 수
            scoring: 평가 지표
            n_jobs: 병렬 처리 수
        """
        self.logger.info(f"=== 교차 검증 시작 (CV={cv_folds}, scoring={scoring}) ===")
        
        # TimeSeriesSplit 사용 (데이터 누출 방지)
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
        cv_results = {}
        
        for name, (base_model, param_grid) in self.model_param_grid.items():
            self.logger.info(f"모델 학습 중: {name}")
            
            try:
                # RandomizedSearchCV로 효율적인 하이퍼파라미터 탐색
                # (모든 조합을 다 시도하면 너무 오래 걸림)
                random_search = RandomizedSearchCV(
                    estimator=base_model,
                    param_distributions=param_grid,
                    n_iter=50,  # 50회 랜덤 샘플링
                    cv=tscv,
                    scoring=scoring,
                    n_jobs=n_jobs,
                    random_state=42,
                    verbose=1
                )
                
                # 학습 (선택된 특성 사용)
                random_search.fit(self.pipeline.X_train_selected, self.pipeline.y_train)
                
                # 최적 모델 저장
                self.best_models[name] = random_search.best_estimator_
                
                # 결과 기록
                cv_results[name] = {
                    'best_score': random_search.best_score_,
                    'best_params': random_search.best_params_,
                    'cv_mean': random_search.cv_results_['mean_test_score'].max(),
                    'cv_std': random_search.cv_results_['std_test_score'][random_search.best_index_]
                }
                
                self.logger.info(f"{name} 최적 성능: {random_search.best_score_:.4f} (±{cv_results[name]['cv_std']:.4f})")
                
            except Exception as e:
                self.logger.error(f"{name} 학습 중 오류: {e}")
                cv_results[name] = {'error': str(e)}
        
        self.training_log['cross_validation_results'] = cv_results
        
        # 성능 순 정렬
        valid_results = {k: v for k, v in cv_results.items() if 'error' not in v}
        if valid_results:
            sorted_models = sorted(valid_results.items(), key=lambda x: x[1]['best_score'], reverse=True)
            self.logger.info("=== 교차 검증 결과 (성능순) ===")
            for rank, (name, result) in enumerate(sorted_models, 1):
                self.logger.info(f"{rank}. {name}: {result['best_score']:.4f} (±{result['cv_std']:.4f})")
        
        return cv_results
    
    def evaluate_models(self):
        """검증 세트에서 모델 성능 평가"""
        self.logger.info("=== 검증 세트 모델 평가 ===")
        
        evaluation_results = {}
        
        for name, model in self.best_models.items():
            self.logger.info(f"모델 평가 중: {name}")
            
            try:
                # 예측
                y_pred = model.predict(self.pipeline.X_val_selected)
                y_pred_proba = model.predict_proba(self.pipeline.X_val_selected)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # 성능 지표 계산
                metrics = {
                    'accuracy': accuracy_score(self.pipeline.y_val, y_pred),
                    'precision': precision_score(self.pipeline.y_val, y_pred, average='weighted'),
                    'recall': recall_score(self.pipeline.y_val, y_pred, average='weighted'),
                    'f1': f1_score(self.pipeline.y_val, y_pred, average='weighted')
                }
                
                if y_pred_proba is not None:
                    metrics['roc_auc'] = roc_auc_score(self.pipeline.y_val, y_pred_proba)
                
                evaluation_results[name] = metrics
                
                # 로깅
                self.logger.info(f"{name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
                if 'roc_auc' in metrics:
                    self.logger.info(f"{name} - ROC AUC: {metrics['roc_auc']:.4f}")
                    
            except Exception as e:
                self.logger.error(f"{name} 평가 중 오류: {e}")
                evaluation_results[name] = {'error': str(e)}
        
        self.training_log['validation_performance'] = evaluation_results
        
        return evaluation_results
    
    def create_ensemble_model(self, top_k=3):
        """
        상위 모델들로 앙상블 생성
        
        Args:
            top_k: 앙상블에 포함할 상위 모델 수
        """
        self.logger.info(f"=== 앙상블 모델 생성 (상위 {top_k}개) ===")
        
        # 검증 성능 기준으로 상위 모델 선택
        if 'validation_performance' not in self.training_log:
            self.logger.warning("검증 성능 데이터가 없습니다. 먼저 evaluate_models()를 실행하세요.")
            return None
        
        valid_models = {k: v for k, v in self.training_log['validation_performance'].items() 
                       if 'error' not in v and 'roc_auc' in v}
        
        if len(valid_models) < top_k:
            top_k = len(valid_models)
            self.logger.warning(f"유효한 모델 수가 {top_k}개뿐입니다.")
        
        # ROC AUC 기준 상위 모델 선택
        top_models = sorted(valid_models.items(), key=lambda x: x[1]['roc_auc'], reverse=True)[:top_k]
        
        # 앙상블 모델 생성
        ensemble_estimators = [(name, self.best_models[name]) for name, _ in top_models]
        
        self.ensemble_model = VotingClassifier(
            estimators=ensemble_estimators,
            voting='soft',  # 확률 기반 투표
            n_jobs=-1
        )
        
        # 앙상블 학습
        self.ensemble_model.fit(self.pipeline.X_train_selected, self.pipeline.y_train)
        
        # 앙상블 성능 평가
        ensemble_pred = self.ensemble_model.predict(self.pipeline.X_val_selected)
        ensemble_proba = self.ensemble_model.predict_proba(self.pipeline.X_val_selected)[:, 1]
        
        ensemble_metrics = {
            'accuracy': accuracy_score(self.pipeline.y_val, ensemble_pred),
            'precision': precision_score(self.pipeline.y_val, ensemble_pred, average='weighted'),
            'recall': recall_score(self.pipeline.y_val, ensemble_pred, average='weighted'),
            'f1': f1_score(self.pipeline.y_val, ensemble_pred, average='weighted'),
            'roc_auc': roc_auc_score(self.pipeline.y_val, ensemble_proba)
        }
        
        self.training_log['ensemble_performance'] = {
            'component_models': [name for name, _ in top_models],
            'metrics': ensemble_metrics
        }
        
        self.logger.info("앙상블 모델 생성 완료")
        self.logger.info(f"구성 모델: {[name for name, _ in top_models]}")
        self.logger.info(f"앙상블 성능 - Accuracy: {ensemble_metrics['accuracy']:.4f}, ROC AUC: {ensemble_metrics['roc_auc']:.4f}")
        
        return self.ensemble_model
    
    def final_test_evaluation(self):
        """테스트 세트에서 최종 성능 평가"""
        self.logger.info("=== 테스트 세트 최종 평가 ===")
        
        if self.ensemble_model is None:
            self.logger.warning("앙상블 모델이 없습니다. 최고 성능 단일 모델을 사용합니다.")
            # 가장 좋은 단일 모델 선택
            best_model_name = max(self.training_log['validation_performance'].items(), 
                                 key=lambda x: x[1].get('roc_auc', 0))[0]
            final_model = self.best_models[best_model_name]
            model_type = 'single'
        else:
            final_model = self.ensemble_model
            model_type = 'ensemble'
        
        # 테스트 예측
        y_test_pred = final_model.predict(self.pipeline.X_test_selected)
        y_test_proba = final_model.predict_proba(self.pipeline.X_test_selected)[:, 1] if hasattr(final_model, 'predict_proba') else None
        
        # 최종 성능 지표
        final_metrics = {
            'accuracy': accuracy_score(self.pipeline.y_test, y_test_pred),
            'precision': precision_score(self.pipeline.y_test, y_test_pred, average='weighted'),
            'recall': recall_score(self.pipeline.y_test, y_test_pred, average='weighted'),
            'f1': f1_score(self.pipeline.y_test, y_test_pred, average='weighted')
        }
        
        if y_test_proba is not None:
            final_metrics['roc_auc'] = roc_auc_score(self.pipeline.y_test, y_test_proba)
        
        # 분류 리포트
        class_report = classification_report(self.pipeline.y_test, y_test_pred, output_dict=True)
        
        # 혼동 행렬
        conf_matrix = confusion_matrix(self.pipeline.y_test, y_test_pred)
        
        self.training_log['final_test_performance'] = {
            'model_type': model_type,
            'metrics': final_metrics,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'test_period': f"{self.pipeline.X_test.index[0]} to {self.pipeline.X_test.index[-1]}",
            'test_samples': len(self.pipeline.y_test)
        }
        
        # 결과 출력
        self.logger.info(f"최종 모델 유형: {model_type}")
        self.logger.info(f"테스트 기간: {self.training_log['final_test_performance']['test_period']}")
        self.logger.info(f"테스트 샘플 수: {len(self.pipeline.y_test)}")
        self.logger.info("=== 최종 성능 지표 ===")
        for metric, value in final_metrics.items():
            self.logger.info(f"{metric.upper()}: {value:.4f}")
        
        return final_metrics, final_model
    
    def analyze_feature_importance(self, top_k=20):
        """특성 중요도 분석"""
        self.logger.info(f"=== 특성 중요도 분석 (상위 {top_k}개) ===")
        
        importance_data = {}
        
        # 개별 모델들의 특성 중요도
        for name, model in self.best_models.items():
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_names = self.pipeline.X_train_selected.columns
                
                # 상위 특성들
                indices = np.argsort(importances)[::-1][:top_k]
                top_features = [(feature_names[i], importances[i]) for i in indices]
                
                importance_data[name] = {
                    'top_features': top_features,
                    'feature_importance_sum': float(np.sum(importances))
                }
                
                self.logger.info(f"{name} 상위 5개 특성:")
                for i, (feature, importance) in enumerate(top_features[:5]):
                    self.logger.info(f"  {i+1}. {feature}: {importance:.4f}")
        
        # 앙상블 모델 특성 중요도 (가능한 경우)
        if self.ensemble_model is not None:
            # 구성 모델들의 평균 중요도 계산
            all_importances = []
            for estimator_name, estimator in self.ensemble_model.named_estimators_.items():
                if hasattr(estimator, 'feature_importances_'):
                    all_importances.append(estimator.feature_importances_)
            
            if all_importances:
                mean_importance = np.mean(all_importances, axis=0)
                feature_names = self.pipeline.X_train_selected.columns
                
                indices = np.argsort(mean_importance)[::-1][:top_k]
                ensemble_top_features = [(feature_names[i], mean_importance[i]) for i in indices]
                
                importance_data['ensemble_mean'] = {
                    'top_features': ensemble_top_features,
                    'feature_importance_sum': float(np.sum(mean_importance))
                }
                
                self.logger.info("앙상블 평균 중요도 상위 5개 특성:")
                for i, (feature, importance) in enumerate(ensemble_top_features[:5]):
                    self.logger.info(f"  {i+1}. {feature}: {importance:.4f}")
        
        self.training_log['feature_importance'] = importance_data
        
        return importance_data
    
    def save_models_and_results(self):
        """모델과 결과 저장"""
        self.logger.info("=== 모델 및 결과 저장 ===")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs('models', exist_ok=True)
        
        # 1. 최적 모델들 저장
        models_file = f"models/spy_best_models_{timestamp}.pkl"
        with open(models_file, 'wb') as f:
            pickle.dump(self.best_models, f)
        
        # 2. 앙상블 모델 저장
        if self.ensemble_model is not None:
            ensemble_file = f"models/spy_ensemble_model_{timestamp}.pkl"
            with open(ensemble_file, 'wb') as f:
                pickle.dump(self.ensemble_model, f)
        
        # 3. 스케일러와 특성 선택기 저장
        pipeline_file = f"models/spy_pipeline_{timestamp}.pkl"
        pipeline_objects = {
            'scaler': self.pipeline.scalers.get('main'),
            'feature_selector': self.pipeline.feature_selectors.get('main'),
            'selected_features': list(self.pipeline.X_train_selected.columns)
        }
        with open(pipeline_file, 'wb') as f:
            pickle.dump(pipeline_objects, f)
        
        # 4. 학습 로그 저장
        self.training_log['end_time'] = datetime.now().isoformat()
        self.training_log['saved_files'] = {
            'models': models_file,
            'ensemble': ensemble_file if self.ensemble_model else None,
            'pipeline': pipeline_file
        }
        
        log_file = f"models/spy_training_log_{timestamp}.json"
        with open(log_file, 'w') as f:
            json.dump(self.training_log, f, indent=2, default=str)
        
        self.logger.info(f"모델 저장: {models_file}")
        if self.ensemble_model:
            self.logger.info(f"앙상블 저장: {ensemble_file}")
        self.logger.info(f"파이프라인 저장: {pipeline_file}")
        self.logger.info(f"학습 로그 저장: {log_file}")
        
        return {
            'models': models_file,
            'ensemble': ensemble_file if self.ensemble_model else None,
            'pipeline': pipeline_file,
            'log': log_file
        }

# 실행 함수
def main():
    """메인 실행 함수"""
    print("=== SPY 머신러닝 모델 학습 시작 ===")
    
    # 1. 데이터 파이프라인 실행
    pipeline = SPYMLPipeline()
    
    # 데이터 로드 및 전처리
    data = pipeline.load_and_explore_data()
    binary_target, multi_target = pipeline.create_target_variable(prediction_horizon=1, threshold=0.001)
    features = pipeline.engineer_features(lookback_periods=[5, 10, 20, 50])
    
    X_train, X_val, X_test, y_train, y_val, y_test = pipeline.prepare_ml_data(
        test_size=0.2, val_size=0.1, target_type='binary'
    )
    
    X_train_scaled, X_val_scaled, X_test_scaled = pipeline.scale_features(scaler_type='robust')
    X_train_selected, X_val_selected, X_test_selected = pipeline.select_features(method='rfe', k=30)
    
    # 2. 모델 학습
    trainer = SPYModelTrainer(pipeline)
    
    # 모델 공간 정의
    trainer.define_model_space()
    
    # 교차 검증
    cv_results = trainer.perform_cross_validation(cv_folds=5, scoring='roc_auc')
    
    # 검증 세트 평가
    val_results = trainer.evaluate_models()
    
    # 앙상블 모델 생성
    ensemble_model = trainer.create_ensemble_model(top_k=3)
    
    # 최종 테스트 평가
    final_metrics, final_model = trainer.final_test_evaluation()
    
    # 특성 중요도 분석
    importance_results = trainer.analyze_feature_importance(top_k=20)
    
    # 결과 저장
    saved_files = trainer.save_models_and_results()
    
    print("=== 학습 완료 ===")
    print(f"최종 성능: {final_metrics}")
    print(f"저장된 파일들: {saved_files}")
    
    return trainer, final_model

if __name__ == "__main__":
    trainer, model = main()