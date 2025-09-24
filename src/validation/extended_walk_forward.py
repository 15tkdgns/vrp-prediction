#!/usr/bin/env python3
"""
Extended Walk-Forward Validation for SPY Returns Prediction
5년 확장 데이터셋을 활용한 회귀 시계열 검증 시스템
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import warnings
warnings.filterwarnings('ignore')

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class RegressionWalkForwardValidator:
    """5년 SPY 회귀 데이터를 위한 Walk-Forward Validation 클래스"""
    
    def __init__(self, data_file: str = "data/processed/extended_spy_features_safe.csv",
                 results_dir: str = "results/validation"):
        self.data_file = data_file
        self.results_dir = results_dir
        self.data = None
        self.results = {}
        
        # 결과 저장 디렉토리 생성
        os.makedirs(results_dir, exist_ok=True)
        
        logger.info(f"Initialized RegressionWalkForwardValidator with data: {data_file}")
        
    def load_extended_data(self) -> pd.DataFrame:
        """확장된 SPY 데이터 로드 및 전처리"""
        logger.info(f"Loading extended data from {self.data_file}")
        
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"Extended data file not found: {self.data_file}")
            
        self.data = pd.read_csv(self.data_file)
        
        # 날짜 컬럼 처리
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data = self.data.sort_values('Date').reset_index(drop=True)
        
        # NaN 값 확인 및 처리
        null_counts = self.data.isnull().sum()
        if null_counts.sum() > 0:
            logger.warning(f"Found {null_counts.sum()} null values, filling with 0")
            self.data = self.data.fillna(0)
        
        # 데이터 요약
        logger.info(f"Data loaded: {len(self.data)} samples")
        logger.info(f"Date range: {self.data['Date'].min()} to {self.data['Date'].max()}")
        returns_stats = self.data['Returns'].describe()
        logger.info(f"Returns statistics: mean={returns_stats['mean']:.6f} ({returns_stats['mean']*100:.4f}%), std={returns_stats['std']:.6f} ({returns_stats['std']*100:.4f}%)")
        logger.info(f"Returns range: {returns_stats['min']:.6f} to {returns_stats['max']:.6f}")
        
        # 특성 컬럼 식별 (회귀 타겟으로 변경)
        feature_columns = [col for col in self.data.columns if col not in ['Date', 'Target', 'Returns']]
        logger.info(f"Feature columns: {len(feature_columns)}")
        
        return self.data
        
    def create_walk_forward_splits(self, 
                                   train_window_months: int = 12,
                                   test_window_months: int = 1,
                                   step_months: int = 1,
                                   min_train_samples: int = 200) -> List[Dict]:
        """월 단위 Walk-Forward 분할 생성 (5년 데이터에 적합)
        
        Args:
            train_window_months: 훈련 윈도우 (개월)
            test_window_months: 테스트 윈도우 (개월)
            step_months: 슬라이딩 스텝 (개월)
            min_train_samples: 최소 훈련 샘플 수
        """
        logger.info(f"Creating Walk-Forward splits with {train_window_months}M train, {test_window_months}M test...")
        
        splits = []
        data_size = len(self.data)
        
        # 날짜 기반 분할
        start_date = self.data['Date'].min()
        end_date = self.data['Date'].max()
        
        current_date = start_date + pd.DateOffset(months=train_window_months)
        split_id = 1
        
        while current_date + pd.DateOffset(months=test_window_months) <= end_date:
            # 훈련 기간
            train_start_date = current_date - pd.DateOffset(months=train_window_months)
            train_end_date = current_date
            
            # 테스트 기간  
            test_start_date = current_date
            test_end_date = current_date + pd.DateOffset(months=test_window_months)
            
            # 인덱스 추출
            train_mask = (self.data['Date'] >= train_start_date) & (self.data['Date'] < train_end_date)
            test_mask = (self.data['Date'] >= test_start_date) & (self.data['Date'] < test_end_date)
            
            train_indices = self.data.index[train_mask].tolist()
            test_indices = self.data.index[test_mask].tolist()
            
            # 최소 샘플 수 확인
            if len(train_indices) < min_train_samples or len(test_indices) < 10:
                current_date += pd.DateOffset(months=step_months)
                continue
                
            split_info = {
                'split_id': split_id,
                'train_indices': train_indices,
                'test_indices': test_indices,
                'train_size': len(train_indices),
                'test_size': len(test_indices),
                'train_start_date': train_start_date.strftime('%Y-%m-%d'),
                'train_end_date': train_end_date.strftime('%Y-%m-%d'),
                'test_start_date': test_start_date.strftime('%Y-%m-%d'),
                'test_end_date': test_end_date.strftime('%Y-%m-%d')
            }
            
            splits.append(split_info)
            
            # 다음 위치로 이동
            current_date += pd.DateOffset(months=step_months)
            split_id += 1
            
        logger.info(f"Created {len(splits)} walk-forward splits")
        if splits:
            logger.info(f"Average train size: {np.mean([s['train_size'] for s in splits]):.1f}")
            logger.info(f"Average test size: {np.mean([s['test_size'] for s in splits]):.1f}")
            logger.info(f"First split: {splits[0]['train_start_date']} to {splits[0]['test_end_date']}")
            logger.info(f"Last split: {splits[-1]['train_start_date']} to {splits[-1]['test_end_date']}")
        
        return splits
        
    def prepare_features(self) -> List[str]:
        """특성 준비 (안전한 특성만 사용)"""
        exclude_cols = ['Date', 'Target']
        feature_columns = [col for col in self.data.columns if col not in exclude_cols]
        
        logger.info(f"Using {len(feature_columns)} features for training")
        logger.info(f"Feature categories breakdown:")
        
        # 특성 카테고리 분석
        categories = {
            'price': [f for f in feature_columns if any(k in f.lower() for k in ['price', 'close', 'open', 'high', 'low'])],
            'volume': [f for f in feature_columns if 'volume' in f.lower()],
            'technical': [f for f in feature_columns if any(k in f.lower() for k in ['sma', 'rsi', 'macd', 'bb', 'atr'])],
            'volatility': [f for f in feature_columns if 'volatility' in f.lower()],
            'other': []
        }
        
        # 분류되지 않은 특성들
        categorized = set()
        for cat_features in categories.values():
            categorized.update(cat_features)
        categories['other'] = [f for f in feature_columns if f not in categorized]
        
        for cat, features in categories.items():
            if features:
                logger.info(f"  {cat}: {len(features)} features")
                
        return feature_columns
        
    def validate_model(self, model_class, model_params: Dict, features: List[str],
                      splits: List[Dict], model_name: str) -> Dict:
        """모델에 대한 Walk-Forward Validation 수행"""
        logger.info(f"Starting Walk-Forward validation for {model_name}...")
        
        results = {
            'model_name': model_name,
            'model_class': model_class.__name__,
            'features': features,
            'feature_count': len(features),
            'splits_count': len(splits),
            'split_results': [],
            'overall_metrics': {},
            'time_series_predictions': []
        }
        
        all_predictions = []
        all_actuals = []
        all_probabilities = []
        all_dates = []
        split_metrics = []
        
        # 스케일링 여부 결정
        use_scaler = model_name in ['LogisticRegression']
        
        for i, split in enumerate(splits):
            try:
                logger.info(f"Processing split {split['split_id']}/{len(splits)} ({split['test_start_date']})")
                
                # 데이터 분할
                train_idx = split['train_indices']
                test_idx = split['test_indices']
                
                X_train = self.data.loc[train_idx, features].copy()
                y_train = self.data.loc[train_idx, 'Target'].copy()
                X_test = self.data.loc[test_idx, features].copy()
                y_test = self.data.loc[test_idx, 'Target'].copy()
                
                # 데이터 검증
                if len(X_train) == 0 or len(X_test) == 0:
                    logger.warning(f"Empty data in split {split['split_id']}, skipping...")
                    continue
                    
                # 스케일링 (필요한 경우)
                if use_scaler:
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    X_train = pd.DataFrame(X_train_scaled, columns=features, index=X_train.index)
                    X_test = pd.DataFrame(X_test_scaled, columns=features, index=X_test.index)
                
                # 모델 훈련
                model = model_class(**model_params)
                model.fit(X_train, y_train)
                
                # 예측
                y_pred = model.predict(X_test)
                y_pred_proba = None
                
                if hasattr(model, 'predict_proba'):
                    try:
                        y_pred_proba = model.predict_proba(X_test)[:, 1]
                    except:
                        y_pred_proba = None
                
                # 메트릭 계산
                metrics = {
                    'split_id': split['split_id'],
                    'test_start_date': split['test_start_date'],
                    'test_end_date': split['test_end_date'],
                    'train_size': len(X_train),
                    'test_size': len(X_test),
                    'target_mean': float(y_test.mean()),
                    'target_std': float(y_test.std()),
                    'mse': float(mean_squared_error(y_test, y_pred)),
                    'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
                    'mae': float(mean_absolute_error(y_test, y_pred)),
                    'r2': float(r2_score(y_test, y_pred))
                }
                
                # MAPE 계산 (영나누기 방지)
                mask = np.abs(y_test) > 1e-8
                if mask.sum() > 0:
                    metrics['mape'] = float(np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100)
                else:
                    metrics['mape'] = 0.0
                
                results['split_results'].append(metrics)
                split_metrics.append(metrics)
                
                # 전체 결과 누적
                all_predictions.extend(y_pred.tolist())
                all_actuals.extend(y_test.tolist())
                
                if y_pred_proba is not None:
                    all_probabilities.extend(y_pred_proba.tolist())
                else:
                    all_probabilities.extend([0.5] * len(y_pred))
                
                # 날짜 정보 추가
                test_dates = self.data.loc[test_idx, 'Date'].dt.strftime('%Y-%m-%d').tolist()
                all_dates.extend(test_dates)
                
                # 시계열 예측 결과 저장
                for j, (date, actual, pred, prob) in enumerate(zip(test_dates, y_test, y_pred, 
                                                                   y_pred_proba if y_pred_proba is not None else [0.5] * len(y_pred))):
                    results['time_series_predictions'].append({
                        'date': date,
                        'actual': int(actual),
                        'predicted': int(pred),
                        'probability': float(prob),
                        'correct': int(actual == pred),
                        'split_id': split['split_id']
                    })
                    
            except Exception as e:
                logger.error(f"Split {split['split_id']} failed: {str(e)}")
                continue
                
        # 전체 회귀 성능 계산
        if all_predictions and all_actuals:
            all_actuals = np.array(all_actuals)
            all_predictions = np.array(all_predictions)
            
            # MAPE 계산
            mask = np.abs(all_actuals) > 1e-8
            if mask.sum() > 0:
                overall_mape = np.mean(np.abs((all_actuals[mask] - all_predictions[mask]) / all_actuals[mask])) * 100
            else:
                overall_mape = 0.0
                
            overall_metrics = {
                'mse': float(mean_squared_error(all_actuals, all_predictions)),
                'rmse': float(np.sqrt(mean_squared_error(all_actuals, all_predictions))),
                'mae': float(mean_absolute_error(all_actuals, all_predictions)),
                'mape': float(overall_mape),
                'r2': float(r2_score(all_actuals, all_predictions)),
                'total_predictions': len(all_predictions),
                'target_mean': float(np.mean(all_actuals)),
                'target_std': float(np.std(all_actuals)),
                'prediction_mean': float(np.mean(all_predictions)),
                'prediction_std': float(np.std(all_predictions)),
                'baseline_mse': float(np.var(all_actuals))  # 분산을 베이스라인으로 사용
            }
                
            results['overall_metrics'] = overall_metrics
            
            # 안정성 메트릭 계산 (회귀)
            if split_metrics:
                split_mapes = [m.get('mape', 0) for m in split_metrics]
                split_r2s = [m.get('r2', 0) for m in split_metrics]
                split_maes = [m.get('mae', 0) for m in split_metrics]
                
                results['stability_metrics'] = {
                    'mean_mape': float(np.mean(split_mapes)),
                    'std_mape': float(np.std(split_mapes)),
                    'mean_r2': float(np.mean(split_r2s)),
                    'std_r2': float(np.std(split_r2s)),
                    'mean_mae': float(np.mean(split_maes)),
                    'std_mae': float(np.std(split_maes)),
                    'cv_mape': float(np.std(split_mapes) / np.mean(split_mapes)) if np.mean(split_mapes) > 0 else float('inf'),
                    'cv_r2': float(np.std(split_r2s) / np.mean(split_r2s)) if np.mean(split_r2s) > 0 else float('inf')
                }
                
        logger.info(f"Walk-Forward validation completed for {model_name}")
        if 'overall_metrics' in results:
            logger.info(f"Overall accuracy: {results['overall_metrics']['accuracy']:.4f}")
            logger.info(f"Baseline accuracy: {results['overall_metrics']['baseline_accuracy']:.4f}")
            
        return results
        
    def run_complete_validation(self, 
                               train_window_months: int = 12,
                               test_window_months: int = 1,
                               step_months: int = 1) -> Dict:
        """전체 Walk-Forward Validation 실행"""
        logger.info("Starting complete Walk-Forward validation on extended dataset...")
        
        # 데이터 로드
        self.load_extended_data()
        
        # Walk-Forward 분할 생성
        splits = self.create_walk_forward_splits(
            train_window_months=train_window_months,
            test_window_months=test_window_months,
            step_months=step_months
        )
        
        if not splits:
            raise ValueError("No valid splits created. Check data and parameters.")
        
        # 특성 준비
        features = self.prepare_features()
        
        # 모델 설정 (현실적인 하이퍼파라미터)
        models_config = {
            'RandomForest': {
                'class': RandomForestClassifier,
                'params': {
                    'n_estimators': 50,  # 줄임 (과적합 방지)
                    'max_depth': 10,     # 제한 (과적합 방지)
                    'min_samples_split': 20,
                    'min_samples_leaf': 10,
                    'random_state': 42,
                    'n_jobs': -1,
                    'class_weight': 'balanced'  # 불균형 데이터 대응
                }
            },
            'GradientBoosting': {
                'class': GradientBoostingClassifier,
                'params': {
                    'n_estimators': 50,  # 줄임
                    'max_depth': 6,      # 제한
                    'learning_rate': 0.1,
                    'min_samples_split': 20,
                    'min_samples_leaf': 10,
                    'random_state': 42,
                    'subsample': 0.8     # 과적합 방지
                }
            },
            'XGBoost': {
                'class': xgb.XGBClassifier,
                'params': {
                    'n_estimators': 50,  # 줄임
                    'max_depth': 6,      # 제한
                    'learning_rate': 0.1,
                    'min_child_weight': 10,  # 과적합 방지
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42,
                    'eval_metric': 'logloss',
                    'use_label_encoder': False,
                    'scale_pos_weight': len(self.data[self.data['Target'] == 0]) / len(self.data[self.data['Target'] == 1])  # 불균형 대응
                }
            },
            'LogisticRegression': {
                'class': LogisticRegression,
                'params': {
                    'random_state': 42,
                    'max_iter': 1000,
                    'C': 1.0,  # 정규화 강화
                    'class_weight': 'balanced'
                }
            }
        }
        
        # 각 모델에 대해 검증 실행
        validation_results = {}
        
        for model_name, config in models_config.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Validating {model_name}")
            logger.info(f"{'='*50}")
            
            try:
                model_results = self.validate_model(
                    config['class'],
                    config['params'],
                    features,
                    splits,
                    model_name
                )
                validation_results[model_name] = model_results
                
            except Exception as e:
                logger.error(f"Model {model_name} validation failed: {str(e)}")
                continue
        
        # 결과 통합
        complete_results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'data_file': self.data_file,
                'total_samples': len(self.data),
                'date_range': {
                    'start': self.data['Date'].min().isoformat(),
                    'end': self.data['Date'].max().isoformat()
                },
                'target_distribution': self.data['Target'].value_counts().to_dict(),
                'validation_config': {
                    'train_window_months': train_window_months,
                    'test_window_months': test_window_months,
                    'step_months': step_months,
                    'total_splits': len(splits)
                }
            },
            'model_results': validation_results,
            'splits_info': splits
        }
        
        # 결과 저장
        self.save_results(complete_results)
        
        # 요약 리포트 출력
        self.print_summary(complete_results)
        
        logger.info("Extended Walk-Forward validation completed successfully!")
        
        return complete_results
        
    def save_results(self, results: Dict, filename: str = "extended_walk_forward_results.json"):
        """결과를 JSON으로 저장"""
        output_file = os.path.join(self.results_dir, filename)
        
        # JSON 직렬화 가능하도록 변환
        serializable_results = self._convert_to_serializable(results)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Extended Walk-Forward results saved: {output_file}")
        return output_file
        
    def _convert_to_serializable(self, obj):
        """JSON 직렬화 가능한 형태로 변환"""
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.int_)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float_)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj):
            return None
        else:
            return obj
            
    def print_summary(self, results: Dict):
        """결과 요약 출력"""
        print("\n" + "="*80)
        print("📊 EXTENDED WALK-FORWARD VALIDATION SUMMARY")
        print("="*80)
        
        metadata = results['metadata']
        print(f"📅 Data Period: {metadata['date_range']['start'][:10]} to {metadata['date_range']['end'][:10]}")
        print(f"📈 Total Samples: {metadata['total_samples']:,}")
        print(f"🎯 Target Distribution: {metadata['target_distribution']}")
        print(f"⚡ Validation Splits: {metadata['validation_config']['total_splits']}")
        print(f"🔄 Training Window: {metadata['validation_config']['train_window_months']} months")
        
        print("\n" + "="*80)
        print("🤖 MODEL PERFORMANCE COMPARISON")
        print("="*80)
        
        model_results = results['model_results']
        performance_data = []
        
        for model_name, model_result in model_results.items():
            if 'overall_metrics' not in model_result:
                continue
                
            metrics = model_result['overall_metrics']
            stability = model_result.get('stability_metrics', {})
            
            print(f"\n🔹 {model_name.upper()}")
            print("-" * 50)
            print(f"  Accuracy:          {metrics['accuracy']:.4f}")
            print(f"  Precision:         {metrics['precision']:.4f}")
            print(f"  Recall:            {metrics['recall']:.4f}")
            print(f"  F1-Score:          {metrics['f1_score']:.4f}")
            print(f"  AUC:               {metrics['auc']:.4f}")
            print(f"  Baseline Accuracy: {metrics['baseline_accuracy']:.4f}")
            
            if stability:
                print(f"  Mean Accuracy:     {stability['mean_accuracy']:.4f} ± {stability['std_accuracy']:.4f}")
                print(f"  Accuracy Range:    {stability['min_accuracy']:.4f} - {stability['max_accuracy']:.4f}")
                print(f"  Stability (CV):    {stability['cv_accuracy']:.4f}")
            
            performance_data.append({
                'model': model_name,
                'accuracy': metrics['accuracy'],
                'f1_score': metrics['f1_score'],
                'stability': stability.get('std_accuracy', float('inf'))
            })
        
        if performance_data:
            # 최고 성능 모델
            best_accuracy = max(performance_data, key=lambda x: x['accuracy'])
            best_f1 = max(performance_data, key=lambda x: x['f1_score'])
            most_stable = min(performance_data, key=lambda x: x['stability'])
            
            print("\n" + "="*80)
            print("🏆 PERFORMANCE HIGHLIGHTS")
            print("="*80)
            print(f"🎯 Best Accuracy:  {best_accuracy['model']} ({best_accuracy['accuracy']:.4f})")
            print(f"⚡ Best F1-Score:  {best_f1['model']} ({best_f1['f1_score']:.4f})")
            print(f"🔄 Most Stable:    {most_stable['model']} (std: {most_stable['stability']:.4f})")
        
        print("\n" + "="*80)
        print("✅ Extended Walk-Forward Validation Complete!")
        print("📊 Realistic performance metrics with proper time-series validation")
        print("🚫 Data leakage eliminated through temporal separation")
        print("="*80)


def main():
    """메인 실행 함수"""
    # 확장된 Walk-Forward Validation 실행
    validator = ExtendedWalkForwardValidator()
    
    try:
        # 검증 실행 (12개월 훈련, 1개월 테스트, 1개월 단위 이동)
        results = validator.run_complete_validation(
            train_window_months=12,
            test_window_months=1,
            step_months=1
        )
        
        print(f"\n✅ Extended Walk-Forward Validation completed!")
        print(f"📁 Results saved in: {validator.results_dir}")
        
        return results
        
    except Exception as e:
        logger.error(f"Extended Walk-Forward validation failed: {e}")
        raise


if __name__ == "__main__":
    main()