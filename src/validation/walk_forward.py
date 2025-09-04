#!/usr/bin/env python3
"""
Walk-Forward Validation 모듈
시계열 데이터에 최적화된 검증으로 더 현실적인 성능 평가
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """Walk-Forward Validation 구현 클래스"""
    
    def __init__(self, data_file: str = "data/raw/integrated_spy_news_data.csv",
                 results_dir: str = "results"):
        self.data_file = data_file
        self.results_dir = results_dir
        self.data = None
        self.results = {}
        
        # 결과 저장 디렉토리 생성
        os.makedirs(results_dir, exist_ok=True)
        
    def load_data(self):
        """데이터 로드 및 전처리"""
        logger.info(f"Loading data from {self.data_file}")
        self.data = pd.read_csv(self.data_file)
        
        # 날짜 컬럼 처리
        if 'date' in self.data.columns:
            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data = self.data.sort_values('date').reset_index(drop=True)
        else:
            logger.warning("No date column found, using index as time order")
            self.data = self.data.reset_index(drop=True)
            
        # NaN 값 제거
        self.data = self.data.dropna()
        
        logger.info(f"Data loaded: {len(self.data)} samples")
        logger.info(f"Date range: {self.data['date'].min()} to {self.data['date'].max()}")
        
    def create_walk_forward_splits(self, train_window: int = 30, test_window: int = 5, 
                                   step_size: int = 1, min_train_size: int = 20) -> List[Dict]:
        """Walk-Forward 분할 생성
        
        Args:
            train_window: 훈련 데이터 윈도우 크기 (일수)
            test_window: 테스트 데이터 윈도우 크기 (일수) 
            step_size: 슬라이딩 스텝 크기 (일수)
            min_train_size: 최소 훈련 데이터 크기
            
        Returns:
            분할 정보 리스트
        """
        splits = []
        data_size = len(self.data)
        
        # 시작 위치에서 충분한 훈련 데이터 확보
        start_idx = max(train_window, min_train_size)
        
        current_idx = start_idx
        split_id = 1
        
        while current_idx + test_window <= data_size:
            # 훈련 데이터 인덱스 범위
            train_start = max(0, current_idx - train_window)
            train_end = current_idx
            
            # 테스트 데이터 인덱스 범위  
            test_start = current_idx
            test_end = min(current_idx + test_window, data_size)
            
            # 실제 테스트 윈도우가 너무 작으면 중단
            if test_end - test_start < test_window:
                break
                
            # 날짜 정보 추가
            split_info = {
                'split_id': split_id,
                'train_indices': list(range(train_start, train_end)),
                'test_indices': list(range(test_start, test_end)),
                'train_size': train_end - train_start,
                'test_size': test_end - test_start
            }
            
            # 날짜 정보가 있는 경우 추가
            if 'date' in self.data.columns:
                split_info.update({
                    'train_start_date': self.data.loc[train_start, 'date'].strftime('%Y-%m-%d'),
                    'train_end_date': self.data.loc[train_end-1, 'date'].strftime('%Y-%m-%d'),
                    'test_start_date': self.data.loc[test_start, 'date'].strftime('%Y-%m-%d'),
                    'test_end_date': self.data.loc[test_end-1, 'date'].strftime('%Y-%m-%d')
                })
                
            splits.append(split_info)
            
            # 다음 위치로 이동
            current_idx += step_size
            split_id += 1
            
        logger.info(f"Created {len(splits)} walk-forward splits")
        logger.info(f"Average train size: {np.mean([s['train_size'] for s in splits]):.1f}")
        logger.info(f"Average test size: {np.mean([s['test_size'] for s in splits]):.1f}")
        
        return splits
        
    def prepare_features(self, exclude_cols: List[str] = None) -> Tuple[List[str], List[str]]:
        """특성 준비 (뉴스 vs 기술적 지표 분류)"""
        if exclude_cols is None:
            exclude_cols = ['date', 'target']
            
        # 뉴스 관련 키워드
        news_keywords = ['sentiment', 'news', 'article', 'impact']
        
        # 뉴스 특성 분류
        news_features = [col for col in self.data.columns 
                        if any(keyword in col.lower() for keyword in news_keywords) 
                        and col not in exclude_cols]
        
        # 기술적 지표 분류
        technical_features = [col for col in self.data.columns 
                             if col not in exclude_cols 
                             and col not in news_features]
        
        logger.info(f"Technical features ({len(technical_features)}): {technical_features}")
        logger.info(f"News features ({len(news_features)}): {news_features}")
        
        return technical_features, news_features
        
    def validate_model(self, model_class, model_params: Dict, features: List[str],
                      splits: List[Dict], model_name: str) -> Dict:
        """특정 모델에 대한 Walk-Forward Validation 수행"""
        logger.info(f"Starting Walk-Forward validation for {model_name}")
        
        results = {
            'model_name': model_name,
            'model_class': model_class.__name__,
            'features': features,
            'feature_count': len(features),
            'splits_count': len(splits),
            'split_results': [],
            'overall_metrics': {},
            'time_series_metrics': []
        }
        
        all_predictions = []
        all_actuals = []
        all_dates = []
        
        # 표준화 스케일러 (로지스틱 회귀의 경우만 사용)
        use_scaler = 'Logistic' in model_name
        
        for split in splits:
            try:
                # 훈련/테스트 데이터 분할
                train_idx = split['train_indices'] 
                test_idx = split['test_indices']
                
                X_train = self.data.loc[train_idx, features]
                y_train = self.data.loc[train_idx, 'target']
                X_test = self.data.loc[test_idx, features]
                y_test = self.data.loc[test_idx, 'target']
                
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
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # 성능 계산
                metrics = {
                    'split_id': split['split_id'],
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, zero_division=0),
                    'recall': recall_score(y_test, y_pred, zero_division=0),
                    'f1_score': f1_score(y_test, y_pred, zero_division=0)
                }
                
                if y_pred_proba is not None:
                    try:
                        metrics['auc'] = roc_auc_score(y_test, y_pred_proba)
                    except:
                        metrics['auc'] = 0.5
                else:
                    metrics['auc'] = 0.5
                    
                # 날짜 정보 추가
                if 'test_start_date' in split:
                    metrics['test_date'] = split['test_start_date']
                    
                results['split_results'].append(metrics)
                
                # 전체 결과 누적
                all_predictions.extend(y_pred.tolist())
                all_actuals.extend(y_test.tolist())
                
                if 'date' in self.data.columns:
                    test_dates = self.data.loc[test_idx, 'date'].dt.strftime('%Y-%m-%d').tolist()
                    all_dates.extend(test_dates)
                    
            except Exception as e:
                logger.warning(f"Split {split['split_id']} failed: {str(e)}")
                continue
                
        # 전체 성능 계산
        if all_predictions and all_actuals:
            overall_metrics = {
                'accuracy': accuracy_score(all_actuals, all_predictions),
                'precision': precision_score(all_actuals, all_predictions, zero_division=0),
                'recall': recall_score(all_actuals, all_predictions, zero_division=0),
                'f1_score': f1_score(all_actuals, all_predictions, zero_division=0),
                'total_predictions': len(all_predictions)
            }
            
            results['overall_metrics'] = overall_metrics
            
            # 시계열 성능 추이
            results['time_series_metrics'] = [
                {
                    'date': all_dates[i] if all_dates else f"sample_{i}",
                    'prediction': int(all_predictions[i]),
                    'actual': int(all_actuals[i]),
                    'correct': int(all_predictions[i] == all_actuals[i])
                }
                for i in range(len(all_predictions))
            ]
            
        logger.info(f"Walk-Forward validation completed for {model_name}")
        logger.info(f"Overall accuracy: {overall_metrics.get('accuracy', 0):.4f}")
        
        return results
        
    def compare_validation_methods(self, models_config: Dict, train_window: int = 30, 
                                   test_window: int = 5) -> Dict:
        """Walk-Forward vs 기존 검증 방법 비교"""
        logger.info("Comparing validation methods...")
        
        # 특성 준비
        technical_features, news_features = self.prepare_features()
        all_features = technical_features + news_features
        
        # Walk-Forward 분할 생성
        wf_splits = self.create_walk_forward_splits(
            train_window=train_window, 
            test_window=test_window
        )
        
        comparison_results = {
            'validation_comparison': {
                'walk_forward': {},
                'traditional': {}
            },
            'feature_comparison': {
                'baseline': {},  # 기술적 지표만
                'enhanced': {}   # 기술적 + 뉴스
            }
        }
        
        # 모델별 검증 수행
        for model_name, config in models_config.items():
            model_class = config['class']
            model_params = config['params']
            
            # Baseline (기술적 지표만) - Walk-Forward
            baseline_results = self.validate_model(
                model_class, model_params, technical_features,
                wf_splits, f"Baseline_{model_name}"
            )
            
            # Enhanced (전체 특성) - Walk-Forward  
            enhanced_results = self.validate_model(
                model_class, model_params, all_features,
                wf_splits, f"Enhanced_{model_name}"
            )
            
            comparison_results['validation_comparison']['walk_forward'][f"Baseline_{model_name}"] = baseline_results
            comparison_results['validation_comparison']['walk_forward'][f"Enhanced_{model_name}"] = enhanced_results
            
            # 기존 방법과 비교를 위한 단일 분할 테스트
            train_size = int(len(self.data) * 0.8)
            train_idx = list(range(train_size))
            test_idx = list(range(train_size, len(self.data)))
            
            single_split = [{
                'split_id': 1,
                'train_indices': train_idx,
                'test_indices': test_idx,
                'train_size': len(train_idx),
                'test_size': len(test_idx)
            }]
            
            # Traditional 검증
            baseline_traditional = self.validate_model(
                model_class, model_params, technical_features,
                single_split, f"Traditional_Baseline_{model_name}"
            )
            
            enhanced_traditional = self.validate_model(
                model_class, model_params, all_features,
                single_split, f"Traditional_Enhanced_{model_name}"
            )
            
            comparison_results['validation_comparison']['traditional'][f"Baseline_{model_name}"] = baseline_traditional
            comparison_results['validation_comparison']['traditional'][f"Enhanced_{model_name}"] = enhanced_traditional
            
        return comparison_results
        
    def analyze_temporal_stability(self, results: Dict) -> Dict:
        """시간대별 모델 안정성 분석"""
        stability_analysis = {}
        
        for method, method_results in results['validation_comparison'].items():
            stability_analysis[method] = {}
            
            for model_name, model_results in method_results.items():
                if 'split_results' not in model_results:
                    continue
                    
                split_accuracies = [s['accuracy'] for s in model_results['split_results']]
                
                stability_metrics = {
                    'mean_accuracy': np.mean(split_accuracies),
                    'std_accuracy': np.std(split_accuracies),
                    'min_accuracy': np.min(split_accuracies),
                    'max_accuracy': np.max(split_accuracies),
                    'stability_score': 1 - (np.std(split_accuracies) / np.mean(split_accuracies)) if np.mean(split_accuracies) > 0 else 0,
                    'coefficient_of_variation': np.std(split_accuracies) / np.mean(split_accuracies) if np.mean(split_accuracies) > 0 else float('inf')
                }
                
                stability_analysis[method][model_name] = stability_metrics
                
        return stability_analysis
        
    def create_performance_plots(self, results: Dict, model_name: str = None):
        """성능 추이 시각화"""
        if not model_name:
            # 첫 번째 Enhanced 모델 선택
            for name in results['validation_comparison']['walk_forward'].keys():
                if 'Enhanced' in name:
                    model_name = name
                    break
                    
        if not model_name:
            logger.warning("No model found for plotting")
            return None
            
        model_results = results['validation_comparison']['walk_forward'][model_name]
        split_results = model_results['split_results']
        
        if not split_results:
            logger.warning(f"No split results found for {model_name}")
            return None
            
        # 성능 추이 그래프
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        splits = [s['split_id'] for s in split_results]
        accuracies = [s['accuracy'] for s in split_results]
        precisions = [s['precision'] for s in split_results]
        recalls = [s['recall'] for s in split_results]
        f1_scores = [s['f1_score'] for s in split_results]
        
        # 정확도 추이
        ax1.plot(splits, accuracies, 'b-o', linewidth=2, markersize=4)
        ax1.set_title('정확도 시간 추이', fontsize=12, fontweight='bold')
        ax1.set_xlabel('분할 번호')
        ax1.set_ylabel('정확도')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=np.mean(accuracies), color='r', linestyle='--', alpha=0.7, label=f'평균: {np.mean(accuracies):.3f}')
        ax1.legend()
        
        # 정밀도 추이
        ax2.plot(splits, precisions, 'g-s', linewidth=2, markersize=4)
        ax2.set_title('정밀도 시간 추이', fontsize=12, fontweight='bold')
        ax2.set_xlabel('분할 번호')
        ax2.set_ylabel('정밀도')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=np.mean(precisions), color='r', linestyle='--', alpha=0.7, label=f'평균: {np.mean(precisions):.3f}')
        ax2.legend()
        
        # 재현율 추이
        ax3.plot(splits, recalls, 'm-^', linewidth=2, markersize=4)
        ax3.set_title('재현율 시간 추이', fontsize=12, fontweight='bold')
        ax3.set_xlabel('분할 번호')
        ax3.set_ylabel('재현율')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=np.mean(recalls), color='r', linestyle='--', alpha=0.7, label=f'평균: {np.mean(recalls):.3f}')
        ax3.legend()
        
        # F1 점수 추이
        ax4.plot(splits, f1_scores, 'c-d', linewidth=2, markersize=4)
        ax4.set_title('F1 점수 시간 추이', fontsize=12, fontweight='bold')
        ax4.set_xlabel('분할 번호')
        ax4.set_ylabel('F1 점수')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=np.mean(f1_scores), color='r', linestyle='--', alpha=0.7, label=f'평균: {np.mean(f1_scores):.3f}')
        ax4.legend()
        
        plt.suptitle(f'Walk-Forward Validation 성능 추이: {model_name}', fontsize=14, fontweight='bold', y=0.95)
        plt.tight_layout()
        
        # 파일 저장
        plot_file = os.path.join(self.results_dir, f'walk_forward_performance_{model_name.lower()}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance plot saved: {plot_file}")
        return plot_file
        
    def save_results(self, results: Dict, output_file: str = None):
        """결과를 JSON으로 저장"""
        if not output_file:
            output_file = os.path.join(self.results_dir, 'walk_forward_validation_results.json')
            
        # 안정성 분석 추가
        stability_analysis = self.analyze_temporal_stability(results)
        results['stability_analysis'] = stability_analysis
        
        # 메타 정보 추가
        results['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'total_samples': len(self.data) if self.data is not None else 0,
            'validation_date_range': {
                'start': self.data['date'].min().isoformat() if 'date' in self.data.columns else None,
                'end': self.data['date'].max().isoformat() if 'date' in self.data.columns else None
            }
        }
        
        # JSON 직렬화 가능한 형태로 변환
        results_serializable = self._convert_to_json_serializable(results)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Walk-Forward validation results saved: {output_file}")
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
        elif pd.isna(obj):
            return None
        else:
            return obj
            
    def run_complete_validation(self, train_window: int = 30, test_window: int = 5):
        """전체 Walk-Forward Validation 실행"""
        logger.info("Starting complete Walk-Forward validation")
        
        # 데이터 로드
        self.load_data()
        
        # 모델 설정
        models_config = {
            'RandomForest': {
                'class': RandomForestClassifier,
                'params': {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1}
            },
            'GradientBoosting': {
                'class': GradientBoostingClassifier, 
                'params': {'n_estimators': 100, 'random_state': 42, 'learning_rate': 0.1}
            },
            'LogisticRegression': {
                'class': LogisticRegression,
                'params': {'random_state': 42, 'max_iter': 1000}
            }
        }
        
        # 검증 실행
        results = self.compare_validation_methods(
            models_config, 
            train_window=train_window,
            test_window=test_window
        )
        
        # 시각화 생성
        self.create_performance_plots(results)
        
        # 결과 저장
        output_file = self.save_results(results)
        
        # 요약 출력
        self._print_summary(results)
        
        logger.info("Walk-Forward validation completed")
        return output_file
        
    def _print_summary(self, results: Dict):
        """결과 요약 출력"""
        print("\n" + "="*60)
        print("📊 WALK-FORWARD VALIDATION 결과 요약")
        print("="*60)
        
        # Walk-Forward vs Traditional 비교
        wf_results = results['validation_comparison']['walk_forward']
        trad_results = results['validation_comparison']['traditional']
        
        print("\n🔍 검증 방법 비교:")
        print("-" * 40)
        
        for model_type in ['Enhanced_GradientBoosting', 'Enhanced_RandomForest']:
            if model_type in wf_results and model_type in trad_results:
                wf_acc = wf_results[model_type]['overall_metrics']['accuracy']
                trad_acc = trad_results[model_type]['overall_metrics']['accuracy'] 
                
                print(f"{model_type}:")
                print(f"  Walk-Forward: {wf_acc:.4f}")
                print(f"  Traditional:  {trad_acc:.4f}")
                print(f"  차이: {wf_acc - trad_acc:+.4f}")
                print()
                
        # 안정성 분석
        if 'stability_analysis' in results:
            print("\n📈 모델 안정성 분석:")
            print("-" * 40)
            
            wf_stability = results['stability_analysis'].get('walk_forward', {})
            for model_name, metrics in wf_stability.items():
                if 'Enhanced' in model_name:
                    print(f"{model_name}:")
                    print(f"  평균 정확도: {metrics['mean_accuracy']:.4f}")
                    print(f"  표준편차: {metrics['std_accuracy']:.4f}")
                    print(f"  안정성 점수: {metrics['stability_score']:.4f}")
                    print()


def main():
    """메인 실행 함수"""
    validator = WalkForwardValidator()
    
    # Walk-Forward Validation 실행
    result_file = validator.run_complete_validation(
        train_window=30,  # 30일 훈련 윈도우
        test_window=5     # 5일 테스트 윈도우
    )
    
    print(f"\n✅ Walk-Forward Validation 완료!")
    print(f"📁 결과 파일: {result_file}")


if __name__ == "__main__":
    main()