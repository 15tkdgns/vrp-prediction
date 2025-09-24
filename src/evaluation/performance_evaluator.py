#!/usr/bin/env python3
"""
성능 평가 시스템
모델 성능 평가 및 비교 분석
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.model_selection import TimeSeriesSplit
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceEvaluator:
    """종합 성능 평가 시스템"""

    def __init__(self):
        self.evaluation_results = {}

    def evaluate_classification_performance(self, y_true, y_pred, y_proba=None, model_name="model"):
        """분류 성능 평가"""
        # 기본 분류 지표
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # 방향 정확도 (특별히 중요)
        direction_accuracy = accuracy

        results = {
            'accuracy': accuracy,
            'direction_accuracy': direction_accuracy * 100,  # 퍼센트로 변환
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }

        # AUC (확률이 있는 경우)
        if y_proba is not None:
            try:
                auc = roc_auc_score(y_true, y_proba)
                results['auc'] = auc
            except Exception:
                results['auc'] = None

        # 클래스별 성능
        class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        results['class_performance'] = class_report

        logger.info(f"{model_name} 성능:")
        logger.info(f"  방향 정확도: {direction_accuracy*100:.2f}%")
        logger.info(f"  정밀도: {precision:.4f}")
        logger.info(f"  재현율: {recall:.4f}")
        logger.info(f"  F1 점수: {f1:.4f}")

        return results

    def evaluate_regression_performance(self, y_true, y_pred, model_name="model"):
        """회귀 성능 평가"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # MAPE 계산 (0으로 나누기 방지)
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if np.any(mask) else np.inf

        results = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }

        logger.info(f"{model_name} 성능:")
        logger.info(f"  MSE: {mse:.6f}")
        logger.info(f"  RMSE: {rmse:.6f}")
        logger.info(f"  MAE: {mae:.6f}")
        logger.info(f"  R²: {r2:.4f}")
        logger.info(f"  MAPE: {mape:.2f}%")

        return results

    def cross_validate_model(self, model, X, y, cv_splits=5, task_type='classification'):
        """교차 검증 성능 평가"""
        tscv = TimeSeriesSplit(n_splits=cv_splits)

        cv_scores = []
        all_predictions = []
        all_true_values = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # 모델 타입에 따른 처리
            if hasattr(model, 'fit'):  # sklearn 모델
                # 2D 데이터로 변환 (필요시)
                X_train_flat = X_train.reshape(X_train.shape[0], -1) if len(X_train.shape) > 2 else X_train
                X_val_flat = X_val.reshape(X_val.shape[0], -1) if len(X_val.shape) > 2 else X_val

                model.fit(X_train_flat, y_train)
                y_pred = model.predict(X_val_flat)
            else:  # PyTorch 모델 등
                # 별도 처리 필요
                continue

            if task_type == 'classification':
                score = accuracy_score(y_val, y_pred)
            else:  # regression
                score = -mean_squared_error(y_val, y_pred)  # 음수로 변환 (높을수록 좋게)

            cv_scores.append(score)
            all_predictions.extend(y_pred)
            all_true_values.extend(y_val)

        # 전체 성능 평가
        if task_type == 'classification':
            overall_performance = self.evaluate_classification_performance(
                all_true_values, all_predictions, model_name=f"CV_{type(model).__name__}"
            )
        else:
            overall_performance = self.evaluate_regression_performance(
                all_true_values, all_predictions, model_name=f"CV_{type(model).__name__}"
            )

        cv_results = {
            'cv_scores': cv_scores,
            'mean_cv_score': np.mean(cv_scores),
            'std_cv_score': np.std(cv_scores),
            'overall_performance': overall_performance
        }

        return cv_results

    def compare_models(self, models_results):
        """모델 성능 비교"""
        comparison = []

        for model_name, results in models_results.items():
            if 'accuracy' in results:
                main_metric = results['accuracy']
                metric_name = 'accuracy'
            elif 'mse' in results:
                main_metric = -results['mse']  # MSE는 낮을수록 좋으므로 음수로 변환
                metric_name = 'neg_mse'
            else:
                continue

            comparison.append({
                'model_name': model_name,
                'main_metric': main_metric,
                'metric_name': metric_name,
                'full_results': results
            })

        # 성능순으로 정렬
        comparison.sort(key=lambda x: x['main_metric'], reverse=True)

        logger.info("모델 성능 순위:")
        for i, model_info in enumerate(comparison, 1):
            logger.info(f"{i}. {model_info['model_name']}: {model_info['main_metric']:.4f}")

        return comparison

    def calculate_ensemble_diversity(self, predictions_dict):
        """앙상블 다양성 계산"""
        model_names = list(predictions_dict.keys())
        predictions = [predictions_dict[name] for name in model_names]

        if len(predictions) < 2:
            return {'diversity_score': 0, 'pairwise_correlations': {}}

        # 예측간 상관관계 계산
        correlations = {}
        correlation_values = []

        for i in range(len(predictions)):
            for j in range(i+1, len(predictions)):
                corr = np.corrcoef(predictions[i], predictions[j])[0, 1]
                if not np.isnan(corr):
                    pair_name = f"{model_names[i]}_vs_{model_names[j]}"
                    correlations[pair_name] = corr
                    correlation_values.append(abs(corr))

        # 다양성 점수 (낮은 상관관계 = 높은 다양성)
        diversity_score = 1 - np.mean(correlation_values) if correlation_values else 0

        return {
            'diversity_score': diversity_score,
            'pairwise_correlations': correlations,
            'mean_abs_correlation': np.mean(correlation_values) if correlation_values else 0
        }

    def generate_performance_report(self, models_results, save_path=None):
        """종합 성능 리포트 생성"""
        report = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'models_evaluated': len(models_results),
            'individual_results': models_results,
            'model_ranking': self.compare_models(models_results)
        }

        # 최고 성능 모델
        if report['model_ranking']:
            best_model = report['model_ranking'][0]
            report['best_model'] = {
                'name': best_model['model_name'],
                'performance': best_model['main_metric'],
                'details': best_model['full_results']
            }

        # 성능 통계
        if models_results:
            if 'accuracy' in list(models_results.values())[0]:
                accuracies = [r.get('accuracy', 0) for r in models_results.values()]
                report['performance_statistics'] = {
                    'mean_accuracy': np.mean(accuracies),
                    'std_accuracy': np.std(accuracies),
                    'min_accuracy': np.min(accuracies),
                    'max_accuracy': np.max(accuracies)
                }

        # 저장
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"성능 리포트 저장: {save_path}")

        return report

    def print_summary(self, report):
        """성능 요약 출력"""
        print("\n" + "="*60)
        print("📊 모델 성능 평가 결과")
        print("="*60)

        if 'best_model' in report:
            best = report['best_model']
            print(f"\n🏆 최고 성능 모델: {best['name']}")

            if 'direction_accuracy' in best['details']:
                print(f"   방향 정확도: {best['details']['direction_accuracy']:.2f}%")
            elif 'accuracy' in best['details']:
                print(f"   정확도: {best['details']['accuracy']*100:.2f}%")

        print(f"\n📈 평가된 모델 수: {report['models_evaluated']}")

        if 'model_ranking' in report:
            print("\n순위:")
            for i, model in enumerate(report['model_ranking'][:5], 1):  # 상위 5개
                score = model['main_metric']
                if model['metric_name'] == 'accuracy':
                    print(f"  {i}. {model['model_name']}: {score*100:.2f}%")
                else:
                    print(f"  {i}. {model['model_name']}: {score:.4f}")

        print("="*60)

if __name__ == "__main__":
    # 테스트
    evaluator = PerformanceEvaluator()

    # 더미 데이터로 테스트
    y_true = np.random.choice([0, 1], 100)
    y_pred = np.random.choice([0, 1], 100)

    results = evaluator.evaluate_classification_performance(y_true, y_pred, model_name="test_model")
    print("✅ 성능 평가 시스템 정상 작동")
    print(f"테스트 정확도: {results['accuracy']:.4f}")