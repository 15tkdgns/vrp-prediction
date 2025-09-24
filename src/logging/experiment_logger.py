#!/usr/bin/env python3
"""
📝 실험 로거 시스템
학술 논문을 위한 완전한 실험 기록 및 추적 도구

주요 기능:
- 실험 설정 및 결과 자동 기록
- 하이퍼파라미터 추적
- 환경 정보 기록
- 실험 비교 및 분석
- JSON/CSV 형태로 저장
"""

import json
import csv
import pickle
import hashlib
import datetime
import os
import sys
import platform
import subprocess
import traceback
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict, field
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ExperimentConfig:
    """실험 설정 정보"""
    experiment_id: str
    experiment_name: str
    description: str
    model_type: str
    model_parameters: Dict[str, Any]
    data_info: Dict[str, Any]
    preprocessing_steps: List[str]
    validation_method: str
    random_seed: int
    tags: List[str] = field(default_factory=list)
    notes: str = ""

@dataclass
class SystemInfo:
    """시스템 환경 정보"""
    python_version: str
    platform: str
    cpu_count: int
    memory_gb: float
    gpu_info: Optional[str]
    installed_packages: Dict[str, str]
    git_commit: Optional[str]
    git_branch: Optional[str]
    working_directory: str
    timestamp: str

@dataclass
class ExperimentMetrics:
    """실험 성능 지표"""
    mae: float
    mse: float
    rmse: float
    r2: float
    direction_accuracy: float
    training_time: float
    prediction_time: float
    memory_usage: float
    custom_metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class ExperimentRecord:
    """완전한 실험 기록"""
    config: ExperimentConfig
    system_info: SystemInfo
    metrics: ExperimentMetrics
    predictions: Optional[List[float]] = None
    feature_importance: Optional[Dict[str, float]] = None
    validation_scores: Optional[Dict[str, float]] = None
    error_log: Optional[str] = None
    duration_seconds: float = 0.0
    status: str = "completed"  # completed, failed, running

class ExperimentLogger:
    """
    포괄적 실험 로깅 시스템

    모든 실험 과정을 자동으로 기록하고 추적
    """

    def __init__(self, log_dir: str = "experiments_log", auto_save: bool = True):
        """
        초기화

        Args:
            log_dir: 로그 저장 디렉토리
            auto_save: 자동 저장 여부
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True, parents=True)
        self.auto_save = auto_save

        # 현재 실험 기록
        self.current_experiment: Optional[ExperimentRecord] = None
        self.experiment_start_time: Optional[datetime.datetime] = None

        # 실험 히스토리
        self.experiment_history: List[ExperimentRecord] = []

        # 시스템 정보 수집
        self.system_info = self._collect_system_info()

    def start_experiment(self, experiment_name: str,
                        description: str,
                        model_type: str,
                        model_parameters: Dict[str, Any],
                        data_info: Dict[str, Any],
                        preprocessing_steps: List[str],
                        validation_method: str,
                        random_seed: int,
                        tags: Optional[List[str]] = None,
                        notes: str = "") -> str:
        """
        실험 시작 및 설정 기록

        Args:
            experiment_name: 실험 이름
            description: 실험 설명
            model_type: 모델 유형
            model_parameters: 모델 파라미터
            data_info: 데이터 정보
            preprocessing_steps: 전처리 단계
            validation_method: 검증 방법
            random_seed: 랜덤 시드
            tags: 실험 태그
            notes: 추가 노트

        Returns:
            실험 ID
        """
        self.experiment_start_time = datetime.datetime.now()

        # 실험 ID 생성 (시간 + 해시)
        experiment_id = self._generate_experiment_id(experiment_name, model_parameters)

        # 실험 설정 생성
        config = ExperimentConfig(
            experiment_id=experiment_id,
            experiment_name=experiment_name,
            description=description,
            model_type=model_type,
            model_parameters=model_parameters.copy(),
            data_info=data_info.copy(),
            preprocessing_steps=preprocessing_steps.copy(),
            validation_method=validation_method,
            random_seed=random_seed,
            tags=tags or [],
            notes=notes
        )

        # 현재 실험 기록 초기화
        self.current_experiment = ExperimentRecord(
            config=config,
            system_info=self.system_info,
            metrics=ExperimentMetrics(0, 0, 0, 0, 0, 0, 0, 0),  # 임시값
            status="running"
        )

        print(f"📝 실험 시작: {experiment_name} (ID: {experiment_id})")
        return experiment_id

    def log_metrics(self, mae: float, mse: float, rmse: float, r2: float,
                   direction_accuracy: float, training_time: float,
                   prediction_time: float, memory_usage: float,
                   custom_metrics: Optional[Dict[str, float]] = None):
        """
        성능 지표 기록

        Args:
            mae: Mean Absolute Error
            mse: Mean Squared Error
            rmse: Root Mean Squared Error
            r2: R-squared
            direction_accuracy: 방향 정확도
            training_time: 훈련 시간
            prediction_time: 예측 시간
            memory_usage: 메모리 사용량
            custom_metrics: 사용자 정의 지표
        """
        if self.current_experiment is None:
            raise ValueError("실험이 시작되지 않았습니다. start_experiment()를 먼저 호출하세요.")

        self.current_experiment.metrics = ExperimentMetrics(
            mae=mae,
            mse=mse,
            rmse=rmse,
            r2=r2,
            direction_accuracy=direction_accuracy,
            training_time=training_time,
            prediction_time=prediction_time,
            memory_usage=memory_usage,
            custom_metrics=custom_metrics or {}
        )

        print(f"📊 지표 기록 완료: MAE={mae:.6f}, R²={r2:.4f}, 방향정확도={direction_accuracy:.1f}%")

    def log_predictions(self, predictions: np.ndarray):
        """예측 결과 기록"""
        if self.current_experiment is None:
            raise ValueError("실험이 시작되지 않았습니다.")

        self.current_experiment.predictions = predictions.tolist()
        print(f"🔮 예측 결과 기록: {len(predictions)}개 예측값")

    def log_feature_importance(self, feature_importance: Dict[str, float]):
        """특징 중요도 기록"""
        if self.current_experiment is None:
            raise ValueError("실험이 시작되지 않았습니다.")

        self.current_experiment.feature_importance = feature_importance.copy()
        print(f"🎯 특징 중요도 기록: {len(feature_importance)}개 특징")

    def log_validation_scores(self, validation_scores: Dict[str, float]):
        """교차 검증 점수 기록"""
        if self.current_experiment is None:
            raise ValueError("실험이 시작되지 않았습니다.")

        self.current_experiment.validation_scores = validation_scores.copy()
        print(f"✅ 검증 점수 기록: {len(validation_scores)}개 폴드")

    def log_error(self, error: Exception):
        """오류 정보 기록"""
        if self.current_experiment is None:
            raise ValueError("실험이 시작되지 않았습니다.")

        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'timestamp': datetime.datetime.now().isoformat()
        }

        self.current_experiment.error_log = json.dumps(error_info)
        self.current_experiment.status = "failed"
        print(f"❌ 오류 기록: {type(error).__name__}")

    def end_experiment(self):
        """실험 종료 및 최종 저장"""
        if self.current_experiment is None:
            raise ValueError("실험이 시작되지 않았습니다.")

        if self.experiment_start_time is not None:
            duration = datetime.datetime.now() - self.experiment_start_time
            self.current_experiment.duration_seconds = duration.total_seconds()

        if self.current_experiment.status == "running":
            self.current_experiment.status = "completed"

        # 실험 히스토리에 추가
        self.experiment_history.append(self.current_experiment)

        # 자동 저장
        if self.auto_save:
            self._save_experiment(self.current_experiment)

        experiment_id = self.current_experiment.config.experiment_id
        print(f"✅ 실험 완료: {experiment_id} ({self.current_experiment.duration_seconds:.1f}초)")

        # 현재 실험 초기화
        self.current_experiment = None
        self.experiment_start_time = None

        return experiment_id

    def _generate_experiment_id(self, experiment_name: str, parameters: Dict[str, Any]) -> str:
        """실험 ID 생성"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # 파라미터 해시 생성
        param_str = json.dumps(parameters, sort_keys=True, default=str)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]

        return f"{experiment_name}_{timestamp}_{param_hash}"

    def _collect_system_info(self) -> SystemInfo:
        """시스템 정보 수집"""
        import psutil

        # GPU 정보 수집
        gpu_info = None
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_info = f"{gpus[0].name} ({gpus[0].memoryTotal}MB)"
        except ImportError:
            try:
                # nvidia-smi 사용
                result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total',
                                       '--format=csv,noheader,nounits'],
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    gpu_info = result.stdout.strip()
            except FileNotFoundError:
                pass

        # 설치된 패키지 정보
        try:
            import pkg_resources
            installed_packages = {pkg.project_name: pkg.version
                                for pkg in pkg_resources.working_set}
        except Exception:
            installed_packages = {}

        # Git 정보
        git_commit, git_branch = self._get_git_info()

        return SystemInfo(
            python_version=platform.python_version(),
            platform=platform.platform(),
            cpu_count=psutil.cpu_count(),
            memory_gb=psutil.virtual_memory().total / (1024**3),
            gpu_info=gpu_info,
            installed_packages=installed_packages,
            git_commit=git_commit,
            git_branch=git_branch,
            working_directory=os.getcwd(),
            timestamp=datetime.datetime.now().isoformat()
        )

    def _get_git_info(self) -> Tuple[Optional[str], Optional[str]]:
        """Git 정보 수집"""
        try:
            # Git commit hash
            commit_result = subprocess.run(['git', 'rev-parse', 'HEAD'],
                                         capture_output=True, text=True)
            commit = commit_result.stdout.strip() if commit_result.returncode == 0 else None

            # Git branch
            branch_result = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                                         capture_output=True, text=True)
            branch = branch_result.stdout.strip() if branch_result.returncode == 0 else None

            return commit, branch
        except FileNotFoundError:
            return None, None

    def _save_experiment(self, experiment: ExperimentRecord):
        """실험 기록 저장"""
        experiment_id = experiment.config.experiment_id

        # JSON 형태로 저장
        json_file = self.log_dir / f"{experiment_id}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(experiment), f, indent=2, ensure_ascii=False, default=str)

        # 바이너리 형태로도 저장 (예측값 등 대용량 데이터)
        pickle_file = self.log_dir / f"{experiment_id}.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(experiment, f)

        print(f"💾 실험 기록 저장: {json_file}")

    def load_experiment(self, experiment_id: str) -> ExperimentRecord:
        """실험 기록 로드"""
        pickle_file = self.log_dir / f"{experiment_id}.pkl"

        if pickle_file.exists():
            with open(pickle_file, 'rb') as f:
                return pickle.load(f)
        else:
            # JSON 파일에서 로드 (pickle이 없는 경우)
            json_file = self.log_dir / f"{experiment_id}.json"
            if json_file.exists():
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # 데이터클래스로 변환 (간단한 버전)
                # 실제로는 더 복잡한 변환 로직 필요
                return data
            else:
                raise FileNotFoundError(f"실험 {experiment_id}을 찾을 수 없습니다.")

    def list_experiments(self, tags: Optional[List[str]] = None,
                        model_type: Optional[str] = None,
                        status: Optional[str] = None) -> List[str]:
        """실험 목록 조회"""
        experiment_files = list(self.log_dir.glob("*.json"))
        experiments = []

        for file in experiment_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # 필터링
                if tags and not any(tag in data['config'].get('tags', []) for tag in tags):
                    continue

                if model_type and data['config'].get('model_type') != model_type:
                    continue

                if status and data.get('status') != status:
                    continue

                experiments.append(data['config']['experiment_id'])

            except Exception as e:
                print(f"실험 파일 {file} 로드 오류: {str(e)}")

        return experiments

    def compare_experiments(self, experiment_ids: List[str]) -> pd.DataFrame:
        """실험 비교 테이블 생성"""
        comparison_data = []

        for exp_id in experiment_ids:
            try:
                with open(self.log_dir / f"{exp_id}.json", 'r', encoding='utf-8') as f:
                    data = json.load(f)

                config = data['config']
                metrics = data['metrics']

                row = {
                    'experiment_id': exp_id,
                    'experiment_name': config['experiment_name'],
                    'model_type': config['model_type'],
                    'mae': metrics['mae'],
                    'rmse': metrics['rmse'],
                    'r2': metrics['r2'],
                    'direction_accuracy': metrics['direction_accuracy'],
                    'training_time': metrics['training_time'],
                    'status': data.get('status', 'unknown'),
                    'duration': data.get('duration_seconds', 0)
                }

                # 모델 파라미터 일부 추가
                for key, value in config['model_parameters'].items():
                    if isinstance(value, (int, float, str, bool)):
                        row[f"param_{key}"] = value

                comparison_data.append(row)

            except Exception as e:
                print(f"실험 {exp_id} 비교 오류: {str(e)}")

        return pd.DataFrame(comparison_data)

    def generate_experiment_summary(self) -> str:
        """실험 요약 보고서 생성"""
        experiments = self.list_experiments()

        if not experiments:
            return "기록된 실험이 없습니다."

        # 실험 통계
        total_experiments = len(experiments)
        completed_experiments = len(self.list_experiments(status="completed"))
        failed_experiments = len(self.list_experiments(status="failed"))

        # 성능 통계 (완료된 실험만)
        completed_exp_data = []
        for exp_id in self.list_experiments(status="completed"):
            try:
                with open(self.log_dir / f"{exp_id}.json", 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    completed_exp_data.append(data['metrics'])
            except Exception:
                continue

        report = [
            "📊 실험 요약 보고서",
            "=" * 50,
            "",
            f"📈 실험 통계:",
            f"   총 실험 수: {total_experiments}",
            f"   완료된 실험: {completed_experiments}",
            f"   실패한 실험: {failed_experiments}",
            f"   성공률: {completed_experiments/total_experiments*100:.1f}%" if total_experiments > 0 else "   성공률: N/A",
            ""
        ]

        if completed_exp_data:
            mae_values = [exp['mae'] for exp in completed_exp_data]
            r2_values = [exp['r2'] for exp in completed_exp_data]
            dir_acc_values = [exp['direction_accuracy'] for exp in completed_exp_data]

            report.extend([
                f"🎯 성능 통계 (완료된 실험):",
                f"   MAE - 평균: {np.mean(mae_values):.6f}, 최고: {np.min(mae_values):.6f}, 최저: {np.max(mae_values):.6f}",
                f"   R² - 평균: {np.mean(r2_values):.4f}, 최고: {np.max(r2_values):.4f}, 최저: {np.min(r2_values):.4f}",
                f"   방향정확도 - 평균: {np.mean(dir_acc_values):.1f}%, 최고: {np.max(dir_acc_values):.1f}%, 최저: {np.min(dir_acc_values):.1f}%",
                ""
            ])

        # 최근 실험들
        recent_experiments = sorted(experiments, reverse=True)[:5]
        report.extend([
            f"🕒 최근 실험 (최대 5개):",
        ])

        for exp_id in recent_experiments:
            try:
                with open(self.log_dir / f"{exp_id}.json", 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    config = data['config']
                    metrics = data['metrics']
                    status = data.get('status', 'unknown')

                    report.append(f"   - {config['experiment_name']}: MAE={metrics['mae']:.6f}, 상태={status}")
            except Exception:
                report.append(f"   - {exp_id}: 로드 오류")

        return "\n".join(report)

def main():
    """테스트 및 예제 실행"""
    print("📝 실험 로거 시스템 테스트")
    print("=" * 50)

    # 실험 로거 초기화
    logger = ExperimentLogger(log_dir="test_experiments_log")

    # 시뮬레이션 데이터 생성
    np.random.seed(42)
    X_train = np.random.randn(100, 5)
    y_train = np.random.randn(100)
    X_test = np.random.randn(30, 5)
    y_test = np.random.randn(30)

    # 테스트 실험 1
    print("\n1. 첫 번째 실험")
    exp_id1 = logger.start_experiment(
        experiment_name="RandomForest_Test",
        description="Random Forest 기본 성능 테스트",
        model_type="RandomForestRegressor",
        model_parameters={"n_estimators": 100, "max_depth": 10, "random_state": 42},
        data_info={"n_samples": 100, "n_features": 5, "target_type": "regression"},
        preprocessing_steps=["StandardScaler", "train_test_split"],
        validation_method="TimeSeriesSplit",
        random_seed=42,
        tags=["baseline", "random_forest"],
        notes="기본 성능 확인용"
    )

    # 모델 시뮬레이션 및 지표 기록
    import time
    start_time = time.time()
    time.sleep(0.1)  # 훈련 시뮬레이션
    training_time = time.time() - start_time

    start_time = time.time()
    predictions = np.random.randn(30)  # 예측 시뮬레이션
    prediction_time = time.time() - start_time

    # 지표 계산 및 기록
    mae = np.mean(np.abs(y_test - predictions))
    mse = np.mean((y_test - predictions)**2)
    rmse = np.sqrt(mse)
    r2 = 1 - mse / np.var(y_test) if np.var(y_test) > 0 else 0

    logger.log_metrics(
        mae=mae, mse=mse, rmse=rmse, r2=r2,
        direction_accuracy=65.0, training_time=training_time,
        prediction_time=prediction_time, memory_usage=50.0,
        custom_metrics={"sharpe_ratio": 1.2, "max_drawdown": 0.15}
    )

    logger.log_predictions(predictions)
    logger.log_feature_importance({"feature_0": 0.3, "feature_1": 0.25, "feature_2": 0.2})
    logger.log_validation_scores({"fold_1": 0.85, "fold_2": 0.82, "fold_3": 0.88})

    logger.end_experiment()

    # 테스트 실험 2 (실패 시뮬레이션)
    print("\n2. 두 번째 실험 (실패)")
    exp_id2 = logger.start_experiment(
        experiment_name="LinearRegression_Test",
        description="Linear Regression 실패 케이스",
        model_type="LinearRegression",
        model_parameters={"fit_intercept": True},
        data_info={"n_samples": 100, "n_features": 5},
        preprocessing_steps=["StandardScaler"],
        validation_method="cross_val_score",
        random_seed=42,
        tags=["linear_model", "failed_test"]
    )

    # 오류 시뮬레이션
    try:
        raise ValueError("시뮬레이션 오류: 데이터 형식 불일치")
    except Exception as e:
        logger.log_error(e)

    logger.end_experiment()

    # 실험 목록 및 비교
    print("\n3. 실험 목록 조회")
    all_experiments = logger.list_experiments()
    print(f"전체 실험: {all_experiments}")

    completed_experiments = logger.list_experiments(status="completed")
    print(f"완료된 실험: {completed_experiments}")

    # 실험 비교
    if len(all_experiments) >= 2:
        print("\n4. 실험 비교")
        comparison_df = logger.compare_experiments(all_experiments[:2])
        print(comparison_df.to_string(index=False))

    # 요약 보고서
    print("\n5. 실험 요약 보고서")
    summary = logger.generate_experiment_summary()
    print(summary)

if __name__ == "__main__":
    main()