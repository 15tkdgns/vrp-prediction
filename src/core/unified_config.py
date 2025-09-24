#!/usr/bin/env python3
"""
통합 설정 관리 시스템
YAML 기반의 계층적 설정 관리 및 환경별 오버라이드 지원
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """설정 관련 예외"""
    pass


@dataclass
class DataConfig:
    """데이터 관련 설정"""
    symbol: str = "SPY"
    period: str = "5y"
    interval: str = "1d"
    directories: Dict[str, str] = field(default_factory=dict)
    validation: Dict[str, Any] = field(default_factory=dict)
    features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """모델 관련 설정"""
    supported_types: list = field(default_factory=list)
    default_config: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    ensemble: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """훈련 관련 설정"""
    validation: Dict[str, Any] = field(default_factory=dict)
    gpu: Dict[str, Any] = field(default_factory=dict)
    optimization: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationConfig:
    """평가 관련 설정"""
    backtesting: Dict[str, Any] = field(default_factory=dict)
    statistical_tests: Dict[str, Any] = field(default_factory=dict)
    performance_thresholds: Dict[str, float] = field(default_factory=dict)
    monitoring: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemConfig:
    """시스템 관련 설정"""
    warnings: Dict[str, Any] = field(default_factory=dict)
    reproducibility: Dict[str, Any] = field(default_factory=dict)
    performance: Dict[str, Any] = field(default_factory=dict)
    error_handling: Dict[str, Any] = field(default_factory=dict)


class UnifiedConfigManager:
    """통합 설정 관리자"""

    def __init__(self, environment: str = None, config_dir: str = None):
        """
        설정 관리자 초기화

        Args:
            environment: 환경 (development, production, test)
            config_dir: 설정 파일 디렉토리 경로
        """
        self.environment = environment or os.getenv('ENVIRONMENT', 'development')
        self.config_dir = Path(config_dir or "config")

        self._raw_config = {}
        self.data = DataConfig()
        self.models = ModelConfig()
        self.training = TrainingConfig()
        self.evaluation = EvaluationConfig()
        self.system = SystemConfig()

        self._load_configuration()

    def _load_configuration(self) -> None:
        """설정 파일들을 로드하고 병합"""
        try:
            # 1. 기본 설정 로드
            default_path = self.config_dir / "default.yaml"
            if default_path.exists():
                with open(default_path, 'r', encoding='utf-8') as f:
                    self._raw_config = yaml.safe_load(f) or {}
            else:
                logger.warning(f"기본 설정 파일을 찾을 수 없습니다: {default_path}")
                self._raw_config = {}

            # 2. 환경별 설정 오버라이드
            env_path = self.config_dir / f"{self.environment}.yaml"
            if env_path.exists():
                with open(env_path, 'r', encoding='utf-8') as f:
                    env_config = yaml.safe_load(f) or {}
                self._raw_config = self._deep_merge(self._raw_config, env_config)
            else:
                logger.warning(f"환경별 설정 파일을 찾을 수 없습니다: {env_path}")

            # 3. 환경 변수로 오버라이드
            self._apply_environment_variables()

            # 4. 타입별 설정 객체 생성
            self._create_config_objects()

            logger.info(f"설정 로드 완료: 환경={self.environment}")

        except Exception as e:
            raise ConfigurationError(f"설정 로드 실패: {e}")

    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """딕셔너리 깊은 병합"""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def _apply_environment_variables(self) -> None:
        """환경 변수로 설정 오버라이드"""
        # ${VAR_NAME} 형태의 환경 변수 참조 해결
        self._raw_config = self._resolve_env_vars(self._raw_config)

    def _resolve_env_vars(self, obj: Any) -> Any:
        """환경 변수 참조 해결"""
        if isinstance(obj, dict):
            return {k: self._resolve_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._resolve_env_vars(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
            env_var = obj[2:-1]
            return os.getenv(env_var, obj)  # 환경 변수가 없으면 원본 반환
        else:
            return obj

    def _create_config_objects(self) -> None:
        """설정 딕셔너리를 타입별 객체로 변환"""
        try:
            # 데이터 설정
            data_config = self._raw_config.get('data', {})
            self.data = DataConfig(
                symbol=data_config.get('symbol', 'SPY'),
                period=data_config.get('period', '5y'),
                interval=data_config.get('interval', '1d'),
                directories=data_config.get('directories', {}),
                validation=data_config.get('validation', {}),
                features=data_config.get('features', {})
            )

            # 모델 설정
            model_config = self._raw_config.get('models', {})
            self.models = ModelConfig(
                supported_types=model_config.get('supported_types', []),
                default_config=model_config.get('default_config', {}),
                ensemble=model_config.get('ensemble', {})
            )

            # 훈련 설정
            training_config = self._raw_config.get('training', {})
            self.training = TrainingConfig(
                validation=training_config.get('validation', {}),
                gpu=training_config.get('gpu', {}),
                optimization=training_config.get('optimization', {}),
                metrics=training_config.get('metrics', {})
            )

            # 평가 설정
            eval_config = self._raw_config.get('evaluation', {})
            self.evaluation = EvaluationConfig(
                backtesting=eval_config.get('backtesting', {}),
                statistical_tests=eval_config.get('statistical_tests', {}),
                performance_thresholds=eval_config.get('performance_thresholds', {}),
                monitoring=eval_config.get('monitoring', {})
            )

            # 시스템 설정
            system_config = self._raw_config.get('system', {})
            self.system = SystemConfig(
                warnings=system_config.get('warnings', {}),
                reproducibility=system_config.get('reproducibility', {}),
                performance=system_config.get('performance', {}),
                error_handling=system_config.get('error_handling', {})
            )

        except Exception as e:
            raise ConfigurationError(f"설정 객체 생성 실패: {e}")

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        점 표기법으로 설정값 가져오기

        Args:
            key_path: 점으로 구분된 키 경로 (예: "models.default_config.xgboost.n_estimators")
            default: 기본값

        Returns:
            설정값
        """
        keys = key_path.split('.')
        value = self._raw_config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key_path: str, value: Any) -> None:
        """
        점 표기법으로 설정값 수정

        Args:
            key_path: 점으로 구분된 키 경로
            value: 설정할 값
        """
        keys = key_path.split('.')
        config = self._raw_config

        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value

        # 설정 객체 다시 생성
        self._create_config_objects()

    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """특정 모델의 설정 반환"""
        return self.models.default_config.get(model_type, {})

    def get_data_directory(self, dir_type: str) -> str:
        """데이터 디렉토리 경로 반환"""
        return self.data.directories.get(dir_type, f"data/{dir_type}")

    def is_gpu_enabled(self) -> bool:
        """GPU 사용 여부 반환"""
        return self.training.gpu.get('enable', False)

    def get_performance_threshold(self, metric: str) -> float:
        """성능 임계값 반환"""
        return self.evaluation.performance_thresholds.get(metric, 0.0)

    def validate_configuration(self) -> bool:
        """설정 검증"""
        errors = []

        # 필수 디렉토리 존재 확인
        for dir_type, dir_path in self.data.directories.items():
            path = Path(dir_path)
            if not path.exists():
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"디렉토리 생성: {path}")
                except Exception as e:
                    errors.append(f"디렉토리 생성 실패 ({dir_type}): {e}")

        # 모델 타입 검증
        if not self.models.supported_types:
            errors.append("지원 모델 타입이 정의되지 않았습니다.")

        # GPU 설정 검증
        if self.training.gpu.get('enable', False):
            try:
                import torch
                if not torch.cuda.is_available():
                    logger.warning("GPU가 활성화되었지만 CUDA를 사용할 수 없습니다.")
            except ImportError:
                logger.warning("GPU가 활성화되었지만 PyTorch가 설치되지 않았습니다.")

        if errors:
            raise ConfigurationError(f"설정 검증 실패: {', '.join(errors)}")

        return True

    def save_current_config(self, file_path: str = None) -> None:
        """현재 설정을 파일로 저장"""
        if file_path is None:
            file_path = f"config/current_{self.environment}.yaml"

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self._raw_config, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"현재 설정 저장: {path}")

    def get_summary(self) -> Dict[str, Any]:
        """설정 요약 정보 반환"""
        return {
            'environment': self.environment,
            'data_symbol': self.data.symbol,
            'data_period': self.data.period,
            'supported_models': self.models.supported_types,
            'gpu_enabled': self.is_gpu_enabled(),
            'validation_method': self.training.validation.get('method', 'time_series_split'),
            'primary_metric': self.training.metrics.get('primary', 'f1_score'),
            'config_source': str(self.config_dir)
        }


# 전역 설정 인스턴스
CONFIG = None


def get_config(environment: str = None, config_dir: str = None) -> UnifiedConfigManager:
    """전역 설정 인스턴스 반환 (싱글톤 패턴)"""
    global CONFIG

    if CONFIG is None:
        CONFIG = UnifiedConfigManager(environment, config_dir)
        CONFIG.validate_configuration()

    return CONFIG


def reload_config(environment: str = None, config_dir: str = None) -> UnifiedConfigManager:
    """설정 강제 재로드"""
    global CONFIG
    CONFIG = None
    return get_config(environment, config_dir)


if __name__ == "__main__":
    # 사용 예시
    config = get_config('development')

    print("✅ 통합 설정 관리 시스템 초기화 완료")
    print(f"환경: {config.environment}")
    print(f"데이터 심볼: {config.data.symbol}")
    print(f"GPU 사용: {config.is_gpu_enabled()}")
    print(f"지원 모델: {config.models.supported_types}")

    # 설정 요약 출력
    summary = config.get_summary()
    print("\n설정 요약:")
    for key, value in summary.items():
        print(f"  {key}: {value}")