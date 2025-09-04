"""
AI Stock Prediction System - Configuration Manager
환경변수 로딩 문제 해결을 위한 안전한 설정 관리자
"""

import os
import logging
from typing import Dict, Optional, Any
from pathlib import Path
from dotenv import load_dotenv
import json

class ConfigManager:
    """
    안전한 환경변수 및 설정 관리
    """
    
    def __init__(self, env_path: Optional[str] = None):
        """
        설정 관리자 초기화
        
        Args:
            env_path: .env 파일 경로 (기본: 프로젝트 루트)
        """
        self.logger = logging.getLogger(__name__)
        
        # .env 파일 경로 설정
        if env_path is None:
            project_root = Path(__file__).parent.parent.parent
            env_path = project_root / '.env'
        
        # 환경변수 로드
        if os.path.exists(env_path):
            load_dotenv(env_path)
            self.logger.info(f"✅ .env 파일 로드됨: {env_path}")
        else:
            self.logger.warning(f"⚠️ .env 파일을 찾을 수 없습니다: {env_path}")
            self.logger.info("💡 .env.example을 .env로 복사하고 API 키를 설정하세요")
        
        # API 키 로드 및 검증
        self.api_keys = self._load_api_keys()
        self._validate_api_keys()
        
        # 시스템 설정 로드
        self.system_config = self._load_system_config()
        
    def _load_api_keys(self) -> Dict[str, Optional[str]]:
        """API 키들을 환경변수에서 로드"""
        keys = {
            'ALPHA_VANTAGE': os.getenv('ALPHA_VANTAGE_KEY'),
            'FMP': os.getenv('FMP_KEY'),
            'TWELVE_DATA': os.getenv('TWELVE_DATA_KEY'),
            'POLYGON': os.getenv('POLYGON_KEY'),
            'IEX_CLOUD': os.getenv('IEX_CLOUD_KEY'),
            'MARKETAUX': os.getenv('MARKETAUX_KEY'),
            'NEWS_API': os.getenv('NEWS_API_KEY'),
            'FINNHUB': os.getenv('FINNHUB_KEY'),
        }
        return keys
    
    def _load_system_config(self) -> Dict[str, Any]:
        """시스템 설정 로드"""
        config = {
            'environment': os.getenv('ENVIRONMENT', 'development'),
            'log_level': os.getenv('LOG_LEVEL', 'INFO'),
            'dashboard_port': int(os.getenv('DASHBOARD_PORT', 8090)),
            'api_rate_limit': int(os.getenv('API_RATE_LIMIT', 60)),
            'prediction_interval': int(os.getenv('PREDICTION_INTERVAL', 300)),
            'confidence_threshold': float(os.getenv('CONFIDENCE_THRESHOLD', 0.75)),
            'batch_size': int(os.getenv('BATCH_SIZE', 32)),
        }
        return config
    
    def _validate_api_keys(self):
        """API 키 유효성 검사"""
        valid_keys = 0
        total_keys = len(self.api_keys)
        
        for service_name, api_key in self.api_keys.items():
            if self._is_valid_api_key(api_key):
                self.logger.info(f"✅ {service_name} API 키 로드됨")
                valid_keys += 1
            else:
                self.logger.warning(f"⚠️ {service_name} API 키가 설정되지 않았거나 유효하지 않음")
        
        if valid_keys == 0:
            self.logger.error("❌ 설정된 API 키가 없습니다. 시스템이 제한적으로 작동합니다.")
            self.logger.info("💡 .env 파일에 실제 API 키를 설정하세요")
        elif valid_keys < total_keys:
            self.logger.warning(f"⚠️ {valid_keys}/{total_keys} API 키만 설정됨")
        else:
            self.logger.info(f"🎉 모든 API 키({total_keys}개) 정상 로드됨")
            
    def _is_valid_api_key(self, api_key: Optional[str]) -> bool:
        """API 키 유효성 체크"""
        if not api_key:
            return False
        
        # 플레이스홀더 키 체크
        invalid_patterns = [
            'your_',
            'change_this',
            'example',
            'demo',
            'test',
            'placeholder'
        ]
        
        api_key_lower = api_key.lower()
        if any(pattern in api_key_lower for pattern in invalid_patterns):
            return False
            
        # 최소 길이 체크 (대부분 API 키는 10자 이상)
        if len(api_key) < 10:
            return False
            
        return True
    
    def get_api_key(self, service: str) -> Optional[str]:
        """
        서비스별 API 키 조회
        
        Args:
            service: 서비스 이름 (예: 'ALPHA_VANTAGE', 'POLYGON')
            
        Returns:
            API 키 문자열 또는 None
        """
        key = self.api_keys.get(service.upper())
        if not self._is_valid_api_key(key):
            self.logger.warning(f"⚠️ {service} API 키를 사용할 수 없습니다")
            return None
        return key
    
    def get_system_config(self, key: str, default: Any = None) -> Any:
        """
        시스템 설정 조회
        
        Args:
            key: 설정 키
            default: 기본값
            
        Returns:
            설정 값
        """
        return self.system_config.get(key, default)
    
    def is_development(self) -> bool:
        """개발 환경 여부 확인"""
        return self.get_system_config('environment', 'development') == 'development'
    
    def is_production(self) -> bool:
        """프로덕션 환경 여부 확인"""
        return self.get_system_config('environment', 'development') == 'production'
    
    def get_available_services(self) -> list:
        """사용 가능한 API 서비스 목록 반환"""
        available = []
        for service, key in self.api_keys.items():
            if self._is_valid_api_key(key):
                available.append(service)
        return available
    
    def get_config_summary(self) -> Dict[str, Any]:
        """설정 요약 정보 반환 (보안 정보 제외)"""
        available_services = self.get_available_services()
        
        summary = {
            'environment': self.get_system_config('environment'),
            'api_services_count': len(available_services),
            'available_services': available_services,
            'system_config': {
                'dashboard_port': self.get_system_config('dashboard_port'),
                'prediction_interval': self.get_system_config('prediction_interval'),
                'confidence_threshold': self.get_system_config('confidence_threshold'),
            }
        }
        return summary
    
    def validate_setup(self) -> Dict[str, Any]:
        """
        전체 설정 검증
        
        Returns:
            검증 결과 딕셔너리
        """
        results = {
            'status': 'ok',
            'warnings': [],
            'errors': [],
            'api_keys': {
                'total': len(self.api_keys),
                'valid': len(self.get_available_services()),
                'missing': []
            }
        }
        
        # API 키 검증
        for service, key in self.api_keys.items():
            if not self._is_valid_api_key(key):
                results['api_keys']['missing'].append(service)
        
        # 경고 및 오류 생성
        if results['api_keys']['valid'] == 0:
            results['status'] = 'error'
            results['errors'].append('API 키가 설정되지 않았습니다')
        elif results['api_keys']['valid'] < results['api_keys']['total']:
            results['warnings'].append(f"{len(results['api_keys']['missing'])}개 API 키가 설정되지 않았습니다")
        
        return results

# 전역 인스턴스 (싱글톤 패턴)
_config_manager = None

def get_config_manager() -> ConfigManager:
    """전역 설정 관리자 인스턴스 반환"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

# 편의 함수들
def get_api_key(service: str) -> Optional[str]:
    """API 키 조회 편의 함수"""
    return get_config_manager().get_api_key(service)

def get_system_config(key: str, default: Any = None) -> Any:
    """시스템 설정 조회 편의 함수"""
    return get_config_manager().get_system_config(key, default)

def is_development() -> bool:
    """개발 환경 확인 편의 함수"""
    return get_config_manager().is_development()

if __name__ == "__main__":
    # 테스트 코드
    config = ConfigManager()
    
    print("=== API 키 상태 ===")
    for service in config.api_keys:
        key = config.get_api_key(service)
        status = "✅ 설정됨" if key else "❌ 미설정"
        print(f"{service}: {status}")
    
    print("\n=== 시스템 설정 ===")
    summary = config.get_config_summary()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    
    print("\n=== 설정 검증 ===")
    validation = config.validate_setup()
    print(json.dumps(validation, indent=2, ensure_ascii=False))