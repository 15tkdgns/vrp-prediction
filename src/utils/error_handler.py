"""
통합 에러 핸들러 - 시스템 전체의 에러 처리 표준화
"""

import logging
import traceback
import sys
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum


class ErrorSeverity(Enum):
    """에러 심각도 수준"""
    LOW = "LOW"
    MEDIUM = "MEDIUM" 
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ErrorCategory(Enum):
    """에러 카테고리"""
    API_ERROR = "API_ERROR"
    DATA_ERROR = "DATA_ERROR"
    MODEL_ERROR = "MODEL_ERROR"
    NETWORK_ERROR = "NETWORK_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    SYSTEM_ERROR = "SYSTEM_ERROR"


class StandardizedError(Exception):
    """표준화된 에러 클래스"""
    
    def __init__(self, 
                 message: str,
                 category: ErrorCategory,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 error_code: Optional[str] = None,
                 context: Optional[Dict[str, Any]] = None,
                 original_error: Optional[Exception] = None):
        self.message = message
        self.category = category
        self.severity = severity
        self.error_code = error_code or f"{category.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.context = context or {}
        self.original_error = original_error
        self.timestamp = datetime.now()
        
        super().__init__(self.message)


class ErrorHandler:
    """통합 에러 핸들러"""
    
    def __init__(self, logger_name: str = "AI_Stock_System"):
        self.logger = logging.getLogger(logger_name)
        self.error_stats = {
            "total_errors": 0,
            "by_category": {},
            "by_severity": {},
            "last_error_time": None
        }
    
    def handle_error(self, 
                    error: Exception,
                    category: ErrorCategory,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    context: Optional[Dict[str, Any]] = None) -> StandardizedError:
        """표준화된 에러 처리"""
        
        # 통계 업데이트
        self._update_error_stats(category, severity)
        
        # 표준화된 에러 객체 생성
        std_error = StandardizedError(
            message=str(error),
            category=category,
            severity=severity,
            context=context,
            original_error=error
        )
        
        # 로깅
        self._log_error(std_error)
        
        # 심각도에 따른 추가 처리
        if severity == ErrorSeverity.CRITICAL:
            self._handle_critical_error(std_error)
        elif severity == ErrorSeverity.HIGH:
            self._handle_high_severity_error(std_error)
        
        return std_error
    
    def _update_error_stats(self, category: ErrorCategory, severity: ErrorSeverity):
        """에러 통계 업데이트"""
        self.error_stats["total_errors"] += 1
        self.error_stats["last_error_time"] = datetime.now()
        
        # 카테고리별 통계
        cat_key = category.value
        if cat_key not in self.error_stats["by_category"]:
            self.error_stats["by_category"][cat_key] = 0
        self.error_stats["by_category"][cat_key] += 1
        
        # 심각도별 통계  
        sev_key = severity.value
        if sev_key not in self.error_stats["by_severity"]:
            self.error_stats["by_severity"][sev_key] = 0
        self.error_stats["by_severity"][sev_key] += 1
    
    def _log_error(self, error: StandardizedError):
        """에러 로깅"""
        log_message = f"""
=== STANDARDIZED ERROR ===
Error Code: {error.error_code}
Category: {error.category.value}
Severity: {error.severity.value}
Message: {error.message}
Timestamp: {error.timestamp}
Context: {error.context}
"""
        
        if error.original_error:
            log_message += f"Original Error: {error.original_error}\n"
            log_message += f"Traceback: {traceback.format_exc()}\n"
        
        log_message += "=========================="
        
        # 심각도에 따른 로그 레벨
        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif error.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def _handle_critical_error(self, error: StandardizedError):
        """Critical 에러 특별 처리"""
        # TODO: 시스템 관리자에게 즉시 알림
        # TODO: 시스템 자동 복구 시도
        self.logger.critical(f"🚨 CRITICAL ERROR DETECTED: {error.error_code}")
        
    def _handle_high_severity_error(self, error: StandardizedError):
        """High 심각도 에러 처리"""
        # TODO: 모니터링 시스템에 알림
        self.logger.error(f"🔴 HIGH SEVERITY ERROR: {error.error_code}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """에러 요약 통계"""
        return {
            "total_errors": self.error_stats["total_errors"],
            "categories": self.error_stats["by_category"],
            "severities": self.error_stats["by_severity"], 
            "last_error": self.error_stats["last_error_time"],
            "system_health": self._calculate_system_health()
        }
    
    def _calculate_system_health(self) -> str:
        """시스템 건강도 계산"""
        total = self.error_stats["total_errors"]
        if total == 0:
            return "EXCELLENT"
        
        critical = self.error_stats["by_severity"].get("CRITICAL", 0)
        high = self.error_stats["by_severity"].get("HIGH", 0)
        
        if critical > 0:
            return "CRITICAL"
        elif high > 5:
            return "POOR"
        elif total > 20:
            return "FAIR"
        else:
            return "GOOD"


# 전역 에러 핸들러 인스턴스
global_error_handler = ErrorHandler()


def handle_api_error(error: Exception, context: Dict[str, Any] = None) -> StandardizedError:
    """API 에러 전용 핸들러"""
    return global_error_handler.handle_error(
        error, ErrorCategory.API_ERROR, ErrorSeverity.HIGH, context
    )


def handle_data_error(error: Exception, context: Dict[str, Any] = None) -> StandardizedError:
    """데이터 에러 전용 핸들러"""
    return global_error_handler.handle_error(
        error, ErrorCategory.DATA_ERROR, ErrorSeverity.MEDIUM, context
    )


def handle_model_error(error: Exception, context: Dict[str, Any] = None) -> StandardizedError:
    """모델 에러 전용 핸들러"""
    return global_error_handler.handle_error(
        error, ErrorCategory.MODEL_ERROR, ErrorSeverity.HIGH, context
    )


def handle_network_error(error: Exception, context: Dict[str, Any] = None) -> StandardizedError:
    """네트워크 에러 전용 핸들러"""
    return global_error_handler.handle_error(
        error, ErrorCategory.NETWORK_ERROR, ErrorSeverity.MEDIUM, context
    )


def handle_validation_error(error: Exception, context: Dict[str, Any] = None) -> StandardizedError:
    """검증 에러 전용 핸들러"""
    return global_error_handler.handle_error(
        error, ErrorCategory.VALIDATION_ERROR, ErrorSeverity.LOW, context
    )


def handle_system_error(error: Exception, context: Dict[str, Any] = None) -> StandardizedError:
    """시스템 에러 전용 핸들러"""
    return global_error_handler.handle_error(
        error, ErrorCategory.SYSTEM_ERROR, ErrorSeverity.CRITICAL, context
    )


def get_system_health() -> Dict[str, Any]:
    """시스템 건강도 조회"""
    return global_error_handler.get_error_summary()