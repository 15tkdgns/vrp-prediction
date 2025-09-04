"""
AI Stock Prediction System - Directory Manager
디렉토리 자동 생성 및 관리 유틸리티
"""

import os
import logging
from pathlib import Path
from typing import List, Dict


class DirectoryManager:
    """
    프로젝트 디렉토리 자동 생성 및 관리
    """
    
    def __init__(self, project_root: str = None):
        """
        디렉토리 관리자 초기화
        
        Args:
            project_root: 프로젝트 루트 디렉토리 (기본: 현재 파일의 3단계 상위)
        """
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent
        
        self.project_root = Path(project_root)
        self.logger = logging.getLogger(__name__)
        
        # 필수 디렉토리 목록 정의
        self.required_directories = [
            # 데이터 디렉토리
            'data/raw',
            'data/processed', 
            'data/models',
            'data/cache',
            'data/backup',
            
            # 결과 디렉토리
            'results/analysis',
            'results/training',
            'results/realtime',
            'results/reports',
            'results/visualizations',
            
            # 로그 디렉토리  
            'logs/system',
            'logs/api',
            'logs/models',
            'logs/dashboard',
            
            # 임시 디렉토리
            'tmp/downloads',
            'tmp/processing',
            'tmp/uploads',
            
            # 설정 디렉토리
            'config/environments',
            'config/models', 
            'config/api',
            
            # 문서 디렉토리
            'docs/api',
            'docs/user_guide',
            'docs/development',
            'docs/reports',
            
            # 대시보드 관련
            'dashboard/uploads',
            'dashboard/logs',
            'dashboard/cache',
            
            # 테스트 관련
            'tests/data',
            'tests/fixtures',
            'tests/outputs',
        ]
        
    def ensure_directories(self, additional_dirs: List[str] = None) -> Dict[str, bool]:
        """
        모든 필수 디렉토리가 존재하는지 확인하고 없으면 생성
        
        Args:
            additional_dirs: 추가로 생성할 디렉토리 목록
            
        Returns:
            생성 결과 딕셔너리 {디렉토리: 성공여부}
        """
        directories = self.required_directories.copy()
        
        if additional_dirs:
            directories.extend(additional_dirs)
            
        results = {}
        created_count = 0
        existed_count = 0
        failed_count = 0
        
        self.logger.info("📁 필수 디렉토리 확인 중...")
        
        for directory in directories:
            dir_path = self.project_root / directory
            
            try:
                if dir_path.exists():
                    results[directory] = True
                    existed_count += 1
                    self.logger.debug(f"✅ 존재: {directory}")
                else:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    results[directory] = True
                    created_count += 1
                    self.logger.info(f"📁 생성: {directory}")
                    
            except Exception as e:
                results[directory] = False
                failed_count += 1
                self.logger.error(f"❌ 생성 실패: {directory} - {e}")
        
        # 결과 요약
        total = len(directories)
        success_rate = (created_count + existed_count) / total * 100
        
        self.logger.info(f"📊 디렉토리 생성 완료:")
        self.logger.info(f"  • 이미 존재: {existed_count}개")
        self.logger.info(f"  • 새로 생성: {created_count}개") 
        self.logger.info(f"  • 실패: {failed_count}개")
        self.logger.info(f"  • 성공률: {success_rate:.1f}%")
        
        if failed_count > 0:
            self.logger.warning("⚠️ 일부 디렉토리 생성에 실패했습니다")
        else:
            self.logger.info("🎉 모든 필수 디렉토리 생성 완료!")
            
        return results
    
    def clean_directories(self, directories: List[str] = None, dry_run: bool = True) -> Dict[str, int]:
        """
        지정된 디렉토리의 임시 파일들을 정리
        
        Args:
            directories: 정리할 디렉토리 목록 (기본: tmp, logs의 하위 디렉토리)
            dry_run: True면 실제로 삭제하지 않고 로그만 출력
            
        Returns:
            정리 결과 {디렉토리: 삭제된_파일_수}
        """
        if directories is None:
            directories = ['tmp', 'logs/system', 'logs/api', 'logs/models']
            
        results = {}
        
        for directory in directories:
            dir_path = self.project_root / directory
            
            if not dir_path.exists():
                continue
                
            file_count = 0
            
            try:
                # 임시 파일 패턴들
                patterns = ['*.tmp', '*.temp', '*.log.*', '*.backup', '*.cache']
                
                for pattern in patterns:
                    for file_path in dir_path.glob(pattern):
                        if file_path.is_file():
                            if dry_run:
                                self.logger.info(f"🗑️ [DRY RUN] 삭제 예정: {file_path}")
                            else:
                                file_path.unlink()
                                self.logger.info(f"🗑️ 삭제됨: {file_path}")
                            file_count += 1
                            
                results[directory] = file_count
                
            except Exception as e:
                self.logger.error(f"❌ {directory} 정리 실패: {e}")
                results[directory] = -1
                
        if not dry_run:
            self.logger.info(f"🧹 총 {sum(r for r in results.values() if r > 0)}개 파일 정리 완료")
        else:
            self.logger.info("ℹ️ dry_run=False로 설정하여 실제 정리를 실행하세요")
            
        return results
    
    def get_directory_sizes(self, directories: List[str] = None) -> Dict[str, int]:
        """
        디렉토리별 크기를 MB 단위로 반환
        
        Args:
            directories: 확인할 디렉토리 목록 (기본: 주요 디렉토리들)
            
        Returns:
            {디렉토리: 크기_MB}
        """
        if directories is None:
            directories = ['data', 'results', 'logs', 'tmp', 'models']
            
        results = {}
        
        for directory in directories:
            dir_path = self.project_root / directory
            
            if not dir_path.exists():
                results[directory] = 0
                continue
                
            try:
                total_size = 0
                for file_path in dir_path.rglob('*'):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
                        
                results[directory] = round(total_size / (1024 * 1024), 2)  # MB
                
            except Exception as e:
                self.logger.error(f"❌ {directory} 크기 계산 실패: {e}")
                results[directory] = -1
                
        return results
    
    def validate_permissions(self) -> Dict[str, bool]:
        """
        주요 디렉토리의 읽기/쓰기 권한 확인
        
        Returns:
            {디렉토리: 권한_OK}
        """
        critical_directories = [
            'data/raw', 'data/processed', 'data/models',
            'results/analysis', 'logs/system'
        ]
        
        results = {}
        
        for directory in critical_directories:
            dir_path = self.project_root / directory
            
            if not dir_path.exists():
                results[directory] = False
                continue
                
            try:
                # 읽기 권한 테스트
                list(dir_path.iterdir())
                
                # 쓰기 권한 테스트
                test_file = dir_path / '.permission_test'
                test_file.write_text('test')
                test_file.unlink()
                
                results[directory] = True
                self.logger.debug(f"✅ 권한 OK: {directory}")
                
            except Exception as e:
                results[directory] = False
                self.logger.warning(f"⚠️ 권한 문제: {directory} - {e}")
                
        return results
    
    def get_status_report(self) -> Dict:
        """
        디렉토리 상태 종합 보고서 생성
        
        Returns:
            상태 보고서 딕셔너리
        """
        self.logger.info("📋 디렉토리 상태 보고서 생성 중...")
        
        return {
            'directory_creation': self.ensure_directories(),
            'directory_sizes': self.get_directory_sizes(), 
            'permissions': self.validate_permissions(),
            'project_root': str(self.project_root),
            'timestamp': os.path.getctime(str(self.project_root))
        }


# 편의 함수들
def ensure_all_directories(project_root: str = None, additional_dirs: List[str] = None):
    """모든 필수 디렉토리 생성 편의 함수"""
    manager = DirectoryManager(project_root)
    return manager.ensure_directories(additional_dirs)

def get_directory_manager(project_root: str = None) -> DirectoryManager:
    """디렉토리 관리자 인스턴스 반환"""
    return DirectoryManager(project_root)


if __name__ == "__main__":
    # 테스트 실행
    logging.basicConfig(level=logging.INFO)
    
    manager = DirectoryManager()
    
    print("=== 디렉토리 생성 테스트 ===")
    results = manager.ensure_directories()
    
    print("\n=== 디렉토리 크기 확인 ===")
    sizes = manager.get_directory_sizes()
    for dir_name, size in sizes.items():
        print(f"{dir_name}: {size} MB")
    
    print("\n=== 권한 확인 ===") 
    permissions = manager.validate_permissions()
    for dir_name, has_permission in permissions.items():
        status = "✅" if has_permission else "❌"
        print(f"{status} {dir_name}")
    
    print("\n=== 상태 보고서 ===")
    import json
    report = manager.get_status_report()
    print(json.dumps(report, indent=2, default=str, ensure_ascii=False))