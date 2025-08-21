#!/usr/bin/env python3
"""
실시간 예측 자동 스케줄러
정기적으로 실시간 예측을 실행하고 결과를 업데이트
"""

import os
import sys
import time
import schedule
import threading
import logging
from datetime import datetime, timezone
from pathlib import Path
import subprocess

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.utils.realtime_predictor import RealtimePredictor

class RealtimeScheduler:
    def __init__(self):
        self.data_dir = project_root / "data" / "raw"
        self.predictor = RealtimePredictor()
        self.is_running = False
        self.scheduler_thread = None
        
        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.data_dir / "scheduler.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)
        
        # 스케줄 설정
        self.setup_schedule()
        
    def setup_schedule(self):
        """스케줄 설정"""
        # 매 5분마다 예측 실행 (시장 시간 중)
        schedule.every(5).minutes.do(self.run_prediction_job)
        
        # 매 시간마다 시스템 상태 확인
        schedule.every().hour.do(self.check_system_health)
        
        # 매일 오전 9시에 로그 정리
        schedule.every().day.at("09:00").do(self.cleanup_logs)
        
        self.logger.info("스케줄 설정 완료")
        
    def is_market_hours(self):
        """시장 시간 확인 (미국 동부 시간 기준 9:30 AM - 4:00 PM)"""
        try:
            # 현재 UTC 시간
            now_utc = datetime.now(timezone.utc)
            
            # 미국 동부 시간으로 변환 (UTC-5 또는 UTC-4, 단순화하여 UTC-5 사용)
            now_et_hour = (now_utc.hour - 5) % 24
            
            # 주말 확인
            weekday = now_utc.weekday()  # 0=월요일, 6=일요일
            if weekday >= 5:  # 토요일(5), 일요일(6)
                return False
                
            # 시장 시간 확인 (9:30 AM - 4:00 PM ET)
            if 9 <= now_et_hour < 16:  # 간단히 9-16시로 설정
                return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"시장 시간 확인 실패: {e}")
            return True  # 에러 시 실행
            
    def run_prediction_job(self):
        """예측 작업 실행"""
        try:
            # 시장 시간이 아니면 건너뛰기
            if not self.is_market_hours():
                self.logger.info("시장 시간이 아님, 예측 건너뛰기")
                return
                
            self.logger.info("스케줄된 예측 작업 시작")
            
            # 예측 실행
            success = self.predictor.run_predictions()
            
            if success:
                self.logger.info("✅ 스케줄된 예측 작업 완료")
                self.update_last_run_status("success")
            else:
                self.logger.error("❌ 스케줄된 예측 작업 실패")
                self.update_last_run_status("failed")
                
        except Exception as e:
            self.logger.error(f"예측 작업 실행 중 오류: {e}")
            self.update_last_run_status("error")
            
    def check_system_health(self):
        """시스템 상태 확인"""
        try:
            self.logger.info("시스템 상태 확인 중...")
            
            # 디스크 사용량 확인
            disk_usage = self.get_disk_usage()
            
            # 메모리 사용량 확인 (간단한 방법)
            memory_info = self.get_memory_info()
            
            # 로그 파일 크기 확인
            log_size = self.get_log_file_size()
            
            health_status = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "disk_usage_percent": disk_usage,
                "memory_info": memory_info,
                "log_file_size_mb": log_size,
                "status": "healthy" if disk_usage < 90 and log_size < 100 else "warning"
            }
            
            # 시스템 상태 저장
            import json
            with open(self.data_dir / "system_health.json", "w") as f:
                json.dump(health_status, f, indent=2)
                
            self.logger.info(f"시스템 상태: {health_status['status']}")
            
        except Exception as e:
            self.logger.error(f"시스템 상태 확인 실패: {e}")
            
    def get_disk_usage(self):
        """디스크 사용량 확인"""
        try:
            import shutil
            total, used, free = shutil.disk_usage(self.data_dir)
            usage_percent = (used / total) * 100
            return round(usage_percent, 2)
        except:
            return 0
            
    def get_memory_info(self):
        """메모리 정보 확인"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return {
                "total_gb": round(memory.total / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
                "percent": memory.percent
            }
        except:
            return {"total_gb": 0, "used_gb": 0, "percent": 0}
            
    def get_log_file_size(self):
        """로그 파일 크기 확인"""
        try:
            log_file = self.data_dir / "realtime_predictor.log"
            if log_file.exists():
                size_mb = log_file.stat().st_size / (1024 * 1024)
                return round(size_mb, 2)
            return 0
        except:
            return 0
            
    def cleanup_logs(self):
        """로그 파일 정리"""
        try:
            self.logger.info("로그 파일 정리 시작")
            
            log_files = [
                "realtime_predictor.log",
                "scheduler.log",
                "realtime_testing.log"
            ]
            
            for log_file in log_files:
                log_path = self.data_dir / log_file
                if log_path.exists():
                    size_mb = log_path.stat().st_size / (1024 * 1024)
                    
                    # 10MB 이상이면 백업하고 새 파일 시작
                    if size_mb > 10:
                        backup_path = self.data_dir / f"{log_file}.backup"
                        log_path.rename(backup_path)
                        self.logger.info(f"로그 파일 백업: {log_file}")
                        
            self.logger.info("로그 파일 정리 완료")
            
        except Exception as e:
            self.logger.error(f"로그 정리 실패: {e}")
            
    def update_last_run_status(self, status):
        """마지막 실행 상태 업데이트"""
        try:
            import json
            status_info = {
                "last_run": datetime.now(timezone.utc).isoformat(),
                "status": status,
                "scheduler_running": self.is_running
            }
            
            with open(self.data_dir / "scheduler_status.json", "w") as f:
                json.dump(status_info, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"상태 업데이트 실패: {e}")
            
    def run_scheduler(self):
        """스케줄러 실행"""
        self.logger.info("스케줄러 시작")
        self.is_running = True
        
        try:
            while self.is_running:
                schedule.run_pending()
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("사용자에 의해 중단됨")
        except Exception as e:
            self.logger.error(f"스케줄러 실행 중 오류: {e}")
        finally:
            self.is_running = False
            self.logger.info("스케줄러 종료")
            
    def start(self):
        """백그라운드에서 스케줄러 시작"""
        if self.is_running:
            self.logger.warning("스케줄러가 이미 실행 중입니다")
            return
            
        self.scheduler_thread = threading.Thread(target=self.run_scheduler, daemon=True)
        self.scheduler_thread.start()
        self.logger.info("백그라운드 스케줄러 시작됨")
        
        # 즉시 한 번 실행
        self.run_prediction_job()
        
    def stop(self):
        """스케줄러 중지"""
        self.is_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        self.logger.info("스케줄러 중지됨")
        
def main():
    """메인 함수"""
    scheduler = RealtimeScheduler()
    
    try:
        # 백그라운드 모드 또는 포그라운드 모드 선택
        import sys
        if "--background" in sys.argv:
            scheduler.start()
            # 백그라운드에서 계속 실행
            while scheduler.is_running:
                time.sleep(60)
        else:
            # 포그라운드에서 실행
            scheduler.run_scheduler()
            
    except KeyboardInterrupt:
        print("\\n스케줄러를 중지합니다...")
        scheduler.stop()
    except Exception as e:
        print(f"스케줄러 실행 실패: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())