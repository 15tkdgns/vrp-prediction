#!/usr/bin/env python3
"""
AI Stock Prediction Monitoring System
실시간 모니터링 및 지속적인 예측을 위한 백그라운드 시스템
"""

import time
import json
import threading
from datetime import datetime
from src.testing.run_realtime_test import RealTimePredictor
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/system/monitoring.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MonitoringSystem:
    def __init__(self, interval_minutes=5):
        """
        모니터링 시스템 초기화
        
        Args:
            interval_minutes: 예측 간격 (분)
        """
        self.interval = interval_minutes * 60  # 초로 변환
        self.predictor = RealTimePredictor()
        self.running = False
        self.tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMZN', 'META', 'NFLX']
        
    def initialize(self):
        """시스템 초기화"""
        logger.info("🚀 AI Stock Monitoring System 시작")
        
        # 모델 로드
        if self.predictor.load_best_model():
            logger.info("✅ 모델 로드 성공")
            return True
        else:
            logger.error("❌ 모델 로드 실패")
            return False
    
    def run_prediction_cycle(self):
        """예측 사이클 실행"""
        try:
            logger.info(f"📊 예측 사이클 시작 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 실시간 예측 수행
            results = self.predictor.run_single_test(self.tickers)
            
            if results:
                logger.info(f"✅ {len(results)}개 종목 예측 완료")
                
                # 결과 요약 로깅
                event_count = 0
                for result in results:
                    ticker = result['ticker']
                    price = result['current_price'] 
                    pred = result['predictions']['gradient_boosting']
                    
                    if pred['prediction'] == 1:
                        event_count += 1
                        logger.warning(f"🔴 {ticker}: 이벤트 감지! (확률: {pred['event_probability']*100:.1f}%)")
                    else:
                        logger.info(f"🟢 {ticker}: 정상 (신뢰도: {pred['confidence']*100:.1f}%)")
                
                if event_count > 0:
                    logger.warning(f"⚠️ 총 {event_count}개 종목에서 이벤트 감지됨")
                else:
                    logger.info("✅ 모든 종목 정상 상태")
                    
            else:
                logger.warning("⚠️ 예측 결과 없음")
                
        except Exception as e:
            logger.error(f"❌ 예측 사이클 실패: {e}")
    
    def start_monitoring(self):
        """지속적 모니터링 시작"""
        if not self.initialize():
            return
        
        self.running = True
        logger.info(f"🔄 모니터링 시작 (간격: {self.interval//60}분)")
        logger.info(f"📈 모니터링 종목: {', '.join(self.tickers)}")
        logger.info("⏹️ 중단하려면 Ctrl+C를 누르세요")
        
        try:
            # 첫 번째 예측 즉시 실행
            self.run_prediction_cycle()
            
            # 주기적 실행
            while self.running:
                time.sleep(self.interval)
                if self.running:  # 중간에 중단되지 않았다면
                    self.run_prediction_cycle()
                    
        except KeyboardInterrupt:
            logger.info("🛑 사용자에 의해 모니터링 중단됨")
        except Exception as e:
            logger.error(f"❌ 모니터링 중 오류: {e}")
        finally:
            self.running = False
            logger.info("🏁 모니터링 시스템 종료")
    
    def stop_monitoring(self):
        """모니터링 중단"""
        self.running = False

def main():
    """메인 실행 함수"""
    print("🎯 AI Stock Prediction Monitoring System")
    print("=" * 50)
    
    # 모니터링 간격 설정 (기본: 5분)
    try:
        interval = input("모니터링 간격을 분 단위로 입력하세요 (기본: 5분): ").strip()
        if interval:
            interval = int(interval)
        else:
            interval = 5
    except ValueError:
        interval = 5
        
    print(f"⏰ 모니터링 간격: {interval}분")
    
    # 모니터링 시스템 시작
    monitor = MonitoringSystem(interval_minutes=interval)
    monitor.start_monitoring()

if __name__ == "__main__":
    main()