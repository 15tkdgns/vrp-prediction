#!/usr/bin/env python3
"""
실시간 다중 심볼 예측 시스템
Dashboard와 호환되는 JSON 형식으로 예측 결과 생성
"""

import os
import sys
import json
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import time
import logging
from pathlib import Path
import joblib
import warnings

warnings.filterwarnings("ignore")

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

class RealtimePredictor:
    def __init__(self):
        self.data_dir = project_root / "data" / "raw"
        self.models_dir = project_root / "data" / "models"
        self.models = {}
        self.scaler = None
        
        # 주식 심볼과 메타데이터
        self.symbols_metadata = {
            "AAPL": {"sector": "technology", "market_cap": "large"},
            "GOOGL": {"sector": "technology", "market_cap": "large"}, 
            "MSFT": {"sector": "technology", "market_cap": "large"},
            "AMZN": {"sector": "consumer_discretionary", "market_cap": "large"},
            "TSLA": {"sector": "consumer_discretionary", "market_cap": "large"},
            "META": {"sector": "technology", "market_cap": "large"},
            "NVDA": {"sector": "technology", "market_cap": "large"},
            "JPM": {"sector": "financials", "market_cap": "large"},
            "JNJ": {"sector": "healthcare", "market_cap": "large"},
            "V": {"sector": "financials", "market_cap": "large"}
        }
        
        # 로깅 설정
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.data_dir / "realtime_predictor.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)
        
    def load_models(self):
        """훈련된 모델들 로드"""
        try:
            # 모델 파일들 확인
            model_files = {
                "random_forest": "random_forest_model.pkl",
                "gradient_boosting": "gradient_boosting_model.pkl", 
                "xgboost": "xgboost_model.pkl"
            }
            
            for name, filename in model_files.items():
                model_path = self.models_dir / filename
                if model_path.exists():
                    self.models[name] = joblib.load(model_path)
                    self.logger.info(f"모델 로드됨: {name}")
                else:
                    self.logger.warning(f"모델 파일 없음: {filename}")
                    
            # 스케일러 로드
            scaler_path = self.models_dir / "scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                self.logger.info("스케일러 로드됨")
            else:
                self.logger.warning("스케일러 파일 없음")
                
            return len(self.models) > 0
            
        except Exception as e:
            self.logger.error(f"모델 로드 실패: {e}")
            return False
            
    def get_stock_data(self, symbol, period="5d"):
        """주식 데이터 가져오기"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval="1h")
            
            if data.empty:
                self.logger.warning(f"{symbol}: 데이터 없음")
                return None
                
            return data
            
        except Exception as e:
            self.logger.error(f"{symbol} 데이터 수집 실패: {e}")
            return None
            
    def calculate_technical_indicators(self, data):
        """기술적 지표 계산"""
        try:
            # 가격 관련 특성
            data['price_change'] = data['Close'].pct_change()
            data['volume_change'] = data['Volume'].pct_change()
            
            # 이동평균
            data['ma_5'] = data['Close'].rolling(window=5).mean()
            data['ma_20'] = data['Close'].rolling(window=20).mean()
            
            # RSI
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))
            
            # 볼린저 밴드
            rolling_mean = data['Close'].rolling(window=20).mean()
            rolling_std = data['Close'].rolling(window=20).std()
            data['bb_upper'] = rolling_mean + (rolling_std * 2)
            data['bb_lower'] = rolling_mean - (rolling_std * 2)
            data['bb_position'] = (data['Close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
            
            # 변동성
            data['volatility'] = data['Close'].rolling(window=10).std()
            
            return data
            
        except Exception as e:
            self.logger.error(f"기술적 지표 계산 실패: {e}")
            return data
            
    def prepare_features(self, data):
        """예측을 위한 특성 준비"""
        try:
            if data is None or len(data) < 20:
                return None
                
            # 기술적 지표 계산
            data = self.calculate_technical_indicators(data)
            
            # 최신 데이터 가져오기
            latest = data.iloc[-1]
            
            # 특성 벡터 생성 (기존 모델과 동일한 순서)
            features = [
                latest.get('price_change', 0),
                latest.get('volume_change', 0),
                latest.get('rsi', 50),
                latest.get('bb_position', 0.5),
                latest.get('volatility', 0),
                (latest.get('Close', 0) - latest.get('ma_5', 0)) / latest.get('ma_5', 1),
                (latest.get('Close', 0) - latest.get('ma_20', 0)) / latest.get('ma_20', 1),
                latest.get('Volume', 0) / data['Volume'].mean() if data['Volume'].mean() > 0 else 1
            ]
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            self.logger.error(f"특성 준비 실패: {e}")
            return None
            
    def make_prediction(self, symbol):
        """단일 심볼에 대한 예측 (임시로 실제 데이터 기반 분석)"""
        try:
            # 주식 데이터 가져오기
            data = self.get_stock_data(symbol)
            if data is None:
                return None
                
            # 현재 가격
            current_price = float(data['Close'].iloc[-1])
            
            # 기술적 분석 기반 예측 (모델 대신 사용)
            data = self.calculate_technical_indicators(data)
            latest = data.iloc[-1]
            
            # 기술적 지표 기반 방향 예측
            signals = []
            
            # RSI 신호
            rsi = latest.get('rsi', 50)
            if rsi < 30:
                signals.append(1)  # 과매도 -> 상승
            elif rsi > 70:
                signals.append(0)  # 과매수 -> 하락
            else:
                signals.append(0.5)  # 중립
                
            # 볼린저 밴드 신호
            bb_pos = latest.get('bb_position', 0.5)
            if bb_pos < 0.2:
                signals.append(1)  # 하단 근처 -> 상승
            elif bb_pos > 0.8:
                signals.append(0)  # 상단 근처 -> 하락
            else:
                signals.append(0.5)  # 중립
                
            # 이동평균 신호
            ma5 = latest.get('ma_5', current_price)
            ma20 = latest.get('ma_20', current_price)
            if current_price > ma5 > ma20:
                signals.append(1)  # 상승 추세
            elif current_price < ma5 < ma20:
                signals.append(0)  # 하락 추세
            else:
                signals.append(0.5)  # 중립
                
            # 가격 변화 신호
            price_change = latest.get('price_change', 0)
            if price_change > 0.02:  # 2% 이상 상승
                signals.append(1)
            elif price_change < -0.02:  # 2% 이상 하락
                signals.append(0)
            else:
                signals.append(0.5)
                
            # 종합 신호
            avg_signal = np.mean(signals)
            direction = "up" if avg_signal > 0.5 else "down"
            confidence = abs(avg_signal - 0.5) * 2  # 0.5에서 멀수록 높은 신뢰도
            
            # 위험도 계산
            volatility = latest.get('volatility', 0) / current_price if current_price > 0 else 0
            if volatility > 0.03:
                risk_level = "high"
            elif volatility > 0.015:
                risk_level = "medium"
            else:
                risk_level = "low"
                
            return {
                "symbol": symbol,
                "current_price": round(current_price, 2),
                "predicted_direction": direction,
                "confidence": round(min(max(confidence, 0.5), 0.95), 3),  # 0.5-0.95 범위
                "risk_level": risk_level,
                "sector": self.symbols_metadata.get(symbol, {}).get("sector", "unknown"),
                "market_cap": self.symbols_metadata.get(symbol, {}).get("market_cap", "unknown"),
                "technical_indicators": {
                    "rsi": round(rsi, 2),
                    "bb_position": round(bb_pos, 3),
                    "price_change": round(price_change * 100, 2),  # 퍼센트
                    "volatility": round(volatility * 100, 2)  # 퍼센트
                }
            }
            
        except Exception as e:
            self.logger.error(f"{symbol} 예측 실패: {e}")
            return None
            
    def calculate_market_summary(self, predictions):
        """시장 요약 계산"""
        try:
            if not predictions:
                return {
                    "overall_sentiment": "neutral",
                    "volatility_index": 0,
                    "trend": "sideways"
                }
                
            # 전체 감정
            up_count = sum(1 for p in predictions if p["predicted_direction"] == "up")
            sentiment_ratio = up_count / len(predictions)
            
            if sentiment_ratio > 0.6:
                overall_sentiment = "bullish"
            elif sentiment_ratio < 0.4:
                overall_sentiment = "bearish" 
            else:
                overall_sentiment = "neutral"
                
            # 변동성 지수 (평균 위험도)
            risk_scores = {"low": 1, "medium": 2, "high": 3}
            avg_risk = np.mean([risk_scores.get(p["risk_level"], 2) for p in predictions])
            volatility_index = round(avg_risk * 5, 1)  # 0-15 스케일
            
            # 트렌드
            avg_confidence = np.mean([p["confidence"] for p in predictions])
            if avg_confidence > 0.7:
                trend = "strong_" + ("upward" if sentiment_ratio > 0.5 else "downward")
            else:
                trend = "sideways"
                
            return {
                "overall_sentiment": overall_sentiment,
                "volatility_index": volatility_index,
                "trend": trend,
                "confidence_level": round(avg_confidence, 3)
            }
            
        except Exception as e:
            self.logger.error(f"시장 요약 계산 실패: {e}")
            return {"overall_sentiment": "neutral", "volatility_index": 0, "trend": "sideways"}
            
    def run_predictions(self):
        """모든 심볼에 대한 예측 실행"""
        try:
            self.logger.info("실시간 예측 시작")
            
            # 모델 로드 (임시로 건너뛰기)
            # if not self.load_models():
            #     self.logger.error("모델 로드 실패")
            #     return False
                
            predictions = []
            symbols = list(self.symbols_metadata.keys())[:5]  # 처음 5개 심볼
            
            for symbol in symbols:
                self.logger.info(f"{symbol} 예측 중...")
                prediction = self.make_prediction(symbol)
                if prediction:
                    predictions.append(prediction)
                    self.logger.info(f"{symbol}: {prediction['predicted_direction']} (신뢰도: {prediction['confidence']})")
                else:
                    self.logger.warning(f"{symbol} 예측 실패")
                    
            # 시장 요약 계산
            market_summary = self.calculate_market_summary(predictions)
            
            # 결과 저장
            result = {
                "predictions": predictions,
                "market_summary": market_summary,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            output_file = self.data_dir / "realtime_results.json"
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)
                
            self.logger.info(f"예측 결과 저장됨: {len(predictions)}개 심볼")
            return True
            
        except Exception as e:
            self.logger.error(f"예측 실행 실패: {e}")
            return False
            
def main():
    """메인 함수"""
    predictor = RealtimePredictor()
    success = predictor.run_predictions()
    
    if success:
        print("✅ 실시간 예측 완료")
    else:
        print("❌ 실시간 예측 실패")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())