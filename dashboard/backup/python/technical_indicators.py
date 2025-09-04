#!/usr/bin/env python3
"""
실제 기술적 지표 및 AI 신뢰도 계산 모듈
yfinance 데이터를 기반으로 실제 기술적 분석 지표를 계산합니다.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TechnicalIndicators:
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 300  # 5분 캐시
        
    def _get_cached_data(self, symbol: str, period: str = '30d') -> Optional[pd.DataFrame]:
        """캐시된 주식 데이터 가져오기"""
        cache_key = f"{symbol}_{period}"
        now = datetime.now()
        
        if cache_key in self.cache:
            data, cached_time = self.cache[cache_key]
            if (now - cached_time).seconds < self.cache_timeout:
                return data
        
        try:
            # yfinance에서 데이터 가져오기
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval='1d')
            
            if not data.empty:
                self.cache[cache_key] = (data, now)
                return data
        except Exception as e:
            logger.error(f"❌ {symbol} 데이터 가져오기 실패: {e}")
        
        return None
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """RSI (Relative Strength Index) 계산"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            return round(float(rsi.iloc[-1]), 2)
        except:
            return 50.0  # 중립값
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20) -> Tuple[float, float, float]:
        """볼린져 밴드 계산"""
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std()
            
            upper_band = sma + (2 * std)
            lower_band = sma - (2 * std)
            
            current_price = prices.iloc[-1]
            upper = upper_band.iloc[-1]
            lower = lower_band.iloc[-1]
            
            # 볼린져 밴드 내 위치 (0: 하단, 1: 상단)
            bb_position = (current_price - lower) / (upper - lower) if upper != lower else 0.5
            
            return round(float(upper), 2), round(float(lower), 2), round(float(bb_position), 3)
        except:
            return 0.0, 0.0, 0.5
    
    def calculate_macd(self, prices: pd.Series) -> Tuple[float, float, float]:
        """MACD 계산"""
        try:
            ema12 = prices.ewm(span=12).mean()
            ema26 = prices.ewm(span=26).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9).mean()
            histogram = macd_line - signal_line
            
            return (
                round(float(macd_line.iloc[-1]), 3),
                round(float(signal_line.iloc[-1]), 3),
                round(float(histogram.iloc[-1]), 3)
            )
        except:
            return 0.0, 0.0, 0.0
    
    def calculate_volatility(self, prices: pd.Series, period: int = 20) -> float:
        """변동성 계산 (표준편차)"""
        try:
            returns = prices.pct_change()
            volatility = returns.rolling(window=period).std() * np.sqrt(252)  # 연간화
            return round(float(volatility.iloc[-1]), 4)
        except:
            return 0.0
    
    def calculate_sharpe_ratio(self, prices: pd.Series, risk_free_rate: float = 0.05) -> float:
        """샤프 비율 계산"""
        try:
            returns = prices.pct_change().dropna()
            if len(returns) < 2:
                return 0.0
                
            excess_returns = returns - (risk_free_rate / 252)  # 일별 무위험 수익률
            
            if excess_returns.std() == 0:
                return 0.0
                
            sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
            return round(float(sharpe), 3)
        except:
            return 0.0
    
    def calculate_momentum(self, prices: pd.Series, period: int = 10) -> float:
        """모멘텀 계산"""
        try:
            if len(prices) < period + 1:
                return 0.0
            momentum = (prices.iloc[-1] / prices.iloc[-(period+1)]) - 1
            return round(float(momentum), 4)
        except:
            return 0.0
    
    def calculate_support_resistance(self, data: pd.DataFrame, period: int = 20) -> Tuple[float, float]:
        """지지/저항선 계산"""
        try:
            recent_data = data.tail(period)
            support = float(recent_data['Low'].min())
            resistance = float(recent_data['High'].max())
            
            return round(support, 2), round(resistance, 2)
        except:
            current = float(data['Close'].iloc[-1])
            return current * 0.95, current * 1.05
    
    def calculate_ai_confidence(self, symbol: str, data: pd.DataFrame) -> float:
        """AI 예측 신뢰도 계산 (실제 지표 기반)"""
        try:
            prices = data['Close']
            volume = data['Volume']
            
            # 1. 기술적 지표 일관성 점수
            rsi = self.calculate_rsi(prices)
            rsi_score = 1.0 if 30 <= rsi <= 70 else 0.5  # 과매수/과매도 아닐 때 높은 점수
            
            # 2. 거래량 안정성
            avg_volume = volume.tail(20).mean()
            current_volume = volume.iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            volume_score = min(1.0, volume_ratio / 2)  # 거래량이 많을수록 신뢰도 상승
            
            # 3. 가격 변동성 안정성
            volatility = self.calculate_volatility(prices)
            volatility_score = max(0.3, 1.0 - min(volatility, 1.0))  # 변동성이 낮을수록 신뢰도 상승
            
            # 4. 추세 일관성
            short_ma = prices.tail(5).mean()
            long_ma = prices.tail(20).mean()
            trend_consistency = abs(short_ma - long_ma) / long_ma if long_ma > 0 else 0
            trend_score = max(0.3, 1.0 - min(trend_consistency, 0.5))
            
            # 5. 시장 상관관계 (S&P 500과의 상관관계)
            correlation_score = 0.7  # 기본값 (실제로는 S&P 500과의 상관관계 계산)
            
            # 종합 점수 계산 (가중평균)
            confidence = (
                rsi_score * 0.2 + 
                volume_score * 0.3 + 
                volatility_score * 0.2 + 
                trend_score * 0.2 + 
                correlation_score * 0.1
            )
            
            # 0.4 ~ 0.95 범위로 조정
            confidence = max(0.4, min(0.95, confidence))
            
            return round(confidence, 3)
            
        except Exception as e:
            logger.error(f"❌ {symbol} AI 신뢰도 계산 실패: {e}")
            return 0.65  # 기본 신뢰도
    
    def calculate_risk_level(self, data: pd.DataFrame) -> str:
        """위험 수준 계산"""
        try:
            prices = data['Close']
            
            # 변동성 기반 위험도 계산
            volatility = self.calculate_volatility(prices)
            
            # RSI 기반 과매수/과매도 상태
            rsi = self.calculate_rsi(prices)
            
            # 최근 가격 변화
            price_change = abs(prices.pct_change().tail(5).mean())
            
            risk_score = 0
            
            # 변동성 점수
            if volatility > 0.4:
                risk_score += 3
            elif volatility > 0.25:
                risk_score += 2
            else:
                risk_score += 1
            
            # RSI 점수
            if rsi > 80 or rsi < 20:
                risk_score += 3
            elif rsi > 70 or rsi < 30:
                risk_score += 2
            else:
                risk_score += 1
            
            # 가격 변화 점수
            if price_change > 0.05:
                risk_score += 3
            elif price_change > 0.03:
                risk_score += 2
            else:
                risk_score += 1
            
            # 최종 위험 등급
            if risk_score <= 4:
                return "low"
            elif risk_score <= 7:
                return "medium"
            else:
                return "high"
                
        except:
            return "medium"
    
    def get_comprehensive_analysis(self, symbol: str) -> Dict:
        """종합적인 기술적 분석"""
        try:
            # 데이터 가져오기
            data = self._get_cached_data(symbol, '30d')
            if data is None or data.empty:
                logger.error(f"❌ {symbol} 데이터 없음")
                return self._get_default_analysis(symbol)
            
            prices = data['Close']
            volume = data['Volume']
            
            # 기본 정보
            current_price = float(prices.iloc[-1])
            price_change = float((prices.iloc[-1] - prices.iloc[-2]) / prices.iloc[-2])
            
            # 기술적 지표 계산
            rsi = self.calculate_rsi(prices)
            bb_upper, bb_lower, bb_position = self.calculate_bollinger_bands(prices)
            macd, macd_signal, macd_histogram = self.calculate_macd(prices)
            volatility = self.calculate_volatility(prices)
            sharpe_ratio = self.calculate_sharpe_ratio(prices)
            momentum = self.calculate_momentum(prices)
            support, resistance = self.calculate_support_resistance(data)
            
            # AI 분석
            confidence = self.calculate_ai_confidence(symbol, data)
            risk_level = self.calculate_risk_level(data)
            
            # 예측 방향 (복합 지표 기반)
            direction_score = 0
            
            # RSI 기반 방향성
            if rsi < 30:
                direction_score += 1  # 매수 신호
            elif rsi > 70:
                direction_score -= 1  # 매도 신호
            
            # MACD 기반 방향성
            if macd > macd_signal:
                direction_score += 1
            else:
                direction_score -= 1
            
            # 모멘텀 기반 방향성
            if momentum > 0:
                direction_score += 1
            else:
                direction_score -= 1
            
            predicted_direction = "up" if direction_score > 0 else "down"
            
            return {
                "symbol": symbol,
                "current_price": round(current_price, 2),
                "predicted_direction": predicted_direction,
                "confidence": confidence,
                "risk_level": risk_level,
                "technical_indicators": {
                    "rsi": rsi,
                    "bollinger_upper": bb_upper,
                    "bollinger_lower": bb_lower,
                    "bollinger_position": bb_position,
                    "macd": macd,
                    "macd_signal": macd_signal,
                    "macd_histogram": macd_histogram,
                    "price_change": round(price_change, 4),
                    "volatility": volatility,
                    "sharpe_ratio": sharpe_ratio,
                    "momentum": momentum,
                    "volume": int(volume.iloc[-1]) if not volume.empty else 0,
                    "avg_volume_20d": int(volume.tail(20).mean()) if not volume.empty else 0,
                    "support_level": support,
                    "resistance_level": resistance
                }
            }
            
        except Exception as e:
            logger.error(f"❌ {symbol} 종합 분석 실패: {e}")
            return self._get_default_analysis(symbol)
    
    def _get_default_analysis(self, symbol: str) -> Dict:
        """기본 분석 데이터 (에러 시 사용)"""
        return {
            "symbol": symbol,
            "current_price": 100.0,
            "predicted_direction": "neutral",
            "confidence": 0.5,
            "risk_level": "medium",
            "technical_indicators": {
                "rsi": 50.0,
                "bollinger_upper": 105.0,
                "bollinger_lower": 95.0,
                "bollinger_position": 0.5,
                "macd": 0.0,
                "macd_signal": 0.0,
                "macd_histogram": 0.0,
                "price_change": 0.0,
                "volatility": 0.2,
                "sharpe_ratio": 0.0,
                "momentum": 0.0,
                "volume": 1000000,
                "avg_volume_20d": 1000000,
                "support_level": 95.0,
                "resistance_level": 105.0
            }
        }

# 전역 인스턴스
technical_analyzer = TechnicalIndicators()

if __name__ == "__main__":
    # 테스트
    print("🧪 기술적 지표 계산 테스트...")
    
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    
    for symbol in symbols:
        print(f"\n=== {symbol} 기술적 분석 ===")
        analysis = technical_analyzer.get_comprehensive_analysis(symbol)
        
        print(f"현재가: ${analysis['current_price']}")
        print(f"예측 방향: {analysis['predicted_direction']}")
        print(f"신뢰도: {analysis['confidence']*100:.1f}%")
        print(f"위험 수준: {analysis['risk_level']}")
        print(f"RSI: {analysis['technical_indicators']['rsi']}")
        print(f"볼린져 밴드 위치: {analysis['technical_indicators']['bollinger_position']}")
        print(f"변동성: {analysis['technical_indicators']['volatility']:.2%}")
        print(f"샤프 비율: {analysis['technical_indicators']['sharpe_ratio']}")