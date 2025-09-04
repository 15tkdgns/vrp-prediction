#!/usr/bin/env python3
"""
실제 주식 API 연동 (고도화된 기술적 지표 포함)
yfinance와 기술적 지표를 사용하여 실제 주식 데이터를 가져옵니다.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Optional
from technical_indicators import technical_analyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealStockAPI:
    def __init__(self):
        self.symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
        self.cache = {}
        self.cache_timeout = 60  # 1분 캐시
        
    def get_historical_data(self, symbol: str, start_date: str, end_date: str) -> Optional[List[float]]:
        """실제 히스토리 데이터 가져오기 (yfinance)"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)
            
            if hist.empty:
                logger.error(f"❌ {symbol} 히스토리 데이터 없음: {start_date}~{end_date}")
                return None
                
            prices = [float(price) for price in hist['Close'].values]
            logger.info(f"✅ {symbol} 히스토리 데이터 {len(prices)}개 로드: {start_date}~{end_date}")
            return prices
            
        except Exception as e:
            logger.error(f"❌ {symbol} 히스토리 데이터 로드 실패: {e}")
            return None

    def get_real_stock_data(self, symbol: str) -> Optional[Dict]:
        """실제 주식 데이터 가져오기 (고도화된 기술적 지표 포함)"""
        try:
            # 캐시 확인
            cache_key = f"{symbol}_data"
            now = datetime.now()
            
            if cache_key in self.cache:
                cached_data, cached_time = self.cache[cache_key]
                if (now - cached_time).seconds < self.cache_timeout:
                    logger.info(f"📋 {symbol} 캐시 데이터 사용")
                    return cached_data
            
            logger.info(f"🔄 {symbol} 실제 데이터 + 기술적 분석 로딩...")
            
            # 기술적 분석 데이터 가져오기
            technical_data = technical_analyzer.get_comprehensive_analysis(symbol)
            
            if not technical_data:
                logger.error(f"❌ {symbol} 기술적 분석 실패")
                return None
            
            # 추가적인 회사 정보 가져오기
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # 기술적 분석 결과와 회사 정보를 결합
            stock_data = {
                "symbol": symbol,
                "current_price": technical_data['current_price'],
                "predicted_direction": technical_data['predicted_direction'],
                "confidence": technical_data['confidence'],
                "risk_level": technical_data['risk_level'],
                "sector": self._get_sector(symbol),
                "market_cap": info.get('marketCap', 'N/A'),
                "technical_indicators": {
                    # 기존 지표들
                    "price_change": technical_data['technical_indicators']['price_change'],
                    "volatility": technical_data['technical_indicators']['volatility'],
                    "volume": technical_data['technical_indicators']['volume'],
                    "avg_volume_20d": technical_data['technical_indicators']['avg_volume_20d'],
                    "day_high": float(info.get('dayHigh', technical_data['current_price'])),
                    "day_low": float(info.get('dayLow', technical_data['current_price'])),
                    "fifty_two_week_high": float(info.get('fiftyTwoWeekHigh', technical_data['current_price'])),
                    "fifty_two_week_low": float(info.get('fiftyTwoWeekLow', technical_data['current_price'])),
                    
                    # 고급 기술적 지표들
                    "rsi": technical_data['technical_indicators']['rsi'],
                    "bollinger_upper": technical_data['technical_indicators']['bollinger_upper'],
                    "bollinger_lower": technical_data['technical_indicators']['bollinger_lower'],
                    "bollinger_position": technical_data['technical_indicators']['bollinger_position'],
                    "macd": technical_data['technical_indicators']['macd'],
                    "macd_signal": technical_data['technical_indicators']['macd_signal'],
                    "macd_histogram": technical_data['technical_indicators']['macd_histogram'],
                    "sharpe_ratio": technical_data['technical_indicators']['sharpe_ratio'],
                    "momentum": technical_data['technical_indicators']['momentum'],
                    "support_level": technical_data['technical_indicators']['support_level'],
                    "resistance_level": technical_data['technical_indicators']['resistance_level']
                },
                "company_info": {
                    "name": info.get('longName', info.get('shortName', symbol)),
                    "industry": info.get('industry', 'Unknown'),
                    "employees": info.get('fullTimeEmployees', 'N/A'),
                    "market_cap_formatted": self._format_market_cap(info.get('marketCap', 0)),
                    "pe_ratio": info.get('forwardPE', info.get('trailingPE', 'N/A')),
                    "dividend_yield": info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
                }
            }
            
            # 캐시에 저장
            self.cache[cache_key] = (stock_data, now)
            
            logger.info(f"✅ {symbol} 실제 데이터 로드 완료: ${technical_data['current_price']}")
            return stock_data
            
        except Exception as e:
            logger.error(f"❌ {symbol} 데이터 로드 실패: {e}")
            return None
    
    def _get_sector(self, symbol: str) -> str:
        """확장된 섹터 매핑"""
        sector_map = {
            'AAPL': 'Technology',
            'GOOGL': 'Technology', 
            'MSFT': 'Technology',
            'AMZN': 'Consumer Discretionary',
            'TSLA': 'Consumer Discretionary',
            'META': 'Technology',
            'NVDA': 'Technology',
            'NFLX': 'Communication Services',
            'V': 'Financial Services',
            'MA': 'Financial Services',
            'JPM': 'Financial Services',
            'JNJ': 'Healthcare',
            'PG': 'Consumer Staples'
        }
        return sector_map.get(symbol, 'Unknown')
    
    def _format_market_cap(self, market_cap: int) -> str:
        """시가총액 포맷팅"""
        if market_cap == 0 or market_cap == 'N/A':
            return 'N/A'
        
        try:
            market_cap = int(market_cap)
            if market_cap >= 1_000_000_000_000:  # 1조 이상
                return f"${market_cap / 1_000_000_000_000:.2f}T"
            elif market_cap >= 1_000_000_000:  # 10억 이상
                return f"${market_cap / 1_000_000_000:.2f}B"
            elif market_cap >= 1_000_000:  # 100만 이상
                return f"${market_cap / 1_000_000:.2f}M"
            else:
                return f"${market_cap:,}"
        except:
            return 'N/A'
    
    def get_multiple_stocks(self, symbols: List[str] = None) -> List[Dict]:
        """여러 주식의 실제 데이터 가져오기"""
        if symbols is None:
            symbols = self.symbols
            
        results = []
        for symbol in symbols:
            data = self.get_real_stock_data(symbol)
            if data:
                results.append(data)
        
        return results
    
    def get_live_stocks(self) -> Dict:
        """실시간 주식 데이터를 API 서버용 형식으로 반환"""
        try:
            stocks_data = self.get_multiple_stocks()
            if not stocks_data:
                return None
                
            predictions = []
            for stock in stocks_data:
                prediction = {
                    'symbol': stock['symbol'],
                    'current_price': stock['current_price'],
                    'predicted_direction': stock['predicted_direction'],
                    'confidence': stock['confidence'],
                    'risk_level': stock['risk_level'],
                    'sector': stock['sector'],
                    'last_update': datetime.now().isoformat()
                }
                predictions.append(prediction)
            
            return {
                'predictions': predictions,
                'timestamp': datetime.now().isoformat(),
                'total_predictions': len(predictions),
                'status': 'success' if predictions else 'no_data'
            }
        except Exception as e:
            logger.error(f"❌ get_live_stocks 실패: {e}")
            return None

    def get_market_summary(self) -> Dict:
        """시장 요약 정보"""
        try:
            # S&P 500 지수 정보
            sp500 = yf.Ticker('^GSPC')
            sp500_hist = sp500.history(period='2d', interval='1d')
            
            if len(sp500_hist) >= 2:
                current = float(sp500_hist['Close'].iloc[-1])
                previous = float(sp500_hist['Close'].iloc[-2])
                change = (current - previous) / previous
                
                return {
                    "sp500_current": round(current, 2),
                    "sp500_change": round(change, 4),
                    "overall_sentiment": "positive" if change > 0 else "negative",
                    "trend": "upward" if change > 0.01 else ("downward" if change < -0.01 else "sideways"),
                    "volatility_index": round(abs(change) * 100, 1),
                    "confidence_level": round(0.6 + abs(change) * 10, 2)
                }
        except Exception as e:
            logger.error(f"❌ 시장 요약 정보 로드 실패: {e}")
        
        # 기본값 반환
        return {
            "sp500_current": 0,
            "sp500_change": 0,
            "overall_sentiment": "neutral",
            "trend": "sideways", 
            "volatility_index": 0,
            "confidence_level": 0.5,
            "market_status": "closed",
            "session_change": 0.0
        }

# 전역 인스턴스
real_api = RealStockAPI()

if __name__ == "__main__":
    # 테스트
    print("🧪 실제 주식 API 테스트...")
    
    # 단일 주식 테스트
    aapl_data = real_api.get_real_stock_data('AAPL')
    if aapl_data:
        print(f"✅ AAPL 현재가: ${aapl_data['current_price']}")
        print(f"   변화율: {aapl_data['technical_indicators']['price_change']*100:.2f}%")
    
    # 여러 주식 테스트
    stocks = real_api.get_multiple_stocks(['AAPL', 'GOOGL'])
    print(f"✅ {len(stocks)}개 주식 데이터 로드 완료")
    
    # 시장 요약 테스트
    market = real_api.get_market_summary()
    print(f"✅ S&P 500: {market['sp500_current']} ({market['sp500_change']*100:+.2f}%)")