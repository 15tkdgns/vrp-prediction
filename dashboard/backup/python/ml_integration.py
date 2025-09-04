#!/usr/bin/env python3
"""
ML 모델 통합 시스템
실시간 예측 및 API 연동을 위한 래퍼 클래스
"""

import os
import sys
import json
import joblib
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
import yfinance as yf

# Add parent directory to path
parent_dir = Path(__file__).parent.parent.absolute()
sys.path.append(str(parent_dir))

# Import YFinance manager (disabled due to dependency issues)
YFINANCE_MANAGER_AVAILABLE = False

logger = logging.getLogger(__name__)

class MLModelIntegration:
    def __init__(self, model_dir="../data/models"):
        self.model_dir = Path(model_dir).resolve()
        self.models = {}
        self.scaler = None
        self.feature_names = ['Open', 'High', 'Low', 'Close', 'Volume', 'unusual_volume', 'price_spike']
        self.load_models()
    
    def load_models(self):
        """훈련된 모델들 로드"""
        try:
            # Random Forest 모델
            rf_path = self.model_dir / "random_forest_model.pkl"
            if rf_path.exists():
                self.models['random_forest'] = joblib.load(rf_path)
                logger.info("✅ Random Forest 모델 로드 성공")
            
            # Gradient Boosting 모델
            gb_path = self.model_dir / "gradient_boosting_model.pkl"
            if gb_path.exists():
                self.models['gradient_boosting'] = joblib.load(gb_path)
                logger.info("✅ Gradient Boosting 모델 로드 성공")
            
            # Scaler 로드
            scaler_path = self.model_dir / "scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info("✅ Scaler 로드 성공")
            
            logger.info(f"🎯 총 {len(self.models)}개 모델 로드 완료")
            
        except Exception as e:
            logger.error(f"❌ 모델 로드 실패: {e}")
            return False
        
        return len(self.models) > 0
    
    def get_stock_data(self, symbol, period="5d"):
        """주식 데이터 가져오기 및 특성 추출 (YFinanceManager 사용)"""
        try:
            # Use direct yfinance (YFinanceManager disabled due to dependency issues)
            logger.debug(f"📊 Fetching {symbol} data using direct yfinance (period={period})")
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                logger.warning(f"⚠️ {symbol}에 대한 데이터가 없습니다")
                return None
            
            # 기본 특성 추출
            latest_data = data.iloc[-1]
            
            # 거래량 이상 탐지 (간단한 버전)
            avg_volume = data['Volume'].mean()
            unusual_volume = 1 if latest_data['Volume'] > avg_volume * 1.5 else 0
            
            # 가격 급등 탐지
            if len(data) >= 2:
                price_change = (latest_data['Close'] - data['Close'].iloc[-2]) / data['Close'].iloc[-2]
                price_spike = 1 if abs(price_change) > 0.05 else 0
            else:
                price_spike = 0
            
            # 특성 벡터 생성
            features = [
                latest_data['Open'],
                latest_data['High'], 
                latest_data['Low'],
                latest_data['Close'],
                latest_data['Volume'],
                unusual_volume,
                price_spike
            ]
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"❌ {symbol} 데이터 처리 실패: {e}")
            return None
    
    def predict_event(self, symbol):
        """특정 종목에 대한 이벤트 예측"""
        if not self.models:
            return {"error": "모델이 로드되지 않음"}
        
        # 데이터 가져오기
        features = self.get_stock_data(symbol)
        if features is None:
            return {"error": f"{symbol} 데이터를 가져올 수 없음"}
        
        # 스케일링
        if self.scaler:
            features_scaled = self.scaler.transform(features)
        else:
            features_scaled = features
        
        predictions = {}
        
        # 각 모델로 예측
        for model_name, model in self.models.items():
            try:
                # 예측 수행
                pred = model.predict(features_scaled)[0]
                
                # 확률 예측 (지원하는 경우)
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(features_scaled)[0]
                    confidence = max(proba)
                else:
                    confidence = 0.7  # 기본값
                
                predictions[model_name] = {
                    'prediction': int(pred),
                    'confidence': float(confidence),
                    'event_type': 'major_event' if pred == 1 else 'normal'
                }
                
            except Exception as e:
                logger.error(f"❌ {model_name} 예측 실패: {e}")
                predictions[model_name] = {
                    'prediction': 0,
                    'confidence': 0.0,
                    'error': str(e)
                }
        
        # 앙상블 예측 (다수결)
        votes = [p['prediction'] for p in predictions.values() if 'error' not in p]
        ensemble_pred = 1 if sum(votes) > len(votes) / 2 else 0
        ensemble_confidence = sum(p['confidence'] for p in predictions.values() if 'error' not in p) / len(votes) if votes else 0
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'individual_predictions': predictions,
            'ensemble_prediction': {
                'prediction': ensemble_pred,
                'confidence': round(ensemble_confidence, 3),
                'event_type': 'major_event' if ensemble_pred == 1 else 'normal'
            },
            'features_used': self.feature_names
        }
    
    def get_live_predictions(self):
        """실시간 예측 데이터 생성 (API 서버용) - Yahoo Finance 실제 가격 사용"""
        if not self.models:
            return None
            
        symbols = ['^GSPC', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']  # S&P 500 지수 추가
        predictions = []
        
        # yfinance 사용 가능성 확인
        yf_available = True
        try:
            import yfinance as yf
        except ImportError:
            yf_available = False
            logger.warning("⚠️ yfinance 없음 - 플레이스홀더 가격 사용")
        
        for symbol in symbols:
            try:
                result = self.predict_event(symbol)
                if 'error' not in result:
                    # 실제 Yahoo Finance 가격 가져오기
                    current_price = 150.0 + hash(symbol) % 100  # 기본값
                    
                    data_date = None
                    if yf_available:
                        try:
                            ticker = yf.Ticker(symbol)
                            hist = ticker.history(period='1d')
                            if not hist.empty:
                                current_price = float(hist['Close'].iloc[-1])
                                data_date = hist.index[-1].strftime('%Y-%m-%d')
                                logger.debug(f"✅ {symbol} 실제 가격: ${current_price:.2f} (날짜: {data_date})")
                            else:
                                logger.warning(f"⚠️ {symbol} 가격 데이터 없음 - 기본값 사용")
                        except Exception as price_error:
                            logger.warning(f"⚠️ {symbol} 가격 가져오기 실패: {price_error} - 기본값 사용")
                    
                    # 섹터 정보 매핑
                    sector_map = {
                        '^GSPC': 'Market Index',
                        'AAPL': 'Technology',
                        'MSFT': 'Technology', 
                        'GOOGL': 'Technology',
                        'AMZN': 'Consumer Discretionary',
                        'TSLA': 'Consumer Discretionary'
                    }
                    
                    prediction = {
                        'symbol': symbol,
                        'current_price': current_price,
                        'predicted_direction': 'up' if result['ensemble_prediction']['prediction'] == 1 else 'down',
                        'confidence': result['ensemble_prediction']['confidence'],
                        'risk_level': 'medium' if result['ensemble_prediction']['confidence'] > 0.7 else 'low',
                        'sector': sector_map.get(symbol, 'Technology'),
                        'last_update': result['timestamp'],
                        'data_date': data_date,
                        'is_real_data': data_date is not None
                    }
                    predictions.append(prediction)
            except Exception as e:
                logger.error(f"❌ {symbol} 예측 실패: {e}")
                continue
        
        # 최신 데이터 날짜 확인
        latest_data_date = None
        if predictions:
            valid_dates = [p['data_date'] for p in predictions if p['data_date']]
            if valid_dates:
                latest_data_date = max(valid_dates)
        
        return {
            'predictions': predictions,
            'timestamp': datetime.now().isoformat(),
            'total_predictions': len(predictions),
            'status': 'success' if predictions else 'no_data',
            'latest_data_date': latest_data_date,
            'data_freshness_warning': latest_data_date is not None and (datetime.now().date() - datetime.strptime(latest_data_date, '%Y-%m-%d').date()).days > 1
        }

    def get_model_status(self):
        """모델 로드 상태 확인"""
        return {
            'models_loaded': list(self.models.keys()),
            'model_count': len(self.models),
            'scaler_loaded': self.scaler is not None,
            'status': 'ready' if self.models else 'no_models'
        }


# 테스트 함수
def test_ml_integration():
    """ML 통합 테스트"""
    print("🧪 ML 모델 통합 테스트 시작")
    
    ml = MLModelIntegration()
    
    # 상태 확인
    status = ml.get_model_status()
    print(f"📊 모델 상태: {status}")
    
    if status['model_count'] > 0:
        # AAPL 예측 테스트
        result = ml.predict_event('AAPL')
        print(f"🎯 AAPL 예측 결과:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result
    else:
        print("❌ 로드된 모델이 없습니다")
        return None

if __name__ == '__main__':
    test_ml_integration()