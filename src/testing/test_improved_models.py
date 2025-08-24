#!/usr/bin/env python3
"""
개선된 모델 테스트 스크립트
현실적인 신뢰도로 실시간 예측 테스트
"""

import os
import json
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from datetime import datetime, timedelta
import logging
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')


class ImprovedModelTester:
    def __init__(self, data_dir="data/raw", models_dir="data/models"):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.models = {}
        self.scaler = None
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)

    def load_improved_models(self):
        """개선된 모델들 로드"""
        print("🔄 개선된 모델 로딩...")
        
        try:
            # 스케일러 로드
            scaler_path = f"{self.models_dir}/scaler_improved.pkl"
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                print("✅ 스케일러 로드 완료")
            else:
                print("⚠️ 개선된 스케일러를 찾을 수 없습니다. 기존 스케일러 사용")
                self.scaler = joblib.load(f"{self.models_dir}/scaler.pkl")
            
            # Random Forest 모델
            rf_path = f"{self.models_dir}/random_forest_improved_model.pkl"
            if os.path.exists(rf_path):
                self.models['random_forest'] = joblib.load(rf_path)
                print("✅ 개선된 Random Forest 모델 로드 완료")
            
            # Gradient Boosting 모델
            gb_path = f"{self.models_dir}/gradient_boosting_improved_model.pkl"
            if os.path.exists(gb_path):
                self.models['gradient_boosting'] = joblib.load(gb_path)
                print("✅ 개선된 Gradient Boosting 모델 로드 완료")
            
            # LSTM 모델
            lstm_path = f"{self.models_dir}/lstm_improved_model.h5"
            if os.path.exists(lstm_path):
                self.models['lstm'] = load_model(lstm_path)
                print("✅ 개선된 LSTM 모델 로드 완료")
            
            if not self.models:
                raise FileNotFoundError("개선된 모델을 찾을 수 없습니다.")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 모델 로드 실패: {e}")
            return False

    def get_realtime_data(self, tickers=['AAPL', 'MSFT', 'GOOGL'], days=5):
        """실시간 테스트를 위한 최근 데이터 수집"""
        print(f"📊 최근 {days}일 데이터 수집 중...")
        
        # 주말/휴일 대응을 위해 더 긴 기간 설정
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days+60)  # 더 많은 데이터 확보
        
        all_data = []
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                data = stock.history(start=start_date, end=end_date)
                
                if data.empty:
                    continue
                
                # 기술적 지표 계산 (훈련 데이터와 동일)
                data['Returns'] = data['Close'].pct_change()
                data['Volatility'] = data['Returns'].rolling(window=5).std()
                data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
                data['Price_MA_5'] = data['Close'].rolling(window=5).mean()
                data['Price_MA_20'] = data['Close'].rolling(window=20).mean()
                data['Price_MA_50'] = data['Close'].rolling(window=50).mean()
                
                # 추가 기술적 지표
                data['RSI'] = self.calculate_rsi(data['Close'])
                data['MACD'], data['MACD_Signal'] = self.calculate_macd(data['Close'])
                data['BB_Upper'], data['BB_Lower'] = self.calculate_bollinger_bands(data['Close'])
                data['ATR'] = self.calculate_atr(data)
                
                data['Price_Change'] = data['Returns'].abs()
                data['Volume_Spike'] = data['Volume'] / data['Volume_MA']
                
                data['ticker'] = ticker
                data = data.reset_index()
                data = data.dropna()
                
                # 최근 데이터만 선택 (최소 1개 보장)
                recent_data = data.tail(max(days, 10))  # 최소 10개 데이터 확보
                all_data.append(recent_data)
                
                print(f"✅ {ticker}: {len(recent_data)}개 최신 레코드")
                
            except Exception as e:
                print(f"❌ {ticker} 데이터 수집 실패: {e}")
                continue
        
        if not all_data:
            return None
        
        return pd.concat(all_data, ignore_index=True)

    def calculate_rsi(self, prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line

    def calculate_bollinger_bands(self, prices, window=20, num_std=2):
        rolling_mean = prices.rolling(window=window).mean()
        rolling_std = prices.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, lower_band

    def calculate_atr(self, data, window=14):
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(window=window).mean()
        return atr

    def prepare_test_features(self, data):
        """테스트 데이터 특성 준비"""
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'Returns', 'Volatility', 'Volume_MA', 'Price_MA_5', 'Price_MA_20', 'Price_MA_50',
            'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'ATR',
            'Price_Change', 'Volume_Spike'
        ]
        
        X = data[feature_columns].fillna(0)
        return X

    def run_realtime_test(self):
        """실시간 테스트 실행"""
        print("\n🎯 개선된 모델 실시간 테스트 시작")
        print("=" * 50)
        
        if not self.load_improved_models():
            return False
        
        # 실시간 데이터 수집
        test_data = self.get_realtime_data()
        if test_data is None:
            print("❌ 테스트 데이터를 수집할 수 없습니다.")
            return False
        
        # 특성 준비
        X_test = self.prepare_test_features(test_data)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"\n📊 테스트 데이터: {len(test_data)}개 샘플")
        print(f"📈 종목: {', '.join(test_data['ticker'].unique())}")
        print(f"📅 기간: {test_data['Date'].min().strftime('%Y-%m-%d')} ~ {test_data['Date'].max().strftime('%Y-%m-%d')}")
        
        # 모델별 예측 결과
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\n🤖 {model_name.upper()} 예측 결과:")
            print("-" * 30)
            
            if model_name == 'lstm':
                # LSTM용 데이터 reshape
                X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
                predictions = model.predict(X_test_lstm, verbose=0).flatten()
            else:
                predictions = model.predict_proba(X_test_scaled)[:, 1]
            
            # 예측 결과 분석
            avg_confidence = np.mean(predictions)
            confidence_std = np.std(predictions)
            max_confidence = np.max(predictions)
            min_confidence = np.min(predictions)
            high_confidence_count = np.sum(predictions > 0.5)
            
            results[model_name] = {
                'predictions': predictions.tolist(),
                'avg_confidence': float(avg_confidence),
                'confidence_std': float(confidence_std),
                'max_confidence': float(max_confidence),
                'min_confidence': float(min_confidence),
                'high_confidence_count': int(high_confidence_count),
                'total_samples': len(predictions)
            }
            
            print(f"  평균 신뢰도: {avg_confidence:.4f} ± {confidence_std:.4f}")
            print(f"  신뢰도 범위: {min_confidence:.4f} ~ {max_confidence:.4f}")
            print(f"  고신뢰도 예측 (>0.5): {high_confidence_count}개 ({high_confidence_count/len(predictions)*100:.1f}%)")
            
            # 상위 예측 결과 표시
            top_indices = np.argsort(predictions)[-3:][::-1]
            print(f"  상위 3개 예측:")
            for i, idx in enumerate(top_indices):
                row = test_data.iloc[idx]
                print(f"    {i+1}. {row['ticker']} ({row['Date'].strftime('%m-%d')}): {predictions[idx]:.4f}")
        
        # 결과 저장
        test_results = {
            'test_timestamp': datetime.now().isoformat(),
            'test_data_info': {
                'samples': len(test_data),
                'tickers': test_data['ticker'].unique().tolist(),
                'date_range': {
                    'start': test_data['Date'].min().isoformat(),
                    'end': test_data['Date'].max().isoformat()
                }
            },
            'model_results': results
        }
        
        # 결과 파일 저장
        with open(f"{self.data_dir}/improved_realtime_test_results.json", "w") as f:
            json.dump(test_results, f, indent=2)
        
        print(f"\n💾 결과 저장됨: {self.data_dir}/improved_realtime_test_results.json")
        
        # 요약 리포트
        print(f"\n📋 테스트 요약:")
        print(f"  테스트 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  총 샘플: {len(test_data)}개")
        print(f"  테스트된 모델: {len(results)}개")
        
        best_model = min(results.keys(), key=lambda x: abs(results[x]['avg_confidence'] - 0.15))  # 0.15에 가장 가까운 모델
        print(f"  가장 현실적인 모델: {best_model} (평균 신뢰도: {results[best_model]['avg_confidence']:.4f})")
        
        return True


if __name__ == "__main__":
    tester = ImprovedModelTester()
    success = tester.run_realtime_test()
    
    if success:
        print("\n✅ 개선된 모델 테스트 완료!")
        print("   이제 현실적인 신뢰도로 예측이 가능합니다.")
    else:
        print("\n❌ 테스트 실패!")