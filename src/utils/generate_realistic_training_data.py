import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os

def generate_realistic_training_data():
    """실제 주식 데이터를 사용한 현실적인 훈련 데이터 생성"""
    print("🔄 실제 주식 데이터 기반 훈련 데이터 생성 시작...")
    
    # S&P500 주요 종목
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
    
    # 지난 6개월 데이터 수집
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    all_features = []
    all_labels = []
    
    print(f"📅 데이터 수집 기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
    
    for ticker in tickers:
        print(f"📈 {ticker} 데이터 수집 중...")
        
        try:
            # 주식 데이터 다운로드
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date)
            
            if data.empty:
                continue
            
            # 기술적 지표 계산
            data['Returns'] = data['Close'].pct_change()
            data['Volatility'] = data['Returns'].rolling(window=5).std()
            data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
            data['Price_MA_5'] = data['Close'].rolling(window=5).mean()
            data['Price_MA_20'] = data['Close'].rolling(window=20).mean()
            data['Price_MA_50'] = data['Close'].rolling(window=50).mean()
            
            # 추가 기술적 지표
            data['RSI'] = calculate_rsi(data['Close'])
            data['MACD'], data['MACD_Signal'] = calculate_macd(data['Close'])
            data['BB_Upper'], data['BB_Lower'] = calculate_bollinger_bands(data['Close'])
            data['ATR'] = calculate_atr(data)
            
            # 이벤트 정의 (현실적)
            data['Price_Change'] = data['Returns'].abs()
            data['Volume_Spike'] = data['Volume'] / data['Volume_MA']
            
            # major_event: 3% 이상 가격 변동 OR 거래량 2배 이상 증가
            data['major_event'] = (
                (data['Price_Change'] > 0.03) | 
                (data['Volume_Spike'] > 2.0)
            ).astype(int)
            
            # price_spike: 2% 이상 가격 변동
            data['price_spike'] = (data['Price_Change'] > 0.02).astype(int)
            
            # unusual_volume: 거래량 1.5배 이상 증가
            data['unusual_volume'] = (data['Volume_Spike'] > 1.5).astype(int)
            
            # ticker 컬럼 추가
            data['ticker'] = ticker
            data = data.reset_index()
            
            # NaN 제거
            data = data.dropna()
            
            if len(data) < 10:  # 최소 데이터 수 확인
                continue
            
            # 특성 데이터프레임 생성
            feature_columns = [
                'Date', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Returns', 'Volatility', 'Volume_MA', 'Price_MA_5', 'Price_MA_20', 'Price_MA_50',
                'RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'ATR',
                'Price_Change', 'Volume_Spike'
            ]
            
            features_df = data[['ticker'] + feature_columns].copy()
            
            # 라벨 데이터프레임 생성
            labels_df = data[['ticker', 'Date', 'major_event', 'price_spike', 'unusual_volume']].copy()
            
            all_features.append(features_df)
            all_labels.append(labels_df)
            
            print(f"✅ {ticker}: {len(data)}개 레코드, 이벤트 비율: {data['major_event'].mean():.3f}")
            
        except Exception as e:
            print(f"❌ {ticker} 데이터 수집 실패: {e}")
            continue
    
    if not all_features:
        print("❌ 수집된 데이터가 없습니다.")
        return False
    
    # 데이터 결합
    final_features = pd.concat(all_features, ignore_index=True)
    final_labels = pd.concat(all_labels, ignore_index=True)
    
    # 디렉토리 생성
    os.makedirs('data/raw', exist_ok=True)
    
    # 파일 저장
    final_features.to_csv('data/raw/training_features.csv', index=False)
    final_labels.to_csv('data/raw/event_labels.csv', index=False)
    
    print(f"\n✅ 훈련 데이터 생성 완료!")
    print(f"📊 총 샘플 수: {len(final_features)}")
    print(f"📈 종목 수: {len(tickers)}")
    print(f"🎯 전체 이벤트 비율: {final_labels['major_event'].mean():.3f}")
    print(f"💾 특성 파일: data/raw/training_features.csv")
    print(f"💾 라벨 파일: data/raw/event_labels.csv")
    
    return True

def calculate_rsi(prices, window=14):
    """RSI (Relative Strength Index) 계산"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """MACD 계산"""
    exp1 = prices.ewm(span=fast).mean()
    exp2 = prices.ewm(span=slow).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal).mean()
    return macd, signal_line

def calculate_bollinger_bands(prices, window=20, num_std=2):
    """볼린저 밴드 계산"""
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

def calculate_atr(data, window=14):
    """Average True Range 계산"""
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(window=window).mean()
    return atr

if __name__ == "__main__":
    success = generate_realistic_training_data()
    if success:
        print("\n🚀 이제 개선된 모델 훈련을 실행할 수 있습니다!")
    else:
        print("\n❌ 데이터 생성에 실패했습니다.")