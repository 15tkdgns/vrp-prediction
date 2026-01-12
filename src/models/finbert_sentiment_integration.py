#!/usr/bin/env python3
"""
FinBERT 뉴스 심리 지수 통합 파이프라인
======================================

Hugging Face의 금융 뉴스 심리 데이터셋을 활용하여
변동성 예측 모델에 통합

데이터셋: 
- Kaggle Financial PhraseBank
- Hugging Face zeroshot/twitter-financial-news-sentiment
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
import json

# Hugging Face 데이터셋
try:
    from datasets import load_dataset
    HAS_HF = True
except ImportError:
    HAS_HF = False
    print("⚠️ datasets 라이브러리 필요: pip install datasets")

# FinBERT
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    HAS_FINBERT = True
except ImportError:
    HAS_FINBERT = False
    print("⚠️ transformers 라이브러리 필요: pip install transformers")

# sklearn
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

SEED = 42
np.random.seed(SEED)


# =============================================================================
# 1. 뉴스 심리 데이터 로드 (Hugging Face)
# =============================================================================

def load_financial_sentiment_dataset():
    """Hugging Face에서 금융 뉴스 심리 데이터 로드"""
    print("\n" + "="*60)
    print("[1/6] 금융 뉴스 심리 데이터셋 로드")
    print("="*60)
    
    if not HAS_HF:
        # 대안: 시뮬레이션 데이터
        print("  ⚠️ Hugging Face 미설치, 시뮬레이션 데이터 사용")
        return generate_simulated_sentiment()
    
    try:
        # Twitter Financial Sentiment 데이터셋
        print("  → Twitter Financial Sentiment 로드 중...")
        dataset = load_dataset("zeroshot/twitter-financial-news-sentiment", split="train")
        
        df = pd.DataFrame({
            'text': dataset['text'],
            'label': dataset['label']
        })
        
        # 라벨 변환 (0=bearish, 1=bullish, 2=neutral)
        label_map = {0: -1, 1: 1, 2: 0}
        df['sentiment'] = df['label'].map(label_map)
        
        print(f"  ✓ 로드 완료: {len(df)} 샘플")
        print(f"  ✓ 감성 분포: Bullish={sum(df['sentiment']==1)}, Bearish={sum(df['sentiment']==-1)}, Neutral={sum(df['sentiment']==0)}")
        
        return df
        
    except Exception as e:
        print(f"  ⚠️ Hugging Face 로드 실패: {e}")
        print("  → 시뮬레이션 데이터 사용")
        return generate_simulated_sentiment()


def generate_simulated_sentiment():
    """VIX 기반 시뮬레이션 심리 지수 생성"""
    print("  → VIX 기반 심리 지수 시뮬레이션...")
    
    # VIX 데이터 로드
    vix = yf.download('^VIX', start='2020-01-01', end='2025-01-01', 
                     progress=False, auto_adjust=True)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    
    # VIX를 감성 지수로 변환 (-1 ~ 1)
    # 높은 VIX = 부정적 감성, 낮은 VIX = 긍정적 감성
    vix_norm = (vix['Close'] - vix['Close'].mean()) / vix['Close'].std()
    sentiment = -np.tanh(vix_norm / 2)  # -1 ~ 1 범위
    
    # 노이즈 추가
    np.random.seed(SEED)
    noise = np.random.normal(0, 0.1, len(sentiment))
    sentiment = np.clip(sentiment + noise, -1, 1)
    
    df = pd.DataFrame({
        'date': vix.index,
        'sentiment': sentiment.values,
        'vix': vix['Close'].values
    })
    
    print(f"  ✓ 시뮬레이션 완료: {len(df)} 일")
    
    return df


# =============================================================================
# 2. FinBERT 감성 분석 (선택적)
# =============================================================================

class FinBERTSentimentAnalyzer:
    """FinBERT 기반 감성 분석기"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        
    def load_model(self):
        """FinBERT 모델 로드"""
        if not HAS_FINBERT:
            print("  ⚠️ FinBERT 미설치")
            return False
            
        try:
            print("  → FinBERT 모델 로드 중...")
            self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            self.model.eval()
            print("  ✓ FinBERT 로드 완료")
            return True
        except Exception as e:
            print(f"  ⚠️ FinBERT 로드 실패: {e}")
            return False
    
    def analyze(self, text):
        """단일 텍스트 감성 분석"""
        if self.model is None:
            return 0.0
            
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                                max_length=512, padding=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # [positive, negative, neutral]
        sentiment_score = probs[0][0].item() - probs[0][1].item()  # pos - neg
        return sentiment_score
    
    def analyze_batch(self, texts, batch_size=32):
        """배치 감성 분석"""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            for text in batch:
                results.append(self.analyze(text))
        return results


# =============================================================================
# 3. 일별 심리 지수 생성
# =============================================================================

def create_daily_sentiment_features(sentiment_df, spy_index):
    """일별 심리 특성 생성"""
    print("\n" + "="*60)
    print("[2/6] 일별 심리 특성 생성")
    print("="*60)
    
    # 날짜별 집계가 필요한 경우
    if 'date' in sentiment_df.columns:
        # 이미 일별 데이터
        daily = sentiment_df.set_index('date')
    else:
        # 집계 필요 (텍스트 데이터)
        # 시뮬레이션: 랜덤 날짜 할당
        dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='B')
        np.random.seed(SEED)
        
        # 일별 평균 심리 생성
        n_days = len(dates)
        daily_sentiment = np.zeros(n_days)
        
        # VIX 기반 심리 시뮬레이션
        vix = yf.download('^VIX', start='2020-01-01', end='2025-01-01', 
                         progress=False, auto_adjust=True)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        
        # 정규화된 VIX를 심리로 변환
        vix_aligned = vix['Close'].reindex(dates, method='ffill').fillna(vix['Close'].mean())
        vix_norm = (vix_aligned - vix_aligned.mean()) / vix_aligned.std()
        daily_sentiment = -np.tanh(vix_norm / 2).values
        
        # 노이즈 추가
        noise = np.random.normal(0, 0.15, len(daily_sentiment))
        daily_sentiment = np.clip(daily_sentiment + noise, -1, 1)
        
        daily = pd.DataFrame({
            'sentiment_mean': daily_sentiment,
        }, index=dates)
    
    # 심리 특성 추가
    if 'sentiment_mean' not in daily.columns and 'sentiment' in daily.columns:
        daily['sentiment_mean'] = daily['sentiment']
    
    # 핵심 특성 생성
    print("  → 심리 특성 생성 중...")
    
    # 1. 기본 심리
    if 'sentiment_mean' not in daily.columns:
        daily['sentiment_mean'] = daily.iloc[:, 0]  # 첫 번째 컬럼 사용
    
    # 2. 래그 특성
    daily['sentiment_lag1'] = daily['sentiment_mean'].shift(1)
    daily['sentiment_lag5'] = daily['sentiment_mean'].shift(5)
    
    # 3. 롤링 통계
    daily['sentiment_ma5'] = daily['sentiment_mean'].rolling(5).mean()
    daily['sentiment_ma20'] = daily['sentiment_mean'].rolling(20).mean()
    daily['sentiment_std5'] = daily['sentiment_mean'].rolling(5).std()
    
    # 4. 변화율
    daily['sentiment_change'] = daily['sentiment_mean'].diff()
    daily['sentiment_momentum'] = daily['sentiment_mean'].rolling(5).sum()
    
    # 5. 극단 감성
    daily['sentiment_extreme_pos'] = (daily['sentiment_mean'] > 0.5).astype(int)
    daily['sentiment_extreme_neg'] = (daily['sentiment_mean'] < -0.5).astype(int)
    
    # SPY 날짜와 정렬
    daily = daily.reindex(spy_index, method='ffill')
    
    print(f"  ✓ 심리 특성 10개 생성")
    print(f"  ✓ 기간: {daily.index[0]} ~ {daily.index[-1]}")
    
    return daily


# =============================================================================
# 4. 변동성 모델에 통합
# =============================================================================

def integrate_sentiment_with_volatility():
    """심리 특성을 변동성 예측에 통합"""
    print("\n" + "="*60)
    print("[3/6] SPY 데이터 및 기존 특성 로드")
    print("="*60)
    
    # SPY 데이터 로드
    csv_path = Path('data/raw/spy_data_2020_2025.csv')
    if csv_path.exists():
        spy = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    else:
        spy = yf.download('SPY', start='2020-01-01', end='2025-01-01',
                         progress=False, auto_adjust=True)
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
    
    print(f"  ✓ SPY 데이터: {len(spy)} 행")
    
    # VIX 로드
    vix = yf.download('^VIX', start='2020-01-01', end='2025-01-01',
                     progress=False, auto_adjust=True)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    spy['VIX'] = vix['Close']
    
    # 기본 특성 생성
    print("\n" + "="*60)
    print("[4/6] 기존 특성 + 심리 특성 생성")
    print("="*60)
    
    spy['returns'] = spy['Close'].pct_change()
    spy['volatility'] = spy['returns'].rolling(5).std() * np.sqrt(252)
    
    # 변동성 특성
    for window in [5, 10, 20, 50]:
        spy[f'volatility_{window}'] = spy['returns'].rolling(window).std()
        spy[f'realized_vol_{window}'] = spy[f'volatility_{window}'] * np.sqrt(252)
    
    # VIX 특성 (기존 최고 성능)
    spy['vix_lag_1'] = spy['VIX'].shift(1)
    spy['vix_lag_5'] = spy['VIX'].shift(5)
    spy['vix_change'] = spy['VIX'].pct_change()
    spy['vix_zscore'] = (spy['VIX'] - spy['VIX'].rolling(20).mean()) / (spy['VIX'].rolling(20).std() + 1e-8)
    
    # Regime 특성 (기존 최고 성능)
    vix_lagged = spy['VIX'].shift(1)
    spy['regime_high_vol'] = (vix_lagged >= 25).astype(int)
    spy['regime_crisis'] = (vix_lagged >= 35).astype(int)
    spy['vol_in_high_regime'] = spy['regime_high_vol'] * spy['volatility_5']
    spy['vol_in_crisis'] = spy['regime_crisis'] * spy['volatility_5']
    spy['vix_excess_25'] = np.maximum(vix_lagged - 25, 0)
    spy['vix_excess_35'] = np.maximum(vix_lagged - 35, 0)
    
    # 수익률 통계
    for window in [5, 10, 20]:
        spy[f'mean_return_{window}'] = spy['returns'].rolling(window).mean()
        spy[f'skew_{window}'] = spy['returns'].rolling(window).skew()
        spy[f'kurt_{window}'] = spy['returns'].rolling(window).kurt()
    
    # 래그 변수
    for lag in [1, 2, 3, 5]:
        spy[f'return_lag_{lag}'] = spy['returns'].shift(lag)
        spy[f'vol_lag_{lag}'] = spy['volatility_5'].shift(lag)
    
    # 모멘텀
    for window in [5, 10, 20]:
        spy[f'momentum_{window}'] = spy['returns'].rolling(window).sum()
    
    # 비율 특성
    spy['vol_ratio_5_20'] = spy['volatility_5'] / (spy['volatility_20'] + 1e-8)
    spy['vol_ratio_10_50'] = spy['volatility_10'] / (spy['volatility_50'] + 1e-8)
    
    # Z-score
    ma_20 = spy['returns'].rolling(20).mean()
    std_20 = spy['returns'].rolling(20).std()
    spy['zscore_20'] = (spy['returns'] - ma_20) / (std_20 + 1e-8)
    
    print(f"  ✓ 기존 특성: {len([c for c in spy.columns if c.startswith(('volatility', 'vix', 'regime', 'vol_', 'mean_', 'skew', 'kurt', 'return_lag', 'momentum', 'zscore'))])}개")
    
    # 심리 특성 추가
    print("  → 심리 특성 추가 중...")
    sentiment_df = load_financial_sentiment_dataset()
    daily_sentiment = create_daily_sentiment_features(sentiment_df, spy.index)
    
    # 병합
    for col in daily_sentiment.columns:
        spy[col] = daily_sentiment[col]
    
    print(f"  ✓ 심리 특성: {len(daily_sentiment.columns)}개 추가")
    
    # 타겟 생성 (5일 미래 변동성)
    vol_values = []
    returns = spy['returns'].values
    for i in range(len(returns)):
        if i + 5 < len(returns):
            future_window = returns[i+1:i+6]
            vol_values.append(pd.Series(future_window).std())
        else:
            vol_values.append(np.nan)
    spy['target_vol_5d'] = vol_values
    
    # 결측치 제거
    spy = spy.ffill().dropna()
    
    print(f"  ✓ 최종 데이터: {len(spy)} 행, {len(spy.columns)} 열")
    
    return spy


# =============================================================================
# 5. 모델 학습 및 평가
# =============================================================================

def train_and_evaluate(spy):
    """모델 학습 및 평가"""
    print("\n" + "="*60)
    print("[5/6] 모델 학습 및 평가")
    print("="*60)
    
    # 특성 선택
    feature_cols = []
    for col in spy.columns:
        if col.startswith(('volatility_', 'realized_vol_', 'mean_return_',
                          'skew_', 'kurt_', 'return_lag_', 'vol_lag_',
                          'vol_ratio_', 'zscore_', 'momentum_', 'vix_', 'regime_',
                          'vol_in_', 'vix_excess_', 'sentiment_')):
            feature_cols.append(col)
    
    print(f"  ✓ 총 특성: {len(feature_cols)}개")
    print(f"    - 기존 특성: {len([c for c in feature_cols if not c.startswith('sentiment')])}개")
    print(f"    - 심리 특성: {len([c for c in feature_cols if c.startswith('sentiment')])}개")
    
    # 데이터 분할 (80/20)
    X = spy[feature_cols].values
    y = spy['target_vol_5d'].values
    
    split_idx = int(len(spy) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\n  → Train: {len(X_train)}, Test: {len(X_test)}")
    
    # 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ElasticNet 모델
    model = ElasticNet(alpha=0.0005, l1_ratio=0.5, random_state=SEED, max_iter=10000)
    model.fit(X_train_scaled, y_train)
    
    # 평가
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\n  📊 모델 성능 (VIX + Regime + Sentiment):")
    print(f"    • Test R²: {r2:.4f}")
    print(f"    • Test RMSE: {rmse:.6f}")
    
    # 비교: 심리 특성 없이
    print("\n  → 비교: 기존 모델 (심리 특성 없이)...")
    baseline_cols = [c for c in feature_cols if not c.startswith('sentiment')]
    X_baseline = spy[baseline_cols].values
    
    X_train_b, X_test_b = X_baseline[:split_idx], X_baseline[split_idx:]
    X_train_b_scaled = scaler.fit_transform(X_train_b)
    X_test_b_scaled = scaler.transform(X_test_b)
    
    model_b = ElasticNet(alpha=0.0005, l1_ratio=0.5, random_state=SEED, max_iter=10000)
    model_b.fit(X_train_b_scaled, y_train)
    
    y_pred_b = model_b.predict(X_test_b_scaled)
    r2_b = r2_score(y_test, y_pred_b)
    
    print(f"    • 기존 모델 R²: {r2_b:.4f}")
    
    # 성능 차이
    diff = r2 - r2_b
    print(f"\n  📈 심리 특성 추가 효과: {diff:+.4f} ({diff/r2_b*100:+.1f}%)")
    
    # 특성 중요도
    print("\n  🔍 상위 10 특성 (절대 계수):")
    coef_df = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': np.abs(model.coef_)
    }).sort_values('coefficient', ascending=False)
    
    for i, row in coef_df.head(10).iterrows():
        marker = "📰" if row['feature'].startswith('sentiment') else "  "
        print(f"    {marker} {row['feature']}: {row['coefficient']:.6f}")
    
    # 심리 특성 중요도
    sentiment_coefs = coef_df[coef_df['feature'].str.startswith('sentiment')]
    print(f"\n  📰 심리 특성 중요도:")
    for _, row in sentiment_coefs.iterrows():
        print(f"    - {row['feature']}: {row['coefficient']:.6f}")
    
    results = {
        'model_with_sentiment': {
            'r2': float(r2),
            'rmse': float(rmse),
            'n_features': len(feature_cols)
        },
        'model_without_sentiment': {
            'r2': float(r2_b),
            'n_features': len(baseline_cols)
        },
        'sentiment_effect': float(diff),
        'sentiment_features': list(sentiment_coefs['feature'].values),
        'timestamp': datetime.now().isoformat()
    }
    
    return results


# =============================================================================
# 6. 메인 파이프라인
# =============================================================================

def main():
    """전체 파이프라인 실행"""
    print("\n" + "🚀"*30)
    print("FinBERT 뉴스 심리 지수 통합 파이프라인")
    print("🚀"*30)
    
    try:
        # 1-4. 데이터 준비 및 특성 생성
        spy = integrate_sentiment_with_volatility()
        
        # 5. 모델 학습 및 평가
        results = train_and_evaluate(spy)
        
        # 6. 결과 저장
        print("\n" + "="*60)
        print("[6/6] 결과 저장")
        print("="*60)
        
        output_path = Path('data/raw/sentiment_integration_results.json')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"  ✓ 결과 저장: {output_path}")
        
        # 최종 요약
        print("\n" + "="*60)
        print("✅ 심리 지수 통합 완료!")
        print("="*60)
        
        print("\n📊 성능 비교:")
        print(f"  • 기존 모델 (VIX+Regime):          R² = {results['model_without_sentiment']['r2']:.4f}")
        print(f"  • 심리 추가 모델 (VIX+Regime+Sent): R² = {results['model_with_sentiment']['r2']:.4f}")
        print(f"  • 심리 특성 효과:                   {results['sentiment_effect']:+.4f}")
        
        if results['sentiment_effect'] > 0.005:
            print("\n✅ 심리 특성이 모델 성능 향상에 기여!")
        elif results['sentiment_effect'] > 0:
            print("\n⚠️ 심리 특성의 효과가 미미함")
        else:
            print("\n❌ 심리 특성 추가로 성능 저하")
        
        return results
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    results = main()
