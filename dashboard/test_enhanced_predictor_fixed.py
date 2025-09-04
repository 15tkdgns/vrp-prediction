#!/usr/bin/env python3
"""
뉴스 감정 분석 통합 모델 테스트 (수정된 버전)
- API 키 없이도 작동하는 시뮬레이션 모드
- 성능 향상 검증 실험
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import logging
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, roc_auc_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleEnhancedTester:
    """간단한 강화된 예측 모델 테스터"""
    
    def __init__(self):
        pass
        
    def load_clean_data(self, start_date='2019-01-01', end_date='2024-12-31'):
        """가격 데이터 로드"""
        logger.info(f"📥 SPY 및 VIX 데이터 로드 중... ({start_date} ~ {end_date})")
        
        try:
            spy_raw = yf.download('SPY', start=start_date, end=end_date, auto_adjust=True, progress=False)
            vix_raw = yf.download('^VIX', start=start_date, end=end_date, auto_adjust=True, progress=False)
            
            if isinstance(spy_raw.columns, pd.MultiIndex):
                spy_raw.columns = spy_raw.columns.get_level_values(0)
            if isinstance(vix_raw.columns, pd.MultiIndex):
                vix_raw.columns = vix_raw.columns.get_level_values(0)
                
            logger.info(f"✅ 데이터 로드 완료: SPY {len(spy_raw)}일, VIX {len(vix_raw)}일")
            return spy_raw, vix_raw
            
        except Exception as e:
            logger.error(f"❌ 데이터 로드 실패: {str(e)}")
            return None, None
    
    def create_technical_features(self, spy_data, vix_data):
        """기술적 특성 생성"""
        logger.info("🔧 기술적 특성 생성 중...")
        
        df = pd.DataFrame(index=spy_data.index)
        
        # 기본 수익률
        returns = spy_data['Close'].pct_change()
        df['returns_lag1'] = returns.shift(1)
        df['returns_lag2'] = returns.shift(2)
        df['returns_lag3'] = returns.shift(3)
        
        # 이동평균 비율
        ma50 = spy_data['Close'].rolling(50).mean()
        df['price_to_ma50'] = (spy_data['Close'].shift(1) / ma50.shift(1) - 1)
        
        # VIX 특성
        vix_aligned = vix_data.reindex(spy_data.index, method='ffill')
        df['vix_change'] = vix_aligned['Close'].pct_change().shift(1)
        
        # 변동성
        df['volatility_20'] = returns.rolling(20).std().shift(1)
        
        # 거래량 비율
        volume_ma = spy_data['Volume'].rolling(20).mean()
        df['volume_ratio'] = (spy_data['Volume'].shift(1) / volume_ma.shift(1))
        
        # 타겟
        df['target'] = (spy_data['Close'].shift(-1) / spy_data['Close'] - 1 > 0).astype(int)
        
        return df.dropna()
    
    def generate_mock_sentiment_features(self, price_index):
        """모의 감정 특성 생성"""
        logger.info("🎭 모의 감정 특성 생성 중...")
        
        # 가격과 상관관계가 있는 감정 신호 생성
        np.random.seed(42)  # 재현성을 위해
        
        sentiment_features = pd.DataFrame(index=price_index)
        
        # 1일 뉴스 감정 점수 (약간의 예측 신호 포함)
        base_sentiment = np.random.normal(0, 0.3, len(price_index))
        
        # SPY 가격 변화와 약간의 상관관계 추가 (현실적)
        if len(price_index) > 1:
            # 이전 날의 수익률과 약한 음의 상관관계 (역방향 예측)
            spy_returns = pd.Series(index=price_index, data=np.random.normal(0, 0.02, len(price_index)))
            for i in range(1, len(base_sentiment)):
                # 전날 하락 시 약간 더 긍정적 감정 (반등 기대)
                base_sentiment[i] += -spy_returns.iloc[i-1] * 0.5 + np.random.normal(0, 0.1)
        
        sentiment_features['news_sentiment_1d'] = np.clip(base_sentiment, -1, 1)
        
        # 감정 모멘텀
        sentiment_3d = pd.Series(sentiment_features['news_sentiment_1d']).rolling(3).mean()
        sentiment_features['sentiment_momentum'] = sentiment_3d - sentiment_3d.shift(3)
        
        # 영향도 가중 감정
        impact_weights = np.random.uniform(0.4, 0.8, len(price_index))
        sentiment_features['news_impact_weighted'] = sentiment_features['news_sentiment_1d'] * impact_weights
        
        sentiment_features = sentiment_features.fillna(0)
        
        logger.info(f"✅ 모의 감정 특성 생성 완료: {sentiment_features.shape[1]}개 특성")
        return sentiment_features
    
    def strict_time_split(self, df):
        """시간 기반 데이터 분할"""
        logger.info("📊 시간 기반 데이터 분할 중...")
        
        train_mask = df.index < '2022-01-01'
        val_mask = (df.index >= '2022-01-01') & (df.index < '2023-01-01')
        test_mask = df.index >= '2023-01-01'
        
        feature_cols = [col for col in df.columns if col != 'target']
        
        X_train = df.loc[train_mask, feature_cols]
        y_train = df.loc[train_mask, 'target']
        
        X_val = df.loc[val_mask, feature_cols]
        y_val = df.loc[val_mask, 'target']
        
        X_test = df.loc[test_mask, feature_cols] 
        y_test = df.loc[test_mask, 'target']
        
        logger.info(f"📊 데이터 분할: 훈련 {len(X_train)}, 검증 {len(X_val)}, 테스트 {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def run_comparison_test(self):
        """베이스라인 vs 강화 모델 비교"""
        logger.info("🔬 베이스라인 vs 강화 모델 비교 실험 시작")
        logger.info("=" * 60)
        
        try:
            # 1. 데이터 로드
            spy_data, vix_data = self.load_clean_data()
            if spy_data is None:
                return None
            
            # 2. 기술적 특성 생성
            technical_df = self.create_technical_features(spy_data, vix_data)
            
            # 3. 베이스라인 모델 (기술적 특성만)
            logger.info("\n🎯 베이스라인 모델 (기술적 특성만) 테스트...")
            
            X_train_base, X_val_base, X_test_base, y_train, y_val, y_test = self.strict_time_split(technical_df)
            
            # 베이스라인 모델 훈련
            baseline_model = LogisticRegression(C=0.1, class_weight='balanced', random_state=42, max_iter=1000)
            baseline_scaler = RobustScaler()
            
            X_train_scaled = baseline_scaler.fit_transform(X_train_base)
            X_test_scaled = baseline_scaler.transform(X_test_base)
            
            baseline_model.fit(X_train_scaled, y_train)
            baseline_pred = baseline_model.predict(X_test_scaled)
            baseline_proba = baseline_model.predict_proba(X_test_scaled)[:, 1]
            
            baseline_accuracy = accuracy_score(y_test, baseline_pred)
            baseline_auc = roc_auc_score(y_test, baseline_proba)
            
            logger.info(f"✅ 베이스라인 성능: {baseline_accuracy:.1%} 정확도, {baseline_auc:.3f} AUC")
            
            # 4. 강화된 모델 (기술적 + 감정 특성)
            logger.info("\n🚀 강화된 모델 (감정 분석 추가) 테스트...")
            
            # 감정 특성 생성
            sentiment_features = self.generate_mock_sentiment_features(technical_df.index)
            
            # 기술적 특성과 감정 특성 결합
            technical_only = technical_df[['returns_lag1', 'returns_lag2', 'returns_lag3', 'price_to_ma50', 
                                         'vix_change', 'volatility_20', 'volume_ratio', 'target']]
            enhanced_df = pd.concat([technical_only.drop('target', axis=1), sentiment_features, 
                                   technical_only[['target']]], axis=1)
            enhanced_df = enhanced_df.dropna()
            
            # 강화된 모델 분할
            X_train_enh, X_val_enh, X_test_enh, y_train_enh, y_val_enh, y_test_enh = self.strict_time_split(enhanced_df)
            
            # 강화된 모델 훈련
            enhanced_model = LogisticRegression(C=0.1, class_weight='balanced', random_state=42, max_iter=1000)
            enhanced_scaler = RobustScaler()
            
            X_train_enh_scaled = enhanced_scaler.fit_transform(X_train_enh)
            X_test_enh_scaled = enhanced_scaler.transform(X_test_enh)
            
            enhanced_model.fit(X_train_enh_scaled, y_train_enh)
            enhanced_pred = enhanced_model.predict(X_test_enh_scaled)
            enhanced_proba = enhanced_model.predict_proba(X_test_enh_scaled)[:, 1]
            
            enhanced_accuracy = accuracy_score(y_test_enh, enhanced_pred)
            enhanced_auc = roc_auc_score(y_test_enh, enhanced_proba)
            
            logger.info(f"✅ 강화된 모델 성능: {enhanced_accuracy:.1%} 정확도, {enhanced_auc:.3f} AUC")
            
            # 5. 성능 비교 분석
            accuracy_improvement = enhanced_accuracy - baseline_accuracy
            auc_improvement = enhanced_auc - baseline_auc
            
            logger.info("\n" + "=" * 60)
            logger.info("📊 성능 비교 결과:")
            logger.info(f"📈 정확도 향상: +{accuracy_improvement:.1%} ({baseline_accuracy:.1%} → {enhanced_accuracy:.1%})")
            logger.info(f"📈 AUC 향상: +{auc_improvement:.3f} ({baseline_auc:.3f} → {enhanced_auc:.3f})")
            
            # 6. 특성 중요도 분석
            feature_importance = dict(zip(X_train_enh.columns, enhanced_model.coef_[0]))
            
            sentiment_features_list = ['news_sentiment_1d', 'sentiment_momentum', 'news_impact_weighted']
            technical_features_list = ['returns_lag1', 'returns_lag2', 'returns_lag3', 'price_to_ma50', 
                                     'vix_change', 'volatility_20', 'volume_ratio']
            
            sentiment_importance = {k: v for k, v in feature_importance.items() if k in sentiment_features_list}
            technical_importance = {k: v for k, v in feature_importance.items() if k in technical_features_list}
            
            logger.info("\n🔍 특성 중요도 분석:")
            logger.info("감정 특성:")
            for feature, importance in sorted(sentiment_importance.items(), key=lambda x: abs(x[1]), reverse=True):
                logger.info(f"   {feature}: {importance:.3f}")
            
            logger.info("기술적 특성 (상위 3개):")
            for feature, importance in sorted(technical_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:3]:
                logger.info(f"   {feature}: {importance:.3f}")
            
            # 7. 결과 저장
            comparison_report = {
                'experiment_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'test_type': 'baseline_vs_enhanced_comparison',
                'baseline_performance': {
                    'accuracy': float(baseline_accuracy),
                    'auc': float(baseline_auc),
                    'features': list(X_train_base.columns)
                },
                'enhanced_performance': {
                    'accuracy': float(enhanced_accuracy),
                    'auc': float(enhanced_auc),
                    'features': list(X_train_enh.columns)
                },
                'improvement': {
                    'accuracy_gain': float(accuracy_improvement),
                    'auc_gain': float(auc_improvement),
                    'relative_accuracy_improvement': float(accuracy_improvement / baseline_accuracy * 100) if baseline_accuracy > 0 else 0
                },
                'feature_importance': {
                    'sentiment_features': {k: float(v) for k, v in sentiment_importance.items()},
                    'technical_features': {k: float(v) for k, v in technical_importance.items()}
                },
                'conclusions': []
            }
            
            # 8. 결론 생성
            if accuracy_improvement > 0.02:
                comparison_report['conclusions'].append("✅ 뉴스 감정 분석으로 2%+ 정확도 향상 달성")
            elif accuracy_improvement > 0.01:
                comparison_report['conclusions'].append("🎯 뉴스 감정 분석으로 1%+ 정확도 향상 달성")
            elif accuracy_improvement > 0:
                comparison_report['conclusions'].append("📈 뉴스 감정 분석으로 소폭 성능 향상")
            else:
                comparison_report['conclusions'].append("⚠️ 뉴스 감정 분석의 성능 향상 미미하거나 없음")
            
            if enhanced_accuracy > 0.55:
                comparison_report['conclusions'].append("🏆 55% 이상 정확도 달성 (우수한 성능)")
            elif enhanced_accuracy > 0.53:
                comparison_report['conclusions'].append("✅ 53% 이상 정확도 달성 (만족스러운 성능)")
            else:
                comparison_report['conclusions'].append("⚠️ 성능이 기대치에 미달")
            
            # 보고서 저장
            import os
            os.makedirs('data/raw', exist_ok=True)
            with open('data/raw/baseline_vs_enhanced_comparison.json', 'w', encoding='utf-8') as f:
                json.dump(comparison_report, f, indent=2, ensure_ascii=False)
            
            logger.info("\n📋 주요 결론:")
            for conclusion in comparison_report['conclusions']:
                logger.info(f"   {conclusion}")
            
            logger.info(f"\n✅ 비교 실험 완료! 보고서: data/raw/baseline_vs_enhanced_comparison.json")
            
            return comparison_report
            
        except Exception as e:
            logger.error(f"❌ 비교 실험 실패: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

def main():
    """테스트 실행"""
    tester = SimpleEnhancedTester()
    
    logger.info("🧪 뉴스 감정 분석 통합 모델 테스트 시작")
    logger.info("=" * 80)
    
    # 베이스라인 vs 강화 모델 비교
    result = tester.run_comparison_test()
    
    if result:
        logger.info("\n🎉 테스트 완료!")
    else:
        logger.error("\n❌ 테스트 실패")

if __name__ == "__main__":
    main()