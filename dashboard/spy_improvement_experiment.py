#!/usr/bin/env python3
"""
SPY 예측 모델 개선 실험
Phase 1: VIX 통합 + 앙상블 + 신뢰도 필터링
"""

import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class SPYImprovementExperiment:
    def __init__(self):
        self.spy_data = None
        self.vix_data = None
        self.prediction_data = None
        self.enhanced_features = None
        self.models = {}
        self.results = {}
        
    def load_current_data(self):
        """현재 SPY 예측 데이터 로드"""
        print("📥 현재 데이터 로딩...")
        
        # SPY 실제 데이터
        with open('data/raw/spy_2025_h1.json', 'r') as f:
            spy_raw = json.load(f)
            
        # AI 예측 데이터
        with open('data/raw/spy_2025_h1_predictions.json', 'r') as f:
            self.prediction_data = json.load(f)
            
        # DataFrame으로 변환
        spy_df = pd.DataFrame(spy_raw['data'])
        spy_df['date'] = pd.to_datetime(spy_df['date'])
        spy_df = spy_df.set_index('date')
        
        self.spy_data = spy_df
        print(f"✅ SPY 데이터 로드 완료: {len(spy_df)} 일")
        
    def collect_vix_data(self):
        """VIX 데이터 수집"""
        print("📈 VIX 데이터 수집 중...")
        
        try:
            # VIX 데이터 다운로드 (2024년 12월부터 여유있게)
            vix_raw = yf.download('^VIX', start='2024-12-01', end='2025-07-01', auto_adjust=True, progress=False)
            
            # 날짜 인덱스 정렬 및 정리
            self.vix_data = vix_raw.copy()
            
            # 2025년 1-6월 데이터만 필터링
            start_date = '2025-01-01'
            end_date = '2025-06-30'
            self.vix_data = self.vix_data[start_date:end_date]
            
            print(f"✅ VIX 데이터 수집 완료: {len(self.vix_data)} 일")
            print(f"VIX 범위: {self.vix_data['Close'].min():.2f} - {self.vix_data['Close'].max():.2f}")
            
        except Exception as e:
            print(f"❌ VIX 데이터 수집 실패: {str(e)}")
            # 임시 VIX 데이터 생성
            self.create_mock_vix_data()
    
    def create_mock_vix_data(self):
        """VIX 데이터 수집 실패시 목 데이터 생성"""
        print("🔧 목 VIX 데이터 생성 중...")
        
        dates = self.spy_data.index
        # 실제 VIX 패턴을 모방한 데이터 (15-35 범위)
        np.random.seed(42)
        vix_values = 20 + 10 * np.sin(np.linspace(0, 4*np.pi, len(dates))) + np.random.normal(0, 3, len(dates))
        vix_values = np.clip(vix_values, 12, 40)  # VIX 현실적 범위
        
        self.vix_data = pd.DataFrame({
            'Close': vix_values,
            'Open': vix_values * 1.02,
            'High': vix_values * 1.05,
            'Low': vix_values * 0.95,
            'Volume': np.random.randint(100000, 500000, len(dates))
        }, index=dates)
        
        print("✅ 목 VIX 데이터 생성 완료")
        
    def create_enhanced_features(self):
        """향상된 특성 생성"""
        print("🔧 향상된 특성 생성 중...")
        
        # 기본 SPY 특성
        spy_features = self.spy_data.copy()
        
        # 가격 기반 특성
        spy_features['returns'] = spy_features['close'].pct_change()
        spy_features['log_returns'] = np.log(spy_features['close'] / spy_features['close'].shift(1))
        
        # 이동평균
        for period in [5, 10, 20, 50]:
            spy_features[f'ma_{period}'] = spy_features['close'].rolling(period).mean()
            spy_features[f'price_to_ma_{period}'] = spy_features['close'] / spy_features[f'ma_{period}']
        
        # RSI
        def calculate_rsi(prices, period=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
            
        spy_features['rsi'] = calculate_rsi(spy_features['close'])
        
        # 볼린저 밴드
        spy_features['bb_middle'] = spy_features['close'].rolling(20).mean()
        bb_std = spy_features['close'].rolling(20).std()
        spy_features['bb_upper'] = spy_features['bb_middle'] + (bb_std * 2)
        spy_features['bb_lower'] = spy_features['bb_middle'] - (bb_std * 2)
        spy_features['bb_position'] = (spy_features['close'] - spy_features['bb_lower']) / (spy_features['bb_upper'] - spy_features['bb_lower'])
        
        # VIX 특성 추가
        vix_aligned = self.vix_data.reindex(spy_features.index, method='ffill')
        spy_features['vix'] = vix_aligned['Close']
        spy_features['vix_ma_5'] = spy_features['vix'].rolling(5).mean()
        spy_features['vix_change'] = spy_features['vix'].pct_change()
        
        # VIX 시그널 (핵심 개선사항)
        spy_features['vix_signal'] = (spy_features['vix'] <= 20).astype(int)  # VIX 낮으면 상승 신호
        spy_features['vix_regime'] = pd.cut(spy_features['vix'], bins=[0, 15, 20, 25, 100], labels=['low', 'normal', 'high', 'extreme'])
        
        # 거래량 특성
        spy_features['volume_ma'] = spy_features['volume'].rolling(20).mean()
        spy_features['volume_ratio'] = spy_features['volume'] / spy_features['volume_ma']
        
        # 변동성 특성
        spy_features['volatility'] = spy_features['returns'].rolling(20).std()
        spy_features['high_low_ratio'] = spy_features['high'] / spy_features['low']
        
        self.enhanced_features = spy_features
        print(f"✅ {len(spy_features.columns)}개 특성 생성 완료")
        
        # 결측치 처리
        self.enhanced_features = self.enhanced_features.fillna(method='ffill').fillna(method='bfill')
        
    def prepare_training_data(self):
        """학습 데이터 준비"""
        print("📊 학습 데이터 준비 중...")
        
        # 예측 데이터를 DataFrame으로 변환
        pred_df = pd.DataFrame(self.prediction_data['predictions'])
        pred_df['date'] = pd.to_datetime(pred_df['date'])
        pred_df = pred_df.set_index('date')
        
        # 특성과 타겟 결합
        combined_data = self.enhanced_features.join(pred_df[['prediction', 'actual_return', 'confidence']], how='inner')
        
        # 타겟 변수: 실제 수익률 방향
        combined_data['target'] = (combined_data['actual_return'] > 0).astype(int)
        
        # 특성 선택 (수치형만)
        feature_columns = [
            'returns', 'log_returns', 'rsi', 'bb_position', 
            'vix', 'vix_change', 'vix_signal', 'volume_ratio', 
            'volatility', 'high_low_ratio', 'price_to_ma_5', 
            'price_to_ma_10', 'price_to_ma_20'
        ]
        
        # 결측치가 있는 행 제거
        combined_data = combined_data.dropna()
        
        X = combined_data[feature_columns]
        y = combined_data['target']
        confidence = combined_data['confidence']
        
        print(f"✅ 학습 데이터 준비 완료: {len(X)} 샘플, {len(feature_columns)} 특성")
        
        return X, y, confidence, combined_data
        
    def train_baseline_model(self, X, y):
        """기존 모델 (기준선)"""
        print("🤖 기준선 모델 학습 중...")
        
        # 기본 Random Forest
        rf_baseline = RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            class_weight='balanced'
        )
        
        # 시계열 교차검증
        tscv = TimeSeriesSplit(n_splits=5)
        scores = cross_val_score(rf_baseline, X, y, cv=tscv, scoring='accuracy')
        
        # 전체 데이터로 학습
        rf_baseline.fit(X, y)
        
        self.models['baseline'] = rf_baseline
        self.results['baseline'] = {
            'cv_mean': scores.mean(),
            'cv_std': scores.std(),
            'model': rf_baseline
        }
        
        print(f"✅ 기준선 정확도: {scores.mean():.3f} ± {scores.std():.3f}")
        
    def train_ensemble_model(self, X, y):
        """앙상블 모델 학습"""
        print("🎯 앙상블 모델 학습 중...")
        
        # 다양한 모델들
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
        
        # VotingClassifier
        ensemble = VotingClassifier(
            estimators=[('rf', rf), ('gb', gb)],
            voting='soft'
        )
        
        # 시계열 교차검증
        tscv = TimeSeriesSplit(n_splits=5)
        scores = cross_val_score(ensemble, X, y, cv=tscv, scoring='accuracy')
        
        # 전체 데이터로 학습
        ensemble.fit(X, y)
        
        self.models['ensemble'] = ensemble
        self.results['ensemble'] = {
            'cv_mean': scores.mean(),
            'cv_std': scores.std(),
            'model': ensemble
        }
        
        print(f"✅ 앙상블 정확도: {scores.mean():.3f} ± {scores.std():.3f}")
        
    def test_confidence_filtering(self, X, y, confidence):
        """신뢰도 필터링 테스트"""
        print("🔍 신뢰도 필터링 테스트 중...")
        
        results = {}
        
        for threshold in [0.5, 0.6, 0.7, 0.8]:
            # 고신뢰도 데이터만 선택
            high_conf_mask = confidence >= threshold
            if high_conf_mask.sum() == 0:
                continue
                
            X_filtered = X[high_conf_mask]
            y_filtered = y[high_conf_mask]
            
            # 앙상블 모델로 예측
            if len(np.unique(y_filtered)) > 1:  # 클래스가 1개 이상인 경우만
                model = self.models['ensemble']
                pred = model.predict(X_filtered)
                accuracy = accuracy_score(y_filtered, pred)
                
                results[f'conf_{threshold}'] = {
                    'accuracy': accuracy,
                    'sample_size': len(X_filtered),
                    'coverage': len(X_filtered) / len(X)
                }
            
        self.results['confidence_filtering'] = results
        
        print("✅ 신뢰도 필터링 결과:")
        for key, result in results.items():
            print(f"  {key}: {result['accuracy']:.3f} (n={result['sample_size']}, 커버리지: {result['coverage']:.1%})")
    
    def evaluate_vix_contribution(self, X, y):
        """VIX 기여도 평가"""
        print("📈 VIX 기여도 분석 중...")
        
        # VIX 없이 학습
        X_no_vix = X.drop(['vix', 'vix_change', 'vix_signal'], axis=1, errors='ignore')
        
        rf_no_vix = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        tscv = TimeSeriesSplit(n_splits=5)
        scores_no_vix = cross_val_score(rf_no_vix, X_no_vix, y, cv=tscv, scoring='accuracy')
        
        # VIX와 함께 학습한 결과와 비교
        baseline_score = self.results['baseline']['cv_mean']
        vix_contribution = baseline_score - scores_no_vix.mean()
        
        self.results['vix_analysis'] = {
            'with_vix': baseline_score,
            'without_vix': scores_no_vix.mean(),
            'vix_contribution': vix_contribution
        }
        
        print(f"✅ VIX 기여도: {vix_contribution:+.3f}")
        print(f"   VIX 포함: {baseline_score:.3f}")
        print(f"   VIX 제외: {scores_no_vix.mean():.3f}")
        
    def generate_improved_predictions(self, X, y, confidence):
        """개선된 예측 생성"""
        print("🔮 개선된 예측 생성 중...")
        
        # 앙상블 모델 예측
        ensemble_pred = self.models['ensemble'].predict(X)
        ensemble_proba = self.models['ensemble'].predict_proba(X)
        
        # 결과 저장
        improved_predictions = []
        
        for i, (date, row) in enumerate(X.iterrows()):
            pred_info = {
                'date': date.strftime('%Y-%m-%d'),
                'original_prediction': int(y.iloc[i]),  # 실제 결과
                'ensemble_prediction': int(ensemble_pred[i]),
                'ensemble_confidence': float(ensemble_proba[i].max()),
                'original_confidence': float(confidence.iloc[i]),
                'vix_value': float(row['vix']) if 'vix' in row else None,
                'vix_signal': int(row['vix_signal']) if 'vix_signal' in row else None
            }
            improved_predictions.append(pred_info)
        
        # 파일로 저장
        output_data = {
            'experiment_info': {
                'date': datetime.now().isoformat(),
                'baseline_accuracy': float(self.results['baseline']['cv_mean']),
                'ensemble_accuracy': float(self.results['ensemble']['cv_mean']),
                'improvement': float(self.results['ensemble']['cv_mean'] - self.results['baseline']['cv_mean']),
                'vix_contribution': float(self.results['vix_analysis']['vix_contribution'])
            },
            'predictions': improved_predictions
        }
        
        with open('data/raw/spy_improved_predictions.json', 'w') as f:
            json.dump(output_data, f, indent=2)
            
        print(f"✅ 개선된 예측 저장 완료: {len(improved_predictions)}개")
        
    def create_comparison_report(self):
        """비교 보고서 생성"""
        print("📋 성과 비교 보고서 생성 중...")
        
        report = {
            'experiment_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'original_model': {
                'accuracy': 0.5455,  # 원래 모델
                'description': 'Technical Analysis (RSI, MACD, Bollinger Bands)'
            },
            'improvements': {}
        }
        
        # 각 개선사항별 결과
        baseline_acc = self.results['baseline']['cv_mean']
        ensemble_acc = self.results['ensemble']['cv_mean']
        
        report['improvements']['baseline_with_vix'] = {
            'accuracy': float(baseline_acc),
            'improvement_vs_original': float(baseline_acc - 0.5455),
            'description': 'RF + VIX integration'
        }
        
        report['improvements']['ensemble'] = {
            'accuracy': float(ensemble_acc),
            'improvement_vs_original': float(ensemble_acc - 0.5455),
            'improvement_vs_baseline': float(ensemble_acc - baseline_acc),
            'description': 'RF + Gradient Boosting Ensemble'
        }
        
        # 신뢰도 필터링 결과
        if 'confidence_filtering' in self.results:
            best_conf_result = max(
                self.results['confidence_filtering'].values(), 
                key=lambda x: x['accuracy']
            )
            
            report['improvements']['confidence_filtering'] = {
                'best_accuracy': float(best_conf_result['accuracy']),
                'improvement_vs_original': float(best_conf_result['accuracy'] - 0.5455),
                'coverage': float(best_conf_result['coverage']),
                'description': 'High confidence predictions only'
            }
        
        # VIX 기여도
        if 'vix_analysis' in self.results:
            report['vix_analysis'] = {
                'contribution': float(self.results['vix_analysis']['vix_contribution']),
                'with_vix': float(self.results['vix_analysis']['with_vix']),
                'without_vix': float(self.results['vix_analysis']['without_vix'])
            }
        
        # 보고서 저장
        with open('data/raw/improvement_experiment_report.json', 'w') as f:
            json.dump(report, f, indent=2)
            
        return report
        
    def run_experiment(self):
        """전체 실험 실행"""
        print("🚀 SPY 예측 모델 개선 실험 시작!")
        print("=" * 50)
        
        # 단계별 실행
        self.load_current_data()
        self.collect_vix_data()
        self.create_enhanced_features()
        
        X, y, confidence, combined_data = self.prepare_training_data()
        
        self.train_baseline_model(X, y)
        self.train_ensemble_model(X, y)
        self.test_confidence_filtering(X, y, confidence)
        self.evaluate_vix_contribution(X, y)
        
        self.generate_improved_predictions(X, y, confidence)
        report = self.create_comparison_report()
        
        print("\n" + "=" * 50)
        print("🏆 실험 결과 요약:")
        print(f"📊 원래 모델: {report['original_model']['accuracy']:.1%}")
        
        if 'baseline_with_vix' in report['improvements']:
            baseline_result = report['improvements']['baseline_with_vix']
            print(f"📈 VIX 통합: {baseline_result['accuracy']:.1%} ({baseline_result['improvement_vs_original']:+.1%})")
            
        if 'ensemble' in report['improvements']:
            ensemble_result = report['improvements']['ensemble']
            print(f"🎯 앙상블: {ensemble_result['accuracy']:.1%} ({ensemble_result['improvement_vs_original']:+.1%})")
            
        if 'confidence_filtering' in report['improvements']:
            conf_result = report['improvements']['confidence_filtering']
            print(f"🔍 신뢰도 필터링: {conf_result['best_accuracy']:.1%} ({conf_result['improvement_vs_original']:+.1%})")
            
        print(f"\n✅ 실험 완료! 상세 보고서는 data/raw/ 폴더에 저장되었습니다.")

def main():
    experiment = SPYImprovementExperiment()
    experiment.run_experiment()

if __name__ == "__main__":
    main()