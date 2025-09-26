#!/usr/bin/env python3
"""
Simple XAI Analysis for SPY Volatility Prediction
의존성 문제 없이 즉시 실행 가능한 XAI 분석
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
import json
from datetime import datetime
from pathlib import Path

# 경고 무시
warnings.filterwarnings('ignore')

# 프로젝트 루트 추가
sys.path.append('/root/workspace')

try:
    import shap
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.inspection import partial_dependence, permutation_importance
    DEPENDENCIES_OK = True
    print("✅ 모든 XAI 의존성 로드 완료")
except ImportError as e:
    print(f"❌ 의존성 부족: {e}")
    DEPENDENCIES_OK = False

try:
    import yfinance as yf
    YFINANCE_OK = True
except ImportError:
    YFINANCE_OK = False

class SimpleXAIAnalyzer:
    """간단한 XAI 분석기 - 즉시 실행 가능"""

    def __init__(self):
        print("🚀 Simple XAI Analyzer 초기화...")
        self.model = None
        self.scaler = None
        self.feature_names = []
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    def generate_spy_features(self, start_date='2020-01-01', end_date='2024-01-01'):
        """SPY 데이터 로드 및 특성 생성"""
        print("📊 SPY 데이터 로드 및 특성 생성 중...")

        if YFINANCE_OK:
            try:
                spy = yf.Ticker("SPY")
                data = spy.history(start=start_date, end=end_date)
                prices = data['Close']
                print(f"✅ 실제 SPY 데이터: {len(data)} 관측치")
            except Exception as e:
                print(f"⚠️ yfinance 오류: {e}, 시뮬레이션 데이터 사용")
                dates = pd.date_range(start=start_date, end=end_date, freq='D')
                np.random.seed(42)
                prices = pd.Series(100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, len(dates)))),
                                 index=dates)
        else:
            print("📊 시뮬레이션 SPY 데이터 생성 중...")
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            np.random.seed(42)
            prices = pd.Series(100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, len(dates)))),
                             index=dates)

        # 수익률 계산
        returns = prices.pct_change()

        # 특성 생성
        features = pd.DataFrame(index=prices.index)

        print("🔧 특성 생성 중...")

        # 1. 변동성 특성 (핵심)
        for window in [5, 10, 20]:
            features[f'vol_{window}'] = returns.rolling(window).std() * np.sqrt(252)

        # 2. 수익률 지연 특성
        for lag in [1, 2, 3]:
            features[f'return_lag_{lag}'] = returns.shift(lag)

        # 3. Z-스코어 특성
        for window in [10, 20]:
            mean_ret = returns.rolling(window).mean()
            std_ret = returns.rolling(window).std()
            features[f'zscore_{window}'] = (returns - mean_ret) / std_ret

        # 4. 모멘텀 특성
        for window in [10, 20]:
            features[f'momentum_{window}'] = (prices / prices.shift(window) - 1) * 100

        # 5. 변동성 비율
        features['vol_5_20_ratio'] = features['vol_5'] / features['vol_20']

        # 6. 변동성 체제
        vol_median = features['vol_20'].rolling(252).median()
        features['vol_regime'] = (features['vol_20'] > vol_median).astype(int)

        # 7. 타겟: 5일 후 변동성
        target = returns.rolling(5).std().shift(-5) * np.sqrt(252)

        # 결측치 제거
        combined = pd.concat([features, target], axis=1).dropna()
        X = combined[features.columns]
        y = combined.iloc[:, -1]  # 타겟

        self.feature_names = list(X.columns)
        print(f"✅ 특성 생성 완료: {X.shape[0]}개 샘플, {X.shape[1]}개 특성")

        return X, y

    def train_ridge_model(self, X, y, test_size=0.2, alpha=1.0):
        """Ridge 모델 훈련"""
        print("🔄 Ridge 모델 훈련 시작...")

        # 시간적 분할
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # 표준화
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Ridge 모델 훈련
        model = Ridge(alpha=alpha, random_state=42)
        model.fit(X_train_scaled, y_train)

        # 예측 및 성능
        y_test_pred = model.predict(X_test_scaled)
        test_r2 = r2_score(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        # 저장
        self.model = model
        self.scaler = scaler
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        print(f"✅ 모델 훈련 완료")
        print(f"📊 Test R²: {test_r2:.4f}")
        print(f"📊 Test RMSE: {test_rmse:.4f}")

        return {
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }

    def analyze_shap(self, sample_size=500):
        """SHAP 분석"""
        print("🔍 SHAP 분석 시작...")

        if not DEPENDENCIES_OK:
            print("❌ SHAP 분석 불가 - 의존성 부족")
            return {}

        try:
            # 샘플 선택
            X_sample = self.X_train.tail(sample_size) if len(self.X_train) > sample_size else self.X_train
            X_sample_scaled = self.scaler.transform(X_sample)

            # SHAP LinearExplainer
            explainer = shap.LinearExplainer(self.model, X_sample_scaled)
            shap_values = explainer.shap_values(X_sample_scaled)

            # 특성 중요도 계산
            feature_importance = np.abs(shap_values).mean(axis=0)

            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)

            print(f"✅ SHAP 분석 완료 ({len(X_sample)}개 샘플)")
            print("🏆 Top 5 중요 특성:")
            for i, row in importance_df.head().iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")

            return {
                'feature_importance': importance_df.to_dict('records'),
                'expected_value': float(explainer.expected_value),
                'sample_size': len(X_sample)
            }

        except Exception as e:
            print(f"❌ SHAP 분석 실패: {e}")
            return {}

    def analyze_ridge_coefficients(self):
        """Ridge 회귀 계수 분석 (SHAP 대안)"""
        print("📊 Ridge 계수 기반 특성 중요도 분석...")

        if self.model is None:
            return {}

        # Ridge 계수
        coefficients = self.model.coef_

        # 절대값 기준 중요도
        abs_coeffs = np.abs(coefficients)

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': coefficients,
            'abs_importance': abs_coeffs
        }).sort_values('abs_importance', ascending=False)

        print("✅ Ridge 계수 분석 완료")
        print("🏆 Top 5 중요 특성 (Ridge 계수 기준):")
        for i, row in importance_df.head().iterrows():
            print(f"  {row['feature']}: {row['coefficient']:.4f} (|{row['abs_importance']:.4f}|)")

        return importance_df.to_dict('records')

    def analyze_permutation_importance(self, n_repeats=5):
        """Permutation Importance 분석"""
        print("🔄 Permutation Importance 분석 시작...")

        if not DEPENDENCIES_OK:
            print("❌ Permutation 분석 불가")
            return {}

        try:
            X_test_scaled = self.scaler.transform(self.X_test)

            perm_importance = permutation_importance(
                self.model, X_test_scaled, self.y_test,
                n_repeats=n_repeats, random_state=42, scoring='r2'
            )

            importance_results = []
            for i, feature_name in enumerate(self.feature_names):
                importance_results.append({
                    'feature': feature_name,
                    'importance_mean': float(perm_importance.importances_mean[i]),
                    'importance_std': float(perm_importance.importances_std[i])
                })

            importance_results.sort(key=lambda x: x['importance_mean'], reverse=True)

            print("✅ Permutation Importance 분석 완료")
            print("🏆 Top 5 중요 특성 (Permutation 기준):")
            for result in importance_results[:5]:
                print(f"  {result['feature']}: {result['importance_mean']:.4f} ± {result['importance_std']:.4f}")

            return importance_results

        except Exception as e:
            print(f"❌ Permutation 분석 실패: {e}")
            return {}

    def generate_financial_insights(self, shap_results=None, coeff_results=None, perm_results=None):
        """금융 도메인 특화 인사이트 생성"""
        print("💡 금융 인사이트 생성 중...")

        insights = {}

        # 모델 성능 해석
        if self.model is not None:
            test_r2 = r2_score(self.y_test, self.model.predict(self.scaler.transform(self.X_test)))

            if test_r2 > 0.3:
                performance_level = "우수"
                business_value = "VIX 옵션, 동적 헤징에 직접 활용 가능"
            elif test_r2 > 0.15:
                performance_level = "양호"
                business_value = "리스크 관리 보조 도구로 활용"
            else:
                performance_level = "개선 필요"
                business_value = "추가 연구 필요"

            insights['model_performance'] = {
                'r2_score': float(test_r2),
                'level': performance_level,
                'business_value': business_value
            }

        # 특성별 금융 해석
        top_features = []

        if shap_results and 'feature_importance' in shap_results:
            top_features = [f['feature'] for f in shap_results['feature_importance'][:5]]
        elif coeff_results:
            top_features = [f['feature'] for f in coeff_results[:5]]
        elif perm_results:
            top_features = [f['feature'] for f in perm_results[:5]]

        feature_insights = []
        for feature in top_features:
            insight = {
                'feature': feature,
                'category': self._categorize_feature(feature),
                'financial_meaning': self._get_financial_meaning(feature),
                'trading_application': self._get_trading_application(feature)
            }
            feature_insights.append(insight)

        insights['feature_insights'] = feature_insights

        # 리스크 관리 인사이트
        vol_features = [f for f in top_features if 'vol' in f.lower()]
        return_features = [f for f in top_features if 'return' in f.lower()]

        insights['risk_insights'] = {
            'volatility_focus': len(vol_features) / len(top_features) if top_features else 0,
            'return_focus': len(return_features) / len(top_features) if top_features else 0,
            'primary_risk_factor': 'Volatility' if len(vol_features) > len(return_features) else 'Returns'
        }

        print("✅ 금융 인사이트 생성 완료")
        return insights

    def _categorize_feature(self, feature_name):
        """특성 카테고리 분류"""
        if 'vol' in feature_name.lower():
            return 'volatility'
        elif 'return' in feature_name.lower():
            return 'returns'
        elif 'momentum' in feature_name.lower():
            return 'momentum'
        elif 'zscore' in feature_name.lower():
            return 'technical'
        else:
            return 'other'

    def _get_financial_meaning(self, feature_name):
        """특성의 금융적 의미"""
        meanings = {
            'vol_5': '단기 변동성 - 최근 5일 시장 불확실성',
            'vol_10': '중기 변동성 - 최근 10일 시장 불확실성',
            'vol_20': '월간 변동성 - 최근 20일 시장 불확실성',
            'return_lag_1': '전일 수익률 - 단기 모멘텀/역모멘텀 효과',
            'return_lag_2': '2일 전 수익률 - 단기 시계열 패턴',
            'momentum_10': '10일 모멘텀 - 중기 가격 추세',
            'momentum_20': '20일 모멘텀 - 장기 가격 추세',
            'vol_regime': '변동성 체제 - 고/저 변동성 상태 구분'
        }
        return meanings.get(feature_name, f'{feature_name}의 시장 영향')

    def _get_trading_application(self, feature_name):
        """트레이딩 응용"""
        applications = {
            'vol_5': 'VIX 옵션 거래, 단기 헤징',
            'vol_10': '중기 변동성 거래, 옵션 전략',
            'vol_20': '월간 옵션, 포트폴리오 리밸런싱',
            'return_lag_1': '일중 거래, 단기 반전 전략',
            'momentum_10': '추세 추종 전략',
            'momentum_20': '장기 추세 전략',
            'vol_regime': '동적 포지션 조정, 리스크 예산 배분'
        }
        return applications.get(feature_name, '일반적 거래 전략 활용')

    def save_results(self, results, filename='xai_analysis_results.json'):
        """결과 저장"""
        output_dir = Path('/root/workspace/data/xai_analysis')
        output_dir.mkdir(exist_ok=True)

        filepath = output_dir / filename

        # 메타데이터 추가
        results['metadata'] = {
            'analysis_date': datetime.now().isoformat(),
            'model_type': 'Ridge Regression',
            'target': '5-day future volatility',
            'features_count': len(self.feature_names)
        }

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"✅ 결과 저장: {filepath}")
        return str(filepath)

    def run_comprehensive_analysis(self):
        """종합 XAI 분석 실행"""
        print("🚀 종합 XAI 분석 시작...")
        print("=" * 60)

        results = {}

        try:
            # 1. 데이터 준비
            X, y = self.generate_spy_features()
            results['data_info'] = {
                'samples': len(X),
                'features': len(self.feature_names),
                'feature_names': self.feature_names
            }

            # 2. 모델 훈련
            model_results = self.train_ridge_model(X, y)
            results['model_performance'] = model_results

            # 3. SHAP 분석 (가능한 경우)
            shap_results = self.analyze_shap()
            if shap_results:
                results['shap_analysis'] = shap_results

            # 4. Ridge 계수 분석
            coeff_results = self.analyze_ridge_coefficients()
            if coeff_results:
                results['coefficient_analysis'] = coeff_results

            # 5. Permutation Importance
            perm_results = self.analyze_permutation_importance()
            if perm_results:
                results['permutation_analysis'] = perm_results

            # 6. 금융 인사이트
            insights = self.generate_financial_insights(shap_results, coeff_results, perm_results)
            results['financial_insights'] = insights

            # 7. 결과 저장
            results_file = self.save_results(results)
            results['results_file'] = results_file

            # 요약 출력
            print("\n" + "=" * 60)
            print("📊 XAI 분석 완료 요약:")
            print(f"📈 Test R²: {model_results['test_r2']:.4f}")
            print(f"🎯 분석 특성 수: {len(self.feature_names)}")

            if shap_results:
                top_feature = shap_results['feature_importance'][0]['feature']
                print(f"🏆 최고 중요 특성 (SHAP): {top_feature}")
            elif coeff_results:
                print(f"🏆 최고 중요 특성 (Ridge): {coeff_results[0]['feature']}")

            if insights and 'model_performance' in insights:
                print(f"💼 비즈니스 가치: {insights['model_performance']['business_value']}")

            print(f"📁 결과 파일: {results_file}")
            print("\n🎉 종합 XAI 분석 완료!")

            return results

        except Exception as e:
            print(f"❌ XAI 분석 실패: {e}")
            return {'error': str(e)}


def main():
    """메인 실행 함수"""
    print("🚀 Simple XAI Analysis 시작...")

    analyzer = SimpleXAIAnalyzer()
    results = analyzer.run_comprehensive_analysis()

    return analyzer, results


if __name__ == '__main__':
    analyzer, results = main()