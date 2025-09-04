#!/usr/bin/env python3
"""
SPY 모델 빠른 검증 및 수정
- 오버피팅 해결
- 데이터 누수 완전 차단
- 정확도 개선
"""

import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif

class QuickModelFix:
    def __init__(self):
        self.results = {}
        
    def load_clean_data(self):
        """깔끔한 데이터 로드"""
        print("📥 깔끔한 데이터 로드 중...")
        
        try:
            # 2019-2024 데이터로 제한 (너무 오래된 데이터 제외)
            spy_raw = yf.download('SPY', start='2019-01-01', end='2024-12-31', auto_adjust=True, progress=False)
            vix_raw = yf.download('^VIX', start='2019-01-01', end='2024-12-31', auto_adjust=True, progress=False)
            
            # MultiIndex 컬럼 정리
            if isinstance(spy_raw.columns, pd.MultiIndex):
                spy_raw.columns = spy_raw.columns.get_level_values(0)
            if isinstance(vix_raw.columns, pd.MultiIndex):
                vix_raw.columns = vix_raw.columns.get_level_values(0)
                
            print(f"✅ SPY: {len(spy_raw)} 일, VIX: {len(vix_raw)} 일")
            return spy_raw, vix_raw
            
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {str(e)}")
            return None, None
    
    def create_simple_features(self, spy_data, vix_data):
        """간단하고 효과적인 특성만 생성"""
        print("🔧 간단한 특성 생성 중...")
        
        df = pd.DataFrame(index=spy_data.index)
        
        # 기본 수익률 (1일 지연으로 누수 방지)
        returns = spy_data['Close'].pct_change()
        df['returns_lag1'] = returns.shift(1)
        df['returns_lag2'] = returns.shift(2)
        df['returns_lag3'] = returns.shift(3)
        df['returns_lag5'] = returns.shift(5)
        
        # 단순 이동평균 (과거만)
        for period in [10, 20, 50]:
            ma = spy_data['Close'].rolling(period).mean()
            df[f'price_to_ma{period}'] = (spy_data['Close'].shift(1) / ma.shift(1) - 1)
        
        # RSI (간단 버전, 과거만)
        def simple_rsi(prices, period=14):
            delta = prices.diff().shift(1)  # 1일 지연
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
            
        df['rsi'] = simple_rsi(spy_data['Close'])
        df['rsi_normalized'] = (df['rsi'] - 50) / 50  # -1 ~ 1로 정규화
        
        # VIX 특성 (과거만)
        vix_aligned = vix_data.reindex(spy_data.index, method='ffill')
        df['vix'] = vix_aligned['Close'].shift(1)  # 1일 지연
        df['vix_normalized'] = (df['vix'] - 20) / 20  # VIX 20 기준 정규화
        df['vix_change'] = df['vix'].pct_change()
        
        # 변동성 (과거만)
        df['volatility_10'] = returns.rolling(10).std().shift(1)
        df['volatility_20'] = returns.rolling(20).std().shift(1)
        
        # 거래량 (과거만)
        volume_ma = spy_data['Volume'].rolling(20).mean()
        df['volume_ratio'] = (spy_data['Volume'].shift(1) / volume_ma.shift(1))
        
        # 타겟: 다음날 수익률 방향
        df['target'] = (spy_data['Close'].shift(-1) / spy_data['Close'] - 1 > 0).astype(int)
        
        print(f"✅ {len(df.columns)}개 특성 생성 완료")
        return df.dropna()
    
    def strict_time_split(self, df):
        """엄격한 시간 분할"""
        print("📊 엄격한 시간 분할 중...")
        
        # 2019-2021: 훈련
        # 2022: 검증
        # 2023-2024: 테스트
        
        train_mask = df.index < '2022-01-01'
        val_mask = (df.index >= '2022-01-01') & (df.index < '2023-01-01')
        test_mask = df.index >= '2023-01-01'
        
        # 특성과 타겟 분리
        feature_cols = [col for col in df.columns if col != 'target']
        
        X_train = df.loc[train_mask, feature_cols]
        y_train = df.loc[train_mask, 'target']
        
        X_val = df.loc[val_mask, feature_cols]
        y_val = df.loc[val_mask, 'target']
        
        X_test = df.loc[test_mask, feature_cols] 
        y_test = df.loc[test_mask, 'target']
        
        print(f"📊 훈련: {len(X_train)} | 검증: {len(X_val)} | 테스트: {len(X_test)}")
        print(f"📊 훈련 클래스 분포: {dict(y_train.value_counts())}")
        print(f"📊 테스트 클래스 분포: {dict(y_test.value_counts())}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def feature_selection(self, X_train, y_train, X_val, X_test, k=8):
        """최적 특성 선택 (오버피팅 방지)"""
        print(f"🎯 최적 특성 {k}개 선택 중...")
        
        # 통계적 특성 선택
        selector = SelectKBest(score_func=f_classif, k=k)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_val_selected = selector.transform(X_val)
        X_test_selected = selector.transform(X_test)
        
        # 선택된 특성명
        selected_features = X_train.columns[selector.get_support()]
        print(f"✅ 선택된 특성: {list(selected_features)}")
        
        # DataFrame으로 변환
        X_train_selected = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train.index)
        X_val_selected = pd.DataFrame(X_val_selected, columns=selected_features, index=X_val.index)
        X_test_selected = pd.DataFrame(X_test_selected, columns=selected_features, index=X_test.index)
        
        return X_train_selected, X_val_selected, X_test_selected, selector
    
    def train_conservative_models(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """보수적인 모델들 (오버피팅 방지)"""
        print("🎯 보수적 모델 훈련 중...")
        
        models = {
            'conservative_rf': RandomForestClassifier(
                n_estimators=50,      # 적게
                max_depth=6,          # 얕게
                min_samples_split=50, # 크게
                min_samples_leaf=20,  # 크게
                max_features=0.5,     # 적게
                class_weight='balanced',
                random_state=42
            ),
            'simple_lr': LogisticRegression(
                C=0.1,               # 강한 정규화
                class_weight='balanced',
                random_state=42,
                max_iter=1000
            )
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n🔧 {name} 훈련 중...")
            
            # 스케일링 (로지스틱 회귀만)
            if 'lr' in name:
                scaler = RobustScaler()
                X_train_proc = scaler.fit_transform(X_train)
                X_val_proc = scaler.transform(X_val)
                X_test_proc = scaler.transform(X_test)
            else:
                X_train_proc = X_train
                X_val_proc = X_val
                X_test_proc = X_test
                scaler = None
            
            # 훈련
            model.fit(X_train_proc, y_train)
            
            # 예측
            train_pred = model.predict(X_train_proc)
            val_pred = model.predict(X_val_proc)
            test_pred = model.predict(X_test_proc)
            
            # 성능
            train_acc = accuracy_score(y_train, train_pred)
            val_acc = accuracy_score(y_val, val_pred)
            test_acc = accuracy_score(y_test, test_pred)
            
            # AUC
            if hasattr(model, 'predict_proba'):
                test_proba = model.predict_proba(X_test_proc)[:, 1]
                test_auc = roc_auc_score(y_test, test_proba)
            else:
                test_auc = 0.5
            
            # 오버피팅 체크
            overfitting_gap = train_acc - val_acc
            
            results[name] = {
                'model': model,
                'scaler': scaler,
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
                'test_accuracy': test_acc,
                'test_auc': test_auc,
                'overfitting_gap': overfitting_gap,
                'overfitting': overfitting_gap > 0.1
            }
            
            print(f"   훈련: {train_acc:.3f} | 검증: {val_acc:.3f} | 테스트: {test_acc:.3f}")
            print(f"   AUC: {test_auc:.3f} | 오버피팅 갭: {overfitting_gap:.3f}")
            
            if overfitting_gap > 0.1:
                print("   ⚠️ 오버피팅 감지!")
            else:
                print("   ✅ 오버피팅 없음")
        
        return results
    
    def cross_validation_check(self, X_train, y_train, model, model_name):
        """교차 검증으로 안정성 체크"""
        print(f"🔍 {model_name} 교차 검증 중...")
        
        # 시계열 교차 검증
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='accuracy')
        
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        print(f"   CV 평균: {cv_mean:.3f} ± {cv_std:.3f}")
        
        if cv_std > 0.05:
            print("   ⚠️ 불안정한 성능")
        else:
            print("   ✅ 안정적 성능")
        
        return cv_mean, cv_std
    
    def create_quick_report(self, results, selector):
        """빠른 보고서 생성"""
        print("📝 빠른 보고서 생성 중...")
        
        report = {
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'models_tested': len(results),
            'selected_features': list(selector.feature_names_in_[selector.get_support()]),
            'model_performance': {},
            'best_model': None,
            'overfitting_issues': [],
            'recommendations': []
        }
        
        # 성능 정리
        for name, data in results.items():
            report['model_performance'][name] = {
                'test_accuracy': float(data['test_accuracy']),
                'test_auc': float(data['test_auc']),
                'overfitting_gap': float(data['overfitting_gap']),
                'has_overfitting': bool(data['overfitting'])
            }
            
            if data['overfitting']:
                report['overfitting_issues'].append(name)
        
        # 최고 모델
        best_model = max(results.keys(), key=lambda k: results[k]['test_accuracy'])
        best_acc = results[best_model]['test_accuracy']
        
        report['best_model'] = best_model
        report['best_accuracy'] = float(best_acc)
        
        # 권장사항
        recommendations = []
        
        if len(report['overfitting_issues']) == 0:
            recommendations.append("✅ 오버피팅 문제 해결됨")
        else:
            recommendations.append(f"⚠️ 오버피팅 모델: {report['overfitting_issues']}")
        
        if best_acc > 0.55:
            recommendations.append("✅ 합리적 성능 달성")
        else:
            recommendations.append("🎯 성능 추가 개선 필요")
            
        recommendations.append("🔧 특성 선택으로 복잡도 감소")
        recommendations.append("📊 엄격한 시간 분할로 누수 방지")
        
        report['recommendations'] = recommendations
        
        # 저장
        with open('data/raw/quick_model_fix_report.json', 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        return report
    
    def run_quick_fix(self):
        """빠른 수정 프로세스 실행"""
        print("🔧 SPY 모델 빠른 수정 시작!")
        print("=" * 50)
        
        # 1. 데이터 로드
        spy_data, vix_data = self.load_clean_data()
        if spy_data is None:
            return
        
        # 2. 간단한 특성 생성
        df = self.create_simple_features(spy_data, vix_data)
        
        # 3. 엄격한 시간 분할
        X_train, X_val, X_test, y_train, y_val, y_test = self.strict_time_split(df)
        
        # 4. 특성 선택 (오버피팅 방지)
        X_train_sel, X_val_sel, X_test_sel, selector = self.feature_selection(
            X_train, y_train, X_val, X_test, k=8
        )
        
        # 5. 보수적 모델 훈련
        results = self.train_conservative_models(
            X_train_sel, X_val_sel, X_test_sel, y_train, y_val, y_test
        )
        
        # 6. 교차 검증
        for name, data in results.items():
            cv_mean, cv_std = self.cross_validation_check(
                X_train_sel if data['scaler'] is None else data['scaler'].transform(X_train_sel),
                y_train, 
                data['model'], 
                name
            )
            results[name]['cv_mean'] = cv_mean
            results[name]['cv_std'] = cv_std
        
        # 7. 보고서 생성
        report = self.create_quick_report(results, selector)
        
        print("\n" + "=" * 50)
        print("🏆 빠른 수정 결과:")
        print(f"🎯 최고 모델: {report['best_model']}")
        print(f"📊 최고 정확도: {report['best_accuracy']:.1%}")
        print(f"🔧 선택된 특성: {len(report['selected_features'])}개")
        print(f"⚠️ 오버피팅 모델: {len(report['overfitting_issues'])}개")
        
        print("\n📋 핵심 개선사항:")
        for rec in report['recommendations']:
            print(f"   {rec}")
        
        print(f"\n✅ 빠른 수정 완료! 보고서: data/raw/quick_model_fix_report.json")
        
        self.results = results
        return results

def main():
    fixer = QuickModelFix()
    fixer.run_quick_fix()

if __name__ == "__main__":
    main()