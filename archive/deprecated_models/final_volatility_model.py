#!/usr/bin/env python3
"""
최종 변동성 예측 모델 파이프라인
훈련된 최적 모델을 저장하고 새로운 데이터 예측을 위한 완전한 파이프라인
"""

import numpy as np
import pandas as pd
import yfinance as yf
import pickle
import os
import json
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')

# 이전 모듈에서 함수 가져오기
import sys
sys.path.append('/root/workspace/src/features')
from improved_volatility_model import load_enhanced_spy_data, create_comprehensive_features, create_future_volatility_targets, create_interaction_features

class VolatilityPredictor:
    """최종 변동성 예측 모델 클래스"""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_trained = False

    def prepare_features(self, data):
        """데이터에서 특성 생성"""
        print("📊 특성 생성 중...")

        # 포괄적인 특성 생성
        comprehensive_features = create_comprehensive_features(data)

        # 상위 15개 특성 (훈련 시 발견한 것)
        top_features = [
            'vix_level', 'intraday_vol_5', 'intraday_vol_10', 'ewm_vol_10',
            'ewm_vol_5', 'vix_ma_5', 'volatility_10', 'realized_vol_10',
            'realized_vol_5', 'volatility_5', 'ewm_vol_20', 'garman_klass_5',
            'garman_klass_10', 'vol_lag_1', 'vix_std_20'
        ]

        # 존재하는 특성만 선택
        available_features = [f for f in top_features if f in comprehensive_features.columns]
        top_features_df = comprehensive_features[available_features]

        # 상호작용 특성 생성
        interaction_features = create_interaction_features(top_features_df, n_top=8)

        # 최종 특성 결합
        final_features = pd.concat([top_features_df, interaction_features], axis=1)

        print(f"✅ 생성된 특성 수: {len(final_features.columns)}")
        return final_features

    def train_final_model(self, save_path='models/'):
        """최종 모델 훈련 및 저장"""
        print("🤖 최종 모델 훈련 시작")
        print("=" * 50)

        # 1. 데이터 로드
        spy_data = load_enhanced_spy_data()

        # 2. 특성 및 타겟 생성
        features = self.prepare_features(spy_data)
        targets = create_future_volatility_targets(spy_data)

        # 3. 데이터 결합 및 정리
        combined_data = pd.concat([features, targets[['target_vol_5d']]], axis=1).dropna()
        X = combined_data[features.columns]
        y = combined_data['target_vol_5d']

        print(f"📊 훈련 데이터: {len(X)} 샘플, {len(X.columns)} 특성")

        # 4. 시간 순서 분할
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # 5. 스케일링
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 6. 최적 모델 훈련 (이전에 발견한 최적 파라미터)
        self.model = Lasso(alpha=0.0005, max_iter=1000, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        self.feature_names = X.columns.tolist()
        self.is_trained = True

        # 7. 성능 평가
        y_pred = self.model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        print(f"📈 최종 모델 성능:")
        print(f"   R² Score: {r2:.4f}")
        print(f"   MAE:      {mae:.6f}")
        print(f"   RMSE:     {np.sqrt(mse):.6f}")

        # 8. 특성 중요도
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': np.abs(self.model.coef_)
        }).sort_values('importance', ascending=False)

        print(f"\n🎯 상위 10개 중요 특성:")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            print(f"  {i+1:2d}. {row['feature']:25}: {row['importance']:.6f}")

        # 9. 모델 및 메타데이터 저장
        os.makedirs(save_path, exist_ok=True)

        # 모델 저장
        with open(f"{save_path}/volatility_model.pkl", "wb") as f:
            pickle.dump(self.model, f)

        # 스케일러 저장
        with open(f"{save_path}/scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        # 메타데이터 저장
        metadata = {
            'model_type': 'Lasso',
            'alpha': 0.0005,
            'max_iter': 1000,
            'feature_names': self.feature_names,
            'feature_count': len(self.feature_names),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'performance': {
                'r2_score': float(r2),
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(np.sqrt(mse))
            },
            'feature_importance': feature_importance.to_dict('records'),
            'trained_date': datetime.now().isoformat(),
            'data_period': '2015-01-01 to 2024-12-31'
        }

        with open(f"{save_path}/model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n💾 모델 저장 완료: {save_path}")
        return metadata

    def load_model(self, model_path='models/'):
        """저장된 모델 로드"""
        print(f"📂 모델 로드 중: {model_path}")

        try:
            # 모델 로드
            with open(f"{model_path}/volatility_model.pkl", "rb") as f:
                self.model = pickle.load(f)

            # 스케일러 로드
            with open(f"{model_path}/scaler.pkl", "rb") as f:
                self.scaler = pickle.load(f)

            # 메타데이터 로드
            with open(f"{model_path}/model_metadata.json", "r") as f:
                metadata = json.load(f)
                self.feature_names = metadata['feature_names']

            self.is_trained = True
            print("✅ 모델 로드 완료")
            return metadata

        except FileNotFoundError as e:
            print(f"❌ 모델 파일을 찾을 수 없습니다: {e}")
            return None

    def predict_volatility(self, data, days_ahead=5):
        """새로운 데이터에 대한 변동성 예측"""
        if not self.is_trained:
            raise ValueError("모델이 훈련되지 않았습니다. train_final_model() 또는 load_model()을 먼저 실행하세요.")

        print(f"🔮 {days_ahead}일 후 변동성 예측 중...")

        # 특성 생성
        features = self.prepare_features(data)

        # 특성 순서 맞추기
        features_ordered = features.reindex(columns=self.feature_names, fill_value=0)

        # 유효한 데이터만 선택 (NaN 제거)
        valid_data = features_ordered.dropna()

        if len(valid_data) == 0:
            print("⚠️ 유효한 데이터가 없습니다.")
            return None

        # 스케일링
        features_scaled = self.scaler.transform(valid_data)

        # 예측
        predictions = self.model.predict(features_scaled)

        # 결과 DataFrame 생성
        results = pd.DataFrame({
            'date': valid_data.index,
            'predicted_volatility': predictions
        })

        print(f"✅ {len(results)}개 예측 완료")
        return results

    def get_latest_prediction(self, symbol='SPY'):
        """최신 데이터로 변동성 예측"""
        print(f"📊 {symbol} 최신 데이터 예측 중...")

        # 최근 6개월 데이터 다운로드 (특성 생성을 위해 충분한 기간)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)

        try:
            data = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'),
                             end=end_date.strftime('%Y-%m-%d'), progress=False)
            data['returns'] = data['Close'].pct_change()

            # VIX 데이터 추가
            vix = yf.download('^VIX', start=start_date.strftime('%Y-%m-%d'),
                            end=end_date.strftime('%Y-%m-%d'), progress=False)
            vix_close = vix['Close'].reindex(data.index, method='ffill')
            data['vix'] = vix_close

            data = data.dropna()

            # 예측
            predictions = self.predict_volatility(data)

            if predictions is not None and len(predictions) > 0:
                latest_prediction = predictions.iloc[-1]
                print(f"🎯 최신 예측 (5일 후 변동성): {latest_prediction['predicted_volatility']:.6f}")
                print(f"📅 예측 날짜: {latest_prediction['date'].strftime('%Y-%m-%d')}")
                return latest_prediction
            else:
                print("❌ 예측 실패")
                return None

        except Exception as e:
            print(f"❌ 데이터 로드 오류: {e}")
            return None

def main():
    """메인 실행 함수"""
    print("🚀 최종 변동성 예측 모델 파이프라인")
    print("=" * 60)

    # 1. 모델 인스턴스 생성
    predictor = VolatilityPredictor()

    # 2. 모델 훈련 및 저장
    metadata = predictor.train_final_model()

    # 3. 모델 테스트 (재로드)
    print(f"\n🧪 모델 재로드 테스트")
    test_predictor = VolatilityPredictor()
    loaded_metadata = test_predictor.load_model()

    if loaded_metadata:
        print(f"✅ 재로드 성공: {loaded_metadata['model_type']}")

        # 4. 최신 예측 테스트
        print(f"\n🔮 최신 SPY 변동성 예측 테스트")
        latest_pred = test_predictor.get_latest_prediction('SPY')

        if latest_pred is not None:
            print(f"✅ 예측 파이프라인 정상 작동")
        else:
            print(f"⚠️ 예측 파이프라인 오류")

    print("=" * 60)
    print("✅ 최종 모델 파이프라인 구축 완료")
    print("📂 저장된 파일:")
    print("   - models/volatility_model.pkl")
    print("   - models/scaler.pkl")
    print("   - models/model_metadata.json")

    return predictor

if __name__ == "__main__":
    predictor = main()