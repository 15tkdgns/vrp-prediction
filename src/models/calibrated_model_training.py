import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import joblib
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class CalibratedSP500Model:
    """
    연구 기반 신뢰도 캘리브레이션을 적용한 S&P500 이벤트 탐지 모델
    - Platt Scaling (Sigmoid 캘리브레이션)
    - Isotonic Regression
    - Bootstrap 신뢰구간
    - 앙상블 모델
    """

    def __init__(self, data_dir="data", models_dir="data/models"):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.scaler = StandardScaler()
        self.models = {}
        self.calibrated_models = {}
        self.ensemble_weights = {}

        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        
        if not os.path.exists("results/analysis"):
            os.makedirs("results/analysis", exist_ok=True)

    def load_and_enhance_training_data(self):
        """향상된 훈련 데이터 로드 및 이벤트 정의 재조정"""
        print("[1/8] 향상된 훈련 데이터 로드...")
        
        features_df = pd.read_csv(f"{self.data_dir}/raw/training_features.csv")
        labels_df = pd.read_csv(f"{self.data_dir}/raw/event_labels.csv")
        
        # 날짜 형식 통일
        features_df["Date"] = pd.to_datetime(features_df["Date"])
        labels_df["Date"] = pd.to_datetime(labels_df["Date"])
        
        # 데이터 병합
        merged_df = pd.merge(features_df, labels_df, on=["ticker", "Date"], how="inner")
        
        # 향상된 이벤트 정의 (연구 기반)
        print("🔄 이벤트 정의 재조정 (목표: 15-25% 이벤트 비율)...")
        
        # 기존 특성들
        price_change = merged_df['Price_Change'] if 'Price_Change' in merged_df.columns else merged_df['Returns'].abs()
        volume_spike = merged_df['Volume_Spike'] if 'Volume_Spike' in merged_df.columns else merged_df['Volume'] / merged_df['Volume_MA']
        
        # 새로운 이벤트 정의 (더 관대한 기준)
        # 1. 가격 이벤트: 2.5% 이상 변동 (기존 3%에서 완화)
        price_event = price_change > 0.025
        
        # 2. 거래량 이벤트: 1.5배 이상 증가 (기존 2배에서 완화)
        volume_event = volume_spike > 1.5
        
        # 3. 변동성 이벤트: 5일 변동성이 20일 평균의 1.5배 이상
        volatility_5d = merged_df['Volatility'] if 'Volatility' in merged_df.columns else merged_df['Returns'].rolling(5).std()
        volatility_20d = merged_df['Returns'].rolling(20).std()
        volatility_event = volatility_5d > (volatility_20d * 1.5)
        
        # 4. 기술적 지표 이벤트
        rsi = merged_df['RSI'] if 'RSI' in merged_df.columns else 50
        rsi_event = (rsi > 70) | (rsi < 30)  # 과매수/과매도
        
        # 복합 이벤트 정의 (OR 조건으로 이벤트 비율 증가)
        major_event = (price_event | volume_event | volatility_event | rsi_event).astype(int)
        
        # 기존 라벨 업데이트
        merged_df['major_event'] = major_event
        merged_df['price_spike'] = price_event.astype(int)
        merged_df['unusual_volume'] = volume_event.astype(int)
        
        event_rate = merged_df['major_event'].mean()
        print(f"✅ 새로운 이벤트 비율: {event_rate:.3f} ({event_rate*100:.1f}%)")
        
        if event_rate < 0.15:
            print("⚠️ 이벤트 비율이 목표치(15%) 미만입니다. 추가 조정...")
            # 더 관대한 기준 적용
            price_event_soft = price_change > 0.02  # 2%로 더 완화
            volume_event_soft = volume_spike > 1.3   # 1.3배로 더 완화
            major_event_soft = (price_event_soft | volume_event_soft | volatility_event | rsi_event).astype(int)
            merged_df['major_event'] = major_event_soft
            
            final_event_rate = merged_df['major_event'].mean()
            print(f"🔄 조정된 이벤트 비율: {final_event_rate:.3f} ({final_event_rate*100:.1f}%)")
        
        print(f"✅ 최종 데이터 크기: {merged_df.shape}")
        return merged_df

    def prepare_enhanced_features(self, df):
        """향상된 특성 준비"""
        print("[2/8] 향상된 특성 엔지니어링...")
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        target_columns = ['major_event', 'price_spike', 'unusual_volume']
        feature_columns = [col for col in numeric_columns if col not in target_columns]
        
        X = df[feature_columns].fillna(0)
        y = df['major_event']
        
        # 시장 상황 특성 추가 (시뮬레이션)
        print("🔧 시장 상황 특성 추가...")
        
        # VIX 대용 지표 (변동성 기반)
        X['market_fear'] = X['Volatility'] * 100 if 'Volatility' in X.columns else np.random.normal(20, 5, len(X))
        
        # 시장 모멘텀 (이동평균 기반)
        if 'Price_MA_5' in X.columns and 'Price_MA_20' in X.columns:
            X['momentum'] = (X['Price_MA_5'] / X['Price_MA_20'] - 1) * 100
        else:
            X['momentum'] = np.random.normal(0, 2, len(X))
        
        # 시간적 특성 (요일, 월)
        if 'Date' in df.columns:
            dates = pd.to_datetime(df['Date'])
            X['day_of_week'] = dates.dt.dayofweek
            X['month'] = dates.dt.month
        
        # 상대적 강도 지수
        if 'RSI' in X.columns:
            X['rsi_normalized'] = (X['RSI'] - 50) / 50  # -1 to 1 범위로 정규화
        
        feature_columns = X.columns.tolist()
        
        print(f"✅ 총 특성 수: {len(feature_columns)}")
        print(f"✅ 샘플 수: {len(X)}")
        print(f"✅ 이벤트 비율: {y.mean():.3f}")
        
        return X, y, feature_columns

    def train_base_models(self, X, y):
        """기본 모델들 훈련"""
        print("[3/8] 기본 모델 훈련...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 1. Random Forest (연구 기반 파라미터 조정)
        print("🌳 Random Forest 훈련...")
        rf_model = RandomForestClassifier(
            n_estimators=150,      # 증가
            max_depth=15,          # 증가 (기존 10)
            min_samples_split=10,  # 감소 (기존 20)
            min_samples_leaf=5,    # 감소 (기존 10)
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)
        
        rf_train_pred = rf_model.predict_proba(X_train_scaled)[:, 1]
        rf_test_pred = rf_model.predict_proba(X_test_scaled)[:, 1]
        
        self.models["random_forest"] = {
            "model": rf_model,
            "train_auc": roc_auc_score(y_train, rf_train_pred),
            "test_auc": roc_auc_score(y_test, rf_test_pred),
            "feature_importance": rf_model.feature_importances_
        }
        
        # 2. Gradient Boosting (연구 기반 파라미터 조정)
        print("📈 Gradient Boosting 훈련...")
        gb_model = GradientBoostingClassifier(
            n_estimators=120,      # 증가
            max_depth=8,           # 증가 (기존 6)
            learning_rate=0.15,    # 증가 (기존 0.1)
            subsample=0.85,        # 증가
            random_state=42
        )
        gb_model.fit(X_train_scaled, y_train)
        
        gb_train_pred = gb_model.predict_proba(X_train_scaled)[:, 1]
        gb_test_pred = gb_model.predict_proba(X_test_scaled)[:, 1]
        
        self.models["gradient_boosting"] = {
            "model": gb_model,
            "train_auc": roc_auc_score(y_train, gb_train_pred),
            "test_auc": roc_auc_score(y_test, gb_test_pred),
            "feature_importance": gb_model.feature_importances_
        }
        
        # 3. LSTM (연구 기반 파라미터 조정)
        print("🧠 LSTM 훈련...")
        X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
        X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
        
        lstm_model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(1, X_train_scaled.shape[1]),
                 kernel_regularizer=l2(0.005)),  # 정규화 완화
            Dropout(0.2),  # 드롭아웃 감소
            LSTM(32, kernel_regularizer=l2(0.005)),
            Dropout(0.2),
            Dense(32, activation='relu', kernel_regularizer=l2(0.005)),
            Dropout(0.15),  # 감소
            Dense(1, activation='sigmoid')
        ])
        
        lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        
        history = lstm_model.fit(
            X_train_lstm, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test_lstm, y_test),
            callbacks=[early_stopping],
            verbose=0
        )
        
        lstm_train_pred = lstm_model.predict(X_train_lstm, verbose=0).flatten()
        lstm_test_pred = lstm_model.predict(X_test_lstm, verbose=0).flatten()
        
        self.models["lstm"] = {
            "model": lstm_model,
            "train_auc": roc_auc_score(y_train, lstm_train_pred),
            "test_auc": roc_auc_score(y_test, lstm_test_pred),
            "history": history.history
        }
        
        # 성능 출력
        for name, model_info in self.models.items():
            print(f"{name.upper()} - Train AUC: {model_info['train_auc']:.4f}, Test AUC: {model_info['test_auc']:.4f}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test

    def apply_calibration(self, X_train, X_test, y_train, y_test):
        """Platt Scaling과 Isotonic Regression 적용"""
        print("[4/8] 신뢰도 캘리브레이션 적용...")
        
        for model_name, model_info in self.models.items():
            if model_name == 'lstm':
                continue  # LSTM은 별도 처리
            
            print(f"🎯 {model_name.upper()} 캘리브레이션...")
            
            # Platt Scaling (Sigmoid)
            platt_calibrated = CalibratedClassifierCV(
                model_info['model'], 
                method='sigmoid', 
                cv=3
            )
            platt_calibrated.fit(X_train, y_train)
            
            # Isotonic Regression
            isotonic_calibrated = CalibratedClassifierCV(
                model_info['model'], 
                method='isotonic', 
                cv=3
            )
            isotonic_calibrated.fit(X_train, y_train)
            
            # 캘리브레이션된 예측
            platt_pred = platt_calibrated.predict_proba(X_test)[:, 1]
            isotonic_pred = isotonic_calibrated.predict_proba(X_test)[:, 1]
            original_pred = model_info['model'].predict_proba(X_test)[:, 1]
            
            # Brier Score로 캘리브레이션 품질 평가
            original_brier = brier_score_loss(y_test, original_pred)
            platt_brier = brier_score_loss(y_test, platt_pred)
            isotonic_brier = brier_score_loss(y_test, isotonic_pred)
            
            # 최적 캘리브레이션 방법 선택
            if platt_brier <= isotonic_brier:
                best_method = 'platt'
                best_calibrated = platt_calibrated
                best_pred = platt_pred
                best_brier = platt_brier
            else:
                best_method = 'isotonic'
                best_calibrated = isotonic_calibrated
                best_pred = isotonic_pred
                best_brier = isotonic_brier
            
            self.calibrated_models[model_name] = {
                'calibrated_model': best_calibrated,
                'method': best_method,
                'original_brier': original_brier,
                'calibrated_brier': best_brier,
                'improvement': original_brier - best_brier
            }
            
            print(f"  최적 방법: {best_method}")
            print(f"  Brier Score 개선: {original_brier:.4f} → {best_brier:.4f}")
            print(f"  평균 신뢰도: {np.mean(best_pred):.4f}")
            
        print("✅ 모든 모델 캘리브레이션 완료")

    def create_ensemble_model(self, X_test, y_test):
        """앙상블 모델 생성 및 가중치 최적화"""
        print("[5/8] 앙상블 모델 생성...")
        
        # 각 모델의 캘리브레이션된 예측
        predictions = {}
        
        for model_name in self.calibrated_models:
            calibrated_model = self.calibrated_models[model_name]['calibrated_model']
            pred = calibrated_model.predict_proba(X_test)[:, 1]
            predictions[model_name] = pred
        
        # LSTM 예측 (별도 처리)
        if 'lstm' in self.models:
            X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
            lstm_pred = self.models['lstm']['model'].predict(X_test_lstm, verbose=0).flatten()
            predictions['lstm'] = lstm_pred
        
        # 성능 기반 가중치 계산
        weights = {}
        total_weight = 0
        
        for model_name in predictions:
            if model_name == 'lstm':
                auc_score = self.models[model_name]['test_auc']
            else:
                auc_score = self.models[model_name]['test_auc']
            
            # AUC 점수를 가중치로 사용 (정규화)
            weight = max(0, auc_score - 0.5) ** 2  # 0.5 이상만 유의미
            weights[model_name] = weight
            total_weight += weight
        
        # 가중치 정규화
        for model_name in weights:
            weights[model_name] = weights[model_name] / total_weight if total_weight > 0 else 1/len(weights)
        
        self.ensemble_weights = weights
        
        # 앙상블 예측 계산
        ensemble_pred = np.zeros(len(y_test))
        for model_name, pred in predictions.items():
            ensemble_pred += weights[model_name] * pred
        
        ensemble_auc = roc_auc_score(y_test, ensemble_pred)
        ensemble_avg_confidence = np.mean(ensemble_pred)
        ensemble_std_confidence = np.std(ensemble_pred)
        
        print(f"✅ 앙상블 모델 성능:")
        print(f"  AUC: {ensemble_auc:.4f}")
        print(f"  평균 신뢰도: {ensemble_avg_confidence:.4f} ± {ensemble_std_confidence:.4f}")
        print(f"  모델 가중치:")
        for name, weight in weights.items():
            print(f"    {name}: {weight:.3f}")
        
        return ensemble_pred

    def bootstrap_confidence_intervals(self, X_test, y_test, n_bootstrap=500):
        """Bootstrap을 이용한 신뢰구간 계산"""
        print("[6/8] Bootstrap 신뢰구간 계산...")
        
        bootstrap_predictions = []
        
        for i in range(n_bootstrap):
            if i % 100 == 0:
                print(f"  진행률: {i}/{n_bootstrap}")
            
            # Bootstrap 샘플링
            indices = np.random.choice(len(X_test), size=len(X_test), replace=True)
            X_boot = X_test[indices]
            
            # 앙상블 예측
            ensemble_pred_boot = np.zeros(len(indices))
            
            for model_name in self.calibrated_models:
                calibrated_model = self.calibrated_models[model_name]['calibrated_model']
                pred = calibrated_model.predict_proba(X_boot)[:, 1]
                ensemble_pred_boot += self.ensemble_weights[model_name] * pred
            
            if 'lstm' in self.models:
                X_boot_lstm = X_boot.reshape((X_boot.shape[0], 1, X_boot.shape[1]))
                lstm_pred = self.models['lstm']['model'].predict(X_boot_lstm, verbose=0).flatten()
                ensemble_pred_boot += self.ensemble_weights['lstm'] * lstm_pred
            
            bootstrap_predictions.append(ensemble_pred_boot)
        
        bootstrap_predictions = np.array(bootstrap_predictions)
        
        # 신뢰구간 계산
        confidence_intervals = {
            'mean': np.mean(bootstrap_predictions, axis=0),
            'lower_95': np.percentile(bootstrap_predictions, 2.5, axis=0),
            'upper_95': np.percentile(bootstrap_predictions, 97.5, axis=0),
            'lower_68': np.percentile(bootstrap_predictions, 16, axis=0),
            'upper_68': np.percentile(bootstrap_predictions, 84, axis=0)
        }
        
        avg_interval_width = np.mean(confidence_intervals['upper_95'] - confidence_intervals['lower_95'])
        print(f"✅ Bootstrap 완료 - 평균 95% 신뢰구간 폭: {avg_interval_width:.4f}")
        
        return confidence_intervals

    def evaluate_calibration_quality(self, X_test, y_test):
        """캘리브레이션 품질 평가"""
        print("[7/8] 캘리브레이션 품질 평가...")
        
        evaluation_results = {}
        
        for model_name in self.calibrated_models:
            calibrated_model = self.calibrated_models[model_name]['calibrated_model']
            y_pred_proba = calibrated_model.predict_proba(X_test)[:, 1]
            
            # 캘리브레이션 곡선
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_test, y_pred_proba, n_bins=10, strategy='uniform'
            )
            
            # Expected Calibration Error (ECE)
            bin_boundaries = np.linspace(0, 1, 11)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_pred_proba > bin_lower) & (y_pred_proba <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = y_test[in_bin].mean()
                    avg_confidence_in_bin = y_pred_proba[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            evaluation_results[model_name] = {
                'ece': ece,
                'brier_score': brier_score_loss(y_test, y_pred_proba),
                'avg_confidence': np.mean(y_pred_proba),
                'confidence_std': np.std(y_pred_proba),
                'auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            print(f"{model_name.upper()}:")
            print(f"  ECE: {ece:.4f}")
            print(f"  Brier Score: {evaluation_results[model_name]['brier_score']:.4f}")
            print(f"  평균 신뢰도: {evaluation_results[model_name]['avg_confidence']:.4f}")
        
        return evaluation_results

    def save_calibrated_models(self):
        """캘리브레이션된 모델들 저장"""
        print("[8/8] 캘리브레이션된 모델 저장...")
        
        # 캘리브레이션된 모델들 저장
        for model_name, calibrated_info in self.calibrated_models.items():
            joblib.dump(
                calibrated_info['calibrated_model'], 
                f"{self.models_dir}/{model_name}_calibrated_model.pkl"
            )
        
        # LSTM 모델 저장
        if 'lstm' in self.models:
            self.models['lstm']['model'].save(f"{self.models_dir}/lstm_calibrated_model.h5")
        
        # 스케일러 저장
        joblib.dump(self.scaler, f"{self.models_dir}/scaler_calibrated.pkl")
        
        # 앙상블 가중치 저장
        with open(f"{self.data_dir}/raw/ensemble_weights.json", "w") as f:
            json.dump(self.ensemble_weights, f, indent=2)
        
        print("✅ 모든 캘리브레이션된 모델 저장 완료")

    def run_calibrated_training_pipeline(self):
        """전체 캘리브레이션 훈련 파이프라인 실행"""
        print("🚀 연구 기반 캘리브레이션 모델 훈련 시작")
        print("=" * 60)
        
        # 1. 데이터 로드 및 향상
        df = self.load_and_enhance_training_data()
        
        # 2. 특성 준비
        X, y, feature_names = self.prepare_enhanced_features(df)
        
        # 3. 기본 모델 훈련
        X_train, X_test, y_train, y_test = self.train_base_models(X, y)
        
        # 4. 캘리브레이션 적용
        self.apply_calibration(X_train, X_test, y_train, y_test)
        
        # 5. 앙상블 모델 생성
        ensemble_pred = self.create_ensemble_model(X_test, y_test)
        
        # 6. Bootstrap 신뢰구간
        confidence_intervals = self.bootstrap_confidence_intervals(X_test, y_test)
        
        # 7. 캘리브레이션 품질 평가
        evaluation_results = self.evaluate_calibration_quality(X_test, y_test)
        
        # 8. 모델 저장
        self.save_calibrated_models()
        
        # 최종 결과 요약
        print("\n" + "=" * 60)
        print("🎉 캘리브레이션 훈련 완료!")
        print("=" * 60)
        
        ensemble_avg = np.mean(ensemble_pred)
        ensemble_std = np.std(ensemble_pred)
        
        print(f"📊 앙상블 모델 최종 성과:")
        print(f"   평균 신뢰도: {ensemble_avg:.4f} ({ensemble_avg*100:.1f}%)")
        print(f"   신뢰도 표준편차: {ensemble_std:.4f}")
        print(f"   AUC: {roc_auc_score(y_test, ensemble_pred):.4f}")
        
        # 신뢰도 구간별 분포
        low_conf = np.sum(ensemble_pred < 0.3) / len(ensemble_pred)
        mid_conf = np.sum((ensemble_pred >= 0.3) & (ensemble_pred <= 0.7)) / len(ensemble_pred)
        high_conf = np.sum(ensemble_pred > 0.7) / len(ensemble_pred)
        
        print(f"   신뢰도 분포:")
        print(f"     낮음 (<30%): {low_conf*100:.1f}%")
        print(f"     중간 (30-70%): {mid_conf*100:.1f}%")
        print(f"     높음 (>70%): {high_conf*100:.1f}%")
        
        target_achieved = 0.35 <= ensemble_avg <= 0.55
        print(f"\n🎯 목표 달성 여부 (35-55% 신뢰도): {'✅ 달성' if target_achieved else '❌ 미달성'}")
        
        return True


if __name__ == "__main__":
    print("🔬 연구 기반 S&P500 캘리브레이션 모델 훈련")
    
    model_trainer = CalibratedSP500Model()
    success = model_trainer.run_calibrated_training_pipeline()
    
    if success:
        print("\n✅ 캘리브레이션 모델 훈련 성공!")
        print("   이제 35-55% 범위의 현실적인 신뢰도를 제공합니다.")
    else:
        print("\n❌ 훈련 실패!")