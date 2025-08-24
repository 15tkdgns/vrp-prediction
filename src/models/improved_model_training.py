import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class ImprovedSP500EventDetectionModel:
    """
    개선된 S&P500 주식 데이터 기반 이벤트 탐지 모델
    - 과적합 방지
    - 교차 검증
    - 정규화 적용
    - 현실적인 신뢰도 점수
    """

    def __init__(self, data_dir="data", models_dir="data/models"):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.scaler = StandardScaler()
        self.models = {}

        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        
        # 결과 저장 디렉토리
        if not os.path.exists("results/analysis"):
            os.makedirs("results/analysis", exist_ok=True)

    def load_training_data(self):
        """훈련 데이터 로드 및 전처리"""
        print("[1/6] 훈련 데이터 로드...")
        
        # 기본 특성 및 라벨 데이터
        try:
            features_df = pd.read_csv(f"{self.data_dir}/raw/training_features.csv")
            labels_df = pd.read_csv(f"{self.data_dir}/raw/event_labels.csv")
        except FileNotFoundError as e:
            print(f"❌ 필수 데이터 파일을 찾을 수 없습니다: {e}")
            return None

        # LLM 특성 데이터 (선택적)
        try:
            llm_features_df = pd.read_csv(f"{self.data_dir}/processed/llm_enhanced_features.csv")
            # date 컬럼이 없을 경우 처리
            if 'date' not in llm_features_df.columns:
                print("⚠️ LLM 특성에 date 컬럼이 없습니다. LLM 특성 제외하고 진행합니다.")
                llm_features_df = pd.DataFrame()
            else:
                llm_features_df = llm_features_df.dropna(subset=["date"])
                llm_features_df["date"] = pd.to_datetime(llm_features_df["date"])
        except FileNotFoundError:
            print("⚠️ LLM 강화 특성 파일을 찾을 수 없습니다. 기본 특성만 사용합니다.")
            llm_features_df = pd.DataFrame()

        # 날짜 형식 통일
        if 'Date' in features_df.columns:
            features_df["Date"] = pd.to_datetime(features_df["Date"])
        if 'Date' in labels_df.columns:
            labels_df["Date"] = pd.to_datetime(labels_df["Date"])

        # ticker와 날짜 기준으로 병합
        if 'ticker' in features_df.columns and 'ticker' in labels_df.columns:
            merged_df = pd.merge(
                features_df, labels_df,
                on=["ticker", "Date"] if 'Date' in features_df.columns else ["ticker"],
                how="inner"
            )
        else:
            # ticker 컬럼이 없는 경우 날짜만으로 병합
            if 'Date' in features_df.columns and 'Date' in labels_df.columns:
                merged_df = pd.merge(features_df, labels_df, on="Date", how="inner")
            else:
                merged_df = pd.concat([features_df, labels_df], axis=1)

        print(f"✅ 병합된 데이터 크기: {merged_df.shape}")
        return merged_df

    def prepare_features(self, df):
        """특성 준비 및 전처리"""
        print("[2/6] 특성 전처리...")
        
        # 필요한 컬럼만 선택 (숫자형 컬럼)
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 타겟 변수들 제외
        target_columns = ['major_event', 'price_spike', 'unusual_volume']
        feature_columns = [col for col in numeric_columns if col not in target_columns]
        
        if not feature_columns:
            print("❌ 사용 가능한 숫자형 특성이 없습니다.")
            return None, None, None

        # 특성과 타겟 분리
        X = df[feature_columns].fillna(0)  # NaN 값을 0으로 처리
        
        # major_event가 있으면 사용, 없으면 임시 타겟 생성
        if 'major_event' in df.columns:
            y = df['major_event']
        else:
            # 임시로 타겟 생성 (Close 가격 변화율 기준)
            if 'Close' in df.columns:
                price_change = df['Close'].pct_change().fillna(0)
                y = (price_change.abs() > 0.02).astype(int)  # 2% 이상 변동을 이벤트로 정의
                print("⚠️ major_event 컬럼이 없어 가격 변동률로 타겟을 생성했습니다.")
            else:
                # 마지막 수단: 랜덤 타겟
                y = np.random.choice([0, 1], size=len(df), p=[0.8, 0.2])
                print("⚠️ 타겟 변수를 찾을 수 없어 임시 타겟을 생성했습니다.")

        print(f"✅ 특성 수: {len(feature_columns)}")
        print(f"✅ 샘플 수: {len(X)}")
        print(f"✅ 이벤트 비율: {y.mean():.3f}")
        
        return X, y, feature_columns

    def train_improved_models(self, X, y):
        """개선된 모델들 훈련"""
        print("[3/6] 개선된 모델 훈련...")
        
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # 특성 스케일링
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 1. 개선된 Random Forest (과적합 방지)
        print("🌳 Random Forest 훈련...")
        rf_model = RandomForestClassifier(
            n_estimators=100,  # 트리 수 감소
            max_depth=10,      # 깊이 제한
            min_samples_split=20,  # 분할 최소 샘플 증가
            min_samples_leaf=10,   # 리프 최소 샘플 증가
            max_features='sqrt',   # 특성 수 제한
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)
        
        # 교차 검증
        cv_scores_rf = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        
        # 예측 및 평가
        rf_train_pred = rf_model.predict_proba(X_train_scaled)[:, 1]
        rf_test_pred = rf_model.predict_proba(X_test_scaled)[:, 1]
        
        self.models["random_forest"] = {
            "model": rf_model,
            "train_auc": roc_auc_score(y_train, rf_train_pred),
            "test_auc": roc_auc_score(y_test, rf_test_pred),
            "cv_auc_mean": cv_scores_rf.mean(),
            "cv_auc_std": cv_scores_rf.std(),
            "feature_importance": rf_model.feature_importances_
        }
        
        print(f"RF - Train AUC: {self.models['random_forest']['train_auc']:.4f}")
        print(f"RF - Test AUC: {self.models['random_forest']['test_auc']:.4f}")
        print(f"RF - CV AUC: {cv_scores_rf.mean():.4f} ± {cv_scores_rf.std():.4f}")

        # 2. 개선된 Gradient Boosting
        print("📈 Gradient Boosting 훈련...")
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,       # 깊이 제한
            learning_rate=0.1, # 학습률 감소
            subsample=0.8,     # 서브샘플링
            random_state=42
        )
        gb_model.fit(X_train_scaled, y_train)
        
        # 교차 검증
        cv_scores_gb = cross_val_score(gb_model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
        
        # 예측 및 평가
        gb_train_pred = gb_model.predict_proba(X_train_scaled)[:, 1]
        gb_test_pred = gb_model.predict_proba(X_test_scaled)[:, 1]
        
        self.models["gradient_boosting"] = {
            "model": gb_model,
            "train_auc": roc_auc_score(y_train, gb_train_pred),
            "test_auc": roc_auc_score(y_test, gb_test_pred),
            "cv_auc_mean": cv_scores_gb.mean(),
            "cv_auc_std": cv_scores_gb.std(),
            "feature_importance": gb_model.feature_importances_
        }
        
        print(f"GB - Train AUC: {self.models['gradient_boosting']['train_auc']:.4f}")
        print(f"GB - Test AUC: {self.models['gradient_boosting']['test_auc']:.4f}")
        print(f"GB - CV AUC: {cv_scores_gb.mean():.4f} ± {cv_scores_gb.std():.4f}")

        # 3. 개선된 LSTM (정규화 적용)
        print("🧠 LSTM 훈련...")
        
        # LSTM용 데이터 준비 (시계열 형태로 reshape)
        X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
        X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
        
        lstm_model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(1, X_train_scaled.shape[1]),
                 kernel_regularizer=l2(0.01)),  # L2 정규화
            Dropout(0.3),  # 드롭아웃
            LSTM(25, kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            Dense(25, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        lstm_model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Early Stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # 훈련
        history = lstm_model.fit(
            X_train_lstm, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test_lstm, y_test),
            callbacks=[early_stopping],
            verbose=0
        )
        
        # 예측 및 평가
        lstm_train_pred = lstm_model.predict(X_train_lstm, verbose=0).flatten()
        lstm_test_pred = lstm_model.predict(X_test_lstm, verbose=0).flatten()
        
        self.models["lstm"] = {
            "model": lstm_model,
            "train_auc": roc_auc_score(y_train, lstm_train_pred),
            "test_auc": roc_auc_score(y_test, lstm_test_pred),
            "history": history.history
        }
        
        print(f"LSTM - Train AUC: {self.models['lstm']['train_auc']:.4f}")
        print(f"LSTM - Test AUC: {self.models['lstm']['test_auc']:.4f}")

        return X_train_scaled, X_test_scaled, y_train, y_test

    def evaluate_models(self, X_test, y_test):
        """모델 평가 및 리포트 생성"""
        print("[4/6] 모델 평가...")
        
        evaluation_results = {}
        
        for name, model_info in self.models.items():
            if name == "lstm":
                # LSTM은 별도 처리
                X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
                y_pred_proba = model_info["model"].predict(X_test_lstm, verbose=0).flatten()
            else:
                y_pred_proba = model_info["model"].predict_proba(X_test)[:, 1]
            
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # 분류 리포트
            report = classification_report(y_test, y_pred, output_dict=True)
            
            evaluation_results[name] = {
                "auc": model_info.get("test_auc", roc_auc_score(y_test, y_pred_proba)),
                "accuracy": report["accuracy"],
                "precision": report["1"]["precision"],
                "recall": report["1"]["recall"],
                "f1_score": report["1"]["f1-score"],
                "avg_confidence": np.mean(y_pred_proba),
                "confidence_std": np.std(y_pred_proba)
            }
            
            print(f"\n{name.upper()} 평가 결과:")
            print(f"  AUC: {evaluation_results[name]['auc']:.4f}")
            print(f"  정확도: {evaluation_results[name]['accuracy']:.4f}")
            print(f"  F1 점수: {evaluation_results[name]['f1_score']:.4f}")
            print(f"  평균 신뢰도: {evaluation_results[name]['avg_confidence']:.4f}")
            print(f"  신뢰도 표준편차: {evaluation_results[name]['confidence_std']:.4f}")
        
        return evaluation_results

    def save_models(self):
        """모델 저장"""
        print("[5/6] 모델 저장...")
        
        # 모델별 저장
        for name, model_info in self.models.items():
            if name == "lstm":
                model_info["model"].save(f"{self.models_dir}/lstm_improved_model.h5")
            else:
                joblib.dump(model_info["model"], f"{self.models_dir}/{name}_improved_model.pkl")
        
        # 스케일러 저장
        joblib.dump(self.scaler, f"{self.models_dir}/scaler_improved.pkl")
        
        # 성능 결과 저장
        performance_results = {}
        for name, model_info in self.models.items():
            performance_results[name] = {
                k: v for k, v in model_info.items() 
                if k != "model" and k != "history"
            }
        
        with open(f"{self.data_dir}/raw/improved_model_performance.json", "w") as f:
            json.dump(performance_results, f, indent=2, default=str)
        
        print("✅ 모든 모델이 저장되었습니다.")

    def run_improved_training_pipeline(self):
        """개선된 전체 훈련 파이프라인 실행"""
        print("=== 개선된 모델 훈련 파이프라인 시작 ===\n")
        
        # 1. 데이터 로드
        df = self.load_training_data()
        if df is None:
            return False
        
        # 2. 특성 준비
        X, y, feature_names = self.prepare_features(df)
        if X is None:
            return False
        
        # 3. 모델 훈련
        X_train, X_test, y_train, y_test = self.train_improved_models(X, y)
        
        # 4. 모델 평가
        evaluation_results = self.evaluate_models(X_test, y_test)
        
        # 5. 모델 저장
        self.save_models()
        
        print("\n=== 훈련 완료 ===")
        
        # 최고 성능 모델 찾기
        best_model = max(evaluation_results.keys(), 
                        key=lambda x: evaluation_results[x]['auc'])
        
        print(f"\n🏆 최고 성능 모델: {best_model}")
        print(f"   AUC: {evaluation_results[best_model]['auc']:.4f}")
        print(f"   평균 신뢰도: {evaluation_results[best_model]['avg_confidence']:.4f} ± {evaluation_results[best_model]['confidence_std']:.4f}")
        
        return True


if __name__ == "__main__":
    print("🚀 개선된 AI 모델 훈련 시작")
    
    model_trainer = ImprovedSP500EventDetectionModel()
    success = model_trainer.run_improved_training_pipeline()
    
    if success:
        print("\n✅ 훈련 성공!")
    else:
        print("\n❌ 훈련 실패!")