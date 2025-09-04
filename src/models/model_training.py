import pandas as pd
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import joblib
import matplotlib.pyplot as plt
import logging
from ..utils.directory_manager import DirectoryManager
from ..core.config_manager import get_config_manager


class SP500EventDetectionModel:
    """
    S&P500 주식 데이터 기반 이벤트 탐지 모델을 훈련, 평가, 저장하는 클래스.

    이 클래스는 데이터 로딩, 특성 전처리, 다양한 머신러닝/딥러닝 모델 훈련,
    성능 평가, 특성 중요도 시각화, 모델 저장 등의 기능을 포함하는
    전체 모델링 파이프라인을 관리합니다.
    """

    def __init__(self, data_dir="data", models_dir="data/models"):
        """
        SP500EventDetectionModel 인스턴스를 초기화합니다.

        Args:
            data_dir (str): 원본 데이터가 저장된 디렉토리 경로.
            models_dir (str): 훈련된 모델과 스케일러가 저장될 디렉토리 경로.
        """
        # 디렉토리 관리자 및 설정 관리자 초기화
        self.directory_manager = DirectoryManager()
        self.config_manager = get_config_manager()
        
        # 필요한 디렉토리들 자동 생성
        additional_dirs = [
            'results/analysis',
            'results/training', 
            'results/visualizations',
            'logs/models'
        ]
        self.directory_manager.ensure_directories(additional_dirs)
        
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.scaler = StandardScaler()
        self.models = {}
        
        # 로거 설정
        self.logger = logging.getLogger(__name__)

        # 모델 저장 디렉토리 생성 (기존 방식 유지하되 로깅 추가)
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
            self.logger.info(f"📁 모델 저장 디렉토리 생성: {self.models_dir}")
        else:
            self.logger.info(f"📁 모델 저장 디렉토리 확인: {self.models_dir}")

    def load_training_data(self):
        """
        훈련에 필요한 특성(features)과 라벨(labels) 데이터를 로드하고 병합합니다.

        Returns:
            pd.DataFrame: 특성과 라벨이 병합된 데이터프레임.
        """
        # 학습 특성 및 이벤트 라벨 로드
        features_df = pd.read_csv(f"{self.data_dir}/raw/training_features.csv")
        labels_df = pd.read_csv(f"{self.data_dir}/raw/event_labels.csv")

        # LLM 강화 특징 로드
        try:
            llm_features_df = pd.read_csv(
                f"{self.data_dir}/processed/llm_enhanced_features.csv"
            )
            llm_features_df = llm_features_df.dropna(
                subset=["date"]
            )  # date 컬럼에 NaN 값이 있는 행 제거
            llm_features_df["date"] = pd.to_datetime(llm_features_df["date"])
        except FileNotFoundError:
            print(
                "LLM enhanced features file not found. Proceeding without LLM features."
            )
            llm_features_df = pd.DataFrame()

        # 날짜 형식 통일
        features_df["Date"] = pd.to_datetime(features_df["Date"])
        labels_df["Date"] = pd.to_datetime(labels_df["Date"])

        # 'ticker'와 날짜를 기준으로 데이터 병합
        merged_df = pd.merge(
            features_df,
            labels_df,
            left_on=["ticker", "Date"],
            right_on=["ticker", "Date"],
            how="inner",
        )

        # LLM 특징 병합 (뉴스 제목과 날짜를 기준으로)
        if not llm_features_df.empty:
            merged_df = pd.merge(
                merged_df,
                llm_features_df,
                left_on=["date", "title"],  # news_data.csv의 title과 매핑
                right_on=["date", "title"],
                how="left",
                suffixes=("", "_llm"),
            )
            # 병합 후 중복 컬럼 제거 (예: title_llm, date_llm 등)
            merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()].copy()

        return merged_df

    def prepare_features(self, df):
        """
        모델 훈련을 위해 특성을 선택, 정제하고 스케일링합니다.

        Args:
            df (pd.DataFrame): 원본 데이터프레임.

        Returns:
            tuple: 스케일링된 특성(X_scaled)과 특성 이름 목록(numeric_features).
        """
        # 모델에 사용할 수치형 특성 목록
        numeric_features = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "sma_20",
            "sma_50",
            "rsi",
            "macd",
            "bb_upper",
            "bb_lower",
            "atr",
            "volatility",
            "obv",
            "price_change",
            "volume_change",
            "unusual_volume",
            "price_spike",
            "news_sentiment",
            "news_polarity",
            "news_count",
            "llm_sentiment_score",
            "uncertainty_score",  # LLM 수치형 특징 추가
        ]

        # 범주형 LLM 특징 원-핫 인코딩
        categorical_llm_features = ["market_sentiment", "event_category"]
        for col in categorical_llm_features:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown")  # 결측값 처리
                df = pd.get_dummies(df, columns=[col], prefix=col)

        # 원-핫 인코딩 후 추가된 컬럼들을 numeric_features에 포함
        # 기존 numeric_features에 없는 새로운 더미 변수 컬럼들을 찾아서 추가
        current_features = set(df.columns)
        new_dummy_features = [
            col
            for col in current_features
            if col.startswith("market_sentiment_") or col.startswith("event_category_")
        ]
        numeric_features.extend(new_dummy_features)

        # 실제 존재하는 컬럼들만 필터링
        available_features = [col for col in numeric_features if col in df.columns]
        print(f"Available features: {available_features}")

        # 특성 데이터 선택 및 결측값 처리 (0으로 채움)
        X = df[available_features].fillna(0)

        # StandardScaler를 이용한 특성 정규화
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, available_features

    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """
        Random Forest 분류 모델을 훈련하고 성능을 평가합니다.

        Args:
            X_train, y_train: 훈련 데이터 (특성 및 라벨).
            X_test, y_test: 테스트 데이터 (특성 및 라벨).
        """
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight="balanced",  # 불균형 데이터 처리를 위한 가중치 조정
        )
        rf_model.fit(X_train, y_train)

        # 훈련 및 테스트 정확도 평가
        train_score = rf_model.score(X_train, y_train)
        test_score = rf_model.score(X_test, y_test)
        print(f"Random Forest - Train: {train_score:.4f}, Test: {test_score:.4f}")

        # 모델 및 관련 정보 저장
        self.models["random_forest"] = {
            "model": rf_model,
            "train_score": train_score,
            "test_score": test_score,
            "feature_importance": rf_model.feature_importances_,
        }

    def train_gradient_boosting(self, X_train, y_train, X_test, y_test):
        """
        Gradient Boosting 분류 모델을 훈련하고 성능을 평가합니다.
        """
        gb_model = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
        )
        gb_model.fit(X_train, y_train)

        train_score = gb_model.score(X_train, y_train)
        test_score = gb_model.score(X_test, y_test)
        print(f"Gradient Boosting - Train: {train_score:.4f}, Test: {test_score:.4f}")

        self.models["gradient_boosting"] = {
            "model": gb_model,
            "train_score": train_score,
            "test_score": test_score,
        }

    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """
        XGBoost 분류 모델을 훈련하고 성능을 평가합니다.
        
        Args:
            X_train, y_train: 훈련 데이터 (특성 및 라벨).
            X_test, y_test: 테스트 데이터 (특성 및 라벨).
        """
        # XGBoost 분류기 설정
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            eval_metric='logloss',  # 이진 분류를 위한 평가 메트릭
            use_label_encoder=False  # 최신 버전에서 권장
        )
        
        # 모델 훈련
        xgb_model.fit(X_train, y_train)
        
        # 훈련 및 테스트 정확도 평가
        train_score = xgb_model.score(X_train, y_train)
        test_score = xgb_model.score(X_test, y_test)
        print(f"XGBoost - Train: {train_score:.4f}, Test: {test_score:.4f}")
        
        # 특성 중요도 추출 (XGBoost는 내장 특성 중요도 제공)
        feature_importance = xgb_model.feature_importances_
        
        # 모델 및 평가 결과 저장
        self.models["xgboost"] = {
            "model": xgb_model,
            "train_score": train_score,
            "test_score": test_score,
            "feature_importance": feature_importance,
        }

    def train_lstm(self, X_train, y_train, X_test, y_test):
        """
        LSTM(Long Short-Term Memory) 딥러닝 모델을 훈련합니다.
        시계열 특성을 고려하기 위해 사용됩니다.
        """
        # LSTM 입력 형식에 맞게 데이터 형태 변환: (samples, timesteps, features)
        X_train_lstm = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_test_lstm = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

        # Keras Sequential API를 사용한 LSTM 모델 구성
        lstm_model = Sequential(
            [
                LSTM(50, return_sequences=True, input_shape=(1, X_train.shape[1])),
                Dropout(0.2),  # 과적합 방지를 위한 Dropout
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25, activation="relu"),
                Dense(1, activation="sigmoid"),  # 이진 분류를 위한 Sigmoid 활성화 함수
            ]
        )

        # 모델 컴파일
        lstm_model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )

        # 모델 훈련
        history = lstm_model.fit(
            X_train_lstm,
            y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test_lstm, y_test),
            verbose=0,  # 훈련 과정 출력 생략
        )

        # 성능 평가
        _, train_score = lstm_model.evaluate(X_train_lstm, y_train, verbose=0)
        _, test_score = lstm_model.evaluate(X_test_lstm, y_test, verbose=0)
        print(f"LSTM - Train: {train_score:.4f}, Test: {test_score:.4f}")

        self.models["lstm"] = {
            "model": lstm_model,
            "train_score": train_score,
            "test_score": test_score,
            "history": history.history,
        }

    def plot_feature_importance(self, feature_names):
        """
        Random Forest 모델의 특성 중요도를 시각화하고 이미지 파일로 저장합니다.
        """
        if "random_forest" in self.models:
            importance = self.models["random_forest"]["feature_importance"]

            # 중요도 순으로 정렬
            indices = np.argsort(importance)[::-1]

            plt.figure(figsize=(12, 8))
            plt.title("Feature Importance (Random Forest)")
            plt.bar(range(len(importance)), importance[indices])
            plt.xticks(
                range(len(importance)),
                [feature_names[i] for i in indices],
                rotation=45,
                ha="right",
            )
            plt.tight_layout()
            # 결과 디렉토리 생성
            os.makedirs("results/analysis", exist_ok=True)
            plt.savefig("results/analysis/feature_importance.png")
            plt.close()
            print(
                "특성 중요도 그래프가 'results/analysis/feature_importance.png'에 저장되었습니다."
            )

    def save_models(self):
        """
        훈련된 모든 모델과 전처리에 사용된 스케일러를 파일로 저장합니다.
        - Scikit-learn 모델: .pkl (joblib)
        - Keras 모델: .h5
        - 모델 성능: .json
        """
        for name, model_info in self.models.items():
            if name == "lstm":
                model_info["model"].save(f"{self.models_dir}/{name}_model.h5")
            else:
                joblib.dump(model_info["model"], f"{self.models_dir}/{name}_model.pkl")

        joblib.dump(self.scaler, f"{self.models_dir}/scaler.pkl")
        print(f"모든 모델과 스케일러가 '{self.models_dir}' 디렉토리에 저장되었습니다.")

        # 모델별 성능 지표를 JSON 파일로 저장
        performance = {
            name: {
                "train_accuracy": info["train_score"],
                "test_accuracy": info["test_score"],
            }
            for name, info in self.models.items()
        }
        with open(f"{self.data_dir}/raw/model_performance.json", "w") as f:
            json.dump(performance, f, indent=4)

    def run_training_pipeline(self):
        """
        전체 모델 훈련 파이프라인을 순차적으로 실행합니다.
        """
        print("--- 모델 훈련 파이프라인 시작 ---")

        print("\n[1/6] 훈련 데이터 로드...")
        df = self.load_training_data()

        print("\n[2/6] 특성 데이터 전처리...")
        X, feature_names = self.prepare_features(df)
        y = df["major_event"].values  # 타겟 변수 설정

        print("\n[3/6] 훈련/테스트 데이터 분할...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y  # 라벨 비율 유지
        )

        print("\n[4/6] 모델 훈련...")
        self.train_random_forest(X_train, y_train, X_test, y_test)
        self.train_gradient_boosting(X_train, y_train, X_test, y_test)
        self.train_xgboost(X_train, y_train, X_test, y_test)
        self.train_lstm(X_train, y_train, X_test, y_test)

        print("\n[5/6] 특성 중요도 시각화...")
        self.plot_feature_importance(feature_names)

        print("\n[6/6] 모델 및 결과 저장...")
        self.save_models()

        print("\n--- 모델 훈련 파이프라인 종료 ---")


if __name__ == "__main__":
    # 클래스 인스턴스 생성 및 파이프라인 실행
    model_trainer = SP500EventDetectionModel()
    model_trainer.run_training_pipeline()
