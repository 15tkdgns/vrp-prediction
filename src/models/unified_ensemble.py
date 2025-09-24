#!/usr/bin/env python3
"""
통합 앙상블 시스템 - 9개 분산 파일들의 핵심 기능 통합
유지보수성, 가독성, 재사용성을 극대화한 단일 앙상블 시스템
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error
from sklearn.preprocessing import RobustScaler
import joblib
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


@dataclass
class EnsembleConfig:
    """앙상블 설정 클래스"""
    voting_strategy: str = "soft"  # "hard", "soft"
    stacking_cv: int = 5
    meta_learner_hidden_size: int = 64
    use_gpu: bool = True
    validation_split: float = 0.2
    early_stopping_patience: int = 20


class BaseEnsembleMethod(ABC):
    """앙상블 방법의 기본 추상 클래스"""

    def __init__(self, name: str, config: EnsembleConfig):
        self.name = name
        self.config = config
        self.models = {}
        self.weights = {}
        self.is_fitted = False
        self.device = torch.device('cuda' if config.use_gpu and torch.cuda.is_available() else 'cpu')

    def add_model(self, name: str, model: Any, weight: float = 1.0) -> None:
        """모델을 앙상블에 추가"""
        self.models[name] = model
        self.weights[name] = weight

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseEnsembleMethod':
        """앙상블 훈련"""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """앙상블 예측"""
        pass

    def get_model_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """개별 모델들의 예측 결과 반환"""
        predictions = {}
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)[:, 1] if len(pred.shape) > 1 else pred
            elif hasattr(model, 'predict'):
                pred = model.predict(X)
            else:
                raise ValueError(f"모델 {name}에 predict 메서드가 없습니다.")
            predictions[name] = pred
        return predictions


class WeightedVotingEnsemble(BaseEnsembleMethod):
    """가중 투표 앙상블"""

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'WeightedVotingEnsemble':
        """가중 투표 앙상블 훈련"""
        # 개별 모델들이 이미 훈련되었다고 가정
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """가중 투표로 예측"""
        if not self.is_fitted:
            raise ValueError("앙상블이 훈련되지 않았습니다.")

        predictions = self.get_model_predictions(X)
        weighted_pred = np.zeros(len(X))

        total_weight = sum(self.weights.values())

        for name, pred in predictions.items():
            weight = self.weights[name] / total_weight
            weighted_pred += weight * pred

        return weighted_pred


class StackingEnsemble(BaseEnsembleMethod):
    """스태킹 앙상블"""

    def __init__(self, name: str, config: EnsembleConfig, meta_learner=None):
        super().__init__(name, config)
        self.meta_learner = meta_learner
        self.stacking_model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'StackingEnsemble':
        """스태킹 앙상블 훈련"""
        if not self.models:
            raise ValueError("앙상블에 모델이 추가되지 않았습니다.")

        # sklearn의 StackingClassifier 사용
        estimators = [(name, model) for name, model in self.models.items()]

        self.stacking_model = StackingClassifier(
            estimators=estimators,
            final_estimator=self.meta_learner,
            cv=self.config.stacking_cv,
            stack_method='predict_proba'
        )

        self.stacking_model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """스태킹으로 예측"""
        if not self.is_fitted:
            raise ValueError("앙상블이 훈련되지 않았습니다.")

        return self.stacking_model.predict_proba(X)[:, 1]


class MetaLearnerNN(nn.Module):
    """PyTorch 기반 메타 학습기"""

    def __init__(self, num_base_models: int, hidden_size: int = 64, dropout: float = 0.3):
        super(MetaLearnerNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(num_base_models, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x).squeeze()


class NeuralStackingEnsemble(BaseEnsembleMethod):
    """신경망 기반 스태킹 앙상블"""

    def __init__(self, name: str, config: EnsembleConfig):
        super().__init__(name, config)
        self.meta_learner = None
        self.scaler = RobustScaler()

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'NeuralStackingEnsemble':
        """신경망 메타 학습기 훈련"""
        if not self.models:
            raise ValueError("앙상블에 모델이 추가되지 않았습니다.")

        # 1단계: 기본 모델들의 예측 생성
        base_predictions = []
        for name, model in self.models.items():
            pred = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X)
            base_predictions.append(pred)

        meta_features = np.column_stack(base_predictions)
        meta_features = self.scaler.fit_transform(meta_features)

        # 2단계: 메타 학습기 훈련
        self.meta_learner = MetaLearnerNN(
            num_base_models=len(self.models),
            hidden_size=self.config.meta_learner_hidden_size
        ).to(self.device)

        # 훈련 설정
        criterion = nn.BCELoss()
        optimizer = optim.AdamW(self.meta_learner.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        # 데이터 준비
        X_tensor = torch.FloatTensor(meta_features).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        # 훈련 루프
        self.meta_learner.train()
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(1000):
            optimizer.zero_grad()
            outputs = self.meta_learner(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.meta_learner.parameters(), max_norm=1.0)
            optimizer.step()

            scheduler.step(loss)

            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    break

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """신경망 메타 학습기로 예측"""
        if not self.is_fitted:
            raise ValueError("앙상블이 훈련되지 않았습니다.")

        # 기본 모델들의 예측 생성
        base_predictions = []
        for name, model in self.models.items():
            pred = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X)
            base_predictions.append(pred)

        meta_features = np.column_stack(base_predictions)
        meta_features = self.scaler.transform(meta_features)

        # 메타 학습기로 최종 예측
        self.meta_learner.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(meta_features).to(self.device)
            predictions = self.meta_learner(X_tensor).cpu().numpy()

        return predictions


class EnsembleFactory:
    """앙상블 생성 팩토리"""

    @staticmethod
    def create_ensemble(ensemble_type: str, config: EnsembleConfig, **kwargs) -> BaseEnsembleMethod:
        """앙상블 타입에 따른 인스턴스 생성"""
        if ensemble_type == "voting":
            return WeightedVotingEnsemble("WeightedVoting", config)
        elif ensemble_type == "stacking":
            meta_learner = kwargs.get('meta_learner', None)
            return StackingEnsemble("Stacking", config, meta_learner)
        elif ensemble_type == "neural_stacking":
            return NeuralStackingEnsemble("NeuralStacking", config)
        else:
            raise ValueError(f"지원하지 않는 앙상블 타입: {ensemble_type}")


class UnifiedEnsembleSystem:
    """통합 앙상블 시스템 - 모든 앙상블 방법들을 관리"""

    def __init__(self, config: EnsembleConfig = None):
        self.config = config or EnsembleConfig()
        self.ensembles = {}
        self.best_ensemble = None
        self.evaluation_results = {}

    def add_ensemble(self, name: str, ensemble_type: str, **kwargs) -> None:
        """앙상블 추가"""
        ensemble = EnsembleFactory.create_ensemble(ensemble_type, self.config, **kwargs)
        self.ensembles[name] = ensemble

    def add_models_to_ensemble(self, ensemble_name: str, models: Dict[str, Any], weights: Dict[str, float] = None) -> None:
        """앙상블에 모델들 추가"""
        if ensemble_name not in self.ensembles:
            raise ValueError(f"앙상블 {ensemble_name}이 존재하지 않습니다.")

        ensemble = self.ensembles[ensemble_name]
        weights = weights or {name: 1.0 for name in models.keys()}

        for name, model in models.items():
            weight = weights.get(name, 1.0)
            ensemble.add_model(name, model, weight)

    def train_all_ensembles(self, X: np.ndarray, y: np.ndarray) -> None:
        """모든 앙상블 훈련"""
        for name, ensemble in self.ensembles.items():
            print(f"앙상블 {name} 훈련 중...")
            ensemble.fit(X, y)

    def evaluate_ensembles(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """앙상블들 평가"""
        results = {}

        for name, ensemble in self.ensembles.items():
            if not ensemble.is_fitted:
                continue

            predictions = ensemble.predict(X_test)
            binary_predictions = (predictions > 0.5).astype(int)

            accuracy = accuracy_score(y_test, binary_predictions)
            f1 = f1_score(y_test, binary_predictions)
            mse = mean_squared_error(y_test, predictions)

            results[name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'mse': mse,
                'predictions': predictions
            }

        self.evaluation_results = results

        # 최고 성능 앙상블 선택 (F1 점수 기준)
        if results:
            best_name = max(results.keys(), key=lambda k: results[k]['f1_score'])
            self.best_ensemble = self.ensembles[best_name]

        return results

    def predict(self, X: np.ndarray, use_best: bool = True) -> np.ndarray:
        """예측 수행"""
        if use_best and self.best_ensemble:
            return self.best_ensemble.predict(X)
        elif len(self.ensembles) == 1:
            return list(self.ensembles.values())[0].predict(X)
        else:
            raise ValueError("사용할 앙상블을 명시하거나 평가를 먼저 수행하세요.")

    def save_system(self, save_path: str) -> None:
        """앙상블 시스템 저장"""
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 각 앙상블 저장
        for name, ensemble in self.ensembles.items():
            if hasattr(ensemble, 'meta_learner') and ensemble.meta_learner is not None:
                # PyTorch 모델 저장
                torch.save(ensemble.meta_learner.state_dict(), save_dir / f"{name}_meta_learner.pth")
            if hasattr(ensemble, 'stacking_model') and ensemble.stacking_model is not None:
                # Sklearn 모델 저장
                joblib.dump(ensemble.stacking_model, save_dir / f"{name}_stacking_model.pkl")

        # 설정 및 결과 저장
        config_data = {
            'config': {
                'voting_strategy': self.config.voting_strategy,
                'stacking_cv': self.config.stacking_cv,
                'meta_learner_hidden_size': self.config.meta_learner_hidden_size,
                'use_gpu': self.config.use_gpu
            },
            'evaluation_results': self.evaluation_results,
            'best_ensemble': self.best_ensemble.name if self.best_ensemble else None
        }

        with open(save_dir / "ensemble_system.json", 'w') as f:
            json.dump(config_data, f, indent=2)

    def get_summary(self) -> Dict[str, Any]:
        """시스템 요약 정보 반환"""
        return {
            'ensemble_count': len(self.ensembles),
            'ensemble_names': list(self.ensembles.keys()),
            'best_ensemble': self.best_ensemble.name if self.best_ensemble else None,
            'evaluation_results': self.evaluation_results,
            'config': {
                'voting_strategy': self.config.voting_strategy,
                'stacking_cv': self.config.stacking_cv,
                'use_gpu': self.config.use_gpu
            }
        }


if __name__ == "__main__":
    # 사용 예시
    config = EnsembleConfig(use_gpu=True, stacking_cv=5)
    system = UnifiedEnsembleSystem(config)

    print("✅ 통합 앙상블 시스템 초기화 완료")
    print(f"설정: GPU={config.use_gpu}, CV={config.stacking_cv}")
    print("지원 앙상블 타입: voting, stacking, neural_stacking")