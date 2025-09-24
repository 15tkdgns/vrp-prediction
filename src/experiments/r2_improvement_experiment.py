#!/usr/bin/env python3
"""
R² 개선 실험 - 단계적 접근법
음수 R² 문제 해결을 위한 체계적 실험

실험 목표:
1. 타겟 변수 재설계로 R² > 0 달성
2. 시장 체제 분할로 예측력 향상
3. 정규화 최적화로 과적합 방지
4. 경제적 실효성 평가
"""

import pandas as pd
import numpy as np
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    """실험 설정"""
    data_path: str = "data/training/sp500_2020_2024_enhanced.csv"
    output_dir: str = "results/r2_improvement/"
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5

class TargetVariableExperiment:
    """타겟 변수 재설계 실험"""

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.results = {}

    def create_alternative_targets(self) -> Dict[str, pd.Series]:
        """다양한 타겟 변수 생성"""
        logger.info("🎯 다양한 타겟 변수 생성 중...")

        targets = {}

        # 현재 타겟 (일일 수익률)
        if 'Returns' in self.data.columns:
            targets['daily_returns'] = self.data['Returns'].copy()

        # 1. 주간/월간 수익률
        if 'Close' in self.data.columns:
            close = self.data['Close']

            # 3일 수익률
            targets['returns_3d'] = (close.shift(-3) - close) / close

            # 5일 수익률
            targets['returns_5d'] = (close.shift(-5) - close) / close

            # 7일 수익률
            targets['returns_7d'] = (close.shift(-7) - close) / close

            # 로그 수익률
            targets['log_returns'] = np.log(close.shift(-1) / close)

            # 2. 변동성 조정 수익률
            if 'Volatility' in self.data.columns:
                daily_ret = (close.shift(-1) - close) / close
                vol = self.data['Volatility'].rolling(20).mean()
                targets['vol_adjusted_returns'] = daily_ret / (vol + 1e-8)

            # 3. 다음날 변동성 예측
            high = self.data.get('High', close)
            low = self.data.get('Low', close)
            targets['next_day_volatility'] = (high.shift(-1) - low.shift(-1)) / close

            # 4. 수익률 범위
            targets['return_range'] = (high.shift(-1) - low.shift(-1)) / close

            # 5. 절대 수익률
            targets['abs_returns'] = np.abs((close.shift(-1) - close) / close)

        # 결측치 제거
        for name, target in targets.items():
            targets[name] = target.dropna()
            logger.info(f"   📊 {name}: {len(targets[name])} 샘플")

        return targets

    def evaluate_target_predictability(self, targets: Dict[str, pd.Series]) -> Dict[str, Dict]:
        """타겟별 예측 가능성 평가"""
        logger.info("📈 타겟 변수별 예측 가능성 평가...")

        results = {}

        # 기본 특성 선택 (가장 안전한 특성들만)
        safe_features = [
            'MA_20', 'MA_50', 'RSI', 'Volatility', 'Volume_ratio',
            'Returns_lag_1', 'Returns_lag_2', 'Returns_lag_3'
        ]

        for target_name, target in targets.items():
            logger.info(f"🔍 {target_name} 평가 중...")

            try:
                # 데이터 정렬
                aligned_data = self.data.loc[target.index]

                # 특성 추출
                available_features = [f for f in safe_features if f in aligned_data.columns]
                if len(available_features) < 3:
                    logger.warning(f"   ⚠️ {target_name}: 충분한 특성 없음")
                    continue

                X = aligned_data[available_features].fillna(method='ffill').fillna(0)
                y = target.values

                # 유효한 데이터만 선택
                valid_mask = ~(np.isnan(y) | np.isinf(y))
                X = X[valid_mask]
                y = y[valid_mask]

                if len(y) < 100:
                    logger.warning(f"   ⚠️ {target_name}: 샘플 수 부족 ({len(y)})")
                    continue

                # 간단한 Ridge 회귀로 예측력 테스트
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)

                # 시계열 교차검증
                tscv = TimeSeriesSplit(n_splits=3)

                models = {
                    'Ridge': Ridge(alpha=1.0),
                    'Lasso': Lasso(alpha=0.01),
                    'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5)
                }

                target_results = {}

                for model_name, model in models.items():
                    scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='r2')
                    target_results[f'{model_name}_r2'] = {
                        'mean': np.mean(scores),
                        'std': np.std(scores),
                        'scores': scores.tolist()
                    }

                # 기본 통계
                target_results['stats'] = {
                    'mean': np.mean(y),
                    'std': np.std(y),
                    'skew': float(pd.Series(y).skew()),
                    'kurtosis': float(pd.Series(y).kurtosis()),
                    'samples': len(y),
                    'features': len(available_features)
                }

                results[target_name] = target_results

                # 최고 성과 출력
                best_r2 = max([target_results[f'{m}_r2']['mean'] for m in models.keys()])
                logger.info(f"   ✅ {target_name}: 최고 R² = {best_r2:.4f}")

            except Exception as e:
                logger.error(f"   ❌ {target_name}: {str(e)}")
                continue

        return results

class MarketRegimeModeling:
    """시장 체제 분할 모델링"""

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()

    def identify_market_regimes(self) -> pd.Series:
        """시장 체제 식별"""
        logger.info("🌊 시장 체제 식별 중...")

        # VIX 기반 공포/탐욕 구분 (VIX 없으면 변동성 사용)
        if 'VIX' in self.data.columns:
            fear_indicator = self.data['VIX']
        else:
            fear_indicator = self.data['Volatility'].rolling(20).mean()

        # 추세 식별 (MA 기반)
        if 'MA_20' in self.data.columns and 'MA_50' in self.data.columns:
            trend = (self.data['MA_20'] / self.data['MA_50'] - 1) * 100
        else:
            trend = self.data['Close'].pct_change(20) * 100

        # 체제 분류
        regimes = pd.Series(index=self.data.index, dtype=str)

        fear_threshold = fear_indicator.quantile(0.7)
        trend_up_threshold = 2.0  # 2% 이상 상승 추세
        trend_down_threshold = -2.0  # 2% 이상 하락 추세

        # 4가지 체제 정의
        bull_calm = (trend > trend_up_threshold) & (fear_indicator < fear_threshold)
        bull_volatile = (trend > trend_up_threshold) & (fear_indicator >= fear_threshold)
        bear_calm = (trend < trend_down_threshold) & (fear_indicator < fear_threshold)
        bear_volatile = (trend < trend_down_threshold) & (fear_indicator >= fear_threshold)
        sideways = (trend >= trend_down_threshold) & (trend <= trend_up_threshold)

        regimes[bull_calm] = 'bull_calm'
        regimes[bull_volatile] = 'bull_volatile'
        regimes[bear_calm] = 'bear_calm'
        regimes[bear_volatile] = 'bear_volatile'
        regimes[sideways] = 'sideways'

        # 통계 출력
        regime_counts = regimes.value_counts()
        logger.info("📊 시장 체제 분포:")
        for regime, count in regime_counts.items():
            pct = count / len(regimes) * 100
            logger.info(f"   {regime}: {count}개 ({pct:.1f}%)")

        return regimes

    def train_regime_specific_models(self, X: pd.DataFrame, y: pd.Series,
                                   regimes: pd.Series) -> Dict[str, object]:
        """체제별 모델 훈련"""
        logger.info("🤖 체제별 모델 훈련 중...")

        models = {}
        regime_performance = {}

        # 인덱스 정렬
        common_index = X.index.intersection(y.index).intersection(regimes.index)
        X_aligned = X.loc[common_index]
        y_aligned = y.loc[common_index] if hasattr(y, 'loc') else pd.Series(y, index=X.index).loc[common_index]
        regimes_aligned = regimes.loc[common_index]

        for regime in regimes_aligned.unique():
            if pd.isna(regime):
                continue

            mask = regimes_aligned == regime
            if mask.sum() < 50:  # 최소 샘플 수
                logger.warning(f"   ⚠️ {regime}: 샘플 수 부족 ({mask.sum()})")
                continue

            X_regime = X_aligned[mask]
            y_regime = y_aligned[mask]

            # 결측치 처리
            valid_idx = ~(X_regime.isna().any(axis=1) | y_regime.isna())
            X_regime = X_regime[valid_idx]
            y_regime = y_regime[valid_idx]

            if len(y_regime) < 30:
                continue

            # 정규화
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_regime)

            # Ridge 회귀 훈련
            model = Ridge(alpha=1.0)
            model.fit(X_scaled, y_regime)

            # 성능 평가
            y_pred = model.predict(X_scaled)
            r2 = r2_score(y_regime, y_pred)

            models[regime] = {
                'model': model,
                'scaler': scaler,
                'r2': r2,
                'samples': len(y_regime)
            }

            logger.info(f"   ✅ {regime}: R² = {r2:.4f}, 샘플 = {len(y_regime)}")

        return models

class R2ImprovementExperiment:
    """R² 개선 통합 실험"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = {}

        # 출력 디렉토리 생성
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        """데이터 로드"""
        logger.info(f"📁 데이터 로드: {self.config.data_path}")

        data = pd.read_csv(self.config.data_path, index_col=0, parse_dates=True)
        logger.info(f"   📊 형태: {data.shape}")
        logger.info(f"   📅 기간: {data.index.min()} ~ {data.index.max()}")

        return data

    def run_experiments(self):
        """전체 실험 실행"""
        logger.info("🚀 R² 개선 실험 시작")

        # 1. 데이터 로드
        data = self.load_data()

        # 2. 타겟 변수 실험
        logger.info("\n" + "="*50)
        logger.info("🎯 Phase 1: 타겟 변수 최적화")
        logger.info("="*50)

        target_exp = TargetVariableExperiment(data)
        targets = target_exp.create_alternative_targets()
        target_results = target_exp.evaluate_target_predictability(targets)

        self.results['target_experiments'] = target_results

        # 최고 성과 타겟 선택
        best_target = self.select_best_target(target_results)
        logger.info(f"🏆 최고 성과 타겟: {best_target}")

        # 3. 시장 체제 모델링
        logger.info("\n" + "="*50)
        logger.info("🌊 Phase 2: 시장 체제 분할 모델링")
        logger.info("="*50)

        regime_modeling = MarketRegimeModeling(data)
        regimes = regime_modeling.identify_market_regimes()

        # 4. 최종 모델 훈련 및 평가
        logger.info("\n" + "="*50)
        logger.info("🤖 Phase 3: 통합 모델 훈련")
        logger.info("="*50)

        final_results = self.train_final_models(data, targets[best_target], regimes)
        self.results['final_models'] = final_results

        # 5. 결과 저장
        self.save_results()

        # 6. 요약 보고서
        self.generate_summary_report()

    def select_best_target(self, target_results: Dict) -> str:
        """최고 성과 타겟 선택"""
        best_score = -np.inf
        best_target = 'daily_returns'

        for target_name, results in target_results.items():
            # Ridge 모델 R² 기준
            if 'Ridge_r2' in results:
                score = results['Ridge_r2']['mean']
                if score > best_score:
                    best_score = score
                    best_target = target_name

        return best_target

    def train_final_models(self, data: pd.DataFrame, target: pd.Series,
                          regimes: pd.Series) -> Dict:
        """최종 모델 훈련"""

        # 안전한 특성 선택
        safe_features = [
            'MA_20', 'MA_50', 'RSI', 'Volatility', 'Volume_ratio',
            'Returns_lag_1', 'Returns_lag_2', 'Returns_lag_3',
            'BB_position', 'ATR'
        ]

        available_features = [f for f in safe_features if f in data.columns]

        # 데이터 정렬
        aligned_data = data.loc[target.index]
        aligned_regimes = regimes.loc[target.index]

        X = aligned_data[available_features].fillna(method='ffill').fillna(0)
        y = target.values

        # 유효 데이터 선택
        valid_mask = ~(np.isnan(y) | np.isinf(y) | X.isna().any(axis=1))
        X = X[valid_mask]
        y = y[valid_mask]
        aligned_regimes = aligned_regimes[valid_mask]

        results = {}

        # 1. 전체 데이터 모델
        logger.info("🔧 전체 데이터 모델 훈련...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        models = {
            'Ridge_optimized': Ridge(alpha=0.1),
            'Lasso_optimized': Lasso(alpha=0.001),
            'ElasticNet_optimized': ElasticNet(alpha=0.001, l1_ratio=0.5)
        }

        for name, model in models.items():
            # 시계열 교차검증
            tscv = TimeSeriesSplit(n_splits=5)
            scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='r2')

            # 전체 데이터 훈련
            model.fit(X_scaled, y)
            y_pred = model.predict(X_scaled)

            results[name] = {
                'cv_r2_mean': np.mean(scores),
                'cv_r2_std': np.std(scores),
                'train_r2': r2_score(y, y_pred),
                'train_mse': mean_squared_error(y, y_pred),
                'train_mae': mean_absolute_error(y, y_pred),
                'model': model,
                'scaler': scaler
            }

            logger.info(f"   {name}: CV R² = {np.mean(scores):.4f} ± {np.std(scores):.4f}")

        # 2. 체제별 모델
        logger.info("🌊 체제별 모델 훈련...")
        regime_modeling = MarketRegimeModeling(data)
        regime_models = regime_modeling.train_regime_specific_models(X, pd.Series(y), aligned_regimes)

        results['regime_models'] = regime_models

        return results

    def save_results(self):
        """결과 저장"""
        logger.info("💾 결과 저장 중...")

        # JSON으로 저장 (모델 제외)
        import json

        # 모델 객체 제거한 결과 생성
        save_results = {}
        for key, value in self.results.items():
            if key == 'final_models':
                save_results[key] = {}
                for model_name, model_data in value.items():
                    if model_name == 'regime_models':
                        save_results[key][model_name] = {
                            regime: {k: v for k, v in regime_data.items() if k not in ['model', 'scaler']}
                            for regime, regime_data in model_data.items()
                        }
                    else:
                        save_results[key][model_name] = {
                            k: v for k, v in model_data.items() if k not in ['model', 'scaler']
                        }
            else:
                save_results[key] = value

        with open(f"{self.config.output_dir}/experiment_results.json", 'w') as f:
            json.dump(save_results, f, indent=2, default=str)

        # 모델들 별도 저장
        if 'final_models' in self.results:
            for name, model_data in self.results['final_models'].items():
                if name != 'regime_models' and 'model' in model_data:
                    joblib.dump(model_data['model'], f"{self.config.output_dir}/{name}_model.pkl")
                    joblib.dump(model_data['scaler'], f"{self.config.output_dir}/{name}_scaler.pkl")

        logger.info(f"   📁 저장 위치: {self.config.output_dir}")

    def generate_summary_report(self):
        """요약 보고서 생성"""
        logger.info("\n" + "="*60)
        logger.info("📊 R² 개선 실험 결과 요약")
        logger.info("="*60)

        # 타겟 변수 결과
        if 'target_experiments' in self.results:
            logger.info("\n🎯 타겟 변수별 최고 R² 성과:")
            for target, results in self.results['target_experiments'].items():
                if 'Ridge_r2' in results:
                    r2 = results['Ridge_r2']['mean']
                    samples = results['stats']['samples']
                    logger.info(f"   {target:20s}: R² = {r2:7.4f} (샘플: {samples})")

        # 최종 모델 결과
        if 'final_models' in self.results:
            logger.info("\n🤖 최종 모델 성과:")
            for name, results in self.results['final_models'].items():
                if name != 'regime_models' and 'cv_r2_mean' in results:
                    cv_r2 = results['cv_r2_mean']
                    cv_std = results['cv_r2_std']
                    train_r2 = results['train_r2']
                    logger.info(f"   {name:20s}: CV R² = {cv_r2:7.4f} ± {cv_std:.4f}, Train R² = {train_r2:7.4f}")

        # 체제별 모델 결과
        if 'final_models' in self.results and 'regime_models' in self.results['final_models']:
            logger.info("\n🌊 체제별 모델 성과:")
            for regime, model_data in self.results['final_models']['regime_models'].items():
                r2 = model_data['r2']
                samples = model_data['samples']
                logger.info(f"   {regime:15s}: R² = {r2:7.4f} (샘플: {samples})")

        # 개선 여부 판단
        best_r2 = -np.inf
        if 'final_models' in self.results:
            for name, results in self.results['final_models'].items():
                if name != 'regime_models' and 'cv_r2_mean' in results:
                    best_r2 = max(best_r2, results['cv_r2_mean'])

        logger.info("\n" + "="*60)
        if best_r2 > 0:
            logger.info("🎉 성공! 양수 R² 달성!")
            logger.info(f"   최고 성과: R² = {best_r2:.4f}")
        else:
            logger.info("⚠️  여전히 음수 R², 추가 개선 필요")
            logger.info(f"   현재 최고: R² = {best_r2:.4f}")

        logger.info("="*60)


def main():
    """메인 실행 함수"""
    config = ExperimentConfig()
    experiment = R2ImprovementExperiment(config)
    experiment.run_experiments()


if __name__ == "__main__":
    main()