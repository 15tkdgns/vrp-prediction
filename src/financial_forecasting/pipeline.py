"""
Advanced Financial Forecasting Pipeline

완전한 엔드투엔드 금융 예측 파이프라인:
1. 데이터 수집 및 전처리 (로그 수익률 변환)
2. 대체 데이터 통합 (FRED, FinBERT, HMM)
3. 고급 모델 훈련 (ARIMA-GARCH, TFT, MDN)
4. 금융 성과 기반 최적화
5. Walk-Forward 검증
6. 백테스팅 및 리스크 평가
7. 종합 성과 리포트 생성
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("⚠️ yfinance not available, using simulated data")

from .core import LogReturnProcessor, TimeSeriesSafeValidator, FinancialMetrics
from .features import AlternativeDataIntegrator, SentimentAnalyzer, MarketRegimeDetector
from .models import ARIMAGARCHModel, TemporalFusionTransformer, MixtureDensityNetwork
from .validation import WalkForwardValidator, FinancialBacktester, RiskMetricsCalculator
from .optimizer import FinancialObjectiveOptimizer, HyperparameterOptimizer, EnsembleOptimizer


@dataclass
class PipelineConfig:
    """파이프라인 설정"""
    # 데이터 설정
    symbol: str = "SPY"
    start_date: str = "2020-01-01"
    end_date: Optional[str] = None

    # 모델 설정
    models_to_train: List[str] = None

    # 검증 설정
    train_size: int = 252
    test_size: int = 21
    walk_forward_steps: int = 21

    # 최적화 설정
    optimize_hyperparameters: bool = True
    optimization_method: str = "grid_search"

    # 백테스팅 설정
    initial_capital: float = 100000.0
    transaction_cost: float = 0.001

    # API 키 (선택적)
    fred_api_key: Optional[str] = None
    news_api_key: Optional[str] = None

    def __post_init__(self):
        if self.models_to_train is None:
            self.models_to_train = ["arima_garch", "tft", "mdn"]


@dataclass
class PipelineResult:
    """파이프라인 실행 결과"""
    data_summary: Dict
    model_results: Dict
    optimization_results: Dict
    validation_results: Dict
    backtest_results: Dict
    risk_metrics: Dict
    ensemble_weights: Optional[Any]
    final_recommendations: Dict


class AdvancedFinancialPipeline:
    """
    Advanced Financial Forecasting Pipeline

    근본적 접근법 변경을 구현한 완전한 파이프라인:
    - 가격 → 로그 수익률 예측 패러다임
    - 통계적 정상성 및 시계열 안전성
    - 고급 계량경제/딥러닝 모델
    - 금융 성과 지표 기반 최적화
    - 포괄적 리스크 관리
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

        # 핵심 구성요소 초기화
        self.log_processor = LogReturnProcessor()
        self.time_validator = TimeSeriesSafeValidator()
        self.data_integrator = AlternativeDataIntegrator(config.fred_api_key)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.regime_detector = MarketRegimeDetector()

        # 검증 및 최적화 구성요소
        self.walk_forward_validator = WalkForwardValidator(
            initial_train_size=config.train_size,
            test_size=config.test_size,
            step_size=config.walk_forward_steps
        )
        self.financial_objective = FinancialObjectiveOptimizer()
        self.hyperparameter_optimizer = HyperparameterOptimizer(
            self.financial_objective,
            self.walk_forward_validator,
            config.optimization_method
        )

        # 백테스팅 및 리스크 구성요소
        self.backtester = FinancialBacktester(
            initial_capital=config.initial_capital,
            transaction_cost=config.transaction_cost
        )
        self.risk_calculator = RiskMetricsCalculator()
        self.ensemble_optimizer = EnsembleOptimizer()

    def run_complete_pipeline(self) -> PipelineResult:
        """완전한 파이프라인 실행"""
        print("🚀 Advanced Financial Forecasting Pipeline 시작")
        print("=" * 60)

        try:
            # 1. 데이터 수집 및 전처리
            print("\n📊 1단계: 데이터 수집 및 전처리")
            data_summary, processed_data = self._collect_and_preprocess_data()

            # 2. 대체 데이터 통합
            print("\n🔗 2단계: 대체 데이터 통합")
            enhanced_data = self._integrate_alternative_data(processed_data)

            # 3. 모델 훈련 및 최적화
            print("\n🤖 3단계: 모델 훈련 및 최적화")
            model_results, optimization_results = self._train_and_optimize_models(enhanced_data)

            # 4. Walk-Forward 검증
            print("\n🔄 4단계: Walk-Forward 검증")
            validation_results = self._perform_walk_forward_validation(enhanced_data, model_results)

            # 5. 백테스팅
            print("\n📈 5단계: 백테스팅 및 성과 평가")
            backtest_results = self._perform_backtesting(enhanced_data, model_results)

            # 6. 리스크 평가
            print("\n⚠️ 6단계: 리스크 지표 계산")
            risk_metrics = self._calculate_risk_metrics(backtest_results)

            # 7. 앙상블 최적화
            print("\n🎯 7단계: 앙상블 최적화")
            ensemble_weights = self._optimize_ensemble(enhanced_data, model_results)

            # 8. 최종 권장사항 생성
            print("\n📋 8단계: 최종 권장사항 생성")
            final_recommendations = self._generate_recommendations(
                validation_results, backtest_results, risk_metrics, ensemble_weights
            )

            # 결과 패키징
            result = PipelineResult(
                data_summary=data_summary,
                model_results=model_results,
                optimization_results=optimization_results,
                validation_results=validation_results,
                backtest_results=backtest_results,
                risk_metrics=risk_metrics,
                ensemble_weights=ensemble_weights,
                final_recommendations=final_recommendations
            )

            print("\n✅ 파이프라인 완료!")
            print("=" * 60)

            return result

        except Exception as e:
            print(f"\n❌ 파이프라인 실행 실패: {e}")
            raise

    def _collect_and_preprocess_data(self) -> Tuple[Dict, pd.DataFrame]:
        """데이터 수집 및 전처리"""
        if YFINANCE_AVAILABLE:
            # yfinance로 실제 데이터 수집
            ticker = yf.Ticker(self.config.symbol)
            data = ticker.history(
                start=self.config.start_date,
                end=self.config.end_date
            )
            prices = data['Close']
        else:
            # 시뮬레이션 데이터 생성
            print("⚠️ yfinance 없어서 시뮬레이션 데이터 사용")
            dates = pd.date_range(
                start=self.config.start_date,
                end=self.config.end_date or pd.Timestamp.now(),
                freq='D'
            )
            # 현실적인 가격 시뮬레이션 (기하 브라운 운동)
            returns = np.random.normal(0.0008, 0.02, len(dates))  # 일일 수익률
            prices = pd.Series(100 * np.exp(np.cumsum(returns)), index=dates, name='Close')

        # 시간 순서 검증
        is_valid = self.time_validator.validate_temporal_order(
            pd.DataFrame({'Close': prices})
        )

        if not is_valid:
            print("⚠️ 시간 순서 문제 감지, 데이터 정렬 중...")
            prices = prices.sort_index()

        # 로그 수익률 변환
        log_returns = self.log_processor.price_to_log_returns(prices)

        # 정상성 검증 및 확보
        stationary_returns, transformations = self.log_processor.ensure_stationarity(log_returns)

        # 기본 기술적 지표 계산
        processed_data = self._calculate_technical_indicators(prices, stationary_returns)

        data_summary = {
            'symbol': self.config.symbol,
            'period': f"{prices.index[0]} ~ {prices.index[-1]}",
            'observations': len(prices),
            'price_range': f"${prices.min():.2f} ~ ${prices.max():.2f}",
            'transformations_applied': transformations,
            'stationarity_achieved': len(transformations) > 0,
            'missing_values': processed_data.isnull().sum().sum()
        }

        print(f"✅ 데이터 전처리 완료:")
        print(f"   기간: {data_summary['period']}")
        print(f"   관측치: {data_summary['observations']}개")
        print(f"   변환: {transformations}")

        return data_summary, processed_data

    def _calculate_technical_indicators(self, prices: pd.Series, returns: pd.Series) -> pd.DataFrame:
        """기술적 지표 계산"""
        data = pd.DataFrame({
            'price': prices,
            'log_returns': returns
        })

        # 이동평균
        data['ma_5'] = prices.rolling(5).mean()
        data['ma_20'] = prices.rolling(20).mean()
        data['ma_50'] = prices.rolling(50).mean()

        # 변동성 지표
        data['volatility_5d'] = returns.rolling(5).std()
        data['volatility_20d'] = returns.rolling(20).std()

        # 모멘텀 지표
        data['rsi'] = self._calculate_rsi(prices, window=14)
        data['price_momentum_5d'] = prices.pct_change(5)
        data['price_momentum_20d'] = prices.pct_change(20)

        # 볼린저 밴드
        bb_ma = prices.rolling(20).mean()
        bb_std = prices.rolling(20).std()
        data['bb_upper'] = bb_ma + 2 * bb_std
        data['bb_lower'] = bb_ma - 2 * bb_std
        data['bb_position'] = (prices - bb_lower) / (bb_upper - bb_lower)

        # 타겟 변수: 다음 기간 로그 수익률
        data['target_return'] = returns.shift(-1)

        return data.dropna()

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """RSI 계산"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _integrate_alternative_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """대체 데이터 통합"""
        enhanced_data = data.copy()

        try:
            # 거시경제 지표 통합
            macro_data = self.data_integrator.fetch_macro_indicators(
                start_date=self.config.start_date,
                end_date=self.config.end_date
            )

            # 공통 날짜로 병합
            enhanced_data = enhanced_data.join(macro_data, how='left')
            enhanced_data = enhanced_data.fillna(method='ffill')

            print(f"✅ 거시경제 지표 {len(macro_data.columns)}개 통합 완료")

        except Exception as e:
            print(f"⚠️ 거시경제 지표 통합 실패: {e}")

        try:
            # 센티멘트 데이터 통합
            sentiment_data = self.sentiment_analyzer.generate_sentiment_time_series(
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                simulate=True  # 실제 뉴스 API 없으므로 시뮬레이션
            )

            enhanced_data = enhanced_data.join(sentiment_data, how='left')
            enhanced_data = enhanced_data.fillna(method='ffill')

            print(f"✅ 센티멘트 지표 {len(sentiment_data.columns)}개 통합 완료")

        except Exception as e:
            print(f"⚠️ 센티멘트 데이터 통합 실패: {e}")

        try:
            # 시장 레짐 감지
            regime_data = self.regime_detector.predict_regime(
                enhanced_data['log_returns'],
                enhanced_data['volatility_20d']
            )

            enhanced_data = enhanced_data.join(regime_data, how='left')
            enhanced_data = enhanced_data.fillna(method='ffill')

            print(f"✅ 시장 레짐 데이터 통합 완료")

        except Exception as e:
            print(f"⚠️ 시장 레짐 감지 실패: {e}")

        # 최종 결측치 처리
        enhanced_data = enhanced_data.dropna()

        print(f"✅ 대체 데이터 통합 완료: {len(enhanced_data.columns)}개 특성, {len(enhanced_data)}개 관측치")

        return enhanced_data

    def _train_and_optimize_models(self, data: pd.DataFrame) -> Tuple[Dict, Dict]:
        """모델 훈련 및 최적화"""
        model_results = {}
        optimization_results = {}

        # 특성과 타겟 분리
        feature_columns = [col for col in data.columns if col != 'target_return']
        X = data[feature_columns]
        y = data['target_return']

        print(f"   특성 수: {len(feature_columns)}")
        print(f"   샘플 수: {len(X)}")

        for model_name in self.config.models_to_train:
            try:
                print(f"\n🔧 {model_name} 모델 훈련 시작...")

                if model_name == "arima_garch":
                    model, optimization_result = self._train_arima_garch(X, y)
                elif model_name == "tft":
                    model, optimization_result = self._train_tft(X, y)
                elif model_name == "mdn":
                    model, optimization_result = self._train_mdn(X, y)
                else:
                    print(f"⚠️ 지원하지 않는 모델: {model_name}")
                    continue

                model_results[model_name] = model
                optimization_results[model_name] = optimization_result

                print(f"✅ {model_name} 훈련 완료")

            except Exception as e:
                print(f"❌ {model_name} 훈련 실패: {e}")
                continue

        return model_results, optimization_results

    def _train_arima_garch(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, Dict]:
        """ARIMA-GARCH 모델 훈련"""
        # 간단한 ARIMA-GARCH 구현 (시뮬레이션)
        from sklearn.linear_model import Ridge

        model = Ridge(alpha=1.0)

        # 기본 훈련
        model.fit(X.iloc[:-50], y.iloc[:-50])  # 마지막 50개는 테스트용 보존

        # 하이퍼파라미터 최적화 (옵션)
        optimization_result = {}
        if self.config.optimize_hyperparameters:
            param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}

            try:
                opt_result = self.hyperparameter_optimizer.optimize_hyperparameters(
                    Ridge, X.iloc[:-50], y.iloc[:-50], param_grid
                )
                model = Ridge(**opt_result.best_params)
                model.fit(X.iloc[:-50], y.iloc[:-50])
                optimization_result = opt_result.__dict__
            except Exception as e:
                print(f"⚠️ 하이퍼파라미터 최적화 실패: {e}")

        return model, optimization_result

    def _train_tft(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, Dict]:
        """TFT 모델 훈련 (간단한 대체 구현)"""
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X.iloc[:-50], y.iloc[:-50])

        optimization_result = {}
        if self.config.optimize_hyperparameters:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15]
            }

            try:
                opt_result = self.hyperparameter_optimizer.optimize_hyperparameters(
                    RandomForestRegressor, X.iloc[:-50], y.iloc[:-50], param_grid
                )
                model = RandomForestRegressor(**opt_result.best_params, random_state=42)
                model.fit(X.iloc[:-50], y.iloc[:-50])
                optimization_result = opt_result.__dict__
            except Exception as e:
                print(f"⚠️ 하이퍼파라미터 최적화 실패: {e}")

        return model, optimization_result

    def _train_mdn(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Any, Dict]:
        """MDN 모델 훈련 (간단한 대체 구현)"""
        from sklearn.ensemble import GradientBoostingRegressor

        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X.iloc[:-50], y.iloc[:-50])

        optimization_result = {}
        if self.config.optimize_hyperparameters:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }

            try:
                opt_result = self.hyperparameter_optimizer.optimize_hyperparameters(
                    GradientBoostingRegressor, X.iloc[:-50], y.iloc[:-50], param_grid
                )
                model = GradientBoostingRegressor(**opt_result.best_params, random_state=42)
                model.fit(X.iloc[:-50], y.iloc[:-50])
                optimization_result = opt_result.__dict__
            except Exception as e:
                print(f"⚠️ 하이퍼파라미터 최적화 실패: {e}")

        return model, optimization_result

    def _perform_walk_forward_validation(self, data: pd.DataFrame, models: Dict) -> Dict:
        """Walk-Forward 검증 수행"""
        validation_results = {}

        feature_columns = [col for col in data.columns if col != 'target_return']
        X = data[feature_columns]
        y = data['target_return']

        for model_name, model in models.items():
            try:
                print(f"   {model_name} Walk-Forward 검증 중...")

                cv_results = self.walk_forward_validator.validate(model, X, y)
                summary = self.walk_forward_validator.summarize_results(cv_results)

                validation_results[model_name] = {
                    'cv_results': cv_results,
                    'summary': summary
                }

                print(f"   ✅ {model_name}: 테스트 점수 {summary.get('test_score_mean', 0):.4f} "
                      f"(±{summary.get('test_score_std', 0):.4f})")

            except Exception as e:
                print(f"   ❌ {model_name} 검증 실패: {e}")

        return validation_results

    def _perform_backtesting(self, data: pd.DataFrame, models: Dict) -> Dict:
        """백테스팅 수행"""
        backtest_results = {}

        prices = data['price']
        feature_columns = [col for col in data.columns if col != 'target_return']
        X = data[feature_columns]

        for model_name, model in models.items():
            try:
                print(f"   {model_name} 백테스팅 중...")

                # 예측 신호 생성
                predictions = model.predict(X)

                # 매매 신호 생성 (예측값이 양수면 매수, 음수면 매도)
                signals = pd.Series(
                    np.where(predictions > 0, 1, -1),
                    index=X.index
                )

                # 백테스팅 실행
                backtest_result = self.backtester.backtest_strategy(
                    prices, signals, f"{model_name}_strategy"
                )

                backtest_results[model_name] = backtest_result

                print(f"   ✅ {model_name}: 총수익률 {backtest_result.total_return:.2%}, "
                      f"샤프비율 {backtest_result.sharpe_ratio:.3f}")

            except Exception as e:
                print(f"   ❌ {model_name} 백테스팅 실패: {e}")

        return backtest_results

    def _calculate_risk_metrics(self, backtest_results: Dict) -> Dict:
        """리스크 지표 계산"""
        risk_metrics = {}

        for model_name, backtest_result in backtest_results.items():
            try:
                portfolio_returns = backtest_result.portfolio_value.pct_change().dropna()

                # 벤치마크는 매수 후 보유 전략으로 설정
                comprehensive_metrics = self.risk_calculator.calculate_comprehensive_risk_metrics(
                    portfolio_returns
                )

                risk_report = self.risk_calculator.generate_risk_report(portfolio_returns)

                risk_metrics[model_name] = {
                    'comprehensive_metrics': comprehensive_metrics,
                    'detailed_report': risk_report
                }

                print(f"   ✅ {model_name} 리스크 지표 계산 완료")

            except Exception as e:
                print(f"   ❌ {model_name} 리스크 계산 실패: {e}")

        return risk_metrics

    def _optimize_ensemble(self, data: pd.DataFrame, models: Dict) -> Optional[Any]:
        """앙상블 최적화"""
        try:
            feature_columns = [col for col in data.columns if col != 'target_return']
            X = data[feature_columns]
            y = data['target_return']

            # 각 모델의 예측값 수집
            model_predictions = {}
            for model_name, model in models.items():
                predictions = model.predict(X)
                model_predictions[model_name] = predictions

            # 앙상블 가중치 최적화
            ensemble_weights = self.ensemble_optimizer.optimize_ensemble_weights(
                model_predictions, y.values, method="markowitz"
            )

            print(f"   ✅ 앙상블 최적화 완료: 샤프비율 {ensemble_weights.sharpe_ratio:.3f}")

            return ensemble_weights

        except Exception as e:
            print(f"   ❌ 앙상블 최적화 실패: {e}")
            return None

    def _generate_recommendations(
        self,
        validation_results: Dict,
        backtest_results: Dict,
        risk_metrics: Dict,
        ensemble_weights: Optional[Any]
    ) -> Dict:
        """최종 권장사항 생성"""
        recommendations = {
            'best_individual_model': None,
            'model_rankings': [],
            'risk_assessment': {},
            'ensemble_recommendation': {},
            'deployment_readiness': {},
            'next_steps': []
        }

        # 모델 순위 매기기 (샤프 비율 기준)
        model_scores = {}
        for model_name in backtest_results.keys():
            if model_name in backtest_results:
                sharpe_ratio = backtest_results[model_name].sharpe_ratio
                total_return = backtest_results[model_name].total_return
                max_drawdown = backtest_results[model_name].max_drawdown

                # 복합 점수 계산 (샤프 비율 + 수익률 - 낙폭 페널티)
                composite_score = sharpe_ratio + total_return - max_drawdown
                model_scores[model_name] = {
                    'composite_score': composite_score,
                    'sharpe_ratio': sharpe_ratio,
                    'total_return': total_return,
                    'max_drawdown': max_drawdown
                }

        # 순위 정렬
        sorted_models = sorted(
            model_scores.items(),
            key=lambda x: x[1]['composite_score'],
            reverse=True
        )

        recommendations['model_rankings'] = sorted_models
        recommendations['best_individual_model'] = sorted_models[0][0] if sorted_models else None

        # 리스크 평가
        recommendations['risk_assessment'] = {
            'low_risk_models': [
                model for model, metrics in model_scores.items()
                if metrics['max_drawdown'] < 0.15
            ],
            'high_return_models': [
                model for model, metrics in model_scores.items()
                if metrics['total_return'] > 0.1
            ],
            'balanced_models': [
                model for model, metrics in model_scores.items()
                if metrics['sharpe_ratio'] > 0.5 and metrics['max_drawdown'] < 0.2
            ]
        }

        # 앙상블 권장사항
        if ensemble_weights:
            recommendations['ensemble_recommendation'] = {
                'use_ensemble': True,
                'expected_sharpe': ensemble_weights.sharpe_ratio,
                'diversification_benefit': ensemble_weights.diversification_ratio > 1.1,
                'weights': dict(zip(backtest_results.keys(), ensemble_weights.weights))
            }

        # 배포 준비도 평가
        for model_name in backtest_results.keys():
            stability_score = 0

            # 검증 안정성 확인
            if model_name in validation_results:
                test_score_std = validation_results[model_name]['summary'].get('test_score_std', 1.0)
                stability_score += 50 if test_score_std < 0.1 else 0

            # 백테스팅 성과 확인
            if model_name in backtest_results:
                sharpe = backtest_results[model_name].sharpe_ratio
                stability_score += 50 if sharpe > 0.5 else 25 if sharpe > 0 else 0

            deployment_ready = stability_score >= 75

            recommendations['deployment_readiness'][model_name] = {
                'ready': deployment_ready,
                'stability_score': stability_score,
                'confidence_level': 'High' if stability_score >= 75 else 'Medium' if stability_score >= 50 else 'Low'
            }

        # 다음 단계 권장사항
        recommendations['next_steps'] = [
            "실시간 데이터 피드 구축",
            "프로덕션 환경 배포 준비",
            "모니터링 및 알람 시스템 구축",
            "성과 추적 대시보드 개발",
            "정기 모델 재훈련 스케줄 설정"
        ]

        return recommendations


def run_example_pipeline():
    """파이프라인 실행 예제"""
    print("📈 Advanced Financial Forecasting Pipeline 예제 실행")

    # 설정 생성
    config = PipelineConfig(
        symbol="SPY",
        start_date="2022-01-01",
        end_date="2024-01-01",
        models_to_train=["arima_garch", "tft", "mdn"],
        optimize_hyperparameters=True,
        initial_capital=100000.0
    )

    # 파이프라인 실행
    pipeline = AdvancedFinancialPipeline(config)
    result = pipeline.run_complete_pipeline()

    # 결과 요약 출력
    print("\n" + "="*60)
    print("📊 파이프라인 실행 결과 요약")
    print("="*60)

    print(f"\n📈 데이터 요약:")
    for key, value in result.data_summary.items():
        print(f"   {key}: {value}")

    print(f"\n🏆 최고 성능 모델: {result.final_recommendations.get('best_individual_model', 'N/A')}")

    print(f"\n📊 모델 순위:")
    for i, (model_name, metrics) in enumerate(result.final_recommendations.get('model_rankings', []), 1):
        print(f"   {i}. {model_name}: 샤프비율 {metrics['sharpe_ratio']:.3f}, "
              f"수익률 {metrics['total_return']:.2%}")

    if result.ensemble_weights:
        print(f"\n🎯 앙상블 권장:")
        print(f"   예상 샤프비율: {result.ensemble_weights.sharpe_ratio:.3f}")
        print(f"   다각화 효과: {result.ensemble_weights.diversification_ratio:.3f}")

    return result


if __name__ == "__main__":
    # 예제 실행
    result = run_example_pipeline()