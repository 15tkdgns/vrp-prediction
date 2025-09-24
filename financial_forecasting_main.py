#!/usr/bin/env python3
"""
Advanced Financial Forecasting System - Main Execution Script

근본적 접근법 변경을 구현한 완전한 금융 예측 시스템:

🔄 패러다임 변경:
- 가격 예측 → 로그 수익률 예측 (통계적 정상성)
- MSE 최적화 → 금융 성과 지표 최적화 (샤프/소르티노/MDD)
- 단순 검증 → Walk-Forward Validation (데이터 누출 방지)

🧠 고급 모델링:
- 계량경제학: ARIMA-GARCH (변동성 군집 모델링)
- 딥러닝: TFT, MDN (시퀀스 모델링 & 불확실성 정량화)
- 대체 데이터: FRED, FinBERT, HMM 레짐 감지

⚖️ 리스크 관리:
- 포트폴리오 이론 기반 앙상블 최적화
- VaR/CVaR, 베타, 추적오차 등 종합 리스크 지표
- 실제 거래 전략 백테스팅

사용법:
    python financial_forecasting_main.py --symbol SPY --start-date 2020-01-01
"""

import sys
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 루트를 파이썬 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from src.financial_forecasting.pipeline import AdvancedFinancialPipeline, PipelineConfig
    from src.financial_forecasting.core import FinancialMetrics
    print("✅ 모든 모듈 가져오기 성공")
except ImportError as e:
    print(f"❌ 모듈 가져오기 실패: {e}")
    print("필요한 패키지를 설치하세요: pip install -r requirements/base.txt")
    sys.exit(1)


def create_argument_parser():
    """명령행 인수 파서 생성"""
    parser = argparse.ArgumentParser(
        description="Advanced Financial Forecasting System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제 사용법:
  # SPY 기본 분석 (2020-2024)
  python financial_forecasting_main.py --symbol SPY

  # QQQ 1년 분석, 하이퍼파라미터 최적화 스킵
  python financial_forecasting_main.py --symbol QQQ --start-date 2023-01-01 --no-optimization

  # 모든 모델 사용, 고급 설정
  python financial_forecasting_main.py --symbol TSLA --models arima_garch tft mdn --capital 500000

참고:
  이 시스템은 근본적 접근법 변경을 구현합니다:
  - 가격 → 로그 수익률 예측 (통계적 정상성 확보)
  - 전통적 ML 지표 → 금융 성과 지표 최적화 (샤프/소르티노/MDD)
  - 시점간 검증 → Walk-Forward Validation (데이터 누출 방지)
        """
    )

    # 기본 매개변수
    parser.add_argument(
        '--symbol', '-s',
        type=str,
        default='SPY',
        help='분석할 심볼 (기본값: SPY)'
    )

    parser.add_argument(
        '--start-date',
        type=str,
        default='2020-01-01',
        help='분석 시작일 (YYYY-MM-DD 형식, 기본값: 2020-01-01)'
    )

    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='분석 종료일 (YYYY-MM-DD 형식, 기본값: 현재)'
    )

    # 모델 설정
    parser.add_argument(
        '--models',
        nargs='+',
        choices=['arima_garch', 'tft', 'mdn'],
        default=['arima_garch', 'tft', 'mdn'],
        help='사용할 모델들 (기본값: 모든 모델)'
    )

    parser.add_argument(
        '--no-optimization',
        action='store_true',
        help='하이퍼파라미터 최적화 스킵 (빠른 실행)'
    )

    parser.add_argument(
        '--optimization-method',
        choices=['grid_search', 'differential_evolution'],
        default='grid_search',
        help='최적화 방법 (기본값: grid_search)'
    )

    # 검증 설정
    parser.add_argument(
        '--train-size',
        type=int,
        default=252,
        help='초기 훈련 크기 (일수, 기본값: 252 = 1년)'
    )

    parser.add_argument(
        '--test-size',
        type=int,
        default=21,
        help='테스트 크기 (일수, 기본값: 21 = 1개월)'
    )

    parser.add_argument(
        '--walk-forward-steps',
        type=int,
        default=21,
        help='Walk-Forward 재훈련 주기 (일수, 기본값: 21)'
    )

    # 백테스팅 설정
    parser.add_argument(
        '--capital',
        type=float,
        default=100000.0,
        help='초기 자본 (USD, 기본값: 100,000)'
    )

    parser.add_argument(
        '--transaction-cost',
        type=float,
        default=0.001,
        help='거래 비용 (비율, 기본값: 0.001 = 0.1%%)'
    )

    # API 키 (선택적)
    parser.add_argument(
        '--fred-api-key',
        type=str,
        default=None,
        help='FRED API 키 (거시경제 지표용, 선택적)'
    )

    parser.add_argument(
        '--news-api-key',
        type=str,
        default=None,
        help='News API 키 (센티멘트 분석용, 선택적)'
    )

    # 출력 설정
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='결과 저장 디렉터리 (기본값: results)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='상세 출력 모드'
    )

    return parser


def print_system_introduction():
    """시스템 소개 출력"""
    print("=" * 80)
    print("🚀 Advanced Financial Forecasting System")
    print("   근본적 접근법 변경 (Fundamental Paradigm Shift)")
    print("=" * 80)
    print()
    print("📊 패러다임 변경:")
    print("   ❌ 기존: 가격 예측 + MSE 최적화 + 일반 교차검증")
    print("   ✅ 새로운: 로그 수익률 예측 + 금융 성과 최적화 + Walk-Forward 검증")
    print()
    print("🧠 고급 모델링 기법:")
    print("   • 계량경제학: ARIMA-GARCH (변동성 군집 효과)")
    print("   • 딥러닝: Temporal Fusion Transformer, Mixture Density Networks")
    print("   • 대체 데이터: FRED 거시지표, FinBERT 센티멘트, HMM 레짐 감지")
    print()
    print("⚖️ 포괄적 리스크 관리:")
    print("   • 포트폴리오 이론 기반 앙상블 최적화")
    print("   • VaR/CVaR, 베타, 추적오차 등 종합 리스크 지표")
    print("   • 실제 거래 전략 백테스팅 및 성과 평가")
    print()
    print("🛡️ 데이터 무결성 보장:")
    print("   • 시계열 안전 검증 (데이터 누출 방지)")
    print("   • 통계적 정상성 확보 (ADF 검정)")
    print("   • Purged Walk-Forward Validation")
    print()


def validate_arguments(args):
    """입력 인수 검증"""
    errors = []

    # 날짜 형식 검증
    try:
        from datetime import datetime
        datetime.strptime(args.start_date, '%Y-%m-%d')
        if args.end_date:
            datetime.strptime(args.end_date, '%Y-%m-%d')
    except ValueError:
        errors.append("날짜는 YYYY-MM-DD 형식이어야 합니다")

    # 수치 검증
    if args.capital <= 0:
        errors.append("초기 자본은 양수여야 합니다")

    if args.transaction_cost < 0 or args.transaction_cost > 0.1:
        errors.append("거래 비용은 0과 0.1 사이여야 합니다")

    if args.train_size < 50:
        errors.append("훈련 크기는 최소 50일 이상이어야 합니다")

    if args.test_size < 1:
        errors.append("테스트 크기는 최소 1일 이상이어야 합니다")

    if errors:
        print("❌ 입력 오류:")
        for error in errors:
            print(f"   • {error}")
        return False

    return True


def save_results(result, output_dir: str, symbol: str):
    """결과 저장"""
    import json
    from datetime import datetime

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 주요 결과를 JSON으로 저장
    summary = {
        'execution_info': {
            'symbol': symbol,
            'timestamp': timestamp,
            'data_summary': result.data_summary
        },
        'model_performance': {},
        'final_recommendations': result.final_recommendations
    }

    # 백테스팅 결과 요약
    for model_name, backtest_result in result.backtest_results.items():
        summary['model_performance'][model_name] = {
            'total_return': backtest_result.total_return,
            'annual_return': backtest_result.annual_return,
            'sharpe_ratio': backtest_result.sharpe_ratio,
            'sortino_ratio': backtest_result.sortino_ratio,
            'max_drawdown': backtest_result.max_drawdown,
            'win_rate': backtest_result.win_rate,
            'trade_count': backtest_result.trade_count
        }

    # JSON 파일 저장
    json_file = output_path / f'{symbol}_forecast_results_{timestamp}.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"📁 결과가 저장되었습니다: {json_file}")

    # 추가로 텍스트 리포트 생성
    report_file = output_path / f'{symbol}_forecast_report_{timestamp}.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("Advanced Financial Forecasting System - Report\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Symbol: {symbol}\n")
        f.write(f"Analysis Period: {result.data_summary.get('period', 'N/A')}\n")
        f.write(f"Observations: {result.data_summary.get('observations', 'N/A')}\n\n")

        f.write("Model Performance Summary:\n")
        f.write("-" * 30 + "\n")

        rankings = result.final_recommendations.get('model_rankings', [])
        for i, (model_name, metrics) in enumerate(rankings, 1):
            f.write(f"{i}. {model_name}:\n")
            f.write(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.3f}\n")
            f.write(f"   Total Return: {metrics['total_return']:.2%}\n")
            f.write(f"   Max Drawdown: {metrics['max_drawdown']:.2%}\n\n")

        best_model = result.final_recommendations.get('best_individual_model')
        if best_model:
            f.write(f"Recommended Model: {best_model}\n\n")

        if result.ensemble_weights:
            f.write("Ensemble Recommendation:\n")
            f.write(f"   Expected Sharpe Ratio: {result.ensemble_weights.sharpe_ratio:.3f}\n")
            f.write(f"   Diversification Ratio: {result.ensemble_weights.diversification_ratio:.3f}\n\n")

        f.write("Next Steps:\n")
        for step in result.final_recommendations.get('next_steps', []):
            f.write(f"   • {step}\n")

    print(f"📄 리포트가 저장되었습니다: {report_file}")


def main():
    """메인 함수"""
    # 시스템 소개
    print_system_introduction()

    # 명령행 인수 파싱
    parser = create_argument_parser()
    args = parser.parse_args()

    # 인수 검증
    if not validate_arguments(args):
        sys.exit(1)

    print(f"🎯 분석 시작: {args.symbol} ({args.start_date} ~ {args.end_date or '현재'})")
    print(f"📊 사용 모델: {', '.join(args.models)}")
    print(f"💰 초기 자본: ${args.capital:,.0f}")
    print(f"⚙️ 하이퍼파라미터 최적화: {'예' if not args.no_optimization else '아니오'}")
    print()

    try:
        # 파이프라인 설정 생성
        config = PipelineConfig(
            symbol=args.symbol,
            start_date=args.start_date,
            end_date=args.end_date,
            models_to_train=args.models,
            train_size=args.train_size,
            test_size=args.test_size,
            walk_forward_steps=args.walk_forward_steps,
            optimize_hyperparameters=not args.no_optimization,
            optimization_method=args.optimization_method,
            initial_capital=args.capital,
            transaction_cost=args.transaction_cost,
            fred_api_key=args.fred_api_key,
            news_api_key=args.news_api_key
        )

        # 파이프라인 실행
        pipeline = AdvancedFinancialPipeline(config)
        result = pipeline.run_complete_pipeline()

        # 결과 저장
        save_results(result, args.output_dir, args.symbol)

        # 최종 요약 출력
        print("\n" + "="*80)
        print("🎉 분석 완료! 주요 결과:")
        print("="*80)

        best_model = result.final_recommendations.get('best_individual_model')
        if best_model and best_model in result.backtest_results:
            backtest = result.backtest_results[best_model]
            print(f"🏆 최고 성능 모델: {best_model}")
            print(f"   💹 총 수익률: {backtest.total_return:.2%}")
            print(f"   📈 연 수익률: {backtest.annual_return:.2%}")
            print(f"   ⚡ 샤프 비율: {backtest.sharpe_ratio:.3f}")
            print(f"   📉 최대 낙폭: {backtest.max_drawdown:.2%}")
            print(f"   🎯 승률: {backtest.win_rate:.1%}")

        if result.ensemble_weights:
            print(f"\n🎯 앙상블 권장사항:")
            print(f"   📊 예상 샤프 비율: {result.ensemble_weights.sharpe_ratio:.3f}")
            print(f"   🔄 다각화 효과: {result.ensemble_weights.diversification_ratio:.3f}")

        deployment_ready = [
            model for model, info in result.final_recommendations.get('deployment_readiness', {}).items()
            if info.get('ready', False)
        ]

        if deployment_ready:
            print(f"\n🚀 배포 준비 완료 모델: {', '.join(deployment_ready)}")
        else:
            print(f"\n⚠️ 추가 최적화가 필요합니다")

        print(f"\n📁 상세 결과는 '{args.output_dir}' 디렉터리에서 확인하세요")

    except KeyboardInterrupt:
        print("\n\n⏹️ 사용자에 의해 중단되었습니다")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ 실행 중 오류 발생: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()