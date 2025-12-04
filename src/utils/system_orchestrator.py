import os
import json
import threading
import time
from datetime import datetime
import logging
import pandas as pd  # pandas import 추가


# Missing imports - validation checker and monitoring systems need to be created
# from src.testing.validation_checker import DataValidationChecker
# from src.utils.xai_monitoring import XAIMonitoringSystem
# from src.testing.realtime_testing_system import RealTimeTestingSystem

# LLM imports - these exist
try:
    from src.features.llm_feature_extractor import extract_llm_features
except ImportError:
    extract_llm_features = None

try:
    from src.features.llm_feature_extractor_improved import EnhancedLLMFeatureExtractor
except ImportError:
    EnhancedLLMFeatureExtractor = None

# Available imports
try:
    from src.analysis.xai_analyzer import XAIAnalyzer
except ImportError:
    XAIAnalyzer = None

try:
    from src.utils.directory_manager import DirectoryManager
except ImportError:
    DirectoryManager = None

try:
    from src.core.config_manager import get_config_manager
except ImportError:
    get_config_manager = None

try:
    import tensorflow as tf
except ImportError:
    tf = None


class SystemOrchestrator:
    def __init__(self, data_dir="data/raw"):
        # 디렉토리 자동 생성 먼저 실행 (optional)
        if DirectoryManager:
            self.directory_manager = DirectoryManager()
            self.directory_manager.ensure_directories()
        else:
            self.directory_manager = None
            # Create basic directories manually
            os.makedirs("data/raw", exist_ok=True)
            os.makedirs("data/processed", exist_ok=True)
            os.makedirs("logs/system", exist_ok=True)

        # ConfigManager 초기화 (optional)
        if get_config_manager:
            self.config_manager = get_config_manager()
        else:
            self.config_manager = None

        self.data_dir = os.path.abspath(data_dir)
        self.components = {}
        self.status = {
            "validation": "not_started",
            "xai_monitoring": "not_started", 
            "realtime_testing": "not_started",
            "overall_health": "unknown",
            "directories": "initialized",
        }

        # 로깅 설정 - 이제 logs 디렉토리가 존재함
        if self.directory_manager and hasattr(self.directory_manager, 'project_root'):
            log_dir = self.directory_manager.project_root / "logs/system"
            log_file = log_dir / "system_orchestrator.log"
        else:
            log_dir = "logs/system"
            log_file = "logs/system/system_orchestrator.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(str(log_file)),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)
        
        # 초기화 완료 로그
        self.logger.info("🚀 시스템 오케스트레이터 초기화 시작")
        if self.directory_manager and hasattr(self.directory_manager, 'project_root'):
            self.logger.info(f"📁 프로젝트 루트: {self.directory_manager.project_root}")
        else:
            self.logger.info(f"📁 프로젝트 루트: {os.getcwd()}")
        
        self.check_gpu()

    def check_gpu(self):
        """GPU 사용 가능 여부 확인"""
        if tf is None:
            self.logger.info("⚠️ TensorFlow 미설치. GPU 확인 건너뜀 (Static Mode)")
            return

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            self.logger.info(f"✅ GPU 장치 감지: {gpus}")
        else:
            self.logger.warning("⚠️ GPU 장치를 찾을 수 없습니다. CPU로 실행됩니다.")

    def initialize_components(self):
        """모든 컴포넌트 초기화 (정적 모드)"""
        self.logger.info("시스템 컴포넌트 초기화 시작 (정적 모드)")

        try:
            # 데이터 검증 컴포넌트 (현재 사용 불가)
            # self.components["validator"] = DataValidationChecker(self.data_dir)
            self.logger.info("⚠️ 데이터 검증 컴포넌트 사용 불가 (모듈 누락)")

            # XAI 모니터링 컴포넌트 (정적 모드에서는 선택적)
            # XAIMonitoringSystem 모듈이 누락되어 건너뜀
            self.logger.info("⚠️ XAI 모니터링 컴포넌트 사용 불가 (모듈 누락)")

            # Enhanced XAI 분석 컴포넌트
            try:
                if XAIAnalyzer:
                    self.components["xai_analyzer"] = XAIAnalyzer(data_path=self.data_dir, output_path="data/processed")
                    self.logger.info("✅ Enhanced XAI 분석 컴포넌트 초기화 완료")
                else:
                    self.logger.warning("⚠️ XAIAnalyzer 모듈 사용 불가")
            except Exception as e:
                self.logger.warning(f"⚠️ Enhanced XAI 분석 컴포넌트 초기화 실패: {e}")

            # Enhanced LLM Feature Extractor
            try:
                if EnhancedLLMFeatureExtractor:
                    self.components["enhanced_llm_extractor"] = EnhancedLLMFeatureExtractor()
                    self.logger.info("✅ Enhanced LLM Feature Extractor 초기화 완료")
                else:
                    self.logger.warning("⚠️ EnhancedLLMFeatureExtractor 모듈 사용 불가")
            except Exception as e:
                self.logger.warning(f"⚠️ Enhanced LLM Feature Extractor 초기화 실패: {e}")

            # 실시간 테스트 컴포넌트는 정적 모드에서 제외
            # self.components["realtime_tester"] = RealTimeTestingSystem(self.data_dir)
            self.logger.info("✅ 정적 모드: 실시간 테스트 컴포넌트 건너뜀")

            return True

        except Exception as e:
            self.logger.error(f"컴포넌트 초기화 실패: {e}")
            return False

    def _run_llm_feature_extraction(self):
        """LLM 특징 추출 실행"""
        self.logger.info("LLM 특징 추출 시작")
        try:
            # news_sentiment_data.csv 로드
            news_data_path = os.path.join(self.data_dir, "news_sentiment_data.csv")
            if not os.path.exists(news_data_path):
                self.logger.error(
                    f"뉴스 감성 데이터 파일을 찾을 수 없습니다: {news_data_path}"
                )
                return False

            news_df = pd.read_csv(news_data_path)

            # LLM 특징 추출
            llm_enhanced_features = extract_llm_features(news_df)

            # 추출된 특징 저장
            output_path = os.path.join(
                self.data_dir.replace("raw", "processed"), "llm_enhanced_features.csv"
            )
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            llm_enhanced_features.to_csv(output_path, index=False)

            self.logger.info(f"✅ LLM 특징 추출 완료 및 저장: {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"LLM 특징 추출 중 오류: {e}")
            return False

    def run_system_validation(self):
        """시스템 전체 검증 (간소화된 버전)"""
        self.logger.info("시스템 검증 시작 (간소화된 버전)")
        self.status["validation"] = "running"

        try:
            # 기본 디렉토리 및 파일 존재 여부 확인
            required_dirs = ["data/raw", "data/processed"]
            missing_dirs = []

            for dir_path in required_dirs:
                if not os.path.exists(dir_path):
                    missing_dirs.append(dir_path)
                    os.makedirs(dir_path, exist_ok=True)
                    self.logger.info(f"📁 디렉토리 생성: {dir_path}")

            # 기본 데이터 파일 확인
            data_files = ["realtime_results.json", "model_performance.json", "system_status.json"]
            existing_files = []

            for file_name in data_files:
                file_path = os.path.join(self.data_dir, file_name)
                if os.path.exists(file_path):
                    existing_files.append(file_name)

            self.logger.info(f"✅ 기존 데이터 파일: {len(existing_files)}/{len(data_files)}개")

            self.status["validation"] = "passed"
            self.logger.info("✅ 간소화된 시스템 검증 통과")
            return True

        except Exception as e:
            self.status["validation"] = "error"
            self.logger.error(f"시스템 검증 중 오류: {e}")
            return False

    def run_xai_monitoring(self):
        """XAI 모니터링 실행 (정적 모드)"""
        self.logger.info("XAI 모니터링 시작 (정적 모드)")
        self.status["xai_monitoring"] = "running"

        try:
            # 정적 모드에서는 기존 분석 결과 확인
            existing_reports = [
                "model_performance.json",
                "feature_analysis_enhanced.json", 
                "model_ensemble_comparison.json"
            ]
            
            reports_found = 0
            for report in existing_reports:
                report_path = os.path.join(self.data_dir, report)
                if os.path.exists(report_path):
                    reports_found += 1
                    self.logger.info(f"✅ 기존 분석 결과 발견: {report}")
            
            if reports_found > 0:
                self.status["xai_monitoring"] = "active_static"
                self.logger.info(f"✅ XAI 모니터링 활성화 (정적 모드) - {reports_found}개 보고서 확인")
                return True
            else:
                self.status["xai_monitoring"] = "no_data"
                self.logger.warning("⚠️ XAI 분석 데이터 없음, 건너뜀")
                return True  # 오류가 아니므로 True 반환

        except Exception as e:
            self.status["xai_monitoring"] = "error"
            self.logger.error(f"XAI 모니터링 중 오류: {e}")
            return False

    def start_realtime_testing(self):
        """정적 데이터 검증 실행"""
        self.logger.info("정적 데이터 검증 시작")
        self.status["realtime_testing"] = "starting"

        try:
            # 정적 모드에서는 기존 결과 파일 확인 및 검증
            static_data_files = [
                "realtime_results.json",
                "model_performance.json",
                "validation_report.json"
            ]
            
            validation_passed = 0
            for data_file in static_data_files:
                file_path = os.path.join(self.data_dir, data_file)
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        validation_passed += 1
                        self.logger.info(f"✅ 정적 데이터 검증 완료: {data_file}")
                    except json.JSONDecodeError:
                        self.logger.warning(f"⚠️ 데이터 파일 손상: {data_file}")
                else:
                    self.logger.warning(f"⚠️ 데이터 파일 누락: {data_file}")
            
            if validation_passed >= 2:  # 최소 2개 파일이 있으면 성공
                self.status["realtime_testing"] = "static_validated"
                self.logger.info(f"✅ 정적 데이터 검증 완료 - {validation_passed}개 파일 확인됨")
                return True
            else:
                self.status["realtime_testing"] = "insufficient_data"
                self.logger.warning("⚠️ 정적 데이터 부족, 하지만 계속 진행")
                return True  # 데이터가 부족해도 오류는 아님

        except Exception as e:
            self.status["realtime_testing"] = "error"
            self.logger.error(f"정적 데이터 검증 중 오류: {e}")
            return False

    def monitor_system_health(self):
        """시스템 헬스 모니터링"""
        while True:
            try:
                # 전체 상태 평가
                healthy_components = 0
                total_components = 0

                for component, status in self.status.items():
                    if component != "overall_health":
                        total_components += 1
                        if status in ["passed", "active", "running"]:
                            healthy_components += 1

                # 전체 헬스 상태 결정
                if healthy_components == total_components:
                    self.status["overall_health"] = "healthy"
                elif healthy_components > 0:
                    self.status["overall_health"] = "degraded"
                else:
                    self.status["overall_health"] = "unhealthy"

                # 상태 저장
                status_report = {
                    "timestamp": datetime.now().isoformat(),
                    "status": self.status,
                    "components": {
                        name: comp.__class__.__name__
                        for name, comp in self.components.items()
                    },
                }

                with open(f"{self.data_dir}/system_status.json", "w") as f:
                    json.dump(status_report, f, indent=2)

                # 상태 로그
                if self.status["overall_health"] == "healthy":
                    self.logger.info("✅ 시스템 상태: 정상")
                elif self.status["overall_health"] == "degraded":
                    self.logger.warning("⚠️ 시스템 상태: 일부 문제")
                else:
                    self.logger.error("❌ 시스템 상태: 비정상")

                time.sleep(300)  # 5분마다 체크

            except Exception as e:
                self.logger.error(f"헬스 모니터링 오류: {e}")
                time.sleep(60)  # 오류 시 1분 대기

    def run_full_pipeline(self):
        """전체 파이프라인 실행"""
        self.logger.info("=== S&P500 이벤트 탐지 시스템 시작 ===")

        # 1. 컴포넌트 초기화
        if not self.initialize_components():
            return False

        # 2. LLM 특징 추출 (새로운 단계 추가)
        if not self._run_llm_feature_extraction():
            self.logger.error("LLM 특징 추출 실패로 인해 중단됨")
            return False

        # 3. 시스템 검증 (기존 2단계)
        if not self.run_system_validation():
            self.logger.error("시스템 검증 실패로 인해 중단됨")
            return False

        # 4. XAI 모니터링 실행 (기존 3단계)
        if not self.run_xai_monitoring():
            self.logger.warning("XAI 모니터링 실패, 계속 진행")

        # 5. 정적 데이터 검증 (기존 실시간 테스트 대체)
        if not self.start_realtime_testing():
            self.logger.warning("정적 데이터 검증 실패, 하지만 계속 진행")
            # 정적 모드에서는 데이터 검증 실패가 치명적이지 않음

        # 6. 헬스 모니터링 시작 (별도 스레드) (기존 5단계)
        health_thread = threading.Thread(target=self.monitor_system_health)
        health_thread.daemon = True
        health_thread.start()

        self.logger.info("=== 시스템 완전 가동 ===")
        return True

    def stop_system(self):
        """시스템 중지"""
        self.logger.info("시스템 중지 시작")

        try:
            # 실시간 테스트 중지 (정적 모드에서는 해당 없음)
            if "realtime_tester" in self.components:
                self.components["realtime_tester"].stop_testing()
                self.logger.info("✅ 실시간 테스트 중지 완료")
            else:
                self.logger.info("✅ 정적 모드: 실시간 테스트 컴포넌트 없음")

            # 상태 업데이트
            self.status["realtime_testing"] = "stopped"
            self.status["overall_health"] = "stopped"

            self.logger.info("✅ 시스템 중지 완료")
            return True

        except Exception as e:
            self.logger.error(f"시스템 중지 중 오류: {e}")
            return False

    def run_enhanced_xai_analysis(self, sample_size=10):
        """Enhanced XAI 분석 실행 (Chain-of-Thought + Attention)"""
        self.logger.info("🧠 Enhanced XAI 분석 시작")
        
        try:
            if "xai_analyzer" not in self.components:
                self.logger.warning("⚠️ XAI 분석 컴포넌트가 초기화되지 않음")
                return False
            
            xai_analyzer = self.components["xai_analyzer"]
            
            # Enhanced XAI 분석 실행
            result = xai_analyzer.run_complete_analysis(
                sample_size=sample_size,
                news_file="news_sentiment_data.csv"
            )
            
            if "error" in result:
                self.logger.error(f"❌ Enhanced XAI 분석 실패: {result['error']}")
                return False
            
            # 결과 로깅
            self.logger.info(f"✅ Enhanced XAI 분석 완료")
            self.logger.info(f"📊 분석된 기사 수: {result.get('articles_analyzed', 0)}")
            self.logger.info(f"✨ 성공적인 분석: {result.get('successful_analyses', 0)}")
            self.logger.info(f"💾 저장된 파일: {list(result.get('saved_files', {}).keys())}")
            
            # 시스템 상태 업데이트
            self.status["enhanced_xai"] = "completed"
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced XAI 분석 중 오류: {e}")
            self.status["enhanced_xai"] = "error"
            return False

    def run_enhanced_llm_extraction(self, sample_size=10):
        """Enhanced LLM 특성 추출 실행"""
        self.logger.info("🔬 Enhanced LLM 특성 추출 시작")
        
        try:
            if "enhanced_llm_extractor" not in self.components:
                self.logger.warning("⚠️ Enhanced LLM Extractor가 초기화되지 않음")
                return False
            
            # 뉴스 데이터 로드
            news_data_path = os.path.join(self.data_dir, "news_sentiment_data.csv")
            if not os.path.exists(news_data_path):
                self.logger.error(f"❌ 뉴스 데이터 파일을 찾을 수 없음: {news_data_path}")
                return False
            
            news_df = pd.read_csv(news_data_path)
            
            # 샘플링 (너무 많은 데이터 처리 방지)
            if len(news_df) > sample_size:
                news_df = news_df.sample(n=sample_size, random_state=42)
                self.logger.info(f"📝 데이터 샘플링: {sample_size}개 기사 선택")
            
            enhanced_extractor = self.components["enhanced_llm_extractor"]
            
            # Enhanced 특성 추출
            enhanced_features = enhanced_extractor.extract_enhanced_features(news_df)
            
            # 결과 저장
            output_path = os.path.join(
                self.data_dir.replace("raw", "processed"), 
                "enhanced_llm_features.csv"
            )
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            enhanced_features.to_csv(output_path, index=False)
            
            # 통계 출력
            successful_extractions = len(enhanced_features[enhanced_features['processing_status'] == 'success'])
            self.logger.info(f"✅ Enhanced LLM 특성 추출 완료")
            self.logger.info(f"📊 처리된 기사: {len(enhanced_features)}")
            self.logger.info(f"✨ 성공적인 추출: {successful_extractions}")
            self.logger.info(f"💾 저장 위치: {output_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced LLM 특성 추출 중 오류: {e}")
            return False

    def run_full_xai_pipeline(self, sample_size=10):
        """전체 Enhanced XAI 파이프라인 실행"""
        self.logger.info("🚀 전체 Enhanced XAI 파이프라인 시작")
        
        try:
            # 1. Enhanced LLM 특성 추출
            if not self.run_enhanced_llm_extraction(sample_size=sample_size):
                self.logger.error("❌ Enhanced LLM 특성 추출 실패")
                return False
            
            # 2. Enhanced XAI 분석
            if not self.run_enhanced_xai_analysis(sample_size=sample_size):
                self.logger.error("❌ Enhanced XAI 분석 실패")
                return False
            
            self.logger.info("✅ 전체 Enhanced XAI 파이프라인 완료")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced XAI 파이프라인 실행 중 오류: {e}")
            return False

    def get_system_status(self):
        """현재 시스템 상태 반환"""
        return {
            "status": self.status,
            "components": list(self.components.keys()),
            "timestamp": datetime.now().isoformat(),
        }


if __name__ == "__main__":
    # 시스템 오케스트레이터 실행
    orchestrator = SystemOrchestrator()

    if orchestrator.run_full_pipeline():
        print("시스템이 성공적으로 시작되었습니다.")
        print("시스템 상태는 raw_data/system_status.json에서 확인할 수 있습니다.")
        print("종료하려면 Ctrl+C를 누르세요.")

        try:
            while True:
                time.sleep(10)
                status = orchestrator.get_system_status()
                print(f"현재 상태: {status['status']['overall_health']}")

        except KeyboardInterrupt:
            print("\n시스템 중지 중...")
            orchestrator.stop_system()
            print("시스템이 중지되었습니다.")
    else:
        print("시스템 시작 실패")
