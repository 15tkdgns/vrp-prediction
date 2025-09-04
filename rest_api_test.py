#!/usr/bin/env python3
"""
종합적인 REST API 테스트 스위트
모든 엔드포인트를 체계적으로 테스트하고 성능 평가
"""

import requests
import json
import time
from datetime import datetime
import concurrent.futures
import statistics

class RESTAPITester:
    def __init__(self, base_url="http://localhost:8091"):
        self.base_url = base_url
        self.results = {}
        
    def test_endpoint(self, endpoint, method="GET", data=None):
        """개별 엔드포인트 테스트"""
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()
        
        try:
            if method == "GET":
                response = requests.get(url, timeout=30)
            elif method == "POST":
                response = requests.post(url, json=data, timeout=30)
            
            response_time = time.time() - start_time
            
            return {
                "endpoint": endpoint,
                "status_code": response.status_code,
                "response_time": round(response_time, 3),
                "success": 200 <= response.status_code < 300,
                "content_type": response.headers.get('Content-Type', ''),
                "response_size": len(response.content),
                "data": response.json() if 'json' in response.headers.get('Content-Type', '') else None
            }
            
        except requests.exceptions.Timeout:
            return {
                "endpoint": endpoint,
                "error": "Timeout",
                "response_time": 30.0,
                "success": False
            }
        except Exception as e:
            return {
                "endpoint": endpoint,
                "error": str(e),
                "response_time": time.time() - start_time,
                "success": False
            }
    
    def performance_test(self, endpoint, iterations=10):
        """성능 테스트 (여러 번 요청)"""
        print(f"🔄 {endpoint} 성능 테스트 ({iterations}회)...")
        results = []
        
        for i in range(iterations):
            result = self.test_endpoint(endpoint)
            results.append(result['response_time'] if 'response_time' in result else 30.0)
            time.sleep(0.1)  # 서버 부하 방지
        
        return {
            "endpoint": endpoint,
            "iterations": iterations,
            "avg_response_time": round(statistics.mean(results), 3),
            "min_response_time": round(min(results), 3),
            "max_response_time": round(max(results), 3),
            "std_dev": round(statistics.stdev(results) if len(results) > 1 else 0, 3)
        }
    
    def concurrent_test(self, endpoint, concurrent_users=5):
        """동시 사용자 테스트"""
        print(f"👥 {endpoint} 동시성 테스트 ({concurrent_users}명)...")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(self.test_endpoint, endpoint) for _ in range(concurrent_users)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        success_count = sum(1 for r in results if r.get('success', False))
        avg_time = statistics.mean([r['response_time'] for r in results if 'response_time' in r])
        
        return {
            "endpoint": endpoint,
            "concurrent_users": concurrent_users,
            "success_rate": round(success_count / concurrent_users * 100, 1),
            "avg_response_time": round(avg_time, 3),
            "results": results
        }
    
    def comprehensive_test(self):
        """종합 테스트 실행"""
        print("🚀 REST API 종합 테스트 시작")
        print("=" * 50)
        
        # 테스트할 엔드포인트들
        endpoints = [
            "/api/status",
            "/api/stocks/live",
            "/api/news/sentiment", 
            "/api/market/volume",
            "/api/models/performance",
            "/api/ml/predict/AAPL",
            "/api/ml/predict/GOOGL",
            "/api/ml/batch_predict"
        ]
        
        # 1. 기본 기능성 테스트
        print("\n1️⃣ 기본 기능성 테스트")
        print("-" * 30)
        
        functionality_results = []
        for endpoint in endpoints:
            result = self.test_endpoint(endpoint)
            status = "✅" if result.get('success') else "❌"
            time_str = f"{result.get('response_time', 'N/A')}s"
            print(f"{status} {endpoint:<25} {time_str:>8} ({result.get('status_code', 'ERR')})")
            functionality_results.append(result)
        
        # 2. 성능 테스트 (주요 엔드포인트만)
        print("\n2️⃣ 성능 테스트")
        print("-" * 30)
        
        performance_endpoints = ["/api/stocks/live", "/api/ml/predict/AAPL", "/api/ml/batch_predict"]
        performance_results = []
        
        for endpoint in performance_endpoints:
            perf_result = self.performance_test(endpoint, iterations=5)
            print(f"📊 {endpoint:<25} 평균: {perf_result['avg_response_time']}s (±{perf_result['std_dev']})")
            performance_results.append(perf_result)
        
        # 3. 동시성 테스트 (가벼운 엔드포인트만)
        print("\n3️⃣ 동시성 테스트")
        print("-" * 30)
        
        concurrency_endpoints = ["/api/status", "/api/models/performance"]
        concurrency_results = []
        
        for endpoint in concurrency_endpoints:
            conc_result = self.concurrent_test(endpoint, concurrent_users=3)
            print(f"👥 {endpoint:<25} 성공률: {conc_result['success_rate']}% 평균: {conc_result['avg_response_time']}s")
            concurrency_results.append(conc_result)
        
        # 4. 데이터 품질 검증
        print("\n4️⃣ 데이터 품질 검증")
        print("-" * 30)
        
        data_quality_results = self.validate_data_quality()
        
        # 결과 종합
        self.results = {
            "test_timestamp": datetime.now().isoformat(),
            "functionality": functionality_results,
            "performance": performance_results,
            "concurrency": concurrency_results,
            "data_quality": data_quality_results
        }
        
        return self.generate_report()
    
    def validate_data_quality(self):
        """데이터 품질 검증"""
        quality_results = {}
        
        # ML 예측 데이터 품질 검증
        ml_result = self.test_endpoint("/api/ml/predict/AAPL")
        if ml_result.get('success') and ml_result.get('data'):
            data = ml_result['data']
            quality_results['ml_prediction'] = {
                "has_ensemble_prediction": "ensemble_prediction" in data,
                "has_individual_predictions": "individual_predictions" in data,
                "model_count": len(data.get('individual_predictions', {})),
                "confidence_range": "valid" if 0 <= data.get('ensemble_prediction', {}).get('confidence', -1) <= 1 else "invalid",
                "features_count": len(data.get('features_used', []))
            }
        
        # 주식 데이터 품질 검증  
        stocks_result = self.test_endpoint("/api/stocks/live")
        if stocks_result.get('success') and stocks_result.get('data'):
            data = stocks_result['data']
            quality_results['stock_data'] = {
                "has_predictions": "predictions" in data,
                "predictions_count": len(data.get('predictions', [])),
                "has_market_summary": "market_summary" in data,
                "data_source": data.get('source', 'unknown')
            }
        
        return quality_results
    
    def generate_report(self):
        """종합 보고서 생성"""
        print("\n" + "=" * 50)
        print("📊 REST API 테스트 결과 보고서")
        print("=" * 50)
        
        # 기능성 요약
        functionality_success = sum(1 for r in self.results['functionality'] if r.get('success'))
        total_endpoints = len(self.results['functionality'])
        success_rate = round(functionality_success / total_endpoints * 100, 1)
        
        print(f"\n🎯 전체 성공률: {success_rate}% ({functionality_success}/{total_endpoints})")
        
        # 성능 요약
        if self.results['performance']:
            avg_perf = statistics.mean([r['avg_response_time'] for r in self.results['performance']])
            print(f"⚡ 평균 응답시간: {round(avg_perf, 3)}초")
        
        # ML 통합 상태
        ml_endpoints = [r for r in self.results['functionality'] if '/ml/' in r.get('endpoint', '')]
        ml_success = sum(1 for r in ml_endpoints if r.get('success'))
        ml_total = len(ml_endpoints)
        
        if ml_total > 0:
            ml_rate = round(ml_success / ml_total * 100, 1) 
            print(f"🤖 ML 통합 성공률: {ml_rate}% ({ml_success}/{ml_total})")
        
        # 데이터 품질 요약
        if self.results['data_quality']:
            print(f"📈 데이터 품질:")
            for key, value in self.results['data_quality'].items():
                if isinstance(value, dict):
                    valid_fields = sum(1 for v in value.values() if v not in [False, 'invalid', 'unknown'])
                    total_fields = len(value)
                    print(f"   - {key}: {valid_fields}/{total_fields} 필드 유효")
        
        # 권장사항
        print(f"\n💡 권장사항:")
        if success_rate < 90:
            print("   - 일부 엔드포인트 오류 수정 필요")
        if avg_perf > 2.0:
            print("   - 응답 속도 최적화 권장")
        
        return self.results


def main():
    """메인 테스트 실행"""
    tester = RESTAPITester()
    
    try:
        # 서버 연결 확인
        response = requests.get(f"{tester.base_url}/api/status", timeout=5)
        if response.status_code != 200:
            print("❌ 서버에 연결할 수 없습니다.")
            return
        
        # 종합 테스트 실행
        results = tester.comprehensive_test()
        
        # 결과를 JSON 파일로 저장
        with open('rest_api_test_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📁 상세 결과는 'rest_api_test_results.json'에 저장되었습니다.")
        
    except requests.exceptions.ConnectionError:
        print("❌ API 서버가 실행되지 않습니다. http://localhost:8091 확인하세요.")
    except Exception as e:
        print(f"❌ 테스트 중 오류 발생: {e}")


if __name__ == "__main__":
    main()