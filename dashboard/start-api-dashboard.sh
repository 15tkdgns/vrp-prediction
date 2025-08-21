#!/bin/bash
# AI Stock Dashboard - API Integration Startup Script

echo "🚀 AI Stock Dashboard API Integration 시작"
echo "======================================="

# Check if virtual environment exists
if [ ! -d "../venv" ]; then
    echo "❌ Python 가상환경이 없습니다. /root/workspace/venv 경로를 확인해주세요."
    exit 1
fi

# Kill any existing processes on the ports
echo "🧹 기존 프로세스 정리 중..."
pkill -f "python.*api_server.py" 2>/dev/null || true
pkill -f "npm run dev" 2>/dev/null || true
pkill -f "http-server" 2>/dev/null || true

# Wait for processes to terminate
sleep 2

echo "📦 필요한 패키지 설치 확인 중..."
cd /root/workspace
source venv/bin/activate
cd dashboard
pip install -q flask flask-cors python-dotenv 2>/dev/null || echo "⚠️ 일부 패키지 설치 실패 (이미 설치되어 있을 수 있음)"

echo ""
echo "🔥 Flask API 서버 시작 중... (포트 8091)"
echo "   - 실시간 주식 데이터: http://localhost:8091/api/stocks/live"
echo "   - 뉴스 감정 분석: http://localhost:8091/api/news/sentiment"
echo "   - 모델 성능 지표: http://localhost:8091/api/models/performance"
echo "   - API 상태: http://localhost:8091/api/status"

# Start Flask API server in background
cd /root/workspace
source venv/bin/activate
cd dashboard
nohup python api_server.py > api_server.log 2>&1 &
API_PID=$!

# Wait for API server to start
echo "⏳ API 서버 시작 대기 중..."
sleep 5

# Check if API server is running
if curl -s http://localhost:8091/api/status > /dev/null; then
    echo "✅ Flask API 서버 시작됨 (PID: $API_PID)"
else
    echo "❌ Flask API 서버 시작 실패"
    cat api_server.log
    exit 1
fi

echo ""
echo "🌐 프론트엔드 대시보드 시작 중... (포트 8080)"
echo "   - 대시보드: http://localhost:8080"

# Start frontend server in background
npm run dev > frontend.log 2>&1 &
FRONTEND_PID=$!

# Wait for frontend server to start
echo "⏳ 프론트엔드 서버 시작 대기 중..."
sleep 3

# Check if frontend server is running
if curl -s http://localhost:8080 > /dev/null; then
    echo "✅ 프론트엔드 대시보드 시작됨 (PID: $FRONTEND_PID)"
else
    echo "❌ 프론트엔드 서버 시작 실패"
    cat frontend.log
    kill $API_PID 2>/dev/null
    exit 1
fi

echo ""
echo "🎉 시스템 완전 시작 완료!"
echo "======================================="
echo "📊 AI Stock Dashboard: http://localhost:8080"
echo "🔌 API 서버: http://localhost:8091"
echo ""
echo "🔄 실시간 데이터 소스:"
echo "   • API 우선: Flask API (8091) → 실제 주식 데이터"
echo "   • 폴백 1: JSON 파일 → 기존 데이터"  
echo "   • 폴백 2: Mock 데이터 → 안정적인 데모"
echo ""
echo "⚙️ 기능:"
echo "   • 자동 새로고침: 60초마다"
echo "   • 실시간 API 호출 with 재시도 로직"
echo "   • 캐싱으로 성능 최적화"
echo "   • 에러 처리 및 폴백 시스템"
echo ""
echo "🛑 서버 중지: Ctrl+C 또는 ./stop-servers.sh"
echo ""

# Create PID file for easy cleanup
echo $API_PID > api_server.pid
echo $FRONTEND_PID > frontend.pid

# Keep script running until user stops
echo "📱 서버들이 백그라운드에서 실행 중입니다..."
echo "🔍 로그 확인: tail -f api_server.log frontend.log"
echo "⏹️ 중지하려면 Ctrl+C를 누르세요"

# Wait for user interrupt
trap "echo ''; echo '🛑 시스템 종료 중...'; kill $API_PID $FRONTEND_PID 2>/dev/null; echo '✅ 모든 서버 종료됨'; exit 0" INT TERM

# Keep running until interrupted
while true; do
    sleep 1
    # Check if processes are still running
    if ! kill -0 $API_PID 2>/dev/null; then
        echo "❌ API 서버가 중단되었습니다"
        break
    fi
    if ! kill -0 $FRONTEND_PID 2>/dev/null; then
        echo "❌ 프론트엔드 서버가 중단되었습니다"
        break
    fi
done