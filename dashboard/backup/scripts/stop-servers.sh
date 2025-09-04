#!/bin/bash
# Stop all dashboard servers

echo "🛑 AI Stock Dashboard 서버 중지 중..."

# Kill processes by PID if PID files exist
if [ -f "api_server.pid" ]; then
    API_PID=$(cat api_server.pid)
    kill $API_PID 2>/dev/null && echo "✅ API 서버 중지됨 (PID: $API_PID)"
    rm -f api_server.pid
fi

if [ -f "frontend.pid" ]; then
    FRONTEND_PID=$(cat frontend.pid)
    kill $FRONTEND_PID 2>/dev/null && echo "✅ 프론트엔드 서버 중지됨 (PID: $FRONTEND_PID)"
    rm -f frontend.pid
fi

# Kill any remaining processes by name/port
pkill -f "python.*api_server.py" 2>/dev/null && echo "🧹 Python API 서버 프로세스 정리됨"
pkill -f "npm run dev" 2>/dev/null && echo "🧹 npm dev 서버 프로세스 정리됨"  
pkill -f "http-server" 2>/dev/null && echo "🧹 http-server 프로세스 정리됨"

# Clean up log files (optional)
if [ "$1" = "clean" ]; then
    rm -f api_server.log frontend.log
    echo "🧹 로그 파일 정리됨"
fi

echo "✅ 모든 서버가 중지되었습니다"