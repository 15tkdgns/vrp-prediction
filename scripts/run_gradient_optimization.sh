#!/bin/bash

# 경사하강법 하이퍼파라미터 최적화 백그라운드 실행 스크립트

echo "🚀 경사하강법 하이퍼파라미터 최적화 시작..."
echo "📅 시작 시간: $(date)"

# 작업 디렉토리 설정
cd /root/workspace

# Python 패스 설정
export PYTHONPATH=/root/workspace:$PYTHONPATH

# 로그 파일 초기화
LOG_FILE="/root/workspace/data/raw/gradient_optimization.log"
PROGRESS_FILE="/root/workspace/data/raw/optimization_progress.json"

echo "📁 로그 파일: $LOG_FILE"
echo "📊 진행상황 파일: $PROGRESS_FILE"

# 기존 로그 백업
if [ -f "$LOG_FILE" ]; then
    mv "$LOG_FILE" "${LOG_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
fi

# Python 최적화 스크립트 백그라운드 실행
echo "⚡ 백그라운드에서 최적화 시작..."
echo "📋 최적화 설정:"
echo "   - 학습률: 0.1"
echo "   - 최대 반복: 500회"
echo "   - 조기 종료: 30회 연속 개선 없음"
echo "   - CV: Purged K-Fold (5-fold)"

# nohup으로 백그라운드 실행
nohup python3 src/optimization/gradient_hyperparameter_optimizer.py > "${LOG_FILE}" 2>&1 &

# 프로세스 ID 저장
OPTIMIZATION_PID=$!
echo "🔢 프로세스 ID: $OPTIMIZATION_PID"
echo $OPTIMIZATION_PID > /root/workspace/data/raw/optimization_pid.txt

echo ""
echo "✅ 백그라운드 최적화 프로세스가 시작되었습니다!"
echo ""
echo "📋 모니터링 명령어:"
echo "   진행상황 확인: tail -f $LOG_FILE"
echo "   프로세스 상태: ps aux | grep $OPTIMIZATION_PID"
echo "   프로세스 종료: kill $OPTIMIZATION_PID"
echo ""
echo "🔍 실시간 모니터링 스크립트 실행:"
echo "   python3 scripts/monitor_optimization.py"
echo ""

# 초기 상태 저장
cat > "$PROGRESS_FILE" << EOF
{
  "status": "started",
  "pid": $OPTIMIZATION_PID,
  "start_time": "$(date -Iseconds)",
  "log_file": "$LOG_FILE",
  "estimated_duration": "30-60 minutes",
  "monitoring_command": "tail -f $LOG_FILE"
}
EOF

echo "💾 진행상황 파일 생성: $PROGRESS_FILE"
echo ""
echo "🎯 최적화가 완료되면 다음 파일에서 결과를 확인하세요:"
echo "   /root/workspace/data/raw/gradient_optimization_results.json"