#!/bin/bash
# 기존 코드를 건드리지 않고 모듈 경로 문제만 해결하는 간단한 스크립트

# 현재 디렉토리를 PYTHONPATH에 추가
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 가상환경 활성화 (있는 경우)
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# 인자로 받은 Python 스크립트 실행
if [ $# -eq 0 ]; then
    echo "사용법: ./run_with_path.sh <python_script>"
    echo "예: ./run_with_path.sh src/utils/system_orchestrator.py"
else
    python3 "$@"
fi