#!/bin/bash
# AI Stock Prediction System - Startup Script
# 모듈 경로 문제 해결 및 환경 설정 자동화

set -e  # 에러 발생시 중단

# 색상 설정
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 AI Stock Prediction System 시작...${NC}"

# 현재 디렉토리 확인
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${YELLOW}📂 작업 디렉토리: $SCRIPT_DIR${NC}"

# Python 환경 확인
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}❌ Python이 설치되어 있지 않습니다.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Python 명령: $PYTHON_CMD${NC}"

# 가상환경 활성화
if [ -d "venv" ]; then
    echo -e "${YELLOW}🐍 가상환경 활성화...${NC}"
    source venv/bin/activate
    echo -e "${GREEN}✅ 가상환경 활성화됨${NC}"
else
    echo -e "${RED}❌ venv 디렉토리를 찾을 수 없습니다. 가상환경을 생성해주세요.${NC}"
    echo -e "${BLUE}💡 다음 명령으로 가상환경을 생성하세요:${NC}"
    echo -e "   python3 -m venv venv"
    echo -e "   source venv/bin/activate"
    echo -e "   pip install -r config/requirements.txt"
    exit 1
fi

# PYTHONPATH 설정
export PYTHONPATH="${PYTHONPATH}:${SCRIPT_DIR}"
echo -e "${GREEN}✅ PYTHONPATH 설정: $PYTHONPATH${NC}"

# 패키지 설치 (개발 모드)
echo -e "${YELLOW}📦 패키지 설치 중...${NC}"
$PYTHON_CMD -m pip install -e . --quiet
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ 패키지 설치 완료${NC}"
else
    echo -e "${RED}❌ 패키지 설치 실패${NC}"
    exit 1
fi

# 필수 디렉토리 생성
echo -e "${YELLOW}📁 필수 디렉토리 확인...${NC}"
directories=(
    "data/raw"
    "data/processed" 
    "data/models"
    "results/analysis"
    "results/training"
    "results/realtime"
    "log"
)

for dir in "${directories[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo -e "${GREEN}✅ 디렉토리 생성: $dir${NC}"
    fi
done

# 환경 변수 파일 확인
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        echo -e "${YELLOW}⚠️ .env 파일이 없습니다. .env.example을 복사하세요.${NC}"
        echo -e "${BLUE}💡 다음 명령을 실행하세요:${NC}"
        echo -e "   cp .env.example .env"
        echo -e "   # 그리고 .env 파일에서 실제 API 키를 설정하세요"
    else
        echo -e "${RED}❌ .env 파일과 .env.example 파일이 모두 없습니다.${NC}"
    fi
else
    echo -e "${GREEN}✅ .env 파일 확인됨${NC}"
fi

# 모듈 임포트 테스트
echo -e "${YELLOW}🧪 모듈 임포트 테스트...${NC}"
$PYTHON_CMD -c "
try:
    import src
    import src.models.model_training
    import src.core.data_collection_pipeline
    import src.utils.system_orchestrator
    print('✅ 모든 핵심 모듈 임포트 성공')
except ImportError as e:
    print(f'❌ 모듈 임포트 실패: {e}')
    exit(1)
" 2>/dev/null

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ 모든 모듈이 정상적으로 로드됩니다${NC}"
else
    echo -e "${RED}❌ 모듈 로드 실패. 의존성을 확인해주세요.${NC}"
fi

echo -e "${BLUE}🎉 시스템 준비 완료!${NC}"
echo -e "${BLUE}📋 사용 가능한 명령들:${NC}"
echo -e "   ai-stock-orchestrator  # 전체 시스템 실행"
echo -e "   ai-stock-train         # 모델 훈련"
echo -e "   ai-stock-test          # 실시간 테스트"
echo -e "   ai-stock-dashboard     # 대시보드 실행"
echo -e ""
echo -e "${BLUE}또는 직접 스크립트 실행:${NC}"
echo -e "   $PYTHON_CMD src/utils/system_orchestrator.py"
echo -e "   $PYTHON_CMD src/models/model_training.py"

# 명령행 인자가 있으면 해당 명령 실행
if [ $# -gt 0 ]; then
    echo -e "${BLUE}🏃 명령 실행: $@${NC}"
    exec "$@"
fi