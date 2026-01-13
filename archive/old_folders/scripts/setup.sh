#!/bin/bash
# SPY Analysis System Setup Script

set -e  # Exit on any error

echo "🚀 SPY Analysis System Setup"
echo "============================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python 3.10+ is installed
print_status "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)
    
    if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 10 ]; then
        print_success "Python $PYTHON_VERSION found"
    else
        print_error "Python 3.10+ required, found $PYTHON_VERSION"
        exit 1
    fi
else
    print_error "Python 3 not found. Please install Python 3.10+"
    exit 1
fi

# Create virtual environment
print_status "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
python -m pip install --upgrade pip

# Install dependencies
print_status "Installing dependencies..."
if [ -f "pyproject.toml" ]; then
    pip install -e ".[dev,test,prod]"
    print_success "Dependencies installed from pyproject.toml"
else
    print_error "pyproject.toml not found"
    exit 1
fi

# Create necessary directories
print_status "Creating project directories..."
mkdir -p data/{raw,processed,models}
mkdir -p logs
mkdir -p results
mkdir -p temp
print_success "Project directories created"

# Set up pre-commit hooks
print_status "Setting up pre-commit hooks..."
if command -v pre-commit &> /dev/null; then
    pre-commit install
    print_success "Pre-commit hooks installed"
else
    print_warning "pre-commit not found, skipping hooks setup"
fi

# Download sample data (if needed)
print_status "Checking for sample data..."
if [ ! -f "data/raw/sample_data.csv" ]; then
    print_status "Sample data will be generated on first run"
else
    print_success "Sample data exists"
fi

# Set up environment variables
print_status "Setting up environment..."
if [ ! -f ".env" ]; then
    cat > .env << EOF
# SPY Analysis Environment Configuration
ENVIRONMENT=development
LOG_LEVEL=INFO
DATA_PATH=./data
MODELS_PATH=./data/models
RESULTS_PATH=./results

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

# Redis Configuration (optional)
REDIS_URL=redis://localhost:6379

# Monitoring
METRICS_PORT=8001
ENABLE_METRICS=true

# External APIs (set your own keys)
# ALPHA_VANTAGE_API_KEY=your_key_here
# NEWS_API_KEY=your_key_here
EOF
    print_success ".env file created"
else
    print_warning ".env file already exists"
fi

# Run basic system check
print_status "Running system check..."
python -c "
import sys
import pandas as pd
import numpy as np
import sklearn
import yfinance as yf

print(f'Python: {sys.version}')
print(f'Pandas: {pd.__version__}')
print(f'NumPy: {np.__version__}')  
print(f'Scikit-learn: {sklearn.__version__}')
print('✅ Core dependencies working')
"

print_success "System check completed"

# Display next steps
echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Set up your API keys in .env file"
echo "3. Run the system: python src/main.py"
echo "4. Or start the API: python src/api/main.py"
echo "5. Run tests: pytest"
echo ""
echo "For more information, see README.md"
echo ""