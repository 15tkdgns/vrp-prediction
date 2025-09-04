#!/usr/bin/env python3
"""
FastAPI server for AI Stock Prediction Dashboard
Modern replacement for the simple HTTP server with enhanced API capabilities
"""
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import urllib.request

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import uvicorn

# Load environment variables
load_dotenv()
parent_env = Path(__file__).parent.parent / '.env'
if parent_env.exists():
    load_dotenv(parent_env)

# FastAPI app configuration
app = FastAPI(
    title="AI Stock Prediction Dashboard API",
    description="FastAPI server for real-time stock prediction and analysis",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
PORT = 8090
DATA_DIR = Path(__file__).parent.parent / "data/raw"

# Mount static files
dashboard_dir = Path(__file__).parent
app.mount("/static", StaticFiles(directory=dashboard_dir), name="static")

@app.get("/")
async def serve_index():
    """Serve the main dashboard page"""
    index_file = dashboard_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    raise HTTPException(status_code=404, detail="Dashboard not found")

@app.get("/api/health")
async def health_check():
    """System health check endpoint"""
    try:
        health_data = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0",
            "server": "FastAPI",
            "components": {
                "realtime_results": (DATA_DIR / "realtime_test_results.json").exists(),
                "model_performance": (DATA_DIR / "model_performance.json").exists(),
                "sp500_predictions": (DATA_DIR / "sp500_prediction_data.json").exists(),
                "market_sentiment": (DATA_DIR / "market_sentiment.json").exists(),
                "trading_volume": (DATA_DIR / "trading_volume.json").exists(),
            }
        }
        return JSONResponse(content=health_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check error: {str(e)}")

@app.get("/api/keys")
async def get_api_keys():
    """Get API keys for external services"""
    try:
        api_keys = {
            "alphaVantage": os.getenv("ALPHA_VANTAGE_KEY"),
            "financialModelingPrep": os.getenv("FMP_KEY"),
            "twelveData": os.getenv("TWELVE_DATA_KEY"),
            "polygon": os.getenv("POLYGON_KEY"),
            "iexCloud": os.getenv("IEX_CLOUD_KEY"),
            "marketaux": os.getenv("MARKETAUX_KEY"),
        }
        # Filter out None values
        api_keys = {k: v for k, v in api_keys.items() if v is not None}
        return JSONResponse(content=api_keys)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API Keys error: {str(e)}")

@app.get("/api/realtime-results")
async def get_realtime_results():
    """Get real-time prediction results"""
    try:
        data_path = DATA_DIR / "realtime_test_results.json"
        if not data_path.exists():
            raise HTTPException(status_code=404, detail="Realtime results not found")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Realtime results error: {str(e)}")

@app.get("/api/model-performance")
async def get_model_performance():
    """Get model performance metrics"""
    try:
        data_path = DATA_DIR / "model_performance.json"
        if not data_path.exists():
            raise HTTPException(status_code=404, detail="Model performance data not found")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model performance error: {str(e)}")

@app.get("/api/sp500-predictions")
async def get_sp500_predictions():
    """Get S&P 500 prediction data"""
    try:
        data_path = DATA_DIR / "sp500_prediction_data.json"
        if not data_path.exists():
            raise HTTPException(status_code=404, detail="SP500 prediction data not found")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SP500 predictions error: {str(e)}")

@app.get("/api/market-sentiment")
async def get_market_sentiment():
    """Get market sentiment analysis data"""
    try:
        data_path = DATA_DIR / "market_sentiment.json"
        if not data_path.exists():
            raise HTTPException(status_code=404, detail="Market sentiment data not found")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Market sentiment error: {str(e)}")

@app.get("/api/trading-volume")
async def get_trading_volume():
    """Get trading volume data"""
    try:
        data_path = DATA_DIR / "trading_volume.json"
        if not data_path.exists():
            raise HTTPException(status_code=404, detail="Trading volume data not found")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trading volume error: {str(e)}")

@app.get("/proxy/rss")
async def rss_proxy(url: str = Query(..., description="RSS feed URL to proxy")):
    """RSS feed proxy to avoid CORS issues"""
    try:
        if not url:
            raise HTTPException(status_code=400, detail="Missing URL parameter")
        
        # Fetch the RSS feed
        with urllib.request.urlopen(url) as response:
            content = response.read().decode("utf-8")
        
        return JSONResponse(content={"content": content}, media_type="application/xml")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")

@app.get("/api/system-status")
async def get_system_status():
    """Get comprehensive system status"""
    try:
        data_files = [
            "realtime_test_results.json",
            "model_performance.json", 
            "sp500_prediction_data.json",
            "market_sentiment.json",
            "trading_volume.json",
            "system_health.json"
        ]
        
        file_status = {}
        for file in data_files:
            file_path = DATA_DIR / file
            file_status[file] = {
                "exists": file_path.exists(),
                "size": file_path.stat().st_size if file_path.exists() else 0,
                "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat() if file_path.exists() else None
            }
        
        status_data = {
            "server": "FastAPI",
            "status": "running",
            "timestamp": datetime.now().isoformat(),
            "port": PORT,
            "files": file_status
        }
        
        return JSONResponse(content=status_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"System status error: {str(e)}")

# Catch-all for serving static files
@app.get("/{file_path:path}")
async def serve_static_files(file_path: str):
    """Serve static files from dashboard directory"""
    file_full_path = dashboard_dir / file_path
    
    # Security check - prevent directory traversal
    if not str(file_full_path).startswith(str(dashboard_dir)):
        raise HTTPException(status_code=404, detail="File not found")
    
    if file_full_path.exists() and file_full_path.is_file():
        return FileResponse(file_full_path)
    
    raise HTTPException(status_code=404, detail="File not found")

def main():
    """Start the FastAPI server"""
    print(f"🚀 Starting FastAPI server on port {PORT}")
    print(f"📊 Dashboard: http://localhost:{PORT}")
    print(f"📚 API Docs: http://localhost:{PORT}/docs")
    print(f"📖 ReDoc: http://localhost:{PORT}/redoc")
    print(f"📁 Working directory: {dashboard_dir}")
    
    uvicorn.run(
        "fastapi_server:app",
        host="0.0.0.0",
        port=PORT,
        reload=True,
        reload_dirs=[str(dashboard_dir)],
        log_level="info"
    )

if __name__ == "__main__":
    main()