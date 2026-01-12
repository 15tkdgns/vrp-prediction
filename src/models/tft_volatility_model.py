#!/usr/bin/env python3
"""
Temporal Fusion Transformer (TFT) 변동성 예측 모델
=================================================

Google Research의 TFT를 SPY 변동성 예측에 적용
- Multi-horizon 예측 (1일, 5일 동시)
- Attention 메커니즘으로 중요 특성 자동 식별
- 해석 가능한 예측

실행 시간: 약 30-60분 (CPU 기준)
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
import json
from datetime import datetime

# PyTorch
import torch
from torch.utils.data import DataLoader

# PyTorch Forecasting
try:
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    HAS_TFT = True
except ImportError:
    HAS_TFT = False
    print("⚠️ PyTorch Forecasting 미설치. 설치 중...")
    print("pip install pytorch-forecasting pytorch-lightning")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# =============================================================================
# 1. 데이터 준비
# =============================================================================

def load_and_prepare_data():
    """SPY 및 VIX 데이터 로드 및 TFT 형식으로 변환"""
    print("\n" + "="*60)
    print("[1/6] 데이터 로드 및 준비")
    print("="*60)
    
    # SPY 데이터 로드
    csv_path = Path('data/raw/spy_data_2020_2025.csv')
    if csv_path.exists():
        spy = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        print(f"  ✓ SPY 데이터: {len(spy)} 행")
    else:
        print("  ⚠️ SPY 데이터 다운로드 중...")
        spy = yf.download('SPY', start='2020-01-01', end='2025-01-01', 
                         progress=False, auto_adjust=True)
        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)
    
    # VIX 데이터 로드
    print("  → VIX 데이터 로드 중...")
    vix = yf.download('^VIX', start='2020-01-01', end='2025-01-01',
                     progress=False, auto_adjust=True)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    spy['VIX'] = vix['Close']
    
    # 결측치 처리
    spy = spy.ffill().dropna()
    
    # 기본 특성 생성
    print("  → 특성 생성 중...")
    spy['returns'] = spy['Close'].pct_change()
    spy['volatility'] = spy['returns'].rolling(5).std() * np.sqrt(252)
    spy['log_volume'] = np.log(spy['Volume'] + 1)
    
    # VIX 특성
    spy['vix_lag1'] = spy['VIX'].shift(1)
    spy['vix_change'] = spy['VIX'].pct_change()
    
    # Regime 특성
    vix_lag = spy['VIX'].shift(1)
    spy['regime_high_vol'] = (vix_lag >= 25).astype(int)
    
    # 타겟 생성 (5일 미래 변동성)
    vol_values = []
    returns = spy['returns'].values
    for i in range(len(returns)):
        if i + 5 < len(returns):
            future_window = returns[i+1:i+6]
            vol_values.append(pd.Series(future_window).std())
        else:
            vol_values.append(np.nan)
    spy['target_vol_5d'] = vol_values
    
    # 1일 타겟도 추가
    spy['target_vol_1d'] = spy['returns'].shift(-1).abs()
    
    spy = spy.dropna()
    
    print(f"  ✓ 최종 데이터: {len(spy)} 행")
    print(f"  ✓ 기간: {spy.index[0]} ~ {spy.index[-1]}")
    
    return spy


def convert_to_timeseries_format(spy):
    """TFT용 TimeSeriesDataSet 형식으로 변환"""
    print("\n" + "="*60)
    print("[2/6] TFT 형식 변환")
    print("="*60)
    
    # TFT는 패널 데이터 기대 (ticker별)
    df = spy.reset_index()
    
    # 컬럼 이름 확인 및 변경
    if 'index' in df.columns:
        df = df.rename(columns={'index': 'date'})
    elif df.columns[0] not in ['date', 'Date']:
        df = df.rename(columns={df.columns[0]: 'date'})
    
    # 단일 시계열을 패널로 변환
    df['ticker'] = 'SPY'
    df['time_idx'] = np.arange(len(df))
    
    # 특성 선택
    feature_cols = [
        'returns', 'volatility', 'log_volume',
        'VIX', 'vix_lag1', 'vix_change',
        'regime_high_vol',
        'target_vol_1d', 'target_vol_5d'
    ]
    
    # date 컬럼이 있는지 확인
    if 'date' in df.columns:
        df = df[['ticker', 'time_idx', 'date'] + feature_cols]
    else:
        df = df[['ticker', 'time_idx'] + feature_cols]
    
    # 카테고리 변수를 문자열로 변환 (TFT 요구사항)
    df['regime_high_vol'] = df['regime_high_vol'].astype(str)
    
    # 결측치 최종 제거
    df = df.dropna()
    
    print(f"  ✓ 변환 완료: {len(df)} 샘플")
    print(f"  ✓ 특성: {len(feature_cols) - 2}개 (타겟 제외)")
    
    return df


# =============================================================================
# 2. TFT 모델 구축
# =============================================================================

def create_tft_datasets(df, max_encoder_length=30, max_prediction_length=5):
    """TimeSeriesDataSet 생성"""
    print("\n" + "="*60)
    print("[3/6] TFT 데이터셋 생성")
    print("="*60)
    
    if not HAS_TFT:
        raise ImportError("PyTorch Forecasting이 설치되지 않았습니다.")
    
    # 학습/검증 분할 (80/20)
    split_idx = int(len(df) * 0.8)
    
    print(f"  → Encoder 길이: {max_encoder_length}일")
    print(f"  → Prediction 길이: {max_prediction_length}일")
    
    # 학습 데이터셋
    training = TimeSeriesDataSet(
        df.iloc[:split_idx],
        time_idx="time_idx",
        target="target_vol_5d",
        group_ids=["ticker"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        
        # 정적 변수 (변하지 않는 값) - 없음
        static_categoricals=[],
        static_reals=[],
        
        # 시간 변화 변수 (알려진 미래 값)
        time_varying_known_reals=["time_idx"],
        time_varying_known_categoricals=[],
        
        # 시간 변화 변수 (예측 대상)
        time_varying_unknown_reals=[
            "returns", "volatility", "log_volume",
            "VIX", "vix_lag1", "vix_change"
        ],
        time_varying_unknown_categoricals=["regime_high_vol"],
        
        # 정규화
        target_normalizer=GroupNormalizer(
            groups=["ticker"], 
            transformation="softplus"
        ),
        
        # 추가 특성
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    
    # 검증 데이터셋
    validation = TimeSeriesDataSet.from_dataset(
        training, 
        df.iloc[split_idx:], 
        predict=True, 
        stop_randomization=True
    )
    
    # DataLoader
    batch_size = 32
    train_dataloader = training.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0
    )
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=batch_size, num_workers=0
    )
    
    print(f"  ✓ 학습 샘플: {len(training)}")
    print(f"  ✓ 검증 샘플: {len(validation)}")
    print(f"  ✓ Batch size: {batch_size}")
    
    return training, validation, train_dataloader, val_dataloader


def build_tft_model(training):
    """TFT 모델 생성"""
    print("\n" + "="*60)
    print("[4/6] TFT 모델 생성")
    print("="*60)
    
    # 경량 TFT 설정 (과적합 방지)
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.01,
        hidden_size=16,  # 작게 시작
        attention_head_size=2,
        dropout=0.2,
        hidden_continuous_size=8,
        output_size=7,  # Quantile outputs
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )
    
    print("  ✓ TFT 모델 구성:")
    print(f"    - Hidden size: 16")
    print(f"    - Attention heads: 2")
    print(f"    - Dropout: 0.2")
    print(f"    - Output: 7 quantiles")
    
    return tft


# =============================================================================
# 3. 모델 학습
# =============================================================================

def train_tft_model(tft, train_dataloader, val_dataloader):
    """TFT 모델 학습"""
    print("\n" + "="*60)
    print("[5/6] TFT 모델 학습")
    print("="*60)
    
    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=10,
        verbose=False,
        mode="min"
    )
    
    # Checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath='data/models',
        filename='tft_volatility',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
    )
    
    # Trainer (CPU 명시)
    trainer = Trainer(
        max_epochs=50,
        accelerator="cpu",
        devices=1,
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback, checkpoint_callback],
        enable_progress_bar=True,
        enable_model_summary=True,
        logger=False,  # 로거 비활성화
    )
    
    print("  → 학습 시작 (최대 50 epochs, Early stopping 적용)")
    print("  → 예상 시간: 10-30분 (CPU)")
    print()
    
    try:
        # fit 메서드 호출 (모델은 LightningModule)
        trainer.fit(
            model=tft,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
    except Exception as e:
        print(f"\n  ⚠️ 학습 중 오류 발생, 간단한 학습 시도: {e}")
        # 간단한 대안
        trainer = Trainer(
            max_epochs=20,
            accelerator="cpu",
            devices=1,
            enable_progress_bar=True,
        )
        trainer.fit(tft, train_dataloader, val_dataloader)
    
    print(f"\n  ✓ 학습 완료")
    print(f"  ✓ Best model saved to: data/models/tft_volatility.ckpt")
    
    return trainer, tft


# =============================================================================
# 4. 평가 및 해석
# =============================================================================

def evaluate_tft_model(tft, val_dataloader, validation_data):
    """TFT 모델 평가"""
    print("\n" + "="*60)
    print("[6/6] 모델 평가 및 해석")
    print("="*60)
    
    # 예측
    predictions = tft.predict(val_dataloader, return_x=True)
    
    # 실제 값과 예측 값 추출
    actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
    pred_values = predictions.output
    
    # 중간값 (median, quantile 0.5) 사용
    pred_median = pred_values[:, :, 3]  # 7개 quantile 중 중간
    
    # 첫 스텝만 (1일 예측)
    y_true = actuals[:, 0].cpu().numpy()
    y_pred = pred_median[:, 0].cpu().numpy()
    
    # 메트릭 계산
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    print("\n  📊 TFT 성능:")
    print(f"    • R²:    {r2:.4f}")
    print(f"    • RMSE:  {rmse:.6f}")
    print(f"    • MAE:   {mae:.6f}")
    
    # 결과 저장
    results = {
        'model': 'Temporal Fusion Transformer',
        'test_r2': float(r2),
        'test_rmse': float(rmse),
        'test_mae': float(mae),
        'config': {
            'hidden_size': 16,
            'attention_heads': 2,
            'dropout': 0.2,
            'max_encoder_length': 30,
            'max_prediction_length': 5,
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open('data/raw/tft_model_performance.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n  ✓ 결과 저장: data/raw/tft_model_performance.json")
    
    # Attention weights 분석
    try:
        interpretation = tft.interpret_output(predictions.output, reduction="sum")
        
        print("\n  🔍 Attention 분석:")
        print("    → Variable importance (변수 중요도)")
        
        # 특성 중요도 출력
        if hasattr(interpretation, 'attention'):
            print("    (Attention weights 계산 완료)")
    except Exception as e:
        print(f"    ⚠️ Attention 분석 실패: {e}")
    
    return results


# =============================================================================
# 5. 메인 파이프라인
# =============================================================================

def main():
    """TFT 전체 파이프라인 실행"""
    print("\n" + "🚀"*30)
    print("Temporal Fusion Transformer 변동성 예측")
    print("🚀"*30)
    
    if not HAS_TFT:
        print("\n❌ PyTorch Forecasting이 설치되지 않았습니다.")
        print("\n설치 명령:")
        print("  pip install pytorch-forecasting pytorch-lightning")
        return None
    
    try:
        # 1. 데이터 로드
        spy = load_and_prepare_data()
        
        # 2. TFT 형식 변환
        df = convert_to_timeseries_format(spy)
        
        # 3. 데이터셋 생성
        training, validation, train_dl, val_dl = create_tft_datasets(df)
        
        # 4. 모델 생성
        tft = build_tft_model(training)
        
        # 5. 학습
        trainer, tft = train_tft_model(tft, train_dl, val_dl)
        
        # 6. 평가
        results = evaluate_tft_model(tft, val_dl, validation)
        
        print("\n" + "="*60)
        print("✅ TFT 파이프라인 완료!")
        print("="*60)
        print(f"\n  🏆 최종 R²: {results['test_r2']:.4f}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    results = main()
