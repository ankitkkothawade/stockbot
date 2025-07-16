#!/usr/bin/env python3
"""
Deploy_backtest.py

Backtesting deployment script using historical 1-minute Alpaca data.
Simulates the same intraday loop logic from live deploy, but on past data.
"""
import os
import sys
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.signal import detrend
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from stable_baselines3 import PPO
from alpaca_trade_api.rest import REST, TimeFrame

# ─── UTF-8 Output ─────────────────────────────────────────────────────────────
# Ensure emoji and unicode print correctly
sys.stdout.reconfigure(encoding='utf-8')

# ─── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── 0) CONFIG ────────────────────────────────────────────────────────────────
TICKERS = ["AAPL", "JPM", "AMZN", "TSLA", "MSFT"]
SEQ_LEN = 60
LOOKBACK_BARS = SEQ_LEN + 20
INITIAL_BALANCE = 10_000.0
TRAIN_YEAR = 2022

# Alpaca credentials from environment
ALPACA_API_KEY = os.getenv("APCA_API_KEY_ID")
ALPACA_SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")
ALPACA_BASE_URL = os.getenv(
    "APCA_BASE_URL", "https://paper-api.alpaca.markets/v2"
)

# Backtest date and market hours
BACKTEST_DATE = "2023-01-04"
MARKET_OPEN = datetime.fromisoformat(f"{BACKTEST_DATE} 09:30:00")
MARKET_CLOSE = datetime.fromisoformat(f"{BACKTEST_DATE} 16:00:00")

# ─── 1) INIT MODELS & SCALERS ────────────────────────────────────────────────
logger.info("Initializing models and scalers...")
# Alpaca REST client for fetching historical bars
api = REST(
    ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version="v2"
)
# Load trained models
cnn_lstm = load_model("models/final_multi_stock_cnn_lstm.h5", compile=False)
agent = PPO.load("cnn_lstm_multi_stock_ppo.zip")

# Load raw feature arrays and fit scalers
raw_X = {}
dates_idx = {}
scaler_X = {}
for ticker in TICKERS:
    tl = ticker.lower()
    raw_path = f"preprocessed_data/{tl}_raw_X.npy"
    tech_csv = f"preprocessed_data/{tl}_tech_scaled_features.csv"
    X_full = np.load(raw_path)
    raw_X[ticker] = X_full

    df_tech = pd.read_csv(tech_csv, index_col=0, parse_dates=True)
    dates_idx[ticker] = df_tech.index
    train_mask = df_tech.index.year <= TRAIN_YEAR
    scaler = MinMaxScaler().fit(X_full[train_mask])
    scaler_X[ticker] = scaler

# Determine model expected feature dimension
_, seq_len_model, feat_dim_model = cnn_lstm.input_shape
max_feat_dim = max(X.shape[1] for X in raw_X.values())
logger.info(f"Model expects sequence length {seq_len_model}, feature dim {feat_dim_model}")

# ─── 2) FETCH HISTORICAL BARS ONCE ───────────────────────────────────────────
logger.info(f"Pre-fetching {BACKTEST_DATE} bars for all tickers…")
start_fetch = MARKET_OPEN - timedelta(minutes=LOOKBACK_BARS)
bars_data = {}
for sym in TICKERS:
    raw = api.get_bars(
        symbol=sym,
        timeframe=TimeFrame.Minute,
        start=start_fetch.isoformat() + "Z",
        end=MARKET_CLOSE.isoformat() + "Z",
        adjustment="raw",
    )
    df = pd.DataFrame([
        {"time": b.t, "open": b.o, "high": b.h, "low": b.l, "close": b.c, "volume": b.v}
        for b in raw
    ])
    if not df.empty:
        df.set_index("time", inplace=True)
        df.sort_index(inplace=True)
        df.index = df.index.tz_localize(None)
    bars_data[sym] = df

# ─── 3) FEATURE EXTRACTION ───────────────────────────────────────────────────
def extract_features_intraday(ticker: str, df: pd.DataFrame) -> np.ndarray:
    """
    Extract features for a given ticker and DataFrame of LOOKBACK_BARS rows.
    Returns a vector of length feat_dim_model + 1 (features + cnn_lstm price pred).
    """
    # Align indices
    positions = dates_idx[ticker].get_indexer(df.index)
    X_window = raw_X[ticker][positions]

    # Pad/truncate raw features to scaler input dims
    n_raw = scaler_X[ticker].n_features_in_
    if X_window.shape[1] < n_raw:
        pad = np.zeros((X_window.shape[0], n_raw - X_window.shape[1]))
        X_window = np.hstack([X_window, pad])
    elif X_window.shape[1] > n_raw:
        X_window = X_window[:, :n_raw]

    # Scale
    X_scaled = scaler_X[ticker].transform(X_window)

    # Pad/truncate to max_feat_dim for CNN
    if X_scaled.shape[1] < max_feat_dim:
        pad2 = np.zeros((X_scaled.shape[0], max_feat_dim - X_scaled.shape[1]))
        X_scaled = np.hstack([X_scaled, pad2])
    elif X_scaled.shape[1] > max_feat_dim:
        X_scaled = X_scaled[:, :max_feat_dim]

    # Sequence slice
    X_seq = X_scaled[-SEQ_LEN:]

    # Ensure matching model input dim
    if X_seq.shape[1] != feat_dim_model:
        logger.debug(f"Adjusting features: got {X_seq.shape[1]}, expected {feat_dim_model}")
        if X_seq.shape[1] < feat_dim_model:
            pad3 = np.zeros((SEQ_LEN, feat_dim_model - X_seq.shape[1]))
            X_seq = np.hstack([X_seq, pad3])
        else:
            X_seq = X_seq[:, :feat_dim_model]

    # Predict scaled price
    y_pred = float(cnn_lstm.predict(X_seq[np.newaxis, :, :], verbose=0).flatten()[0])
    last_f = X_seq[-1]
    return np.hstack([last_f, y_pred])

# ─── 4) BACKTEST LOOP ─────────────────────────────────────────────────────────
logger.info(f"Backtesting {BACKTEST_DATE} from {MARKET_OPEN.time()} to {MARKET_CLOSE.time()}...")
portfolio = {sym: {"cash": INITIAL_BALANCE, "shares": 0} for sym in TICKERS}
current_time = MARKET_OPEN

while current_time <= MARKET_CLOSE:
    print(current_time.strftime("%H:%M"), end="  ")
    for sym in TICKERS:
        df_all = bars_data.get(sym, pd.DataFrame())
        df_bar = df_all.loc[:current_time].tail(LOOKBACK_BARS)
        if len(df_bar) < LOOKBACK_BARS:
            continue

        try:
            price = float(df_bar["close"].iloc[-1])
            feats = extract_features_intraday(sym, df_bar)
            state = portfolio[sym]
            obs = np.concatenate([feats, [price, state["cash"], state["shares"]]])
            action, _ = agent.predict(obs.astype(np.float32), deterministic=True)

            if action == 1 and state["cash"] >= price:
                qty = int(state["cash"] // price)
                state["cash"] -= qty * price
                state["shares"] += qty
                print(f"[{sym}] BUY@{price:.2f}×{qty}", end="  ")
            elif action == 2 and state["shares"] > 0:
                qty = state["shares"]
                state["cash"] += qty * price
                state["shares"] = 0
                print(f"[{sym}] SELL@{price:.2f}×{qty}", end="  ")
        except Exception as e:
            logger.exception(f"Error processing {sym} at {current_time}")

    print()
    current_time += timedelta(minutes=1)

# ─── 5) FINAL SUMMARY ─────────────────────────────────────────────────────────
total_val = 0.0
print(f"\n💰 Final Portfolio on {BACKTEST_DATE}:")
for sym, state in portfolio.items():
    last_price = bars_data.get(sym, pd.DataFrame())["close"].iloc[-1] if not bars_data.get(sym, pd.DataFrame()).empty else 0.0
    val = state["cash"] + state["shares"] * last_price
    total_val += val
    print(f"  {sym}: Cash=${state['cash']:.2f}, Shares={state['shares']}, Last=${last_price:.2f}, Value=${val:.2f}")

print(f"\n🏁 Total Portfolio Value: ${total_val:.2f}\n")
