#!/usr/bin/env python3
"""
deploy.py

Intraday deployment script (1-minute bars) that:
1. Fetches recent OHLCV bars from Alpaca
2. Computes technicals + CEEMDAN IMFs
3. Builds the same SEQ_LEN-window features you used in training
4. Runs CNN–LSTM to get last-step features + scaled price prediction
5. Feeds that into your PPO agent to pick 0/hold,1/buy,2/sell
6. Submits market orders on Alpaca paper API
"""

import os
import time
import numpy as np
import pandas as pd

from scipy.signal            import detrend
from PyEMD                    import CEEMDAN
from ta.momentum             import RSIIndicator
from ta.trend                import MACD
from ta.volatility           import BollingerBands
from sklearn.preprocessing   import MinMaxScaler
from tensorflow.keras.models import load_model
from stable_baselines3       import PPO
from alpaca_trade_api.rest   import REST, TimeFrame

# ─── 0) USER CONFIG ─────────────────────────────────────────────────────────────
TICKERS         = ["AAPL","JPM","AMZN","TSLA","MSFT"]
SEQ_LEN         = 60
INITIAL_BALANCE = 10_000
TRAIN_YEAR      = 2022

ALPACA_API_KEY    = "PK4NLA1IF8I62FZVEJMT"
ALPACA_SECRET_KEY = "U9bNYdnf4nWRGxZ3bjUVSAimfSP8d3bwHNa5r3S9"
ALPACA_BASE_URL   = "https://paper-api.alpaca.markets"

# For 1-minute bars; change to TimeFrame.Minute5 or TimeFrame.Hour for other granularities
TIMEFRAME      = TimeFrame.Minute
LOOKBACK_BARS  = SEQ_LEN + 20
SLEEP_INTERVAL = 60    # seconds between loops (≈1 bar)

# ─── 1) LOAD OFFLINE MODELS & RECONSTRUCT SCALERS ───────────────────────────────
# 1a) CNN–LSTM feature→scaled-price predictor
cnn_lstm = load_model("models/final_multi_stock_cnn_lstm.h5")

# 1b) PPO agent
agent    = PPO.load("cnn_lstm_multi_stock_ppo.zip")

# 1c) Re-build each ticker’s MinMax scaler exactly as in training
raw_X     = {}
dates_idx = {}
scaler_X  = {}

for ticker in TICKERS:
    tl        = ticker.lower()
    X_full    = np.load(f"preprocessed_data/{tl}_raw_X.npy")                  # (T, F_ticker)
    df_tech   = pd.read_csv(f"preprocessed_data/{tl}_tech_scaled_features.csv",
                            index_col=0, parse_dates=True)
    idx       = df_tech.index
    train_mask= idx.year <= TRAIN_YEAR

    scaler    = MinMaxScaler().fit(X_full[train_mask.values])
    raw_X[ticker]   = X_full
    dates_idx[ticker] = idx
    scaler_X[ticker]  = scaler

# determine max feature dimension (for padding)
max_feat_dim = max(X.shape[1] for X in raw_X.values())

# ─── 2) ALPACA CLIENT SETUP ─────────────────────────────────────────────────────
api = REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version='v2')

# ─── 3) HELPERS ─────────────────────────────────────────────────────────────────
def compute_technical(df: pd.DataFrame) -> pd.DataFrame:
    """Compute RSI, MACD, Bollinger Bands on df['close'], df['volume']."""
    close  = df["close"]
    volume = df["volume"]
    rsi    = RSIIndicator(close=close, window=14, fillna=True).rsi()
    macd_o = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    bb     = BollingerBands(close=close, window=20, window_dev=2)

    tech = pd.concat([
        volume.rename("volume"),
        rsi.rename("rsi"),
        macd_o.macd().rename("macd"),
        macd_o.macd_signal().rename("macd_signal"),
        bb.bollinger_hband().rename("bb_high"),
        bb.bollinger_lband().rename("bb_low"),
    ], axis=1).ffill().bfill()
    return tech

def compute_ceemdan_imfs(close: pd.Series) -> np.ndarray:
    """Detrend & run CEEMDAN, select IMFs by energy/correlation thresholds."""
    sig    = close.to_numpy(dtype=np.float64)
    sig_dt = detrend(sig)
    ceemd  = CEEMDAN(trials=100, noise_width=0.05, max_imf=6, parallel=True, n_jobs=-1)
    imfs   = ceemd.ceemdan(sig_dt, np.arange(len(sig_dt), dtype=float))
    energies      = (imfs**2).sum(axis=1)
    ratios        = energies / energies.sum()
    cors          = np.array([np.corrcoef(imf, sig_dt)[0,1] for imf in imfs])
    keep          = (ratios>=0.02) & (np.abs(cors)>=0.1)
    return imfs[keep]

def extract_features_intraday(ticker: str, df: pd.DataFrame) -> np.ndarray:
    """
    Given a DataFrame of at least LOOKBACK_BARS rows with cols ['open','high','low','close','volume'],
    compute tech + CEEMDAN, pad, scale, CNN–LSTM predict, and return (feat_dim+1,) vector.
    """
    tech      = compute_technical(df)
    imfs      = compute_ceemdan_imfs(df["close"])
    imf_chan  = imfs.T                          # (T, n_sel)
    X_raw     = np.hstack([imf_chan, tech.values])  # (T, feat_ticker)

    # pad to max_feat_dim
    if X_raw.shape[1] < max_feat_dim:
        pad    = np.zeros((X_raw.shape[0], max_feat_dim - X_raw.shape[1]))
        X_raw  = np.hstack([X_raw, pad])

    # scale
    X_scaled = scaler_X[ticker].transform(X_raw)   # (T, max_feat_dim)
    X_seq    = X_scaled[-SEQ_LEN:]                 # last SEQ_LEN rows
    y_pred   = float(cnn_lstm.predict(X_seq[np.newaxis,:,:]).flatten()[0])
    last_f   = X_seq[-1]                           # (max_feat_dim,)

    return np.hstack([last_f, y_pred])             # (max_feat_dim+1,)

def get_recent_bars(sym: str) -> pd.DataFrame:
    """
    Fetch the last LOOKBACK_BARS bars from Alpaca at TIMEFRAME.
    Returns df with index=time and columns ['open','high','low','close','volume'].
    """
    bars = api.get_bars(
        symbol    = sym,
        timeframe = TIMEFRAME,
        limit     = LOOKBACK_BARS,
        adjustment= "raw"
    )
    data = [{
        "t": b.t, "open": b.o, "high": b.h,
        "low": b.l, "close": b.c, "volume": b.v
    } for b in bars]
    df = pd.DataFrame(data).set_index("t")
    return df

# ─── 4) PORTFOLIO TRACKING ─────────────────────────────────────────────────────
portfolio = { sym: {"cash": INITIAL_BALANCE, "shares": 0} for sym in TICKERS }

# ─── 5) MAIN LOOP ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("▶️  Starting intraday live deploy. Ctrl+C to exit.")
    try:
        while True:
            for sym in TICKERS:
                df_bar = get_recent_bars(sym)
                price  = float(df_bar["close"].iloc[-1])

                feats  = extract_features_intraday(sym, df_bar)
                state  = portfolio[sym]
                obs    = np.concatenate([
                    feats,
                    [price, state["cash"], state["shares"]]
                ]).astype(np.float32)

                action, _ = agent.predict(obs, deterministic=True)
                # 0=Hold, 1=Buy, 2=Sell

                if action == 1 and state["cash"] >= price:
                    qty = int(state["cash"] // price)
                    if qty > 0:
                        api.submit_order(
                            symbol        = sym,
                            qty           = qty,
                            side          = "buy",
                            type          = "market",
                            time_in_force = "gtc"
                        )
                        state["cash"]  -= qty * price
                        state["shares"]+= qty
                        print(f"[{sym}] BUY  {qty} @ {price:.2f}")

                elif action == 2 and state["shares"] > 0:
                    qty = int(state["shares"])
                    api.submit_order(
                        symbol        = sym,
                        qty           = qty,
                        side          = "sell",
                        type          = "market",
                        time_in_force = "gtc"
                    )
                    state["cash"]  += qty * price
                    state["shares"] = 0
                    print(f"[{sym}] SELL {qty} @ {price:.2f}")

            print(f"⏱  Sleeping {SLEEP_INTERVAL}s …\n")
            time.sleep(SLEEP_INTERVAL)

    except KeyboardInterrupt:
        print("🛑  Stop by user")

    except Exception as e:
        print("⚠️  Error in loop:", e)
        time.sleep(5)
