#!/usr/bin/env python3
"""
Deploy_backtest.py

Backtesting deployment script using historical daily Alpaca data.
Simulates the same model + RL agent over 2023–2024 with a lookback window.
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

# ─── UTF-8 OUTPUT ─────────────────────────────────────────────────────────────
# allow emojis/unicode in Windows consoles
sys.stdout.reconfigure(encoding='utf-8')

# ─── LOGGING SETUP ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def main():
    # ─── 0) CONFIG ──────────────────────────────────────────────────────────────
    TICKERS        = ["AAPL", "JPM", "AMZN", "TSLA", "MSFT"]
    SEQ_LEN        = 60
    LOOKBACK_BARS  = SEQ_LEN + 20
    INITIAL_BALANCE = 10_000.0
    TRAIN_YEAR     = 2022

    # Alpaca credentials from environment
    APCA_KEY    = os.getenv("APCA_API_KEY_ID")
    APCA_SECRET = os.getenv("APCA_API_SECRET_KEY")
    APCA_URL    = os.getenv(
        "APCA_BASE_URL", "https://paper-api.alpaca.markets/v2"
    )
    if not APCA_KEY or not APCA_SECRET:
        logger.error("Please set APCA_API_KEY_ID and APCA_API_SECRET_KEY env vars")
        sys.exit(1)

    # Backtest window (daily bars)
    BACKTEST_START = "2023-01-01"
    BACKTEST_END   = "2024-12-31"
    TIMEFRAME      = TimeFrame.Day

    # ─── 1) INIT MODELS & SCALERS ─────────────────────────────────────────────
    logger.info("Loading models and scalers…")
    api       = REST(APCA_KEY, APCA_SECRET, APCA_URL, api_version="v2")
    cnn_lstm  = load_model("models/final_multi_stock_cnn_lstm.h5", compile=False)
    agent     = PPO.load("cnn_lstm_multi_stock_ppo.zip")

    raw_X     = {}
    dates_idx = {}
    scaler_X  = {}
    for ticker in TICKERS:
        tl       = ticker.lower()
        X_full   = np.load(f"preprocessed_data/{tl}_raw_X.npy")  # (T, F_ticker)
        raw_X[ticker] = X_full

        df_tech = pd.read_csv(
            f"preprocessed_data/{tl}_tech_scaled_features.csv",
            index_col=0, parse_dates=True
        )
        dates_idx[ticker] = df_tech.index
        mask = df_tech.index.year <= TRAIN_YEAR
        scaler_X[ticker] = MinMaxScaler().fit(X_full[mask])

    # determine expected model input dims
    _, seq_model, feat_model = cnn_lstm.input_shape
    max_feat_dim = max(x.shape[1] for x in raw_X.values())
    logger.info(f"Model expects sequence length={seq_model}, features={feat_model}")

    # ─── 2) PRE-FETCH DAILY BARS ────────────────────────────────────────────────
    logger.info(f"Fetching daily bars {BACKTEST_START} → {BACKTEST_END}")
    bars_data = {}
    for sym in TICKERS:
        raw = api.get_bars(
            symbol    = sym,
            timeframe = TIMEFRAME,
            start     = BACKTEST_START + "T00:00:00Z",
            end       = BACKTEST_END   + "T23:59:59Z",
            adjustment= "raw",
        )
        df = pd.DataFrame([{
            "date": b.t.date(),
            "open": b.o, "high": b.h,
            "low": b.l,   "close": b.c,
            "volume": b.v
        } for b in raw])
        if not df.empty:
            df.set_index("date", inplace=True)
            df.sort_index(inplace=True)
        bars_data[sym] = df

    # ─── 3) FEATURE EXTRACTION ─────────────────────────────────────────────────
    def extract_features(ticker: str, df: pd.DataFrame) -> np.ndarray:
        # align the precomputed raw_X to these dates
        pos = dates_idx[ticker].get_indexer(df.index)
        X_win = raw_X[ticker][pos]  # (LOOKBACK_BARS, F_ticker)

        # pad/truncate to scaler input dims
        n_raw = scaler_X[ticker].n_features_in_
        if X_win.shape[1] < n_raw:
            pad = np.zeros((X_win.shape[0], n_raw - X_win.shape[1]))
            X_win = np.hstack([X_win, pad])
        elif X_win.shape[1] > n_raw:
            X_win = X_win[:, :n_raw]

        # scale
        X_scl = scaler_X[ticker].transform(X_win)

        # pad/truncate to max_feat_dim
        if X_scl.shape[1] < max_feat_dim:
            pad2 = np.zeros((X_scl.shape[0], max_feat_dim - X_scl.shape[1]))
            X_scl = np.hstack([X_scl, pad2])
        elif X_scl.shape[1] > max_feat_dim:
            X_scl = X_scl[:, :max_feat_dim]

        # slice sequence
        X_seq = X_scl[-SEQ_LEN:]

        # align to model’s feat dim
        if X_seq.shape[1] != feat_model:
            if X_seq.shape[1] < feat_model:
                pad3 = np.zeros((SEQ_LEN, feat_model - X_seq.shape[1]))
                X_seq = np.hstack([X_seq, pad3])
            else:
                X_seq = X_seq[:, :feat_model]

        # predict
        y_pred = float(cnn_lstm.predict(X_seq[np.newaxis, :, :], verbose=0).flatten()[0])
        last_f = X_seq[-1]
        return np.hstack([last_f, y_pred])

    # ─── 4) BACKTEST LOOP ───────────────────────────────────────────────────────
    # find trading dates common to all symbols
    common_dates = sorted(
        set(bars_data[TICKERS[0]].index)
        .intersection(*(bars_data[s].index for s in TICKERS[1:]))
    )
    logger.info(f"Running backtest on {len(common_dates)} days")
    portfolio = {s: {"cash": INITIAL_BALANCE, "shares": 0} for s in TICKERS}

    for current_date in common_dates:
        print(current_date, end="  ")
        for sym in TICKERS:
            df_all = bars_data[sym]
            window = df_all.loc[:current_date].tail(LOOKBACK_BARS)
            if len(window) < LOOKBACK_BARS:
                continue

            price = float(window["close"].iloc[-1])
            feats = extract_features(sym, window)
            state = portfolio[sym]
            obs   = np.concatenate([feats, [price, state["cash"], state["shares"]]])
            action, _ = agent.predict(obs.astype(np.float32), deterministic=True)

            if action == 1 and state["cash"] >= price:
                qty = int(state["cash"] // price)
                state["cash"]  -= qty * price
                state["shares"]+= qty
                print(f"[{sym}] BUY×{qty}", end="  ")
            elif action == 2 and state["shares"] > 0:
                qty = state["shares"]
                state["cash"]  += qty * price
                state["shares"] = 0
                print(f"[{sym}] SELL×{qty}", end="  ")
        print()

    # ─── 5) FINAL SUMMARY ───────────────────────────────────────────────────────
    total_val = 0.0
    print("\n💰 Final Portfolio Value:")
    for sym, s in portfolio.items():
        price = float(bars_data[sym]["close"].iloc[-1]) if not bars_data[sym].empty else 0.0
        val   = s["cash"] + s["shares"] * price
        total_val += val
        print(f"  {sym}: Cash=${s['cash']:.2f}, Shares={s['shares']}, Last=${price:.2f}, Value=${val:.2f}")
    print(f"\n🏁 Total: ${total_val:.2f}\n")

if __name__ == "__main__":
    main()
