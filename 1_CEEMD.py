#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.signal    import detrend
from ta.momentum     import RSIIndicator
from ta.trend        import MACD
from ta.volatility   import BollingerBands
from PyEMD           import CEEMDAN

# ─── CONFIG ────────────────────────────────────────────────────────────────────
TICKERS = ["AAPL", "JPM", "AMZN", "TSLA", "MSFT"]
START   = "2000-01-01"
END     = "2025-01-01"
OUTDIR  = "preprocessed_data"
os.makedirs(OUTDIR, exist_ok=True)

# ─── LOOP OVER TICKERS ─────────────────────────────────────────────────────────
for ticker in TICKERS:
    # 1) Download price & volume
    df = yf.download(ticker, start=START, end=END, progress=False)
    # flatten any MultiIndex (e.g. from yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)

    close  = df["Close"]
    volume = df["Volume"]

    # 2) Compute technical indicators
    rsi      = RSIIndicator(close=close, window=14, fillna=True).rsi()
    macd_obj = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    bb_obj   = BollingerBands(close=close, window=20, window_dev=2)

    tech = pd.concat([
        volume.rename("Volume"),
        rsi.rename("RSI"),
        macd_obj.macd().rename("MACD"),
        macd_obj.macd_signal().rename("MACD_Signal"),
        bb_obj.bollinger_hband().rename("BB_High"),
        bb_obj.bollinger_lband().rename("BB_Low"),
    ], axis=1).ffill().bfill()

    # 3) CEEMDAN on detrended close
    sig      = close.to_numpy(dtype=np.float64)
    sig_dt   = detrend(sig)
    ceemdan  = CEEMDAN(trials=100, noise_width=0.05, max_imf=6, parallel=True, n_jobs=-1)
    time     = np.arange(len(sig_dt), dtype=np.float64)
    imfs     = ceemdan.ceemdan(sig_dt, time)  # shape (n_imfs, T)

    # 4) Select IMFs by energy & correlation
    energies      = (imfs ** 2).sum(axis=1)
    energy_ratios = energies / energies.sum()
    cors          = np.array([np.corrcoef(imf, sig_dt)[0,1] for imf in imfs])
    keep          = (energy_ratios >= 0.02) & (np.abs(cors) >= 0.1)
    sel_imfs      = imfs[keep]
    imf_chan      = sel_imfs.T  # shape (T, n_sel)

    # 5) Build X and y
    X_raw = np.hstack([imf_chan, tech.values])       # features
    y_raw = close.to_numpy(dtype=np.float32)         # target

    # 6) Save per-ticker outputs
    tl = ticker.lower()
    np.save(os.path.join(OUTDIR, f"{tl}_all_imfs.npy"),       imfs)
    np.save(os.path.join(OUTDIR, f"{tl}_filtered_imfs.npy"),  sel_imfs)
    np.save(os.path.join(OUTDIR, f"{tl}_denoised_signal.npy"), sel_imfs.sum(axis=0))
    tech.to_csv(os.path.join(OUTDIR, f"{tl}_tech_scaled_features.csv"))
    np.save(os.path.join(OUTDIR, f"{tl}_raw_X.npy"),          X_raw)
    np.save(os.path.join(OUTDIR, f"{tl}_raw_y.npy"),          y_raw)

    print(f"✅ {ticker} preprocessed → files in {OUTDIR}/")

