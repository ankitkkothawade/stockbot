#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np, random, torch

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)


# In[2]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics      import mean_squared_error, mean_absolute_error
from tensorflow.keras.models     import Sequential
from tensorflow.keras.layers     import Conv1D, MaxPooling1D, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks  import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ─── CONFIG ────────────────────────────────────────────────────────────────────
TICKERS    = ["AAPL", "JPM", "AMZN", "TSLA", "MSFT"]
SEQ_LEN    = 60
TRAIN_YEAR = 2022

os.makedirs("models", exist_ok=True)
os.makedirs("processed_data", exist_ok=True)

def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)

# ─── DETERMINE MAX FEATURE DIMENSION ──────────────────────────────────────────
feature_dims = []
for ticker in TICKERS:
    tl = ticker.lower()
    X_raw = np.load(f"preprocessed_data/{tl}_raw_X.npy")
    feature_dims.append(X_raw.shape[1])
max_feat_dim = max(feature_dims)

# ─── 1) LOAD & PREPARE ALL TICKERS ─────────────────────────────────────────────
all_X_train, all_y_train = [], []
all_X_test,  all_y_test  = [], []
all_test_dates          = []

for ticker in TICKERS:
    tl = ticker.lower()

    # 1a) Load raw features & target
    X_raw = np.load(f"preprocessed_data/{tl}_raw_X.npy")   # (T, F_ticker)
    # pad to max_feat_dim
    if X_raw.shape[1] < max_feat_dim:
        pad = np.zeros((X_raw.shape[0], max_feat_dim - X_raw.shape[1]), dtype=X_raw.dtype)
        X_raw = np.hstack([X_raw, pad])

    y_raw = np.load(f"preprocessed_data/{tl}_raw_y.npy")   # (T,)

    # 1b) Load dates for train/test split
    df_tech = pd.read_csv(f"preprocessed_data/{tl}_tech_scaled_features.csv",
                          index_col=0, parse_dates=True)
    dates = df_tech.index

    # 1c) Train/Test boolean masks
    train_mask = dates.year <= TRAIN_YEAR
    test_mask  = dates.year >= (TRAIN_YEAR + 1)

    # 1d) Scale using train only
    scaler_X = MinMaxScaler().fit(X_raw[train_mask])
    X_scaled = scaler_X.transform(X_raw)

    scaler_y = MinMaxScaler().fit(y_raw[train_mask].reshape(-1,1))
    y_scaled = scaler_y.transform(y_raw.reshape(-1,1)).flatten()

    # 1e) Build sequences
    X_seq, y_seq   = create_sequences(X_scaled, y_scaled, SEQ_LEN)
    seq_dates      = dates[SEQ_LEN:]
    train_seq_mask = train_mask[SEQ_LEN:]
    test_seq_mask  = test_mask[SEQ_LEN:]

    # 1f) Accumulate
    all_X_train.append(X_seq[train_seq_mask])
    all_y_train.append(y_seq[train_seq_mask])
    all_X_test.append( X_seq[test_seq_mask])
    all_y_test.append( y_seq[test_seq_mask])
    all_test_dates.append(seq_dates[test_seq_mask])

# ─── 2) CONCATENATE ACROSS TICKERS ─────────────────────────────────────────────
X_train    = np.vstack(all_X_train)
y_train    = np.hstack(all_y_train)
X_test     = np.vstack(all_X_test)
y_test     = np.hstack(all_y_test)
test_dates = np.concatenate(all_test_dates)

# ─── 3) DEFINE & COMPILE SINGLE CNN–LSTM ───────────────────────────────────────
model = Sequential([
    Conv1D(32, 3, activation='relu', padding='same',
           input_shape=(SEQ_LEN, max_feat_dim)),
    MaxPooling1D(2),
    LSTM(64),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer=Adam(1e-4), loss='mse', metrics=['mae'])

# ─── 4) TRAIN WITH CALLBACKS ───────────────────────────────────────────────────
callbacks = [
    EarlyStopping(  monitor='val_loss', patience=8, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6),
    ModelCheckpoint("models/best_multi_stock_cnn_lstm.h5",
                    monitor='val_loss', save_best_only=True)
]
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=60,
    batch_size=32,
    callbacks=callbacks,
    verbose=2
)

# ─── 5) EVALUATE & PLOT ON COMBINED TEST SET ───────────────────────────────────
model.load_weights("models/best_multi_stock_cnn_lstm.h5")
y_pred = model.predict(X_test).flatten()

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
print(f"Combined Test RMSE: {rmse:.6f}   MAE: {mae:.6f}")

start_idx = 0
for ticker, y_true_part, dates_part in zip(TICKERS, all_y_test, all_test_dates):
    n = len(y_true_part)
    y_pred_part = y_pred[start_idx : start_idx + n]

    # Per-ticker metrics
    rmse_scaled = np.sqrt(mean_squared_error(y_true_part, y_pred_part))
    mae_scaled  = mean_absolute_error(y_true_part, y_pred_part)
    print(f"{ticker} Test RMSE (scaled): {rmse_scaled:.6f}   MAE (scaled): {mae_scaled:.6f}")

    # Plot Actual vs Predicted
    plt.figure(figsize=(12, 5))
    plt.plot(dates_part, y_true_part,  label='Actual',   alpha=0.7)
    plt.plot(dates_part, y_pred_part,  label='Predicted', alpha=0.7)
    plt.title(f"{ticker} CNN–LSTM Actual vs Predicted (Test)\nRMSE={rmse_scaled:.4f}")
    plt.xlabel("Date")
    plt.ylabel("Scaled Close")
    plt.legend()
    plt.tight_layout()
    plt.show()

    start_idx += n

# ─── 6) SAVE FINAL MODEL & STATE MATRICES ────────────────────────────────────
model.save("models/final_multi_stock_cnn_lstm.h5")
print("✅ Saved multi-stock CNN-LSTM model to models/final_multi_stock_cnn_lstm.h5")

# 6b) Save per-ticker RL state matrices
start_idx = 0
for ticker, X_test_part, dates_part in zip(TICKERS, all_X_test, all_test_dates):
    tl = ticker.lower()
    length = X_test_part.shape[0]
        # CORRECT: extract only the last time‐step features for each sequence
    X_seq_block = X_test[start_idx:start_idx+length]    # shape (n, SEQ_LEN, feat)
    X_base      = X_seq_block[:, -1, :]                  # shape (n, feat)
    y_pred_part = y_pred[start_idx:start_idx+length].reshape(-1,1)  # (n,1)
    state_matrix = np.hstack([X_base, y_pred_part])      # (n, feat+1)

    np.save(f"processed_data/{tl}_state_cnnlstm_test.npy", state_matrix)
    print(f"✅ Saved {tl} RL state matrix to processed_data/{tl}_state_cnnlstm_test.npy")
    start_idx += length

