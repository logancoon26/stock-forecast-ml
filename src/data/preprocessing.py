import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
 
 
def make_windows(prices: np.ndarray, window_size: int = 20) -> np.ndarray:
    """
    Slides a window across the full price signal to create samples.
 
    Args:
        prices      (np.ndarray): Raw price signal, shape (T,)
        window_size (int):        Number of timesteps per sample. Default: 20
 
    Returns:
        np.ndarray: shape (N, window_size) where N = T - window_size
 
    Example:
        signals = make_windows(prices, window_size=20)
        # signals[0] = prices[0:20]
        # signals[1] = prices[1:21]
    """
    N = len(prices) - window_size - 1
    return np.stack([prices[t : t + window_size] for t in range(N)])
 
 
def make_labels(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates classification and regression labels from a price DataFrame.
 
    Classification: 1 if next close is higher than current, 0 otherwise.
    Regression:     Percentage change from current close to next close.
 
    Args:
        df (pd.DataFrame): DataFrame with a Close column.
 
    Returns:
        y_class (np.ndarray): Binary labels, shape (T,)
        y_reg   (np.ndarray): Percentage change labels, shape (T,)
    """
    close = df["Close"]
 
    y_class = (close.shift(-1) > close).astype(int)
    y_reg   = close.pct_change().shift(-1).astype(float)
 
    # Drop last row (NaN from shift) and reset index
    y_class = y_class.iloc[:-1].reset_index(drop=True).to_numpy()
    y_reg   = y_reg.iloc[:-1].reset_index(drop=True).to_numpy()
 
    return y_class, y_reg
 
 
def preprocess(
    prices: np.ndarray,
    features: np.ndarray,
    df: pd.DataFrame,
    window_size: int = 20,
    feat_scaler: StandardScaler = None,
) -> tuple:
    """
    Windows the price signal, aligns features and labels, and normalizes inputs.
 
    Args:
        prices        (np.ndarray):    Raw close prices, shape (T,)
        features      (np.ndarray):    Handcrafted features, shape (T, n_features)
        df            (pd.DataFrame):  DataFrame for label generation
        window_size   (int):           Window size.
        signal_scaler (StandardScaler): Pre-fit scaler for signals. If None, fits a new one.
        feat_scaler   (StandardScaler): Pre-fit scaler for features. If None, fits a new one.
 
    Returns:
        signals       (np.ndarray): Normalized price windows, shape (N, window_size)
        X             (np.ndarray): Normalized handcrafted features, shape (N, n_features)
        y_class       (np.ndarray): Classification labels, shape (N,)
        y_reg         (np.ndarray): Regression labels, shape (N,)
        signal_scaler (StandardScaler): Fitted signal scaler (save for inference)
        feat_scaler   (StandardScaler): Fitted feature scaler (save for inference)
 
        )
    """
    # Window the signal
    signals = make_windows(prices, window_size)         # (N, window_size)
 
    # Generate labels and trim to match windowed signals
    y_class, y_reg = make_labels(df)
    n = len(signals)
    y_class = y_class[-n:]
    y_reg   = y_reg[-n:]
 
    # Align features to windowed signals
    X = features[-n:]                                   # (N, n_features)
 
    # Normalize signal windows on a "per-window" basis
    mean = signals.mean(axis=1, keepdims=True)
    std  = signals.std(axis=1,  keepdims=True)
    signals = (signals - mean) / std
 
    # Normalize features w/ StandardScaler
    if feat_scaler is None:
        feat_scaler = StandardScaler()
        X = feat_scaler.fit_transform(X)
    else:
        X = feat_scaler.transform(X)
 
    return signals, X, y_class, y_reg, feat_scaler