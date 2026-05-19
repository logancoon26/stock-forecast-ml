import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Return all features as DataFrame

def return_all_features(signal: np.ndarray, window: int = 20) -> pd.DataFrame:
    """Return a single DataFrame with time-domain and frequency-domain features."""
    time_df = rolling_time_features(signal, window=window)
    freq_df = fft_features(signal, window=window)
    return pd.concat([time_df, freq_df], axis=1)

def make_transformer_sequences(df, seq_len):
    sequences = None

    for ticker, group in df.groupby("ticker"):
        for iSeq in range(len(group)//seq_len):
            if sequences is None:
                sequences = np.expand_dims(group.iloc[iSeq*seq_len:(iSeq+1)*seq_len, :-1].to_numpy(), axis = 0)
            else:
                sequences = np.concatenate((sequences, np.expand_dims(group.iloc[iSeq*seq_len:(iSeq+1)*seq_len, :-1].to_numpy(), axis = 0)), axis = 0)

    return sequences

# Rolling time-domain features 

def rolling_time_features(signal: np.ndarray, window: int = 20) -> pd.DataFrame:
    """Mean, std, skewness, kurtosis over a rolling window."""
    s = pd.Series(signal)
    df = pd.DataFrame({
    "mean": s.rolling(window).mean(),
    "std": s.rolling(window).std(),
    "skew": s.rolling(window).skew(),
    "kurtosis": s.rolling(window).kurt(),
    })
    return df

#  Windowed FFT features (same length as rolling time features) 

def fft_features(signal: np.ndarray, window: int = 20) -> pd.DataFrame:
    """
    Compute FFT features over a rolling window so the output length
    matches rolling_time_features(..., window=window).

    Rows 0 .. window-2 are NaN (mirrors pandas rolling behaviour).
    """
    n = len(signal)
    freq_ratio = np.full(n, np.nan)
    dominant_freq = np.full(n, np.nan)
    spec_entropy = np.full(n, np.nan)

    for i in range(window - 1, n):
        segment = signal[i - window + 1 : i + 1]
        power = compute_power_spectrum(segment)

        freq_ratio[i] = low_high_freq_ratio(power)
        dominant_freq[i] = dominant_frequency(power)
        spec_entropy[i] = spectral_entropy(power)

    return pd.DataFrame({
    "freq_ratio": freq_ratio,
    "dominant_freq": dominant_freq,
    "spectral_entropy": spec_entropy,
    })


# FFT helpers (operate on a 1-D array) 

def compute_power_spectrum(signal: np.ndarray) -> np.ndarray:
    signal = signal - np.mean(signal)
    fft_vals = np.fft.fft(signal)
    power = np.abs(fft_vals) ** 2
    return power[: len(power) // 2]


def low_high_freq_ratio(power: np.ndarray, split_var: int = 4) -> float:
    split = len(power) // split_var
    low = power[:split].sum()
    high = power[(split_var - 1) * split :].sum()
    return high / (low + 1e-10)


def dominant_frequency(power: np.ndarray) -> float:
    return float(np.argmax(power[1:]) + 1)/len(power) 


def spectral_entropy(power: np.ndarray) -> float:
    prob = power / (np.sum(power) + 1e-8)
    return float(-np.sum(prob * np.log(prob + 1e-8)))