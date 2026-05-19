import numpy as np

def compute_power_spectrum(signal):
    # Remove mean to avoid DC component dominating
    signal = signal - np.mean(signal)
    fft_vals = np.fft.fft(signal)
    power = np.abs(fft_vals)**2
    return power[:len(power)//2]

def make_signal_windows(
    signal: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Slides a window across the full price signal to create samples.
 
    Args:
        prices      (np.ndarray): Raw price signal, shape (T,)
        window_size (int):        Number of timesteps per sample
 
    Returns:
        np.ndarray: shape (N, window_size) where N = T - window_size
 
    Example:
        signals = make_windows(prices, window_size=20)
        # signals[0] = prices[0:20]
        # signals[1] = prices[1:21]
        # signals[2] = prices[2:22]
    """
    N = len(signal) - window_size
 
    signals_out  = np.stack([signal[t : t + window_size] for t in range(N)])   # (N, window_size)
 
    return signals_out
