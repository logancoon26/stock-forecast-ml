import torch
import numpy as np
import pandas as pd

from src.data import get_data
from src.data.features import return_all_features
from src.models.model_library import get_model

device = "cuda" if torch.cuda.is_available() else "cpu"

TICKERS = [
        # Big Tech
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "ORCL", "INTC", "AMD",

        # Financials
        "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK", "SCHW", "COF",

        # Healthcare
        "JNJ", "UNH", "PFE", "ABBV", "MRK", "TMO", "ABT", "AMGN", "GILD", "CVS",

        # Consumer
        "WMT", "HD", "MCD", "SBUX", "NKE", "TGT", "COST", "LOW", "TJX", "BKNG",

        # Industrials
        "HON", "UPS", "CAT", "GE", "BA", "LMT", "RTX", "DE", "MMM", "EMR",

        # Energy
        "XOM", "CVX", "COP", "SLB", "OXY", "MPC", "PSX", "VLO", "HAL", "EOG",

        # Communications
        "T", "VZ", "TMUS", "CMCSA", "DIS", "NFLX", "V", "MA", "PYPL", "ADBE",
    ]

_, data_test, _ = get_data(
        TICKERS,
        "2012-01-01",
        "2026-01-01",
        split_date_val="2021-01-01",
        split_date_test="2023-01-01",
        feature_fn=return_all_features
    )

def load_model():
    # Recreate model architecture
    model = get_model(
        "hybrid",
        signal_length=data_test.iloc[:, 9:].shape[1],
        n_handcrafted=data_test.iloc[:, 0:7].shape[1]
    ).to(device)

    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()

    return model


def predict(model, X_signal, X_features):
    """
    X_signal: np.array (same format as training)
    X_features: np.array (same format as training)
    """
    with torch.no_grad():
        pred, prob = model.evaluate(X_signal, X_features)
    return pred, prob


def main():
    model = load_model()

    # Predict the next day price movement (use final set of test data to predict next day)
    X_signal = data_test.iloc[-1:, 9:-1].to_numpy()
    X_features = data_test.iloc[-1:, 0:7].to_numpy()

    pred, prob = predict(model, X_signal, X_features)

    print("Predictions:", pred)
    print("Probabilities:", prob)


if __name__ == "__main__":
    main()