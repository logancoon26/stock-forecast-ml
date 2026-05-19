import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from src.data import get_data
from src.data.features import return_all_features
from src.models.model_library import get_model

device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    # -----------------------------
    # Load Data
    # -----------------------------
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
    
    data_train, data_test, data_val = get_data(
        TICKERS,
        "2012-01-01",
        "2026-01-01",
        split_date_val="2021-01-01",
        split_date_test="2023-01-01",
        feature_fn=return_all_features
    )

    # -----------------------------
    # Build Model
    # -----------------------------
    model = get_model(
        "hybrid",
        signal_length=data_train.iloc[:, 9:].shape[1],
        n_handcrafted=data_train.iloc[:, 0:7].shape[1]
    ).to(device)

    # -----------------------------
    # Train
    # -----------------------------
    history = model.fit(
        data_train.iloc[:, 9:-1].to_numpy(),
        data_train.iloc[:, 0:7].to_numpy(),
        data_train.iloc[:, 7].to_numpy(),
        data_val.iloc[:, 9:-1].to_numpy(),
        data_val.iloc[:, 0:7].to_numpy(),
        data_val.iloc[:, 7].to_numpy(),
        batch_size=64,
        epochs=8
    )

    # -----------------------------
    # Evaluate
    # -----------------------------
    pred, prob = model.evaluate(
        data_test.iloc[:, 9:-1].to_numpy(),
        data_test.iloc[:, 0:7].to_numpy()
    )

    acc = accuracy_score(data_test.iloc[:, 7].to_numpy(), pred)
    print("Test Accuracy:", acc)

    # -----------------------------
    # Save model
    # -----------------------------
    torch.save(model.state_dict(), "model.pth")
    print("Model saved to model.pth")


if __name__ == "__main__":
    main()