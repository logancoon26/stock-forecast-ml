import yfinance as yf
import pandas as pd
import numpy as np


def load_data(
    ticker: str,
    start: str,
    end: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Downloads historical stock data for a given ticker and date range.

    Args:
        ticker (str): Stock ticker symbol e.g. "AAPL"
        start  (str): Start date in "YYYY-MM-DD" format
        end    (str): End date in "YYYY-MM-DD" format

    Returns:
        prices (np.ndarray):  Close prices, shape (T,)
        df     (pd.DataFrame): DataFrame with DatetimeIndex

    """
    df = yf.download(ticker, start=start, end=end, progress=False)

    if df.empty:
        raise ValueError(f"No data found for ticker '{ticker}' between {start} and {end}.")

    prices = df["Close"].to_numpy().flatten()

    return prices, df


def train_val_test_split_by_date(
    df: pd.DataFrame,
    split_date_val: str,
    split_date_test: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits a DataFrame into train and test sets by date.

    Args:
        df         (pd.DataFrame): DataFrame with DatetimeIndex
        split_date (str):          Date to split on in "YYYY-MM-DD" format.
                                   Everything before this date is train, after is test.

    Returns:
        df_train (pd.DataFrame): Training data
        df_test  (pd.DataFrame): Test data

    Example:
        df_train, df_test = train_test_split_by_date(df, "2022-01-01")
    """
    df_train = df[df.index < split_date_val]
    df_val   = df[(df.index >= split_date_val) & (df.index < split_date_test)]
    df_test  = df[df.index >= split_date_test]
    
    return df_train, df_val, df_test