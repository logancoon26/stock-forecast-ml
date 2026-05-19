from .loader import load_data, train_val_test_split_by_date
from .preprocessing import preprocess
import pandas as pd
 
 
def get_data(
    tickers: str,
    start: str,
    end: str,
    split_date_val: str,
    split_date_test: str,
    feature_fn,
    window_size: int = 20,
    shuffle_train: bool = True
) -> dict:
    """
    Full data pipeline entry point. Loads, splits, generates features,
    and preprocesses train and validation data in one call.
 
    Args:
        ticker      (str):      Stock ticker e.g. "AAPL"
        start       (str):      Start date "YYYY-MM-DD"
        end         (str):      End date "YYYY-MM-DD"
        split_date  (str):      Train/test split date "YYYY-MM-DD"
        feature_fn  (callable): Function that takes a price array and returns
                                a feature matrix, shape (T, n_features).
                                e.g. return_all_features
        window_size (int):      Lookback window size.
 
    Returns:
        dict with keys:
            "signals_train", "X_train", "y_class_train", "y_reg_train",
            "signals_val",   "X_val",   "y_class_val",   "y_reg_val",
            "signal_scaler", "feat_scaler"
 
    """
    all_train, all_val, all_test = [], [], []
    # Load full data and split by date
    for ticker in tickers:
        try:
            _, df           = load_data(ticker, start, end)
            if len(df) < 100:
                print(f"Dataset too small for {ticker}")
                continue
        except Exception as e:
            print(f"No data found for {ticker}")
            continue

        df_train, df_val, df_test    = train_val_test_split_by_date(df, split_date_val, split_date_test)
    
        prices_train = df_train["Close"].to_numpy().flatten()
        prices_val = df_val["Close"].to_numpy().flatten()
        prices_test  = df_test["Close"].to_numpy().flatten()
    
        # Generate handcrafted features
        feats_train = feature_fn(prices_train)
        feats_val= feature_fn(prices_val)
        feats_test  = feature_fn(prices_test)
    
        # Preprocess train — fits scalers
        signals_train, X_train, y_class_train, y_reg_train, feat_scaler = preprocess(
            prices_train, feats_train, df_train, window_size
        )

        # Preprocess val — reuses train scalers 
        signals_val, X_val, y_class_val, y_reg_val, _ = preprocess(
            prices_val, feats_val, df_val, window_size,
            feat_scaler=feat_scaler,
        )
    
        # Preprocess test — reuses train scalers 
        signals_test, X_test, y_class_test, y_reg_test, _ = preprocess(
            prices_test, feats_test, df_test, window_size,
            feat_scaler=feat_scaler,
        )

        # Training Dataframe
        train_df = pd.DataFrame(X_train)
        train_df["y_class"] = y_class_train
        train_df["y_reg"] = y_reg_train

        # Add signal as Dataframe columns to enable data shuffling: signals shape (N, window_size)
        signals_df = pd.DataFrame(signals_train, columns=[f"w_sample_{i}" for i in range(window_size)])
        train_df = pd.concat([train_df, signals_df], axis=1)
        train_df["ticker"] = ticker

        # Testing Dataframe
        test_df = pd.DataFrame(X_test)
        test_df["y_class"] = y_class_test
        test_df["y_reg"] = y_reg_test

        # Add signal as Dataframe columns to enable data shuffling: signals shape (N, window_size)
        signals_df = pd.DataFrame(signals_test, columns=[f"w_sample_{i}" for i in range(window_size)])
        test_df = pd.concat([test_df, signals_df], axis=1)
        test_df["ticker"] = ticker

        # Validation Dataframe
        val_df = pd.DataFrame(X_val)
        val_df["y_class"] = y_class_val
        val_df["y_reg"] = y_reg_val

        # Add signal as Dataframe columns to enable data shuffling: signals shape (N, window_size)
        signals_df = pd.DataFrame(signals_val, columns=[f"w_sample_{i}" for i in range(window_size)])
        val_df = pd.concat([val_df, signals_df], axis=1)
        val_df["ticker"] = ticker

        # Append to lists (faster, will convert to dataframe after)
        all_train.append(train_df)
        all_val.append(val_df)
        all_test.append(test_df)

    # Form one Dataframe from list of data frames
    # Shuffle the training data
    if shuffle_train:
        train_df_total = pd.concat(all_train, ignore_index=True).sample(frac=1).reset_index(drop=True)
    else:
        train_df_total = pd.concat(all_train, ignore_index=True).reset_index(drop=True)
    val_df_total   = pd.concat(all_val,   ignore_index=True)
    test_df_total  = pd.concat(all_test,  ignore_index=True)
 
    return train_df_total, test_df_total, val_df_total

def get_inference_data(
    ticker: str,
    start: str,
    end: str,
    feature_fn,
    window_size: int = 20
) -> pd.DataFrame:
    """
    Simplified data pipeline for inference — no train/val/test split.
    Returns a single processed dataframe for the given ticker and date range.
    """
    _, df = load_data(ticker, start, end)
    print(df.columns.tolist())
    print(type(df.columns))

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    print(df.columns.tolist())
    print(type(df.columns))
    if len(df) < 100:
        raise ValueError(f"Not enough data for {ticker}")

    prices = df["Close"].to_numpy().flatten()
    feats  = feature_fn(prices, window=window_size)

    signals, X, y_class, y_reg, _ = preprocess(
        prices, feats, df, window_size
    )

    print("feats length:", len(feats))
    print("prices length:", len(prices))
    print("df length:", len(df))
    print("X length:", len(X))
    print("y_class length:", len(y_class))
    print("signals length:", len(signals))

    result_df = pd.DataFrame(X)
    result_df["y_class"] = y_class
    result_df["y_reg"]   = y_reg

    signals_df = pd.DataFrame(signals, columns=[f"w_sample_{i}" for i in range(window_size)])
    result_df  = pd.concat([result_df, signals_df], axis=1)
    result_df = result_df.reset_index(drop=True)
    result_df["ticker"] = pd.Series([ticker] * len(result_df)).values

    return result_df