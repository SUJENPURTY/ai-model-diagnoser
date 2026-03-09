import pandas as pd
import numpy as np


def ensure_dataframe(X):
    """
    Ensure input data is converted to pandas DataFrame.
    """

    if isinstance(X, pd.DataFrame):
        return X

    if isinstance(X, np.ndarray):
        return pd.DataFrame(X)

    return pd.DataFrame(X)


def ensure_series(y):
    """
    Ensure target variable is pandas Series.
    """

    if isinstance(y, pd.Series):
        return y

    return pd.Series(y)


def dataset_summary(X):
    """
    Return basic dataset information.
    """

    X = ensure_dataframe(X)

    summary = {
        "num_rows": X.shape[0],
        "num_columns": X.shape[1],
        "missing_values": X.isnull().sum().to_dict(),
        "data_types": X.dtypes.astype(str).to_dict()
    }

    return summary


def check_missing_values(X):
    """
    Detect missing values in dataset.
    """

    X = ensure_dataframe(X)

    missing = X.isnull().sum()

    missing = missing[missing > 0]

    return missing.to_dict()