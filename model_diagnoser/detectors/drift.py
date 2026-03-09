import numpy as np
from scipy.stats import ks_2samp
import pandas as pd


def detect_data_drift(X_train, X_test, threshold=0.05):
    """
    Detects data drift using Kolmogorov-Smirnov test.
    """

    drifted_features = []

    if not isinstance(X_train, pd.DataFrame):
        return {"issue": "Drift check skipped (X_train not DataFrame)"}

    for column in X_train.columns:

        train_values = X_train[column].dropna()
        test_values = X_test[column].dropna()

        stat, p_value = ks_2samp(train_values, test_values)

        if p_value < threshold:
            drifted_features.append(column)

    if drifted_features:
        return {
            "issue": "Data drift detected",
            "features": drifted_features,
            "severity": "medium"
        }

    return {"issue": "No significant data drift detected"}