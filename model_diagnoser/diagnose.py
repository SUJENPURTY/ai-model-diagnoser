import pandas as pd
import numpy as np

from .detectors import (
    detect_overfitting,
    detect_class_imbalance,
    detect_data_leakage,
    detect_data_drift
)

from .metrics.performance import evaluate_model


def run_diagnostics(model, X_train, X_test, y_train, y_test):
    """
    Run all diagnostics on the machine learning model and dataset.
    """

    # 🔹 Ensure data is DataFrame for drift and leakage checks
    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train)

    if isinstance(X_test, np.ndarray):
        X_test = pd.DataFrame(X_test)

    results = {}

    # Model performance metrics
    results["performance"] = evaluate_model(
        model,
        X_test,
        y_test
    )

    # Overfitting detection
    results["overfitting"] = detect_overfitting(
        model,
        X_train,
        X_test,
        y_train,
        y_test
    )

    # Class imbalance detection
    results["class_imbalance"] = detect_class_imbalance(
        y_train
    )

    # Data leakage detection
    results["data_leakage"] = detect_data_leakage(
        X_train,
        y_train
    )

    # Data drift detection
    results["data_drift"] = detect_data_drift(
        X_train,
        X_test
    )

    return results