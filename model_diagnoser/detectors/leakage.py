import pandas as pd


def detect_data_leakage(X, y, threshold=0.9):
    """
    Detects potential data leakage by checking correlation with target.
    """

    if not isinstance(X, pd.DataFrame):
        return {"issue": "Data leakage check skipped (X not DataFrame)"}

    correlations = X.corrwith(pd.Series(y))

    leakage_features = correlations[abs(correlations) > threshold]

    if not leakage_features.empty:
        return {
            "issue": "Potential data leakage detected",
            "features": leakage_features.to_dict(),
            "severity": "high"
        }

    return {"issue": "No data leakage detected"}