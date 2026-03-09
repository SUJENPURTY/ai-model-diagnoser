import numpy as np


def detect_class_imbalance(y, threshold=0.8):
    """
    Detects class imbalance in target variable.
    """

    classes, counts = np.unique(y, return_counts=True)
    total = len(y)

    ratios = counts / total

    imbalance = max(ratios)

    if imbalance > threshold:
        return {
            "issue": "Class imbalance detected",
            "class_distribution": dict(zip(classes, counts)),
            "severity": "medium"
        }

    return {
        "issue": "No class imbalance detected",
        "class_distribution": dict(zip(classes, counts))
    }