def generate_suggestions(results):

    suggestions = []

    overfitting = results.get("overfitting", {}).get("issue", "").lower()
    imbalance = results.get("class_imbalance", {}).get("issue", "").lower()
    leakage = results.get("data_leakage", {}).get("issue", "").lower()
    drift = results.get("data_drift", {}).get("issue", "").lower()

    if "detected" in overfitting and "no" not in overfitting:
        suggestions.append(
            "Reduce model complexity or apply regularization."
        )

    if "detected" in imbalance and "no" not in imbalance:
        suggestions.append(
            "Consider using SMOTE or class weighting."
        )

    if "detected" in leakage and "no" not in leakage:
        suggestions.append(
            "Check if any feature contains target information."
        )

    if "detected" in drift and "no" not in drift:
        suggestions.append(
            "Retrain model with updated data."
        )

    if not suggestions:
        suggestions.append(
            "No major issues detected. Model looks stable."
        )

    return suggestions