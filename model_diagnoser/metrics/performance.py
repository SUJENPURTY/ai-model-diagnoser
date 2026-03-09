from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance using common classification metrics.
    """

    predictions = model.predict(X_test)

    results = {}

    results["accuracy"] = accuracy_score(y_test, predictions)

    results["precision"] = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    results["recall"] = recall_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    results["f1_score"] = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    try:
        probabilities = model.predict_proba(X_test)
        results["roc_auc"] = roc_auc_score(
            y_test,
            probabilities[:, 1]
        )
    except Exception:
        results["roc_auc"] = None

    results["confusion_matrix"] = confusion_matrix(
        y_test,
        predictions
    ).tolist()

    results["classification_report"] = classification_report(
        y_test,
        predictions,
        output_dict=True
    )

    return results