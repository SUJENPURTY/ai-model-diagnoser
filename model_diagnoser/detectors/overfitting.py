from sklearn.metrics import accuracy_score


def detect_overfitting(model, X_train, X_test, y_train, y_test, threshold=0.1):
    """
    Detects overfitting by comparing train and test accuracy.
    """

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    gap = train_acc - test_acc

    if gap > threshold:
        return {
            "issue": "Overfitting detected",
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "gap": gap,
            "severity": "high"
        }

    return {
        "issue": "No overfitting detected",
        "gap": gap
    }