import pytest

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from model_diagnoser import diagnose_model


def test_diagnose_model_runs():

    X, y = make_classification(
        n_samples=200,
        n_features=10,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2
    )

    model = RandomForestClassifier()

    model.fit(X_train, y_train)

    results = diagnose_model(
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        generate_report=False
    )

    assert isinstance(results, dict)


def test_results_have_keys():

    X, y = make_classification(
        n_samples=200,
        n_features=10,
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2
    )

    model = RandomForestClassifier()

    model.fit(X_train, y_train)

    results = diagnose_model(
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        generate_report=False
    )

    assert "performance" in results
    assert "overfitting" in results