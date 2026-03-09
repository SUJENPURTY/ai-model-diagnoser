from .diagnose import run_diagnostics
from .reports.html_report import generate_html_report
from .reports.pdf_report import generate_pdf_report
from .ai.suggestions import generate_suggestions
from prettytable import PrettyTable


def diagnose_model(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    generate_report=True,
    report_format="html"
):

    print("Running model diagnostics...")

    results = run_diagnostics(
        model,
        X_train,
        X_test,
        y_train,
        y_test
    )

    # 🔹 Performance metrics
    perf = results.get("performance", {})

    print("\nModel Performance")
    print("-----------------")
    print(f"Accuracy  : {perf.get('accuracy', 0):.3f}")
    print(f"Precision : {perf.get('precision', 0):.3f}")
    print(f"Recall    : {perf.get('recall', 0):.3f}")
    print(f"F1 Score  : {perf.get('f1_score', 0):.3f}")
    print(f"ROC AUC   : {perf.get('roc_auc', 0):.3f}")

    # 🔹 Diagnostics summary table
    table = PrettyTable()
    table.field_names = ["Check", "Result"]

    diagnostic_keys = [
        "overfitting",
        "class_imbalance",
        "data_leakage",
        "data_drift"
    ]

    for key in diagnostic_keys:

        value = results.get(key, {})
        issue = value.get("issue", "N/A")

        table.add_row([key, issue])

    print("\nModel Diagnostics Summary:\n")
    print(table)

    # 🔹 AI Suggestions
    suggestions = generate_suggestions(results) or []
    results["suggestions"] = suggestions

    print("\nAI Suggestions")
    print("----------------")

    for s in suggestions:
        print(f"• {s}")

    # 🔹 Generate report
    if generate_report:

        if report_format == "html":
            generate_html_report(results)

        elif report_format == "pdf":
            generate_pdf_report(results)

    print("\nDiagnostics completed.")

    return results