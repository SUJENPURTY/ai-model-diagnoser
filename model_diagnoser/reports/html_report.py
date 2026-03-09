from datetime import datetime


def generate_html_report(results, output_file="model_diagnosis_report.html"):
    """
    Generate an HTML report for model diagnostics.
    """

    html_content = f"""
    <html>
    <head>
        <title>Model Diagnosis Report</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
            }}
            h1 {{
                color: #2c3e50;
            }}
            h2 {{
                color: #34495e;
            }}
            .issue {{
                color: red;
                font-weight: bold;
            }}
            .ok {{
                color: green;
                font-weight: bold;
            }}
            pre {{
                background: #f4f4f4;
                padding: 10px;
            }}
        </style>
    </head>
    <body>

        <h1>Model Diagnosis Report</h1>
        <p>Generated on: {datetime.now()}</p>

        <h2>Performance Metrics</h2>
        <pre>{results.get("performance")}</pre>

        <h2>Overfitting Check</h2>
        <pre>{results.get("overfitting")}</pre>

        <h2>Class Imbalance</h2>
        <pre>{results.get("class_imbalance")}</pre>

        <h2>Data Leakage</h2>
        <pre>{results.get("data_leakage")}</pre>

        <h2>Data Drift</h2>
        <pre>{results.get("data_drift")}</pre>

    </body>
    </html>
    """

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"HTML report generated: {output_file}")