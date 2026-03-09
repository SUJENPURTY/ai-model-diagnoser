<<<<<<< HEAD
# model-diagnoser

> AI-powered machine learning model diagnostics and debugging toolkit

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0-green.svg)](CHANGELOG.md)

**model-diagnoser** is a Python toolkit that automatically diagnoses common issues in machine learning models — overfitting, class imbalance, data leakage, and data drift — and provides actionable suggestions to fix them. It generates detailed HTML or PDF reports and optionally leverages OpenAI for deeper explanations.

---

## Features

- **Overfitting Detection** — Compares train vs. test accuracy to flag generalization issues
- **Class Imbalance Detection** — Identifies skewed class distributions in target variables
- **Data Leakage Detection** — Checks for features highly correlated with the target
- **Data Drift Detection** — Uses the Kolmogorov-Smirnov test to detect distribution shifts between train and test sets
- **Performance Metrics** — Computes accuracy, precision, recall, F1 score, ROC AUC, confusion matrix, and classification report
- **AI-Powered Suggestions** — Generates actionable recommendations based on detected issues
- **LLM Explanations** — Optional OpenAI integration for natural-language explanations of detected issues
- **HTML & PDF Reports** — Auto-generates diagnostic reports in your preferred format

---

## Installation

```bash
pip install model-diagnoser
```

Or install from source:

```bash
git clone https://github.com/your-username/model-diagnoser.git
cd model-diagnoser
pip install -e .
```

---

## Quick Start

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from model_diagnoser import diagnose_model

# Generate synthetic dataset
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    random_state=42
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Run diagnostics
results = diagnose_model(
    model,
    X_train, X_test,
    y_train, y_test,
    generate_report=True,
    report_format="html"  # or "pdf"
)
```

This will print a diagnostics summary to the console and generate a `model_diagnosis_report.html` file.

---

## Output Example

```
Running model diagnostics...

Model Performance
-----------------
Accuracy  : 0.925
Precision : 0.926
Recall    : 0.925
F1 Score  : 0.925
ROC AUC   : 0.982

Model Diagnostics Summary:

+------------------+-----------------------------------+
|      Check       |              Result               |
+------------------+-----------------------------------+
|   overfitting    |      No overfitting detected      |
| class_imbalance  |   No class imbalance detected     |
|  data_leakage    |     No data leakage detected      |
|   data_drift     | No significant data drift detected|
+------------------+-----------------------------------+

AI Suggestions
----------------
• No major issues detected. Model looks stable.

Diagnostics completed.
```

---

## Methodology

When you call `diagnose_model()`, the toolkit runs a sequential pipeline of diagnostic checks on your trained model and data. Each step uses a well-defined statistical or heuristic method. Below is a detailed breakdown of every stage.

### 1. Data Preprocessing

Before any checks run, the pipeline ensures all input data is in a consistent format. NumPy arrays passed as `X_train` or `X_test` are automatically converted to pandas DataFrames. This is necessary because several detectors (drift, leakage) operate on named columns and use pandas correlation/statistical methods internally.

### 2. Performance Evaluation

**Module:** `model_diagnoser/metrics/performance.py`

The model is evaluated on the **test set** using standard scikit-learn classification metrics:

| Metric | How It's Computed |
|---|---|
| **Accuracy** | Fraction of correct predictions: $\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Samples}}$ |
| **Precision** | Weighted average across classes. Measures how many predicted positives are truly positive: $\text{Precision} = \frac{TP}{TP + FP}$ |
| **Recall** | Weighted average across classes. Measures how many actual positives are correctly identified: $\text{Recall} = \frac{TP}{TP + FN}$ |
| **F1 Score** | Weighted harmonic mean of precision and recall: $F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ |
| **ROC AUC** | Area under the Receiver Operating Characteristic curve, computed from `predict_proba()`. Measures the model's ability to distinguish between classes across all thresholds. Falls back gracefully if the model doesn't support probability estimates. |
| **Confusion Matrix** | A matrix showing counts of true positives, true negatives, false positives, and false negatives. |
| **Classification Report** | Per-class precision, recall, F1, and support counts. |

All metrics use `zero_division=0` to handle edge cases (e.g., classes with no predictions) without raising errors.

### 3. Overfitting Detection

**Module:** `model_diagnoser/detectors/overfitting.py`

Overfitting occurs when a model memorizes training data instead of learning generalizable patterns. The detector measures this by comparing training accuracy to test accuracy:

$$\text{Gap} = \text{Accuracy}_{\text{train}} - \text{Accuracy}_{\text{test}}$$

**Decision logic:**

- If $\text{Gap} > 0.1$ (configurable threshold) → **Overfitting detected** (severity: `high`)
- Otherwise → No overfitting detected

**Why this works:** A well-generalized model should perform similarly on both train and test sets. A large gap indicates the model has learned noise or patterns specific to the training data that don't transfer to unseen data.

**What the output includes:**
- `train_accuracy` — accuracy on training set
- `test_accuracy` — accuracy on test set
- `gap` — the numeric difference
- `severity` — `"high"` when overfitting is detected

### 4. Class Imbalance Detection

**Module:** `model_diagnoser/detectors/imbalance.py`

Class imbalance is a common problem where one class dominates the dataset, causing the model to be biased toward the majority class. The detector works as follows:

1. Count the occurrences of each unique class in `y_train`
2. Compute the ratio of each class: $\text{ratio}_i = \frac{\text{count}_i}{\text{total samples}}$
3. Take the **maximum ratio** across all classes

**Decision logic:**

- If $\max(\text{ratios}) > 0.8$ (configurable threshold) → **Class imbalance detected** (severity: `medium`)
- Otherwise → No class imbalance detected

**Example:** In a binary classification dataset with 950 positive and 50 negative samples, the max ratio is $\frac{950}{1000} = 0.95$, which exceeds 0.8, so imbalance is flagged.

**What the output includes:**
- `class_distribution` — dictionary mapping each class label to its count
- `severity` — `"medium"` when imbalance is detected

### 5. Data Leakage Detection

**Module:** `model_diagnoser/detectors/leakage.py`

Data leakage occurs when features contain information that is directly derived from or highly correlated with the target variable. This leads to unrealistically high performance that won't hold in production. The detector identifies this using **Pearson correlation**:

1. Compute the correlation between each feature column in `X_train` and the target `y_train` using `DataFrame.corrwith()`
2. Flag any feature where $|\text{correlation}| > 0.9$ (configurable threshold)

**Decision logic:**

- If any feature exceeds the correlation threshold → **Potential data leakage detected** (severity: `high`)
- Otherwise → No data leakage detected

**Why this works:** A feature that correlates almost perfectly with the target is likely either derived from it, is a proxy for it, or contains future information. In all cases, the feature should be investigated and potentially removed.

**What the output includes:**
- `features` — dictionary of flagged feature names and their correlation values
- `severity` — `"high"` when leakage is detected

### 6. Data Drift Detection

**Module:** `model_diagnoser/detectors/drift.py`

Data drift occurs when the statistical distribution of features changes between training and inference time. The detector uses the **Kolmogorov-Smirnov (KS) two-sample test** — a non-parametric test that compares two distributions without assuming normality.

For each feature column:

1. Extract the values from `X_train` and `X_test` (dropping NaNs)
2. Run `scipy.stats.ks_2samp(train_values, test_values)`
3. The KS test returns a **test statistic** (maximum distance between the two empirical CDFs) and a **p-value**

**Decision logic (per feature):**

- If $p\text{-value} < 0.05$ (configurable threshold) → drift detected for that feature
- The null hypothesis is that both samples come from the same distribution; a low p-value rejects this

**Overall decision:**

- If **any** feature shows drift → **Data drift detected** (severity: `medium`)
- Otherwise → No significant data drift detected

**What the output includes:**
- `features` — list of feature names where drift was detected
- `severity` — `"medium"` when drift is detected

### 7. AI-Powered Suggestions

**Module:** `model_diagnoser/ai/suggestions.py`

After all detectors run, a rule-based suggestion engine examines the results and generates actionable recommendations:

| Detected Issue | Suggestion |
|---|---|
| Overfitting | Reduce model complexity or apply regularization |
| Class Imbalance | Consider using SMOTE or class weighting |
| Data Leakage | Check if any feature contains target information |
| Data Drift | Retrain model with updated data |
| No issues | No major issues detected. Model looks stable |

### 8. LLM Explanations (Optional)

**Module:** `model_diagnoser/ai/explanation_llm.py`

For deeper explanations, the toolkit can send detected issues to OpenAI's GPT-4.1-mini model. The LLM receives the raw diagnostic result and returns a plain-language explanation of what the issue means and how to fix it. This requires an `OPENAI_API_KEY` environment variable and is completely optional — the core pipeline functions without it.

### 9. Report Generation

**Module:** `model_diagnoser/reports/`

The final step compiles all results into a structured report:

- **HTML report** — A styled, browser-viewable document with sections for performance metrics, each diagnostic check, and suggestions. Generated using Python string templating with inline CSS.
- **PDF report** — A multi-page document generated with ReportLab, containing the same information in a print-friendly format. Automatically handles page breaks for long outputs.

### Pipeline Flow Diagram

```
Input (model, X_train, X_test, y_train, y_test)
  │
  ▼
┌─────────────────────────┐
│   Data Preprocessing    │  Convert arrays → DataFrames
└────────────┬────────────┘
             │
  ┌──────────┼──────────────────────────────┐
  ▼          ▼              ▼               ▼
┌──────┐ ┌──────────┐ ┌──────────┐ ┌────────────┐
│Overfit│ │Imbalance │ │ Leakage  │ │   Drift    │
│Check  │ │  Check   │ │  Check   │ │   Check    │
└──┬───┘ └────┬─────┘ └────┬─────┘ └─────┬──────┘
   │          │             │             │
   └──────────┴──────┬──────┴─────────────┘
                     ▼
          ┌─────────────────────┐
          │ Performance Metrics │
          └─────────┬───────────┘
                    ▼
          ┌─────────────────────┐
          │  AI Suggestions     │  Rule-based recommendations
          └─────────┬───────────┘
                    ▼
          ┌─────────────────────┐
          │  Report Generation  │  HTML or PDF
          └─────────┬───────────┘
                    ▼
              Output (results dict + report file)
```

---

## API Reference

### `diagnose_model(model, X_train, X_test, y_train, y_test, generate_report=True, report_format="html")`

Runs the full diagnostics pipeline on a trained model.

| Parameter | Type | Description |
|---|---|---|
| `model` | estimator | A trained scikit-learn compatible model |
| `X_train` | array-like | Training feature data |
| `X_test` | array-like | Test feature data |
| `y_train` | array-like | Training labels |
| `y_test` | array-like | Test labels |
| `generate_report` | bool | Whether to generate a report file (default: `True`) |
| `report_format` | str | Report format — `"html"` or `"pdf"` (default: `"html"`) |

**Returns:** A dictionary containing performance metrics, diagnostic results, and suggestions.

---

## Project Structure

```
model_diagnoser/
├── __init__.py              # Package entry point (exports diagnose_model)
├── automl_debugger.py       # Main orchestrator
├── diagnose.py              # Runs all diagnostic checks
├── detectors/
│   ├── overfitting.py       # Train vs. test accuracy gap detection
│   ├── imbalance.py         # Class distribution analysis
│   ├── leakage.py           # Feature-target correlation check
│   └── drift.py             # KS-test based distribution drift
├── metrics/
│   └── performance.py       # Classification metrics evaluation
├── ai/
│   ├── suggestions.py       # Rule-based fix suggestions
│   └── explanation_llm.py   # OpenAI-powered explanations
├── reports/
│   ├── html_report.py       # HTML report generator
│   └── pdf_report.py        # PDF report generator
└── utils/
    └── helpers.py           # DataFrame/Series conversion utilities
```

---

## LLM Explanations (Optional)

To enable AI-powered natural-language explanations via OpenAI, set your API key:

```bash
export OPENAI_API_KEY="your-api-key"
```

Then use the `explain_issue` function directly:

```python
from model_diagnoser.ai import explain_issue

explanation = explain_issue({"issue": "Overfitting detected", "gap": 0.15})
print(explanation)
```

> This feature is optional. The core diagnostics and suggestions work without an API key.

---

## Development

### Setup

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Run Tests

```bash
pytest
```

### Code Formatting

```bash
black .
flake8
```

---

## Dependencies

| Package | Version |
|---|---|
| numpy | >= 1.23 |
| pandas | >= 1.5 |
| scikit-learn | >= 1.2 |
| scipy | >= 1.9 |
| matplotlib | >= 3.7 |
| seaborn | >= 0.12 |
| jinja2 | >= 3.1 |
| reportlab | >= 4.0 |
| tqdm | >= 4.65 |
| openai | >= 1.0 |

---

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release history.

---

Made by [Sujen Purty](https://github.com/SUJENPURTY)
=======
# ai-model-diagnoser
AI-powered Python package for diagnosing machine learning models, detecting issues, and providing intelligent explanations with suggested fixes.
>>>>>>> f89c69a825940eb1adc542797f2abcc41e497c03
