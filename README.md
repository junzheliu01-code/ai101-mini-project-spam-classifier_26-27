# Spam Classifier Mini-Project

[![CI](https://github.com/junzheliu01-code/ai101-mini-project-spam-classifier_26-27/actions/workflows/ci.yml/badge.svg)](https://github.com/junzheliu01-code/ai101-mini-project-spam-classifier_26-27/actions/workflows/ci.yml)

A local SMS spam classifier built with scikit-learn and Gradio. The repository keeps the original 200-message AI 101 exercise as a teaching baseline and adds a reproducible external-data benchmark for the runnable application.

The complete development history, evaluation decisions, interview evidence, and resume-ready wording are collected in [PROJECT_DEVELOPMENT_LOG.md](PROJECT_DEVELOPMENT_LOG.md).

## Project outcome

- Educational pipeline: word-level TF-IDF plus Logistic Regression, with C selected using only the training split.
- Application pipeline: word unigrams and bigrams plus MultinomialNB, trained on the cleaned UCI SMS Spam Collection.
- Gradio interface: returns SPAM or NOT SPAM together with both class probabilities.
- Decision threshold: spam probability >= 0.30 is classified as SPAM. The threshold is selected on training-only out-of-fold predictions to favor macro-F1 while preserving useful spam recall.

## Evaluation results

### Original 200-message teaching dataset

| Evaluation | Accuracy | Macro-F1 | Spam precision | Spam recall |
| --- | ---: | ---: | ---: | ---: |
| Fixed 20% test split | 80.0% | 0.795 | 0.731 | 0.950 |
| Nested 5-fold CV, 2 repeats | 92.0% ± 3.1% | 0.920 ± 3.1% | — | — |

The main improvement was removing English stop-word filtering. Short-message words such as your, account, and free can carry important classification signal.

### External UCI benchmark

The external corpus contains 5,574 English SMS messages. After removing exact duplicate rows, this project used 5,169 examples: 4,516 not_spam and 653 spam. The raw download is excluded from Git; see data/external/README.md for the official source and setup instructions.

| Model | Holdout accuracy | Holdout macro-F1 | Holdout spam recall | 5-fold CV macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Stop-word Logistic Regression | 95.6% | 0.886 | 0.672 | 0.891 |
| Logistic Regression, unigrams | 97.2% | 0.932 | 0.817 | 0.949 |
| Logistic Regression, word 1-2 grams | 95.8% | 0.890 | 0.672 | 0.892 |
| Calibrated LinearSVC, word 1-2 grams | 97.7% | 0.948 | 0.924 | 0.962 |
| **MultinomialNB, word 1-2 grams** | **98.4%** | **0.962** | **0.908** | **0.966** |

The exported application artifact uses the last row. The detailed reproducibility values are stored in reports/uci_external_model.json.
The baseline rows use each estimator's default class decision; the exported application row uses the selected 0.30 spam-probability threshold.

The threshold is selected from candidate values between 0.20 and 0.60 using 5-fold out-of-fold predictions on the training split only. The fixed holdout is evaluated after that choice, and the candidate results are saved in reports/uci_external_model.json.

## Model flow

~~~text
message
  -> lowercase TF-IDF word unigrams and bigrams
  -> MultinomialNB(alpha=0.1)
  -> probability threshold 0.30
  -> SPAM / NOT SPAM + probabilities
~~~

This remains a lightweight classical ML system: it is fast on Windows, produces probabilities, and is easier to explain than a large language model. It should not be treated as a production-grade security filter.

## Run locally on Windows

~~~powershell
git clone https://github.com/junzheliu01-code/ai101-mini-project-spam-classifier_26-27.git
cd ai101-mini-project-spam-classifier_26-27
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
~~~

Open http://127.0.0.1:7860/ in Chrome. The exported model is included, so the application runs immediately after dependency installation.

## Reproduce the original teaching experiment

Open train_spam_classifier.ipynb in Jupyter or Google Colab and run the cells in order. The notebook loads data.csv, creates the fixed stratified split, selects Logistic Regression C on training data, reports metrics and errors, runs nested cross-validation, and saves model/spam_classifier.joblib.

## Reproduce the external benchmark and application model

1. Download the UCI SMS Spam Collection from the link in data/external/README.md.
2. Extract SMSSpamCollection into data/external/uci_sms_spam/.
3. Compare lightweight models:

~~~powershell
python scripts/benchmark_external_models.py
~~~

4. Recreate the exported application artifact and metrics report:

~~~powershell
python scripts/train_external_model.py
~~~

The second command writes model/spam_classifier.joblib and reports/uci_external_model.json. The UCI corpus itself remains local and is not committed.

## Run quality checks

Install the development dependencies and run the same checks used by GitHub Actions:

~~~powershell
python -m pip install -r requirements-dev.txt
python -m pytest
python -m compileall -q app.py scripts tests
python -m ruff check app.py scripts tests
python -m ruff format --check app.py scripts tests
python scripts/check_readme.py
~~~

The workflow in .github/workflows/ci.yml runs these checks automatically on pushes and pull requests to main.

## Example predictions

| Message | Expected result |
| --- | --- |
| WIN FREE MONEY NOW | SPAM |
| Hey, are we still meeting at 3 pm? | NOT SPAM |
| Free pizza at the CS and AI meeting tonight. | NOT SPAM |

## Limitations and next steps

- Both corpora are English SMS datasets and may not represent current messaging behavior.
- The external benchmark is a stronger validation step, but it is still not an independent deployment test set.
- The probability threshold is a policy choice: lower thresholds catch more spam but can increase false positives.
- Useful next experiments are independently labeled hard negatives, an untouched external test set, character n-grams, a batch CSV mode, and a comparison with a calibrated LinearSVC or lightweight transformer.

## Project files

| File | Purpose |
| --- | --- |
| app.py | Gradio interface, model loading, thresholded prediction |
| PROJECT_DEVELOPMENT_LOG.md | Complete development history and resume evidence |
| data.csv | Original 200-message teaching dataset |
| train_spam_classifier.ipynb | Reproducible teaching experiment |
| scripts/benchmark_external_models.py | External model comparison |
| scripts/train_external_model.py | External-data training and export |
| scripts/check_readme.py | README documentation contract check |
| tests/test_app.py | Inference and model-path tests |
| .github/workflows/ci.yml | Automated Windows CI |
| data/external/README.md | UCI download, attribution, and placement instructions |
| model/spam_classifier.joblib | Exported application model |
| reports/uci_external_model.json | Reproducible external metrics |
| requirements.txt | Pinned runtime dependencies |
| requirements-dev.txt | Test and lint dependencies |
| pyproject.toml | Pytest and Ruff configuration |

## License

Apache License 2.0. See LICENSE.
