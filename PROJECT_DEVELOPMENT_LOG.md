# Project Development Log / 项目开发记录

This document records the complete evolution of the SMS Spam Classifier from the original AI 101 workshop implementation to the current resume-ready version. It is intended to preserve the engineering decisions, evidence, limitations, and resume language in one place.

## 1. Project identity

- **Project:** SMS Spam Detection System
- **Repository:** `junzheliu01-code/ai101-mini-project-spam-classifier_26-27`
- **Primary stack:** Python, pandas, scikit-learn, Joblib, Gradio, pytest, Ruff, GitHub Actions
- **Task:** Binary text classification: `spam` versus `not_spam`
- **Current application model:** TF-IDF word unigrams/bigrams + Multinomial Naive Bayes
- **Current interface:** Local Gradio application with class probabilities

## 2. Starting point

The original repository was an educational AI 101 mini-project with a small, balanced dataset of 200 labeled messages. Its vertical slice was:

~~~text
data.csv -> training notebook -> spam_classifier.joblib -> app.py Gradio interface
~~~

The original modeling approach used word-level TF-IDF with English stop-word removal and Logistic Regression. The application loaded the model with a relative path, which meant starting the program from a different working directory could cause a model-not-found error.

The original version was useful as a teaching exercise, but it had three limitations for a resume-facing project:

1. The dataset was too small to support a strong generalization claim.
2. The stop-word setting removed short-message words that could be meaningful spam signals.
3. The repository had no automated tests or continuous integration evidence.

## 3. Phase 1: diagnose and improve the teaching baseline

### Problem discovered

The initial stop-word configuration removed terms such as `your`, `account`, and `free`. These terms can be important in short SMS messages, so the preprocessing choice was hurting the model rather than helping it.

### Change

The teaching pipeline was changed to:

~~~python
TfidfVectorizer(
    lowercase=True,
    stop_words=None,
    ngram_range=(1, 1),
)

LogisticRegression(
    C=3.0,
    max_iter=3000,
)
~~~

The regularization value was selected using cross-validation on the training portion only. The fixed 80/20 split remained stratified with `random_state=42`.

### Teaching-dataset result

| Evaluation | Accuracy | Macro-F1 | Spam precision | Spam recall |
| --- | ---: | ---: | ---: | ---: |
| Earlier stop-word configuration | 77.5% | approximately 0.73 | — | 90.0% |
| Improved fixed test split | 80.0% | 0.795 | 0.731 | 95.0% |
| Nested 5-fold CV, 2 repeats | — | 0.920 ± 0.031 | — | — |

The result showed that the main improvement came from preserving domain-relevant words, not from choosing a more complicated algorithm.

## 4. Phase 2: add a larger external benchmark

To test whether the improvement was specific to the 200-message classroom dataset, the project added the UCI SMS Spam Collection. The raw corpus contains 5,574 English SMS messages. After removing exact duplicate rows, the local benchmark contains 5,169 examples:

- `not_spam`: 4,516
- `spam`: 653

The raw corpus is intentionally excluded from Git. Download and attribution instructions are stored in `data/external/README.md`.

### Compared models

The external benchmark compares five lightweight, probability-producing pipelines:

1. Stop-word Logistic Regression
2. Logistic Regression with word unigrams
3. Logistic Regression with word unigrams and bigrams
4. Calibrated LinearSVC with word unigrams and bigrams
5. MultinomialNB with word unigrams and bigrams

The benchmark keeps the same stratified 80/20 split with `random_state=42` and also runs 5-fold cross-validation. Model selection was based primarily on macro-F1, with spam recall as a business-relevant secondary metric.

### Model-selection evidence

The external benchmark showed that MultinomialNB was the strongest balanced classical model under the selected feature representation. Calibrated LinearSVC achieved higher spam recall in some comparisons, but MultinomialNB provided the better macro-F1 tradeoff and retained straightforward probability output.

The comparison is implemented in `scripts/benchmark_external_models.py`.

## 5. Phase 3: make threshold selection reproducible

The application uses a probability threshold instead of blindly accepting the estimator's default class decision. The initial version used a manually selected threshold of `0.35`. That was useful experimentally, but its selection process was not recorded rigorously enough.

The final training process now:

1. Creates the fixed stratified 80/20 split.
2. Fits the selected model on the training split.
3. Generates 5-fold out-of-fold probabilities using only the training split.
4. Evaluates candidate thresholds from `0.20` through `0.60`.
5. Selects the threshold that maximizes macro-F1, then spam recall, then prefers the lower threshold.
6. Evaluates the selected threshold once on the held-out test split.
7. Re-trains the final model on all cleaned UCI examples.
8. Stores the selected threshold as model metadata in `spam_classifier.joblib`.

The selected threshold is `0.30`. All candidate results and the selection method are recorded in `reports/uci_external_model.json`.

This change makes the evaluation more defensible. The final thresholded holdout score is slightly lower than the earlier manually selected result, but it no longer depends on selecting the threshold after looking at the test set.

## 6. Phase 4: improve application reliability

The Gradio application was updated to:

- Resolve the model path relative to `app.py`, not the current shell directory.
- Cache the loaded model during the process lifetime.
- Read the trained threshold metadata from the exported model.
- Preserve the existing `SPAM` / `NOT SPAM` interface.
- Return both class probabilities.
- Handle empty input, missing model, unloadable model, and prediction errors.

The path change was important because the application should behave consistently when launched from a different working directory.

## 7. Phase 5: add automated quality evidence

The repository now contains five pytest tests in `tests/test_app.py`, covering:

- The model artifact loads successfully.
- Empty and whitespace-only input is handled safely.
- A known spam message is classified as `SPAM`.
- A normal scheduling message is classified as `NOT SPAM`.
- Probabilities contain both classes and sum to approximately 1.
- A subprocess launched from a different working directory can still locate the model.

The GitHub Actions workflow in `.github/workflows/ci.yml` runs on Windows with Python 3.11 and checks:

- Installation from `requirements-dev.txt`
- pytest
- Python bytecode compilation
- Ruff linting
- Ruff formatting
- README structure and placeholder checks

The current CI run is green. The development dependencies and tool configuration are in `requirements-dev.txt` and `pyproject.toml`.

## 8. Final evaluation result

The current exported application model is trained on all 5,169 cleaned UCI examples, while the metrics below are calculated before the final full-data refit.

| Evaluation | Result | Notes |
| --- | ---: | --- |
| Fixed holdout accuracy | 98.4% | 20% stratified test split, threshold selected from training only |
| Fixed holdout macro-F1 | 0.962 | Primary balanced metric |
| Fixed holdout spam precision | 0.960 | Thresholded application decision |
| Fixed holdout spam recall | 0.908 | Business-oriented secondary metric |
| 5-fold out-of-fold accuracy | 98.5% | Fixed selected threshold |
| 5-fold out-of-fold macro-F1 | 0.966 | Fixed selected threshold |
| 5-fold out-of-fold spam-F1 | 0.940 | Fixed selected threshold |

The detailed machine-readable report is `reports/uci_external_model.json`.

## 9. Reproducibility commands

~~~powershell
# Install runtime dependencies
python -m pip install -r requirements.txt

# Download and place the UCI corpus according to data/external/README.md

# Compare candidate models
python scripts/benchmark_external_models.py

# Select the threshold, train, export the model, and write the report
python scripts/train_external_model.py

# Run quality checks
python -m pip install -r requirements-dev.txt
python -m pytest
python -m compileall -q app.py scripts tests
python -m ruff check app.py scripts tests
python -m ruff format --check app.py scripts tests
python scripts/check_readme.py

# Launch the local application
python app.py
~~~

## 10. Resume-ready project description

### Recommended project title

**SMS Spam Detection System | Python, scikit-learn, Gradio**

### Recommended resume bullets

- Built a reproducible SMS spam classification pipeline using TF-IDF word unigrams/bigrams and Multinomial Naive Bayes; benchmarked five lightweight models on 5,169 deduplicated SMS messages.
- Selected the spam decision threshold using training-only out-of-fold predictions and achieved 98.4% holdout accuracy, 0.962 macro-F1, and 0.908 spam recall.
- Integrated a Gradio inference interface and added pytest coverage, Windows GitHub Actions CI, model-path validation, linting, and reproducible evaluation reports.

### Claims to avoid

- Do not call this a production security filter.
- Do not call the local Gradio application a public deployment.
- Do not describe the UCI holdout as an independent deployment test set.
- Do not report the old 98.5%/0.966 thresholded holdout combination; the current stricter evaluation is 98.4%/0.962.

## 11. Interview discussion points

### Why remove English stop words?

Short messages contain highly informative words that generic English stop-word lists can remove. In this dataset, preserving those words improved the teaching baseline substantially.

### Why use MultinomialNB instead of a larger model?

The dataset is moderate in size, and the project needs fast training, simple deployment, interpretable feature behavior, and probability output. MultinomialNB provided the best macro-F1 tradeoff among the tested lightweight models.

### How did you avoid test leakage during threshold selection?

The final threshold is selected from out-of-fold predictions generated only on the training split. The fixed holdout is evaluated only after the threshold is chosen.

### What would you do next?

Add an independently labeled hard-negative set, evaluate on an untouched external corpus, measure the precision-recall tradeoff at several operating points, and then compare character n-grams or a lightweight transformer if the data volume justifies it.

## 12. Development timeline

| Commit | Development milestone |
| --- | --- |
| `f03ee8f` | Complete local spam classifier implementation |
| `0a4afa8` | Improve preprocessing, model selection, and reproducibility |
| `7ad4b30` | Add external benchmark and application model |
| `5f74b36` | Add pytest and GitHub Actions CI |
| `cbf282b` | Use Node 24-compatible GitHub Actions runtimes |
| `e5e0c85` | Make threshold selection reproducible |

## 13. Current limitations

- The corpora are English SMS datasets and may not represent current messaging behavior.
- The UCI benchmark is external to the original classroom dataset, but it is not an untouched production deployment test set.
- A small classical ML model can still fail on new slang, obfuscation, multilingual messages, or context-dependent legitimate uses of words such as `free` and `urgent`.
- The application is local; no public hosted demo or production API is claimed.
