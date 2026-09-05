"""Train and export the production-oriented model on the UCI SMS corpus."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
from benchmark_external_models import DEFAULT_DATA_PATH, build_models, load_dataset
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model" / "spam_classifier.joblib"
REPORT_PATH = PROJECT_ROOT / "reports" / "uci_external_model.json"
THRESHOLD_CANDIDATES = (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6)


def get_spam_probability(model, messages):
    probabilities = model.predict_proba(messages)
    spam_index = list(model.classes_).index("spam")
    return probabilities[:, spam_index]


def threshold_predictions(probabilities, threshold):
    return [
        "spam" if probability >= threshold else "not_spam"
        for probability in probabilities
    ]


def select_threshold(model, X_train, y_train, cv):
    """Select a threshold using only out-of-fold predictions from training data."""
    oof_probabilities = cross_val_predict(
        model,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )
    spam_index = list(model.classes_).index("spam")
    spam_probabilities = oof_probabilities[:, spam_index]

    candidates = []
    for threshold in THRESHOLD_CANDIDATES:
        predictions = threshold_predictions(spam_probabilities, threshold)
        candidates.append(
            {
                "threshold": threshold,
                "macro_f1": f1_score(y_train, predictions, average="macro"),
                "spam_precision": precision_score(
                    y_train,
                    predictions,
                    pos_label="spam",
                ),
                "spam_recall": recall_score(
                    y_train,
                    predictions,
                    pos_label="spam",
                ),
            }
        )

    selected = max(
        candidates,
        key=lambda result: (
            result["macro_f1"],
            result["spam_recall"],
            -result["threshold"],
        ),
    )
    return selected["threshold"], candidates


def main() -> None:
    data = load_dataset(DEFAULT_DATA_PATH)
    X = data["message"]
    y = data["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = build_models()["multinomial_nb"]
    model.fit(X_train, y_train)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    spam_threshold, threshold_candidates = select_threshold(
        model,
        X_train,
        y_train,
        cv,
    )
    probabilities = get_spam_probability(model, X_test)
    predictions = threshold_predictions(probabilities, spam_threshold)
    binary_labels = (y_test == "spam").astype(int)

    cv_probabilities = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )
    cv_spam_probabilities = cv_probabilities[:, list(model.classes_).index("spam")]
    cv_predictions = threshold_predictions(cv_spam_probabilities, spam_threshold)

    metrics = {
        "dataset": "UCI SMS Spam Collection",
        "examples_after_cleaning": len(data),
        "class_counts": y.value_counts().to_dict(),
        "train_examples": len(X_train),
        "test_examples": len(X_test),
        "model": "TfidfVectorizer(word unigrams and bigrams) + MultinomialNB",
        "alpha": 0.1,
        "spam_threshold": spam_threshold,
        "threshold_selection": {
            "method": "5-fold out-of-fold predictions on the training split only",
            "objective": (
                "maximize macro-F1, then spam recall, then prefer lower threshold"
            ),
            "candidates": threshold_candidates,
        },
        "holdout": {
            "accuracy": accuracy_score(y_test, predictions),
            "macro_f1": f1_score(y_test, predictions, average="macro"),
            "spam_precision": precision_score(
                y_test,
                predictions,
                pos_label="spam",
            ),
            "spam_recall": recall_score(y_test, predictions, pos_label="spam"),
            "pr_auc": average_precision_score(binary_labels, probabilities),
            "roc_auc": roc_auc_score(binary_labels, probabilities),
        },
        "five_fold_cv": {
            "evaluation": "fixed threshold selected on training split",
            "accuracy": accuracy_score(y, cv_predictions),
            "macro_f1": f1_score(y, cv_predictions, average="macro"),
            "spam_f1": f1_score(y, cv_predictions, pos_label="spam"),
        },
    }

    final_model = clone(model)
    final_model.fit(X, y)
    final_model.spam_threshold_ = spam_threshold
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)

    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(metrics, indent=2, default=float) + "\n",
        encoding="utf-8",
    )

    print(f"Saved external-data model: {MODEL_PATH}")
    print(f"Saved metrics report: {REPORT_PATH}")
    print(
        "Holdout: "
        f"accuracy={metrics['holdout']['accuracy']:.3f}, "
        f"macro_f1={metrics['holdout']['macro_f1']:.3f}, "
        f"spam_recall={metrics['holdout']['spam_recall']:.3f}"
    )
    print(
        "5-fold CV: "
        f"accuracy={metrics['five_fold_cv']['accuracy']:.3f}, "
        f"macro_f1={metrics['five_fold_cv']['macro_f1']:.3f}, "
        f"spam_f1={metrics['five_fold_cv']['spam_f1']:.3f}"
    )


if __name__ == "__main__":
    main()
