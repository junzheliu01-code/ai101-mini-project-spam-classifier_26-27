"""Train and export the production-oriented model on the UCI SMS corpus."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import make_scorer

from benchmark_external_models import DEFAULT_DATA_PATH, build_models, load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model" / "spam_classifier.joblib"
REPORT_PATH = PROJECT_ROOT / "reports" / "uci_external_model.json"
SPAM_THRESHOLD = 0.35


def get_spam_probability(model, messages):
    probabilities = model.predict_proba(messages)
    spam_index = list(model.classes_).index("spam")
    return probabilities[:, spam_index]


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
    probabilities = get_spam_probability(model, X_test)
    predictions = [
        "spam" if probability >= SPAM_THRESHOLD else "not_spam"
        for probability in probabilities
    ]
    binary_labels = (y_test == "spam").astype(int)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "macro_f1": "f1_macro",
            "spam_f1": make_scorer(f1_score, pos_label="spam"),
        },
        n_jobs=-1,
    )

    metrics = {
        "dataset": "UCI SMS Spam Collection",
        "examples_after_cleaning": len(data),
        "class_counts": y.value_counts().to_dict(),
        "train_examples": len(X_train),
        "test_examples": len(X_test),
        "model": "TfidfVectorizer(word unigrams and bigrams) + MultinomialNB",
        "alpha": 0.1,
        "spam_threshold": SPAM_THRESHOLD,
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
            "accuracy_mean": cv_results["test_accuracy"].mean(),
            "accuracy_std": cv_results["test_accuracy"].std(),
            "macro_f1_mean": cv_results["test_macro_f1"].mean(),
            "macro_f1_std": cv_results["test_macro_f1"].std(),
            "spam_f1_mean": cv_results["test_spam_f1"].mean(),
            "spam_f1_std": cv_results["test_spam_f1"].std(),
        },
    }

    final_model = clone(model)
    final_model.fit(X, y)
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
        f"accuracy={metrics['five_fold_cv']['accuracy_mean']:.3f}"
        f"+/-{metrics['five_fold_cv']['accuracy_std']:.3f}, "
        f"macro_f1={metrics['five_fold_cv']['macro_f1_mean']:.3f}"
        f"+/-{metrics['five_fold_cv']['macro_f1_std']:.3f}"
    )


if __name__ == "__main__":
    main()
