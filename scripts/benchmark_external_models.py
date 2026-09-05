"""Benchmark lightweight spam classifiers on the external UCI SMS corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = (
    PROJECT_ROOT / "data" / "external" / "uci_sms_spam" / "SMSSpamCollection"
)


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the tab-separated UCI corpus and normalize its label names."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing external dataset: {path}\n"
            "Download SMS Spam Collection from the UCI link in "
            "data/external/README.md."
        )

    data = pd.read_csv(
        path,
        sep="\t",
        names=["label", "message"],
        header=None,
        encoding="utf-8",
    )
    data = data.dropna(subset=["label", "message"]).drop_duplicates()
    data["label"] = data["label"].replace({"ham": "not_spam"})
    data = data[data["label"].isin(["spam", "not_spam"])].reset_index(drop=True)
    return data


def build_models() -> dict[str, Pipeline]:
    """Return comparable probability-producing model pipelines."""
    word_features = TfidfVectorizer(
        lowercase=True,
        stop_words=None,
        ngram_range=(1, 1),
    )
    word_bigram_features = TfidfVectorizer(
        lowercase=True,
        stop_words=None,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    return {
        "current_stopword_logreg": Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        lowercase=True,
                        stop_words="english",
                        ngram_range=(1, 1),
                    ),
                ),
                ("classifier", LogisticRegression(C=1.0, max_iter=3000)),
            ]
        ),
        "logreg_unigram": Pipeline(
            [
                ("tfidf", word_features),
                ("classifier", LogisticRegression(C=3.0, max_iter=3000)),
            ]
        ),
        "logreg_word_bigram": Pipeline(
            [
                ("tfidf", word_bigram_features),
                ("classifier", LogisticRegression(C=1.0, max_iter=3000)),
            ]
        ),
        "multinomial_nb": Pipeline(
            [
                ("tfidf", word_bigram_features),
                ("classifier", MultinomialNB(alpha=0.1)),
            ]
        ),
        "linear_svc_calibrated": Pipeline(
            [
                ("tfidf", word_bigram_features),
                (
                    "classifier",
                    CalibratedClassifierCV(
                        LinearSVC(C=1.0),
                        method="sigmoid",
                        cv=3,
                    ),
                ),
            ]
        ),
    }


def spam_probability(model: Pipeline, messages: pd.Series) -> pd.Series:
    """Return the probability assigned to the spam class."""
    probabilities = model.predict_proba(messages)
    spam_index = list(model.classes_).index("spam")
    return probabilities[:, spam_index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    args = parser.parse_args()

    data = load_dataset(args.data_path)
    X = data["message"]
    y = data["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "macro_f1": "f1_macro",
        "spam_f1": make_scorer(f1_score, pos_label="spam"),
    }

    print(f"Examples after cleaning: {len(data)}")
    print(f"Class counts: {y.value_counts().to_dict()}")
    print(f"Train/test sizes: {len(X_train)}/{len(X_test)}")
    print()

    for name, model in build_models().items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = spam_probability(model, X_test)
        binary_labels = (y_test == "spam").astype(int)

        report = classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        )
        cv_results = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )

        spam_precision = precision_score(y_test, predictions, pos_label="spam")
        spam_recall = recall_score(y_test, predictions, pos_label="spam")

        print(name)
        print(
            "  holdout: "
            f"accuracy={accuracy_score(y_test, predictions):.3f}, "
            f"macro_f1={f1_score(y_test, predictions, average='macro'):.3f}, "
            f"spam_precision={spam_precision:.3f}, "
            f"spam_recall={spam_recall:.3f}, "
            f"pr_auc={average_precision_score(binary_labels, probabilities):.3f}, "
            f"roc_auc={roc_auc_score(binary_labels, probabilities):.3f}"
        )
        print(
            "  5-fold CV: "
            f"accuracy={cv_results['test_accuracy'].mean():.3f}"
            f"+/-{cv_results['test_accuracy'].std():.3f}, "
            f"macro_f1={cv_results['test_macro_f1'].mean():.3f}"
            f"+/-{cv_results['test_macro_f1'].std():.3f}, "
            f"spam_f1={cv_results['test_spam_f1'].mean():.3f}"
            f"+/-{cv_results['test_spam_f1'].std():.3f}"
        )
        print(
            "  report: "
            f"not_spam_f1={report['not_spam']['f1-score']:.3f}, "
            f"spam_f1={report['spam']['f1-score']:.3f}"
        )
        print()


if __name__ == "__main__":
    main()
