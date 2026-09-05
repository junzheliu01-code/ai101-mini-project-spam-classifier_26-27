"""Tests for the local Gradio inference function."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import app

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def clear_model_cache():
    """Keep model-loading tests independent from one another."""
    app.load_model.cache_clear()
    yield
    app.load_model.cache_clear()


def test_model_loads():
    model = app.load_model()

    assert model is not None
    assert model is not app.LOAD_ERROR
    assert hasattr(model, "predict_proba")


def test_empty_input_is_handled():
    assert app.classify_message("") == ("Enter a message first.", {})
    assert app.classify_message("   ") == ("Enter a message first.", {})


def test_spam_and_normal_messages_are_classified():
    spam_prediction, _ = app.classify_message("WIN FREE MONEY NOW")
    normal_prediction, _ = app.classify_message("Hey, are we still meeting at 3 pm?")

    assert spam_prediction == "SPAM"
    assert normal_prediction == "NOT SPAM"


def test_probabilities_sum_to_one():
    _, probabilities = app.classify_message("WIN FREE MONEY NOW")

    assert set(probabilities) == {"not_spam", "spam"}
    assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-9)


def test_model_path_works_from_a_different_working_directory(tmp_path):
    code = (
        "from app import classify_message; "
        "prediction, probabilities = classify_message('WIN FREE MONEY NOW'); "
        "assert prediction == 'SPAM'; "
        "assert abs(sum(probabilities.values()) - 1.0) < 1e-9"
    )
    environment = os.environ.copy()
    pythonpath = [str(PROJECT_ROOT)]
    if environment.get("PYTHONPATH"):
        pythonpath.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(pythonpath)

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
