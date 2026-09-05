"""Local Gradio interface for the AI 101 Spam Classifier workshop."""

from functools import lru_cache
from pathlib import Path

import gradio as gr
import joblib

MODEL_PATH = Path(__file__).resolve().parent / "model" / "spam_classifier.joblib"
SPAM_THRESHOLD = 0.30
LOAD_ERROR = object()


@lru_cache(maxsize=1)
def load_model():
    """Return the trained pipeline, None when absent, or a load-error sentinel."""
    if not MODEL_PATH.exists():
        return None

    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return LOAD_ERROR


def classify_message(message: str):
    """Classify one message and return a friendly label plus class probabilities."""
    if not message or not message.strip():
        return "Enter a message first.", {}

    model = load_model()

    if model is None:
        return (
            "Model not found. Run the documented training step and place "
            "spam_classifier.joblib inside the model/ folder.",
            {},
        )

    if model is LOAD_ERROR:
        return (
            "We found your model, but couldn't load it. Check that it is named "
            "spam_classifier.joblib and is inside the model/ folder.",
            {},
        )

    try:
        probabilities = model.predict_proba([message])[0]
        probability_map = {
            str(label): float(probability)
            for label, probability in zip(model.classes_, probabilities)
        }
        spam_threshold = float(getattr(model, "spam_threshold_", SPAM_THRESHOLD))
        prediction = (
            "spam" if probability_map.get("spam", 0.0) >= spam_threshold else "not_spam"
        )
    except Exception:
        return (
            "The model could not classify this message. Please re-export it from "
            "the workshop notebook and try again.",
            {},
        )

    friendly_prediction = "SPAM" if prediction == "spam" else "NOT SPAM"
    return friendly_prediction, probability_map


with gr.Blocks(title="Spam Classifier") as demo:
    gr.Markdown(
        """
        # Spam Classifier

        Determine whether a text message looks like spam using a machine-learning
        model you trained yourself.
        """
    )

    message_input = gr.Textbox(
        label="Message",
        placeholder="Enter a text message...",
        lines=4,
    )
    classify_button = gr.Button("Classify Message", variant="primary")
    prediction_output = gr.Textbox(label="Prediction", interactive=False)
    confidence_output = gr.Label(label="Model Confidence", num_top_classes=2)

    classify_button.click(
        fn=classify_message,
        inputs=message_input,
        outputs=[prediction_output, confidence_output],
    )


if __name__ == "__main__":
    demo.launch()
