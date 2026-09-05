# Spam Classifier Mini-Project

A small, local text-classification system that detects whether a message is spam or not_spam. The project combines a reproducible scikit-learn training pipeline with a Gradio application.

## What I built

- Loaded and inspected 200 labeled messages.
- Used a stratified 80/20 train/test split.
- Converted messages into TF-IDF features.
- Selected Logistic Regression regularization using 5-fold cross-validation on the training set.
- Evaluated accuracy, precision, recall, macro-F1, a confusion matrix, and model mistakes.
- Refit the selected pipeline on all labeled data for the exported application model.
- Integrated the model into a local Gradio interface with class probabilities.

## Results

| Evaluation | Accuracy | Macro-F1 | Spam precision | Spam recall |
| --- | ---: | ---: | ---: | ---: |
| Fixed 20% test split | 80.0% | 0.795 | 0.731 | 0.950 |
| Nested 5-fold CV, 2 repeats | 92.0% ± 3.1% | 0.920 ± 3.1% | — | — |

The dataset is small and designed for learning, so these results are not a production spam-filtering benchmark. The most important improvement was preserving short-message words instead of removing English stop words; terms such as your, account, and free carry useful signal in this dataset.

## Model

~~~text
message
  -> TF-IDF vectorizer
  -> Logistic Regression (C selected on training data)
  -> spam / not_spam + probabilities
~~~

The final pipeline uses lowercase word-level TF-IDF features without stop-word removal and Logistic Regression with max_iter=3000. On the fixed split, cross-validation selected C=3.0. The trained artifact is stored at model/spam_classifier.joblib.

## Run locally on Windows

~~~powershell
git clone https://github.com/junzheliu01-code/ai101-mini-project-spam-classifier_26-27.git
cd ai101-mini-project-spam-classifier_26-27
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
~~~

Open http://127.0.0.1:7860/ in a browser. The exported model is included, so the application can run immediately after installation.

## Reproduce training

Open train_spam_classifier.ipynb in Jupyter or Google Colab and run the cells in order. The notebook:

1. Loads data.csv.
2. Creates the fixed stratified test split.
3. Selects C using only the training data.
4. Reports test metrics and confusion-matrix errors.
5. Runs nested cross-validation for a stability estimate.
6. Inspects influential features.
7. Refits the selected pipeline on all data.
8. Saves model/spam_classifier.joblib.

## Example predictions

| Message | Prediction |
| --- | --- |
| WIN FREE MONEY NOW | SPAM |
| Hey, are we still meeting at 3 pm? | NOT SPAM |
| Free pizza at the CS and AI meeting tonight. | NOT SPAM |

## Limitations and next steps

- The dataset contains only 200 examples and may not represent real-world messages.
- The model can still overreact to words such as free, click, or urgent in unusual contexts.
- A stronger next experiment would add independently labeled hard negatives and an external test set.
- Future extensions could include threshold tuning, character n-grams, a batch CSV mode, and online deployment.

## Project files

| File | Purpose |
| --- | --- |
| app.py | Gradio interface and prediction logic |
| data.csv | Labeled training data |
| train_spam_classifier.ipynb | Reproducible training and evaluation notebook |
| model/spam_classifier.joblib | Full-data trained model artifact |
| requirements.txt | Pinned runtime dependencies |

## License

Apache License 2.0. See LICENSE.
