# Spam Classifier Mini-Project

Your team has inherited a spam-detection application. The interface is already built. There is just one problem: **the AI does not exist yet.**

Your job is to train a machine-learning model and connect it to the application. By the end, you will have trained and evaluated a real text classifier, investigated its mistakes, exported it as a model artifact, and used it inside a local app.

## What You'll Learn

- Training data and labels
- Training versus testing data
- Why text must become numerical features
- TF-IDF and logistic regression at a high level
- Accuracy versus an individual prediction's confidence
- Why simple models make mistakes
- How a model artifact becomes part of an AI product
- How a tiny specialized model differs from a general-purpose AI chatbot

## How the Project Works

```text
data.csv → Google Colab training notebook → spam_classifier.joblib
                                                ↓
                                      app.py + Gradio interface
                                                ↓
                             spam / not_spam prediction + confidence
```

The `data.csv` file contains labeled examples. In Google Colab, you will build a pipeline that turns text into TF-IDF numerical features and then uses logistic regression to classify them. You will save that trained pipeline to `spam_classifier.joblib`. The local app loads that file and presents its predictions.

## Project Files

| File | Purpose |
| --- | --- |
| `app.py` | Finished Gradio interface and model-loading/prediction logic. |
| `data.csv` | 200 labeled messages used for training. |
| `requirements.txt` | Pinned packages for the local app. |
| `train_spam_classifier.ipynb` | Attendee training notebook; a hosted copy will be linked below. |
| `train_spam_classifier_solution.ipynb` | Completed facilitator reference notebook. |
| `model/.gitkeep` | Keeps the initially empty model folder in the project. |

The model file is intentionally absent at first. **Nothing is broken.** The interface exists, but the machine-learning model does not—yet.

## Milestone 1: Get the Project

Either clone the repository:

```bash
git clone <repository-url>
cd spam-classifier
```

Or download the repository as a ZIP, extract it, and open a terminal inside the extracted folder.

## Milestone 2: Install Dependencies

Confirm that Python is available:

```bash
python --version
```

Some systems use `python3` instead. Then install the local app dependencies:

```bash
pip install -r requirements.txt
```

If needed, use `pip3 install -r requirements.txt`.

## Milestone 3: Run the Unfinished App

Start the local application:

```bash
python app.py
```

Open the local URL printed by the terminal, usually `http://127.0.0.1:7860`.

Enter a message and select **Classify Message**. Before you train a model, the expected result is a friendly **Model not found** message. This is the intended starting point of the activity.

## Milestone 4: Open the Colab Notebook

Open the hosted workshop notebook: [Train Your Spam Classifier](<COLAB_LINK_HERE>).

Until it is hosted, you can upload `train_spam_classifier.ipynb` to Google Colab yourself. The notebook uses this sequence:

1. Upload `data.csv`.
2. Explore the labels and examples.
3. Split the data into training and testing sets.
4. Create a TF-IDF + Logistic Regression pipeline.
5. Train and evaluate the model.
6. Try your own messages and inspect confidence.
7. Try to break the model and inspect learned features.
8. Export and download `spam_classifier.joblib`.

The attendee notebook has two short code-completion exercises: choose a 20% test split and pass `X_train`/`y_train` to `pipeline.fit`. Work through them before using the facilitator solution notebook.

## Milestone 5: Upload `data.csv`

The first notebook code cell opens an upload picker. Select the `data.csv` file from this project. Then run the cells that display the first rows, total example count, and label counts.

Each row has:

- `message`: text the model will receive as input
- `label`: the correct answer, either `spam` or `not_spam`

The dataset includes obvious cases, ambiguous cases, and counterexamples. For example, legitimate messages may contain words such as “free,” “winner,” or “click,” while spam can avoid stereotypical words. That design is intentional: it makes model mistakes useful to discuss.

## Milestone 6: Train the Model

The notebook holds out 20% of examples for testing. It trains this pipeline:

```text
message → TF-IDF vectorizer → logistic regression classifier → label
```

TF-IDF converts text into numerical features based on word use and how informative words are in the dataset. Logistic regression learns statistical associations between those features and the two labels.

Before `pipeline.fit(X_train, y_train)`, the pipeline is an algorithm. After it, it is a trained model containing patterns learned from this particular data.

## Milestone 7: Evaluate and Experiment

The notebook reports test accuracy on held-out messages. Accuracy answers: “How often did the model get this testing set right?” It does not mean that every individual prediction has that same confidence.

For an individual message, probabilities show the model's preference:

```text
Spam: 51%       Not Spam: 49%    ← uncertain
Spam: 99%       Not Spam: 1%     ← decisive
```

Use the `classify()` helper in the notebook to test your own messages.

## Challenge: Try to Break the Model

Find each of the following:

- A message the model gets right with high confidence
- A message near 50/50
- A legitimate message incorrectly labeled spam
- A spam message incorrectly labeled not spam
- Two almost identical messages with noticeably different confidence

Good starting messages include:

```text
Congratulations on winning the hackathon!
Free pizza at the CS+AI meeting tonight.
Click this link to download the lecture notes.
URGENT: please send me the homework.
We've been trying to reach you regarding your account.
```

Afterward, use the feature-inspection section. It lists words most associated with each class and can help explain why a model made a surprising decision. The model is not reasoning about spam like a person; it learned patterns from a small labeled dataset.

## Milestone 8: Download the Model

The notebook exports a **model artifact** named `spam_classifier.joblib`, then downloads it. The artifact contains both the fitted TF-IDF vectorizer and the trained logistic regression classifier.

## Milestone 9: Add the Model to the App

Move the downloaded file to this exact location:

```text
model/spam_classifier.joblib
```

Your structure should look like:

```text
spam-classifier/
├── app.py
├── data.csv
├── model/
│   ├── .gitkeep
│   └── spam_classifier.joblib
└── requirements.txt
```

Then run `python app.py` again, or keep the running app open and classify another message. It should now show a prediction and both class probabilities.

## Milestone 10: Test the Completed Product

Try examples from the challenge in the local Gradio interface. Notice that the app is more than a model artifact: it provides model loading, input validation, friendly error messages, output formatting, probability visualization, and a user interface.

```text
MODEL + APPLICATION LOGIC + USER INTERFACE = USABLE AI SYSTEM
```

## Challenge: Compare Against an AI Chatbot

This activity happens **outside** the Gradio app. Open ChatGPT, Gemini, Claude, or another general-purpose chatbot, and use the same messages you tested locally.

Suggested prompt:

```text
You are a spam classifier.

Classify the following message as exactly one of:

spam
not spam

Do not provide an explanation.

Message:
[YOUR MESSAGE]
```

Record the results:

| Message | Tiny ML Model | Confidence | Chatbot | Expected Label |
| --- | --- | ---: | --- | --- |
| WIN FREE MONEY NOW | spam | 99% | spam | spam |
| Free pizza at the club tonight | spam | 71% | not spam | not_spam |
| Call me when you get home | not_spam | 96% | not spam | not_spam |

Exact results will vary. The point is to compare behavior, not to get a guaranteed score.

## Discussion Questions

1. Where did the tiny model and chatbot disagree?
2. Which context-heavy examples did the chatbot understand more naturally?
3. Which learned words/features help explain the tiny classifier's errors?
4. Does a more capable model automatically make the best engineering choice?

The chatbot has broad pretraining and richer language representations. The tiny classifier is local, fast, inexpensive, predictable, easy to deploy, and specialized. A real engineering decision depends on constraints, risks, cost, privacy, latency, accuracy needs, and maintenance—not only raw capability.

## Troubleshooting

### `python` or `pip` is not found

Try `python3 --version`, `python3 app.py`, and `pip3 install -r requirements.txt`.

### A package is missing

Run the install command again from the project folder:

```bash
pip install -r requirements.txt
```

### The app still says model not found

Check that the file is exactly `model/spam_classifier.joblib`. Common mistakes include:

- `spam_classifier.joblib.joblib`
- `models/spam_classifier.joblib`
- leaving the file in Downloads

### The app says it cannot load the model

Re-export the artifact from the workshop notebook. The local package versions are pinned for compatibility; a model created with a substantially different scikit-learn version can fail to load.

### The Gradio page does not open

Use the exact local URL printed by the terminal, typically `http://127.0.0.1:7860`.

## Workshop Checklist

- [ ] Download or clone the repository
- [ ] Install Python dependencies
- [ ] Launch the Gradio app
- [ ] Confirm the app says the model is missing
- [ ] Open the Colab training notebook
- [ ] Upload `data.csv`
- [ ] Split the data and train the model
- [ ] Measure test accuracy
- [ ] Try custom messages and find a mistake
- [ ] Inspect learned features
- [ ] Download `spam_classifier.joblib`
- [ ] Move it into `model/`
- [ ] Use the completed Gradio app
- [ ] Compare results with an AI chatbot

## Final Reflection

You built an AI system without using a large language model. Your classifier learned from labeled examples, transformed text into numerical features, produced predictions, was saved as an artifact, and became useful only after it was integrated into an application. The chatbot can solve the same task differently—but the biggest model is not automatically the best model for every product.
