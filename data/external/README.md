# External evaluation data

The local benchmark uses the official UCI SMS Spam Collection:

- Source: https://archive.ics.uci.edu/dataset/228/sms%2Bspam%2Bcollection
- Corpus: 5,574 English SMS messages labeled ham or spam
- License: CC BY 4.0

Download the ZIP from the UCI page and extract SMSSpamCollection into:

~~~text
data/external/uci_sms_spam/SMSSpamCollection
~~~

The raw corpus is intentionally excluded from the Git repository. Run:

~~~powershell
python scripts/benchmark_external_models.py
~~~

The benchmark maps ham to the project's not_spam label and removes exact duplicate rows before evaluation.
