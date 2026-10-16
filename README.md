# Customer Message Intelligence System 📩

A beginner-friendly NLP project that classifies customer messages into four support categories using Python and Machine Learning.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- TF-IDF
- Logistic Regression
- Regular Expressions

## Support Categories
- Billing
- Login
- Delivery
- Product Support

## How It Works
1. Load customer messages from a CSV file.
2. Convert text into numerical features using TF-IDF.
3. Train a Logistic Regression classifier.
4. Compare unigram and bigram representations.
5. Test new customer messages.
6. Analyse prediction scores and model errors.
7. Extract order IDs using regular expressions.

## Project Files
- `support_messages.csv` — Customer message dataset
- `nlp_classifier.py` — NLP training and testing pipeline
- `rule_extractor.py` — Order ID extraction using regex
- `observations.md` — Experiment results and observations

## How to Run

Install the dependencies:

```bash
python3 -m pip install pandas scikit-learn
```

Run the classifier:

```bash
python3 nlp_classifier.py
```

Run the order ID extractor:

```bash
python3 rule_extractor.py
```

## Key Learnings
- Text representation using TF-IDF
- Unigrams versus bigrams
- Text classification with Logistic Regression
- Prediction score analysis
- Rule-based versus learned NLP
- Limitations of small training datasets

## Limitations
This project uses a small teaching dataset of 40 messages. Predictions may be inaccurate for unfamiliar wording, sarcasm, code-mixed language, or messages containing multiple issues.

## Author
Sri Harsha Vemula
