# Customer Message Intelligence System 📩

A beginner-friendly Natural Language Processing (NLP) project that automatically classifies customer support messages into the correct support category using Python and Machine Learning.

## 🎯 Project Objective

The goal is to understand how computers process customer messages and predict which support team should handle them.

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorisation
- Logistic Regression
- Regular Expressions (Regex)

## 📂 Support Categories

- 💳 Billing
- 🔐 Login
- 🚚 Delivery
- 🛠️ Product Support

## ⚙️ How It Works

1. Load customer messages from a CSV dataset.
2. Convert text into numerical features using TF-IDF.
3. Train a Logistic Regression classifier.
4. Compare unigrams with unigrams and bigrams.
5. Test the models on new customer messages.
6. Analyse prediction scores and model mistakes.
7. Extract order IDs using regular expressions.

## 🧪 Experiments

- Compared two text representations.
- Evaluated predictions on unseen test messages.
- Tested unfamiliar, informal, ambiguous, and difficult messages.
- Investigated why some predictions were incorrect.
- Explored the difference between rule-based extraction and learned text classification.

## ▶️ How to Run

Clone this repository:

```bash
git clone YOUR_REPOSITORY_URL
```

Open the project folder and create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

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

## 📊 Dataset

The project uses a small teaching dataset containing 40 labelled customer messages across four support categories.

## 💡 Key Learnings

- How text can be represented numerically using TF-IDF.
- The difference between unigrams and bigrams.
- How to train and evaluate a text classifier.
- Why prediction scores do not guarantee correctness.
- How limited training data affects predictions.
- When regular expressions are useful for predictable text patterns.

## ⚠️ Limitations

This project uses a small dataset for learning purposes. It may struggle with unfamiliar wording, sarcasm, code-mixed language, and messages containing multiple issues.

## 🔮 Future Improvements

- Expand the dataset with realistic customer messages.
- Include more informal and multilingual examples.
- Improve handling of messages containing multiple issues.
- Evaluate performance on a larger, more representative dataset.

## 👨‍💻 Author

Created as part of a practical learning assignment in Natural Language Processing and Machine Learning.
