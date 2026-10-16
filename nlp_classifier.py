import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Load the dataset
data = pd.read_csv("support_messages.csv")

print("DATASET PREVIEW")
print(data.head())
print("\nTotal messages:", len(data))
print("\nMessages per category:")
print(data["category"].value_counts())

# Step 2: Separate messages and labels
X = data["message"]
y = data["category"]

# Step 3: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("\nTraining messages:", len(X_train))
print("Test messages:", len(X_test))

# Step 4: Version A - Unigrams
unigram_model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 1))),
    ("classifier", LogisticRegression(max_iter=1000))
])

unigram_model.fit(X_train, y_train)

unigram_predictions = unigram_model.predict(X_test)
unigram_accuracy = accuracy_score(y_test, unigram_predictions)

print("\nUNIGRAM ACCURACY:", round(unigram_accuracy, 3))

# Step 5: Version B - Unigrams and Bigrams
bigram_model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
    ("classifier", LogisticRegression(max_iter=1000))
])

bigram_model.fit(X_train, y_train)

bigram_predictions = bigram_model.predict(X_test)
bigram_accuracy = accuracy_score(y_test, bigram_predictions)

print("UNIGRAM + BIGRAM ACCURACY:", round(bigram_accuracy, 3))

# Step 6: Compare test predictions
results = pd.DataFrame({
    "message": X_test,
    "actual": y_test,
    "unigram_prediction": unigram_predictions,
    "bigram_prediction": bigram_predictions
})

print("\nTEST PREDICTIONS")
print(results.to_string(index=False))

# Step 7: Test new customer messages
new_messages = [
    "Money disappeared from my card after checkout",
    "I cannot get into my account",
    "My parcel should have arrived yesterday",
    "The app closes as soon as I open it",
    "I need my money back",
    "The sign in code is missing"
]

print("\nNEW MESSAGE TESTS")

for message in new_messages:
    unigram_result = unigram_model.predict([message])[0]
    bigram_result = bigram_model.predict([message])[0]

    print("\nMessage:", message)
    print("Unigram prediction:", unigram_result)
    print("Bigram prediction:", bigram_result)

# Step 8: Show prediction scores
print("\nPREDICTION CONFIDENCE SCORES")

probabilities = bigram_model.predict_proba(new_messages)
classes = bigram_model.classes_

for message, scores in zip(new_messages, probabilities):
    ranked = sorted(
        zip(classes, scores),
        key=lambda item: item[1],
        reverse=True
    )

    print("\nMessage:", message)

    for category, score in ranked:
        print(category, "=", round(score, 3))

# Step 9: NLP stress test
tricky_messages = [
    "I can login but my payment keeps failing",
    "My order arrived but the device does not work",
    "Great, another package that never arrived",
    "payment deduct ayyindi but order confirm avvaledu",
    "I need help with something"
]

print("\nNLP STRESS TEST")

for message, prediction in zip(
    tricky_messages,
    bigram_model.predict(tricky_messages)
):
    print("\nMessage:", message)
    print("Prediction:", prediction)
# Step 10: My Own NLP Experiment

my_messages = [
    "Please refund the money taken from my account",
    "I am unable to enter my profile",
    "My parcel is late and my device is broken",
    "payment deduct ayyindi",
    "Everything is completely useless and nothing works"
]

expected_categories = [
    "Billing",
    "Login",
    "Delivery",
    "Billing",
    "Product Support"
]

print("\nMY OWN NLP EXPERIMENT")

my_predictions = bigram_model.predict(my_messages)
my_probabilities = bigram_model.predict_proba(my_messages)

for message, expected, predicted, scores in zip(
    my_messages,
    expected_categories,
    my_predictions,
    my_probabilities
):
    ranked = sorted(
        zip(bigram_model.classes_, scores),
        key=lambda item: item[1],
        reverse=True
    )

    top_category, top_score = ranked[0]

    print("\nMessage:", message)
    print("Expected:", expected)
    print("Predicted:", predicted)
    print("Top score:", round(top_score, 3))
    print("Correct:", expected == predicted)
    print("Top two categories:", ranked[:2])