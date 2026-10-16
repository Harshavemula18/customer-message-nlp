# Assignment 5 — Customer Message Intelligence System

## 1. Dataset Understanding

**What is one training example?**

One customer message paired with its correct support category.

**What is the target label?**

The category assigned to each message.

**What are the four possible classes?**

Billing, Login, Delivery, and Product Support.

**Why must we represent text numerically?**

A normal machine learning classifier works with numeric features rather than raw sentences.

## 2. Training and Testing

The dataset contains 40 messages.

Training messages: 30

Test messages: 10

**Why test on unseen messages?**

To check whether the model can classify messages it did not learn from during training.

## 3. Unigram Experiment

Unigram accuracy: [Copy actual accuracy]

Incorrect test predictions: [Count from terminal output]

Correct prediction: [Write one example]

Incorrect prediction: [Write one example, if any]

Observation: Unigrams represent individual words as features.

## 4. Unigram and Bigram Experiment

Unigram + Bigram accuracy: [Copy actual accuracy]

**Did adding bigrams improve the result?**

[Compare the two actual accuracy values.]

**Why can bigrams be useful?**

Bigrams represent two-word combinations, such as "password reset" and "payment failed".

## 5. Six New Message Tests

1. Money disappeared from my card after checkout — Expected: Billing
2. I cannot get into my account — Expected: Login
3. My parcel should have arrived yesterday — Expected: Delivery
4. The app closes as soon as I open it — Expected: Product Support
5. I need my money back — Expected: Billing
6. The sign in code is missing — Expected: Login

Record the actual predictions from both models.

## 6. NLP Stress Test

**Message:** I can login but my payment keeps failing

Expected category: Billing

Observed prediction: Billing

Observation: The message contains two possible topics.

**Message:** My order arrived but the device does not work

Expected category: Product Support

Observed prediction: Product Support

Observation: The delivery is complete, but the device has a problem.

**Message:** Great, another package that never arrived

Expected category: Delivery

Observed prediction: Delivery

Observation: Sarcasm may be difficult to interpret.

**Message:** payment deduct ayyindi but order confirm avvaledu

Expected category: Billing

Observed prediction: Billing

Observation: The message contains code-mixed language.

**Message:** I need help with something

Expected category: Uncertain

Observed prediction: [Copy actual prediction]

Observation: The message provides very little information.

## 7. Rule-Based NLP

**Why are rules useful for extracting order IDs?**

Order IDs follow a predictable pattern, so a regular expression can find them.

**Why are fixed rules harder for delivery problems?**

Customers describe similar problems in many different ways.

**Which benefits more from learned language patterns?**

Message routing benefits more from learned language patterns, while order-ID extraction can use a simple rule.

## 8. My Own NLP Experiment

Create five new messages:

1. Clear message
2. Unfamiliar wording
3. Two possible topics
4. Informal or code-mixed language
5. A difficult message

For each, record the expected category, actual prediction, top score, correctness, and explanation.

## 9. Proposed Improvement

One possible improvement is to add more realistic customer messages, particularly examples involving unfamiliar wording and multiple topics.

## 10. Final NLP Inspector

Original message: Money disappeared from my card after checkout

Important words: money, card, checkout

Text representation: TF-IDF using unigrams and bigrams

Model: Logistic Regression

Predicted category: Delivery

Expected category: Billing

Prediction score: [Copy the actual Billing and Delivery scores]

Was the result correct? No.

What likely influenced the result? The training dataset contains limited examples of alternative ways to describe payment problems.

What limitation did this reveal? The model can struggle when a customer uses wording that differs from its training examples.

## 11. Final Reflection

My NLP system did not understand language like a human. It learned patterns from labelled customer messages, and it struggled when the wording differed from the training examples.

## Final NLP Inspector

**Original message:** Money disappeared from my card after checkout

**Important words / phrases:** money, disappeared, card, checkout

**How was the text represented?** TF-IDF with unigrams and bigrams.

**What model received these features?** Logistic Regression.

**Predicted category:** Delivery

**Expected category:** Billing

**Prediction score:** Copy the actual Billing and Delivery scores from the terminal.

**Was the result correct?** No.

**What likely influenced the result?** The model had limited examples of different ways to describe payment problems. It assigned a higher score to Delivery than Billing.

**What limitation did this example reveal?** The model can struggle when customer wording differs from the examples it learned from.

**Final reflection:**

My NLP system did not understand language like a human. It learned patterns from labelled customer messages, and it struggled when the wording differed from the training examples.