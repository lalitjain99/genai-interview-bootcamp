# 💡 Assignment — Why Models Overfit

---

# 🎯 Assignment 1 — Choose the Better Model

---

## Scenario A

Model A

```text
Training Accuracy : 95%

Validation Accuracy : 93%
```

Model B

```text
Training Accuracy : 100%

Validation Accuracy : 79%
```

Which model would you deploy?

### Answer

I would deploy **Model A**, assuming all other factors such as train-validation split, data leakage, and data distribution are correct.

Model A has a very small gap between training and validation accuracy, which indicates that it has learned the underlying patterns in the data rather than memorizing the training examples.

Model B achieves perfect training accuracy but performs significantly worse on validation data, indicating overfitting.

---

## Scenario B

Two neural networks are trained on the same dataset.

Network A

- 2 Hidden Layers
- 100K Parameters

Network B

- 40 Hidden Layers
- 150 Million Parameters

The dataset contains only 5,000 training examples.

Which network is more likely to overfit?

### Answer

**Network B** is much more likely to overfit.

It contains significantly more parameters and therefore has much higher learning capacity.

```text
More Parameters
        ↓
Higher Capacity
        ↓
Can Learn Complex Patterns
        ↓
Can Also Memorize Noise
```

Large models are useful when we have massive and complex datasets.

However, using such a large model on only 5,000 training examples greatly increases the chance of memorization instead of generalization.

---

## Scenario C

A student memorizes every previous year's exam paper.

Another student studies concepts and solves different kinds of questions.

Which student is more likely to perform well if the exam pattern changes?

### Answer

The second student is more likely to perform well.

They understand the concepts rather than memorizing a fixed set of questions.

The same principle applies in Machine Learning.

A model that learns the underlying patterns can generalize to unseen data, whereas a model that memorizes the training dataset performs poorly when presented with new examples.

---

## Scenario D

You continue training a model for another 200 epochs.

Training loss keeps decreasing.

Validation loss starts increasing.

Should you continue training?

### Answer

No.

I would stop training and investigate why the validation loss is increasing.

A decreasing training loss together with an increasing validation loss is a strong indication that the model has started memorizing the training dataset instead of learning general patterns.

Continuing the training is likely to make overfitting even worse.

---

# 🧠 Assignment 2 — Explain the Engineering Reasoning

---

## Q1. Why isn't achieving 100% training accuracy always a good objective?

### Answer

Consider two students preparing for an exam.

### Student A

- Understands concepts
- Solves unseen questions

### Student B

- Memorizes previous question papers
- Scores 100% on practice tests
- Fails when the questions change

Machine Learning models behave similarly.

A model can achieve 100% training accuracy simply by memorizing the training dataset.

If it performs poorly on unseen data, it is still a poor model despite its perfect training accuracy.

---

## Q2. Why is learning patterns more valuable than memorizing examples?

### Answer

Learning patterns allows the model to make accurate predictions on unseen data.

Memorizing examples only helps on the training dataset.

The true objective of Machine Learning is not memorization—it is **generalization**.

---

## Q3. Explain Signal vs Noise using any real-world example.

### Answer

**Signal** represents the true relationship that we want the model to learn.

Examples:

- Cats have whiskers.
- Larger houses generally cost more.
- Positive reviews usually contain positive words.

These relationships continue to hold beyond the training dataset.

---

**Noise** refers to random information that accidentally appears in the training data.

Examples:

- Camera angle
- Lighting conditions
- Background objects
- Typographical errors
- Sensor noise

Noise does not generalize.

A good model ignores noise.

An overfit model memorizes it.

---

## Q4. Why do larger neural networks generally require more careful training than smaller models?

### Answer

Large neural networks contain more parameters.

```text
More Parameters
        ↓
Higher Capacity
        ↓
Can Learn Complex Patterns
        ↓
Can Also Memorize Noise
```

Their higher capacity allows them to solve very complex problems.

However, the same capacity also makes it easier to memorize the training dataset.

Therefore, larger models require more careful training and stronger regularization to achieve good generalization.

---

# 🚀 Assignment 3 — Think Like the Inventor

Imagine you are one of the earliest Deep Learning researchers.

You observe:

```text
Training Loss ↓

Validation Loss ↑
```

What would you conclude?

### Answer

This behavior suggests that the model has started overfitting.

Instead of learning general patterns, it has begun memorizing the training data, including its noise.

Possible reasons include:

- Small training dataset
- Model capacity is too large for the available data
- Poor train-validation split
- Data leakage
- Validation data distribution differs significantly from the training data

My first intuition would be:

> **Prevent the model from continuing to memorize the training data.**

One immediate practical action would be to stop training once validation performance starts degrading.

---

# ⚡ Assignment 4 — True or False

---

### A model with lower training loss is always better.

❌ False

---

### Overfitting means the model has learned useful patterns.

❌ False

---

### Large models generally have higher learning capacity.

✅ True

---

### Noise helps models generalize better.

❌ False

---

### A model can achieve perfect training accuracy and still fail in production.

✅ True

---

# 🌍 Staff Engineer Challenge

Your company has built an AI model for medical diagnosis.

After deployment:

```text
Training Accuracy : 99%

Validation Accuracy : 95%

Production Accuracy : 68%
```

Management asks:

> "Should we simply train the model for more epochs?"

### Answer

No.

Training for more epochs is unlikely to solve the problem because both the training and validation performance are already strong.

Instead, I would first investigate the data and deployment pipeline.

Questions I would ask include:

- Is the data preparation pipeline correct?
- Was the train-validation split performed correctly?
- Is there any data leakage?
- Is the production data significantly different from the training and validation data?
- Is the chosen model appropriate for this problem?

Only after understanding the root cause would I recommend any changes to the model or training process.