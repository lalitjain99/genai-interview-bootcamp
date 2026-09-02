# Assignment.md
# Lecture 2 — Bias-Variance Tradeoff

---

# Assignment 1 — Scenario Based Questions

## Scenario A

Two models are trained for the same task.

| Model | Training Accuracy | Validation Accuracy |
|--------|-------------------|---------------------|
| Model A | 94% | 92% |
| Model B | 99% | 78% |

Your manager wants to deploy **Model B** because it has the highest training accuracy.

Do you agree? Explain your reasoning.

---

### Answer

No.

Training accuracy alone is not a reliable metric for deployment.

Model A has a very small gap between training and validation accuracy, indicating **low variance** and good generalization.

Model B shows a large gap between training and validation accuracy, indicating **high variance (overfitting)**.

Instead of learning the true pattern, Model B has memorized the training data, which is why it performs poorly on unseen data.

For production systems, a model that generalizes well is far more valuable than one with extremely high training accuracy.

---

## Scenario B

A model achieves:

- Training Accuracy = 61%
- Validation Accuracy = 59%

What does this indicate?

How would you improve the model?

---

### Answer

This is a classic **high bias (underfitting)** problem.

The model is too simple and has made overly simplified assumptions, preventing it from learning the true relationship in the data.

Possible improvements:

- Increase model complexity
- Add more useful features
- Train for more epochs (if the model has not converged)

---

## Scenario C

A teammate suggests:

> "Validation accuracy is poor. Let's simply add more hidden layers."

Would you agree?

---

### Answer

Not necessarily.

Poor validation accuracy alone does not justify making the model larger.

If the model is already overfitting, adding more layers only increases its capacity to memorize noise, potentially making validation performance even worse.

The root cause should be identified first before increasing model complexity.

---

## Scenario D

You train the same model on five different training datasets.

Each trained model gives noticeably different predictions on the same unseen test data.

What property of the model does this indicate?

How could you improve it?

---

### Answer

This indicates **high variance**.

The model is overly sensitive to the training data and has not learned stable patterns.

Possible improvements:

- Collect more training data
- Reduce model complexity
- Apply regularization

---

# Assignment 2 — Conceptual Questions

---

## Q1.

Why is generalization more important than achieving very high training accuracy?

---

### Answer

Because a model is ultimately evaluated on unseen data.

A model that generalizes has learned the underlying pattern rather than memorizing the training dataset, making it reliable in production.

---

## Q2.

Explain High Bias and High Variance using the analogy of two students preparing for an exam.

---

### Answer

### High Bias Student

- Learns one oversimplified rule
- Applies it everywhere
- Performs consistently poorly

This resembles **underfitting**.

---

### High Variance Student

- Memorizes every practice paper
- Scores perfectly on familiar questions
- Performs poorly when questions change

This resembles **overfitting**.

---

### Ideal Student

The ideal student understands concepts rather than memorizing answers.

Similarly, a good ML model learns general patterns instead of either oversimplifying or memorizing.

---

## Q3.

A model achieves:

- Training Accuracy = 99%
- Validation Accuracy = 75%

Would collecting more data help? Why?

---

### Answer

Yes.

This is a classic high variance problem.

More training data makes memorization harder and encourages the model to learn stable, generalizable patterns instead of fitting noise.

---

# Assignment 3 — Analytical Question

Two models fail differently.

- **Model A** performs poorly even on training data.
- **Model B** performs perfectly on training data but poorly on unseen data.

Explain why these problems have different names and require different solutions.

---

### Answer

### Model A

The model cannot even learn the training data.

This is **underfitting (High Bias)**.

The model is too simple and lacks sufficient learning capacity.

Typical solutions:

- Increase model complexity
- Add useful features
- Reduce overly simplified assumptions

---

### Model B

The model memorizes the training data but fails on unseen data.

This is **overfitting (High Variance)**.

Typical solutions:

- Collect more data
- Apply regularization
- Reduce model complexity
- Use Early Stopping

Although both models perform poorly, they fail for completely different reasons and therefore require opposite solutions.

---

# Assignment 4 — True or False

1. A model with low Bias always has low Variance.
> **False**

2. Increasing model complexity always improves performance.
> **False**

3. A model can have poor training accuracy and still generalize well.
> **False**

4. High Variance models are highly sensitive to changes in training data.
> **True**

5. The goal of Machine Learning is to eliminate both Bias and Variance completely.
> **False**

---

# Staff Engineer Challenge

A recommendation model shows the following metrics:

| Dataset | Accuracy |
|----------|----------|
| Training | 98% |
| Validation | 97% |
| Production (after 2 weeks) | 82% |

A teammate concludes:

> "This is a High Variance problem."

Do you agree?

---

### Answer

Not immediately.

A High Variance model typically shows a large gap between training and validation accuracy. Here, the gap is only **1%**, indicating good generalization during offline evaluation.

Since the performance dropped only after deployment, other causes are more likely.

Possible reasons include:

- **Data Drift / Concept Drift** — user behavior or product catalog changed after training.
- **Train-Serve Skew** — production feature pipeline differs from the training pipeline.
- **Feedback Loops** — recommendations influence future user behavior.
- **Data Leakage** during offline validation, making validation accuracy artificially high.
- **Metric Mismatch** — offline accuracy may not reflect the actual production objective (CTR, conversion rate, etc.).

Before blaming the model, I would investigate the data pipeline, feature consistency, evaluation strategy, and production environment.
