# 🚫 Common Mistakes — Bias–Variance Tradeoff

---

## ❌ Mistake 1

"High training accuracy means the model is good."

### ✅ Reality

Training accuracy alone is meaningless.

Always compare it with validation performance.

---

## ❌ Mistake 2

"More layers always improve performance."

### ✅ Reality

More layers increase learning capacity.

They also increase the risk of overfitting.

---

## ❌ Mistake 3

"High Bias means the model hasn't trained enough."

### ✅ Reality

High Bias usually means the model is too simple or has made overly strong assumptions.

---

## ❌ Mistake 4

"Training longer always improves accuracy."

### ✅ Reality

After a point, additional training often increases Variance and causes overfitting.

---

## ❌ Mistake 5

"Variance means prediction randomness."

### ✅ Reality

Variance measures how sensitive a model is to changes in the training data.

---

## ❌ Mistake 6

"Low Bias automatically means low Variance."

### ✅ Reality

Reducing Bias often increases Variance.

This is why it's called the **Bias–Variance Tradeoff**.

---

## ❌ Mistake 7

"The goal is zero Bias and zero Variance."

### ✅ Reality

That is practically impossible.

The goal is to achieve the best balance between them.

---

## ❌ Mistake 8

"Overfitting happens only in deep neural networks."

### ✅ Reality

Any sufficiently complex model can overfit:

- Decision Trees
- Random Forests
- SVMs
- Neural Networks

---

## ❌ Mistake 9

"Validation accuracy dropped, so increase the model size."

### ✅ Reality

A falling validation score usually suggests **High Variance**, not insufficient complexity.

Increasing complexity may make the problem worse.

---

## ❌ Mistake 10

"High Bias and High Variance are opposite metrics."

### ✅ Reality

They are different sources of error.

A model can have both.

Improving one can affect the other.

---

# 🎯 Final Interview Tip

When diagnosing any model, always answer these three questions:

1. How does it perform on the training set?
2. How does it perform on the validation/test set?
3. Is the train–validation gap small or large?

These three observations are usually enough to identify whether the problem is:

- High Bias
- High Variance
- Good Generalization