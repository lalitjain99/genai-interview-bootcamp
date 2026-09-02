# 📌 Bias–Variance Tradeoff Cheat Sheet

---

# 🎯 Goal of Machine Learning

Learn the **true underlying pattern** so the model performs well on **unseen data**.

Not too simple.
Not too complex.

---

# 🧠 Bias

**Definition**

Error caused by overly simplified assumptions.

**Characteristics**

- Model too simple
- Underfits
- Misses true relationships
- Low Training Accuracy
- Low Validation Accuracy

**Typical Causes**

- Small model
- Too few features
- Oversimplified assumptions
- Insufficient training

**How to Reduce Bias**

- Increase model capacity
- Add better features
- Train longer
- Reduce restrictive assumptions

---

# 🧠 Variance

**Definition**

Sensitivity of a model to changes in training data.

**Characteristics**

- Model too complex
- Overfits
- Memorizes noise
- High Training Accuracy
- Low Validation Accuracy

**Typical Causes**

- Large model
- Small dataset
- Excessive training
- Weak regularization

**How to Reduce Variance**

- More training data
- Simpler model
- Regularization
- Early stopping
- Better train/validation split

---

# ⚖️ Bias–Variance Tradeoff

```
Too Simple                     Balanced                     Too Complex

High Bias  ------------------>  Sweet Spot  <------------------ High Variance

Underfitting                                         Overfitting
```

The objective is **not** to eliminate Bias or Variance completely.

The objective is to **balance both**.

---

# 📊 Accuracy Patterns

### High Bias

Training Accuracy ↓

Validation Accuracy ↓

Small Gap

---

### High Variance

Training Accuracy ↑

Validation Accuracy ↓

Large Gap

---

### Good Generalization

Training Accuracy ↑

Validation Accuracy ↑

Small Gap

---

# 🧠 Easy Memory Trick

High Bias

> "Model cannot learn."

High Variance

> "Model learns too much."

Balanced Model

> "Model learns the right amount."

---

# 🚨 Interview Formula

High Training + Low Validation

→ Overfitting
→ High Variance

Low Training + Low Validation

→ Underfitting
→ High Bias

High Training + High Validation

→ Good Generalization

---

# 💡 Golden Rule

Don't chase the highest training accuracy.

Chase the highest validation performance with the smallest gap.

That is the model most likely to succeed in production.