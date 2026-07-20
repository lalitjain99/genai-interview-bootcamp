# 📝 Revision Notes — Bias-Variance Tradeoff

---

# 🎯 Core Idea

Every Machine Learning model can fail in **one of two ways**.

```text
Model Too Simple
        │
        ▼
Cannot Learn the Pattern
        │
        ▼
High Bias
```

```text
Model Too Complex
        │
        ▼
Memorizes the Training Data
        │
        ▼
High Variance
```

The goal is **not** to eliminate Bias or Variance.

The goal is to **balance both**.

---

# 🧠 Why Was This Concept Introduced?

In the previous lecture, we learned about **Overfitting**.

Researchers initially thought:

> "If a large model overfits, let's simply make it smaller."

But smaller models created another problem.

```text
Large Model
        │
        ▼
Overfitting

Small Model
        │
        ▼
Underfitting
```

Researchers realized that **both extremes are harmful**.

This led to the concept of **Bias-Variance Tradeoff**.

---

# 🔵 Bias

## Definition

Bias is the **systematic error introduced because the model is too simple**.

The model cannot learn the true relationship in the data.

```text
Model Too Simple
        │
        ▼
Low Capacity
        │
        ▼
Fails to Learn
        │
        ▼
High Bias
```

---

## Characteristics

- Model is too simple.
- Cannot capture complex patterns.
- High training loss.
- High validation loss.
- Underfitting.

---

## Why is it called "Bias"?

A simple model makes a **built-in assumption** about the data.

Example:

Linear Regression assumes

```text
Output ≈ Straight Line
```

If the real relationship is complex,

this assumption introduces a **systematic error**.

That systematic error is called **Bias**.

---

# 🔴 Variance

## Definition

Variance measures **how sensitive a model is to changes in the training data**.

A highly complex model learns not only the real pattern,

but also the noise present in the training data.

```text
Model Too Complex
        │
        ▼
High Capacity
        │
        ▼
Memorizes Training Data
        │
        ▼
High Variance
```

---

## Characteristics

- Model is overly complex.
- Learns noise and outliers.
- Very low training loss.
- High validation loss.
- Overfitting.

---

## Why is it called "Variance"?

Imagine training the same model using slightly different datasets.

```text
Dataset A
      │
      ▼
Model A

Dataset B
      │
      ▼
Model B

Dataset C
      │
      ▼
Model C
```

If small changes in data produce very different models,

the predictions **vary** significantly.

Hence the name **Variance**.

---

# ⚖️ Bias vs Variance

| Bias | Variance |
|------|----------|
| Model is too simple | Model is too complex |
| Underfitting | Overfitting |
| High training loss | Low training loss |
| High validation loss | High validation loss |
| Cannot learn enough | Learns too much |
| Poor training accuracy | Excellent training accuracy |
| Stable model | Sensitive to data changes |

---

# 📈 The Tradeoff

As model complexity increases,

```text
Bias

High
 │\
 │ \
 │  \
 │   \
 │    \
 └──────────────► Model Complexity
```

Bias decreases.

---

Variance behaves in the opposite way.

```text
Variance

      /
     /
    /
   /
  /
 /
└──────────────► Model Complexity
```

Variance increases.

---

The ideal model lies somewhere in the middle.

```text
Too Simple
      │
      ▼
High Bias

──────────── Sweet Spot ────────────

High Variance
      ▲
      │
Too Complex
```

This balance is called the **Bias-Variance Tradeoff**.

---

# 💡 First Principle

Increasing model complexity

```text
Bias ↓

Variance ↑
```

Decreasing model complexity

```text
Variance ↓

Bias ↑
```

Improving one often worsens the other.

---

# 🌍 Why This Matters

Almost every regularization technique exists to manage this tradeoff.

```text
Bias-Variance Tradeoff
          │
          ▼
Need Better Generalization
          │
          ▼
L1 Regularization
          │
          ▼
L2 Regularization
          │
          ▼
Weight Decay
          │
          ▼
Early Stopping
          │
          ▼
Data Augmentation
          │
          ▼
Dropout
          │
          ▼
Batch Normalization
```

Their common goal is to reduce **Variance** while keeping **Bias** as low as possible.

---

# 🎯 Interview Takeaways

- A model can fail because it is **too simple** or **too complex**.
- Bias is caused by **overly simple assumptions**.
- Variance measures **sensitivity to the training data**.
- High Bias leads to **Underfitting**.
- High Variance leads to **Overfitting**.
- Increasing complexity decreases Bias but increases Variance.
- The objective is to find the **optimal balance** between the two.

---

# 📌 One-Line Summary

> **Bias means the model cannot learn enough. Variance means the model learns too much. Good Machine Learning is the art of balancing both.**