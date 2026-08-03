# Revision Notes — Elastic Net Regularization

---

# 🎯 Why Was Elastic Net Introduced?

We already had two regularization techniques.

### L1 Regularization

- Performs feature selection.
- Removes less useful features.
- Produces sparse models.

Problem:

- May remove useful correlated features.

---

### L2 Regularization

- Keeps every feature.
- Shrinks large weights.
- Produces dense models.

Problem:

- Cannot remove irrelevant features.

---

Researchers asked:

> Can we combine the strengths of both?

This led to **Elastic Net Regularization.**

---

# 🧠 Core Idea

Elastic Net combines

- Feature Selection (L1)
- Weight Shrinkage (L2)

into a single regularization technique.

---

# 📐 Elastic Net Formula

```text
Total Loss
    =
Prediction Loss
    +
λ × [ α × Σ|w| + (1 − α) × Σw² ]
```

---

# 📌 Components of the Equation

### Prediction Loss

Measures how accurately the model predicts.

---

### L1 Term

```text
α × Σ|w|
```

Purpose

- Pushes some weights to exactly zero.
- Performs feature selection.

---

### L2 Term

```text
(1 − α) × Σw²
```

Purpose

- Shrinks large weights.
- Keeps all useful features.

---

# 🎚️ Role of λ (Lambda)

Controls the **overall strength** of regularization.

```text
λ = 0
```

No regularization.

---

```text
Small λ
```

Mild regularization.

Usually preferred.

---

```text
Large λ
```

Strong regularization.

May cause underfitting.

---

# 🎚️ Role of α (Alpha)

Controls the mixture of L1 and L2.

```text
α = 1
```

Pure L1

---

```text
α = 0
```

Pure L2

---

```text
0 < α < 1
```

Elastic Net

Combination of both.

---

# 🏠 Example

Features

```text
Area

Built-up Area

Bedrooms

Parking

Wall Color

Owner Name
```

After Elastic Net

```text
Area           = 3.9

Built-up Area  = 3.8

Bedrooms       = 2.0

Parking        = 0.7

Wall Color     = 0

Owner Name     = 0
```

Notice

- Important correlated features remain.
- Large weights become smaller.
- Useless features disappear.

---

# ⚖️ L1 vs L2 vs Elastic Net

| L1 | L2 | Elastic Net |
|----|----|-------------|
| Removes features | Keeps all features | Removes useless features + shrinks useful ones |
| Sparse model | Dense model | Semi-sparse model |
| Good for feature selection | Good for correlated features | Good for both |

---

# 🔗 Behavior with Correlated Features

Suppose

```text
Area

Built-up Area
```

Both describe house size.

### L1

Keeps one.

Removes the other.

---

### L2

Keeps both.

Shrinks both.

---

### Elastic Net

Usually keeps both correlated features,

while still removing truly irrelevant ones.

---

# 💼 When Should We Use Elastic Net?

Use Elastic Net when

✅ Dataset has many features.

✅ Many features are correlated.

✅ Some features are irrelevant.

Typical applications

- Finance
- Healthcare
- Bioinformatics
- Recommendation Systems
- High-dimensional tabular data

---

# 🎯 Advantages

✅ Combines L1 and L2.

✅ Performs feature selection.

✅ Shrinks large weights.

✅ Handles correlated features better than L1.

✅ Produces stable models.

✅ Helps reduce overfitting.

---

# ⚠️ Limitations

- Two hyperparameters must be tuned (λ and α).
- Slightly more computationally expensive than using only L1 or L2.
- Not always better than pure L1 or pure L2—the choice depends on the dataset.

---

# 📝 Interview One-Liners

### Why Elastic Net?

> Elastic Net combines the feature selection capability of L1 with the weight shrinkage capability of L2.

---

### Role of λ?

> Controls the overall strength of regularization.

---

### Role of α?

> Controls the proportion of L1 and L2 regularization.

---

### When should Elastic Net be preferred?

> When the dataset contains many correlated features along with some irrelevant features.

---

### Does Elastic Net produce sparse models?

> It can produce partially sparse models because the L1 component can drive some weights to zero, while the L2 component shrinks the remaining weights.

---

# 🚀 Key Takeaways

- Elastic Net combines L1 and L2.
- L1 removes features.
- L2 shrinks weights.
- λ controls regularization strength.
- α controls the L1/L2 mixture.
- Useful for correlated and high-dimensional datasets.
- Balances sparsity, stability, and generalization.
```