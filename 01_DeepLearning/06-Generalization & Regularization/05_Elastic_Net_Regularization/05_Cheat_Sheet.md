# 📋 Elastic Net Regularization — Cheat Sheet

---

# 🎯 Why was Elastic Net introduced?

L1 and L2 both have strengths and weaknesses.

### L1

✅ Performs feature selection

❌ Removes useful correlated features

---

### L2

✅ Handles correlated features well

❌ Cannot remove irrelevant features

---

### Elastic Net

✅ Performs feature selection

✅ Preserves correlated features

✅ Shrinks large weights

---

# 📐 Elastic Net Loss Function

```text
Total Loss
=
Prediction Loss
+
λ × [ α × Σ|w| + (1 − α) × Σw² ]
```

---

# 🎛 Hyperparameters

## λ (Lambda)

Controls **overall regularization strength**

```text
Small λ
→ Weak regularization

Large λ
→ Strong regularization
```

---

## α (Alpha)

Controls the mixture of L1 and L2.

```text
α = 1
→ Pure L1

α = 0
→ Pure L2

0 < α < 1
→ Elastic Net
```

---

# ⚖️ Role of Each Penalty

## L1 Component

```text
α × Σ|w|
```

- Removes irrelevant features
- Produces sparse models
- Performs feature selection

---

## L2 Component

```text
(1−α) × Σw²
```

- Shrinks large weights
- Handles correlated features
- Produces stable models

---

# 🔄 Optimizer Objective

The optimizer minimizes

```text
Prediction Loss
+
L1 Penalty
+
L2 Penalty
```

It simultaneously balances:

- 🎯 Prediction Accuracy
- 🎯 Feature Selection
- 🎯 Weight Shrinkage

---

# 📊 Comparison

| Property | L1 | L2 | Elastic Net |
|----------|----|----|-------------|
| Feature Selection | ✅ | ❌ | ✅ |
| Sparse Model | ✅ | ❌ | ✅ Partial |
| Weight Shrinkage | ❌ | ✅ | ✅ |
| Correlated Features | ❌ | ✅ | ✅ |
| Removes Noise | ✅ | ❌ | ✅ |
| Stable Model | ❌ | ✅ | ✅ |

---

# 🧠 When to Use What?

## ✅ L1

- Few useful features
- Need feature selection
- High interpretability

---

## ✅ L2

- Most features are useful
- Strong multicollinearity
- Want stable coefficients

---

## ✅ Elastic Net

- Many correlated features
- Many noisy features
- Need feature selection + stability

---

# 📈 Effect of λ

```text
λ = 0
```

- No regularization
- High variance
- Risk of overfitting

↓

```text
Small λ
```

- Best balance
- Good generalization

↓

```text
Large λ
```

- Strong shrinkage
- Underfitting

---

# 🎯 Effect of α

```text
α = 1
```

➡ Pure L1

↓

```text
0 < α < 1
```

➡ Elastic Net

↓

```text
α = 0
```

➡ Pure L2

---

# 💡 Advantages

✅ Combines strengths of L1 and L2

✅ Removes noisy features

✅ Preserves correlated features

✅ More stable than L1

✅ Better generalization on high-dimensional datasets

---

# ⚠️ Limitations

❌ Two hyperparameters to tune (λ and α)

❌ Higher computational cost

❌ Not always better than pure L1 or L2

❌ Requires Cross Validation for tuning

---

# 🎤 Interview One-Liners

### Why Elastic Net?

> Combines L1's feature selection with L2's weight shrinkage and grouping effect.

---

### Why better than L1?

> L1 arbitrarily removes correlated features; Elastic Net preserves them through its L2 component.

---

### Why better than L2?

> L2 shrinks all weights but cannot remove irrelevant features. Elastic Net can.

---

### What does λ control?

> Overall regularization strength.

---

### What does α control?

> Balance between L1 and L2.

---

### When is Elastic Net preferred?

> When the dataset contains both correlated features and irrelevant/noisy features.

---

# 🚀 Staff Engineer Takeaway

Regularization technique selection should depend on:

- Dataset characteristics
- Feature correlation
- Noise level
- Interpretability requirements
- Business constraints

There is **no universally best regularization technique**.