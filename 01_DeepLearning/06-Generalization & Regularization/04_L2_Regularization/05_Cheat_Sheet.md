# 📌 Cheat Sheet — L2 Regularization (Ridge Regularization)

---

# 🎯 Why was L2 Regularization Introduced?

L1 Regularization removes some features by making their weights exactly zero.

But in many real-world problems:

- Every feature contributes something.
- Some features are more important than others.
- We don't want to remove useful features.

L2 solves this by **shrinking weights instead of removing them.**

---

# 🧠 Core Idea

> Keep every feature, but discourage excessively large weights.

---

# 📐 L2 Regularization Formula

### Prediction Loss

\[
Loss_{prediction}
\]

### Regularization Loss

\[
\lambda \sum w_i^2
\]

### Total Loss

\[
Loss_{total}
=
Loss_{prediction}
+
\lambda \sum w_i^2
\]

---

# 🎯 Why Square the Weights?

Large weights receive much larger penalties.

Example:

| Weight | L1 Penalty | L2 Penalty |
|---------|-----------:|-----------:|
| 1 | 1 | 1 |
| 2 | 2 | 4 |
| 5 | 5 | 25 |
| 10 | 10 | 100 |

Large weights become expensive.

Small weights are only mildly penalized.

---

# ⚖️ Optimizer's Objective

The optimizer minimizes **one combined loss**, not two separate losses.

```
Prediction Loss
        ↓
Increase weights

Regularization Loss
        ↓
Decrease weights

Optimizer
        ↓
Finds the best balance
```

---

# 🎚️ Lambda (λ)

Controls regularization strength.

| λ Value | Effect |
|----------|--------|
| λ = 0 | No regularization |
| Small λ | Good balance |
| Large λ | Underfitting |

---

# 🏠 House Price Example

Before L2

```
Area        = 6.2
Bedrooms    = 3.0
Location    = 7.1
Parking     = 1.2
Age         = 1.8
```

After L2

```
Area        = 5.3
Bedrooms    = 2.7
Location    = 6.0
Parking     = 0.9
Age         = 1.5
```

Notice:

- No feature disappears.
- Every weight becomes smaller.

---

# 🔥 Why Is It Called Weight Decay?

Every optimization step slightly pulls weights toward zero.

Weights gradually become smaller over time.

Hence the name:

> **Weight Decay**

---

# 📉 Gradient Update

Without L2

\[
w=w-\eta\frac{\partial Loss}{\partial w}
\]

With L2

\[
w
=
w
-
\eta
\left(
\frac{\partial Loss}{\partial w}
+
2\lambda w
\right)
\]

Extra gradient:

\[
2\lambda w
\]

---

# ⚖️ L1 vs L2

| L1 | L2 |
|----|----|
| Uses \|w\| | Uses w² |
| Sparse model | Dense model |
| Removes features | Keeps all features |
| Some weights become exactly zero | Weights become smaller |
| Performs feature selection | Shrinks feature influence |

---

# 🔗 Correlated Features

Suppose:

```
House Area

Built-up Area
```

### L1

✅ Keeps one feature

❌ Removes the other

---

### L2

✅ Keeps both

✅ Shrinks both

---

# 🎯 Why Doesn't L2 Make Every Weight Zero?

As a weight becomes smaller:

- Prediction loss starts increasing.
- Benefit from shrinking it further becomes very small.

The optimizer naturally reaches a balance.

---

# 🤖 Where Is L2 Used?

✅ Linear Regression (Ridge)

✅ Logistic Regression

✅ Deep Neural Networks

✅ CNNs

✅ Recommendation Systems

---

# ⚠️ SGD vs AdamW

For **SGD**

```
L2 ≈ Weight Decay
```

For **Adam**

```
L2 ≠ Weight Decay
```

This is why **AdamW** was introduced.

---

# 🎯 Interview One-Liners

### Why L2?

> L2 discourages excessively large weights while keeping every feature in the model.

---

### Why square the weights?

> Squaring penalizes large weights much more aggressively than small ones.

---

### Why Weight Decay?

> Every optimization step gradually shrinks weights toward zero.

---

### Why doesn't L2 perform feature selection?

> Because its gradient becomes smaller as weights approach zero, so weights shrink but rarely become exactly zero.

---

### Why is L2 preferred in Deep Learning?

> Most deep learning features contribute useful information, so shrinking all weights is usually better than removing some features completely.

---

# 📝 Quick Revision

✅ Penalizes squared weights

✅ Keeps all features

✅ Produces dense models

✅ Controls model complexity

✅ Reduces overfitting

✅ Weight Decay = gradual shrinking of weights

✅ λ controls accuracy vs simplicity

✅ Large λ → Underfitting

✅ Small λ → Better generalization

✅ Preferred regularization technique in modern Deep Learning