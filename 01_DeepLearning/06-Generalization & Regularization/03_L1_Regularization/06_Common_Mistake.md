# ❌ Common Mistakes — L1 Regularization

---

# Mistake 1

## ❌ "Regularization improves training accuracy."

### ✅ Correct Understanding

Regularization usually **reduces training accuracy slightly**.

Its purpose is **not** to fit the training data better.

Its purpose is to improve **generalization** so the model performs better on unseen data.

---

# Mistake 2

## ❌ "Prediction Loss alone is enough."

### ✅ Correct Understanding

Prediction Loss only measures how well the model fits the **training dataset**.

The optimizer cannot distinguish

- genuine relationships
- accidental correlations
- noise
- outliers

Without regularization, it may happily assign large weights to noisy features if they reduce the training loss.

---

# Mistake 3

## ❌ "L1 removes important features."

### ✅ Correct Understanding

L1 tries to shrink **every weight**.

However, important features are protected by the Prediction Loss.

Removing an important feature would increase the Prediction Loss significantly.

Therefore, the optimizer keeps useful weights while gradually removing unimportant ones.

---

# Mistake 4

## ❌ "L1 explicitly detects correlated features."

### ✅ Correct Understanding

L1 never checks whether two features are correlated.

Instead, it minimizes

```text
Prediction Loss

+

λ × Regularization Loss
```

If two features provide almost identical information, keeping both unnecessarily increases the regularization penalty.

The optimizer therefore often keeps one feature and removes the other.

Feature selection is an **emergent property**, not an explicit algorithm.

---

# Mistake 5

## ❌ "L1 always removes the same correlated feature."

### ✅ Correct Understanding

Not necessarily.

When two correlated features contribute almost equally,

either one may survive.

The choice depends on optimization dynamics, initialization, and the training data.

---

# Mistake 6

## ❌ "A larger λ is always better."

### ✅ Correct Understanding

Lambda controls the balance between

- prediction accuracy
- model simplicity

A very large λ shrinks even useful weights.

This increases Bias and leads to **underfitting**.

Choosing λ is a hyperparameter tuning problem.

---

# Mistake 7

## ❌ "Regularization only changes the loss function."

### ✅ Correct Understanding

Regularization changes the **optimization objective**.

Because Total Loss changes,

its gradient also changes.

The optimizer therefore updates every weight using

```text
Prediction Gradient

+

Regularization Gradient
```

Regularization directly affects Gradient Descent.

---

# Mistake 8

## ❌ "The weights are constants, so the derivative of the L1 loss should be zero."

### ✅ Correct Understanding

During optimization,

the weights are **variables**, not constants.

The optimizer computes

```text
∂Loss/∂w
```

to determine how the loss changes when a weight changes slightly.

Since the Regularization Loss depends on the weights,

its derivative contributes to every weight update.

---

# Mistake 9

## ❌ "L1 only shrinks large weights."

### ✅ Correct Understanding

L1 applies a penalty to **every weight**.

Small weights simply reach zero much sooner because they contribute little to reducing the Prediction Loss.

---

# Mistake 10

## ❌ "L1 and Feature Selection are two different steps."

### ✅ Correct Understanding

Feature selection happens naturally during optimization.

As unimportant weights become zero,

those features automatically disappear from the model.

No separate feature-selection algorithm is required.

---

# Mistake 11

## ❌ "If a feature is useless, increasing its weight should never reduce the loss."

### ✅ Correct Understanding

On the **training dataset**, even useless features may accidentally correlate with the target because of

- random noise
- sampling effects
- outliers
- coincidental patterns

The optimizer cannot recognize these coincidences.

If increasing a weight reduces the training loss, it will increase that weight.

Regularization discourages the optimizer from relying too heavily on such accidental relationships.

---

# Mistake 12

## ❌ "L1 Regularization guarantees the best model."

### ✅ Correct Understanding

L1 is one tool for controlling model complexity.

Its effectiveness depends on

- λ value
- dataset size
- feature engineering
- model architecture
- optimization

Regularization improves the chances of better generalization—it does not guarantee it.

---

# 🎯 Final Takeaway

Remember this simple philosophy:

```text
Prediction Loss

↓

Fit the Training Data

Regularization

↓

Control Model Complexity

Together

↓

Generalize Well
```

A good model is **not** the one with the highest training accuracy.

A good model is the one that performs consistently on **unseen data**.