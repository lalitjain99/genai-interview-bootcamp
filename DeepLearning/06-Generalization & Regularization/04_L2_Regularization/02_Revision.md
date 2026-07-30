# 📘 Revision Notes — L2 Regularization (Ridge Regularization)

---

# 🎯 Core Idea

L2 Regularization discourages **large weights** by adding the **sum of squared weights** to the loss function.

Instead of removing features, it **shrinks their influence**.

---

# 📌 Why Do We Need L2?

Large weights often indicate that the model is relying too heavily on certain features.

This can cause:

- Overfitting
- Poor generalization
- Unstable predictions

L2 encourages the model to distribute learning across multiple features instead of depending excessively on a few.

---

# 📌 L2 Regularization Equation

```text
Regularization Loss

=

Σ wi²
```

Total Loss becomes

```text
Total Loss

=

Prediction Loss

+

λ × Σ wi²
```

---

# 📌 Why Square the Weights?

Squaring provides two important benefits:

✅ Large weights receive a much larger penalty.

✅ Negative and positive weights contribute equally.

Example

```text
Weight = 1

Penalty = 1
```

```text
Weight = 5

Penalty = 25
```

Large weights become expensive.

---

# 📌 Effect on Weights

Before L2

```text
Area         = 4.8

Bedrooms     = 2.4

Location     = 5.3

Parking      = 0.8
```

After L2

```text
Area         = 4.2

Bedrooms     = 2.1

Location     = 4.7

Parking      = 0.6
```

Notice

- No feature disappears.
- Every feature remains.
- All weights become smaller.

---

# 📌 How Does the Optimizer Balance Two Losses?

The optimizer does **not** minimize Prediction Loss and Regularization Loss separately.

It minimizes only

```text
Total Loss

=

Prediction Loss

+

λ × Regularization Loss
```

During backpropagation

```text
∂(Total Loss)/∂w

=

∂(Prediction Loss)/∂w

+

λ × ∂(Regularization Loss)/∂w
```

Think of two competing forces:

📈 **Prediction Gradient**

- Improve model accuracy.
- May increase or decrease weights.

📉 **Regularization Gradient**

- Pulls weights toward zero.
- Prevents overly complex models.

The optimizer updates weights using the **combined gradient**.

---

# 📌 Role of λ (Lambda)

λ controls the strength of regularization.

Small λ

```text
Prediction Loss dominates.
```

Large λ

```text
Regularization dominates.
```

Typical values

```text
0.0001

0.001

0.01

0.1
```

Usually selected through hyperparameter tuning.

---

# 📌 Why Doesn't L2 Produce Zero Weights?

L2 continuously shrinks weights.

As weights become smaller,

the improvement in Regularization Loss also becomes smaller.

Eventually,

the increase in Prediction Loss outweighs the benefit of shrinking further.

The optimizer stops before weights reach exactly zero.

Result

```text
Small weights

≠

Zero weights
```

---

# 📌 L2 and Correlated Features

Suppose two highly correlated features exist.

```text
House Area

Living Area
```

Instead of removing one,

L2 usually keeps both

with smaller weights.

Learning is distributed across correlated features.

---

# 📌 Why Is L2 Called Weight Decay?

Every optimization step gently pulls weights toward zero.

Weights gradually become smaller over training.

Hence the name

```text
Weight Decay
```

---

# 📌 L1 vs L2

| Property | L1 | L2 |
|----------|----|----|
| Penalty | \|w\| | w² |
| Removes Features | ✅ Yes | ❌ No |
| Sparse Model | ✅ Yes | ❌ No |
| Shrinks Weights | Moderate | Strong |
| Feature Selection | ✅ Automatic | ❌ No |
| Correlated Features | Keeps one | Keeps both |

---

# 🎯 Interview Answers

### Why is L2 called Ridge Regression?

Because it adds the squared magnitude of weights to the loss function.

---

### Why square the weights?

To penalize large weights much more heavily while treating positive and negative weights equally.

---

### Why doesn't L2 perform feature selection?

Because squaring produces a smooth penalty that continuously shrinks weights instead of driving them exactly to zero.

---

### Why is L2 called Weight Decay?

Because each optimization step gradually pulls weights toward zero.

---

### How does the optimizer balance Prediction Loss and Regularization Loss?

It minimizes the **Total Loss**, not each loss independently.

The weight update uses the combined gradient

```text
Prediction Gradient

+

Regularization Gradient
```

which naturally balances model accuracy and model complexity.

---

# ⚠️ Common Mistakes

❌ L2 removes useless features.

✅ L2 keeps almost all features but reduces their weights.

---

❌ L2 makes every weight zero.

✅ L2 usually produces small, non-zero weights.

---

❌ Weight Decay and Prediction Loss are optimized separately.

✅ The optimizer minimizes one objective:

```text
Total Loss

=

Prediction Loss

+

λ × Regularization Loss
```

---

❌ Increasing λ always improves generalization.

✅ A very large λ can cause underfitting.

---

# 🧠 One-Line Memory Trick

> **L1 asks:** "Should this feature exist?"

> **L2 asks:** "How much influence should this feature have?"

L1 creates **sparse models**.

L2 creates **smooth models**.