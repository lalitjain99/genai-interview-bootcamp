# 📘 Revision — L1 Regularization

---

# 🎯 Learning Objectives

After revising this chapter, you should be able to explain:

- Why regularization is needed.
- What overfitting actually means.
- How L1 Regularization works.
- Why L1 performs feature selection.
- How λ controls the balance between accuracy and simplicity.
- How L1 affects Gradient Descent.
- Why correlated features often disappear.

---

# 1️⃣ Why Do We Need Regularization?

A model should not simply memorize the training dataset.

Our real goal is:

```text
Training Data

        ↓

Learn General Patterns

        ↓

Perform Well on Unseen Data
```

A model that performs well on unseen data is said to **generalize well**.

---

# 2️⃣ What is Overfitting?

Overfitting occurs when the model starts learning

- Noise
- Outliers
- Random fluctuations
- Accidental correlations

instead of learning the true relationship in the data.

Typical symptom:

```text
Training Accuracy   → High

Validation Accuracy → Low
```

The model memorizes instead of understanding.

---

# 3️⃣ Core Idea Behind Regularization

Instead of minimizing only Prediction Loss,

we also penalize model complexity.

```text
Total Loss

=

Prediction Loss

+

Regularization Loss
```

The optimizer now tries to

✔ Reduce prediction error

✔ Keep the model simple

at the same time.

---

# 4️⃣ L1 Regularization

L1 adds the absolute value of every weight to the loss.

```text
Regularization Loss

=

|w₁|

+

|w₂|

+

...

+

|wₙ|
```

Therefore,

```text
Total Loss

=

Prediction Loss

+

λ × L1 Loss
```

---

# 5️⃣ What Does λ (Lambda) Do?

λ controls the trade-off.

Small λ

↓

Focus more on prediction accuracy.

Large λ

↓

Focus more on reducing weights.

```text
Small λ

↓

Better Fit

Higher Risk of Overfitting


Large λ

↓

Simpler Model

Higher Risk of Underfitting
```

---

# 6️⃣ Why Doesn't L1 Remove Every Weight?

The optimizer minimizes

```text
Prediction Loss

+

λ × Regularization Loss
```

Two forces act on every weight.

```text
Prediction Loss

↓

Keep useful weights.


Regularization

↓

Shrink every weight.
```

Important features significantly reduce prediction error.

Prediction Loss fights to keep them.

Unimportant features contribute very little.

Regularization gradually pushes them toward zero.

---

# 7️⃣ Why Does L1 Perform Feature Selection?

Suppose two features contain almost identical information.

Example

```text
Area

Rooms
```

Both explain house price almost equally well.

Keeping both increases

```text
|w₁| + |w₂|
```

without significantly improving prediction.

The optimizer therefore tends to

```text
Keep One

↓

Remove the Other
```

This is why L1 often performs automatic feature selection.

> **Note:** L1 does not explicitly detect correlated features. It naturally prefers sparse solutions when multiple features provide similar predictive power.

---

# 8️⃣ How Does L1 Affect Gradient Descent?

The optimizer minimizes

```text
Total Loss

=

Prediction Loss

+

λ × Regularization Loss
```

Weight update becomes

```text
Gradient

=

Prediction Gradient

+

Regularization Gradient
```

Think of it as two competing forces.

```text
Prediction Gradient

↓

Improve Accuracy


Regularization Gradient

↓

Reduce Weight Magnitude
```

Every update is a balance between these objectives.

---

# 9️⃣ Engineering Insight

The optimizer **cannot distinguish**

✔ Genuine patterns

from

✔ Accidental correlations.

It only knows one thing:

> "Does this reduce the Total Loss?"

Regularization prevents the optimizer from assigning unnecessarily large weights to patterns that only appear useful in the training data.

---

# 🔟 Key Takeaways

✅ Generalization is more important than training accuracy.

✅ Overfitting means memorizing the training data.

✅ L1 adds a penalty proportional to the absolute value of weights.

✅ λ controls the balance between accuracy and simplicity.

✅ L1 encourages sparse models.

✅ Sparse models naturally perform feature selection.

✅ Important weights survive because they reduce Prediction Loss.

✅ Unimportant weights gradually become zero.

---

# 🧠 Mental Model

Think of two managers reviewing every weight update.

```text
Manager 1

↓

"Improve prediction accuracy."


Manager 2

↓

"Keep the model simple."
```

The optimizer listens to both.

Useful features survive.

Redundant features disappear.

That balance is the essence of **L1 Regularization**.