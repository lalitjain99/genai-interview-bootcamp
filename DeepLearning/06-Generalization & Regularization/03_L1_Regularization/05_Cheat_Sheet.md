# ⚡ Cheat Sheet — L1 Regularization

---

# 🎯 Core Idea

L1 Regularization prevents **overfitting** by discouraging unnecessarily large weights.

Instead of minimizing only Prediction Loss, it minimizes

```text
Total Loss

=

Prediction Loss

+

λ × L1 Regularization Loss
```

---

# 📌 L1 Regularization Formula

```text
L1 Regularization Loss

=

|w₁|

+

|w₂|

+

...

+

|wₙ|
```

```text
Total Loss

=

Prediction Loss

+

λ ×

(

|w₁|

+

|w₂|

+

...

+

|wₙ|

)
```

---

# 🎯 Goal

✔ Improve Generalization

✔ Reduce Overfitting

✔ Build Simpler Models

✔ Perform Automatic Feature Selection

---

# ⚖️ Lambda (λ)

Small λ

```text
↓

Less Regularization

↓

Better Training Fit

↓

Higher Overfitting Risk
```

Large λ

```text
↓

Stronger Regularization

↓

Smaller Weights

↓

Higher Underfitting Risk
```

---

# 🧠 Optimizer Perspective

The optimizer minimizes

```text
Prediction Loss

+

Regularization Loss
```

Two forces act on every weight.

```text
Prediction Loss

↓

Keep useful weights


Regularization

↓

Shrink weights
```

The final weight is a compromise between these two objectives.

---

# 🎯 Why Don't Important Weights Become Zero?

Because removing them would increase Prediction Loss significantly.

Prediction Loss pushes them back up.

Only features that contribute very little to prediction gradually become zero.

---

# 🎯 Why Does L1 Perform Feature Selection?

When multiple features contain similar information,

keeping all of them increases

```text
|w₁| + |w₂| + ...
```

without improving prediction much.

The optimizer therefore tends to

```text
Keep One

↓

Remove Others
```

Result:

✔ Sparse Model

✔ Automatic Feature Selection

---

# 📉 Effect on Gradient Descent

Weight update depends on

```text
Prediction Gradient

+

Regularization Gradient
```

Regularization directly influences every optimization step.

---

# 📊 Characteristics

| Property | L1 Regularization |
|----------|-------------------|
| Prevents Overfitting | ✅ |
| Improves Generalization | ✅ |
| Produces Sparse Models | ✅ |
| Performs Feature Selection | ✅ |
| Removes Redundant Features | Often |
| Can Cause Underfitting (Large λ) | ✅ |

---

# 🧩 Typical Use Cases

✅ High-dimensional datasets

✅ Many irrelevant features

✅ Need feature selection

✅ Better model interpretability

---

# ⚠️ Limitations

❌ Very large λ can cause underfitting.

❌ Feature selection among correlated features is not deterministic.

❌ Choosing λ requires experimentation (often using cross-validation).

---

# 💼 Interview One-Liners

### Why do we use L1?

> To improve generalization by penalizing large weights and encouraging sparse models.

---

### Why does L1 perform feature selection?

> Because unimportant or redundant features contribute little to Prediction Loss but increase Regularization Loss, so the optimizer naturally drives their weights to zero.

---

### Does L1 shrink every weight?

> It tries to, but Prediction Loss preserves important weights while unimportant ones gradually become zero.

---

### Does L1 explicitly detect correlated features?

> No. It simply minimizes Total Loss. Feature selection emerges naturally during optimization.

---

# 🧠 Mental Model

Think of two managers reviewing every weight.

```text
Manager 1

↓

Improve Prediction Accuracy


Manager 2

↓

Keep Model Simple
```

The optimizer balances both.

Useful features survive.

Redundant features disappear.

---

# 🚀 Remember

```text
Prediction Loss

↓

Fit the Data


L1 Regularization

↓

Keep the Model Simple


Together

↓

Better Generalization
```