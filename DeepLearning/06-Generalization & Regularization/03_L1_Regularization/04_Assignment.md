# 📝 Assignment — L1 Regularization

---

# Assignment 1 — Scenario Based

---

## Scenario A — Memorizing Noise

A bank is building a loan approval model.

The training accuracy reaches **99%**, but the validation accuracy is only **80%**.

One engineer says:

> "The optimizer has done a great job. Let's deploy it."

Do you agree?

### Answer

No.

This is a classic sign of **overfitting**.

The model has memorized the training dataset instead of learning generalizable patterns.

The large gap between training and validation accuracy indicates **high variance**.

Introducing **L1 Regularization** can help by shrinking unnecessary weights and removing features that contribute little to prediction.

---

## Scenario B — Correlated Features

A salary prediction model contains the following features.

```text
Years of Experience

Months of Experience
```

Both features carry almost identical information.

How will L1 Regularization likely behave?

### Answer

L1 will usually keep one feature while shrinking the other toward zero.

Since both features contribute nearly the same information, keeping both unnecessarily increases the regularization penalty.

The optimizer therefore prefers a sparse solution with fewer active features.

---

## Scenario C — Choosing λ

You train the same model three times.

```text
λ = 0

Training Accuracy = 99%

Validation Accuracy = 82%
```

```text
λ = 0.1

Training Accuracy = 96%

Validation Accuracy = 94%
```

```text
λ = 50

Training Accuracy = 78%

Validation Accuracy = 76%
```

Which model would you deploy?

### Answer

The second model.

Reason:

- λ = 0 → overfitting.
- λ = 50 → underfitting.
- λ = 0.1 achieves the best balance between prediction accuracy and generalization.

---

## Scenario D — Weight Updates

A teammate says:

> "Regularization only changes the loss function. It doesn't affect Gradient Descent."

Is the statement correct?

### Answer

No.

Regularization changes the **Total Loss**, so its gradient is also included during optimization.

Gradient Descent therefore updates weights using both

- Prediction Gradient
- Regularization Gradient

Regularization directly influences every weight update.

---

# Assignment 2 — Concept Building

---

## Q1.

Why isn't minimizing Prediction Loss alone sufficient for building a good model?

### Answer

Because Prediction Loss only measures how well the model fits the training data.

The optimizer cannot distinguish genuine patterns from accidental correlations.

Without regularization, it may assign large weights to noisy or unimportant features if doing so reduces the training loss.

---

## Q2.

If L1 penalizes every weight, why don't important features disappear?

### Answer

The optimizer minimizes

```text
Prediction Loss

+

λ × Regularization Loss
```

Prediction Loss tries to preserve useful features.

Regularization tries to shrink every weight.

Useful features remain because removing them would significantly increase Prediction Loss.

Unimportant features gradually shrink toward zero.

---

## Q3.

Why is L1 called a feature selection technique?

### Answer

Because L1 encourages sparse solutions.

Features that contribute very little to prediction gradually receive zero weights.

Once a feature's weight becomes zero, it no longer contributes to the model.

---

## Q4.

Why does L1 often remove one of two correlated features?

### Answer

Correlated features provide similar information.

Keeping both increases the L1 penalty without significantly improving prediction accuracy.

The optimizer therefore tends to keep one feature and remove the other to minimize the Total Loss.

---

# Assignment 3 — Debug the Model

A housing price model uses the following weights after training.

```text
Area            = 4.2

Location        = 3.8

Bedrooms        = 2.1

Parking         = 0.6

Wall Color      = 0.02

Owner Name      = 0.01
```

After applying L1 Regularization for several epochs, the weights become

```text
Area            = 3.9

Location        = 3.6

Bedrooms        = 1.9

Parking         = 0.3

Wall Color      = 0

Owner Name      = 0
```

What can you conclude?

### Answer

L1 has identified that **Wall Color** and **Owner Name** contribute very little to prediction.

The optimizer gradually shrank their weights to zero while preserving important features.

This demonstrates automatic feature selection.

---

# Assignment 4 — True or False

---

- L1 Regularization minimizes only the Regularization Loss. **False**

- L1 encourages sparse models. **True**

- Regularization affects Gradient Descent because it changes the Total Loss. **True**

- L1 explicitly checks whether two features are correlated. **False**

- Lambda controls the balance between prediction accuracy and model simplicity. **True**

---

# ⭐ Staff Engineer Challenge

Your team is training a recommendation system.

The dataset contains **250 engineered features**.

After applying L1 Regularization,

only **70 features** remain active.

A teammate argues:

> "We've lost information. Let's remove L1 and keep all 250 features."

Would you agree?

### Expected Answer

Not immediately.

Removing features is not necessarily harmful if those features contribute little to prediction.

L1 removes features only when the reduction in Prediction Loss is too small to justify their complexity.

Before making a decision, I would evaluate:

- Validation performance
- Cross-validation stability
- Feature importance
- Business impact
- Model latency
- Inference cost
- Explainability requirements

If validation accuracy remains unchanged while the model becomes simpler and faster, retaining the L1-regularized model is generally the better engineering choice.