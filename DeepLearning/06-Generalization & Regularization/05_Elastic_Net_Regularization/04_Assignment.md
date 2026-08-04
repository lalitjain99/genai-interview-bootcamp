# 📝 Assignment — Elastic Net Regularization

---

# 🟧 Assignment 1 — Scenario Based

---

## 📌 Scenario A

A fraud detection model uses **L1 Regularization**.

The dataset contains several highly correlated features:

- Transaction Amount
- Daily Spending
- Monthly Spending

After training, only **Transaction Amount** remains while the other two features receive zero weights.

### Questions

1. Why did this happen?
2. Why might this be undesirable?
3. Which regularization technique would you recommend instead?
4. Explain your reasoning.

---

### ✅ Solution

### 1️⃣ Why did this happen?

L1 Regularization performs feature selection by driving some weights exactly to zero.

When multiple features are highly correlated, they carry similar information. Instead of distributing weights across all of them, L1 typically keeps one feature and pushes the others to zero. The choice of which feature survives is often arbitrary and may change with small variations in the training data.

---

### 2️⃣ Why might this be undesirable?

This can create several problems:

- Model becomes unstable across retraining.
- Useful information from the removed correlated features may be lost.
- Harder to justify feature selection during audits.
- Loss of redundancy makes the model more fragile if the surviving feature has poor data quality.

---

### 3️⃣ Which regularization technique would you recommend?

✅ **Elastic Net Regularization**

---

### 4️⃣ Explain your reasoning.

Elastic Net combines:

- L1's feature selection
- L2's grouping effect

Instead of arbitrarily removing correlated features, Elastic Net encourages them to share the weights while still eliminating truly irrelevant features.

This produces a model that is:

- More stable
- More interpretable
- Better at handling correlated features
- Less prone to overfitting

---

## 📌 Scenario B

A medical diagnosis model uses **L2 Regularization**.

The dataset contains **10,000 features**.

After training, every feature still has a non-zero weight, including many features known to be irrelevant.

### Questions

1. Why did L2 fail to remove these features?
2. Which regularization technique would better address this issue?
3. Explain why.

---

### ✅ Solution

### 1️⃣ Why did L2 fail to remove these features?

L2 Regularization shrinks weights toward zero but almost never makes them exactly zero.

As a result, even irrelevant features remain in the model with very small weights.

---

### 2️⃣ Which regularization technique would better address this?

✅ **L1 Regularization**

or

✅ **Elastic Net** (if correlated features also exist).

---

### 3️⃣ Explain why.

L1 can drive irrelevant feature weights exactly to zero, automatically performing feature selection.

This results in:

- Simpler models
- Better interpretability
- Reduced noise
- Easier regulatory explanation

If correlated features are also present, Elastic Net is usually preferred because it combines feature selection with stable weight sharing.

---

## 📌 Scenario C

A recommendation system contains:

- Thousands of noisy features
- Many highly correlated user behavior features

The team must choose between:

- L1
- L2
- Elastic Net

### Questions

1. Which regularization technique would you recommend?
2. Explain why the other two are less suitable.

---

### ✅ Solution

### 1️⃣ Recommendation

✅ **Elastic Net**

---

### 2️⃣ Why?

The dataset has two problems simultaneously:

- Noisy features
- Correlated features

L1 removes noisy features but may arbitrarily discard useful correlated features.

L2 preserves correlated features but cannot eliminate noisy ones.

Elastic Net combines both strengths by:

- Removing irrelevant features (L1)
- Preserving correlated feature groups (L2)

making it the most suitable choice.

---

## 📌 Scenario D

During hyperparameter tuning, the team gradually increases λ.

They observe:

- Training loss increases.
- Validation loss initially decreases.
- Later, both training and validation loss increase.

### Questions

1. Explain why this happens.
2. What does this indicate about the chosen λ?
3. How would you select a better value?

---

### ✅ Solution

### 1️⃣ Why does this happen?

As λ increases, regularization becomes stronger.

Initially:

- Overfitting decreases.
- Validation performance improves.

However, after a certain point:

- The model becomes too simple.
- Important weights shrink excessively.
- Underfitting begins.

As a result:

- Training loss increases.
- Validation loss also increases.

This is the classic **bias-variance tradeoff**.

---

### 2️⃣ What does this indicate?

The current λ is **too large**.

The optimal λ lies near the minimum point of the validation loss curve.

---

### 3️⃣ How would you select a better λ?

- Perform Cross Validation.
- Search λ on a logarithmic scale.
- Choose the λ with the lowest validation loss.
- Validate the final model on an unseen test set.

---

# 🟨 Assignment 2 — Conceptual Questions

---

## Q1. Explain how Elastic Net combines the strengths of L1 and L2 Regularization.

### ✅ Solution

Elastic Net combines both regularization techniques in a single loss function.

```text
Loss
=
Prediction Loss
+
λ [ αΣ|w| + (1−α)Σw² ]
```

The **L1 component** performs feature selection by removing irrelevant features.

The **L2 component** shrinks weights and preserves correlated features.

As a result, Elastic Net produces models that are both sparse and stable.

---

## Q2. Explain the roles of λ and α.

### ✅ Solution

### λ (Lambda)

Controls the overall strength of regularization.

- Small λ → weak regularization
- Large λ → strong regularization

---

### α (Alpha)

Controls the balance between L1 and L2.

```text
α = 1
→ Pure L1

α = 0
→ Pure L2

0 < α < 1
→ Elastic Net
```

Together,

- λ decides **how much** regularization is applied.
- α decides **what type** of regularization is applied.

---

## Q3. Why is Elastic Net generally preferred over pure L1 for correlated features?

### ✅ Solution

L1 tends to keep one correlated feature while removing the others.

This makes feature selection unstable and may discard useful information.

Elastic Net's L2 component encourages correlated features to share weights instead of eliminating them.

This produces:

- More stable models
- Better generalization
- Better utilization of correlated information

---

# 🟩 Assignment 3 — True or False

---

### 1️⃣ Elastic Net always performs better than L1 and L2.

❌ **False**

Performance depends on the dataset and problem.

---

### 2️⃣ Setting α = 1 produces pure L1 Regularization.

✅ **True**

---

### 3️⃣ Setting α = 0 produces pure L2 Regularization.

✅ **True**

---

### 4️⃣ Increasing λ always improves model performance.

❌ **False**

A very large λ causes underfitting.

---

### 5️⃣ Elastic Net can both remove irrelevant features and shrink important weights.

✅ **True**

---

# ⭐ Assignment 4 — Staff Engineer Challenge

A company is building a **customer churn prediction model**.

The dataset contains:

- 25,000 engineered features
- Many correlated customer behavior metrics
- Moderate overfitting

The team proposes using Elastic Net.

---

## Questions

1. Do you agree with this decision?
2. What benefits does Elastic Net provide?
3. What additional experiments would you perform?
4. Besides regularization, what other techniques would improve generalization?

---

## ✅ Solution

### 1️⃣ Do you agree?

Yes.

Elastic Net is a strong choice because the dataset contains both correlated and noisy features.

However, I would still validate this decision by comparing it against pure L1, pure L2, and strong non-linear baselines before deployment.

---

### 2️⃣ Benefits

- Handles correlated features effectively.
- Removes irrelevant engineered features.
- Reduces overfitting.
- Produces a more interpretable model.
- Improves stability across retraining.

---

### 3️⃣ Additional experiments

- Compare with L1 and L2.
- Perform Grid Search or Randomized Search over λ and α.
- Use Cross Validation.
- Evaluate on temporal holdout data.
- Compare against XGBoost or LightGBM.
- Validate feature stability across multiple runs.

---

### 4️⃣ Other techniques

Besides regularization, I would also consider:

- Feature selection before training.
- Dimensionality reduction (PCA).
- Tree-based ensemble models.
- Handling class imbalance.
- Probability calibration.
- Explainability using SHAP.
- Data leakage audits.
- Production drift monitoring.