# Assignment — L2 Regularization (Ridge Regularization)

---

# 🎯 Assignment 1 — Scenario Based Questions

---

## Scenario A

Your team builds a House Price Prediction model.

All features are useful:

- Area
- Bedrooms
- Location
- Parking
- Age of House

A teammate suggests using **L1 Regularization** because "it automatically removes unnecessary features."

Do you agree?

If not, which regularization technique would you recommend and why?

---

### Answer

No.

Since every feature contributes useful information, removing any feature completely may result in loss of valuable information.

L1 drives some weights exactly to zero, effectively removing those features.

In this case, **L2 Regularization** is a better choice because it reduces the influence of less important features while keeping every feature in the model.

---

## Scenario B

During training you observe:

```
Training Accuracy = 99%

Validation Accuracy = 94%
```

Someone proposes increasing λ from **0.001** to **20** to eliminate overfitting.

Would you recommend this?

Why or why not?

---

### Answer

No.

A very large λ aggressively shrinks all weights, including important ones.

Although this may reduce overfitting, it can also cause underfitting by making the model too simple.

Instead, λ should be treated as a hyperparameter and selected using validation or cross-validation.

---

## Scenario C

Two features are highly correlated.

```
House Area

Built-up Area
```

How would L1 and L2 Regularization behave differently?

---

### Answer

**L1**

- Usually keeps one feature.
- Drives the other feature's weight to zero.
- Performs feature selection.

**L2**

- Keeps both features.
- Shrinks both weights together.
- Produces a more stable solution.

---

## Scenario D

Your colleague says:

> "L2 Regularization removes features from the model."

Is this statement correct?

Explain.

---

### Answer

No.

L2 does not remove features.

Instead, it shrinks their weights toward zero.

All features remain part of the model, resulting in a **dense model** rather than a sparse one.

---

# 🎯 Assignment 2 — Conceptual Questions

---

## Q1.

Why was L2 Regularization introduced even though L1 Regularization already existed?

---

### Answer

L1 performs feature selection by making some weights exactly zero.

However, in many real-world problems every feature contributes something useful.

L2 was introduced to reduce the influence of less important features without removing them completely.

---

## Q2.

Why does L2 square the weights instead of using absolute values?

---

### Answer

Squaring makes large weights much more expensive than small ones.

As a result, L2 penalizes excessively large weights much more aggressively while only mildly penalizing smaller weights.

---

## Q3.

What role does λ play in L2 Regularization?

---

### Answer

Lambda controls the balance between:

- Prediction accuracy
- Model simplicity

Small λ prioritizes prediction accuracy.

Large λ prioritizes smaller weights.

---

## Q4.

Why is L2 also called Weight Decay?

---

### Answer

During every optimization step, L2 gently pulls every weight toward zero.

The weights gradually become smaller over time instead of disappearing instantly.

Hence the name **Weight Decay**.

---

# 🎯 Assignment 3 — Analytical Thinking

---

A model uses L2 Regularization.

After training you observe:

```
Area        = 5.4
Bedrooms    = 2.1
Location    = 6.0
Parking     = 0.4
Age         = 1.3
```

None of the weights are zero.

Does this mean L2 failed?

Explain.

---

### Answer

No.

This is exactly how L2 is expected to behave.

Its objective is not to eliminate features but to reduce excessively large weights.

The model keeps every feature while reducing the influence of less important ones.

---

# 🎯 Assignment 4 — True or False

---

### 1. L2 Regularization produces sparse models.

**Answer:** False

---

### 2. L2 penalizes larger weights more heavily than smaller weights.

**Answer:** True

---

### 3. Setting λ = 0 removes the effect of regularization.

**Answer:** True

---

### 4. L2 usually keeps all features in the model.

**Answer:** True

---

### 5. Increasing λ indefinitely will always improve generalization.

**Answer:** False

---

# 💼 Staff Engineer Challenge

A recommendation system contains nearly **300 engineered features**.

After analysis, your team concludes that:

- Almost every feature contributes useful information.
- The model is overfitting.
- A junior engineer recommends using **L1 Regularization** because it will automatically simplify the model.

As the Staff ML Engineer:

1. Would you recommend L1 or L2?
2. Explain your reasoning.
3. What risks would you discuss with the team?
4. How would you determine the best value of λ?

---

### Answer

I would recommend **L2 Regularization**.

Since almost every feature contains useful information, removing features using L1 could discard valuable predictive signals.

L2 reduces overfitting by shrinking weights while preserving all features.

### Risks to discuss

- Choosing λ too small may not sufficiently reduce overfitting.
- Choosing λ too large may lead to underfitting by shrinking even important weights.
- The best λ depends on the dataset and should not be guessed.

### Choosing λ

I would treat λ as a hyperparameter and perform hyperparameter tuning using cross-validation.

The selected λ should provide the best validation performance while maintaining a good balance between prediction accuracy and model simplicity.