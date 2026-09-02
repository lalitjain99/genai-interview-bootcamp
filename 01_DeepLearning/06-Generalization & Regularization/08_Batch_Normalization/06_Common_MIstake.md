# ⚠️ Batch Normalization (L4) — Common Mistakes

---

# ❌ Mistake 1: BatchNorm normalizes the original dataset

### Wrong ❌

> BatchNorm normalizes the input dataset before training.

### Correct ✅

BatchNorm normalizes the **activations produced by each layer** during every forward pass, **not** the original dataset.

---

# ❌ Mistake 2: BatchNorm permanently forces Mean = 0 and Variance = 1

### Wrong ❌

> Every layer always has Mean = 0 and Variance = 1.

### Correct ✅

BatchNorm first normalizes activations, then uses **γ (Gamma)** and **β (Beta)** to learn the most suitable scale and shift.

```text
Normalize
      ↓
Scale (γ)
      ↓
Shift (β)
```

The final activations are **not necessarily** Mean = 0 and Variance = 1.

---

# ❌ Mistake 3: Gamma and Beta are fixed values

### Wrong ❌

> Gamma and Beta are predefined constants.

### Correct ✅

Gamma (γ) and Beta (β) are **learnable parameters** updated during backpropagation just like other model parameters.

---

# ❌ Mistake 4: Gamma changes the mean

### Wrong ❌

> Gamma shifts the activations.

### Correct ✅

Gamma controls the **scale (spread/variance)** of the activations.

Beta controls the **shift (mean)** of the activations.

```text
γ → Scale

β → Shift
```

---

# ❌ Mistake 5: BatchNorm uses Batch Mean during inference

### Wrong ❌

> During prediction, BatchNorm computes Mean and Variance from the current input batch.

### Correct ✅

During inference, BatchNorm uses the **Running Mean** and **Running Variance** collected during training.

This ensures deterministic predictions.

---

# ❌ Mistake 6: Running Mean is computed only after training

### Wrong ❌

> Running statistics are calculated after the model finishes training.

### Correct ✅

Running Mean and Running Variance are updated continuously during training using every mini-batch.

---

# ❌ Mistake 7: Epsilon improves model accuracy

### Wrong ❌

> Epsilon is added to improve learning.

### Correct ✅

Epsilon is added only for **numerical stability** to prevent division by zero (or very small values).

```text
Variance = 0

↓

Division by Zero

↓

Add ε
```

---

# ❌ Mistake 8: BatchNorm eliminates Internal Covariate Shift completely

### Wrong ❌

> BatchNorm completely removes Internal Covariate Shift.

### Correct ✅

BatchNorm **reduces** fluctuations in activation distributions, making optimization more stable.

It does **not** completely eliminate Internal Covariate Shift.

---

# ❌ Mistake 9: BatchNorm always improves every model

### Wrong ❌

> Every neural network should use BatchNorm.

### Correct ✅

BatchNorm is very effective in many deep networks, but its usefulness depends on:

- Batch size
- Model architecture
- Training setup

It is **not** universally beneficial.

---

# ❌ Mistake 10: BatchNorm works equally well for every batch size

### Wrong ❌

> Batch size doesn't matter.

### Correct ✅

BatchNorm relies on accurate batch statistics.

Very small batches produce noisy Mean and Variance estimates, reducing BatchNorm's effectiveness.

---

# ❌ Mistake 11: BatchNorm and Dropout solve the same problem

### Wrong ❌

> They are interchangeable.

### Correct ✅

They solve different problems.

| BatchNorm | Dropout |
|-----------|----------|
| Stabilizes optimization | Reduces overfitting |
| Normalizes activations | Randomly drops neurons |
| Speeds up training | Improves generalization |

---

# ❌ Mistake 12: BatchNorm removes the need for tuning

### Wrong ❌

> Once BatchNorm is added, learning rate and other hyperparameters no longer matter.

### Correct ✅

BatchNorm often makes training **more robust** and allows larger learning rates, but proper hyperparameter tuning is still essential.

---

# 🎯 Quick Revision

✅ BatchNorm normalizes **activations**, not the dataset.

✅ Normalization happens for **every mini-batch** during training.

✅ γ controls **scale**.

✅ β controls **shift**.

✅ ε prevents **division by zero**.

✅ Running Mean and Running Variance are collected during training.

✅ Inference always uses **Running Statistics**, not batch statistics.

✅ BatchNorm mainly improves **optimization**, while Dropout mainly improves **generalization**.

✅ BatchNorm is powerful but **not universally beneficial**—its effectiveness depends on the architecture and batch size.