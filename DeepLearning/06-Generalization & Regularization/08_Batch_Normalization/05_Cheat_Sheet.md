# 📘 Batch Normalization (L4) — Cheat Sheet

---

# 🎯 Why was Batch Normalization introduced?

Deep neural networks become difficult to train because the distribution of activations keeps changing during training.

This leads to:

- 🐢 Slow convergence
- ⚠️ Difficult optimization
- 🎯 Sensitivity to initialization
- 📉 Sensitivity to learning rate
- 🔄 Unstable training

Batch Normalization stabilizes the activations flowing through the network, making optimization much easier.

---

# 🎯 Internal Covariate Shift

**Definition**

The distribution of inputs to a layer keeps changing because previous layers continuously update their weights.

```text
Weights Updated
       ↓
Activations Change
       ↓
Input Distribution Changes
       ↓
Next Layer Must Relearn
```

BatchNorm reduces this problem by normalizing activations for every mini-batch.

---

# 🎯 Batch Normalization Pipeline

```text
Input Activations
        ↓
Compute Batch Mean (μ)
        ↓
Compute Batch Variance (σ²)
        ↓
Normalize
        ↓
Scale (γ)
        ↓
Shift (β)
        ↓
Output
```

---

# 🎯 BatchNorm Equations

### Step 1 — Mean

```text
μ = (1/m) Σx
```

---

### Step 2 — Variance

```text
σ² = (1/m) Σ(x − μ)²
```

---

### Step 3 — Normalize

```text
x̂ = (x − μ)
/ √(σ² + ε)
```

---

### Step 4 — Scale & Shift

```text
y = γx̂ + β
```

---

# 🎯 Why is ε (Epsilon) added?

Without ε:

```text
Variance = 0

↓

Division by Zero
```

Adding a tiny value prevents numerical instability.

```text
ε ≈ 0.00001
```

---

# 🎯 Role of γ (Gamma)

🎚️ Controls the **scale** of normalized activations.

- Larger γ → Larger variance
- Smaller γ → Smaller variance

---

# 🎯 Role of β (Beta)

📍 Controls the **shift** of normalized activations.

- Positive β → Shift right
- Negative β → Shift left

---

# 🎯 Why are γ and β needed?

Normalization forces:

```text
Mean = 0

Variance = 1
```

But every layer may not work best with this distribution.

Gamma and Beta restore the network's flexibility by learning the most suitable activation distribution.

---

# 🎯 Training vs Inference

| Training 🏋️ | Inference 🚀 |
|--------------|--------------|
| Uses Batch Mean | Uses Running Mean |
| Uses Batch Variance | Uses Running Variance |
| Statistics change every mini-batch | Statistics remain fixed |
| Running statistics are updated | Running statistics are not updated |

---

# 🎯 Running Mean & Running Variance

During training:

```text
Mini Batch

↓

Compute Mean

↓

Compute Variance

↓

Update Running Statistics
```

During inference:

```text
Running Mean

↓

Running Variance

↓

Normalize
```

This guarantees deterministic predictions.

---

# 🎯 Advantages of BatchNorm

✅ Faster convergence

✅ Allows higher learning rates

✅ More stable optimization

✅ Less sensitive to initialization

✅ Easier training of deep networks

✅ Slight regularization effect

---

# 🎯 Limitations

⚠️ Depends on batch size

⚠️ Adds extra computation

⚠️ Small batches produce noisy statistics

⚠️ Not suitable for every architecture

---

# 🎯 BatchNorm vs Dropout

| Feature | 🟦 BatchNorm | 🟨 Dropout |
|----------|-------------|------------|
| Primary Goal | Stabilize training | Reduce overfitting |
| Main Idea | Normalize activations | Randomly drop neurons |
| Speeds up training | ✅ | ❌ |
| Reduces overfitting | Slightly | Strongly |
| Uses batch statistics | ✅ | ❌ |
| Removes neurons | ❌ | ✅ |

---

# 🎯 Important Interview Points

⭐ Internal Covariate Shift = Changing input distribution during training.

⭐ BatchNorm normalizes **activations**, not the original dataset.

⭐ Every mini-batch has its own Mean and Variance.

⭐ γ (Gamma) learns the best **scale**.

⭐ β (Beta) learns the best **shift**.

⭐ ε prevents division by zero.

⭐ During inference, BatchNorm uses **Running Mean** and **Running Variance**, not batch statistics.

⭐ BatchNorm improves optimization but is **not** primarily an overfitting technique.

⭐ BatchNorm allows higher learning rates because activations remain well-scaled.

⭐ BatchNorm is **not guaranteed** to improve every model—its effectiveness depends on the architecture, batch size, and training setup.

---

# 🚀 One-Line Summary

> **Batch Normalization stabilizes the distribution of activations flowing through a neural network by normalizing each mini-batch and then learning the optimal scale (γ) and shift (β), resulting in faster and more stable training.**