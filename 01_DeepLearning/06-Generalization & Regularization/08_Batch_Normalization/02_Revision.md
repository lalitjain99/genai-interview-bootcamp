# 📝 Batch Normalization (L4) — Revision Notes

---

# 🎯 Why was Batch Normalization introduced?

Neural networks become difficult to train as they get deeper because the distribution of activations keeps changing during training.

This makes optimization unstable and slow.

Batch Normalization stabilizes these activations by normalizing them before passing them to the next layer.

---

# 🚨 Problems solved by Batch Normalization

- ⚡ Faster convergence
- 📉 More stable gradients
- 🚀 Allows higher learning rates
- 🛡️ Reduces overfitting (slight regularization effect)
- 🏗️ Enables training of deeper neural networks

---

# 🔄 Batch Normalization Workflow

For each mini-batch:

```text
Input Activations
        │
        ▼
Compute Batch Mean (μ)
        │
        ▼
Compute Batch Variance (σ²)
        │
        ▼
Normalize
        │
        ▼
Scale (γ)
        │
        ▼
Shift (β)
        │
        ▼
Output
```

---

# 📌 Batch Normalization Equations

### Step 1️⃣ Compute Mean

```text
μ = (1/m) Σx
```

---

### Step 2️⃣ Compute Variance

```text
σ² = (1/m) Σ(x − μ)²
```

---

### Step 3️⃣ Normalize

```text
x̂ = (x − μ) / √(σ² + ε)
```

---

### Step 4️⃣ Scale & Shift

```text
y = γx̂ + β
```

---

# 🎯 Role of ε (Epsilon)

Small constant added to variance.

Purpose:

- Prevent division by zero
- Improve numerical stability

Typical value:

```text
ε = 1e−5
```

---

# 🎯 Role of γ (Gamma)

γ is a **learnable parameter**.

Purpose:

- Controls the spread (scale) of activations
- Allows the network to decide how much variance it wants

Think of γ as:

> 🔍 **Zoom In / Zoom Out**

Small γ

```text
Compress Activations
```

Large γ

```text
Stretch Activations
```

---

# 🎯 Role of β (Beta)

β is also **learnable**.

Purpose:

- Shifts the normalized activations
- Allows the network to choose the desired mean

Think of β as:

> 📍 **Move Left / Move Right**

Positive β

```text
Shift Right
```

Negative β

```text
Shift Left
```

---

# 💡 Why do we need γ and β?

Without them,

every layer would always receive:

```text
Mean = 0

Variance = 1
```

This may remove useful information.

Instead,

BatchNorm first stabilizes training through normalization,

then allows the network to learn the best scale and shift using γ and β.

---

# 🧠 Intuition

BatchNorm says:

> "Normalize first for easier optimization."

Then γ and β say:

> "Now I'll learn the most useful distribution."

---

# 🎯 Training vs Inference

## 🏋️ Training

Uses statistics of the current mini-batch.

```text
Current Batch

↓

Mean

↓

Variance

↓

Normalize
```

---

## 🚀 Inference

Uses stored running averages collected during training.

```text
Running Mean

+

Running Variance
```

No batch statistics are computed.

---

# 📊 Running Statistics

During training,

BatchNorm keeps updating:

- Running Mean
- Running Variance

These are later used during inference for consistent predictions.

---

# ⚖️ BatchNorm vs Dropout

| Feature | BatchNorm | Dropout |
|----------|-----------|----------|
| 🎯 Purpose | Stabilize training | Reduce overfitting |
| ⚙️ Works by | Normalizing activations | Randomly dropping neurons |
| 🚀 Speeds up training | ✅ Yes | ❌ No |
| 🛡️ Regularization | Slight | Strong |
| 🧩 Trainable parameters | γ, β | None |

---

# ⚠️ Common Interview Point

BatchNorm **does not permanently force** activations to have:

```text
Mean = 0

Variance = 1
```

It first normalizes,

then γ and β learn the most suitable distribution.

---

# 🎯 Key Takeaways

✅ Solves unstable activation distributions

✅ Speeds up convergence

✅ Enables higher learning rates

✅ Makes deeper networks easier to train

✅ Uses mini-batch statistics during training

✅ Uses running statistics during inference

✅ γ controls scaling

✅ β controls shifting

✅ ε prevents division by zero

✅ BatchNorm improves optimization while preserving model flexibility

---

# 🧠 One-Line Memory Trick

> **BatchNorm = Normalize → Scale (γ) → Shift (β). Normalize for stability, then let the network learn the best distribution.**