# ⚡ Adam Optimizer — Cheat Sheet

> **2-Minute Revision Guide**

---

# 🚀 One-Line Summary

> **Adam combines Momentum and RMSProp with Bias Correction to achieve both better optimization direction and adaptive learning rates.**

---

# 🌍 Evolution of Optimizers

```text
Gradient Descent
        │
        ▼
Momentum
(Remembers Direction)
        │
        ▼
AdaGrad
(Adaptive Learning Rate)
        │
        ▼
RMSProp
(Recent Gradients Matter More)
        │
        ▼
Adam
(Direction + Adaptive LR + Bias Correction)
        │
        ▼
AdamW
(Fixes Weight Decay)
```

---

# 🧠 Core Intuition

An ideal optimizer must answer **two questions**:

### 🎯 1. Which direction should I move?

Solved by **Momentum**

```text
Average Gradient
```

---

### 📏 2. How large should my step be?

Solved by **RMSProp**

```text
Average Squared Gradient
```

Adam combines both.

---

# 📚 Two Memories

| Memory | Stores | Purpose |
|---------|---------|----------|
| 🟢 First Moment (m) | Average Gradient | Better Direction |
| 🔵 Second Moment (v) | Average Squared Gradient | Adaptive Learning Rate |

---

# 🧮 Adam Equations

### Step 1️⃣ Momentum

```text
m = β₁m + (1 − β₁)g
```

---

### Step 2️⃣ RMSProp

```text
v = β₂v + (1 − β₂)g²
```

---

### Step 3️⃣ Bias Correction

```text
m̂ = m / (1 − β₁ᵗ)

v̂ = v / (1 − β₂ᵗ)
```

---

### Step 4️⃣ Weight Update

```text
W = W − η × (m̂ / (√v̂ + ε))
```

---

# 🎯 What Does Each Symbol Mean?

| Symbol | Meaning |
|---------|----------|
| **g** | Current Gradient |
| **m** | First Moment (Momentum Memory) |
| **v** | Second Moment (RMSProp Memory) |
| **β₁** | Momentum Decay (usually 0.9) |
| **β₂** | RMSProp Decay (usually 0.999) |
| **η** | Learning Rate |
| **ε** | Small constant for numerical stability |

---

# 💡 Why Bias Correction?

Initially,

```text
m = 0

v = 0
```

Both moving averages are biased toward zero.

Bias correction removes this initialization bias.

Without it,

- ❌ Early updates are poorly scaled
- ❌ Initial learning becomes unreliable

After many iterations,

```text
βᵗ → 0
```

Bias correction becomes negligible.

---

# ⚖️ Optimizer Comparison

| Optimizer | Direction Memory | Adaptive LR |
|-----------|------------------|-------------|
| Gradient Descent | ❌ | ❌ |
| Momentum | ✅ | ❌ |
| AdaGrad | ❌ | ✅ |
| RMSProp | ❌ | ✅ |
| Adam | ✅ | ✅ |

---

# 🎯 Advantages

✅ Fast convergence

✅ Stable optimization

✅ Adaptive learning rate

✅ Handles sparse gradients

✅ Works well for deep neural networks

✅ Default choice for many NLP and Transformer models

---

# ⚠️ Limitations

❌ More memory than SGD

❌ Requires tuning of learning rate

❌ Can generalize worse than SGD + Momentum

❌ Weight decay is coupled with gradient updates

➡️ Solved by **AdamW**

---

# 🧩 Common Interview Questions

### ❓ Why wasn't Momentum enough?

➡️ Same learning rate for every parameter.

---

### ❓ Why wasn't RMSProp enough?

➡️ No directional memory.

---

### ❓ Why does Adam use two memories?

➡️ One decides **direction**.

➡️ One decides **step size**.

---

### ❓ Why is Bias Correction needed?

➡️ EMA starts from zero, producing biased estimates during early iterations.

---

### ❓ Does Adam automatically choose the learning rate?

❌ No.

It adapts the **effective learning rate**, but the base learning rate **η** must still be chosen.

---

### ❓ Why divide by √v instead of v?

➡️ Because **v stores squared gradients**.

Taking the square root restores the gradient's original scale.

---

# 🚗 Remember This Analogy

Imagine driving a car.

🧭 **Momentum**

```text
GPS
```

Tells you

> Which direction to go.

---

🚗 **RMSProp**

```text
Cruise Control
```

Decides

> How fast to drive.

---

🧠 **Adam**

```text
GPS
      +
Cruise Control
```

Direction **+** Speed

= Better driving

= Better optimization

---

# 🏆 Interview Takeaway

> **Momentum tells the optimizer where to move.**

> **RMSProp tells the optimizer how aggressively to move.**

> **Adam combines both and corrects the initialization bias, making it one of the most widely used optimizers in modern Deep Learning.**

---

# 🔜 Next Lecture

➡️ **AdamW Optimizer**

> Why did researchers invent AdamW if Adam was already so successful?