# 📝 Batch Normalization (L4) — Assignment

---

# 🎯 Assignment 1 — Scenario Based

---

# 📌 Scenario A — Slow Training

A team is training a **20-layer neural network**.

They observe:

- Training is very slow.
- The model is highly sensitive to learning rate.
- Different weight initializations produce very different results.

---

## ✅ Solution

### **1️⃣ Why is training unstable?**

As the network learns, every layer continuously updates its weights.

These weight updates change the activations produced by that layer.

As a result, deeper layers keep receiving inputs whose distribution is constantly changing.

They must continuously adapt to this moving target, making optimization slower and more sensitive to initialization and learning rate.

---

### **2️⃣ What is Internal Covariate Shift?**

Internal Covariate Shift is the phenomenon where the **distribution of inputs to a layer changes during training** because the previous layers continuously update their weights.

---

### **3️⃣ How does Batch Normalization help?**

Batch Normalization normalizes the activations produced by the previous layer for every mini-batch.

This provides a more stable input distribution to the next layer, allowing faster and more stable optimization.

---

### **4️⃣ Explain the Batch Normalization pipeline.**

```text
Input Activations
        ↓
Compute Batch Mean (μ)
        ↓
Compute Batch Variance (σ²)
        ↓
Normalize Activations
        ↓
Scale using γ
        ↓
Shift using β
        ↓
Output
```

---

### **5️⃣ Why does BatchNorm allow higher learning rates?**

Normalization keeps activations and gradients in a well-scaled range.

This reduces unstable updates, allowing the optimizer to safely take larger gradient steps without causing divergence.

---

# 📌 Scenario B — Inference Problem

A model was trained using Batch Normalization.

During deployment, an engineer accidentally computes **Batch Mean** and **Batch Variance** for every incoming inference request instead of using Running Mean and Running Variance.

---

## ✅ Solution

### **1️⃣ What mistake has been made?**

The engineer is using the current batch statistics during inference instead of the stored Running Mean and Running Variance collected during training.

---

### **2️⃣ Why should inference use Running Mean and Running Variance?**

During inference, requests often contain only one sample or a very small batch.

Such batches cannot provide reliable statistics.

Running statistics collected during training better represent the overall data distribution.

---

### **3️⃣ What problems can occur if batch statistics are used during inference?**

- Same input may produce different outputs.
- Predictions become inconsistent.
- Small batches generate noisy statistics.
- Model becomes unreliable in production.

---

### **4️⃣ Why must predictions remain deterministic?**

Production systems require reproducible predictions.

The same input should always produce the same output regardless of when it is evaluated.

---

# 📌 Scenario C — Removing Gamma & Beta

A junior engineer says:

> "BatchNorm already normalizes activations to Mean = 0 and Variance = 1. We don't need Gamma and Beta."

---

## ✅ Solution

### **1️⃣ Do you agree?**

❌ No.

---

### **2️⃣ Why were Gamma and Beta introduced?**

Always forcing activations to have Mean = 0 and Variance = 1 may reduce the network's flexibility.

Gamma and Beta allow each layer to learn the activation distribution that is most useful for the task.

---

### **3️⃣ What does Gamma control?**

Gamma (γ) controls the **scale** of the normalized activations.

---

### **4️⃣ What does Beta control?**

Beta (β) shifts the normalized activations by learning a new mean.

---

### **5️⃣ Can BatchNorm learn to undo normalization?**

✅ Yes.

Since Gamma and Beta are learnable parameters, BatchNorm can learn almost any suitable scale and shift, effectively recovering the activation distribution that the network needs.

---

# 📌 Scenario D — Very Small Batch Size

A team trains a neural network using:

```text
Batch Size = 2
```

Training becomes unstable.

---

## ✅ Solution

### **1️⃣ Why can BatchNorm struggle with very small batch sizes?**

Batch Mean and Batch Variance are computed from only two samples.

These estimates become noisy and do not accurately represent the true data distribution.

---

### **2️⃣ How does batch size affect Mean and Variance estimation?**

Smaller batch sizes produce noisier estimates of Mean and Variance.

Larger batches produce more reliable estimates.

---

### **3️⃣ What happens when these statistics become noisy?**

Noisy normalization introduces instability during training.

Optimization becomes slower and Running Mean/Variance also become less reliable.

---

### **4️⃣ Would BatchNorm always be your first choice here?**

❌ No.

For extremely small batch sizes, BatchNorm becomes less effective because its statistics are unreliable.

Other normalization techniques designed for small batches are usually preferred.

---

# 🎯 Assignment 2 — True / False

| # | Statement | ✅ Answer |
|---|-----------|-----------|
| 1 | BatchNorm normalizes activations for every mini-batch. | ✅ True |
| 2 | Gamma and Beta are learnable parameters. | ✅ True |
| 3 | During inference, BatchNorm computes Mean and Variance from the current input batch. | ❌ False |
| 4 | BatchNorm allows higher learning rates. | ✅ True |
| 5 | BatchNorm permanently forces activations to Mean = 0 and Variance = 1. | ❌ False |
| 6 | Running Mean and Running Variance are collected during training. | ✅ True |
| 7 | Epsilon is added to prevent division by zero. | ✅ True |
| 8 | BatchNorm provides a slight regularization effect. | ✅ True |
| 9 | BatchNorm works equally well regardless of batch size. | ❌ False |
| 10 | BatchNorm always improves every deep learning model. | ❌ False |

---

# ⭐ Staff Engineer Challenge

Your team is building a **30-layer image classification model**.

A senior developer proposes:

> **"Let's add Batch Normalization after every layer because BatchNorm always improves performance."**

---

## ✅ Solution

### **1️⃣ Do you agree with this proposal?**

❌ No.

Batch Normalization is a powerful optimization technique, but it does **not always** improve performance.

Its effectiveness depends on the architecture, dataset, batch size, and training setup.

Engineering decisions should be validated through experiments rather than assumptions.

---

### **2️⃣ What trade-offs would you discuss with the team?**

- Additional computation per layer.
- Depends on batch size for reliable statistics.
- Uses Running Mean and Running Variance during inference.
- May provide little benefit if the model already trains stably.

---

### **3️⃣ Why shouldn't engineering decisions be based on the word "always"?**

Deep learning techniques are highly dependent on the problem being solved.

A technique that works well for one model or dataset may not work equally well for another.

Production decisions should always be supported by empirical evidence.

---

### **4️⃣ What experiments would you run before approving this design?**

- Compare training with and without BatchNorm.
- Measure convergence speed.
- Compare validation accuracy.
- Evaluate different batch sizes.
- Measure computational overhead.

---

### **5️⃣ Besides validation accuracy, what other factors would you evaluate before deployment?**

- 🚀 Training speed
- ⚡ Inference latency
- 💾 Memory usage
- 📈 Convergence stability
- 🔁 Prediction consistency during inference
- 💰 Overall computational cost

---

# ✅ End of Assignment