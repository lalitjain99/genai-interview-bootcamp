# 🚀 Dropout Regularization — Cheat Sheet

---

# 🎯 Purpose of Dropout

✅ Reduce **Overfitting**

✅ Improve **Generalization**

✅ Prevent **Co-adaptation** between neurons

---

# 🧠 Why was Dropout introduced?

Deep Neural Networks have millions of parameters.

They can easily **memorize** the training data instead of learning general patterns.

Dropout forces neurons to learn **independently**.

---

# ⚠️ Co-adaptation

**Definition**

Neurons become overly dependent on specific neighboring neurons.

Instead of learning useful features individually,

they only perform well when certain other neurons are present.

👉 Dropout breaks this dependency.

---

# ⚙️ How Dropout Works

During every training iteration:

🎲 Randomly generate a binary mask.

```text
Mask

1 0 1 1 0 1 0
```

- **1 → Keep neuron**
- **0 → Drop neuron**

Dropped neurons behave as if they do not exist.

A **new mask is generated for every mini-batch**.

---

# 🔄 Forward Propagation

✅ Generate Dropout Mask

✅ Drop selected neurons

✅ Remaining neurons continue computation

✅ Apply Inverted Dropout scaling

---

# 🔙 Backpropagation

Only active neurons receive gradients.

```text
Dropped neuron

Gradient = 0
```

Their weights are **not updated** in that iteration.

---

# 📈 Inverted Dropout

Suppose

```text
Dropout Rate = 0.5

Keep Probability = 0.5
```

Remaining neurons are scaled by

```text
1 / Keep Probability

= 1 / 0.5

= 2
```

This keeps the **expected activation** unchanged.

---

# 🏋️ Training vs 🔍 Inference

| Training | Inference |
|-----------|-----------|
| Random neurons dropped | No neurons dropped |
| New mask every mini-batch | Entire network used |
| Scaling applied | No scaling required |

---

# 🎚️ Choosing Dropout Rate

| Dropout Rate | Effect |
|---------------|--------|
| Too Low | Overfitting may continue |
| Moderate (0.2–0.5) | Usually good balance |
| Too High | Underfitting |

---

# 🧩 Dropout vs L1 vs L2

| Property | L1 | L2 | Dropout |
|----------|----|----|----------|
| Penalizes weights | ✅ | ✅ | ❌ |
| Feature selection | ✅ | ❌ | ❌ |
| Shrinks weights | ❌ | ✅ | Indirectly |
| Removes neurons | ❌ | ❌ | ✅ (temporarily) |
| Prevents Co-adaptation | ❌ | ❌ | ✅ |
| Mostly used in Deep Learning | ❌ | ❌ | ✅ |

---

# 💡 Why Dropout Works

Instead of training **one large network**,

Dropout effectively trains **many smaller subnetworks** that share the same weights.

During inference,

all neurons are used together.

This behaves similarly to an **ensemble** of many neural networks.

---

# 📝 Interview One-Liners

✅ Dropout is a **regularization technique** used to reduce overfitting.

✅ It randomly disables neurons during training.

✅ A **new dropout mask** is generated for every mini-batch.

✅ Dropped neurons receive **zero gradients**.

✅ Inverted Dropout scales surviving neurons during training.

✅ During inference, **no neurons are dropped**.

✅ Dropout reduces **co-adaptation** between neurons.

✅ Very high Dropout can cause **underfitting**.

✅ Moderate Dropout usually lies between **0.2 and 0.5**.

✅ Modern LLMs generally use **little or no Dropout during large-scale pretraining**.

---

# ⚡ Remember

> **L1 removes Features**

> **L2 shrinks Weights**

> **Dropout removes Neurons (temporarily)**