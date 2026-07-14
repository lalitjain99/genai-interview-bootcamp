# 📝 Assignment.md — Adam Optimizer

---

# 🎯 Assignment 1 — Choosing the Right Optimizer

## Scenario A

**Problem**

A small image classification dataset where the primary goal is **stable training and good generalization**, not training speed.

### My Answer

I would choose **Batch Gradient Descent (BGD)**.

Reasoning:

- The dataset is small, so computing gradients over the full dataset is computationally feasible.
- Full-batch updates produce stable gradients with very little noise.
- Stable optimization generally improves convergence on small datasets.
- Since speed is not the priority, the higher computation per update is acceptable.

---

## Scenario B

**Problem**

A text classification dataset containing thousands of rare words.

### My Answer

I would choose **AdaGrad**.

Reasoning:

AdaGrad was specifically designed for sparse features.

It keeps track of the historical squared gradients for every parameter.

Parameters that rarely receive gradients maintain relatively large learning rates.

This prevents rare features from becoming inactive and allows them to continue learning throughout training.

---

## Scenario C

**Problem**

A production model receives new data every day while still needing to retain previous knowledge.

### My Answer

I would choose **Adam**.

Reasoning:

Adam maintains two memories.

1. First Moment (Momentum)

Tracks the average gradient.

```text
m = β₁m + (1 − β₁)g
```

This provides a stable optimization direction.

2. Second Moment (RMSProp)

Tracks the average squared gradient.

```text
v = β₂v + (1 − β₂)g²
```

This adapts the learning rate for every parameter.

Therefore Adam can both preserve historical optimization trends while adapting efficiently to new incoming data.

---

## Scenario D

**Problem**

A robot suddenly starts operating on a completely different surface.

### My Answer

I would choose **Adam** (RMSProp would also be a reasonable choice).

Reasoning:

The optimizer must adapt quickly to the new gradient distribution.

Adaptive learning rates allow different parameters to adjust independently.

Momentum provides stable movement while RMSProp adjusts the step size.

Combining both allows Adam to react quickly without producing unstable updates.

---

# 🎯 Assignment 2 — What Happens If We Remove One Memory?

## Case A

Remove the second moment (v).

### My Answer

Adam becomes almost identical to **Momentum**.

It still remembers previous directions,

but loses adaptive learning rates.

Every parameter now receives the same global learning rate.

---

## Case B

Remove the first moment (m).

### My Answer

Adam becomes essentially **RMSProp**.

It still adapts learning rates,

but no longer remembers historical movement.

Optimization becomes less stable because every update depends mainly on the current gradient.

---

# 🎯 Assignment 3 — Why Bias Correction Matters

Without bias correction,

both moving averages start from zero.

During the first few iterations,

their estimates become much smaller than the true statistics.

As a result,

the update rule becomes poorly scaled.

Early optimization steps become unreliable.

Bias correction removes this initialization bias.

```text
m̂ = m / (1 − β₁ᵗ)

v̂ = v / (1 − β₂ᵗ)
```

As training progresses,

βᵗ approaches zero,

making the correction factor approach one.

Eventually,

bias correction has almost no effect because the moving averages have accumulated sufficient history.

---

# 🎯 Assignment 4 — Why Can't We Simply Use Current Gradients?

Current gradients only describe the present optimization step.

They contain no information about

- previous movement
- historical gradient magnitude
- long-term optimization trends

An effective optimizer should answer two questions:

1. Which direction should I move?
2. How large should the step be?

Current gradients alone cannot answer both reliably.

---

# 🎯 Assignment 5 — True / False

| Statement | Answer |
|-----------|--------|
| Adam remembers previous movement. | ✅ True |
| Adam gives every parameter the same learning rate. | ❌ False |
| Adam combines Momentum and RMSProp. | ✅ True |
| Bias correction is important throughout the entire training process. | ❌ False |
| Adam completely replaces the need for learning rate tuning. | ❌ False |

---

# 🎯 Assignment 6 — Evolution of Optimizers

| Optimizer | Problem Solved | New Limitation Introduced |
|-----------|----------------|---------------------------|
| Gradient Descent | Enables learning using gradients | No memory, slow convergence, oscillations |
| Momentum | Remembers previous movement for better direction | Same learning rate for every parameter |
| AdaGrad | Adaptive learning rates for each parameter | Learning rate continually shrinks |
| RMSProp | Prevents learning rate from shrinking indefinitely | No directional memory |
| Adam | Combines Momentum + RMSProp | Weight decay handling (AdamW) and sensitivity during early iterations (handled by bias correction) |

---

# ⭐ Staff Engineering Challenge

| Team | Recommended Optimizer | Reason |
|------|-----------------------|--------|
| Image Classification | SGD + Momentum | Often achieves the best final generalization on vision tasks |
| Fraud Detection | AdamW | Handles sparse and dense features while converging quickly |
| Recommendation System | Adam / AdaGrad-family | Adaptive learning rates work well for sparse embeddings |
| Large Language Models | AdamW | Stable optimization across billions of parameters with decoupled weight decay |
| Robotics Control | RMSProp or Adam | Rapid adaptation to changing environments while maintaining stable updates |

---

# 🎤 Mock Interview Exercise

Without looking at your notes, answer the following in under one minute each.

1. Why wasn't Momentum enough?
2. Why wasn't RMSProp enough?
3. Why does Adam maintain two memories?
4. What is Bias Correction?
5. Why is Bias Correction needed?
6. What happens if Bias Correction is removed?
7. Why is Adam adaptive?
8. Why is Adam widely used for LLM training?
9. Why isn't Adam always the best optimizer?
10. Why was AdamW invented?


# 🚀 Success Criteria

You are ready to move to **AdamW** if you can confidently explain:

- Why Adam was invented
- Why two moving averages are required
- Why Bias Correction exists
- Why Adam works well for modern deep learning
- Why Adam still has limitations
- Why researchers eventually developed AdamW

without referring to your notes.