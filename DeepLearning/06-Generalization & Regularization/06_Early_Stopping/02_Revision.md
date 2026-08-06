# 📝 Early Stopping — Revision Notes

> These notes are designed for **5-minute interview revision**.

---

# 🎯 What is Early Stopping?

Early Stopping is a **regularization technique** that prevents overfitting by **stopping training when validation performance stops improving.**

> **Idea:** Don't train until the lowest training loss. Train until the best validation performance.

---

# 🎯 Why was Early Stopping introduced?

Training a neural network for too long causes it to:

- Memorize training data
- Learn noise instead of patterns
- Reduce generalization

Instead of changing the model or loss function, Early Stopping changes **when training stops**.

---

# 🎯 Core Intuition

```text
Train Model
      ↓
Learn Useful Patterns
      ↓
Best Validation Performance
      ↓
Starts Memorizing Noise
      ↓
Overfitting
```

Early Stopping stops training **before memorization begins.**

---

# 🎯 Training Loss vs Validation Loss

```text
Training Loss
\
 \
  \
   \
    \
     \

Validation Loss
\
 \
  \
   \
    \__
       \
        \
         \
```

### Observation

- ✅ Training Loss keeps decreasing.
- ✅ Validation Loss first decreases, then starts increasing.

The **best model** is at the **lowest validation loss**, **not** the lowest training loss.

---

# 🎯 What does Early Stopping monitor?

It monitors the **validation metric**, such as:

- Validation Loss
- Validation Accuracy
- Validation F1 Score
- Validation AUC

Training stops when the chosen metric no longer improves.

---

# 🎯 What is Patience?

Patience tells the model to **wait for a few epochs** before stopping.

Example:

```text
Patience = 3

Epoch 18 ✅ Best Model

Epoch 19 ❌

Epoch 20 ❌

Epoch 21 ❌

Stop Training
```

### Why?

Validation metrics fluctuate slightly during training.

Patience prevents stopping because of temporary fluctuations.

---

# 🎯 Early Stopping Workflow

```text
Train One Epoch
        ↓
Evaluate Validation Metric
        ↓
Improved?
      /    \
    Yes     No
     ↓       ↓
Save Model   Increase Patience
Reset Counter
       ↓
Continue

Patience Exceeded?
        ↓
      Stop
        ↓
Restore Best Model
```

---

# 🎯 Why does Early Stopping work?

It finds the optimal point in the **Bias-Variance Tradeoff**.

```text
Few Epochs
      ↓
Underfitting

Optimal Epoch
      ↓
Best Generalization

Too Many Epochs
      ↓
Overfitting
```

---

# 🎯 Advantages

- ✅ Very easy to implement
- ✅ No change to model architecture
- ✅ No change to loss function
- ✅ Saves training time
- ✅ Improves generalization
- ✅ Works with almost all deep learning models

---

# 🎯 Limitations

- ❌ Requires a validation dataset
- ❌ Patience must be tuned
- ❌ Sensitive to noisy validation metrics
- ❌ Can stop too early if patience is too small

---

# 🎯 Early Stopping vs L1 vs L2

| Technique | Main Idea |
|-----------|-----------|
| **L1** | Removes unnecessary weights |
| **L2** | Shrinks large weights |
| **Elastic Net** | Combines L1 + L2 |
| **Early Stopping** | Stops training before memorization |

---

# 🎯 Interview One-Liners

### ⭐ What is Early Stopping?

A regularization technique that stops training when validation performance stops improving.

---

### ⭐ Does Early Stopping modify the loss function?

No.

It modifies the **training process**, not the objective function.

---

### ⭐ Which loss should we monitor?

Validation Loss (or another validation metric), **not Training Loss**.

---

### ⭐ Why don't we stop immediately after one bad epoch?

Because validation metrics fluctuate.

Patience allows temporary fluctuations before stopping.

---

### ⭐ Which model is finally used?

The model checkpoint with the **best validation performance**, not the last trained model.

---

# 🎯 Memory Trick

```text
L1
↓
Remove Weights

L2
↓
Shrink Weights

Elastic Net
↓
L1 + L2

Early Stopping
↓
Stop Training Early
```

---

# 🚀 Quick Revision

- ✅ Prevents overfitting by stopping training early
- ✅ Monitors validation performance
- ✅ Training loss may continue decreasing after the best model is reached
- ✅ Uses **Patience** to avoid stopping too early
- ✅ Restores the best model checkpoint
- ✅ Changes **training duration**, not the model or loss function
```