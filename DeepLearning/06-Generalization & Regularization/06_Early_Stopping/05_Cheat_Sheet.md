# ⚡ Early Stopping — Cheat Sheet

> A one-page quick revision guide for interviews.

---

# 🎯 Definition

**Early Stopping** is a **regularization technique** that prevents overfitting by **stopping training when validation performance stops improving.**

---

# 🎯 Why was it introduced?

As training continues:

- ✅ Model first learns useful patterns.
- ❌ Later starts memorizing noise.
- ❌ Validation performance degrades.

Early Stopping prevents this by stopping training at the optimal point.

---

# 🎯 Core Idea

```text
Start Training
       ↓
Learn Useful Patterns
       ↓
Best Validation Performance
       ↓
Starts Memorizing Noise
       ↓
Overfitting
```

**Stop training before overfitting begins.**

---

# 🎯 Early Stopping Workflow

```text
Train One Epoch
       ↓
Evaluate Validation Metric
       ↓
Improved?
   /        \
 Yes         No
 ↓            ↓
Save Model    Increase Counter
Reset Counter
       ↓
Patience Exceeded?
       ↓
      Stop
       ↓
Restore Best Model
```

---

# 🎯 Training vs Validation Loss

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
   \__
      \
       \
        \
```

### Remember

- ✅ Training Loss usually keeps decreasing.
- ✅ Validation Loss eventually starts increasing.
- 🎯 Best model = **Lowest Validation Loss**

---

# 🎯 What is Patience?

Patience defines how many consecutive epochs without improvement are allowed before stopping.

Example:

```text
Patience = 3

Epoch 15 ✅ Best

Epoch 16 ❌

Epoch 17 ❌

Epoch 18 ❌

Stop Training
```

---

# 🎯 Which metric is monitored?

Common choices include:

- ✅ Validation Loss (Most Common)
- ✅ Validation Accuracy
- ✅ Validation F1 Score
- ✅ Validation AUC

---

# 🎯 Advantages

- ✅ Prevents overfitting
- ✅ Improves generalization
- ✅ Saves training time
- ✅ Easy to implement
- ✅ No change to model architecture
- ✅ No change to loss function

---

# 🎯 Limitations

- ❌ Requires a validation dataset
- ❌ Patience must be tuned
- ❌ Sensitive to noisy validation metrics
- ❌ May stop too early if Patience is too small

---

# 🎯 Early Stopping vs Other Regularization Techniques

| Technique | Primary Goal |
|-----------|--------------|
| **L1** | Remove unnecessary weights |
| **L2 / Weight Decay** | Shrink large weights |
| **Dropout** | Reduce neuron co-adaptation |
| **Batch Normalization** | Stabilize training |
| **Early Stopping** | Stop training before overfitting |

---

# 🎯 Interview One-Liners

### ⭐ What is Early Stopping?

Stops training automatically when validation performance stops improving.

---

### ⭐ Which loss is monitored?

Validation Loss (or another validation metric), **not Training Loss**.

---

### ⭐ Which model is deployed?

The checkpoint with the **best validation performance**, not the final epoch.

---

### ⭐ Why is Patience needed?

To avoid stopping because of temporary fluctuations in validation performance.

---

### ⭐ Does Early Stopping modify the loss function?

❌ No.

It modifies the **training process**, not the objective function.

---

# 🚀 30-Second Revision

- ✅ Prevents overfitting
- ✅ Stops training based on validation performance
- ✅ Uses Patience to avoid premature stopping
- ✅ Restores the best model checkpoint
- ✅ Changes training duration, not the model or loss function
- ✅ Commonly used alongside L1, L2, and Dropout

---

# 🧠 Memory Trick

```text
L1
↓
Remove Weights

L2
↓
Shrink Weights

Dropout
↓
Remove Neurons (Temporarily)

BatchNorm
↓
Normalize Activations

Early Stopping
↓
Stop Training Early
```