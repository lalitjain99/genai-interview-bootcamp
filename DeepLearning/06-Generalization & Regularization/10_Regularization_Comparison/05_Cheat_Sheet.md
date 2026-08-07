# ⚡ Regularization Comparison — Cheat Sheet

> One-page quick revision for the **entire Generalization & Regularization** module.

---

# 🎯 Why Regularization?

```
Complex Model
       ↓
Memorizes Training Data
       ↓
Overfitting
       ↓
Poor Generalization
```

**Goal of Regularization**

✔ Reduce Overfitting

✔ Improve Generalization

✔ Build Robust Models

---

# 🎯 Why So Many Regularization Techniques?

Different techniques solve different problems.

| Problem | Best Technique |
|----------|----------------|
| Too many irrelevant features | L1 |
| Large weights | L2 / Weight Decay |
| Training too long | Early Stopping |
| Small dataset | Data Augmentation |
| Neuron co-adaptation | Dropout |
| Unstable training | Batch Normalization |

---

# 🎯 Classification of Regularization

```text
                    Regularization
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        │                  │                  │
 Weight Based      Training Based      Data Based
        │                  │                  │
        │                  │                  │
   L1 Regularization   Early Stopping   Data Augmentation
   L2 Regularization
   Weight Decay
                           │
                           │
                    Architecture Based
                           │
                    ┌──────┴──────┐
                    │             │
                 Dropout     BatchNorm
```

---

# ⚖️ Weight-Based Regularization

| Technique | Purpose |
|-----------|----------|
| L1 | Feature Selection |
| L2 | Shrink Weights |
| Weight Decay | Control Weight Growth |

---

# ⏹️ Training-Based Regularization

| Technique | Purpose |
|-----------|----------|
| Early Stopping | Stop training before overfitting |

---

# 🖼️ Data-Based Regularization

| Technique | Purpose |
|-----------|----------|
| Data Augmentation | Increase effective dataset size |

---

# 🧩 Architecture-Based Regularization

| Technique | Purpose |
|-----------|----------|
| Dropout | Reduce neuron co-adaptation |
| BatchNorm | Stabilize and speed up training |

---

# 🎯 Complete Comparison

| Technique | Main Goal | Commonly Used In |
|-----------|-----------|------------------|
| L1 | Remove irrelevant features | Linear Models |
| L2 | Shrink large weights | ML Models |
| Weight Decay | Control weight growth | Deep Learning |
| Early Stopping | Prevent over-training | Deep Learning |
| Data Augmentation | Increase data diversity | CNNs, Fine-tuning |
| Dropout | Reduce co-adaptation | Fully Connected Networks |
| BatchNorm | Stabilize optimization | CNNs |

---

# 🎯 Which Technique Should You Choose?

| Situation | Best Choice |
|-----------|-------------|
| Too many irrelevant features | L1 |
| Correlated features | L2 |
| Deep Neural Networks | Weight Decay |
| Small Dataset | Data Augmentation |
| Validation loss increasing | Early Stopping |
| Neurons relying on each other | Dropout |
| Slow or unstable training | BatchNorm |

---

# 🎯 Modern Deep Learning

## 🖼️ CNNs

Commonly use:

- ✅ Weight Decay
- ✅ BatchNorm
- ✅ Data Augmentation
- ✅ Early Stopping

---

## 🤖 Transformers / LLMs

Commonly use:

- ✅ Weight Decay
- ✅ LayerNorm
- ✅ Early Stopping (Fine-tuning)

Very little Dropout during large-scale pretraining.

---

# 🎯 Can We Combine Techniques?

✅ Yes.

Different techniques solve different problems.

Example:

```
Weight Decay
      +
BatchNorm
      +
Data Augmentation
      +
Early Stopping
```

---

# 🎯 Interview One-Liners

### L1

> Removes irrelevant features by pushing some weights exactly to zero.

---

### L2

> Shrinks weights to reduce model complexity.

---

### Weight Decay

> Preferred regularization technique for modern optimizers like AdamW.

---

### Early Stopping

> Stops training before the model starts memorizing the training data.

---

### Data Augmentation

> Creates label-preserving variations to improve generalization.

---

### Dropout

> Randomly drops neurons to reduce co-adaptation.

---

### BatchNorm

> Normalizes activations to stabilize and accelerate training.

---

# 🧠 Memory Trick

```text
Too many Features?
        │
       L1
        │
Large Weights?
        │
     L2 / Weight Decay
        │
Training Too Long?
        │
 Early Stopping
        │
Small Dataset?
        │
Data Augmentation
        │
Neuron Dependency?
        │
    Dropout
        │
Training Unstable?
        │
   BatchNorm
```

---

# 🚀 30-Second Revision

- Regularization improves **generalization** by reducing **overfitting**.
- **L1** → Feature Selection.
- **L2** → Shrink Weights.
- **Weight Decay** → Control weight growth in deep networks.
- **Early Stopping** → Stop before overfitting.
- **Data Augmentation** → Increase effective dataset size.
- **Dropout** → Reduce neuron co-adaptation.
- **BatchNorm** → Stabilize and speed up training.
- **There is no best regularization technique**—choose the one that matches the problem your model is facing.