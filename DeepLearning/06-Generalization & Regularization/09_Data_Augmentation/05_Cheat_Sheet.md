# ⚡ Data Augmentation — Cheat Sheet

> One-page quick revision for interviews and last-minute preparation.

---

# 🎯 Definition

**Data Augmentation** is a **regularization technique** that artificially increases the size and diversity of the training dataset using **label-preserving transformations**.

**Goal:** Reduce overfitting and improve generalization.

---

# 🎯 Why do we need Data Augmentation?

Without augmentation:

```
Small Dataset
      ↓
Model Memorizes Data
      ↓
Overfitting
      ↓
Poor Generalization
```

With augmentation:

```
Small Dataset
      ↓
Many Variations
      ↓
Model Learns Robust Features
      ↓
Better Generalization
```

---

# 🎯 Common Image Augmentations

| Technique | Purpose |
|-----------|---------|
| 🔄 Horizontal Flip | Learn left-right invariance |
| 🔁 Rotation | Handle different viewing angles |
| ↔️ Translation | Handle object position changes |
| ✂️ Random Crop | Learn from partial objects |
| 🔍 Zoom | Handle scale variation |
| ☀️ Brightness | Handle lighting changes |
| 🎨 Contrast | Handle camera differences |
| 🌫️ Noise | Improve robustness |

---

# 🎯 Training vs Inference

## 🏋️ Training

```
Image
   ↓
Random Augmentation
   ↓
Model
```

✔ Different augmented image every epoch.

---

## 🚀 Inference

```
Image
   ↓
Model
   ↓
Prediction
```

✔ No augmentation (except optional Test-Time Augmentation).

---

# 🎯 Advanced Augmentation

## 🎨 Mixup

```
Image A
     +
Image B
     ↓
Mixed Image
```

✔ Mixes both images and labels.

---

## ✂️ CutMix

```
Image A

⬜⬜🐶⬜
```

✔ Replaces a patch with another image.

---

## ⬛ Random Erasing

```
🐱

↓

🐱⬛
```

✔ Simulates occlusion.

---

## 🔍 Test-Time Augmentation (TTA)

```
Original
Flip
Crop
Brightness
      ↓
Average Prediction
```

✔ Improves prediction stability.

---

# 🎯 NLP Augmentation

Since text cannot be rotated or flipped:

- 🔤 Synonym Replacement
- 🌍 Back Translation
- ✍️ Paraphrasing
- 🤖 Synthetic Data Generation

---

# 🎯 Audio Augmentation

- 🔊 Add Noise
- 🎵 Pitch Shift
- ⏩ Speed Change
- 🎚️ Volume Adjustment
- ⏱️ Time Shift

---

# 🎯 AutoAugment vs RandAugment

| Feature | AutoAugment | RandAugment |
|----------|-------------|-------------|
| Finds Best Policy | ✅ | ❌ |
| Search Cost | High | Low |
| Simplicity | Lower | Higher |

---

# 🎯 Data Augmentation in LLMs

### 📚 Pretraining

- Trained on billions of tokens
- Very little augmentation required

### 🎯 Fine-Tuning

Useful techniques:

- Back Translation
- Paraphrasing
- Instruction Rewriting
- Synthetic Q&A Generation

---

# 🎯 Advantages

- ✅ Reduces overfitting
- ✅ Improves generalization
- ✅ Increases effective dataset size
- ✅ Improves robustness
- ✅ Better real-world performance

---

# 🎯 Limitations

- ❌ Unrealistic transformations may change labels
- ❌ Domain-specific tuning is required
- ❌ Increases training time
- ❌ Not every augmentation suits every task

---

# 🎯 Common Interview Questions

✔ Why is Data Augmentation needed?

✔ How does it reduce overfitting?

✔ Why is augmentation applied only during training?

✔ Explain Mixup and CutMix.

✔ What is Test-Time Augmentation (TTA)?

✔ Why is NLP augmentation different from image augmentation?

✔ Why do LLMs use little augmentation during pretraining?

---

# 🧠 Memory Trick

```
Data Augmentation
│
├── Basic
│   ├── Flip
│   ├── Rotate
│   ├── Crop
│   ├── Zoom
│   ├── Brightness
│   ├── Contrast
│   └── Noise
│
├── Advanced
│   ├── Mixup
│   ├── CutMix
│   ├── Random Erasing
│   └── TTA
│
├── NLP
│   ├── Synonym
│   ├── Back Translation
│   └── Paraphrasing
│
└── Audio
    ├── Noise
    ├── Pitch
    ├── Speed
    └── Time Shift
```

---

# 🚀 30-Second Revision

- Data Augmentation is a **regularization technique** that creates new training samples using **label-preserving transformations**.
- It reduces **overfitting** by increasing training data diversity.
- Common image augmentations include **Flip, Rotation, Crop, Zoom, Brightness, Contrast, and Noise**.
- Advanced techniques include **Mixup, CutMix, Random Erasing, and Test-Time Augmentation**.
- NLP and Audio use task-specific augmentation methods.
- Large LLMs rely very little on augmentation during pretraining but may use it during fine-tuning.
```