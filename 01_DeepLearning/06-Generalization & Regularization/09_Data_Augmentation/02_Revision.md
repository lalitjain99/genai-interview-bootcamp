# 📝 Data Augmentation — Revision Notes

> Quick revision notes for interviews and last-minute preparation.

---

# 🎯 Definition

**Data Augmentation** is a **regularization technique** that artificially creates new training samples by applying **label-preserving transformations** to existing data.

---

# 🎯 Why was Data Augmentation introduced?

Deep Neural Networks easily **memorize** small datasets.

This leads to:

- ❌ Overfitting
- ❌ Poor generalization

Instead of collecting more data,

we generate more training examples from existing data.

---

# 🎯 Why not collect more data?

Collecting real-world data is often:

- 💰 Expensive
- ⏳ Time-consuming
- 🏷️ Requires manual labeling
- 🔒 Restricted by privacy
- 📉 Difficult for rare events

Data Augmentation is a cheaper alternative.

---

# 🎯 Core Idea

```text
One Training Image
        │
        ▼
Multiple Augmented Images
        │
        ▼
Model Learns Robust Features
        │
        ▼
Better Generalization
```

---

# 🎯 Common Image Augmentation Techniques

| Technique | Purpose |
|-----------|---------|
| 🔄 Horizontal Flip | Learn left-right invariance |
| 🔁 Rotation | Handle different viewing angles |
| ↔️ Translation | Handle object position changes |
| ✂️ Random Crop | Learn from partial objects |
| 🔍 Zoom | Handle different object sizes |
| ☀️ Brightness | Handle lighting variations |
| 🎨 Contrast | Handle camera/environment changes |
| 🌫️ Noise | Improve robustness to noisy images |

---

# 🎯 Why does Data Augmentation reduce overfitting?

Without augmentation,

the model memorizes:

- Background
- Lighting
- Camera angle
- Object position

With augmentation,

every epoch presents a slightly different version of the same image.

The model learns:

- ✅ Edges
- ✅ Shapes
- ✅ Textures
- ✅ Object identity

instead of memorizing exact images.

---

# 🎯 What is Invariance?

The model learns that certain transformations **do not change the class label**.

Example:

```text
🐱 → Rotate → 🐱 ✅

🐱 → Brightness → 🐱 ✅

🐱 → Crop → 🐱 ✅

🐱 → Shift → 🐱 ✅
```

The label remains unchanged.

---

# 🎯 Training vs Inference

### During Training

```text
Image
   │
Random Augmentation
   │
Model
```

Different augmented versions are shown in every epoch.

---

### During Inference

```text
Image
   │
Model
   │
Prediction
```

No augmentation is applied.

---

# 🎯 Advanced Data Augmentation

## 🎨 Mixup

Combine two images and their labels.

```text
Cat + Dog
      │
      ▼
Mixed Image
```

✔️ Learns smoother decision boundaries.

---

## ✂️ CutMix

Copy a patch from one image into another.

```text
Cat Image

⬜⬜🐶⬜
```

✔️ Encourages the model to use the whole object instead of one small region.

---

## ⬛ Random Erasing

Randomly removes part of an image.

```text
🐱

↓

🐱⬛
```

✔️ Improves robustness to occlusion.

---

## 🔍 Test-Time Augmentation (TTA)

Apply augmentation during inference.

```text
Original
│
├── Flip
├── Crop
├── Brightness
│
▼
Average Predictions
```

✔️ Produces more stable predictions.

---

# 🎯 Data Augmentation in NLP

Unlike images, text cannot be rotated or flipped.

Common techniques:

- 🔤 Synonym Replacement
- ❌ Random Word Deletion
- 🔀 Random Word Swap
- 🌍 Back Translation

---

# 🎯 Data Augmentation in Audio

Examples:

- 🔊 Add Noise
- 🎵 Change Pitch
- ⏩ Speed Change
- 🎚️ Volume Adjustment
- ⏱️ Time Shift

---

# 🎯 AutoAugment vs RandAugment

| Feature | AutoAugment | RandAugment |
|----------|-------------|-------------|
| Finds Best Policy | ✅ Yes | ❌ No |
| Search Cost | High | Low |
| Simplicity | Lower | Higher |

---

# 🎯 Data Augmentation in LLMs

### During Pretraining

Large LLMs (GPT, LLaMA, Gemini) are trained on **billions of tokens**.

➡️ Very little augmentation is needed.

---

### During Fine-Tuning

Fine-tuning datasets are much smaller.

Useful augmentation techniques include:

- Back Translation
- Paraphrasing
- Synthetic Question Generation
- Instruction Rewriting

---

# 🎯 Advantages

- ✅ Reduces overfitting
- ✅ Improves generalization
- ✅ Increases effective dataset size
- ✅ Makes models more robust
- ✅ No change in model architecture

---

# 🎯 Limitations

- ❌ Unrealistic augmentation may change the label.
- ❌ Different tasks require different augmentations.
- ❌ Increases training time.
- ❌ Some augmentations require domain expertise.

---

# 🚀 30-Second Interview Revision

- Data Augmentation creates new training samples using **label-preserving transformations**.
- It reduces overfitting by exposing the model to diverse versions of the same data.
- Common techniques include **Flip, Rotation, Crop, Zoom, Brightness, Contrast, and Noise**.
- Advanced methods include **Mixup, CutMix, Random Erasing, and Test-Time Augmentation**.
- NLP and Audio use domain-specific augmentation methods.
- Large LLMs use little augmentation during pretraining but may use augmentation during fine-tuning.

---

# 🧠 Memory Trick

```text
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
└── Advanced
    ├── Mixup
    ├── CutMix
    ├── Random Erasing
    ├── TTA
    ├── NLP Augmentation
    ├── Audio Augmentation
    ├── AutoAugment
    └── RandAugment
```