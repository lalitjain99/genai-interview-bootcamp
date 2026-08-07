# 💡 Assignment — Data Augmentation

> Practice what you've learned before moving to the next lecture.

---

# 📘 Assignment 1 — Scenario Based

## 🟢 Scenario A — Image Classification

A CNN is trained to classify cats and dogs using **8,000 training images**.

Training Accuracy = **99%**

Validation Accuracy = **83%**

The team currently trains using the original images only.

### Questions

### 1️⃣ Why is the model overfitting?

The model is repeatedly seeing the exact same training images and has started memorizing them instead of learning general features. This results in excellent training performance but poor generalization to unseen validation images.

---

### 2️⃣ How can Data Augmentation help?

Data Augmentation generates multiple variations of the same image using transformations like rotation, crop, flip, and brightness adjustment. This increases the diversity of the training data and forces the model to learn robust features instead of memorizing specific images.

---

### 3️⃣ Which image augmentations would you try first?

Start with safe augmentations such as:

- Horizontal Flip
- Small Rotation (±10° to ±15°)
- Random Crop
- Brightness Adjustment
- Random Zoom

---

### 4️⃣ Should Data Augmentation be applied during inference?

No.

During inference, the original image should normally be used so predictions remain deterministic and consistent.

---

### 5️⃣ What benefit does Data Augmentation provide?

- Reduces overfitting
- Improves generalization
- Increases effective dataset size
- Makes the model more robust to real-world variations

---

# 🟢 Scenario B — Medical Imaging

A hospital is building a CNN to detect lung diseases from chest X-rays.

An engineer proposes:

- Horizontal Flip
- 180° Rotation
- Random Crop
- Brightness Adjustment

### Questions

### 1️⃣ Would you approve all these augmentations?

No.

Medical images require domain-specific augmentation because orientation and anatomical structure can carry important diagnostic information.

---

### 2️⃣ Which augmentations would you question?

- Horizontal Flip
- 180° Rotation

These transformations may create unrealistic medical images or alter clinically important structures.

---

### 3️⃣ Which augmentation is relatively safer?

Mild brightness or contrast adjustments are generally safer because they simulate different imaging conditions without changing anatomy.

---

### 4️⃣ Why should augmentation always be domain-aware?

Because an augmentation that preserves the label in one domain may completely change the meaning in another.

For example:

- Horizontal flip is usually fine for cats.
- Horizontal flip may not be appropriate for chest X-rays.

---

# 🟢 Scenario C — NLP Fine-Tuning

You are fine-tuning a sentiment analysis model using only **3,000 labeled reviews**.

The team suggests rotating and flipping the text data similar to image augmentation.

### Questions

### 1️⃣ Do you agree?

No.

Image augmentation techniques cannot be directly applied to text.

---

### 2️⃣ Which NLP augmentation techniques would you recommend?

- Back Translation
- Paraphrasing
- Synonym Replacement
- Synthetic Data Generation

---

### 3️⃣ Why is NLP augmentation more difficult than image augmentation?

Because even small word changes can alter the meaning of a sentence.

The augmentation must preserve the original semantics.

---

### 4️⃣ Which NLP augmentation generally preserves meaning best?

Back Translation is one of the safest techniques because it usually keeps the sentence meaning intact while changing the wording.

---

# 📘 Assignment 2 — True / False

| # | Statement | Answer |
|---|-----------|--------|
| 1 | Data Augmentation helps reduce overfitting. | ✅ True |
| 2 | Data Augmentation creates completely new labeled data from scratch. | ❌ False |
| 3 | Augmentation is generally applied only during training. | ✅ True |
| 4 | Mixup combines both images and their labels. | ✅ True |
| 5 | CutMix replaces a patch of one image with another. | ✅ True |
| 6 | Test-Time Augmentation performs multiple augmented predictions during inference. | ✅ True |
| 7 | Horizontal flipping is always safe for every application. | ❌ False |
| 8 | Large LLMs rely heavily on Data Augmentation during pretraining. | ❌ False |
| 9 | Data Augmentation increases the diversity of the training data. | ✅ True |
| 10 | Unrealistic augmentation can hurt model performance. | ✅ True |

---

# ⭐ Staff Engineer Challenge

Your team is training an image classification model using **15,000 product images**.

The team proposes:

- Horizontal Flip
- 90° Rotation
- Random Crop
- Brightness Adjustment
- Mixup
- CutMix

They want to apply **every augmentation with maximum intensity** because they believe "more augmentation always gives better generalization."

---

## 1️⃣ Do you agree with the proposal?

No.

Data Augmentation should improve diversity while preserving the original label. Applying every augmentation aggressively can create unrealistic training samples and may actually reduce model performance.

---

## 2️⃣ What risks do you see?

- Unrealistic training images
- Label-changing transformations
- Slower training
- Poor convergence
- Reduced validation accuracy

---

## 3️⃣ How would you determine the best augmentation strategy?

Perform an **ablation study** by enabling one augmentation at a time (or in small combinations) and compare validation performance. Keep only the augmentations that consistently improve generalization.

---

## 4️⃣ What metrics would you monitor?

- Validation Accuracy
- Precision
- Recall
- F1-score
- Validation Loss

rather than relying only on training accuracy.

---

## 5️⃣ What principle should guide augmentation selection?

Choose augmentations that **preserve the semantic meaning of the data** while introducing realistic variations the model is likely to encounter in production.