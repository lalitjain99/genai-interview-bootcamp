# 💡 Assignment — Regularization Comparison

> This assignment focuses on **choosing the right regularization technique** for different scenarios. It is designed to test your understanding of the **entire Generalization & Regularization module**, not just individual techniques.

---

# 📌 Assignment 1 — Scenario Based

---

# 🖥️ Scenario A — Small Dataset

You are building a CNN for classifying handwritten characters.

- Training Images: **3,500**
- Validation Images: **800**
- Training Accuracy: **99%**
- Validation Accuracy: **83%**

### Questions

### 1️⃣ Why is the model overfitting?

✅ **Answer**

The dataset is too small, so the model memorizes the training images instead of learning general patterns.

---

### 2️⃣ Which regularization technique should you try first?

✅ **Answer**

**Data Augmentation**, because it increases the effective size and diversity of the training dataset.

---

### 3️⃣ Name three augmentation techniques you would apply.

✅ **Answer**

- Rotation
- Random Crop
- Brightness Adjustment

---

### 4️⃣ Would Weight Decay alone solve this problem?

✅ **Answer**

No.

Weight Decay controls large weights, but it cannot compensate for the lack of training data.

---

### 5️⃣ What additional regularization technique could be combined with Data Augmentation?

✅ **Answer**

**Early Stopping**, to stop training before the model starts memorizing the training data.

---

# 🖥️ Scenario B — Training Too Long

A neural network has been training for 150 epochs.

You observe:

- Training Loss keeps decreasing.
- Validation Loss starts increasing after epoch 80.

### Questions

### 1️⃣ What problem is occurring?

✅ **Answer**

The model is **overfitting** after prolonged training.

---

### 2️⃣ Which regularization technique is most suitable?

✅ **Answer**

**Early Stopping.**

---

### 3️⃣ Should training continue after validation loss starts increasing?

✅ **Answer**

No.

Training should stop once validation performance stops improving.

---

### 4️⃣ Why is Early Stopping considered a regularization technique?

✅ **Answer**

Because it prevents the model from learning unnecessary details and memorizing the training data.

---

### 5️⃣ Can Early Stopping be combined with Weight Decay?

✅ **Answer**

Yes.

They solve different problems and are commonly used together.

---

# 🖥️ Scenario C — Large Weights

A deep neural network has extremely large parameter values and performs well on training data but poorly on validation data.

### Questions

### 1️⃣ Which regularization technique is most appropriate?

✅ **Answer**

**Weight Decay**.

---

### 2️⃣ Why is Weight Decay preferred over traditional L2 Regularization in modern deep learning?

✅ **Answer**

Because modern optimizers like **AdamW** decouple Weight Decay from the loss function, making regularization more effective.

---

### 3️⃣ Would L1 Regularization be a better choice here?

✅ **Answer**

No.

L1 is mainly used for **feature selection**, not for controlling weight growth in deep neural networks.

---

### 4️⃣ Can Weight Decay and Early Stopping be used together?

✅ **Answer**

Yes.

Weight Decay controls parameter growth, while Early Stopping prevents over-training.

---

### 5️⃣ What happens if Weight Decay is too large?

✅ **Answer**

Weights become too small, causing **underfitting**.

---

# 🖥️ Scenario D — Choosing the Right Technique

For each situation, identify the most appropriate regularization technique.

| Situation | Answer |
|-----------|--------|
| Too many irrelevant features | ✅ L1 Regularization |
| Very large weights | ✅ L2 / Weight Decay |
| Small dataset | ✅ Data Augmentation |
| Training for too many epochs | ✅ Early Stopping |
| Neurons become dependent on each other | ✅ Dropout |
| Training becomes unstable | ✅ Batch Normalization |

---

# 📌 Assignment 2 — True / False

| # | Statement | Answer |
|---|-----------|--------|
| 1 | L1 Regularization performs feature selection. | ✅ True |
| 2 | Weight Decay is commonly used with AdamW. | ✅ True |
| 3 | Early Stopping reduces overfitting by stopping training early. | ✅ True |
| 4 | Data Augmentation increases dataset diversity. | ✅ True |
| 5 | Dropout modifies the training data. | ❌ False |
| 6 | Batch Normalization stabilizes training. | ✅ True |
| 7 | Every regularization technique solves the same problem. | ❌ False |
| 8 | Multiple regularization techniques can be combined. | ✅ True |
| 9 | Choosing the correct regularization depends on the problem being solved. | ✅ True |
| 10 | Weight Decay and L1 Regularization have the same primary objective. | ❌ False |

---

# ⭐ Staff Engineer Challenge

Your team is developing a deep learning model for a production system.

A junior engineer says:

> **"Let's apply L1, L2, Weight Decay, Dropout, Batch Normalization, Early Stopping and Data Augmentation together. More regularization will always produce a better model."**

---

### 1️⃣ Do you agree?

✅ **Answer**

No.

There is no universal combination that works best for every problem. The correct techniques depend on the model, dataset, and the actual cause of overfitting.

---

### 2️⃣ Why is blindly applying every technique a bad idea?

✅ **Answer**

Because it can:

- Increase training time
- Add unnecessary complexity
- Cause excessive regularization
- Lead to underfitting
- Make hyperparameter tuning more difficult

---

### 3️⃣ How would you choose the right regularization technique?

✅ **Answer**

First identify the root cause of the problem.

Examples:

- Small dataset → Data Augmentation
- Large weights → Weight Decay
- Too many epochs → Early Stopping
- Neuron co-adaptation → Dropout
- Unstable training → Batch Normalization

---

### 4️⃣ How would you validate your decision?

✅ **Answer**

Perform **ablation studies** by adding or removing one regularization technique at a time and comparing validation performance.

---

### 5️⃣ What is the biggest takeaway from the Generalization & Regularization module?

✅ **Answer**

> **The goal is not to use more regularization techniques, but to choose the right technique for the specific problem.**