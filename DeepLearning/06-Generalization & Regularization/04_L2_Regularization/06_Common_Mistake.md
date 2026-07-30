# ❌ Common Mistakes — L2 Regularization

---

# 1. Thinking L2 Removes Features

### ❌ Incorrect

> L2 makes unimportant feature weights exactly zero.

### ✅ Correct

L2 **shrinks weights toward zero** but usually **does not make them exactly zero**.

It keeps all features in the model.

---

# 2. Confusing L1 and L2

### ❌ Incorrect

> L1 and L2 both remove unnecessary features.

### ✅ Correct

- **L1** performs feature selection by driving some weights to exactly zero.
- **L2** keeps every feature but reduces the magnitude of their weights.

---

# 3. Thinking Large λ Always Improves Generalization

### ❌ Incorrect

> Increasing λ always makes the model better.

### ✅ Correct

A larger λ increases regularization.

If λ becomes too large, the model becomes too simple and starts **underfitting**.

There is always an optimal range of λ.

---

# 4. Believing L2 Penalizes Only Large Weights

### ❌ Incorrect

> L2 only affects large weights.

### ✅ Correct

L2 penalizes **every weight**.

However, because the penalty is proportional to **w²**, larger weights receive a much stronger penalty than smaller ones.

---

# 5. Assuming Important Features Escape Regularization

### ❌ Incorrect

> L2 leaves important features untouched.

### ✅ Correct

L2 penalizes **all weights**, including important ones.

If a feature is genuinely useful, the reduction in prediction loss outweighs the regularization penalty, so the optimizer naturally keeps its weight relatively large.

---

# 6. Thinking the Optimizer Minimizes Two Losses Separately

### ❌ Incorrect

> The optimizer first minimizes Prediction Loss and then minimizes Regularization Loss.

### ✅ Correct

The optimizer minimizes a **single combined objective**:

\[
Total\ Loss
=
Prediction\ Loss
+
\lambda \times Regularization\ Loss
\]

Both objectives influence every gradient update simultaneously.

---

# 7. Assuming Prediction Loss and Regularization Loss "Fight Forever"

### ❌ Incorrect

> Prediction Loss always increases weights while Regularization always decreases them, so the optimizer never converges.

### ✅ Correct

The optimizer finds a balance where the improvement in prediction no longer justifies increasing the regularization penalty.

This balance is the optimal solution.

---

# 8. Thinking L2 Makes Every Weight Zero Eventually

### ❌ Incorrect

> Since every update shrinks weights, all weights must eventually become zero.

### ✅ Correct

As weights become smaller, the regularization gradient also becomes smaller.

Eventually, reducing the weight further hurts prediction accuracy more than it helps regularization, so the optimizer stops shrinking it.

---

# 9. Believing Weight Decay Means "Delete the Weights"

### ❌ Incorrect

> Weight Decay means weights disappear.

### ✅ Correct

Weight Decay simply means weights **gradually decrease in magnitude** during training.

They usually remain non-zero.

---

# 10. Assuming L2 Reduces the Number of Parameters

### ❌ Incorrect

> L2 reduces the number of neurons or parameters in the network.

### ✅ Correct

L2 keeps the same architecture.

It only reduces the **magnitude** of the parameters, lowering the model's **effective complexity**.

---

# 11. Thinking L2 Is Always Better Than L1

### ❌ Incorrect

> L2 should always be preferred.

### ✅ Correct

Choose based on the problem:

- **L1** → when many features are irrelevant and feature selection is desired.
- **L2** → when most features are useful and should be retained.

---

# 12. Assuming L2 Solves All Overfitting Problems

### ❌ Incorrect

> Adding L2 completely prevents overfitting.

### ✅ Correct

L2 is only one regularization technique.

Other approaches include:

- More training data
- Dropout
- Early Stopping
- Data Augmentation
- Better model architecture

---

# 13. Thinking L2 and Weight Decay Are Always the Same

### ❌ Incorrect

> L2 Regularization and Weight Decay are identical for every optimizer.

### ✅ Correct

They are mathematically equivalent for **vanilla SGD**.

For adaptive optimizers like **Adam**, they behave differently.

This is why **AdamW** was introduced.

---

# 14. Forgetting the Role of λ

### ❌ Incorrect

> λ is just another learning rate.

### ✅ Correct

- **Learning Rate (η)** controls **how big each update is**.
- **Lambda (λ)** controls **how strongly large weights are penalized**.

They serve completely different purposes.

---

# 🎯 Interview Tip

A common interview trap is:

> **"Does L2 remove features?"**

A strong answer is:

> **No. L2 shrinks all weights toward zero but rarely makes them exactly zero. Therefore, it produces a dense model, unlike L1, which performs feature selection by driving some weights exactly to zero.**