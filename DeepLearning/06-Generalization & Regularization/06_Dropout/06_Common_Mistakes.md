# ❌ Common Mistakes — Dropout Regularization

---

# 🚫 Mistake 1: Thinking Dropout permanently removes neurons

### ❌ Wrong

> Dropout permanently deletes neurons from the network.

### ✅ Correct

Dropout removes neurons **only during the current training iteration**.

A new random Dropout mask is generated for every mini-batch, so the same neuron may be active in the next iteration.

---

# 🚫 Mistake 2: Generating a new Dropout mask during Backpropagation

### ❌ Wrong

A separate Dropout mask is created during backpropagation.

### ✅ Correct

The **same Dropout mask** generated during forward propagation is reused during backpropagation.

Only neurons that participated in the forward pass receive gradients.

---

# 🚫 Mistake 3: Assuming dropped neurons still receive gradients

### ❌ Wrong

Every neuron gets updated during backpropagation.

### ✅ Correct

Dropped neurons receive

```text
Gradient = 0
```

Their weights are **not updated** during that iteration.

---

# 🚫 Mistake 4: Forgetting Inverted Dropout scaling

### ❌ Wrong

Simply drop neurons without scaling the remaining activations.

### ✅ Correct

The surviving neurons are scaled by

```text
1 / Keep Probability
```

during training so that the expected activation remains unchanged.

---

# 🚫 Mistake 5: Applying Dropout during inference

### ❌ Wrong

Randomly drop neurons while making predictions.

### ✅ Correct

During inference, **all neurons remain active**.

No Dropout mask is generated.

---

# 🚫 Mistake 6: Believing higher Dropout is always better

### ❌ Wrong

More Dropout always improves generalization.

### ✅ Correct

Too much Dropout reduces the network's learning capacity and leads to **underfitting**.

The goal is to find a balance.

---

# 🚫 Mistake 7: Confusing Dropout with L1/L2 Regularization

### ❌ Wrong

Dropout penalizes model weights.

### ✅ Correct

- **L1** penalizes absolute weights.
- **L2** penalizes squared weights.
- **Dropout** randomly disables neurons.

Dropout never adds a penalty term to the loss function.

---

# 🚫 Mistake 8: Thinking Dropout performs Feature Selection

### ❌ Wrong

Dropout permanently removes unimportant features.

### ✅ Correct

Dropout temporarily disables **neurons**, not input features.

Feature selection is a property of **L1 Regularization**, not Dropout.

---

# 🚫 Mistake 9: Assuming the same neurons are dropped every iteration

### ❌ Wrong

The same neurons remain inactive throughout training.

### ✅ Correct

Every mini-batch gets a **new random Dropout mask**.

Different neurons are dropped in different iterations.

---

# 🚫 Mistake 10: Thinking Dropout reduces Overfitting by shrinking weights

### ❌ Wrong

Dropout reduces overfitting by making weights smaller.

### ✅ Correct

Dropout reduces overfitting by preventing **co-adaptation** between neurons.

The weights are updated normally for active neurons; there is no explicit weight penalty.

---

# 📝 Quick Interview Recap

✅ Dropout is **temporary**, not permanent.

✅ Same mask is used in forward and backward propagation.

✅ Dropped neurons receive **zero gradients**.

✅ Inverted Dropout scales surviving neurons during training.

✅ During inference, **no neurons are dropped**.

✅ Higher Dropout is **not always better**.

✅ Dropout prevents **co-adaptation**, not large weights.

✅ L1 → Feature Selection

✅ L2 → Weight Shrinkage

✅ Dropout → Temporary Neuron Removal