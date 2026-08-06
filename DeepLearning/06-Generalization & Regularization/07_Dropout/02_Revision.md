# 📝 Revision — Dropout Regularization

---

# 🎯 Why was Dropout introduced?

In previous lectures, we studied:

- ✅ L1 Regularization
- ✅ L2 Regularization
- ✅ Elastic Net

These techniques regularize a model by **penalizing weights**.

However, researchers observed that deep neural networks were still overfitting.

They discovered another problem:

> **Neurons were becoming too dependent on one another.**

This phenomenon is called **Co-adaptation**.

---

# 🤝 What is Co-adaptation?

Co-adaptation occurs when neurons rely heavily on specific neighboring neurons instead of learning useful features independently.

Example:

Imagine a team where one employee always solves every difficult problem.

Eventually,

everyone else depends on that employee.

If that employee is unavailable,

the whole team struggles.

The same thing can happen inside a neural network.

---

# 💡 Core Idea Behind Dropout

Instead of penalizing weights,

Dropout randomly **turns off neurons during training**.

Each training iteration uses a different subset of neurons.

Example:

Iteration 1

```text
● ● X ● X
```

Iteration 2

```text
X ● ● X ●
```

Iteration 3

```text
● X ● ● ●
```

This forces every neuron to learn useful features independently.

---

# 🎯 What is Dropout Rate?

Dropout Rate specifies the fraction of neurons that will be randomly removed during training.

Example:

```text
Dropout Rate = 0.4
```

means

40% of neurons are dropped,

60% remain active.

---

# 🧩 Dropout Mask

For every forward pass,

a random binary mask is generated.

Example:

```text
Mask

1 0 1 1 0
```

```text
1 → Keep neuron

0 → Drop neuron
```

A new mask is generated for every mini-batch.

---

# ⚙️ Forward Propagation

Only active neurons participate in computation.

Dropped neurons output:

```text
0
```

They behave as if they do not exist during that iteration.

---

# 🔄 Backpropagation

Only active neurons receive gradients.

Dropped neurons receive:

```text
Gradient = 0
```

Therefore,

their weights are **not updated** during that iteration.

They are available again in future iterations when selected by a new dropout mask.

---

# 📈 Inverted Dropout

Dropping neurons reduces the total activation of the layer.

To maintain a consistent expected activation,

the surviving neurons are scaled during training.

Example:

```text
Dropout Rate = 0.5

Keep Probability = 0.5
```

Surviving neurons are divided by

```text
0.5
```

(or multiplied by 2).

This technique is called **Inverted Dropout**.

---

# 🏋️ Training vs 🎯 Inference

## During Training

- Random neurons are dropped.
- A new mask is generated every mini-batch.
- Surviving neurons are scaled.

---

## During Inference

- No neurons are dropped.
- The complete network is used.
- No scaling is required because scaling was already handled during training.

---

# ❓ Why Isn't Dropout Applied to the Output Layer?

Hidden layers learn intermediate features.

The output layer produces the model's final prediction.

If output neurons were randomly dropped:

- Prediction may become incomplete or invalid.
- Loss cannot be computed correctly.
- Backpropagation would fail.

Therefore,

Dropout is almost always applied only to **hidden layers**.

---

# 🎯 Typical Dropout Rates

| Dropout Rate | Usage |
|--------------|------|
| 0.1 | Light regularization |
| 0.2–0.3 | Common in shallow networks |
| 0.5 | Standard for fully connected layers |
| >0.7 | Usually too aggressive |

---

# ✅ Advantages

- Reduces overfitting.
- Prevents co-adaptation.
- Encourages independent feature learning.
- Improves generalization.
- Acts like training many subnetworks.

---

# ⚠️ Limitations

- Slower convergence.
- High dropout can cause underfitting.
- Less useful in some modern architectures.
- Often reduced or omitted in modern LLM pretraining.

---

# 📊 Comparison

| Property | L1 | L2 | Dropout |
|----------|----|----|----------|
| Penalizes weights | ✅ | ✅ | ❌ |
| Removes features | ✅ | ❌ | ❌ |
| Shrinks weights | ❌ | ✅ | Indirectly |
| Removes neurons | ❌ | ❌ | ✅ (temporarily) |
| Reduces co-adaptation | ❌ | ❌ | ✅ |
| Mainly used in deep networks | ❌ | ❌ | ✅ |

---

# 🚀 Key Takeaways

- ✅ Dropout randomly disables neurons during **training only**.
- ✅ A new dropout mask is created for every mini-batch.
- ✅ Dropped neurons do not participate in forward or backward propagation.
- ✅ Inverted Dropout scales surviving neurons to maintain consistent activations.
- ✅ During inference, all neurons are active.
- ✅ Dropout reduces co-adaptation and improves generalization.
- ✅ Dropout is applied to hidden layers, not the output layer.
- ✅ It acts like training many different subnetworks that share the same weights.