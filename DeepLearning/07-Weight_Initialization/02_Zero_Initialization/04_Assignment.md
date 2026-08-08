# 💡 Assignment — Zero Initialization

> **Focus:** Understanding the symmetry problem and why neural networks need symmetry breaking.

---

# 📌 Assignment 1 — Scenario Based

## 🧠 Scenario A — All Hidden Weights Are Zero

You have a neural network:

```text
Input Layer
    ↓
Hidden Layer (3 neurons)
    ↓
Output Layer
```

All weights connected to the hidden layer are initialized as:

```text
w₁ = 0
w₂ = 0
w₃ = 0
```

### Questions

### 1️⃣ What will happen during forward propagation?

✅ **Answer**

The hidden neurons have identical weights and receive the same inputs.

Therefore, they produce identical outputs.

```text
Same Weights
     ↓
Same Outputs
```

---

### 2️⃣ What happens during backpropagation?

✅ **Answer**

Because the hidden neurons behave identically, they receive the same gradients.

```text
Same Outputs
     ↓
Same Gradients
```

---

### 3️⃣ What happens when Gradient Descent updates the weights?

✅ **Answer**

The same gradients produce the same weight updates.

Therefore, the neurons remain identical.

```text
Same Gradients
     ↓
Same Updates
     ↓
Same Weights
```

---

### 4️⃣ What is the name of this problem?

✅ **Answer**

This is called the **Symmetry Problem**.

---

### 5️⃣ How can we solve this problem?

✅ **Answer**

Use **Random Initialization** so that different neurons start with different weights.

```text
0.002
-0.004
0.001
```

This breaks the symmetry and allows neurons to learn different features.

---

# 🧠 Scenario B — Why Multiple Neurons?

Suppose a hidden layer contains four neurons.

The goal is for them to learn different features.

### Questions

### 1️⃣ Why is it useful for different neurons to learn different features?

✅ **Answer**

Different neurons can detect different patterns.

For example:

```text
Neuron 1 → Edges
Neuron 2 → Corners
Neuron 3 → Textures
Neuron 4 → Shapes
```

This allows the network to learn richer representations.

---

### 2️⃣ What happens if all four neurons remain identical?

✅ **Answer**

They effectively learn the same representation.

Therefore, having four neurons provides much less benefit than having four independent feature detectors.

---

### 3️⃣ What does Random Initialization achieve?

✅ **Answer**

It gives neurons different starting points.

This breaks symmetry and allows their weights to evolve differently during training.

---

# 🧩 Scenario C — Logistic Regression vs Neural Network

You are asked to initialize the parameters of two models with zeros.

### Model A

```text
Logistic Regression

Input
  ↓
Single Output
```

### Model B

```text
Neural Network

Input
  ↓
Hidden Layer
  ↓
Output
```

### Questions

### 1️⃣ Can Zero Initialization work for Model A?

✅ **Answer**

Yes.

Logistic Regression has a single output neuron, so there is no hidden-layer symmetry problem.

---

### 2️⃣ Can Zero Initialization be used for the hidden-layer weights of Model B?

✅ **Answer**

No.

Multiple hidden neurons initialized identically will remain identical and fail to learn different features.

---

### 3️⃣ What is the key difference between the two models?

✅ **Answer**

The neural network contains multiple hidden neurons that need to learn different representations.

Logistic Regression does not have this hidden-layer symmetry problem.

---

# 📌 Assignment 2 — True / False

| #  | Statement                                                                          | Answer  |
| -- | ---------------------------------------------------------------------------------- | ------- |
| 1  | Zero Initialization sets all weights to zero.                                      | ✅ True  |
| 2  | Zero Initialization always fails for every machine learning model.                 | ❌ False |
| 3  | Zero initialization can work for Logistic Regression.                              | ✅ True  |
| 4  | Identical hidden-layer weights can create a symmetry problem.                      | ✅ True  |
| 5  | Identical neurons receive different gradients automatically.                       | ❌ False |
| 6  | Identical gradients produce identical weight updates.                              | ✅ True  |
| 7  | Random Initialization helps break symmetry.                                        | ✅ True  |
| 8  | Zero initialization allows hidden neurons to learn different features.             | ❌ False |
| 9  | Different neurons should generally be able to learn different features.            | ✅ True  |
| 10 | Zero initialization is problematic mainly because hidden neurons remain identical. | ✅ True  |

---

# ⭐ Staff Engineer Challenge

### Scenario

A junior engineer says:

> **"I don't understand why zero initialization is a problem. Gradient Descent will update the weights during training, so eventually every neuron should become different."**

### 1️⃣ Do you agree?

✅ **Answer**

No.

If the neurons start identically, they receive the same inputs, produce the same outputs, and receive the same gradients.

Therefore, Gradient Descent applies the same update to every neuron.

They remain identical.

---

### 2️⃣ Explain the problem using a simple chain.

✅ **Answer**

```text
Identical Initialization
        ↓
Identical Outputs
        ↓
Identical Gradients
        ↓
Identical Updates
        ↓
Identical Weights
        ↓
No Feature Specialization
```

This is the **symmetry problem**.

---

### 3️⃣ What would you change in the initialization strategy?

✅ **Answer**

Use **Random Initialization** for the weights.

For example:

```text
Neuron 1 →  0.003
Neuron 2 → -0.002
Neuron 3 →  0.001
```

These small differences break symmetry.

---

### 4️⃣ Is "random" alone enough to guarantee a good initialization?

✅ **Answer**

No.

Random initialization solves the **symmetry problem**, but the magnitude of the weights also matters.

Weights that are too large or too small can cause problems such as:

* Exploding gradients
* Vanishing gradients

This is why more carefully designed methods such as **Xavier and He Initialization** are used.

---

# 🎯 Final Challenge

Complete this statement:

> **"We don't use zero initialization for hidden-layer weights because..."**

✅ **Answer**

> We don't use zero initialization for hidden-layer weights because all neurons would start identically, produce identical outputs, receive identical gradients, and receive identical updates. They would remain copies of one another instead of learning different features. This is known as the **symmetry problem**.
