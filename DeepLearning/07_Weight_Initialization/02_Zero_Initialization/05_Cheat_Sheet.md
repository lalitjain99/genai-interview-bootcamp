# ⚡ Cheat Sheet — Zero Initialization

> **Core Idea:** Zero initialization causes hidden neurons to remain identical → **Symmetry Problem**.

---

## 🎯 What Is Zero Initialization?

```text
Every weight starts as:

w = 0
```

Example:

```text
w₁ = 0
w₂ = 0
w₃ = 0
```

---

## 🚨 Why Does It Fail?

```text
Zero Initialization
        ↓
Identical Weights
        ↓
Identical Outputs
        ↓
Identical Gradients
        ↓
Identical Updates
        ↓
Identical Weights
        ↓
❌ Symmetry Problem
```

---

## 🧠 The Symmetry Problem

Multiple hidden neurons start exactly the same:

```text
Neuron 1 → w = 0
Neuron 2 → w = 0
Neuron 3 → w = 0
```

They continue receiving the same updates.

Therefore:

> **They behave like copies of one another.**

---

## 🔄 Forward Propagation

Basic equation:

[
z = wx + b
]

If:

[
w = 0
]

then hidden neurons start from the same state and produce identical activations.

---

## 🔙 Backpropagation

Identical neurons produce identical gradients:

[
dw_1 = dw_2 = dw_3
]

Gradient Descent:

[
w_{new} = w - \eta dw
]

Same gradient → same update → neurons remain identical.

---

## 💡 Why Do We Need Different Neurons?

Different neurons should learn different features.

```text
Neuron 1 → Edges
Neuron 2 → Corners
Neuron 3 → Textures
Neuron 4 → Shapes
```

Zero initialization prevents this specialization.

---

## 🔓 Symmetry Breaking

Use different initial weights:

```text
Neuron 1 →  0.003
Neuron 2 → -0.002
Neuron 3 →  0.001
```

Then:

```text
Different Weights
       ↓
Different Outputs
       ↓
Different Gradients
       ↓
Different Updates
       ↓
Different Features
```

---

# ⚖️ Logistic Regression vs Neural Network

|                        | Logistic Regression | Neural Network           |
| ---------------------- | ------------------- | ------------------------ |
| Hidden neurons         | ❌                   | ✅                        |
| Symmetry problem       | ❌                   | ✅                        |
| Zero initialization    | ✅ Can work          | ❌ Not for hidden weights |
| Need symmetry breaking | ❌                   | ✅                        |

---

## ⚠️ Important Distinction

The fundamental problem is **identical initialization of multiple neurons**.

It isn't simply:

> "The number zero is bad."

The issue is:

```text
Same Starting Point
       +
Same Inputs
       +
Same Gradient
       ↓
Same Updates
```

---

## 🧩 What About Biases?

Zero biases are generally fine.

Typical idea:

```text
Weights → Random / carefully initialized
Biases  → Often 0
```

The important part is breaking symmetry between the **weights of different neurons**.

---

# 🚀 Zero vs Random Initialization

| Zero Initialization               | Random Initialization      |
| --------------------------------- | -------------------------- |
| All neurons start identically     | Neurons start differently  |
| Symmetry remains                  | Symmetry is broken         |
| Hidden neurons become copies      | Neurons can specialize     |
| ❌ Not suitable for hidden weights | ✅ Common starting strategy |

---

# 🎤 Interview One-Liner

> **Zero initialization fails for hidden layers because identical weights produce identical outputs and gradients, causing identical updates and preventing neurons from learning different features. This is called the symmetry problem.**

---

# ⭐ Remember This

```text
ZERO
 ↓
SAME
 ↓
SAME
 ↓
SAME
 ↓
SYMMETRY
 ↓
❌
```

### 🔑 Golden Rule

> **Hidden neurons need different starting points so they can learn different features.**

---

### ➡️ Next

**Random Initialization**

> If zero initialization fails, can we simply use random weights?

The next challenge is deciding **how large those random weights should be**.
