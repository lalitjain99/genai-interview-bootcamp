# 💡 Assignment — Weight Initialization Comparison

> **Focus:** Choose the correct initialization based on symmetry, weight scale, activation function, and forward/backward signal flow.

---

## 📌 Assignment 1 — Scenario Based

### 🧠 Scenario A — Identical Hidden Weights

You have a neural network with 100 hidden neurons.

A developer initializes every hidden weight to:

0.05

### Questions

### 1️⃣ Is this better than Zero Initialization?

❌ **Answer: No**

The value is non-zero, but all hidden neurons still start identically.

So:

Same Weights  
↓  
Same Behavior  
↓  
Same Gradients  
↓  
Same Updates  
↓  
Symmetry Problem

The real issue is not specifically zero.

It is:

> **Identical hidden-weight initialization**

---

### 2️⃣ What should be done instead?

✅ **Answer**

Use different random starting weights.

For example:

- Random Initialization
- Xavier Initialization
- He Initialization

depending on the activation function and network structure.

---

## 🎲 Scenario B — Random but Poorly Scaled

A developer initializes all weights randomly between:

-100 and +100

### 1️⃣ Is symmetry broken?

✅ **Answer**

Yes.

The weights are different, so neurons do not start identically.

---

### 2️⃣ Is this necessarily a good initialization?

❌ **Answer: No**

The values may be far too large.

This can cause:

Large Activations  
↓  
Large Gradients  
↓  
Exploding Gradient Risk

So:

Randomness  
≠  
Good Initialization

We need:

Randomness  
+  
Appropriate Scale

---

## 🔵 Scenario C — Tanh Network

You are building a deep network with:

Activation = Tanh

Layer:

fan_in = 200  
fan_out = 200

### 1️⃣ Which initialization would you choose?

✅ **Answer**

**Xavier Initialization**

Because Xavier is commonly suited to `tanh` and sigmoid-style activations.

---

### 2️⃣ Calculate the Xavier variance.

✅ **Answer**

Formula:

Var(W)  
=  
2 / (fan_in + fan_out)

So:

Var(W)  
=  
2 / (200 + 200)

=  
2 / 400

=  
0.005

---

### 3️⃣ Why not simply use He because it gives a larger variance?

✅ **Answer**

Larger is not automatically better.

For Tanh, overly large pre-activations can push the activation into saturation.

That can produce very small derivatives and increase vanishing-gradient risk.

So the initializer should match the activation behavior.

---

## 🟢 Scenario D — ReLU Network

Now consider:

Activation = ReLU

fan_in = 200  
fan_out = 200

### 1️⃣ Which initialization would you choose?

✅ **Answer**

**He Initialization**

because He is designed for ReLU-family activations.

---

### 2️⃣ Calculate the common forward-preserving He variance.

✅ **Answer**

Var(W)  
=  
2 / fan_in

=  
2 / 200

=  
0.01

---

### 3️⃣ Compare it with Xavier.

✅ **Answer**

Xavier:

Var(W) = 0.005

He:

Var(W) = 0.01

So:

He Variance  
=  
2 × Xavier Variance

for this equal-width layer.

The larger He variance compensates for ReLU zeroing negative activations.

---

## 🔙 Scenario E — Forward vs Backward Preservation

You have a ReLU layer:

fan_in = 100  
fan_out = 500

### 1️⃣ What is the forward-preserving He variance?

✅ **Answer**

Var(W)  
=  
2 / fan_in

=  
2 / 100

=  
0.02

---

### 2️⃣ What is the backward-preserving He variance?

✅ **Answer**

Var(W)  
=  
2 / fan_out

=  
2 / 500

=  
0.004

---

### 3️⃣ Why are they different?

✅ **Answer**

Because:

fan_in ≠ fan_out

Forward signal preservation depends mainly on incoming connectivity.

Backward gradient preservation depends mainly on outgoing connectivity.

So one variance cannot perfectly preserve both directions when layer widths are very different.

---

## 🧮 Scenario F — Parameter Count vs Weight Scale

Consider a dense layer:

fan_in = 50  
fan_out = 100

### 1️⃣ How many weights are present?

✅ **Answer**

Total Weights  
=  
fan_in × fan_out

=  
50 × 100

=  
5000

---

### 2️⃣ Does `5000` directly determine Xavier or He variance?

❌ **Answer: No**

The total parameter count tells us:

> **How many weights exist**

It does not directly tell us:

> **How large those weights should be**

For Xavier:

Var(W)  
=  
2 / (50 + 100)

For He forward mode:

Var(W)  
=  
2 / 50

So:

fan_in × fan_out  
→ Parameter Count

Initialization Formula  
→ Parameter Scale

---

# 📌 Assignment 2 — True / False

| # | Statement | Answer |
|---|---|---|
| 1 | Zero initialization of all hidden weights can create a symmetry problem. | ✅ True |
| 2 | Initializing every hidden weight to the same non-zero value breaks symmetry. | ❌ False |
| 3 | Random Initialization breaks symmetry but does not automatically guarantee proper scaling. | ✅ True |
| 4 | Xavier is commonly associated with Tanh and Sigmoid-style activations. | ✅ True |
| 5 | He Initialization is designed mainly for ReLU-family activations. | ✅ True |
| 6 | `fan_in × fan_out` gives the number of weights in a dense layer. | ✅ True |
| 7 | Xavier variance is commonly `2 / (fan_in + fan_out)`. | ✅ True |
| 8 | He forward-preserving variance is commonly `2 / fan_in`. | ✅ True |
| 9 | A larger initialization variance is always better. | ❌ False |
| 10 | Good initialization guarantees gradients can never vanish or explode. | ❌ False |

---

# ⭐ Staff Engineer Challenge

## Scenario

You are reviewing the following architecture:

Input  
↓  
Dense 512  
↓  
Tanh  
↓  
Dense 1024  
↓  
ReLU  
↓  
Dense 256  
↓  
ReLU

A junior engineer says:

> "Let's use the same Xavier Initialization for every layer because consistency is easier."

### 1️⃣ Would you approve this?

❌ **Answer: Not as the default choice.**

The initialization should match the activation behavior of each layer.

---

### 2️⃣ What would you choose for the Tanh layer?

✅ **Answer**

For the layer followed by Tanh:

> **Xavier Initialization**

because Xavier is commonly suitable for Tanh-style activations.

---

### 3️⃣ What would you choose for the ReLU layers?

✅ **Answer**

For layers followed by ReLU:

> **He / Kaiming Initialization**

because He compensates for ReLU zeroing negative activations.

---

### 4️⃣ Why shouldn't one initializer automatically be used everywhere?

✅ **Answer**

Different activation functions affect signal variance differently.

Tanh:

Large inputs  
↓  
Can Saturate  
↓  
Small Derivatives

ReLU:

Negative inputs  
↓  
Become 0  
↓  
Part of Signal Removed

Therefore:

Different Activation Behavior  
↓  
Different Initialization Requirement

---

### 5️⃣ What principle would you explain to the team?

✅ **Answer**

I would explain:

> **Choose initialization according to signal propagation, not convenience or parameter count.**

A good practical default is:

Tanh / Sigmoid  
→ Xavier

ReLU / Leaky ReLU  
→ He / Kaiming

---

# 🎯 Final Challenge

Complete the initialization flow:

All Hidden Weights Identical  
↓  
________________________

Different Random Weights  
↓  
________________________

Random Scale Too Small / Large  
↓  
________________________

Tanh / Sigmoid  
↓  
________________________

ReLU  
↓  
________________________

### ✅ Answer

All Hidden Weights Identical  
↓  
Symmetry Problem

Different Random Weights  
↓  
Symmetry Broken

Random Scale Too Small / Large  
↓  
Vanishing / Exploding Gradient Risk

Tanh / Sigmoid  
↓  
Xavier Initialization

ReLU  
↓  
He Initialization

---

# 🧠 Final Mental Model

Zero / Identical  
↓  
❌ Symmetry

Random  
↓  
✅ Symmetry Broken  
↓  
But Scale Unknown

Controlled Initialization  
↓  

Tanh / Sigmoid  
→ Xavier

ReLU / Leaky ReLU  
→ He

Remember:

fan_in × fan_out  
→ How many weights?

Xavier / He formulas  
→ How large should those weights initially be?

---

# ⭐ Golden Rule

> **First break symmetry, then choose a scale that matches the activation function and signal-flow behavior of the network.**