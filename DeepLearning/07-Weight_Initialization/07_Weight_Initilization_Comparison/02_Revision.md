# 📝 Revision — Weight Initialization Comparison

> **Core Idea:** The right initialization depends on the activation function and the signal scale we want to preserve. Zero fails because of symmetry, Random breaks symmetry but may use the wrong scale, Xavier is suited to tanh/sigmoid-style activations, and He is designed for ReLU-family activations.

---

## 🎯 1. Why Weight Initialization Matters

A good initialization should help with:

1. **Breaking symmetry**
2. **Maintaining healthy activation scale**
3. **Maintaining healthy gradient scale**

Conceptually:

Good Initialization  
↓  
Different Neurons  
+  
Stable Forward Signal  
+  
Stable Backward Signal  
↓  
Better Training

---

## 🚫 2. Zero Initialization

Zero Initialization means:

All weights = 0

This causes:

Same Weights  
↓  
Same Outputs  
↓  
Same Gradients  
↓  
Same Updates  
↓  
❌ Symmetry Problem

So hidden neurons remain identical.

### Important Exception

Zero initialization is usually fine for:

- Biases
- Simple models such as Logistic Regression

---

## 🎲 3. Random Initialization

Random Initialization gives different neurons different starting values.

Example:

+0.03  
-0.01  
+0.06  
-0.04

This solves:

> **Symmetry**

But random values can still have the wrong scale.

Too Small  
↓  
Vanishing Risk

Too Large  
↓  
Exploding Risk

Therefore:

Randomness alone is not enough.

---

## 📉 4. Why Scale Matters

During deep-network training:

Very Small Initial Scale  
↓  
Signals / Gradients may shrink  
↓  
Vanishing Gradient Risk

Very Large Initial Scale  
↓  
Signals / Gradients may grow  
↓  
Exploding Gradient Risk

So we need:

Randomness  
+  
Controlled Scale

This leads to:

- Xavier Initialization
- He Initialization

---

## 🔵 5. Xavier Initialization

Xavier is also called:

> **Glorot Initialization**

A common variance formula is:

Var(W) = 2 / (fan_in + fan_out)

Its goal is to keep forward activations and backward gradients at a reasonably stable scale.

It is commonly associated with:

- Tanh
- Sigmoid-style activations

---

## 🟢 6. He Initialization

He is also called:

> **Kaiming Initialization**

It is designed mainly for ReLU-family activations.

Common forward-preserving variance:

Var(W) = 2 / fan_in

Backward-preserving form:

Var(W) = 2 / fan_out

Why the larger variance?

Because ReLU sets negative activations to zero.

ReLU  
↓  
Part of Signal Removed  
↓  
Need Compensation  
↓  
He Initialization

---

## 🔢 7. `fan_in` and `fan_out`

### `fan_in`

Number of incoming connections to each output neuron.

### `fan_out`

Number of outgoing outputs from the layer.

Example:

100 inputs  
↓  
500 outputs

Then:

fan_in = 100  
fan_out = 500

---

## 🧮 8. Total Number of Weights

For a fully connected layer:

Total Weights  
=  
fan_in × fan_out

Example:

100 × 500  
=  
50,000 weights

Important:

fan_in × fan_out  
↓  
Parameter Count

Initialization Formula  
↓  
Parameter Scale

Do not confuse the two.

---

## ⚖️ 9. Xavier vs He — Formula

### Xavier

Var(W) = 2 / (fan_in + fan_out)

### He — Forward

Var(W) = 2 / fan_in

### He — Backward

Var(W) = 2 / fan_out

---

## 🧠 10. Why Does Xavier Use `fan_in + fan_out`?

Forward propagation approximately prefers:

Var(W) ≈ 1 / fan_in

Backward propagation approximately prefers:

Var(W) ≈ 1 / fan_out

Xavier balances both:

Var(W) = 2 / (fan_in + fan_out)

So:

fan_in  
↓  
Forward Scale

fan_out  
↓  
Backward Scale

Balance Both  
↓  
Xavier

---

## ✌️ 11. Why Does He Use the Factor `2`?

ReLU behaves as:

x > 0 → x  
x < 0 → 0

For a roughly symmetric pre-activation distribution, many negative values become zero.

So part of the signal is removed.

He compensates with a larger variance:

1 / fan_in  
↓  
ReLU Signal Reduction  
↓  
Compensate  
↓  
2 / fan_in

---

## 🔙 12. He and Backpropagation

ReLU derivative:

x > 0 → 1  
x < 0 → 0

So some backward gradient paths are also blocked.

For backward variance preservation:

Var(W) ≈ 2 / fan_out

Therefore:

fan_in  
↓  
Forward Activation Preservation

fan_out  
↓  
Backward Gradient Preservation

---

## ⚖️ 13. What If `fan_in = fan_out`?

Suppose:

fan_in = 100  
fan_out = 100

He forward:

2 / 100 = 0.02

He backward:

2 / 100 = 0.02

So both requirements are approximately the same.

---

## ⚠️ 14. What If `fan_in ≠ fan_out`?

Suppose:

fan_in = 100  
fan_out = 500

Forward He:

2 / 100 = 0.02

Backward He:

2 / 500 = 0.004

So one variance cannot perfectly preserve both directions.

This is why initialization should be viewed as:

> **A good starting condition, not a guarantee of perfect signal preservation.**

---

## 🧮 15. Xavier vs He Numerical Example

Suppose:

fan_in = 100  
fan_out = 100

### Xavier

Var(W)

= 2 / (100 + 100)

= 0.01

Std:

= 0.1

### He

Var(W)

= 2 / 100

= 0.02

Std:

≈ 0.141

Therefore:

He Variance  
≈  
2 × Xavier Variance

for equal-width layers.

---

## 🎯 16. Why Is He Larger Than Xavier?

Because Xavier does not specifically compensate for ReLU zeroing negative activations.

He does.

So:

Xavier  
↓  
Controlled Signal Scale

He  
↓  
Controlled Signal Scale  
+  
ReLU Compensation

---

## 🔵 17. Why Not Use He with Tanh Everywhere?

Tanh and Sigmoid can saturate when inputs become too large.

Larger Weight Scale  
↓  
Large Pre-Activations  
↓  
Saturation  
↓  
Small Derivatives  
↓  
Vanishing Gradient Risk

So larger variance is not automatically better.

---

## 🟢 18. Why Not Use Xavier with ReLU Everywhere?

ReLU removes negative activations.

If the scale is too small:

Layer  
↓  
ReLU Removes Part of Signal  
↓  
Next Layer  
↓  
ReLU Removes More  
↓  
Activations May Shrink

He is specifically designed to compensate for this behavior.

---

## 🎲 19. Normal vs Uniform

Both Xavier and He can use:

- Normal distributions
- Uniform distributions

The key idea is:

> Different distribution shapes can still target the same variance.

---

## 📐 20. Common Formulas

| Method | Common Formula |
|---|---|
| Xavier Normal | `Var = 2 / (fan_in + fan_out)` |
| Xavier Uniform | `±sqrt(6 / (fan_in + fan_out))` |
| He Normal | `Var = 2 / fan_in` |
| He Uniform | `±sqrt(6 / fan_in)` |

---

## 🧠 21. Zero vs Random vs Xavier vs He

| Method | Symmetry | Scale Control | Typical Use |
|---|---|---|---|
| Zero | ❌ Problem | ❌ | Avoid for hidden weights |
| Random | ✅ Broken | ❌ | Basic symmetry breaking |
| Xavier | ✅ | ✅ | Tanh / Sigmoid-style |
| He | ✅ | ✅ | ReLU-family |

---

## 🎯 22. Activation → Initialization Mapping

### Tanh

Think:

> **Xavier**

### Sigmoid

Think:

> **Xavier**

### ReLU

Think:

> **He / Kaiming**

### Leaky ReLU

Think:

> **He / Kaiming with suitable gain**

---

## 🧩 23. Bias Initialization

A common pattern is:

Weights → Xavier / He  
Biases → 0

Zero biases are generally fine because randomized weights already break neuron symmetry.

---

## 🚨 24. Identical Non-Zero Weights Are Still Bad

Suppose:

w₁ = 0.05  
w₂ = 0.05  
w₃ = 0.05

This is still identical initialization.

So the problem is not specifically:

Zero

The problem is:

> **Symmetry**

---

## ⚠️ 25. Does Good Initialization Solve Everything?

No.

Good initialization reduces poor signal-flow risk, but training can still require:

- Batch Normalization
- Suitable activation functions
- Gradient clipping
- Appropriate learning rate
- Appropriate optimizer
- Better architecture

Initialization is the starting point, not the entire optimization strategy.

---

## 🔄 26. Full Decision Flow

Hidden weights identical?  
↓  
Yes  
↓  
❌ Symmetry Problem

Use random values  
↓  
Symmetry Broken

Now ask:

Is the scale controlled?  
↓  
No  
↓  
Vanishing / Exploding Risk

Now check activation:

Tanh / Sigmoid  
↓  
Xavier

ReLU-family  
↓  
He

---

## 🎤 27. 30-Second Interview Answer

> **Zero initialization causes hidden-layer symmetry, while random initialization breaks symmetry but may use an inappropriate scale. Xavier Initialization uses a controlled variance, commonly `2 / (fan_in + fan_out)`, and is well suited to tanh or sigmoid-style activations. He Initialization is designed for ReLU-family activations and commonly uses `2 / fan_in` to compensate for ReLU zeroing negative activations. The overall goal is to maintain healthy activation and gradient scales at the start of training.**

---

## ⚡ Quick Mental Model

Zero  
↓  
Symmetry ❌

Random  
↓  
Symmetry Broken ✅  
But Scale Unknown

Xavier  
↓  
Controlled Scale  
↓  
Tanh / Sigmoid

He  
↓  
ReLU-Aware Scale  
↓  
ReLU / Leaky ReLU

---

## ⭐ Golden Rule

> **Choose initialization based on signal flow and activation behavior: Xavier for tanh-style activations, He for ReLU-style activations, and avoid identical hidden-weight initialization.**