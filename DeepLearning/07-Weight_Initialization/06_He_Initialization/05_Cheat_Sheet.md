```markdown
# ⚡ Cheat Sheet — He Initialization

> **Core Idea:** He Initialization is a controlled random initialization method designed mainly for **ReLU-family activations**. It compensates for the signal reduction caused by ReLU.

---

## 🎯 What Is He Initialization?

He Initialization is also called:

- **Kaiming Initialization**
- **He/Kaiming Initialization**

Common forward-preserving formula:

Var(W) = 2 / fan_in

Standard deviation:

Std(W) = sqrt(2 / fan_in)

---

## 🟢 Why He for ReLU?

ReLU:

x > 0 → x  
x < 0 → 0

So negative activations are removed.

Conceptually:

Input Signal  
↓  
ReLU  
↓  
Negative Values → 0  
↓  
Part of Signal Lost

He compensates by using a larger weight variance.

---

## ✌️ Why the Factor `2`?

Without considering ReLU:

Var(W) ≈ 1 / fan_in

But ReLU removes roughly half of a symmetric signal.

So:

1 / fan_in  
↓  
Compensate for ReLU  
↓  
2 / fan_in

Think:

ReLU removes part of signal  
↓  
Increase initialization variance  
↓  
Healthier activation scale

---

## 🔢 What Is `fan_in`?

`fan_in` = number of inputs coming into each output neuron.

Example:

100 inputs  
↓  
50 output neurons

Then:

fan_in = 100  
fan_out = 50

Total weights:

100 × 50 = 5000

---

## 🧮 Number of Weights vs He Variance

For a fully connected layer:

fan_in × fan_out  
↓  
Total number of weights

But:

2 / fan_in  
↓  
Forward-preserving He variance

So:

100 × 50 = 5000 weights

but:

Var(W) = 2 / 100 = 0.02

Then all 5000 weights are randomly sampled using that scale.

---

# 🔄 Forward Propagation

For one neuron:

z = w₁x₁ + w₂x₂ + ... + wₙxₙ

Approximately:

Var(z)
≈
fan_in × Var(W) × Var(x)

After ReLU:

Var(output)
≈
1/2 × fan_in × Var(W) × Var(input)

To preserve variance:

1/2 × fan_in × Var(W) ≈ 1

Therefore:

Var(W) ≈ 2 / fan_in

---

# 🔙 Backpropagation

ReLU derivative:

x > 0 → 1  
x < 0 → 0

So approximately half the gradient paths may be blocked.

Backward variance is approximately:

Var(gradient_previous)
≈
fan_out × Var(W) × 1/2 × Var(gradient_next)

To preserve gradient variance:

Var(W) ≈ 2 / fan_out

---

# ⚖️ `fan_in` vs `fan_out`

| Mode | Formula | Main Goal |
|---|---|---|
| `fan_in` | `2 / fan_in` | Preserve forward activation scale |
| `fan_out` | `2 / fan_out` | Preserve backward gradient scale |

Mental model:

fan_in  
↓  
Forward Pass

fan_out  
↓  
Backward Pass

---

# 🧠 If `fan_in = fan_out`

Example:

fan_in = 100  
fan_out = 100

Forward:

2 / 100 = 0.02

Backward:

2 / 100 = 0.02

So both requirements are approximately the same.

---

# ⚠️ If `fan_in ≠ fan_out`

Example:

fan_in = 100  
fan_out = 500

Forward-preserving variance:

2 / 100 = 0.02

Backward-preserving variance:

2 / 500 = 0.004

So one variance cannot perfectly preserve both directions.

---

# 🎲 He Normal

For He Normal:

Mean = 0

Variance:

2 / fan_in

Standard deviation:

sqrt(2 / fan_in)

Example:

fan_in = 100

Variance = 0.02  
Std ≈ 0.141

---

# 🎲 He Uniform

For He Uniform:

W ~ Uniform(-a, +a)

where:

a = sqrt(6 / fan_in)

Example:

fan_in = 100

a ≈ 0.245

So:

Weights ∈ [-0.245, +0.245]

---

# ⚖️ He Normal vs He Uniform

| He Normal | He Uniform |
|---|---|
| Normal distribution | Uniform distribution |
| Std = `sqrt(2 / fan_in)` | Range = `±sqrt(6 / fan_in)` |
| No hard min/max | Fixed range |
| Same variance goal | Same variance goal |

---

# 🔵 Xavier vs He

| Xavier | He |
|---|---|
| Glorot Initialization | Kaiming Initialization |
| Common with tanh / sigmoid | Designed for ReLU-family |
| `2 / (fan_in + fan_out)` | `2 / fan_in` |
| Balances forward/backward connectivity | ReLU-aware scaling |
| Smaller variance for equal-width layers | Larger variance for equal-width layers |

---

# 🧮 Quick Example — Xavier vs He

Suppose:

fan_in = 100  
fan_out = 100

### Xavier

Var(W)

= 2 / (100 + 100)

= 0.01

### He

Var(W)

= 2 / 100

= 0.02

So:

He Variance  
≈  
2 × Xavier Variance

for equal-width layers.

Why?

> **He compensates for ReLU zeroing negative activations.**

---

# 🟢 Activation Selection

Quick rule:

Tanh / Sigmoid  
↓  
Xavier

ReLU  
↓  
He

Leaky ReLU  
↓  
He / Kaiming with appropriate gain

---

# 🚨 Important Interview Traps

### ❌ Wrong

> `2 / fan_in` is the actual value of every weight.

### ✅ Correct

It is the **variance of the weight distribution**.

---

### ❌ Wrong

> He completely prevents vanishing gradients.

### ✅ Correct

He provides a healthier starting scale and **reduces the risk**.

---

### ❌ Wrong

> `fan_in × fan_out` is used in the He variance formula.

### ✅ Correct

`fan_in × fan_out` gives the **number of weights**.

---

# 🔗 Initialization Flow

Zero Initialization  
↓  
Symmetry Problem  
↓  
Random Initialization  
↓  
Symmetry Broken  
↓  
Scale Problem  
↓  
Xavier Initialization  
↓  
Good controlled scale  
↓  
But ReLU removes negative activations  
↓  
He Initialization  
↓  
ReLU-Aware Scale

---

# 🎤 Interview One-Liner

> **He Initialization, also called Kaiming Initialization, is designed mainly for ReLU-family activations. It compensates for ReLU zeroing negative activations by using a larger weight variance, commonly `2 / fan_in`, to preserve forward activation scale. A `2 / fan_out` form can be used when backward gradient preservation is the priority.**

---

# 🧠 Remember These 5 Things

1. He = Kaiming
2. Mainly for ReLU-family activations
3. Forward variance = `2 / fan_in`
4. Backward variance = `2 / fan_out`
5. Factor `2` compensates for ReLU gating

---

# ⭐ Golden Rule

> **He Initialization is ReLU-aware controlled randomness: `fan_in` helps preserve forward activations, `fan_out` helps preserve backward gradients, and the factor `2` compensates for ReLU removing part of the signal.**
```
