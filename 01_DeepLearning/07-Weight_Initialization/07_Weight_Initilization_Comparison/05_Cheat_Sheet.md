```markdown id="weight-init-compare-cheatsheet"
# ⚡ Cheat Sheet — Weight Initialization Comparison

> **Core Idea:** First break symmetry, then choose a weight scale that matches the activation function and preserves healthy signal flow.

---

## 🎯 Main Goal of Weight Initialization

A good initialization should help with:

- Breaking symmetry
- Stable forward activations
- Stable backward gradients
- Reducing vanishing/exploding risk

---

# 🚫 Zero Initialization

All weights:

w = 0

Problem:

Same Weights  
↓  
Same Outputs  
↓  
Same Gradients  
↓  
Same Updates  
↓  
❌ Symmetry Problem

### Use

- ❌ Hidden-layer weights
- ✅ Biases are usually fine
- ✅ Can work in simple models like Logistic Regression

---

# 🎲 Random Initialization

Different random values:

+0.03  
-0.01  
+0.05  
-0.04

Main benefit:

> **Breaks symmetry**

But:

Randomness  
≠  
Correct Scale

Too Small  
↓  
Vanishing Risk

Too Large  
↓  
Exploding Risk

---

# 🔵 Xavier Initialization

Also called:

> **Glorot Initialization**

Common variance:

Var(W) = 2 / (fan_in + fan_out)

Best-known use:

- Tanh
- Sigmoid-style activations

Main idea:

fan_in  
↓  
Forward Signal

fan_out  
↓  
Backward Signal

Balance Both  
↓  
Xavier

---

# 🟢 He Initialization

Also called:

> **Kaiming Initialization**

Designed mainly for:

- ReLU
- Leaky ReLU
- ReLU-family activations

Common forward variance:

Var(W) = 2 / fan_in

Backward-preserving form:

Var(W) = 2 / fan_out

Why factor `2`?

ReLU  
↓  
Negative activations → 0  
↓  
Part of signal removed  
↓  
Need compensation  
↓  
Larger variance

---

# 🔢 `fan_in` vs `fan_out`

`fan_in`

= Number of incoming connections

`fan_out`

= Number of outgoing connections

Example:

100 inputs  
↓  
500 outputs

Then:

fan_in = 100  
fan_out = 500

---

# 🧮 Total Number of Weights

For a dense layer:

Total Weights  
=  
fan_in × fan_out

Example:

100 × 500  
=  
50,000 weights

Remember:

fan_in × fan_out  
→ **How many weights?**

Initialization formula  
→ **How large should the weights be?**

---

# ⚖️ Xavier vs He

| Xavier | He |
|---|---|
| Glorot | Kaiming |
| Tanh / Sigmoid-style | ReLU-family |
| `2 / (fan_in + fan_out)` | `2 / fan_in` |
| Balances forward/backward connectivity | Compensates for ReLU gating |
| Smaller variance for equal-width layers | Larger variance |

---

# 🧮 Equal-Width Example

Suppose:

fan_in = 100  
fan_out = 100

### Xavier

Var(W)

= 2 / 200

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

---

# 🎯 Activation → Initialization

| Activation | Good Starting Choice |
|---|---|
| Tanh | Xavier |
| Sigmoid | Xavier |
| ReLU | He |
| Leaky ReLU | He / Kaiming with suitable gain |

---

# 🔄 Forward vs Backward

### Xavier

Forward preference:

Var(W) ≈ 1 / fan_in

Backward preference:

Var(W) ≈ 1 / fan_out

Balanced using:

Var(W) = 2 / (fan_in + fan_out)

---

### He

Forward-preserving:

Var(W) = 2 / fan_in

Backward-preserving:

Var(W) = 2 / fan_out

---

# ⚠️ If `fan_in ≠ fan_out`

Example:

fan_in = 100  
fan_out = 500

He Forward:

2 / 100 = 0.02

He Backward:

2 / 500 = 0.004

So one variance cannot perfectly preserve both directions.

> Initialization gives a good starting condition, not a perfect guarantee.

---

# 🎲 Normal vs Uniform

## Xavier Normal

Var(W) = 2 / (fan_in + fan_out)

## Xavier Uniform

Range:

±sqrt(6 / (fan_in + fan_out))

## He Normal

Var(W) = 2 / fan_in

## He Uniform

Range:

±sqrt(6 / fan_in)

---

# 🧠 Quick Comparison

| Method | Breaks Symmetry | Controls Scale | Best Known Use |
|---|---:|---:|---|
| Zero | ❌ | ❌ | Biases |
| Naive Random | ✅ | ❌ | Basic symmetry breaking |
| Xavier | ✅ | ✅ | Tanh / Sigmoid |
| He | ✅ | ✅ | ReLU-family |

---

# 🚨 Important Traps

### ❌ Wrong

> Zero is the only bad identical initialization.

### ✅ Correct

Any identical hidden-weight initialization can create symmetry.

---

### ❌ Wrong

> Random initialization automatically solves everything.

### ✅ Correct

Randomness breaks symmetry, but scale still matters.

---

### ❌ Wrong

> He is always better because it has larger variance.

### ✅ Correct

Initialization should match the activation function.

---

### ❌ Wrong

> Xavier/He start closer to the optimal weights.

### ✅ Correct

They aim to provide healthy **signal and gradient scales**, not guess the final solution.

---

# 🧩 Practical Decision Flow

Hidden weights identical?  
↓  
YES  
↓  
❌ Fix symmetry first

Use random weights  
↓  
Is scale controlled?  
↓  
NO  
↓  
Vanishing / Exploding Risk

Check activation:

Tanh / Sigmoid  
↓  
Xavier

ReLU / Leaky ReLU  
↓  
He / Kaiming

---

# 🎤 Interview One-Liner

> **Zero initialization creates symmetry, random initialization breaks symmetry but may use a poor scale, Xavier uses `fan_in` and `fan_out` to control variance for tanh/sigmoid-style activations, and He uses a larger ReLU-aware variance, commonly `2 / fan_in`, for ReLU-family networks.**

---

# 🧠 Remember These 6 Things

1. Zero hidden weights → symmetry problem  
2. Random → breaks symmetry  
3. Random scale can still be wrong  
4. Xavier → `2 / (fan_in + fan_out)`  
5. He → `2 / fan_in` for forward ReLU scaling  
6. Match initialization to activation behavior

---

# ⭐ Golden Rule

> **Break symmetry first, then control the weight scale according to the activation function: Xavier for tanh-style activations and He for ReLU-style activations.**
```
