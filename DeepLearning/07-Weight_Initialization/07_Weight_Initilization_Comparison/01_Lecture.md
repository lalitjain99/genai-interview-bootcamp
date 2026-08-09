```markdown id="v7m2cx"
# 📘 Lecture 07 — Weight Initialization Comparison

> **Core Idea:** There is no single initialization method that is best for every network. The right choice depends mainly on the activation function, layer structure, and the kind of signal flow we want to preserve.

---

# 🎯 Learning Objectives

By the end of this lecture, you will understand:

- How Zero, Random, Xavier, and He Initialization differ
- What problem each initialization method solves
- Which initialization is appropriate for which activation function
- How Xavier and He differ mathematically
- Why `fan_in` and `fan_out` matter
- How forward and backward variance influence initialization
- How to choose initialization in practical scenarios
- Common interview decision rules

---

# 1️⃣ Big Picture

We studied initialization in this progression:

Zero Initialization  
↓  
Symmetry Problem  
↓  
Random Initialization  
↓  
Symmetry Broken  
↓  
But Scale Can Be Wrong  
↓  
Vanishing / Exploding Gradients  
↓  
Need Controlled Randomness  
↓  
Xavier Initialization  
↓  
He Initialization

Each technique solves a different part of the initialization problem.

---

# 2️⃣ Why Initialization Matters

Before training begins, the network needs initial parameter values.

Good initialization should help with three major goals:

### 1. Break Symmetry

Different neurons should not start identically.

### 2. Maintain Healthy Activation Scale

Forward signals should not rapidly shrink or explode.

### 3. Maintain Healthy Gradient Scale

Backward gradients should remain usable across many layers.

Conceptually:

Good Initialization  
↓  
Symmetry Breaking  
+  
Healthy Forward Signal  
+  
Healthy Backward Signal  
↓  
More Stable Training

---

# 3️⃣ Zero Initialization

Zero Initialization means:

All weights = 0

Example:

w₁ = 0  
w₂ = 0  
w₃ = 0

The problem is that hidden neurons start identically.

Same Weights  
↓  
Same Outputs  
↓  
Same Gradients  
↓  
Same Updates  
↓  
Neurons Remain Identical

This is the:

> **Symmetry Problem**

---

## When Is Zero Initialization Acceptable?

Zero initialization is generally fine for:

- Bias parameters
- Some simple models such as Logistic Regression

But hidden-layer weights in a neural network should not all be initialized identically.

---

# 4️⃣ Random Initialization

Random Initialization gives different neurons different starting values.

Example:

+0.03  
-0.01  
+0.07  
-0.04

This solves:

> **Symmetry**

But arbitrary randomness introduces another problem:

What should the scale be?

If weights are too small:

Small Weights  
↓  
Signals Shrink  
↓  
Vanishing Gradient Risk

If weights are too large:

Large Weights  
↓  
Signals Grow  
↓  
Exploding Gradient Risk

So random initialization alone is incomplete.

---

# 5️⃣ Xavier Initialization

Xavier Initialization is also called:

> **Glorot Initialization**

It introduces controlled random scaling.

A common variance is:

Var(W) = 2 / (fan_in + fan_out)

The goal is to balance:

Forward activation variance

and:

Backward gradient variance

Xavier is commonly associated with:

- Tanh
- Sigmoid-style activations

---

# 6️⃣ He Initialization

He Initialization is also called:

> **Kaiming Initialization**

It is designed primarily for ReLU-family activations.

Common forward-preserving variance:

Var(W) = 2 / fan_in

Why larger than Xavier?

Because ReLU sets negative activations to zero.

ReLU  
↓  
Part of Signal Removed  
↓  
Need Compensation  
↓  
Larger Initial Variance

For backward gradient preservation, the corresponding condition is approximately:

Var(W) = 2 / fan_out

---

# 7️⃣ High-Level Comparison

| Method | Main Idea | Main Problem Solved |
|---|---|---|
| Zero | All weights identical | None for hidden layers |
| Random | Different random weights | Symmetry |
| Xavier | Controlled layer-aware random scale | Signal variance |
| He | ReLU-aware controlled random scale | ReLU signal reduction |

---

# 8️⃣ Activation Function Decision Rule

A very useful default rule is:

Tanh  
↓  
Xavier

Sigmoid  
↓  
Xavier

ReLU  
↓  
He

Leaky ReLU  
↓  
He / Kaiming with appropriate gain

This is not an absolute law, but it is a strong practical default.

---

# 9️⃣ Xavier vs He — Formula Comparison

### Xavier

Var(W) = 2 / (fan_in + fan_out)

### He

Forward-focused:

Var(W) = 2 / fan_in

Backward-focused:

Var(W) = 2 / fan_out

The main difference is:

Xavier  
↓  
Balances general forward/backward connectivity

He  
↓  
Adjusts for ReLU gating behavior

---

# 🔟 Equal-Width Layer Example

Suppose:

fan_in = 100  
fan_out = 100

### Xavier

Var(W)

= 2 / (100 + 100)

= 0.01

Std:

= sqrt(0.01)

= 0.1

### He

Var(W)

= 2 / 100

= 0.02

Std:

≈ 0.141

So for equal-width layers:

He Variance  
≈  
2 × Xavier Variance

Why?

Because He compensates for ReLU zeroing negative activations.

---

# 1️⃣1️⃣ Why Not Use He Everywhere?

At first, He may seem better because it uses a larger variance.

But larger is not automatically better.

For Tanh or Sigmoid, excessively large pre-activations can push neurons into saturation.

Conceptually:

Weights Too Large  
↓  
Large z Values  
↓  
Tanh / Sigmoid Saturation  
↓  
Small Derivatives  
↓  
Vanishing Gradient Risk

So initialization must match the activation behavior.

---

# 1️⃣2️⃣ Why Not Use Xavier Everywhere?

Xavier is not specifically designed for ReLU's zeroing behavior.

In a deep ReLU network:

Xavier Scale  
↓  
ReLU Removes Negative Signal  
↓  
Signal Can Shrink  
↓  
Repeated Across Layers  
↓  
Poorer Variance Preservation

He compensates for this using a larger variance.

---

# 1️⃣3️⃣ Parameter Count vs Initialization Scale

This distinction applies across the entire module.

For a fully connected layer:

Total Weights  
=  
fan_in × fan_out

But parameter count does not directly tell us the correct variance.

Example:

100 inputs  
↓  
500 outputs

Then:

fan_in = 100  
fan_out = 500

Total weights:

100 × 500  
=  
50,000

But Xavier might use:

2 / (100 + 500)

while He might use:

2 / 100

So:

fan_in × fan_out  
↓  
How many parameters?

Initialization Formula  
↓  
How large should those parameters initially be?

---

# 1️⃣4️⃣ Forward vs Backward Perspective

Initialization affects both directions.

## Forward Pass

We want:

Activation Variance  
≈  
Reasonably Stable

If it continually shrinks:

Vanishing signal

If it continually grows:

Exploding signal

---

## Backward Pass

We want:

Gradient Variance  
≈  
Reasonably Stable

If it continually shrinks:

Vanishing gradients

If it continually grows:

Exploding gradients

---

# 1️⃣5️⃣ How Xavier Thinks About Forward and Backward Flow

For simple symmetric activations:

Forward preference:

Var(W) ≈ 1 / fan_in

Backward preference:

Var(W) ≈ 1 / fan_out

Xavier balances them using:

Var(W) = 2 / (fan_in + fan_out)

So:

fan_in  
↓  
Forward consideration

fan_out  
↓  
Backward consideration

Balance both  
↓  
Xavier

---

# 1️⃣6️⃣ How He Thinks About Forward and Backward Flow

For ReLU:

Forward preservation approximately prefers:

Var(W) = 2 / fan_in

Backward preservation approximately prefers:

Var(W) = 2 / fan_out

The factor `2` appears because ReLU gates part of the signal.

So:

ReLU Forward  
↓  
2 / fan_in

ReLU Backward  
↓  
2 / fan_out

---

# 1️⃣7️⃣ What If `fan_in ≈ fan_out`?

Suppose:

fan_in ≈ fan_out

Then forward and backward preservation conditions become similar.

Example:

fan_in = 256  
fan_out = 256

He forward:

2 / 256

He backward:

2 / 256

Same value.

This makes signal preservation easier.

---

# 1️⃣8️⃣ What If Layer Width Changes Drastically?

Suppose:

fan_in = 100  
fan_out = 1000

Then:

He forward variance:

2 / 100  
=  
0.02

He backward variance:

2 / 1000  
=  
0.002

These are very different.

One single variance cannot perfectly preserve both directions.

That is why:

> Initialization improves signal flow; it does not guarantee perfect variance preservation in every architecture.

---

# 1️⃣9️⃣ Normal vs Uniform Initialization

Both Xavier and He can use:

- Normal distributions
- Uniform distributions

The important concept is:

> The distribution shape can differ while the target variance remains appropriate.

---

## Xavier Normal

Var(W) = 2 / (fan_in + fan_out)

---

## Xavier Uniform

W ∈ [-a, +a]

where:

a = sqrt(6 / (fan_in + fan_out))

---

## He Normal

Var(W) = 2 / fan_in

---

## He Uniform

For standard ReLU forward scaling:

a = sqrt(6 / fan_in)

---

# 2️⃣0️⃣ Comparison of Common Formulas

| Method | Common Scale |
|---|---|
| Naive Random | Arbitrary |
| Xavier Normal | `Var = 2 / (fan_in + fan_out)` |
| Xavier Uniform | `±sqrt(6 / (fan_in + fan_out))` |
| He Normal | `Var = 2 / fan_in` |
| He Uniform | `±sqrt(6 / fan_in)` |

---

# 2️⃣1️⃣ Scenario — Tanh Network

Suppose you build:

Input  
↓  
Dense Layer  
↓  
Tanh  
↓  
Dense Layer  
↓  
Tanh

Which initialization?

### Recommended Starting Choice

> **Xavier**

Why?

Tanh benefits from keeping its pre-activation values reasonably scaled.

Xavier helps prevent values from immediately entering heavily saturated regions.

---

# 2️⃣2️⃣ Scenario — Deep ReLU Network

Suppose:

Input  
↓  
Dense  
↓  
ReLU  
↓  
Dense  
↓  
ReLU  
↓  
Dense  
↓  
ReLU

Which initialization?

### Recommended Starting Choice

> **He / Kaiming**

Why?

ReLU removes negative activations.

He compensates for that variance reduction.

---

# 2️⃣3️⃣ Scenario — Leaky ReLU

Suppose:

Input  
↓  
Dense  
↓  
Leaky ReLU

The natural initialization family is:

> **He / Kaiming**

But the ideal gain should account for the negative slope.

So the exact scaling can be slightly different from standard ReLU.

---

# 2️⃣4️⃣ Scenario — Bias Initialization

Suppose the weights use Xavier or He.

Can biases be zero?

Usually:

> **Yes**

Typical pattern:

Weights → Xavier / He  
Biases → Zero

Zero bias does not create the same hidden-neuron symmetry problem when the weights are already different.

---

# 2️⃣5️⃣ Scenario — All Weights Same Random Number

Suppose someone says:

> "I'll initialize every weight to 0.05. It isn't zero, so symmetry is broken."

Incorrect.

All weights are still identical.

The problem is not specifically:

Zero

The problem is:

Identical Initialization

So:

All weights = 0.05  
↓  
Still Symmetric  
↓  
Bad for hidden neurons

---

# 2️⃣6️⃣ Scenario — Random Values Between -100 and +100

Symmetry?

✅ Broken

Good initialization?

❌ Probably not

Why?

Scale is too large.

So:

Randomness  
≠  
Correct Initialization

We need:

Randomness  
+  
Appropriate Scale

---

# 2️⃣7️⃣ Scenario — Random Values Around 0.0000001

Symmetry?

✅ Broken

Good initialization?

Not necessarily.

The scale may be too small.

Deep network  
↓  
Signals shrink  
↓  
Gradients shrink  
↓  
Vanishing risk

---

# 2️⃣8️⃣ Practical Selection Table

| Situation | Good Starting Choice |
|---|---|
| Hidden weights all identical | ❌ Avoid |
| Need symmetry breaking only | Random |
| Tanh hidden layers | Xavier |
| Sigmoid hidden layers | Xavier |
| ReLU hidden layers | He |
| Leaky ReLU | He/Kaiming with suitable gain |
| Bias parameters | Often zero |

---

# 2️⃣9️⃣ Problem → Initialization Mapping

### Symmetry Problem

Think:

> Random Initialization

---

### Tanh/Sigmoid Signal Scaling

Think:

> Xavier

---

### ReLU Signal Scaling

Think:

> He

---

### Exploding/Vanishing Due to Bad Initial Scale

Think:

> Principled initialization such as Xavier/He

---

# 3️⃣0️⃣ Does Initialization Replace Other Stabilization Techniques?

No.

Good initialization helps provide a healthy starting point.

But deep-network training may still use:

- Batch Normalization
- Suitable activation functions
- Gradient clipping
- Appropriate optimizers
- Appropriate learning rates
- Architectural techniques

Initialization is one part of stable training.

---

# 3️⃣1️⃣ Does the Optimizer Fix Bad Initialization Automatically?

Not necessarily.

The optimizer updates parameters after training begins.

But poor initialization can cause:

Vanishing gradients  
↓  
Optimizer receives almost no useful signal

or:

Exploding gradients  
↓  
Optimizer receives unstable signal

So a good optimizer does not make initialization irrelevant.

---

# 3️⃣2️⃣ Does Batch Normalization Make Initialization Irrelevant?

No.

Batch Normalization can make networks less sensitive to some initialization choices, but initialization still matters.

Good initialization provides:

Healthy Initial Signal Flow

Batch Normalization provides:

Additional Training Stability

They can work together.

---

# 3️⃣3️⃣ Does Good Initialization Mean Starting Near the Optimum?

No.

This is an important misconception.

Xavier and He do not know:

- What the final weights should be
- Where the optimum is
- What exact features neurons should learn

Their goal is:

> **Provide a healthy starting scale for learning.**

---

# 3️⃣4️⃣ Quick Decision Tree

Use this mental model:

What activation function is used?

Tanh / Sigmoid  
↓  
Xavier

ReLU  
↓  
He

Leaky ReLU  
↓  
He/Kaiming with adjusted gain

No activation information?  
↓  
Inspect architecture and activation before deciding

And always:

Avoid identical hidden-weight initialization

---

# 3️⃣5️⃣ Full Comparison

| Feature | Zero | Naive Random | Xavier | He |
|---|---|---|---|---|
| Random weights | ❌ | ✅ | ✅ | ✅ |
| Breaks symmetry | ❌ | ✅ | ✅ | ✅ |
| Controls scale | ❌ | ❌ | ✅ | ✅ |
| Layer-aware | ❌ | ❌ | ✅ | ✅ |
| ReLU-aware | ❌ | ❌ | ❌ specifically | ✅ |
| Tanh-friendly | ❌ | Maybe | ✅ | Not default |
| ReLU-friendly | ❌ | Maybe | Not preferred | ✅ |
| Main formula | `0` | Arbitrary | `2/(fan_in+fan_out)` | `2/fan_in` |

---

# 🧠 Complete Mental Model

Initialization Problem
        ↓
Are all hidden weights identical?
        ↓
YES
        ↓
❌ Symmetry Problem
        ↓
Use random values

Now ask:

Is the scale controlled?
        ↓
NO
        ↓
Vanishing / Exploding Risk

Now inspect activation:

Tanh / Sigmoid
        ↓
Xavier
        ↓
2 / (fan_in + fan_out)

ReLU-family
        ↓
He
        ↓
2 / fan_in
        ↓
ReLU-aware signal preservation
```

---

# 🎤 Interview Perspective

If an interviewer asks:

> **How do you choose between Xavier and He Initialization?**

A strong answer is:

> **I choose the initializer mainly based on the activation function and the signal variance I want to preserve. Xavier uses a common variance of `2 / (fan_in + fan_out)` and is typically suited to tanh or sigmoid-style activations. He Initialization uses a common forward variance of `2 / fan_in` and is designed for ReLU-family activations because ReLU zeroes negative activations and reduces signal variance. Both methods use controlled random weights to reduce vanishing and exploding gradient risk.**

---

# 🎯 30-Second Module Answer

> **Zero initialization fails for hidden weights because it creates symmetry. Random initialization breaks symmetry but may use the wrong scale. Xavier controls the random scale using `fan_in` and `fan_out` and is commonly used for tanh/sigmoid networks. He Initialization adapts the variance for ReLU networks using `2 / fan_in`, compensating for ReLU zeroing negative activations. The goal of all principled initialization methods is healthy signal and gradient flow, not starting close to the final solution.**

---

# 📌 Key Takeaways

* Zero hidden-weight initialization creates symmetry
* Random initialization breaks symmetry
* Random scale can still be too small or too large
* Xavier controls variance using `fan_in` and `fan_out`
* Xavier is commonly suited to tanh/sigmoid-style activations
* He is designed for ReLU-family activations
* ReLU motivates the factor `2` in He scaling
* `fan_in × fan_out` counts weights
* Initialization formulas determine weight scale
* `fan_in` relates strongly to forward signal flow
* `fan_out` relates strongly to backward signal flow
* He `fan_in` mode focuses on forward activation preservation
* He `fan_out` mode focuses on backward gradient preservation
* No initializer guarantees perfect gradient flow forever
* Initialization should match the activation and architecture

---

# ⭐ Golden Rule

> **Choose initialization based on signal flow, not simply on parameter count. Xavier is the natural default for tanh-style activations, while He is the natural default for ReLU-style activations.**

---

# ➡️ Next Step

After completing the revision, interview, assignment, cheat sheet, and common mistakes for this comparison topic, we will finish the entire module with:

# ⭐ Weight Initialization — Module Summary

That will connect:

Zero
→ Random
→ Vanishing/Exploding
→ Xavier
→ He

into one final interview-ready mental model.

```
```
