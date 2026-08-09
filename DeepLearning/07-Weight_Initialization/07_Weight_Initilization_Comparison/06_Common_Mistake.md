# 🚫 Common Mistakes — Weight Initialization Comparison

> **Core Idea:** Most initialization mistakes come from confusing **symmetry**, **weight count**, **weight scale**, and **activation-function compatibility**.

---

## ❌ Mistake 1: Thinking only Zero Initialization causes symmetry

### Wrong ❌

> Symmetry happens only when all weights are zero.

### Correct ✅

The real issue is **identical initialization**.

For example:

w₁ = 0.05  
w₂ = 0.05  
w₃ = 0.05

can still create symmetry.

So remember:

Identical Hidden Weights  
↓  
Same Behavior  
↓  
Same Gradients  
↓  
Same Updates  
↓  
❌ Symmetry Problem

---

## ❌ Mistake 2: Thinking Random Initialization automatically solves everything

### Wrong ❌

> As long as the weights are random, initialization is good.

### Correct ✅

Randomness only guarantees:

> **Symmetry breaking**

It does not guarantee a healthy scale.

Too Small  
↓  
Vanishing Risk

Too Large  
↓  
Exploding Risk

So:

Randomness  
+  
Appropriate Scale  
↓  
Good Initialization

---

## ❌ Mistake 3: Confusing parameter count with initialization scale

### Wrong ❌

> The more weights a layer has, the smaller each weight should be based directly on total weight count.

### Correct ✅

For a fully connected layer:

fan_in × fan_out  
↓  
Total Number of Weights

But Xavier and He reason about:

> **Signal flow per neuron**

So:

fan_in  
↓  
Forward connectivity

fan_out  
↓  
Backward connectivity

Parameter count and parameter scale are different concepts.

---

## ❌ Mistake 4: Thinking Xavier should use `fan_in × fan_out`

### Wrong ❌

> Xavier should divide by the total number of weights.

### Correct ✅

Xavier tries to balance:

Forward preference:

Var(W) ≈ 1 / fan_in

Backward preference:

Var(W) ≈ 1 / fan_out

So a common Xavier variance is:

Var(W) = 2 / (fan_in + fan_out)

The product:

fan_in × fan_out

is used for parameter counting, not Xavier scaling.

---

## ❌ Mistake 5: Thinking He is simply a larger version of Xavier

### Wrong ❌

> He is basically Xavier with bigger numbers.

### Correct ✅

He has a specific reason for using a larger variance.

ReLU:

Negative Activations  
↓  
0  
↓  
Part of Signal Removed

So He compensates for ReLU's gating behavior.

Common forward He:

Var(W) = 2 / fan_in

So He is:

> **ReLU-aware initialization**

not simply "bigger Xavier."

---

## ❌ Mistake 6: Thinking larger initialization variance is always better

### Wrong ❌

> Bigger weights help gradients flow better.

### Correct ✅

Large weights can create:

Large Activations  
↓  
Saturation / Instability  
↓  
Large or Poor Gradients

For Tanh and Sigmoid, large pre-activations can push the activation into saturation.

So larger is not automatically better.

The correct scale depends on the activation function.

---

## ❌ Mistake 7: Using He for every activation function

### Wrong ❌

> He is the best initializer for every network.

### Correct ✅

A useful default rule is:

Tanh / Sigmoid  
→ Xavier

ReLU / Leaky ReLU  
→ He / Kaiming

Initialization should match the activation behavior.

---

## ❌ Mistake 8: Using Xavier for every activation function

### Wrong ❌

> Xavier works everywhere, so there is no need for He.

### Correct ✅

Xavier does not specifically compensate for ReLU zeroing negative activations.

In deep ReLU networks, He is generally the more appropriate default.

---

## ❌ Mistake 9: Thinking `2 / fan_in` is the actual He weight value

### Wrong ❌

Suppose:

fan_in = 100

Then:

2 / fan_in = 0.02

So every weight should be `0.02`.

### Correct ✅

`0.02` is the **variance** of the weight distribution.

For He Normal:

Std(W)
=
sqrt(0.02)

≈
0.141

Then different random weights are sampled.

---

## ❌ Mistake 10: Thinking Xavier's formula gives the actual weight values

### Wrong ❌

> Xavier calculates one number and assigns that value to all weights.

### Correct ✅

Xavier calculates the **scale of a random distribution**.

Different weights are still randomly sampled.

So Xavier still provides:

✅ Symmetry breaking  
✅ Controlled scale

---

## ❌ Mistake 11: Thinking `fan_in` and `fan_out` always play the same role

### Wrong ❌

> `fan_in` and `fan_out` are interchangeable.

### Correct ✅

Conceptually:

fan_in  
↓  
Strongly related to forward activation scale

fan_out  
↓  
Strongly related to backward gradient scale

This distinction is especially important when layer widths differ significantly.

---

## ❌ Mistake 12: Thinking He `2 / fan_in` perfectly preserves backward gradients

### Wrong ❌

> `2 / fan_in` guarantees both forward and backward variance are preserved.

### Correct ✅

The common He formula:

Var(W) = 2 / fan_in

primarily targets:

> **Forward activation variance**

A backward-preserving condition is approximately:

Var(W) = 2 / fan_out

If:

fan_in ≈ fan_out

both requirements are similar.

If they differ greatly, one variance cannot perfectly satisfy both.

---

## ❌ Mistake 13: Thinking initialization completely eliminates Vanishing and Exploding Gradients

### Wrong ❌

> Xavier or He guarantees gradients will never vanish or explode.

### Correct ✅

Good initialization reduces the risk by providing healthy starting conditions.

But gradient problems can still occur because of:

- Network depth
- Activation functions
- Training dynamics
- Architecture
- Learning rate
- Other optimization issues

So say:

✅ Reduces risk

not:

❌ Guarantees elimination

---

## ❌ Mistake 14: Thinking Batch Normalization makes initialization unnecessary

### Wrong ❌

> If BatchNorm is present, initialization does not matter.

### Correct ✅

Batch Normalization can make training less sensitive to some initialization choices, but initialization still determines the initial signal scale.

They solve different parts of the problem.

Good Initialization  
+  
Batch Normalization  
↓  
More Stable Optimization

---

## ❌ Mistake 15: Thinking a good initializer starts close to the optimum

### Wrong ❌

> Xavier and He work because they guess weights close to the final solution.

### Correct ✅

Neither Xavier nor He knows the final optimal weights.

Their goal is:

> **Healthy signal and gradient propagation at the beginning of training.**

Think:

Stable Starting Conditions

not:

Close to Final Solution

---

# 🧠 Don't Confuse These Concepts

| Concept | Meaning |
|---|---|
| Identical weights | Can cause symmetry |
| Random weights | Break symmetry |
| `fan_in × fan_out` | Total number of weights |
| Xavier | Controlled scale for tanh/sigmoid-style activations |
| He | ReLU-aware controlled scale |
| `fan_in` | Forward connectivity |
| `fan_out` | Backward connectivity |
| Variance | Controls distribution spread |

---

# ⚡ Quick Mental Model

Zero / Identical  
↓  
❌ Symmetry

Random  
↓  
✅ Symmetry Broken  
↓  
But Scale May Be Wrong

Controlled Initialization  
↓  

Tanh / Sigmoid  
→ Xavier

ReLU / Leaky ReLU  
→ He

---

# 🎯 Interview Trap

### Weak Answer ❌

> "Use Xavier for small networks and He for large networks."

This is incorrect.

### Strong Answer ✅

> **The choice is mainly driven by activation behavior and signal propagation. Xavier is commonly used with tanh/sigmoid-style activations, while He is designed for ReLU-family activations because it compensates for ReLU zeroing negative activations.**

---

# ⭐ Golden Rule

> **Do not choose an initializer because it is newer, larger, or more popular. Choose it because its variance assumptions match the activation function and signal-flow behavior of the network.**