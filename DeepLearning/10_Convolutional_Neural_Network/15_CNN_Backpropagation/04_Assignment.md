# 💡 Assignment — CNN Backpropagation

> **Focus:** Practice gradient flow through convolution, shared weights, ReLU, pooling, GAP, and the end-to-end CNN training pipeline.

---

# 🧩 Scenario 1 — Shared Weight Gradient

A `2×2` kernel is applied over a `3×3` input.

Kernel:

```text
a  b
c  d
```

The weight `a` is reused at four spatial positions.

Suppose the upstream gradients from those four output positions are:

`1, 2, 3, 4`

and the corresponding input values multiplied by `a` were:

`2, 5, 1, 3`

### Question

What is `dL/da`?

### ✅ Answer

Because `a` is shared:

`dL/da`

is the sum of all path contributions.

So:

`dL/da = 1×2 + 2×5 + 3×1 + 4×3`

`= 2 + 10 + 3 + 12`

`= 27`

Therefore:

> **dL/da = 27**

---

# 🧠 Scenario 2 — Why Sum the Contributions?

A developer says:

> “Since there is only one weight `a`, we should only use one output location to calculate its gradient.”

### Correct or incorrect?

### ✅ Answer

Incorrect.

The same parameter influenced:

> multiple output activations

So by the chain rule:

> all gradient paths must contribute.

Therefore:

```text
One Shared Parameter
↓
Many Uses
↓
Many Gradient Contributions
↓
SUM
```

---

# ⚡ Scenario 3 — ReLU Backward

Suppose upstream gradient is:

`6`

### Case A

Pre-activation:

`z = 4`

### Case B

Pre-activation:

`z = -3`

### Question

What gradient passes backward in each case?

### ✅ Answer

For ReLU:

```text
derivative = 1 if z > 0
derivative = 0 if z < 0
```

### Case A

`6 × 1 = 6`

### Case B

`6 × 0 = 0`

So:

```text
z = 4  → gradient = 6
z = -3 → gradient = 0
```

---

# 🏊 Scenario 4 — MaxPool Backward

Forward pooling window:

```text
2  9
5  3
```

MaxPool output:

`9`

Suppose upstream gradient is:

`8`

### Question

What gradient goes back to the four inputs?

### ✅ Answer

The max came from the top-right position.

So:

```text
0  8
0  0
```

Rule:

> **MaxPool routes the gradient only to the argmax location.**

---

# 🌊 Scenario 5 — AvgPool Backward

A `2×2` AvgPool receives upstream gradient:

`12`

### Question

What gradient does each input receive?

### ✅ Answer

There are four inputs.

So:

`12 / 4 = 3`

Each input receives:

`3`

Backward:

```text
3  3
3  3
```

---

# 🌍 Scenario 6 — GAP Backward

A feature channel has spatial size:

`4×4`

Global Average Pooling reduces it to one number.

Upstream gradient from the classifier is:

`8`

### Question

What gradient does each spatial position receive?

### ✅ Answer

There are:

`4 × 4 = 16`

positions.

So each receives:

`8 / 16`

`= 0.5`

Therefore:

> **each location receives gradient 0.5**

---

# 🧩 Scenario 7 — Input Gradient From Overlapping Windows

A center pixel participates in:

`4`

different convolution output calculations.

The four backward contributions to that input pixel are:

`1.5, -0.5, 2, 3`

### Question

What is the final gradient for that input pixel?

### ✅ Answer

Add all paths:

`1.5 - 0.5 + 2 + 3`

`= 6`

Therefore:

> **dL/dx_center = 6**

---

# 🎛️ Scenario 8 — Multiple Channels

A convolution uses:

* `C_in = 3`
* `C_out = 16`
* Kernel = `3×3`

### Question

Are the weights for the three input channels identical because of parameter sharing?

### ✅ Answer

No.

The full weight tensor is:

`3×3×3×16`

Different input-channel slices can contain:

> different learned weights.

Parameter sharing means:

> the same full filter is reused across spatial positions.

It does NOT mean:

> all input channels use identical weights.

---

# 🧮 Scenario 9 — Bias Gradient

One output channel has upstream gradients:

```text
1  2
4  3
```

### Question

What is the bias gradient for that channel?

### ✅ Answer

The same bias was added at all four positions.

So:

`dL/db = 1 + 2 + 4 + 3`

`= 10`

Therefore:

> **bias gradient = 10**

---

# 👥 Scenario 10 — Batch Gradient

A shared convolution weight receives gradient contributions from three examples:

```text
Example 1 → 4
Example 2 → 6
Example 3 → 2
```

Suppose the loss uses **sum reduction**.

### Question

What is the resulting gradient contribution?

### ✅ Answer

`4 + 6 + 2`

`= 12`

So:

> **gradient = 12**

If the loss instead used mean reduction over these three examples:

`12 / 3 = 4`

So exact gradient scale depends on:

> the loss reduction convention.

---

# 🧠 Scenario 11 — Trainable vs Differentiable

Consider these operations:

* Conv
* ReLU
* MaxPool
* Flatten
* Linear

### Question

Which have trainable parameters?

### ✅ Answer

### Trainable

* Conv
* Linear

### No trainable parameters

* ReLU
* MaxPool
* Flatten

But all of them can:

> participate in gradient propagation.

So:

> **No parameters ≠ no backpropagation**

---

# 🧩 Scenario 12 — Padding Backward

A convolution uses zero padding.

A developer says:

> “The padded zeros will receive gradients and then be updated during optimization.”

### Is that correct?

### ✅ Answer

No.

Padding values are:

> artificial values, not trainable parameters.

Backward may conceptually produce gradient values at padded positions while computing the operation.

But those gradients are:

> discarded outside the real input region.

Padding itself is never optimized.

---

# ➡️ Scenario 13 — Stride Backward

Forward convolution uses:

`stride = 2`

### Question

Does stride introduce trainable parameters during backward?

### ✅ Answer

No.

Stride only changes:

> which spatial connections exist.

Backward follows the same connectivity used during forward.

So stride affects:

* gradient routing geometry

but has:

> **zero trainable parameters**

---

# 📦 Scenario 14 — Flatten Backward

Forward:

```text
4×4×8
↓
Flatten
↓
128
```

### Question

What happens to a 128-dimensional gradient during backward?

### ✅ Answer

It is reshaped back to:

`4×4×8`

No values are learned or altered by Flatten itself.

So:

> **Flatten backward = reshape gradient**

---

# 🔄 Scenario 15 — Backprop vs Optimizer

Suppose after backward:

`dL/dw = 0.8`

Current weight:

`w = 2`

Learning rate:

`0.1`

Using simple gradient descent:

### Question

What happens?

### ✅ Answer

Backprop gives:

`dL/dw = 0.8`

Optimizer update:

`w_new = 2 - 0.1 × 0.8`

`= 1.92`

Therefore:

> **Backprop computed 0.8; optimizer changed the weight to 1.92.**

---

# ✅ True / False

| #  | Statement                                                                       | Answer  |
| -- | ------------------------------------------------------------------------------- | ------- |
| 1  | CNN backpropagation still relies on the chain rule.                             | ✅ True  |
| 2  | Each spatial location gets its own copy of the convolution weight.              | ❌ False |
| 3  | Shared weights receive gradient contributions from every spatial use.           | ✅ True  |
| 4  | Backpropagation itself updates the weights.                                     | ❌ False |
| 5  | MaxPool sends gradient to the argmax location.                                  | ✅ True  |
| 6  | AvgPool sends all gradient to one position.                                     | ❌ False |
| 7  | ReLU can block gradient on inactive paths.                                      | ✅ True  |
| 8  | Flatten has trainable parameters.                                               | ❌ False |
| 9  | Padding values are updated by the optimizer.                                    | ❌ False |
| 10 | One input pixel can receive gradients from multiple outputs.                    | ✅ True  |
| 11 | `dL/dW` has the same shape as `W`.                                              | ✅ True  |
| 12 | A parameter-free operation can still propagate gradients.                       | ✅ True  |
| 13 | Stride changes gradient-routing geometry.                                       | ✅ True  |
| 14 | Mathematical gradient accumulation and PyTorch accumulation are the same thing. | ❌ False |

---

# ⭐ Staff Engineer Challenge

## Scenario

Consider this network:

```text
Input
↓
Conv 3×3
↓
ReLU
↓
MaxPool 2×2
↓
Conv 3×3
↓
GAP
↓
Linear
↓
Loss
```

During backward:

* the classifier sends gradient `12` to one GAP output channel
* the feature map before GAP is `2×2`
* one MaxPool window selected the bottom-left activation
* one shared convolution weight was used at four spatial locations

---

## Question 1

What gradient does each position before GAP receive?

### ✅ Answer

The feature map is:

`2×2`

So there are:

`4`

positions.

GAP distributes:

`12 / 4`

`= 3`

Therefore:

```text
3  3
3  3
```

---

## Question 2

What does MaxPool do with its incoming gradient?

### ✅ Answer

For each pooling window:

> the gradient is routed to the location that was the maximum during forward.

If bottom-left was the argmax:

```text
0  0
g  0
```

where `g` is the upstream gradient for that pooled output.

---

## Question 3

What happens if that gradient then reaches a ReLU whose forward pre-activation was negative?

### ✅ Answer

ReLU derivative there is:

`0`

So:

> the gradient becomes zero along that path.

---

## Question 4

Suppose a shared convolution weight `w` was used at four locations and receives these path contributions:

`2, -1, 4, 3`

What is `dL/dw`?

### ✅ Answer

Add them:

`2 - 1 + 4 + 3`

`= 8`

Therefore:

> **dL/dw = 8**

---

## Question 5

How many copies of `w` are updated?

### ✅ Answer

Only:

> **one**

There is one shared parameter.

The four spatial uses generate four gradient contributions, but all are accumulated into:

> one final gradient for the single parameter.

---

## Question 6

Suppose a center input activation contributed to three outputs and receives backward contributions:

`5, -2, 1`

What is its input gradient?

### ✅ Answer

`5 - 2 + 1`

`= 4`

Therefore:

> **dL/dx = 4**

---

## Question 7

What happens after all convolution, pooling, activation, and linear gradients have been computed?

### ✅ Answer

The optimizer uses the trainable parameter gradients to update:

* Conv weights
* Conv biases
* Linear weights
* Linear biases

Parameter-free operations such as:

* ReLU
* MaxPool
* GAP

are not updated.

---

## Question 8

Why is this considered end-to-end learning?

### ✅ Answer

Because the final loss sends gradient information through:

```text
Classifier
↓
Deep Features
↓
Pooling
↓
Early Features
↓
Initial Convolution Filters
```

So even early local filters are optimized according to:

> their contribution to the final task performance.

---

# 🧠 Final Reasoning Exercise

Complete:

```text
Shared Conv Weight
→ ______________________

MaxPool Backward
→ ______________________

AvgPool Backward
→ ______________________

ReLU Backward
→ ______________________

Backpropagation
→ ______________________

Optimizer
→ ______________________
```

### ✅ Answer

```text
Shared Conv Weight
→ Sum gradients from every spatial use

MaxPool Backward
→ Route gradient to argmax

AvgPool Backward
→ Distribute gradient equally

ReLU Backward
→ Pass or block gradient based on activation

Backpropagation
→ Compute gradients

Optimizer
→ Update trainable parameters
```

---

# 🧠 Final Mental Model

```text
FORWARD
Input
↓
Shared Conv Filters
↓
Activations
↓
Pooling
↓
Deep Features
↓
Prediction
↓
Loss


BACKWARD
Loss
↓
Classifier Gradient
↓
Pooling / Activation Gradient
↓
Conv Gradient Paths
↓
SUM SHARED CONTRIBUTIONS
↓
dL/dW, dL/db, dL/dX
↓
Optimizer Update
```

---

# ⭐ Golden Rule

> **Whenever a CNN value or parameter affects the loss through multiple paths, its final gradient is the sum of all those contributions; convolution makes this especially important because weights are shared and input windows overlap.**
