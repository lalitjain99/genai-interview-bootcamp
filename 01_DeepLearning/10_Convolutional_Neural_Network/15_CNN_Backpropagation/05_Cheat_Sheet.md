# ⚡ Cheat Sheet — CNN Backpropagation

> **Core Idea:** CNN backpropagation uses the same chain rule as any neural network. The key CNN-specific point is that **shared convolution weights receive gradient contributions from every spatial location where they were used**.

---

# 🎯 1. What Backpropagation Does

Backpropagation computes:

`dL/d(parameter)`

for trainable parameters.

Example:

`dL/dW`

Then the optimizer updates:

`W`

So remember:

```text id="cbp-cs-1"
Backpropagation → Computes Gradients
Optimizer       → Updates Parameters
```

---

# 🧠 2. CNN Backprop Still Uses Chain Rule

CNNs do not use a new mathematical rule.

Example:

```text id="cbp-cs-2"
X
↓
Conv
↓
Z
↓
ReLU
↓
A
↓
Loss
```

Conceptually:

`dL/dX = dL/dA × dA/dZ × dZ/dX`

The extra CNN complexity comes from:

* shared weights
* spatial structure
* overlapping windows
* multiple channels

---

# ⭐ 3. Most Important CNN-Specific Rule

Forward:

```text id="cbp-cs-3"
One Filter Weight
↓
Used at Many Spatial Locations
```

Backward:

```text id="cbp-cs-4"
Many Gradient Contributions
↓
SUM
↓
One dL/dw
```

So:

> **Shared weight → summed gradient from every use**

---

# 🧮 4. Shared Weight Formula

If weight `w` affects outputs:

`y1, y2, y3 ...`

then:

`dL/dw = Σ (dL/dyi × dyi/dw)`

That sum exists because:

> one parameter influences the loss through multiple paths.

---

# 🔄 5. Forward vs Backward Mental Model

```text id="cbp-cs-5"
FORWARD
One Filter → Many Locations

BACKWARD
Many Locations → One Filter Gradient
```

---

# 📦 6. Conv Backward Computes Three Things

For a convolution layer:

### Weight Gradient

`dL/dW`

### Bias Gradient

`dL/db`

### Input Gradient

`dL/dX`

Purpose:

```text id="cbp-cs-6"
dL/dW → update filters
dL/db → update biases
dL/dX → train previous layer
```

---

# 📐 7. Gradient Shapes

If:

`X.shape = H_in × W_in × C_in`

then:

`dL/dX`

has the same shape.

If:

`W.shape = K_h × K_w × C_in × C_out`

then:

`dL/dW`

has the same shape.

If:

`b.shape = C_out`

then:

`dL/db.shape = C_out`

Rule:

> **one gradient per trainable parameter**

---

# ⚡ 8. ReLU Backward

ReLU:

`ReLU(z) = max(0,z)`

Derivative:

```text id="cbp-cs-7"
1 if z > 0
0 if z < 0
```

Backward:

```text id="cbp-cs-8"
Positive Pre-Activation
→ Gradient Passes

Negative Pre-Activation
→ Gradient = 0
```

---

# ⚠️ 9. ReLU Zero Gradient Is Local

If one ReLU path is inactive:

> that path gets zero gradient.

It does NOT imply:

> the entire CNN receives zero gradient.

---

# 🏊 10. MaxPool Backward

Forward:

```text id="cbp-cs-9"
2  9
5  3
```

Output:

`9`

If upstream gradient is:

`8`

Backward:

```text id="cbp-cs-10"
0  8
0  0
```

Rule:

> **MaxPool sends gradient only to the argmax location from forward pass.**

---

# 🌊 11. AvgPool Backward

For `2×2 AvgPool`:

If upstream gradient is:

`8`

each input receives:

`8 / 4 = 2`

So:

```text id="cbp-cs-11"
2  2
2  2
```

Rule:

> **AvgPool distributes gradient equally across the window.**

---

# 🌍 12. GAP Backward

For feature map:

`H × W`

GAP computes:

> average of all H×W values.

If upstream gradient is:

`g`

each position receives:

`g / (H×W)`

Example:

`4×4 map`

`g = 8`

Each position:

`8/16 = 0.5`

---

# 📦 13. Flatten Backward

Forward:

`H×W×C → vector`

Backward:

`vector gradient → H×W×C`

Flatten simply:

> **reshapes the gradient**

It has:

* no trainable parameters
* no learned update

---

# 🧠 14. No Parameters ≠ No Backprop

Operations with no trainable parameters can still propagate gradients.

Examples:

* ReLU
* MaxPool
* AvgPool
* GAP
* Flatten

So:

```text id="cbp-cs-12"
No Trainable Parameters
≠
No Gradient Flow
```

---

# 🎯 15. Trainable vs Differentiable

### Trainable

Has parameters updated by optimizer.

Examples:

* Conv
* Linear
* BatchNorm `gamma/beta`

### Differentiable

Allows gradients to flow.

Examples:

* ReLU
* AvgPool
* GAP
* Flatten

An operation can be:

> differentiable without being trainable.

---

# 🔁 16. Input Gradient and Overlapping Windows

A single input pixel may participate in many convolution windows.

Therefore:

```text id="cbp-cs-13"
Output 1 ─┐
Output 2 ─┤
Output 3 ─┼→ Same Input
Output 4 ─┘
```

The input gradient is:

> the sum of all backward contributions from outputs that depended on it.

---

# ⭐ 17. General Gradient Rule

Whenever one variable influences the loss through multiple paths:

> **total gradient = sum of contributions from all paths**

This explains both:

### Shared Weight

One weight → many outputs.

### Overlapping Input

One input → many outputs.

---

# 🎛️ 18. Multiple Channels

Standard Conv2D weight tensor:

`K_h × K_w × C_in × C_out`

Each parameter belongs to:

* one kernel coordinate
* one input channel
* one output filter

That parameter is shared:

> spatially.

It is NOT automatically shared:

> across channels.

---

# ⚠️ 19. Spatial Sharing ≠ Channel Sharing

Wrong idea:

```text id="cbp-cs-14"
all input-channel slices use identical weights
```

Correct:

> different input-channel slices may have different weights.

The **full filter** is reused across spatial locations.

---

# 🧮 20. Bias Gradient

One bias is typically shared across:

> every H×W position of one output channel.

Therefore:

`dL/db`

is summed over:

* height
* width
* usually batch examples

Example:

```text id="cbp-cs-15"
1  2
3  4
```

Bias gradient:

`1+2+3+4 = 10`

---

# 👥 21. Batch Gradient

The same CNN parameters process all batch examples.

So parameter gradients combine contributions from:

```text id="cbp-cs-16"
Example 1
+
Example 2
+
...
+
Example N
```

Exact scale depends on loss reduction:

* mean
* sum

---

# ⚠️ 22. Mean vs Sum Loss

### Mean Reduction

Gradients are averaged/scaled by batch size.

### Sum Reduction

Gradients are summed.

So:

> same data + different reduction can change gradient magnitude.

---

# ➡️ 23. Stride Backward

Stride changes:

> which forward connections exist.

For `S=2`, fewer output positions are created.

Backward:

> follows those same connections.

Stride:

* has no trainable parameters
* changes gradient geometry

---

# 🧱 24. Padding Backward

Padding adds artificial values during forward.

Backward may conceptually produce gradients over padded positions.

But:

> padding is not trainable.

Gradients outside the real input are discarded.

Returned:

`dL/dX`

matches the original input shape.

---

# 🕳️ 25. Dilation Backward

Dilated convolution uses spaced connectivity.

Backward follows:

> the same dilated connectivity pattern used in forward.

Only connected input positions receive contributions.

---

# 🔄 26. Final Classifier Backward

Typical flow:

```text id="cbp-cs-17"
CNN Features
↓
GAP
↓
Linear
↓
Logits
↓
Loss
```

Backward:

```text id="cbp-cs-18"
Loss
↓
Linear
↓
GAP
↓
Deep Conv Layers
↓
Early Conv Layers
```

This trains the network:

> end-to-end.

---

# 🧠 27. Local Filters Learn From Global Loss

A convolution filter operates locally.

But its update can depend on:

> the final task loss.

```text id="cbp-cs-19"
Local Filter
↓
Deep Network
↓
Prediction
↓
Loss
↓
Gradient Returns
↓
Filter Updated
```

This is how hierarchical features become task-relevant.

---

# 🔗 28. Backprop + Feature Hierarchy

Forward:

```text id="cbp-cs-20"
Pixels
↓
Simple Features
↓
Complex Features
↓
Prediction
```

Backward:

```text id="cbp-cs-21"
Loss
↓
Complex Features
↓
Simple Features
↓
Early Filters
```

So:

> the entire hierarchy is optimized by the final task objective.

---

# 🔗 29. Residual Connections

Residual block:

`y = F(x) + x`

Derivative:

`dy/dx = dF/dx + 1`

The shortcut gives:

> another gradient path.

This helps train deep CNNs.

But:

> residual connections do not guarantee zero vanishing/exploding-gradient problems.

---

# 🧠 30. BatchNorm Backward

BatchNorm learns:

* `gamma`
* `beta`

Backward computes:

`dL/dgamma`

`dL/dbeta`

and also propagates gradient through the normalization step.

Same principle:

> chain rule.

---

# 🎯 31. Backprop vs Optimizer

This distinction is essential.

```text id="cbp-cs-22"
Backprop
↓
Compute dL/dW

Optimizer
↓
Use dL/dW to update W
```

Example SGD:

`W_new = W_old - η × dL/dW`

---

# 🔄 32. Typical Training Loop

Conceptually:

```text id="cbp-cs-23"
Forward
↓
Loss
↓
Backward
↓
Optimizer Step
↓
Next Batch
```

PyTorch-style:

```text id="cbp-cs-24"
optimizer.zero_grad()

output = model(x)
loss = criterion(output, y)

loss.backward()
optimizer.step()
```

---

# ⚠️ 33. Two Types of Gradient Accumulation

### Mathematical Accumulation

Shared weight receives:

> contributions from many spatial locations.

This comes from:

> chain rule.

### Framework Accumulation

Gradients from multiple backward calls may accumulate unless cleared.

Example:

```text id="cbp-cs-25"
Batch 1 gradient
+
Batch 2 gradient
```

Different concepts.

---

# ⭐ 34. Interview Trap

Question:

> Why does a convolution filter gradient accumulate?

Correct:

> **Because the same parameter is used in many output computations, so chain-rule contributions from all dependent outputs are summed.**

Incomplete/wrong answer:

> “Because PyTorch accumulates gradients.”

---

# 🧠 35. Autograd

Modern frameworks automatically construct a computational graph.

Then:

```text id="cbp-cs-26"
loss.backward()
```

performs automatic differentiation.

Autograd handles:

> chain-rule bookkeeping.

---

# 📦 36. Why Training Needs More Memory

Backward often needs forward information such as:

* activations
* ReLU state
* pooling argmax locations
* intermediate tensors

So training memory includes much more than:

> model parameters.

---

# 🧠 37. Gradient Checkpointing

Trade-off:

```text id="cbp-cs-27"
Store Fewer Activations
↓
Memory ↓
But Recompute Forward Pieces
↓
Compute ↑
```

Useful for:

> memory-limited training.

---

# ⚠️ 38. Backward Does Not Mean Filter “Slides Backward”

Backpropagation means:

> compute derivatives through the convolution.

It does NOT mean:

> physically scan the kernel in reverse.

---

# ⚠️ 39. Transposed Conv ≠ Backpropagation

Transposed convolution is:

> a trainable forward operation.

It is mathematically related to the transpose of convolution's linear operator.

But:

> **it is not another name for CNN backward propagation.**

---

# ⚡ Quick Backward Table

| Operation   | Backward Rule                       |
| ----------- | ----------------------------------- |
| Conv Weight | Sum gradient from every spatial use |
| Conv Bias   | Sum across spatial positions/batch  |
| Conv Input  | Sum from all dependent outputs      |
| ReLU        | Pass if active, zero if inactive    |
| MaxPool     | Gradient → argmax                   |
| AvgPool     | Gradient shared equally             |
| GAP         | Gradient spread over full H×W       |
| Flatten     | Reshape gradient                    |
| Stride      | Follow forward connectivity         |
| Padding     | Ignore artificial padded region     |
| Linear      | Standard dense backprop             |

---

# 🚫 Interview Traps

### ❌ Backprop updates weights

No.

> Optimizer updates weights.

---

### ❌ Every Conv location has separate weights

No.

> Convolution shares weights spatially.

---

### ❌ Shared weight receives only one gradient

No.

> Contributions are summed.

---

### ❌ ReLU always passes gradient

No.

> Negative pre-activation paths get zero.

---

### ❌ MaxPool distributes gradient equally

No.

> It routes to argmax.

---

### ❌ AvgPool routes gradient only to max

No.

> It distributes equally.

---

### ❌ No parameters means no gradient

Wrong.

> Many parameter-free operations still backpropagate.

---

### ❌ Padding gets optimized

No.

---

### ❌ `dL/dW` has feature-map shape

No.

> It has the same shape as `W`.

---

### ❌ PyTorch gradient accumulation explains CNN parameter sharing

No.

> Different concepts.

---

### ❌ Transposed convolution is CNN backward

No.

---

# 🎤 30-Second Interview Answer

> **CNN backpropagation uses the ordinary chain rule. The important convolution-specific behavior is parameter sharing: because the same kernel weight contributes to many spatial outputs, its final gradient is the sum of all gradient paths through those outputs. A convolution layer computes gradients for its weights, biases, and inputs. ReLU masks gradients, MaxPool routes them to the argmax, AvgPool and GAP distribute them, and the optimizer finally uses the parameter gradients to update the model.**

---

# 🧠 Mental Model

```text id="cbp-cs-28"
FORWARD

Input
↓
Shared Filters
↓
Feature Maps
↓
Prediction
↓
Loss


BACKWARD

Loss
↓
Feature Gradients
↓
All Spatial Dependencies
↓
Gradient Contributions
↓
SUM
↓
dL/dW, dL/db, dL/dX
↓
Optimizer
↓
Updated Parameters
```

---

# ⭐ Remember These 8 Things

1. 🧠 CNN backprop = chain rule.
2. 🔄 Shared weights accumulate gradients from all spatial uses.
3. 📦 Conv backward computes `dL/dW`, `dL/db`, and `dL/dX`.
4. ⚡ ReLU passes or blocks gradient based on forward state.
5. 🏊 MaxPool → argmax; AvgPool → equal distribution.
6. 🌍 GAP spreads gradient over the entire spatial map.
7. 🎯 Backprop computes gradients; optimizer updates parameters.
8. ⚠️ Mathematical accumulation ≠ framework accumulation across batches.

---

# ⭐ Golden Rule

> **Forward sharing means one filter affects many outputs; backward learning means all of those output paths contribute to one gradient for that shared filter.**
