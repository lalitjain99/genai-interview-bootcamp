# 📝 Revision — CNN Backpropagation

> **Core Idea:** CNN backpropagation uses the same chain rule as any neural network. The main CNN-specific idea is that **convolution weights are shared across spatial positions**, so each shared weight receives gradient contributions from every place where it was used.

---

# 🎯 1. What Does Backpropagation Do?

Backpropagation computes:

> **how the loss changes with respect to each trainable parameter**

For a weight `w`:

`dL/dw`

The optimizer then uses this gradient to update the weight.

Example:

```text id="rev-cnn-bp-1"
w_new = w_old - learning_rate × dL/dw
```

Important:

> **Backpropagation computes gradients. The optimizer updates parameters.**

---

# 🧠 2. CNN Backprop Uses the Same Chain Rule

CNN backpropagation is not a different mathematical algorithm.

It still follows:

> **the chain rule**

Example:

```text id="rev-cnn-bp-2"
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

The main additional complexity comes from:

* spatial structure
* overlapping windows
* multiple channels
* parameter sharing

---

# ⭐ 3. Most Important CNN-Specific Idea

A convolution filter is:

> **reused across many spatial locations**

Forward:

```text id="rev-cnn-bp-3"
One Shared Filter
↓
Applied at Many Locations
```

Backward:

```text id="rev-cnn-bp-4"
Gradient Contributions
from Many Locations
↓
Summed
↓
One Gradient for Each Shared Weight
```

So:

> **shared parameter → accumulated gradient**

---

# 🧩 4. Small Convolution Example

Input:

```text id="rev-cnn-bp-5"
1  2  3
4  5  6
7  8  9
```

Kernel:

```text id="rev-cnn-bp-6"
a  b
c  d
```

Stride:

`1`

Output:

```text id="rev-cnn-bp-7"
y11  y12
y21  y22
```

For example:

`y11 = 1a + 2b + 4c + 5d`

`y12 = 2a + 3b + 5c + 6d`

The same weight `a` appears in multiple output calculations.

---

# 🧮 5. Gradient of a Shared Weight

For weight `a`:

```text id="rev-cnn-bp-8"
dL/da
=
dL/dy11 × 1
+
dL/dy12 × 2
+
dL/dy21 × 4
+
dL/dy22 × 5
```

Why do we add all terms?

Because:

> the same variable `a` influenced the loss through multiple computational paths.

General rule:

> **If one variable affects the loss through multiple paths, total gradient is the sum of all path contributions.**

---

# 🔄 6. Forward Sharing vs Backward Accumulation

Remember:

```text id="rev-cnn-bp-9"
FORWARD
One Filter → Many Locations

BACKWARD
Many Locations → One Filter Gradient
```

This is parameter sharing viewed from both directions.

---

# 📦 7. What Does a Conv Layer Compute During Backward?

A convolution layer usually needs:

### Weight Gradient

`dL/dW`

Used to update the filters.

### Bias Gradient

`dL/db`

Used to update biases.

### Input Gradient

`dL/dX`

Passed to the previous layer.

So:

```text id="rev-cnn-bp-10"
Conv Backward
↓
dL/dW
dL/db
dL/dX
```

---

# 🧠 8. Why Do We Need `dL/dX`?

Because earlier layers also need the training signal.

Example:

```text id="rev-cnn-bp-11"
Layer 1
↓
Layer 2
↓
Loss
```

Layer 2 must pass:

`dL/dX`

back to Layer 1.

That gradient becomes:

> the upstream gradient for the previous layer.

---

# 📐 9. Gradient Shapes

For a convolution:

```text id="rev-cnn-bp-12"
Input X:
H_in × W_in × C_in

Weights W:
K_h × K_w × C_in × C_out

Output Y:
H_out × W_out × C_out
```

Backward:

```text id="rev-cnn-bp-13"
dL/dX → same shape as X

dL/dW → same shape as W

dL/db → same shape as bias
```

Each trainable parameter needs:

> exactly one corresponding gradient.

---

# ⚡ 10. Backprop Through ReLU

ReLU:

`ReLU(z) = max(0, z)`

Derivative:

```text id="rev-cnn-bp-14"
1 if z > 0
0 if z < 0
```

So:

```text id="rev-cnn-bp-15"
Positive pre-activation
→ gradient passes

Negative pre-activation
→ gradient blocked
```

Example:

Upstream gradient:

`5`

If:

`z = 3`

then:

`gradient = 5`

If:

`z = -2`

then:

`gradient = 0`

---

# ⚠️ 11. ReLU Blocking Is Local

If one ReLU receives a negative pre-activation:

> that particular path gets zero gradient.

This does NOT mean:

> the whole CNN receives zero gradient.

Other active paths can still contribute.

---

# 🏊 12. MaxPool Backpropagation

Forward window:

```text id="rev-cnn-bp-16"
1  5
3  2
```

Output:

`5`

Suppose upstream gradient:

`7`

Backward:

```text id="rev-cnn-bp-17"
0  7
0  0
```

Rule:

> **MaxPool routes gradient only to the argmax location selected during the forward pass.**

---

# 🌊 13. Average Pooling Backpropagation

For a `2×2` AvgPool:

```text id="rev-cnn-bp-18"
output = (x1 + x2 + x3 + x4) / 4
```

If upstream gradient is:

`8`

each input receives:

`8 / 4 = 2`

Backward:

```text id="rev-cnn-bp-19"
2  2
2  2
```

Rule:

> **AvgPool distributes the upstream gradient equally across its window.**

---

# ⚡ 14. MaxPool vs AvgPool

| Operation | Backward Behavior            |
| --------- | ---------------------------- |
| MaxPool   | Gradient → argmax only       |
| AvgPool   | Gradient distributed equally |

Shortcut:

```text id="rev-cnn-bp-20"
MaxPool → Selective
AvgPool → Distributed
```

---

# 🎛️ 15. Multiple Channels

For standard Conv2D:

`W.shape = K_h × K_w × C_in × C_out`

Each kernel weight belongs to:

* one kernel location
* one input channel
* one output filter

The full filter is shared:

> across spatial positions.

During backward:

> each individual weight accumulates gradient across all spatial positions where that exact parameter was used.

---

# ⚠️ 16. Spatial Sharing ≠ Channel Sharing

Do not assume:

```text id="rev-cnn-bp-21"
W[:,:,0]
=
W[:,:,1]
=
W[:,:,2]
```

Different channel slices can have different weights.

What is shared is:

> **the same full filter across different spatial locations**

not one identical weight across all channels.

---

# 🧮 17. Bias Gradient

One convolution bias usually belongs to:

> one output channel.

That same bias is added to every spatial activation in that channel.

Therefore:

`dL/db`

is accumulated over:

* H positions
* W positions
* usually batch examples

Example upstream gradients:

```text id="rev-cnn-bp-22"
1  2
3  4
```

Then:

`dL/db = 1 + 2 + 3 + 4 = 10`

---

# 👥 18. Batch Gradient Accumulation

The same CNN parameters process every sample in a batch.

So parameter gradients include contributions from:

```text id="rev-cnn-bp-23"
Example 1
+
Example 2
+
...
+
Example N
```

Exact scale depends on whether the loss uses:

* sum reduction
* mean reduction

---

# ⚠️ 19. Sum vs Mean Loss

If batch loss is:

> mean of per-example losses

then gradients are correspondingly averaged.

If batch loss is:

> sum of per-example losses

gradient magnitudes are larger by the corresponding scaling.

This is a general neural-network concept, not unique to CNNs.

---

# 🔁 20. Input Gradient Accumulation

An input pixel can participate in:

> multiple overlapping convolution windows.

Therefore `dL/dX` at that pixel may receive gradient from:

> several output activations.

So:

```text id="rev-cnn-bp-24"
Many Outputs
↓
Same Input Position
↓
Gradient Contributions Sum
```

---

# ⭐ 21. Shared Weights and Overlapping Inputs Use the Same Principle

Two important cases:

### One Weight → Many Outputs

Because of:

> parameter sharing.

### One Input → Many Outputs

Because of:

> overlapping windows.

In both cases:

> total gradient = sum of contributions from all paths.

---

# ➡️ 22. Backprop Through Stride

Stride affects:

> forward connectivity.

If stride is `2`, fewer windows are evaluated.

During backward:

> gradients flow only through those connections that existed in the forward pass.

Stride:

* has no trainable parameters
* changes gradient-routing geometry

---

# 🧱 23. Backprop Through Padding

Padding may create temporary padded locations during the forward convolution.

Backward can conceptually compute gradients over those locations.

But:

> padded values are artificial and not trainable input pixels.

So gradients outside the original input region are:

> discarded.

The returned input gradient matches:

> the original input shape.

---

# 🕳️ 24. Backprop Through Dilation

Dilated convolution uses spaced kernel connections.

Backward follows:

> the same connectivity pattern used in forward.

Only positions that were connected through the dilated operation receive gradient contributions from that output.

---

# 🌍 25. Global Average Pooling Backward

If feature map size is:

`7×7`

then GAP averages:

`49 values`

per channel.

If upstream gradient is:

`g`

each position receives:

`g / 49`

So:

> **GAP distributes gradient uniformly across every spatial location of that channel.**

---

# 📦 26. Flatten Backward

Flatten:

```text id="rev-cnn-bp-25"
H×W×C
↓
Vector
```

changes only shape.

Backward:

> reshape incoming gradient back to `H×W×C`.

Flatten:

* has no parameters
* does not block gradients

---

# 🧠 27. No Parameters ≠ No Gradient

Operations such as:

* ReLU
* MaxPool
* AvgPool
* GAP
* Flatten

may have:

> zero trainable parameters

but still participate in:

> gradient propagation.

Important distinction:

```text id="rev-cnn-bp-26"
Trainable
≠
Differentiable
```

---

# 🎯 28. Trainable vs Differentiable

### Trainable Operation

Has parameters that the optimizer updates.

Examples:

* Conv
* Linear
* BatchNorm `gamma/beta`

### Differentiable Operation

Allows gradients to propagate through the computation.

Examples:

* ReLU
* AvgPool
* Flatten
* GAP

An operation can therefore be:

> differentiable without having trainable parameters.

---

# 🧠 29. Final Linear Layer Backprop

Typical classification flow:

```text id="rev-cnn-bp-27"
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

Backward starts at the loss and flows through:

```text id="rev-cnn-bp-28"
Loss
↓
Linear
↓
GAP
↓
Deep CNN Layers
↓
Early CNN Layers
```

This allows the final classification objective to train:

> the entire backbone.

---

# 🔄 30. End-to-End Learning

Forward:

```text id="rev-cnn-bp-29"
Image
↓
Features
↓
Prediction
↓
Loss
```

Backward:

```text id="rev-cnn-bp-30"
Loss
↓
Deep Layers
↓
Middle Layers
↓
Early Layers
↓
Filter Gradients
```

So early filters learn according to:

> how useful they are for the final task.

---

# 🧠 31. Local Filter, Global Training Signal

A convolution filter only sees:

> a local region during the forward pass.

But its update may depend on:

> the final loss of the entire network.

So:

```text id="rev-cnn-bp-31"
Local Operation
↓
Many Layers
↓
Global Prediction
↓
Loss
↓
Gradient Returns
↓
Local Filter Updated
```

This is how useful feature hierarchies emerge.

---

# 🔗 32. Backprop and Feature Hierarchy

Earlier we learned:

```text id="rev-cnn-bp-32"
Pixels
↓
Simple Features
↓
Complex Features
↓
Prediction
```

Backpropagation trains this hierarchy from the other direction:

```text id="rev-cnn-bp-33"
Loss
↓
High-Level Features
↓
Middle Features
↓
Early Features
```

So:

> **feature hierarchy is learned end-to-end through the final task objective.**

---

# 🔗 33. Residual Connections and Gradient Flow

Residual block:

`y = F(x) + x`

Derivative:

`dy/dx = dF/dx + 1`

The shortcut provides:

> an additional gradient path.

This helps very deep CNNs optimize more effectively.

But:

> residual connections do not guarantee that gradients can never vanish or explode.

---

# 🧠 34. BatchNorm Backward

BatchNorm has trainable:

* `gamma`
* `beta`

So backward computes:

```text id="rev-cnn-bp-34"
dL/dgamma
dL/dbeta
```

and also propagates gradients through the normalization operation to earlier layers.

Detailed derivatives are more involved, but:

> the chain-rule principle remains unchanged.

---

# ⚠️ 35. Backpropagation vs Optimizer

Do not confuse them.

### Backpropagation

Computes:

`dL/dW`

### Optimizer

Uses that gradient to change:

`W`

So:

```text id="rev-cnn-bp-35"
Backprop
↓
Gradient

Optimizer
↓
Parameter Update
```

---

# 🏋️ 36. CNN Training Loop

Typical sequence:

```text id="rev-cnn-bp-36"
Forward Pass
↓
Compute Loss
↓
Clear Old Gradients
↓
Backward Pass
↓
Optimizer Step
```

PyTorch-style:

```text id="rev-cnn-bp-37"
optimizer.zero_grad()

prediction = model(x)
loss = criterion(prediction, y)

loss.backward()
optimizer.step()
```

---

# ⚠️ 37. Two Meanings of “Gradient Accumulation”

This is an important interview trap.

## Mathematical Accumulation

A shared convolution parameter receives:

> gradient contributions from many spatial positions.

This is required by the chain rule.

## Training-Loop Accumulation

Framework gradients may accumulate across several backward calls if not cleared.

Example:

```text id="rev-cnn-bp-38"
Batch 1 gradient
+
Batch 2 gradient
```

These are:

> **different concepts**

---

# ⭐ 38. Strong Interview Distinction

If asked:

> Why does a CNN filter weight receive accumulated gradients?

Answer:

> **Because the same parameter is reused across many spatial output calculations, and the chain rule sums all gradient paths that depend on that shared parameter.**

Do not answer only:

> “Because PyTorch accumulates gradients.”

---

# 🧠 39. Autograd

Modern frameworks automatically build a:

> computational graph.

Then:

```text id="rev-cnn-bp-39"
loss.backward()
```

computes gradients using automatic differentiation.

Autograd essentially handles:

> chain-rule bookkeeping.

But understanding the underlying dependency structure remains important.

---

# 📦 40. Why Training Uses More Memory

Backward often needs forward-pass information such as:

* intermediate activations
* activation states
* pooling argmax locations
* tensors needed to compute derivatives

Therefore training memory can be much larger than inference memory.

---

# 🧠 41. Gradient Checkpointing

Gradient checkpointing trades:

```text id="rev-cnn-bp-40"
Memory ↓
Compute ↑
```

Instead of storing every activation:

* save selected checkpoints
* recompute some activations during backward

Useful for:

> large/deep models with memory constraints.

---

# ⚠️ 42. Backward Does Not Mean “Slide Filter Backwards”

The convolution filter does not physically reverse its forward motion.

Backward means:

> compute derivatives through the convolution operation.

Implementations may use convolution-like operations internally, but conceptually:

```text id="rev-cnn-bp-41"
Backward
=
Gradient Propagation
```

not:

> reverse scanning.

---

# ⚠️ 43. Transposed Convolution Is Not Backpropagation

Transposed convolution is:

> a forward trainable layer with mathematical relation to the transpose of a convolution operator.

It is often used for:

* upsampling
* decoder networks

But:

> **transposed convolution is not another name for CNN backpropagation.**

---

# ⚡ 44. Quick Backward Rules

| Component    | Gradient Behavior                        |
| ------------ | ---------------------------------------- |
| Conv weights | Sum contributions from every use         |
| Conv bias    | Sum over spatial positions/batch         |
| Conv input   | Sum contributions from dependent outputs |
| ReLU         | Pass if active, zero if inactive         |
| MaxPool      | Route to argmax                          |
| AvgPool      | Distribute equally                       |
| GAP          | Distribute across entire map             |
| Flatten      | Reshape                                  |
| Stride       | Follow forward connectivity              |
| Padding      | Discard gradients outside real input     |
| Linear       | Standard weight/input gradient           |

---

# ⚠️ 45. Common Misconceptions

### ❌ Backprop updates filters

Backprop computes gradients.

Optimizer updates filters.

---

### ❌ Every convolution location has separate weights

No.

Weights are shared spatially.

---

### ❌ Shared weight receives only one gradient

No.

It receives contributions from all spatial uses.

---

### ❌ MaxPool sends gradient everywhere

No.

It sends gradient to argmax.

---

### ❌ Pooling cannot backprop because it has no parameters

Wrong.

Gradient propagation does not require trainable parameters.

---

### ❌ Flatten stops gradient

No.

It only reshapes.

---

### ❌ Padding is trained

No.

Padding has no trainable parameters.

---

### ❌ `dL/dW` has feature-map shape

No.

It has:

> the same shape as the weight tensor.

---

### ❌ Transposed convolution = Conv backward

Not the same concept.

---

# 🎤 46. 30-Second Interview Answer

> **CNN backpropagation uses the same chain rule as any neural network. The main CNN-specific behavior comes from parameter sharing: because the same convolution filter weight is reused across many spatial locations, its gradient is the sum of contributions from every output activation that depends on it. A convolution layer also computes gradients with respect to its input so earlier layers can continue receiving the training signal. ReLU masks gradients, MaxPool routes them to the argmax, AvgPool and GAP distribute them, and the optimizer finally uses the computed parameter gradients to update the network.**

---

# 🧠 Final Mental Model

```text id="rev-cnn-bp-42"
FORWARD

Input
↓
Shared Filters
↓
Feature Maps
↓
Deep Features
↓
Prediction
↓
Loss


BACKWARD

Loss Gradient
↓
Deep Layers
↓
Pooling / Activations
↓
Conv Outputs
↓
Many Gradient Paths
↓
SUM CONTRIBUTIONS
↓
One Gradient per Shared Parameter
↓
Optimizer Update
```

---

# ⭐ Remember These 8 Things

1. 🧠 CNN backprop still uses the chain rule.
2. 🔄 Shared weights receive summed gradient contributions.
3. 📦 Conv backward computes `dL/dW`, `dL/db`, and `dL/dX`.
4. ⚡ ReLU masks gradients based on forward activation.
5. 🏊 MaxPool → argmax; AvgPool → equal distribution.
6. 🌍 GAP distributes gradient across the full spatial map.
7. 🎯 Backprop computes gradients; optimizer updates parameters.
8. ⚠️ Mathematical gradient accumulation is different from framework accumulation across backward calls.

---

# ⭐ Golden Rule

> **Forward convolution reuses one filter across many locations; backward convolution collects gradient information from all those locations back into one gradient for each shared filter parameter.**
