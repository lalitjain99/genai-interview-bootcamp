# 💡 Assignment — CNN Architecture

> **Focus:** Practice reading CNN architectures, tracking tensor shapes, calculating parameters, reasoning about compute, and evaluating architecture choices.

---

# 🧩 Scenario 1 — Track the Shape

Consider:

```text
Input: 64×64×3

Conv 3×3, 32 filters, S=1, P=1
ReLU
MaxPool 2×2, S=2
```

### Question

What is the output shape after the convolution and after pooling?

### ✅ Answer

After Conv:

```text
64×64×32
```

Why?

* `P=1`
* `S=1`
* `K=3`

so H and W stay unchanged.

Channels become:

`32`

because there are 32 filters.

After Pool:

```text
32×32×32
```

So:

```text
64×64×3
↓
64×64×32
↓
32×32×32
```

---

# 🧩 Scenario 2 — Channel Growth

Architecture:

```text
Input: 128×128×3

Conv 3×3, 64
Pool
Conv 3×3, 128
Pool
Conv 3×3, 256
```

### Question

What architectural trend do you observe?

### ✅ Answer

A common pattern is:

```text
Spatial Resolution ↓
Channels ↑
```

The network gradually trades:

> spatial detail

for:

> richer feature capacity and broader context.

---

# 🧠 Scenario 3 — Conv Parameter Count

A convolution has:

* Kernel = `3×3`
* `C_in = 64`
* `C_out = 128`
* Bias enabled

### Question

How many trainable parameters?

### ✅ Answer

Weights:

`3 × 3 × 64 × 128`

`= 73,728`

Biases:

`128`

Total:

> **73,856 parameters**

---

# 🧩 Scenario 4 — Same Params, Different Compute

Compare two layers.

### Layer A

Input:

`224×224×64`

Conv:

`3×3, 64→128`

### Layer B

Input:

`28×28×64`

Conv:

`3×3, 64→128`

### Question

Do they have the same parameter count?

### ✅ Answer

Yes.

Both use:

`3×3×64×128`

weights.

Parameter sharing means H and W do not directly affect learned parameter count.

But:

> **Layer A requires much more computation**

because the convolution is applied at many more spatial locations.

---

# 🧠 Scenario 5 — Parameter Count vs Compute

An engineer says:

> “Model A has fewer parameters than Model B, so Model A must be faster.”

### Is this always correct?

### ✅ Answer

No.

Runtime also depends on:

* spatial resolution
* number of operations
* operation type
* hardware
* memory access
* implementation efficiency

So:

> **parameter count ≠ runtime**

A model with relatively few parameters can still be computationally expensive.

---

# 🏊 Scenario 6 — Pooling vs Strided Conv

You need to reduce:

`64×64`

to:

`32×32`.

Two options:

### Option A

`MaxPool 2×2, S=2`

### Option B

`Conv 3×3, S=2`

### Question

What is the main difference?

### ✅ Answer

MaxPool performs:

> **fixed downsampling**

A stride-2 convolution performs:

> **learned feature transformation + downsampling**

So:

```text
Pooling
= Fixed

Strided Conv
= Learned
```

Both can reduce spatial size.

---

# 🌍 Scenario 7 — Flatten vs GAP

Suppose the final feature tensor is:

`8×8×256`

### Option A

Flatten

### Option B

Global Average Pooling

### Question

What output vector size does each produce?

### ✅ Answer

Flatten:

`8 × 8 × 256`

`= 16,384`

features.

GAP:

> one value per channel

So:

`256`

features.

Therefore:

```text
Flatten → 16,384

GAP → 256
```

---

# 🧮 Scenario 8 — Why GAP Can Reduce Parameters

Using the previous example:

`8×8×256`

Task:

`10 classes`

### Question

Compare classifier parameters.

### ✅ Answer

## Flatten + Linear

Input features:

`16,384`

Weights:

`16,384 × 10`

`= 163,840`

Bias:

`10`

Total:

> **163,850**

## GAP + Linear

Input features:

`256`

Weights:

`256 × 10`

`= 2,560`

Bias:

`10`

Total:

> **2,570**

So GAP dramatically reduces classifier parameters.

---

# 🧠 Scenario 9 — Backbone vs Head

Architecture:

```text
Image
↓
Several Conv Blocks
↓
Feature Tensor
↓
GAP
↓
Linear 512→100
```

### Question

Which part is backbone and which part is head?

### ✅ Answer

### Backbone

The convolutional feature extractor:

```text
Image
↓
Conv Blocks
↓
Feature Tensor
```

### Head

The task-specific prediction component:

```text
GAP
↓
Linear
↓
100 logits
```

So:

> **backbone learns representations; head maps them to the task output.**

---

# 🧩 Scenario 10 — Residual Shape Mismatch

Suppose:

```text
x = 56×56×64
```

Residual branch produces:

```text
F(x) = 28×28×128
```

### Question

Can you directly compute:

`F(x) + x`?

### ✅ Answer

No.

The shapes differ in:

* spatial size
* channel count

A projection shortcut can be used, for example:

`1×1 Conv, S=2`

to transform:

```text
56×56×64
↓
28×28×128
```

Then addition becomes valid.

---

# 🧠 Scenario 11 — Multiple Conv Before Pool

Compare:

### Model A

```text
Conv
Pool
Conv
Pool
```

### Model B

```text
Conv
Conv
Pool
Conv
Conv
Pool
```

### Question

Why might Model B learn richer features before downsampling?

### ✅ Answer

Model B performs more learned transformations:

> while spatial resolution is still preserved.

This allows the network to build richer feature combinations before discarding spatial detail.

However:

> it also increases compute and depth.

---

# 🎛️ Scenario 12 — Kernel Design

Compare:

### Option A

One `5×5` Conv

### Option B

Two `3×3` Conv layers

Assume equal channel width `C`.

### Question

Compare approximate weight count.

### ✅ Answer

One `5×5`:

`25C²`

Two `3×3`:

`2 × 9C²`

`=18C²`

So:

> two `3×3` layers can use fewer weights in this simplified same-channel case

and also provide an extra nonlinear transformation.

But:

> they are not identical architectures.

---

# 📱 Scenario 13 — Production Design

You have two models:

### Model A

Accuracy:

`94%`

Latency:

`250 ms`

### Model B

Accuracy:

`93.5%`

Latency:

`30 ms`

The application is a real-time mobile camera.

### Question

Which model may be more appropriate?

### ✅ Answer

Model B may be more appropriate.

Why?

Because production architecture selection considers:

* latency
* memory
* power usage
* accuracy

not accuracy alone.

For real-time mobile inference:

> a small accuracy reduction may be worth a large latency improvement.

---

# 🗺️ Scenario 14 — Classification vs Segmentation

A classification CNN reduces:

```text
256×256
↓
128×128
↓
64×64
↓
32×32
↓
16×16
↓
8×8
```

### Question

Why might using this architecture unchanged for segmentation be problematic?

### ✅ Answer

Segmentation needs:

> precise spatial localization.

Repeated downsampling removes:

* fine boundaries
* small structures
* exact spatial detail

So segmentation architectures often add:

* skip connections
* upsampling
* decoder stages
* multi-scale features

to recover spatial information.

---

# 🧠 Scenario 15 — Read the Full Architecture

Consider:

```text
Input: 32×32×3

Conv 3×3, 32, P=1
ReLU
MaxPool 2×2, S=2

Conv 3×3, 64, P=1
ReLU
MaxPool 2×2, S=2

Conv 3×3, 128, P=1
ReLU

GAP
Linear 128→10
```

### Question

What is the complete shape flow?

### ✅ Answer

Start:

`32×32×3`

Conv 1:

`32×32×32`

Pool:

`16×16×32`

Conv 2:

`16×16×64`

Pool:

`8×8×64`

Conv 3:

`8×8×128`

GAP:

`128`

Linear:

`10`

So:

```text
32×32×3
↓
32×32×32
↓
16×16×32
↓
16×16×64
↓
8×8×64
↓
8×8×128
↓
128
↓
10
```

---

# ✅ True / False

| #  | Statement                                                               | Answer  |
| -- | ----------------------------------------------------------------------- | ------- |
| 1  | CNN architecture means only convolution layers.                         | ❌ False |
| 2  | Pooling usually reduces H/W while preserving channels.                  | ✅ True  |
| 3  | Increasing channels always means more output classes.                   | ❌ False |
| 4  | Flatten is mandatory before classification.                             | ❌ False |
| 5  | GAP can reduce classifier parameter count substantially.                | ✅ True  |
| 6  | Convolution parameter count directly depends on H and W.                | ❌ False |
| 7  | Convolution compute depends on spatial resolution.                      | ✅ True  |
| 8  | Strided convolution can perform learned downsampling.                   | ✅ True  |
| 9  | More parameters always means slower inference.                          | ❌ False |
| 10 | Residual addition requires compatible tensor shapes.                    | ✅ True  |
| 11 | Aggressive downsampling may hurt localization tasks.                    | ✅ True  |
| 12 | More channels always improve accuracy.                                  | ❌ False |
| 13 | Early high-resolution layers can consume substantial activation memory. | ✅ True  |
| 14 | Architecture choice should consider deployment constraints.             | ✅ True  |

---

# ⭐ Staff Engineer Challenge

## Scenario

You are reviewing this CNN for a 20-class classification problem:

```text
Input: 256×256×3

Stage 1
Conv 3×3, 64, S=1, P=1
Conv 3×3, 64, S=1, P=1
MaxPool 2×2, S=2

Stage 2
Conv 3×3, 128, S=1, P=1
Conv 3×3, 128, S=1, P=1
MaxPool 2×2, S=2

Stage 3
Conv 3×3, 256, S=1, P=1
Conv 3×3, 256, S=1, P=1
MaxPool 2×2, S=2

Flatten
Linear → 4096
Linear → 20
```

The model has high memory usage and too many parameters.

---

## Question 1

What is the feature-map shape after Stage 3?

### ✅ Answer

Start:

`256×256`

After Pool 1:

`128×128×64`

After Stage 2 and Pool 2:

`64×64×128`

After Stage 3 and Pool 3:

> **32×32×256**

---

## Question 2

How many features are created by Flatten?

### ✅ Answer

`32 × 32 × 256`

`= 262,144`

So Flatten creates:

> **262,144 features**

---

## Question 3

How many weights does the first dense layer alone require?

### ✅ Answer

Dense layer:

`262,144 → 4,096`

Weights:

`262,144 × 4,096`

`= 1,073,741,824`

That is:

> **over 1 billion weights**

before even counting biases.

This is the major architectural problem.

---

## Question 4

How could you drastically reduce this parameter count?

### ✅ Answer

Replace:

```text
Flatten
↓
Linear 4096
```

with something like:

```text
Global Average Pooling
↓
Small Linear Head
```

For example:

```text
32×32×256
↓
GAP
↓
256
↓
Linear 256→20
```

---

## Question 5

How many classifier parameters would that GAP approach use?

### ✅ Answer

Weights:

`256 × 20`

`= 5,120`

Bias:

`20`

Total:

> **5,140 parameters**

This is dramatically smaller than the billion-weight dense head.

---

## Question 6

Would this automatically make the entire model extremely fast?

### ✅ Answer

Not necessarily.

It would greatly reduce:

* classifier parameters
* parameter memory

But convolutional compute may still be significant, especially at:

`256×256`

and:

`128×128`

resolutions.

So we must separately inspect:

> convolutional compute and activation memory.

---

## Question 7

Where would you look next if latency is still too high?

### ✅ Answer

I would inspect:

* high-resolution convolution stages
* channel widths
* repeated blocks
* downsampling schedule
* kernel sizes
* efficient convolution alternatives
* hardware utilization

The early large feature maps are often important compute/memory targets.

---

## Question 8

Would simply downsampling the input to `64×64` solve everything?

### ✅ Answer

It would significantly reduce compute.

But it could also remove:

* fine details
* small objects
* useful texture information

So input resolution should be reduced only after evaluating:

> accuracy and task requirements.

---

## Question 9

Suppose this CNN must now perform segmentation instead of classification. Would GAP still be appropriate as the final representation?

### ✅ Answer

Not as the main segmentation output path.

GAP collapses:

> all spatial dimensions.

Segmentation requires spatial output.

So a segmentation architecture needs to preserve or recover H/W using mechanisms such as:

* decoder
* upsampling
* skip connections

GAP may still be useful in auxiliary/global-context components, but it cannot by itself produce pixel-level predictions.

---

# 🎯 Final Architecture Exercise

Complete the following:

```text
Parameters depend mainly on:
________________________

Compute additionally depends strongly on:
________________________

Channels provide:
________________________

Downsampling trades:
________________________

GAP converts:
________________________
```

### ✅ Answer

```text
Parameters depend mainly on:
Kernel Size × C_in × C_out

Compute additionally depends strongly on:
H_out × W_out

Channels provide:
Feature representation capacity

Downsampling trades:
Spatial Detail for Efficiency / Broader Context

GAP converts:
H × W × C → C
```

---

# 🧠 Final Mental Model

When reviewing any CNN:

```text
INPUT
↓
Track H × W × C
↓
Find Feature Blocks
↓
Find Downsampling
↓
Track Channel Growth
↓
Estimate RF Growth
↓
Calculate Parameters
↓
Estimate Compute / Memory
↓
Inspect Final Representation
↓
Check Task Head
↓
Evaluate Task + Deployment Trade-Offs
```

---

# ⭐ Golden Rule

> **Never judge a CNN only by depth or parameter count — track tensor shapes, channels, receptive field, computation, memory, and whether the architecture preserves the information required by the final task.**
