# ⚡ Cheat Sheet — CNN Architecture Comparison

> **Core Idea:** Remember each CNN architecture by the **problem it solved**, the **design idea it introduced**, and the **trade-off it created**.

---

# 🎯 1. Architecture Evolution in One Line

```text
LeNet
→ Basic CNN

AlexNet
→ Deep CNN at Scale

VGG
→ Small 3×3 Stacks

Inception
→ Multi-Scale Branches

ResNet
→ Residual Connections

DenseNet
→ Feature Reuse

MobileNet
→ Efficient Convolution

EfficientNet
→ Balanced Scaling
```

---

# 🧱 2. LeNet

## Identity

> **Basic CNN blueprint**

Typical flow:

```text
Conv
↓
Pool
↓
Conv
↓
Pool
↓
Fully Connected
```

### Key Idea

* local feature extraction
* pooling
* classification

### Remember

> **LeNet = early proof that CNNs work**

---

# 🚀 3. AlexNet

## Identity

> **Deep CNN breakthrough at scale**

Important ideas:

* deeper CNN
* ReLU
* dropout
* pooling
* GPU training

### Main Contribution

> showed that deep CNNs could dominate large-scale image classification.

### Main Trade-Off

* large parameter count
* heavy dense layers

### Remember

> **AlexNet = deep CNN + ReLU + scale**

---

# 🧩 4. VGG

## Identity

> **Repeated small `3×3` convolutions**

Typical block:

```text
3×3 Conv
↓
3×3 Conv
↓
Pool
```

### Why `3×3`?

Two `3×3` layers:

* roughly `5×5` nominal RF
* extra nonlinearity
* often fewer weights than one `5×5` under equal-channel assumptions

### Main Strength

> simple and uniform architecture

### Main Weakness

> high compute and parameter cost

### Remember

> **VGG = simple + deep + 3×3**

---

# 🌐 5. Inception / GoogLeNet

## Identity

> **Parallel multi-scale processing**

Concept:

```text
             ┌→ 1×1 Conv ─┐
Input ───────┼→ 3×3 Conv ─┤
             ├→ 5×5 Conv ─┤→ Concatenate
             └→ Pool ─────┘
```

### Main Benefit

Captures:

* small-scale patterns
* medium-scale patterns
* large-scale patterns

simultaneously.

### Merge

> **Concatenation**

### Main Trade-Off

> more complex topology

### Remember

> **Inception = multi-scale branches**

---

# 🎛️ 6. `1×1` Convolution

Used heavily in:

* Inception
* ResNet bottlenecks
* MobileNet pointwise Conv

### Purpose

* mix channels
* reduce channels
* expand channels
* create bottlenecks

Formula:

```text
H×W×C_in
↓
1×1 Conv
↓
H×W×C_out
```

### Remember

> **1×1 Conv = channel projection**

---

# 🔗 7. ResNet

## Identity

> **Residual / skip connections**

Formula:

`y = F(x) + x`

Concept:

```text
x ────────────┐
↓             │
F(x)          │
↓             │
+ ◄───────────┘
↓
y
```

### Main Problem Solved

> optimization of very deep networks

### Main Benefit

* better information flow
* better gradient flow
* easier identity mapping
* enables much deeper networks

### Merge

> **Addition**

### Remember

> **ResNet = Add**

---

# 🧱 8. ResNet Bottleneck

Typical:

```text
1×1 Reduce
↓
3×3 Spatial Conv
↓
1×1 Restore
```

Purpose:

> keep expensive `3×3` processing at smaller channel width.

Commonly associated with deeper ResNets.

---

# 🌿 9. DenseNet

## Identity

> **Dense feature reuse through concatenation**

Each layer receives earlier features:

`Concat(x0, x1, x2, ...)`

### Main Benefit

* feature reuse
* short gradient paths
* strong information flow

### Merge

> **Concatenation**

### Main Trade-Off

> activation memory can grow significantly

### Remember

> **DenseNet = Concatenate**

---

# 📈 10. DenseNet Growth Rate

If each layer adds:

`k`

new feature maps:

> `k = growth rate`

Example:

```text
Initial = 64
k = 32

Layer 1 → 96
Layer 2 → 128
Layer 3 → 160
```

---

# 🔍 11. ResNet vs DenseNet

| ResNet                       | DenseNet                      |
| ---------------------------- | ----------------------------- |
| Addition                     | Concatenation                 |
| `F(x)+x`                     | `Concat(previous features)`   |
| Width stays controlled       | Channels grow                 |
| Strong shortcut path         | Strong feature reuse          |
| Usually more memory-friendly | Activation memory can be high |

Shortcut:

```text
ResNet   → ADD
DenseNet → CONCAT
```

---

# 📱 12. MobileNet

## Identity

> **Depthwise separable convolution**

Instead of standard Conv:

```text
Spatial Filtering
+
Channel Mixing
```

MobileNet separates them:

```text
Depthwise Conv
↓
Pointwise 1×1 Conv
```

### Remember

> **Depthwise = spatial**
>
> **Pointwise = channel mixing**

---

# 🧮 13. Standard Conv vs Depthwise Separable Conv

### Standard

`K² × C_in × C_out`

### Depthwise

`K² × C_in`

### Pointwise

`C_in × C_out`

Total:

`K²C_in + C_inC_out`

---

# 🧮 14. MobileNet Example

Given:

* `K=3`
* `C_in=64`
* `C_out=128`

### Standard Conv

`3×3×64×128`

`= 73,728`

### Depthwise

`3×3×64`

`= 576`

### Pointwise

`64×128`

`= 8,192`

Total:

`8,768`

So:

```text
Standard           → 73,728
Depthwise Separable → 8,768
```

---

# ⚠️ 15. FLOPs ≠ Real Latency

Fewer FLOPs does NOT guarantee faster execution.

Latency also depends on:

* hardware
* memory bandwidth
* operator support
* kernel implementation
* parallelism

So:

> **benchmark on target hardware**

---

# ⚖️ 16. EfficientNet

## Identity

> **Compound scaling**

Scale three dimensions together:

```text
Depth ↑
Width ↑
Resolution ↑
```

Instead of scaling only one.

### Remember

> **EfficientNet = balanced scaling**

---

# 🧠 17. What Does Each Scaling Dimension Do?

### Depth

More layers.

Benefit:

> more transformations.

Cost:

> more compute and training complexity.

---

### Width

More channels.

Benefit:

> more feature capacity.

Cost:

> Conv cost grows strongly with `C_in × C_out`.

---

### Resolution

Larger H/W.

Benefit:

> more spatial detail.

Cost:

> more spatial positions and activation memory.

---

# ⭐ 18. Compound Scaling Principle

Do not blindly do:

```text
Only Depth ↑
```

or:

```text
Only Width ↑
```

or:

```text
Only Resolution ↑
```

Instead:

> scale them in a coordinated way.

---

# 🏗️ 19. Architecture Topology

### VGG

Sequential:

```text
Layer
↓
Layer
↓
Layer
```

### Inception

Parallel:

```text
       ┌→ Branch A ─┐
Input ─┼→ Branch B ─┼→ Merge
       └→ Branch C ─┘
```

### ResNet

Skip connection:

```text
x ───────┐
↓        │
F(x)     │
↓        │
+ ◄──────┘
```

### DenseNet

Dense connectivity:

```text
Earlier Features
↓
All Later Layers
```

---

# 🔗 20. Addition vs Concatenation

### Addition

Example:

```text
56×56×64
+
56×56×64
=
56×56×64
```

Channel count stays:

`64`

---

### Concatenation

```text
56×56×64
Concat
56×56×64
=
56×56×128
```

Channels add.

---

# ⚠️ 21. Addition vs Concatenation Trade-Off

### Addition

* keeps width controlled
* memory-efficient
* ideal for residual shortcut

### Concatenation

* preserves features separately
* improves feature reuse
* increases channel count / activation size

---

# 🎯 22. Which Architecture Solved What?

| Architecture | Problem Solved                                    |
| ------------ | ------------------------------------------------- |
| LeNet        | Establish basic CNN learning                      |
| AlexNet      | Scale deep CNNs to harder vision tasks            |
| VGG          | Build simple deep architecture with small kernels |
| Inception    | Efficient multi-scale processing                  |
| ResNet       | Train very deep networks                          |
| DenseNet     | Improve feature reuse                             |
| MobileNet    | Reduce cost for edge/mobile devices               |
| EfficientNet | Scale CNNs systematically                         |

---

# ⚡ 23. Architecture Identity Table

| Architecture | Keyword                  |
| ------------ | ------------------------ |
| LeNet        | Basic CNN                |
| AlexNet      | Deep CNN breakthrough    |
| VGG          | `3×3` stacks             |
| Inception    | Multi-scale              |
| ResNet       | Residual addition        |
| DenseNet     | Dense concatenation      |
| MobileNet    | Depthwise separable Conv |
| EfficientNet | Compound scaling         |

---

# 🧠 24. VGG vs ResNet

### VGG

```text
Conv
↓
Conv
↓
Conv
```

### ResNet

```text
x ────────┐
↓         │
F(x)      │
↓         │
+ ◄───────┘
```

Main distinction:

> **VGG = plain deep stack**

> **ResNet = deep stack with shortcuts**

Do not say:

> ResNet is better only because it has more layers.

---

# 🌐 25. Inception vs ResNet

### Inception

Goal:

> multi-scale feature extraction.

Mechanism:

> parallel branches + concatenation.

### ResNet

Goal:

> easier deep optimization.

Mechanism:

> skip connection + addition.

Shortcut:

```text
Inception → Multi-Scale
ResNet    → Deep Optimization
```

---

# 🌿 26. DenseNet vs ResNet

### ResNet

```text
Old + New
```

### DenseNet

```text
Old || New
```

where `||` means:

> channel concatenation.

Remember:

```text
ResNet → combine features
DenseNet → preserve/reuse features
```

---

# 📱 27. MobileNet vs Standard Conv

### Standard Conv

Spatial filtering and channel mixing happen:

> together.

### MobileNet

Separates them:

```text
Depthwise
→ Spatial

Pointwise
→ Channels
```

---

# ⚖️ 28. MobileNet vs EfficientNet

### MobileNet

Question:

> How can the convolution operation itself become cheaper?

### EfficientNet

Question:

> How should the overall network scale with more compute?

So:

```text
MobileNet   → Efficient Operation
EfficientNet → Efficient Scaling
```

---

# 📦 29. Architecture Cost Dimensions

When comparing CNNs, look at:

```text
Parameters
Compute / FLOPs
Activation Memory
Training Memory
Latency
Accuracy
```

Never compare models using:

> parameter count alone.

---

# 🎯 30. Typical Architecture Trade-Offs

### VGG

* simple
  − heavy

### Inception

* multi-scale / efficient
  − complex

### ResNet

* deep optimization
  − can still be compute-heavy

### DenseNet

* feature reuse
  − activation memory

### MobileNet

* lightweight
  − capacity/accuracy trade-off

### EfficientNet

* balanced efficiency
  − more involved scaling/design

---

# 🧠 31. Backbone Thinking

These architectures are often used as:

> **feature-extraction backbones**

Example:

```text
Image
↓
ResNet Backbone
↓
Feature Maps
↓
Detection Head
```

or:

```text
Image
↓
MobileNet Backbone
↓
Segmentation Head
```

So architecture family and task head can be:

> separated.

---

# 🔢 32. Architecture Variants

Example:

```text
ResNet-18
ResNet-34
ResNet-50
ResNet-101
ResNet-152
```

Do not assume:

> every ResNet has the same block design.

Commonly:

* ResNet-18/34 → basic blocks
* ResNet-50+ → bottleneck blocks

---

# 🚫 Interview Traps

### ❌ AlexNet introduced residual connections

No.

> ResNet did.

---

### ❌ VGG's main idea is multi-scale branching

No.

> Inception.

---

### ❌ ResNet uses concatenation

Primarily:

> addition.

---

### ❌ DenseNet uses addition

Primarily:

> concatenation.

---

### ❌ `1×1` Conv is useless

Wrong.

> It mixes and projects channels.

---

### ❌ Depthwise Conv mixes all channels

No.

> Pointwise Conv performs cross-channel mixing.

---

### ❌ MobileNet always runs faster

Not guaranteed.

---

### ❌ EfficientNet only increases depth

No.

> depth + width + resolution.

---

### ❌ Fewer parameters = less total memory

Not necessarily.

> activations matter.

---

### ❌ More layers = automatically better architecture

No.

---

# 🎤 30-Second Interview Answer

> **Major CNN architectures are best distinguished by their core design innovation. LeNet established the basic CNN pattern, AlexNet showed deep CNNs could work at scale, and VGG used deep stacks of small 3×3 kernels. Inception introduced multi-scale parallel branches and 1×1 bottlenecks. ResNet used residual addition to improve deep-network optimization, while DenseNet reused earlier features through concatenation. MobileNet reduced convolution cost using depthwise and pointwise operations, and EfficientNet introduced coordinated scaling of network depth, width, and input resolution.**

---

# 🧠 Final Mental Model

```text
LeNet
→ Basic CNN

AlexNet
→ Deep at Scale

VGG
→ Small Kernels

Inception
→ Multi-Scale

ResNet
→ Add Shortcuts

DenseNet
→ Concatenate Features

MobileNet
→ Factorize Convolution

EfficientNet
→ Scale Intelligently
```

---

# ⭐ Remember These 8 Things

1. 🧱 **LeNet** → basic CNN blueprint.
2. 🚀 **AlexNet** → deep CNN breakthrough.
3. 🧩 **VGG** → stacked `3×3`.
4. 🌐 **Inception** → parallel multi-scale branches.
5. 🔗 **ResNet** → residual addition.
6. 🌿 **DenseNet** → dense feature concatenation.
7. 📱 **MobileNet** → depthwise + pointwise convolution.
8. ⚖️ **EfficientNet** → compound scaling.

---

# ⭐ Golden Rule

> **Do not memorize architecture names in isolation. Remember: problem → design idea → benefit → trade-off.**
