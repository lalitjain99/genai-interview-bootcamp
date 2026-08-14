# ⚡ Cheat Sheet — CNN Architecture

> **Core Idea:** A CNN architecture organizes convolution, activation, normalization, downsampling, feature stages, and a task-specific head into one complete network.

---

# 🎯 1. CNN Architecture

A CNN architecture is:

> **the complete arrangement of layers, blocks, stages, and connections that transform an image into a prediction**

Typical flow:

```text
Input
↓
Stem
↓
Feature Stages
↓
Backbone Output
↓
GAP / Flatten
↓
Task Head
↓
Prediction
```

---

# 🧱 2. Layer vs Block vs Stage vs Architecture

### Layer

One operation.

Examples:

* Conv
* ReLU
* BatchNorm
* Pool

### Block

Group of layers.

Example:

```text
Conv
↓
BN
↓
ReLU
↓
Conv
↓
BN
↓
ReLU
```

### Stage

One or more blocks, often operating at similar spatial resolution.

### Architecture

The complete network.

Shortcut:

```text
Layer
↓
Block
↓
Stage
↓
Architecture
```

---

# 🌱 3. Stem

The stem is:

> **the initial part of the CNN that converts raw pixels into initial feature maps**

Example:

```text
Input
↓
Conv
↓
BN
↓
ReLU
↓
Pool
```

---

# 🦴 4. Backbone

The backbone is:

> **the main feature extractor**

Conceptually:

```text
Input
↓
Stem
↓
Stage 1
↓
Stage 2
↓
Stage 3
↓
Final Feature Maps
```

---

# 🎯 5. Head

The head performs:

> **task-specific prediction**

For classification:

```text
Feature Vector
↓
Linear Layer
↓
Class Logits
```

So:

```text
Backbone → learns features
Head     → solves final task
```

---

# 📐 6. Always Track Shape

At every major layer, track:

`H × W × C`

Example:

```text
224×224×3
↓
224×224×64
↓
112×112×64
↓
112×112×128
↓
56×56×128
```

This helps catch:

* wrong output dimensions
* wrong channel counts
* downsampling mistakes
* incompatible skip connections

---

# 🔍 7. Convolution Shape

Example:

```text
Input:
224×224×3

Conv:
3×3
64 filters
S=1
P=1

Output:
224×224×64
```

Why?

* `S=1`
* `P=1`
* `K=3`

preserve H/W.

And:

> **64 filters → 64 output channels**

---

# ⚡ 8. Conv + Activation

Common basic unit:

```text
Conv
↓
ReLU
```

Modern variant:

```text
Conv
↓
BatchNorm
↓
ReLU
```

Activation adds:

> **nonlinearity**

which allows rich feature composition.

---

# 🧩 9. Multiple Conv Before Downsampling

Common pattern:

```text
Conv
↓
ReLU
↓
Conv
↓
ReLU
↓
Pool
```

Why?

> More feature transformation happens before spatial detail is reduced.

---

# 🏊 10. Pooling

Example:

```text
224×224×64
↓
MaxPool 2×2, S=2
↓
112×112×64
```

Typically:

```text
H ↓
W ↓
C unchanged
```

for standard spatial pooling.

---

# ➡️ 11. Pooling vs Strided Conv

### Pooling

```text
MaxPool 2×2, S=2
```

> Fixed downsampling.

### Strided Convolution

```text
Conv 3×3, S=2
```

> Learned transformation + downsampling.

Shortcut:

```text
Pooling      → Fixed
Strided Conv → Learned
```

---

# 📉 12. Why Downsample?

Downsampling helps:

* compute ↓
* activation memory ↓
* jump ↑
* receptive field grows faster
* broader contextual processing

But:

> **spatial detail decreases**

---

# ⚖️ 13. Downsampling Trade-Off

Remember:

```text
Efficiency ↑
Context ↑
Spatial Precision ↓
```

So:

> **Downsampling is a trade-off, not a free improvement.**

---

# 🎛️ 14. Common Shape Pattern

A common CNN progression:

```text
224×224×64
↓
112×112×128
↓
56×56×256
↓
28×28×512
```

Usually:

```text
H/W ↓
Channels ↑
RF ↑
Feature Abstraction ↑
```

---

# 🧠 15. Why Channels Increase

More channels provide:

> **greater feature representation capacity**

Early:

```text
High Resolution
Fewer Channels
```

Deep:

```text
Lower Resolution
More Channels
```

But:

> **Channels ≠ Classes**

---

# 🏗️ 16. CNN Stage Pattern

Example:

```text
Stage 1
Conv → Conv → Pool

Stage 2
Conv → Conv → Pool

Stage 3
Conv → Conv → Conv → Pool
```

Often:

* within stage → similar H/W
* between stages → H/W decreases
* channels may increase

---

# 🌍 17. Global Average Pooling

GAP:

> averages every H×W location independently for each channel.

Example:

```text
7×7×512
↓
GAP
↓
512
```

General rule:

`H × W × C → C`

---

# ⭐ 18. Why GAP Is Useful

Compared with large Flatten + Dense heads, GAP:

* drastically reduces parameter count
* creates compact representation
* summarizes each feature channel
* reduces dependence on huge fully connected layers

---

# 📦 19. Flatten

Flatten converts:

`H×W×C`

into:

`H × W × C`

features.

Example:

```text
7×7×512
↓
25,088 features
```

A large dense layer after this can become extremely expensive.

---

# ⚠️ 20. GAP vs Flatten

Example:

`8×8×256`

### Flatten

`16,384 features`

### GAP

`256 features`

So:

```text
Flatten → preserves every activation
GAP     → one summary value per channel
```

---

# 🎯 21. Classification Output

Example:

```text
GAP
↓
512
↓
Linear 512→10
↓
10 logits
```

The final linear layer usually outputs:

> **logits**

not probabilities.

---

# 🧠 22. Logits → Probabilities

For multiclass classification:

```text
Logits
↓
Softmax
↓
Probabilities
```

During training, softmax and cross-entropy are often combined internally by the framework.

---

# 🧮 23. Conv Parameter Formula

For standard convolution:

`K_h × K_w × C_in × C_out`

With bias:

`+ C_out`

Example:

`3×3, 64→128`

Weights:

`3 × 3 × 64 × 128`

`= 73,728`

With bias:

`73,856`

---

# ⭐ 24. Parameters Do NOT Directly Depend on H/W

A:

`3×3, 64→128`

Conv has the same number of filter parameters whether applied to:

`224×224×64`

or:

`28×28×64`

because:

> **the same weights are reused spatially**

---

# 🧮 25. Approximate Conv Compute

Rough multiplication count:

`H_out × W_out × K_h × K_w × C_in × C_out`

So compute depends heavily on:

> spatial resolution.

---

# ⚖️ 26. Parameters vs Compute

### Parameters

Mainly depend on:

```text
Kernel Size
×
C_in
×
C_out
```

### Compute

Also strongly depends on:

```text
H_out
×
W_out
```

Therefore:

> **few parameters ≠ necessarily low compute**

---

# 🧠 27. Why Early Layers Can Be Expensive

Early layers operate on large feature maps.

Example:

`224×224×64`

contains:

`3,211,264 activations`

So early layers may consume large amounts of:

* compute
* activation memory

even with moderate channel counts.

---

# 📦 28. Activation Memory

CNN memory is not only:

> weights.

Feature maps also consume memory.

During training, activations often need to be retained for:

> backpropagation.

So:

```text
High H/W
→ Large Activation Memory
```

---

# 🔄 29. Training vs Inference

Training usually needs:

* parameters
* activations
* gradients
* optimizer state

Inference usually does not need:

* gradients
* optimizer state

Therefore:

> **training memory is usually much larger than inference memory**

---

# 🎛️ 30. Small vs Large Kernels

Simplified equal-channel comparison:

### One 5×5 Conv

`25C²`

### Two 3×3 Conv

`18C²`

Two 3×3 layers also allow:

> another nonlinearity between them.

But:

> **large kernels are not inherently wrong**

---

# 🔗 31. Residual Connections

Residual block:

`y = F(x) + x`

Shortcut:

```text
x ──────────────┐
↓               │
F(x)            │
↓               │
+ ◄─────────────┘
↓
y
```

Purpose:

> improve optimization of deep networks.

---

# ⚠️ 32. Residual Shape Rule

For:

`F(x) + x`

the shapes must be compatible.

If:

```text
x    = 56×56×64
F(x) = 28×28×128
```

direct addition is impossible.

Possible solution:

```text
x
↓
1×1 Conv, S=2
↓
28×28×128
```

Then add.

---

# 🧠 33. Architecture and Feature Hierarchy

A common progression:

```text
High Resolution
Local Features
↓
Lower Resolution
Richer Features
↓
Larger RF
↓
Higher-Level Representation
```

So:

> **architecture organizes how feature hierarchy develops**

---

# 🎯 34. Classification vs Detection vs Segmentation

### Classification

Needs mainly:

> What is present?

Can tolerate more spatial compression.

### Detection

Needs:

> What + Where?

Needs more spatial information.

### Segmentation

Needs:

> Precise spatial output.

Requires strong localization detail.

---

# 🗺️ 35. Why Segmentation Needs Different Design

Classification may reduce:

```text
224
↓
112
↓
56
↓
28
↓
14
↓
7
```

But segmentation often needs to recover high resolution.

Common tools:

* skip connections
* decoder
* upsampling
* multi-scale features

---

# 📱 36. Production Constraints

Architecture selection also depends on:

* latency
* memory
* compute budget
* power usage
* hardware
* model size
* accuracy requirements

So:

> **highest accuracy model is not automatically the best production model**

---

# ⚠️ 37. Parameter Count ≠ Runtime

A model with fewer parameters may still be slower because of:

* large feature maps
* expensive operations
* poor hardware utilization
* memory bandwidth

So evaluate:

> parameters + FLOPs/compute + latency + memory

---

# 🧠 38. How to Read Any CNN

Use this checklist:

```text
1. Input H×W×C
2. Kernel sizes
3. Strides
4. Padding
5. Channel changes
6. Shape after each stage
7. Downsampling locations
8. RF growth
9. Block structure
10. Skip connections
11. Final feature representation
12. Task head
13. Parameters
14. Compute / memory
```

---

# 🧮 39. Full Shape Example

Architecture:

```text
Input: 32×32×3

Conv 3×3, 32, P=1
Pool 2×2
Conv 3×3, 64, P=1
Pool 2×2
Conv 3×3, 128, P=1
GAP
Linear → 10
```

Flow:

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

# 🧮 40. Parameter Example

### Conv 1

`3×3×3×32 + 32`

`= 896`

### Conv 2

`3×3×32×64 + 64`

`= 18,496`

### Conv 3

`3×3×64×128 + 128`

`= 73,856`

### Classifier

`128×10 + 10`

`= 1,290`

---

# ⚡ Quick Recall Table

| Concept      | Key Idea                     |
| ------------ | ---------------------------- |
| Architecture | Complete network design      |
| Stem         | Initial feature extraction   |
| Block        | Group of layers              |
| Stage        | Blocks at similar scale      |
| Backbone     | Main feature extractor       |
| Head         | Task-specific output         |
| Conv         | Learned local transformation |
| Activation   | Nonlinearity                 |
| Pooling      | Fixed downsampling           |
| Strided Conv | Learned downsampling         |
| Channels ↑   | Feature capacity ↑           |
| H/W ↓        | Compute ↓, detail ↓          |
| RF ↑         | Broader context              |
| GAP          | `H×W×C → C`                  |
| Flatten      | All activations → vector     |
| Conv Params  | `K_hK_wC_inC_out`            |
| Conv Compute | Also depends on `H_outW_out` |

---

# 🚫 Interview Traps

### ❌ More layers = better CNN

No.

---

### ❌ More channels = more classes

No.

---

### ❌ Flatten is mandatory

No.

---

### ❌ Every CNN must use pooling

No.

---

### ❌ Fewer parameters = faster model

Not necessarily.

---

### ❌ H/W affects Conv parameter count

Not directly.

---

### ❌ H/W does not affect compute

Wrong.

It strongly affects compute.

---

### ❌ Downsampling is always good

No.

It may destroy useful spatial detail.

---

### ❌ GAP can be used as the main segmentation output

No.

It removes spatial dimensions.

---

### ❌ Architecture determines exactly what each filter learns

No.

Architecture defines structure; training learns weights.

---

# 🎤 30-Second Interview Answer

> **A CNN architecture organizes convolutional blocks, activations, normalization, downsampling stages, and a task-specific head into a complete network. I analyze it by tracking `H×W×C` through every stage, identifying where channels increase and spatial dimensions decrease, following receptive-field growth, and separating parameter count from computational cost. Modern classification CNNs often use a backbone followed by global average pooling and a small linear head, while localization-heavy tasks preserve or recover more spatial detail.**

---

# 🧠 Final Mental Model

```text
INPUT
  ↓
STEM
  ↓
HIGH-RESOLUTION FEATURES
  ↓
REPEATED BLOCKS
  ↓
DOWNSAMPLING
  ↓
H/W ↓
Channels ↑
RF ↑
Feature Richness ↑
  ↓
FINAL FEATURE MAP
  ↓
GAP / TASK-SPECIFIC PROCESSING
  ↓
HEAD
  ↓
OUTPUT
```

---

# ⭐ Remember These 8 Things

1. 🧱 Architecture = full organization, not just Conv layers.
2. 📐 Always track `H×W×C`.
3. 🎛️ Channels often increase as resolution decreases.
4. 👁️ Receptive field grows through depth/downsampling.
5. 🧮 Params depend on `K`, `C_in`, `C_out`; compute also depends on H/W.
6. 🌍 GAP can dramatically reduce classifier parameters.
7. ⚖️ Downsampling trades spatial detail for efficiency/context.
8. 🎯 Architecture must match both the task and deployment constraints.

---

# ⭐ Golden Rule

> **Understand any CNN by tracking shape, channels, receptive field, parameters, compute, and the final task requirement through every stage.**
