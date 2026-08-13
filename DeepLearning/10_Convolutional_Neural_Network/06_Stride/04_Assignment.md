# 💡 Assignment — Stride in CNNs

> **Focus:** Practice how stride changes filter movement, output size, computation, and spatial resolution.

---

# 🧩 Scenario 1 — Basic Stride Movement

A filter is currently positioned at column `1`.

### Question

If:

`stride = 1`

where will the next filter position begin?

### ✅ Answer

It will move:

`1 pixel`

So the next position begins at:

`column 2`

---

### Question

If:

`stride = 2`

where will the next filter position begin?

### ✅ Answer

It will move:

`2 pixels`

So the next position begins at:

`column 3`

---

# 🔍 Scenario 2 — 1D Scanning

Input:

`[1, 2, 3, 4, 5, 6, 7]`

Kernel size:

`3`

Stride:

`1`

### Question

List all valid regions.

### ✅ Answer

`[1,2,3]`

`[2,3,4]`

`[3,4,5]`

`[4,5,6]`

`[5,6,7]`

So:

> **5 convolution positions**

---

# ⏩ Scenario 3 — Same Input, Larger Stride

Use the same input:

`[1, 2, 3, 4, 5, 6, 7]`

Kernel size:

`3`

Stride:

`2`

### Question

List all valid regions.

### ✅ Answer

`[1,2,3]`

`[3,4,5]`

`[5,6,7]`

So:

> **3 convolution positions**

Compared with stride 1:

`5 positions → 3 positions`

---

# 📉 Scenario 4 — Feature-Map Size

You have:

Input:

`5 × 5`

Kernel:

`3 × 3`

No padding.

### Question 1

What is the output size with:

`stride = 1`?

### ✅ Answer

The filter has:

`3 valid horizontal positions`

and:

`3 valid vertical positions`

So:

> **Output = 3 × 3**

---

### Question 2

What is the output size with:

`stride = 2`?

### ✅ Answer

The valid starting positions are reduced to:

`2 horizontally`

and:

`2 vertically`

So:

> **Output = 2 × 2**

---

# ⚡ Scenario 5 — Computation

Two CNN layers use the same:

* Input
* Kernel size
* Number of filters

Layer A:

`stride = 1`

Layer B:

`stride = 2`

### Question

Which layer usually performs fewer convolution operations?

### ✅ Answer

> **Layer B**

because stride 2 visits fewer spatial positions.

So:

Larger Stride
↓
Fewer Positions
↓
Fewer Convolution Operations

---

# 🔍 Scenario 6 — Spatial Detail

A computer-vision model needs very precise localization of tiny defects in images.

You can choose:

### Option A

`stride = 1`

### Option B

`stride = 4`

### Question

Which would generally preserve more fine spatial detail?

### ✅ Answer

> **Option A — stride = 1**

because it samples the input more densely.

A very large stride can skip important small features.

---

# 📉 Scenario 7 — Downsampling

A convolutional layer uses:

`stride = 2`

### Question

What general effect does this have?

### ✅ Answer

It usually:

> **reduces the spatial resolution of the output**

This is called:

> **Downsampling**

Conceptually:

Large Feature Map
↓
Stride 2
↓
Smaller Feature Map

---

# 📦 Scenario 8 — Stride vs Output Channels

A layer contains:

`64 filters`

You change:

`stride = 1`

to:

`stride = 2`

### Question

Does the number of output channels change?

### ✅ Answer

No.

The output channels remain:

`64`

because:

> **Number of filters determines output channels**

Stride mainly affects:

`H_out` and `W_out`

---

# 🎛️ Scenario 9 — Stride vs Filter Weights

A developer says:

> "When stride changes from 1 to 2, the filter must learn a new set of weights."

### Is this correct?

### ✅ Answer

No.

Stride does not define the filter weights.

It only controls:

> **where the same filter is applied**

So changing stride changes movement, not the meaning of the learned parameters.

---

# 📏 Scenario 10 — Kernel Size vs Stride

A layer has:

Kernel size:

`5 × 5`

Stride:

`2`

### Question

What does each number control?

### ✅ Answer

### `5 × 5`

controls:

> **the size of the local region examined**

### `2`

controls:

> **how far the filter moves after each operation**

So:

> **Kernel Size ≠ Stride**

---

# ↔️ Scenario 11 — Different Horizontal and Vertical Strides

A convolution uses:

`stride = (2,1)`

### Question

What does this mean?

### ✅ Answer

It means:

* move `2` positions vertically
* move `1` position horizontally

So the filter downsamples more aggressively in height than in width.

---

# 🧠 Scenario 12 — Architecture Decision

Suppose a CNN has already extracted many useful low-level features.

The feature map is still very large:

`128 × 128 × 64`

You want to reduce computation in later layers.

### Question

Why might a stride-2 convolution be useful?

### ✅ Answer

It can:

* Reduce spatial resolution
* Reduce memory usage
* Reduce later computation
* Still perform a learned transformation while downsampling

So strided convolution can both:

> **extract features and downsample**

---

# ✅ True / False

| #  | Statement                                             | Answer  |
| -- | ----------------------------------------------------- | ------- |
| 1  | Stride controls how far the filter moves.             | ✅ True  |
| 2  | Stride 2 means the filter moves two pixels at a time. | ✅ True  |
| 3  | Larger stride usually produces a larger feature map.  | ❌ False |
| 4  | Larger stride usually reduces computation.            | ✅ True  |
| 5  | Stride changes the number of learned filter weights.  | ❌ False |
| 6  | Stride directly determines output channels.           | ❌ False |
| 7  | Larger stride can reduce fine spatial detail.         | ✅ True  |
| 8  | Strided convolution can perform downsampling.         | ✅ True  |
| 9  | Kernel size and stride mean the same thing.           | ❌ False |
| 10 | Horizontal and vertical stride can be different.      | ✅ True  |

---

# ⭐ Staff Engineer Challenge

## Scenario

You are designing a CNN for medical-image analysis.

Input feature map:

`256 × 256 × 32`

You are considering two layers.

### Layer A

* Kernel = `3 × 3`
* Filters = `64`
* Stride = `1`

### Layer B

* Kernel = `3 × 3`
* Filters = `64`
* Stride = `2`

The task involves detecting very small abnormalities.

---

### Question 1

Which layer preserves more spatial detail?

### ✅ Answer

> **Layer A**

because stride 1 samples the input more densely.

---

### Question 2

Which layer is computationally cheaper?

### ✅ Answer

> **Layer B**

because stride 2 evaluates fewer spatial positions.

---

### Question 3

Do both layers produce the same number of output channels?

### ✅ Answer

Yes.

Both have:

`64 filters`

Therefore both produce:

> **64 output channels**

---

### Question 4

What is the main risk of using Layer B too early?

### ✅ Answer

Small abnormalities may be:

* skipped
* blurred by later processing
* harder to localize

because spatial resolution is reduced early.

So aggressive early downsampling can remove important fine-grained information.

---

### Question 5

When might Layer B still be a good choice?

### ✅ Answer

Layer B can be useful when:

* Some local features have already been extracted
* Lower spatial resolution is acceptable
* Computation needs to be reduced
* The architecture intentionally downsamples at that stage

---

# 🎯 Final Exercise

Complete the flow:

Increase Stride
↓
Filter Moves __________
↓
Visits __________ Positions
↓
Produces __________ Activations
↓
Feature Map Becomes __________
↓
Computation Becomes __________
↓
Spatial Detail Becomes __________

### ✅ Answer

Increase Stride
↓
Filter Moves **Farther**
↓
Visits **Fewer** Positions
↓
Produces **Fewer** Activations
↓
Feature Map Becomes **Smaller**
↓
Computation Becomes **Lower**
↓
Spatial Detail Becomes **Lower**

---

# 🧠 Final Mental Model

Stride
↓
Controls Filter Movement
↓
Controls Number of Spatial Positions
↓
Controls Feature-Map Resolution
↓
Affects Computation and Spatial Detail

---

# ⭐ Golden Rule

> **Larger stride means fewer sampled positions: you gain efficiency and downsampling, but you may lose fine spatial information.**
