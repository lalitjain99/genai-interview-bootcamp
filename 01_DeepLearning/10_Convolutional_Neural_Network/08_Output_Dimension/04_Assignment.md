# 💡 Assignment — Output Dimension in CNNs

> **Focus:** Practice calculating convolution output height, width, and channels using kernel size, padding, stride, and filter count.

---

# 🧩 Scenario 1 — Basic Output Size

Input:

`5 × 5`

Kernel:

`3 × 3`

Padding:

`0`

Stride:

`1`

### Question

What is the output spatial size?

### ✅ Answer

Use:

`Output = floor((Input + 2P - K) / S) + 1`

So:

`floor((5 + 0 - 3)/1) + 1`

`= 2 + 1`

`= 3`

Therefore:

> **Output = 3 × 3**

---

# 🧱 Scenario 2 — Add Padding

Input:

`5 × 5`

Kernel:

`3 × 3`

Padding:

`1`

Stride:

`1`

### Question

What is the output spatial size?

### ✅ Answer

`floor((5 + 2×1 - 3)/1) + 1`

`= floor(4) + 1`

`= 5`

Therefore:

> **Output = 5 × 5**

---

# ➡️ Scenario 3 — Stride 2

Input:

`7 × 7`

Kernel:

`3 × 3`

Padding:

`0`

Stride:

`2`

### Question

What is the output size?

### ✅ Answer

`floor((7 - 3)/2) + 1`

`= floor(4/2) + 1`

`= 2 + 1`

`= 3`

So:

> **Output = 3 × 3**

---

# ⚠️ Scenario 4 — Why floor() Matters

Input:

`8 × 8`

Kernel:

`3 × 3`

Padding:

`0`

Stride:

`2`

### Question

What is the output size?

### ✅ Answer

`floor((8 - 3)/2) + 1`

`= floor(5/2) + 1`

`= 2 + 1`

`= 3`

So:

> **Output = 3 × 3**

The incomplete final kernel position is ignored.

---

# 🟰 Scenario 5 — Same-Size Output

Input:

`32 × 32`

Kernel:

`3 × 3`

Stride:

`1`

### Question

What padding should be used to preserve the spatial size?

### ✅ Answer

For an odd kernel with stride 1:

`P = (K - 1) / 2`

So:

`P = (3 - 1)/2`

`= 1`

Therefore:

> **Padding = 1**

and:

> **Output = 32 × 32**

---

# 📏 Scenario 6 — Larger Kernel

Input:

`64 × 64`

Kernel:

`5 × 5`

Padding:

`2`

Stride:

`1`

### Question

What is the output size?

### ✅ Answer

`floor((64 + 4 - 5)/1) + 1`

`= 63 + 1`

`= 64`

So:

> **Output = 64 × 64**

---

# 📉 Scenario 7 — Common Downsampling

Input:

`64 × 64`

Kernel:

`3 × 3`

Padding:

`1`

Stride:

`2`

### Question

What is the output size?

### ✅ Answer

`floor((64 + 2 - 3)/2) + 1`

`= floor(63/2) + 1`

`= 31 + 1`

`= 32`

Therefore:

> **Output = 32 × 32**

---

# 📦 Scenario 8 — Add Output Channels

Input:

`32 × 32 × 3`

Convolution:

* Kernel = `3 × 3`
* Padding = `1`
* Stride = `1`
* Filters = `64`

### Question

What is the complete output shape?

### ✅ Answer

Spatial size:

`floor((32 + 2 - 3)/1) + 1`

`= 32`

So:

`H_out = 32`

`W_out = 32`

Number of filters:

`64`

So:

> **Output = 32 × 32 × 64**

---

# 🌈 Scenario 9 — Input Channels vs Output Channels

Input:

`28 × 28 × 16`

Layer uses:

`128 filters`

### Question

Assuming padding and stride preserve spatial dimensions, how many output channels are produced?

### ✅ Answer

> **128 output channels**

because:

`C_out = Number of Filters`

The input channel count:

`16`

does not determine the output channel count.

---

# ↔️ Scenario 10 — Non-Square Input

Input:

`28 × 40`

Kernel:

`3 × 5`

Padding:

`1 × 2`

Stride:

`1 × 2`

### Question

Calculate the output spatial dimensions.

### ✅ Answer

Height:

`floor((28 + 2×1 - 3)/1) + 1`

`= floor(27) + 1`

`= 28`

Width:

`floor((40 + 2×2 - 5)/2) + 1`

`= floor(39/2) + 1`

`= 19 + 1`

`= 20`

Therefore:

> **Output = 28 × 20**

---

# 🧮 Scenario 11 — Full Non-Square Output

Input:

`28 × 40 × 16`

Kernel:

`3 × 5`

Padding:

`1 × 2`

Stride:

`1 × 2`

Filters:

`64`

### Question

What is the complete output shape?

### ✅ Answer

From the previous calculation:

`H_out = 28`

`W_out = 20`

Filters:

`64`

So:

> **Output = 28 × 20 × 64**

---

# 🏗️ Scenario 12 — Multi-Layer CNN

Input:

`224 × 224 × 3`

### Conv 1

* Kernel = `3 × 3`
* Padding = `1`
* Stride = `1`
* Filters = `64`

### Conv 2

* Kernel = `3 × 3`
* Padding = `1`
* Stride = `2`
* Filters = `128`

### Question

What is the output after Conv 2?

### ✅ Answer

## Conv 1

Spatial size:

`floor((224 + 2 - 3)/1) + 1`

`= 224`

Output:

`224 × 224 × 64`

---

## Conv 2

Now use:

`224 × 224 × 64`

as the new input.

Spatial size:

`floor((224 + 2 - 3)/2) + 1`

`= floor(223/2) + 1`

`= 111 + 1`

`= 112`

Output channels:

`128`

Therefore:

> **Final Output = 112 × 112 × 128**

---

# 🧠 Scenario 13 — Spot the Mistake

A developer calculates:

Input:

`32`

Kernel:

`3`

Padding:

`1`

Stride:

`1`

as:

`(32 + 1 - 3)/1`

`= 30`

### What are the mistakes?

### ✅ Answer

There are two mistakes.

### Mistake 1

Padding must be counted on both sides:

`2P`

not:

`P`

### Mistake 2

They forgot:

`+1`

Correct:

`floor((32 + 2×1 - 3)/1) + 1`

`= 32`

---

# ⚠️ Scenario 14 — Channels in the Formula

Input:

`64 × 64 × 32`

A developer writes:

`H_out = floor((64 + 32 + 2P - K)/S) + 1`

because there are 32 input channels.

### Is this correct?

### ✅ Answer

No.

Input channels do not belong in the spatial-dimension formula.

Correct formula:

`H_out = floor((H_in + 2P_h - K_h)/S_h) + 1`

Channels affect other things such as:

* Filter depth
* Parameter count
* Computation

but not the direct H/W formula.

---

# ✅ True / False

| #  | Statement                                                                       | Answer  |
| -- | ------------------------------------------------------------------------------- | ------- |
| 1  | Output size depends on input size, kernel, padding, and stride.                 | ✅ True  |
| 2  | Padding contributes `P`, not `2P`, to one spatial dimension.                    | ❌ False |
| 3  | Larger stride usually reduces output spatial size.                              | ✅ True  |
| 4  | More filters increase output height and width.                                  | ❌ False |
| 5  | Number of filters determines output channels.                                   | ✅ True  |
| 6  | Input channels directly appear in the H/W output formula.                       | ❌ False |
| 7  | `floor()` removes incomplete final filter positions.                            | ✅ True  |
| 8  | `+1` counts the first valid filter position.                                    | ✅ True  |
| 9  | Height and width should always be calculated separately when dimensions differ. | ✅ True  |
| 10 | Multi-layer CNN shapes should be calculated layer by layer.                     | ✅ True  |

---

# ⭐ Staff Engineer Challenge

## Scenario

You are given:

Input:

`256 × 256 × 3`

### Layer 1

* Kernel = `7 × 7`
* Padding = `3`
* Stride = `2`
* Filters = `64`

### Layer 2

* Kernel = `3 × 3`
* Padding = `1`
* Stride = `2`
* Filters = `128`

### Layer 3

* Kernel = `3 × 3`
* Padding = `1`
* Stride = `1`
* Filters = `256`

---

### Question 1

What is the output after Layer 1?

### ✅ Answer

`floor((256 + 6 - 7)/2) + 1`

`= floor(255/2) + 1`

`= 127 + 1`

`= 128`

So:

> **128 × 128 × 64**

---

### Question 2

What is the output after Layer 2?

### ✅ Answer

Input is now:

`128 × 128 × 64`

Spatial size:

`floor((128 + 2 - 3)/2) + 1`

`= floor(127/2) + 1`

`= 63 + 1`

`= 64`

So:

> **64 × 64 × 128**

---

### Question 3

What is the output after Layer 3?

### ✅ Answer

Input:

`64 × 64 × 128`

Spatial size:

`floor((64 + 2 - 3)/1) + 1`

`= 64`

Filters:

`256`

Therefore:

> **64 × 64 × 256**

---

### Question 4

Why does Layer 3 preserve spatial size while increasing channels?

### ✅ Answer

Because:

* `K = 3`
* `P = 1`
* `S = 1`

preserves H/W.

At the same time:

`256 filters`

produce:

> **256 output channels**

So spatial geometry and channel depth are controlled separately.

---

### Question 5

What is the overall pattern in this network?

### ✅ Answer

Spatial dimensions:

`256 → 128 → 64 → 64`

Channels:

`3 → 64 → 128 → 256`

So the network is:

> **reducing spatial resolution while increasing feature-channel capacity**

This is a common CNN design pattern.

---

# 🎯 Final Exercise

Complete the formula:

`Output = ______((Input + ______ - ______) / ______) + ______`

### ✅ Answer

`Output = floor((Input + 2P - K) / S) + 1`

---

# 🧠 Final Mental Model

Input
↓
Add `2P`
↓
Subtract `K`
↓
Divide by `S`
↓
Take `floor()`
↓
Add `1`
↓
Output Spatial Dimension

Then:

Number of Filters
↓
Output Channels

---

# ⭐ Golden Rule

> **Calculate height and width with `floor((Input + 2P - K)/S) + 1`, calculate channels from filter count, and solve multi-layer CNNs one layer at a time.**
