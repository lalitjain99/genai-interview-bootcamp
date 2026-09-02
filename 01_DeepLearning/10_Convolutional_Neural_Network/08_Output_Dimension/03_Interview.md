# 🎤 Interview — Output Dimension in CNNs

> **Focus:** Understand how kernel size, padding, stride, and filter count determine CNN output shape.

---

# 🟢 Beginner

## 1️⃣ What is the formula for convolution output size?

For one spatial dimension:

`Output = floor((Input + 2P - K) / S) + 1`

Where:

* `Input` = input size
* `P` = padding
* `K` = kernel size
* `S` = stride

For 2D convolution, calculate height and width separately.

---

## 2️⃣ Why do we add `2P` in the formula?

Because padding is added on both sides of a dimension.

For width:

Left Padding
+
Right Padding

So total added width is:

`2P`

Similarly for height:

Top Padding
+
Bottom Padding

---

## 3️⃣ Why is there a `+1` in the formula?

Because the division tells us how many moves or jumps the kernel can make.

But we must also count:

> **the first kernel position**

So we add:

`+1`

---

## 4️⃣ What determines the number of output channels?

The number of:

> **filters**

For example:

`64 filters → 64 output channels`

Kernel size, stride, and padding mainly determine output height and width.

---

# 🟡 Intermediate

## 5️⃣ Calculate the output size for a 5×5 input, 3×3 kernel, stride 1, and no padding.

Given:

`Input = 5`

`K = 3`

`P = 0`

`S = 1`

Formula:

`floor((5 + 0 - 3)/1) + 1`

`= 2 + 1`

`= 3`

So:

> **Output = 3 × 3**

---

## 6️⃣ Calculate the output size for a 5×5 input, 3×3 kernel, stride 1, and padding 1.

Formula:

`floor((5 + 2×1 - 3)/1) + 1`

`= 4 + 1`

`= 5`

So:

> **Output = 5 × 5**

This preserves the spatial size.

---

## 7️⃣ What happens to output size when stride increases?

Larger stride means:

Filter Moves Farther
↓
Fewer Valid Positions
↓
Fewer Activations
↓
Smaller Output

So:

> **Increasing stride generally reduces spatial dimensions.**

---

## 8️⃣ What happens to output size if kernel size increases while padding and stride remain fixed?

A larger kernel occupies more of the input at each position.

Therefore:

> **The output usually becomes smaller.**

From the formula:

`Input + 2P - K`

increasing `K` reduces the available movement range.

---

## 9️⃣ For a 32×32 input, 3×3 kernel, stride 1, and padding 1, what is the output size?

Formula:

`floor((32 + 2 - 3)/1) + 1`

`= 31 + 1`

`= 32`

So:

> **Output = 32 × 32**

If there are 64 filters:

> **Final output = 32 × 32 × 64**

---

# 🔴 Advanced

## 🔟 Why is `floor()` used in the output-dimension formula?

Because the kernel must fit completely at a valid starting position.

If the final stride step would leave part of the kernel outside the usable input region, that position is not counted.

Example:

Input:

`8`

Kernel:

`3`

Stride:

`2`

Padding:

`0`

Calculation:

`(8 - 3)/2 = 2.5`

We cannot have half a filter position.

So:

`floor(2.5) + 1`

`= 3`

---

## 1️⃣1️⃣ Calculate the output for input 32×32, kernel 3×3, padding 1, stride 2.

Formula:

`floor((32 + 2 - 3)/2) + 1`

`= floor(31/2) + 1`

`= 15 + 1`

`= 16`

So:

> **Output = 16 × 16**

This is a common downsampling configuration.

---

## 1️⃣2️⃣ How do you calculate output size for a non-square input?

Calculate height and width separately.

Example:

Input:

`28 × 40`

Kernel:

`3 × 5`

Padding:

`1 × 2`

Stride:

`1 × 2`

Height:

`floor((28 + 2 - 3)/1) + 1`

`= 28`

Width:

`floor((40 + 4 - 5)/2) + 1`

`= floor(39/2) + 1`

`= 19 + 1`

`= 20`

So:

> **Output = 28 × 20**

---

## 1️⃣3️⃣ Does input-channel count affect output height and width?

No.

Input channels affect:

* Filter depth
* Parameter count
* Computation

But they do not directly appear in:

`H_out`

or:

`W_out`

The spatial formula only uses:

* Input spatial size
* Kernel size
* Padding
* Stride

---

## 1️⃣4️⃣ Given input 224×224×64, kernel 3×3, padding 1, stride 2, and 128 filters, what is the output shape?

Height:

`floor((224 + 2 - 3)/2) + 1`

`= floor(223/2) + 1`

`= 111 + 1`

`= 112`

Width:

`112`

Output channels:

`128`

So:

> **Output = 112 × 112 × 128**

---

## 1️⃣5️⃣ What is the difference between spatial output dimensions and output channels?

Spatial dimensions:

`H_out × W_out`

are determined by:

* Input H/W
* Kernel size
* Padding
* Stride

Output channels:

`C_out`

are determined by:

> **number of filters**

So they should always be calculated separately.

---

# ⭐ Staff Engineer Challenge

## Scenario

You are reviewing this CNN:

Input:

`256 × 256 × 3`

### Conv Layer 1

* Kernel = `7 × 7`
* Padding = `3`
* Stride = `2`
* Filters = `64`

### Conv Layer 2

* Kernel = `3 × 3`
* Padding = `1`
* Stride = `2`
* Filters = `128`

A developer claims:

> "The output after Layer 2 should be 128 × 128 × 128 because the input is halved only once."

---

### Question 1

What is the output of Layer 1?

### ✅ Answer

Height:

`floor((256 + 6 - 7)/2) + 1`

`= floor(255/2) + 1`

`= 127 + 1`

`= 128`

So Layer 1 output is:

> **128 × 128 × 64**

---

### Question 2

What is the output of Layer 2?

### ✅ Answer

Layer 2 input spatial size:

`128 × 128`

Height:

`floor((128 + 2 - 3)/2) + 1`

`= floor(127/2) + 1`

`= 63 + 1`

`= 64`

So Layer 2 output is:

> **64 × 64 × 128**

---

### Question 3

Why was the developer's answer wrong?

### ✅ Answer

Because stride-based downsampling happens:

> **at each layer**

Layer 1:

`256 → 128`

Layer 2:

`128 → 64`

You must calculate output shape:

> **layer by layer**

not directly from the original input every time.

---

### Question 4

Why does the channel dimension increase from 64 to 128?

### ✅ Answer

Because Layer 2 uses:

`128 filters`

Therefore:

> **128 output feature maps → 128 output channels**

This is independent of the spatial downsampling.

---

### Question 5

What is the safest way to solve multi-layer CNN shape questions?

### ✅ Answer

For every layer:

1. Take the previous layer's output as the new input.
2. Calculate `H_out`.
3. Calculate `W_out`.
4. Set `C_out = number of filters`.
5. Use that full shape as the next layer's input.

Never skip layers.

---

# 🎯 30-Second Interview Answer

> **The convolution output size is `floor((Input + 2P - K)/S) + 1`. Padding increases the usable spatial boundary, kernel size determines the local window size, and stride determines how far the filter moves. Height and width are calculated separately, while output channels equal the number of filters. In multi-layer CNNs, output shape must be calculated layer by layer.**

---

# 🧠 Interview Mental Model

Input
↓
Add `2P`
↓
Subtract Kernel Size
↓
Divide by Stride
↓
Floor Incomplete Position
↓
Add 1 for First Position
↓
Spatial Output

Then:

Number of Filters
↓
Output Channels

---

# ⭐ Golden Rule

> **Use the convolution formula for height and width, use filter count for channels, and always calculate multi-layer CNN shapes one layer at a time.**
