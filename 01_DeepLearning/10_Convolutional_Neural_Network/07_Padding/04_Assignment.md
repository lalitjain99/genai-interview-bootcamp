# 💡 Assignment — Padding in CNNs

> **Focus:** Practice why padding is needed, how it affects borders and spatial size, and how `valid` and `same` padding behave.

---

# 🧩 Scenario 1 — No Padding

You have:

Input:

`5 × 5`

Kernel:

`3 × 3`

Stride:

`1`

Padding:

`0`

### Question

What is the output spatial size?

### ✅ Answer

The filter can occupy:

`3` horizontal positions

and:

`3` vertical positions

So:

> **Output = 3 × 3**

---

# 🧱 Scenario 2 — Add Padding

Use the same input:

`5 × 5`

Kernel:

`3 × 3`

Stride:

`1`

Now use:

`padding = 1`

### Question

What is the output spatial size?

### ✅ Answer

Padding adds one value around every side.

Effective spatial size becomes:

`7 × 7`

Now the `3 × 3` kernel can occupy:

`5 × 5`

valid positions.

So:

> **Output = 5 × 5**

---

# 0️⃣ Scenario 3 — Zero Padding

Original input:

1 2 3
4 5 6
7 8 9

Apply:

`padding = 1`

### Question

Write the padded input.

### ✅ Answer

0 0 0 0 0
0 1 2 3 0
0 4 5 6 0
0 7 8 9 0
0 0 0 0 0

The original:

`3 × 3`

input becomes:

> **5 × 5**

---

# ⚠️ Scenario 4 — Border Pixels

A developer says:

> "Padding is only used to keep output size large. It has nothing to do with border pixels."

### Is this correct?

### ✅ Answer

No.

Without padding, border pixels participate in fewer convolution windows.

Padding gives the filter more room to operate near the edges.

So padding helps with both:

* Spatial-size control
* Boundary handling

---

# ✅ Scenario 5 — Valid Padding

A convolution layer uses:

`padding = valid`

### Question

What does this usually mean?

### ✅ Answer

It usually means:

> **No padding**

So:

`P = 0`

The filter operates only where it completely fits inside the input.

---

# 🟰 Scenario 6 — Same Padding

You have:

Input:

`32 × 32`

Kernel:

`3 × 3`

Stride:

`1`

### Question

What padding is commonly used to preserve the same height and width?

### ✅ Answer

For a `3 × 3` kernel:

`padding = 1`

So:

`32 × 32`

↓

`3 × 3 kernel, stride 1, padding 1`

↓

`32 × 32`

---

# 📏 Scenario 7 — Kernel Size and Padding

Suppose stride = `1`.

### Question 1

What padding is commonly used with:

`5 × 5`

kernel to preserve spatial size?

### ✅ Answer

`padding = 2`

---

### Question 2

What padding is commonly used with:

`7 × 7`

kernel?

### ✅ Answer

`padding = 3`

---

### Rule

For odd kernel size:

`P = (K - 1) / 2`

when preserving spatial size with stride 1.

---

# ➡️ Scenario 8 — Same Padding with Larger Stride

Input:

`32 × 32`

Kernel:

`3 × 3`

Stride:

`2`

Padding is chosen using a same-style rule.

A developer says:

> "The output must still be exactly 32 × 32 because the padding is same."

### Is this correct?

### ✅ Answer

No.

With:

`stride > 1`

the output is still downsampled.

So with stride `2`, the output will generally be much smaller, often around:

`16 × 16`

depending on the framework's exact same-padding rule.

So:

> **Same padding does not cancel stride-based downsampling.**

---

# 🎛️ Scenario 9 — Padding vs Kernel Size

A layer uses:

Kernel:

`3 × 3`

Padding:

`1`

A developer says:

> "Padding 1 makes the kernel effectively 5 × 5."

### Is this correct?

### ✅ Answer

No.

The kernel remains:

`3 × 3`

Padding only changes:

> **the effective input boundary**

So:

Kernel Size
≠
Padding

---

# 📦 Scenario 10 — Padding vs Output Channels

Input:

`64 × 64 × 3`

Filters:

`128`

Kernel:

`3 × 3`

Padding:

`1`

Stride:

`1`

### Question

How many output channels are produced?

### ✅ Answer

`128`

because:

> **Number of filters determines output channels**

Padding affects spatial dimensions, not channel count.

---

# 🧠 Scenario 11 — Does Padding Add Real Information?

A developer says:

> "By adding zeros, padding gives the model extra image information."

### Is this correct?

### ✅ Answer

No.

Padding values are artificial boundary values.

They are not:

* Real pixels
* Learned features
* Additional semantic information

Padding simply changes how convolution behaves near the borders.

---

# 🏗️ Scenario 12 — Deep CNN Without Padding

Suppose a deep CNN repeatedly uses:

* `3 × 3` kernels
* stride = `1`
* padding = `0`

### Question

What happens to the spatial dimensions?

### ✅ Answer

They shrink after every convolution.

For one dimension:

`100`

↓

`98`

↓

`96`

↓

`94`

...

This can reduce spatial resolution very quickly.

Using:

`padding = 1`

can preserve the spatial size for these stride-1 `3 × 3` convolutions.

---

# 🔗 Scenario 13 — Padding in Residual Blocks

Residual block:

`y = F(x) + x`

Suppose `F(x)` contains a stride-1 `3 × 3` convolution.

### Question

Why might `padding = 1` be useful?

### ✅ Answer

Because it can preserve:

`H × W`

So:

`F(x)`

and:

`x`

can maintain compatible spatial shapes for element-wise addition.

---

# ✅ True / False

| #  | Statement                                                            | Answer  |
| -- | -------------------------------------------------------------------- | ------- |
| 1  | Padding adds values around the input boundary.                       | ✅ True  |
| 2  | Zero padding is commonly used in CNNs.                               | ✅ True  |
| 3  | `valid` usually means padding is added to preserve size.             | ❌ False |
| 4  | A 3×3 kernel with stride 1 and padding 1 can preserve H/W.           | ✅ True  |
| 5  | Padding directly determines output channels.                         | ❌ False |
| 6  | Padding changes the learned kernel weights.                          | ❌ False |
| 7  | Padding helps border pixels participate in more convolution windows. | ✅ True  |
| 8  | Same padding with stride 2 must preserve exact input size.           | ❌ False |
| 9  | Zero padding adds real image information.                            | ❌ False |
| 10 | Padding can help prevent rapid spatial shrinkage in deep CNNs.       | ✅ True  |

---

# ⭐ Staff Engineer Challenge

## Scenario

You are designing a residual CNN block.

Input:

`128 × 128 × 64`

Main path:

* `3 × 3` convolution
* stride = `1`
* 64 filters

Shortcut:

> Identity connection

A developer configures:

`padding = 0`

---

### Question 1

What problem can this cause?

### ✅ Answer

A stride-1 `3 × 3` convolution with no padding reduces:

`128 × 128`

to:

`126 × 126`

So:

Main Path:

`126 × 126 × 64`

Shortcut:

`128 × 128 × 64`

These tensors cannot be directly added because their spatial dimensions differ.

---

### Question 2

What padding would commonly fix the spatial mismatch?

### ✅ Answer

Use:

`padding = 1`

with:

* kernel = `3 × 3`
* stride = `1`

Then the main path can preserve:

`128 × 128`

---

### Question 3

Does adding padding change the number of filters?

### ✅ Answer

No.

The layer still has:

`64 filters`

and therefore:

> **64 output channels**

---

### Question 4

Does padding guarantee the shortcut and main path always have compatible shapes?

### ✅ Answer

No.

Padding can preserve spatial dimensions in this case, but other differences may still exist, such as:

* Different stride
* Different channel count

Those may require:

> **a projection shortcut**

So padding solves only the relevant spatial-dimension issue here.

---

### Question 5

Why not simply use very large padding everywhere?

### ✅ Answer

Excessive padding can:

* Add large artificial boundary regions
* Increase output size unnecessarily
* Increase computation
* Distort the relative importance of real vs padded values near boundaries

Padding should be chosen intentionally based on:

> **kernel size, stride, and desired output geometry**

---

# 🎯 Final Exercise

Complete the flow:

No Padding
↓
Filter Must Stay __________
↓
Border Pixels Participate in __________ Windows
↓
Output Usually Becomes __________

With Padding
↓
Add __________ Around Input
↓
Filter Can Operate Near __________
↓
Output Size Becomes Easier to __________

### ✅ Answer

No Padding
↓
Filter Must Stay **Inside the Original Input**
↓
Border Pixels Participate in **Fewer** Windows
↓
Output Usually Becomes **Smaller**

With Padding
↓
Add **Boundary Values** Around Input
↓
Filter Can Operate Near **Edges**
↓
Output Size Becomes Easier to **Control**

---

# 🧠 Final Mental Model

Input
↓
Add Padding Around Boundary
↓
Apply Same Learned Filter
↓
More Boundary Positions Become Valid
↓
Better Border Handling
+
Better Spatial-Size Control

---

# ⭐ Golden Rule

> **Padding does not change what the filter learns; it changes the boundary on which the filter operates, helping preserve spatial dimensions and giving border information more opportunity to contribute.**
