# ⚡ Cheat Sheet — Output Dimension in CNNs

> **Core Idea:** Convolution output height and width depend on **input size, kernel size, padding, and stride**. Output channels depend on the **number of filters**.

---

# 📐 Main Formula

For one spatial dimension:

`Output = floor((Input + 2P - K) / S) + 1`

Where:

* `Input` = input size
* `P` = padding
* `K` = kernel size
* `S` = stride

For 2D:

`H_out = floor((H_in + 2P_h - K_h) / S_h) + 1`

`W_out = floor((W_in + 2P_w - K_w) / S_w) + 1`

---

# 🧠 Formula Meaning

Start with:

`Input`

Add padding on both sides:

`Input + 2P`

Subtract space occupied by kernel:

`Input + 2P - K`

Divide by stride:

`/ S`

Take:

`floor()`

to remove incomplete final positions.

Then:

`+1`

to count the first valid kernel position.

---

# 🧩 Basic Example

Input:

`5`

Kernel:

`3`

Padding:

`0`

Stride:

`1`

Calculation:

`floor((5 - 3)/1) + 1`

`= 3`

So:

`5 × 5 → 3 × 3`

---

# 🧱 Padding Example

Input:

`5`

Kernel:

`3`

Padding:

`1`

Stride:

`1`

Calculation:

`floor((5 + 2 - 3)/1) + 1`

`= 5`

So:

`5 × 5 → 5 × 5`

---

# 🟰 Same-Size Shortcut

For odd kernels with:

`stride = 1`

a common choice is:

`P = (K - 1) / 2`

Examples:

| Kernel  | Padding |
| ------- | ------: |
| `3 × 3` |     `1` |
| `5 × 5` |     `2` |
| `7 × 7` |     `3` |

This commonly preserves:

> **H and W**

---

# ➡️ Stride Example

Input:

`32`

Kernel:

`3`

Padding:

`1`

Stride:

`2`

Calculation:

`floor((32 + 2 - 3)/2) + 1`

`= floor(31/2) + 1`

`= 15 + 1`

`= 16`

So:

`32 × 32 → 16 × 16`

---

# ⚠️ Why floor()?

Example:

Input:

`8`

Kernel:

`3`

Padding:

`0`

Stride:

`2`

`(8 - 3)/2 = 2.5`

We cannot have half a valid filter position.

So:

`floor(2.5) + 1`

`= 3`

> **Incomplete final windows are ignored.**

---

# ➕ Why +1?

The division counts:

> how many jumps the kernel can make

But the kernel already has:

> one initial position

So:

`+1`

counts that first valid position.

---

# 🧱 Why 2P?

Padding is added on:

* Left + Right
* Top + Bottom

So for one dimension:

> **Total padding contribution = 2P**

---

# ↔️ Height and Width

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

Result:

`H_out = 28`

`W_out = 20`

So:

> **Output spatial size = 28 × 20**

---

# 📦 Output Channels

Spatial dimensions use:

* Input H/W
* Kernel
* Padding
* Stride

Output channels use:

> **Number of filters**

Example:

Input:

`32 × 32 × 3`

Filters:

`64`

If spatial dimensions are preserved:

> **Output = 32 × 32 × 64**

---

# 🌈 Important Channel Rule

`C_in`

does NOT directly appear in:

`H_out`

or:

`W_out`

Instead:

`C_in`

affects things such as:

* filter depth
* parameter count
* computation

While:

`C_out = Number of Filters`

---

# 🧮 Full Shape Formula

Input:

`H_in × W_in × C_in`

Conv:

* Kernel = `K_h × K_w`
* Padding = `P_h, P_w`
* Stride = `S_h, S_w`
* Filters = `C_out`

Output:

`H_out × W_out × C_out`

Where:

`H_out = floor((H_in + 2P_h - K_h)/S_h) + 1`

`W_out = floor((W_in + 2P_w - K_w)/S_w) + 1`

`C_out = Number of Filters`

---

# 🧮 Common Interview Example

Input:

`224 × 224 × 64`

Conv:

* Kernel = `3 × 3`
* Padding = `1`
* Stride = `2`
* Filters = `128`

Spatial size:

`floor((224 + 2 - 3)/2) + 1`

`= 112`

Therefore:

> **Output = 112 × 112 × 128**

---

# 🏗️ Layer-by-Layer Example

Input:

`224 × 224 × 3`

### Conv 1

`K=3, P=1, S=1, Filters=64`

Output:

`224 × 224 × 64`

### Conv 2

`K=3, P=1, S=2, Filters=128`

Output:

`112 × 112 × 128`

### Conv 3

`K=3, P=1, S=2, Filters=256`

Output:

`56 × 56 × 256`

Common pattern:

Spatial Size
↓ decreases

Channels
↑ increase

---

# ⚡ Fast Interview Method

1. Write input:

`H × W × C_in`

2. Identify:

`K, P, S, Filters`

3. Calculate:

`H_out`

4. Calculate:

`W_out`

5. Set:

`C_out = Filters`

6. Write:

`H_out × W_out × C_out`

---

# 🚫 Common Traps

### Forgetting `+1`

Wrong:

`(Input + 2P - K)/S`

Correct:

`floor((Input + 2P - K)/S) + 1`

---

### Forgetting `2P`

Wrong:

`Input + P`

Correct:

`Input + 2P`

---

### Forgetting floor()

Always discard:

> incomplete final filter positions

---

### Using channels in H/W formula

Wrong:

`Input + C_in + 2P`

Correct:

> Channels are handled separately.

---

### Using input channels as output channels

Wrong:

`C_out = C_in`

Correct:

> `C_out = Number of Filters`

---

# ⚡ Quick Recall Table

| Concept       | Effect                          |
| ------------- | ------------------------------- |
| Larger Kernel | Usually smaller H/W             |
| More Padding  | Preserves/increases H/W         |
| Larger Stride | Smaller H/W                     |
| More Filters  | More output channels            |
| `floor()`     | Removes incomplete final window |
| `+1`          | Counts first position           |
| `2P`          | Padding on both sides           |
| `C_in`        | Does not directly determine H/W |
| `C_out`       | Number of filters               |

---

# 🧠 Mental Model

Input
↓
`+ 2P`
↓
`- K`
↓
`/ S`
↓
`floor()`
↓
`+ 1`
↓
Spatial Output

Then:

Number of Filters
↓
Output Channels

---

# 🎤 Interview One-Liner

> **The convolution output size is `floor((Input + 2P - K)/S) + 1`. Kernel size, padding, and stride determine the spatial dimensions, while the number of filters determines the output channels.**

---

# ⭐ Remember These 5 Things

1. 📐 `Output = floor((Input + 2P - K)/S) + 1`
2. 🧱 Padding contributes `2P`
3. ➡️ Larger stride → smaller H/W
4. 🎛️ Larger kernel → usually smaller H/W if padding is unchanged
5. 📦 Number of filters = output channels

---

# ⭐ Golden Rule

> **Calculate H and W with the convolution formula, calculate channels from filter count, and track multi-layer CNN shapes one layer at a time.**
