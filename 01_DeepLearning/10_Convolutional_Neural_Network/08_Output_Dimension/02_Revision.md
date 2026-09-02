# 📝 Revision — Output Dimension in CNNs

> **Core Idea:** Output height and width depend on **input size, kernel size, padding, and stride**, while output channels depend on the **number of filters**.

---

# 📐 1. Main Output Formula

For one spatial dimension:

`Output = floor((Input + 2P - K) / S) + 1`

Where:

* `Input` = input size
* `K` = kernel size
* `P` = padding
* `S` = stride

For 2D convolution:

`H_out = floor((H_in + 2P_h - K_h) / S_h) + 1`

`W_out = floor((W_in + 2P_w - K_w) / S_w) + 1`

---

# 🧠 2. Why This Formula Works

Start with:

`Input`

Padding adds space on both sides:

`Input + 2P`

Kernel occupies:

`K`

Remaining movement range:

`Input + 2P - K`

Stride controls how many jumps fit:

`(Input + 2P - K) / S`

Then:

* `floor()` removes incomplete final positions
* `+1` counts the first valid position

So:

> **Output = floor((Input + 2P - K) / S) + 1**

---

# 🧩 3. Basic Example

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

`= 2 + 1`

`= 3`

So:

`5 × 5`

with a:

`3 × 3`

kernel produces:

> **3 × 3**

---

# 🧱 4. Example with Padding

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

`= 4 + 1`

`= 5`

So:

> **3×3 kernel + stride 1 + padding 1 preserves spatial size**

---

# 🟰 5. Same-Size Rule

For odd kernel sizes with:

`stride = 1`

a common padding choice is:

`P = (K - 1) / 2`

Examples:

| Kernel | Padding |
| ------ | ------: |
| `3`    |     `1` |
| `5`    |     `2` |
| `7`    |     `3` |

This commonly gives:

> **Input H/W = Output H/W**

---

# ➡️ 6. Example with Stride

Input:

`7`

Kernel:

`3`

Padding:

`0`

Stride:

`2`

Calculation:

`floor((7 - 3)/2) + 1`

`= 2 + 1`

`= 3`

So:

> **Output = 3**

Larger stride means:

* fewer valid positions
* fewer activations
* smaller output

---

# ⚠️ 7. Why floor() Is Required

Example:

Input:

`8`

Kernel:

`3`

Padding:

`0`

Stride:

`2`

Without floor:

`(8 - 3)/2 + 1`

`= 2.5 + 1`

But we cannot have:

`3.5`

filter positions.

So:

`floor(2.5) + 1`

`= 3`

The incomplete final window is ignored.

---

# 📉 8. Common Downsampling Example

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

`32 × 32`

becomes:

> **16 × 16**

This is a very common CNN pattern.

---

# ↔️ 9. Height and Width Are Calculated Separately

For non-square inputs or kernels:

calculate:

`H_out`

and:

`W_out`

independently.

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

`= 20`

So:

> **Output spatial size = 28 × 20**

---

# 📦 10. Output Channels Are Separate

Output height and width depend on:

* Kernel
* Padding
* Stride
* Input spatial size

But:

> **Output channels = Number of filters**

Example:

Input:

`32 × 32 × 3`

Filters:

`64`

Kernel:

`3 × 3`

Padding:

`1`

Stride:

`1`

Output:

> **32 × 32 × 64**

---

# 🧠 11. Full Shape Mental Model

Input:

`H_in × W_in × C_in`

Conv Layer:

* Kernel = `K_h × K_w`
* Padding = `P_h, P_w`
* Stride = `S_h, S_w`
* Filters = `C_out`

Output:

`H_out × W_out × C_out`

Where:

`H_out = floor((H_in + 2P_h - K_h)/S_h) + 1`

`W_out = floor((W_in + 2P_w - K_w)/S_w) + 1`

and:

`C_out = Number of Filters`

---

# 🧮 12. Interview Example

Input:

`224 × 224 × 3`

Conv:

* Kernel = `3 × 3`
* Padding = `1`
* Stride = `1`
* Filters = `64`

Height:

`floor((224 + 2 - 3)/1) + 1`

`= 224`

Width:

`224`

Channels:

`64`

Output:

> **224 × 224 × 64**

---

# 🧮 13. Interview Example — Downsampling

Input:

`224 × 224 × 64`

Conv:

* Kernel = `3 × 3`
* Padding = `1`
* Stride = `2`
* Filters = `128`

Height:

`floor((224 + 2 - 3)/2) + 1`

`= floor(223/2) + 1`

`= 111 + 1`

`= 112`

Output:

> **112 × 112 × 128**

---

# 🎯 14. Fast Interview Method

When solving a CNN shape problem:

### Step 1

Write the input:

`H × W × C_in`

### Step 2

Identify:

* Kernel `K`
* Padding `P`
* Stride `S`
* Number of filters

### Step 3

Calculate:

`H_out`

### Step 4

Calculate:

`W_out`

### Step 5

Set:

`C_out = Number of Filters`

### Step 6

Write final shape:

`H_out × W_out × C_out`

---

# ⚠️ 15. Common Calculation Traps

### Forgetting `+1`

Correct:

`floor((Input + 2P - K)/S) + 1`

---

### Forgetting `2P`

Padding is added on both sides.

So:

`Input + 2P`

---

### Mixing Channels into Spatial Formula

Do NOT use `C_in` when calculating H/W.

---

### Using Input Channels as Output Channels

Output channels depend on:

> **filter count**

not input channels.

---

### Forgetting floor()

If stride does not fit evenly:

> discard the incomplete final position

---

# 🏗️ 16. Layer-by-Layer Example

Input:

`224 × 224 × 3`

### Conv 1

* K = 3
* P = 1
* S = 1
* Filters = 64

Output:

`224 × 224 × 64`

---

### Conv 2

* K = 3
* P = 1
* S = 2
* Filters = 128

Output:

`112 × 112 × 128`

---

### Conv 3

* K = 3
* P = 1
* S = 2
* Filters = 256

Output:

`56 × 56 × 256`

Common CNN pattern:

Spatial Size
↓ decreases

Channels
↑ increase

---

# ⚡ Quick Recall Table

| Concept        | Effect                          |
| -------------- | ------------------------------- |
| Larger Kernel  | Usually smaller H/W             |
| More Padding   | Preserves/increases H/W         |
| Larger Stride  | Smaller H/W                     |
| More Filters   | More output channels            |
| Input Channels | Do not directly affect H/W      |
| `floor()`      | Removes incomplete final window |
| `+1`           | Counts first valid position     |
| `2P`           | Padding on both sides           |

---

# 🧠 Mental Model

Input Size
↓
Add Padding
↓
Subtract Kernel Size
↓
Divide Movement Range by Stride
↓
Floor Incomplete Position
↓
Add First Position
↓
Output Dimension

---

# 🎤 30-Second Interview Answer

> **The convolution output size is `floor((Input + 2P - K) / S) + 1`. Padding increases the usable boundary, kernel size determines the window size, and stride determines how far the filter moves. Height and width are calculated independently, while output channels are simply equal to the number of filters.**

---

# ⭐ Remember These 5 Things

1. 📐 `Output = floor((Input + 2P - K)/S) + 1`
2. 🧱 Padding contributes `2P`
3. ➡️ Larger stride → smaller spatial output
4. 🎛️ Larger kernel → usually smaller output if padding is unchanged
5. 📦 Number of filters = output channels

---

# ⭐ Golden Rule

> **Kernel, padding, and stride determine output height and width; the number of filters determines output channels.**
