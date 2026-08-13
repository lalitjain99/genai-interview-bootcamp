# ⚡ Cheat Sheet — Stride in CNNs

> **Core Idea:** Stride controls how far the filter moves after each convolution step. Larger stride means fewer visited positions, smaller feature maps, and lower spatial resolution.

---

# ➡️ What Is Stride?

Stride is:

> **The movement step of the filter**

Examples:

`stride = 1`
→ move 1 pixel

`stride = 2`
→ move 2 pixels

`stride = 3`
→ move 3 pixels

---

# 🔍 Stride = 1

With:

`stride = 1`

the filter scans densely.

This gives:

* More positions
* More activations
* Larger feature map
* More spatial detail
* More computation

---

# ⏩ Stride = 2

With:

`stride = 2`

the filter skips intermediate starting positions.

This gives:

* Fewer positions
* Fewer activations
* Smaller feature map
* Less spatial detail
* Less computation

---

# 🧩 Simple 1D Example

Input:

`[1, 2, 3, 4, 5, 6, 7]`

Kernel size:

`3`

### Stride = 1

`[1,2,3]`

`[2,3,4]`

`[3,4,5]`

`[4,5,6]`

`[5,6,7]`

### Stride = 2

`[1,2,3]`

`[3,4,5]`

`[5,6,7]`

So:

> **Larger stride → fewer positions**

---

# 📉 Why Output Gets Smaller

One valid filter position:

→ One activation

So:

Increase Stride
↓
Visit Fewer Positions
↓
Produce Fewer Activations
↓
Smaller Feature Map

---

# ⚡ Stride and Computation

Larger stride usually means:

Fewer Convolution Positions
↓
Fewer Multiply-and-Sum Operations
↓
Lower Computation

It also reduces the size of later feature maps.

---

# 🔍 Main Trade-Off

### Smaller Stride

* More detail
* Higher spatial resolution
* More computation

### Larger Stride

* Less detail
* Lower spatial resolution
* Less computation

So:

> **Spatial Detail ↔ Efficiency**

---

# 📉 Stride as Downsampling

A stride larger than 1 can reduce spatial resolution.

Example:

`64 × 64`

↓ stride 2 convolution

approximately

`32 × 32`

Exact size depends on:

* Kernel size
* Padding
* Input dimensions

---

# ↔️ Stride in Two Dimensions

Stride can be written as:

`(S_h, S_w)`

Example:

`stride = (2,1)`

means:

* move 2 positions vertically
* move 1 position horizontally

Common choices:

`(1,1)`

`(2,2)`

---

# 🎛️ Stride Does NOT Change Filter Weights

Changing:

`stride = 1`

to:

`stride = 2`

changes:

> **where the filter is applied**

It does NOT change:

> **the filter parameters**

---

# 📏 Stride vs Kernel Size

### Kernel Size

Controls:

> size of the local region

Example:

`3 × 3`

### Stride

Controls:

> movement distance

Example:

`stride = 2`

So:

> **Kernel Size ≠ Stride**

---

# 🧱 Stride vs Padding

### Stride

Controls:

> filter movement

### Padding

Controls:

> boundary handling

Both can affect output size, but in different ways.

---

# 📦 Stride vs Output Channels

Stride affects:

`H_out`

and:

`W_out`

Number of filters determines:

`C_out`

So:

`64 filters + stride 2`

still gives:

> **64 output channels**

but with smaller spatial dimensions.

---

# 📐 Quick 2D Example

Input:

`5 × 5`

Kernel:

`3 × 3`

No padding.

### Stride = 1

Output:

`3 × 3`

### Stride = 2

Output:

`2 × 2`

So:

`9 activations → 4 activations`

---

# 🏗️ Why Use Strided Convolution?

Strided convolution can:

* Downsample feature maps
* Reduce memory usage
* Reduce computation
* Compress spatial information
* Replace or complement pooling

---

# ⚡ Quick Recall Table

| Concept             | Quick Meaning              |
| ------------------- | -------------------------- |
| Stride              | Filter movement step       |
| Stride 1            | Dense scanning             |
| Stride 2            | Skip more positions        |
| Larger Stride       | Smaller feature map        |
| Smaller Feature Map | Less computation           |
| Downsampling        | Reduced spatial resolution |
| Kernel Size         | Local region size          |
| Padding             | Boundary handling          |
| Output Channels     | Determined by filter count |

---

# 🧠 Shape Mental Model

Input:

`H × W × C_in`

Convolution with stride:

`S`

Output:

`H_out × W_out × C_out`

Where:

* Stride mainly affects `H_out` and `W_out`
* Number of filters determines `C_out`

---

# 🎤 Interview One-Liner

> **Stride is the number of pixels a convolutional filter moves after each operation. Larger stride means fewer sampled positions, a smaller feature map, lower computation, and reduced spatial resolution.**

---

# ⭐ Remember These 5 Things

1. ➡️ Stride = filter movement step
2. `stride = 1` → dense scanning
3. `stride = 2` → downsampling effect
4. 📉 Larger stride → smaller spatial output
5. ⚖️ Larger stride trades spatial detail for efficiency

---

# ⭐ Golden Rule

> **Stride controls how densely the filter samples space: larger stride means fewer positions, smaller feature maps, and lower spatial resolution.**
