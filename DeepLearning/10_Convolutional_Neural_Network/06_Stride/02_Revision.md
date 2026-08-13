# 📝 Revision — Stride in CNNs

> **Core Idea:** Stride tells us how many pixels the filter moves after each convolution step. Larger stride means fewer visited positions and therefore a smaller feature map.

---

# ➡️ 1. What Is Stride?

Stride is:

> **The movement step of the filter**

If:

`Stride = 1`

the filter moves:

`1 pixel`

If:

`Stride = 2`

the filter moves:

`2 pixels`

So:

> **Stride controls how densely the input is scanned.**

---

# 🔍 2. Stride = 1

With:

`Stride = 1`

the filter moves one position at a time.

This means:

* More positions are visited
* Neighboring regions overlap more
* More activations are produced
* Spatial resolution remains higher

So:

> **Stride 1 gives dense scanning.**

---

# ⏩ 3. Stride = 2

With:

`Stride = 2`

the filter moves two positions at a time.

This means:

* Some intermediate starting positions are skipped
* Fewer locations are visited
* Fewer activations are produced
* Output spatial size becomes smaller

So:

> **Stride 2 performs more sparse spatial sampling.**

---

# 🧩 4. Simple 1D Example

Input:

`[1, 2, 3, 4, 5, 6, 7]`

Kernel size:

`3`

### Stride = 1

Regions:

`[1,2,3]`

`[2,3,4]`

`[3,4,5]`

`[4,5,6]`

`[5,6,7]`

---

### Stride = 2

Regions:

`[1,2,3]`

`[3,4,5]`

`[5,6,7]`

Stride 2 visits fewer positions.

---

# 📉 5. Why Larger Stride Creates Smaller Feature Maps

Remember:

One valid filter position
→ One activation

Therefore:

More Positions
→ More Activations
→ Larger Feature Map

Larger Stride
↓
Fewer Positions
↓
Fewer Activations
↓
Smaller Feature Map

---

# ⚡ 6. Stride and Computation

Every visited location requires a convolution calculation.

So:

Larger Stride
↓
Fewer Filter Positions
↓
Fewer Convolution Operations
↓
Lower Computation

Smaller output feature maps can also reduce computation in later layers.

---

# 🔍 7. Spatial Detail Trade-Off

### Smaller Stride

Example:

`stride = 1`

Provides:

* More spatial detail
* More activations
* Higher resolution

But:

* More computation
* Larger outputs

---

### Larger Stride

Example:

`stride = 2`

Provides:

* Smaller outputs
* Less computation
* Downsampling

But may lose:

> **fine spatial detail**

So stride involves a trade-off:

> **Spatial Resolution ↔ Computational Efficiency**

---

# 📉 8. Stride Can Perform Downsampling

Downsampling means:

> **reducing spatial resolution**

For example:

Large Feature Map
↓
Stride 2 Convolution
↓
Smaller Feature Map

Conceptually:

`H × W`

may become approximately:

`H/2 × W/2`

depending on:

* Kernel size
* Padding
* Exact dimensions

---

# ↔️ 9. Horizontal and Vertical Stride

Stride can be represented as:

`(S_h, S_w)`

where:

`S_h`

= vertical movement

`S_w`

= horizontal movement

Example:

`stride = (2,1)`

means:

* Move 2 positions vertically
* Move 1 position horizontally

Common cases are:

`(1,1)`

and:

`(2,2)`

---

# 🎛️ 10. Stride Does NOT Change Filter Weights

Changing:

`stride = 1`

to:

`stride = 2`

does NOT change:

> **the filter parameters**

It only changes:

> **where the filter is applied**

So:

Filter Weights
→ Same

Movement Step
→ Changes

Feature-Map Size
→ Changes

---

# 📏 11. Stride vs Kernel Size

Do not confuse them.

### Kernel Size

Controls:

> **how large the local region is**

Example:

`3 × 3`

### Stride

Controls:

> **how far the filter moves**

Example:

`2`

So:

> **Kernel Size ≠ Stride**

---

# 🧱 12. Stride vs Padding

### Stride

Controls:

> movement step

### Padding

Controls:

> extra values around input boundaries

Both affect output spatial size, but for different reasons.

---

# 📐 13. Simple 2D Example

Input:

`5 × 5`

Kernel:

`3 × 3`

No padding.

### Stride = 1

Valid positions:

`3 × 3`

Output:

`3 × 3`

---

### Stride = 2

Valid positions:

`2 × 2`

Output:

`2 × 2`

So:

Stride 1
→ `9 activations`

Stride 2
→ `4 activations`

---

# 📦 14. Stride vs Number of Filters

Suppose:

Number of Filters = `64`

Stride = `2`

Stride affects:

> `H_out` and `W_out`

Number of filters determines:

> `C_out`

So output looks like:

`H_out × W_out × 64`

Remember:

> **Stride changes spatial dimensions, not the number of output channels.**

---

# 🏗️ 15. Why CNN Architectures Use Strided Convolution

Strided convolution can be used to:

* Reduce feature-map resolution
* Reduce computation
* Compress spatial information
* Perform downsampling
* Replace or complement pooling

A common architecture pattern is:

Large Height × Width

* Fewer Channels

↓

Smaller Height × Width

* More Channels

---

# ⚡ Quick Recall Table

| Concept             | Meaning                    |
| ------------------- | -------------------------- |
| Stride              | Filter movement step       |
| Stride 1            | Move 1 position            |
| Stride 2            | Move 2 positions           |
| Larger Stride       | Fewer visited positions    |
| Fewer Positions     | Smaller feature map        |
| Smaller Feature Map | Less computation           |
| Downsampling        | Reduced spatial resolution |
| Kernel Size         | Local region size          |
| Padding             | Boundary handling          |
| Number of Filters   | Controls output channels   |

---

# 🧠 Mental Model

Place Filter
↓
Compute Activation
↓
Move by `Stride`
↓
Compute Next Activation
↓
Repeat

Increase Stride
↓
Move Farther
↓
Visit Fewer Positions
↓
Produce Fewer Activations
↓
Smaller Feature Map

---

# 🎤 30-Second Interview Answer

> **Stride is the number of pixels a convolutional filter moves after each operation. A stride of 1 scans the input densely, while a larger stride skips more positions and produces fewer activations. As a result, larger stride reduces the spatial dimensions of the feature map and computation, but it can also reduce spatial detail. Strided convolutions are therefore commonly used for downsampling in CNNs.**

---

# ⭐ Remember These 5 Things

1. ➡️ Stride = movement step
2. `stride = 1` → dense scanning
3. `stride = 2` → fewer visited positions
4. 📉 Larger stride → smaller feature map
5. ⚖️ Larger stride trades spatial detail for efficiency

---

# ⭐ Golden Rule

> **Stride controls how far the filter moves: increasing stride reduces the number of convolution positions and therefore reduces spatial resolution.**
