# 🎤 Interview — Stride in CNNs

> **Focus:** Understand what stride controls, how it affects feature-map size, computation, spatial resolution, and why strided convolution is used for downsampling.

---

# 🟢 Beginner

## 1️⃣ What is stride in a CNN?

Stride is:

> **The number of pixels the filter moves after each convolution operation.**

Examples:

`stride = 1`
→ move 1 pixel

`stride = 2`
→ move 2 pixels

So:

> **Stride controls the movement step of the filter.**

---

## 2️⃣ What happens when stride = 1?

With:

`stride = 1`

the filter moves one position at a time.

This gives:

* Dense scanning
* More filter positions
* More activations
* Larger feature maps
* Higher spatial resolution

---

## 3️⃣ What happens when stride = 2?

With:

`stride = 2`

the filter moves two positions at a time.

This means:

* Some intermediate positions are skipped
* Fewer convolution operations are performed
* Fewer activations are produced
* Feature-map size becomes smaller

---

## 4️⃣ Does stride change the filter weights?

No.

Stride only changes:

> **where the filter is applied**

It does not change:

> **the learned filter weights**

So:

Filter Weights
→ Same

Movement Step
→ Changes

---

# 🟡 Intermediate

## 5️⃣ Why does increasing stride reduce feature-map size?

Because each valid filter position produces one activation.

So:

Larger Stride
↓
Filter Moves Farther
↓
Fewer Positions Visited
↓
Fewer Activations
↓
Smaller Feature Map

---

## 6️⃣ How does stride affect computation?

Larger stride usually reduces computation.

Why?

Because:

Fewer Positions
↓
Fewer Convolution Calculations

It also produces smaller feature maps, which can reduce the workload of later layers.

---

## 7️⃣ What is the trade-off when using a larger stride?

The main trade-off is:

> **Spatial Detail vs Efficiency**

### Smaller stride

* More detail
* Higher resolution
* More computation

### Larger stride

* Lower resolution
* Less computation
* Possible loss of fine-grained information

---

## 8️⃣ How is stride different from kernel size?

### Kernel Size

Controls:

> **the size of the local region being examined**

Example:

`3 × 3`

### Stride

Controls:

> **how far the filter moves**

Example:

`stride = 2`

So:

> **Kernel Size ≠ Stride**

---

## 9️⃣ Can horizontal and vertical stride be different?

Yes.

Stride can be written as:

`(S_h, S_w)`

For example:

`stride = (2,1)`

means:

* Move 2 positions vertically
* Move 1 position horizontally

Common CNN configurations usually use the same value in both directions.

---

# 🔴 Advanced

## 🔟 How does stride perform downsampling?

Downsampling means:

> **reducing spatial resolution**

A convolution with:

`stride > 1`

samples fewer spatial positions.

For example:

Large Feature Map
↓
Stride 2 Convolution
↓
Smaller Feature Map

So strided convolution can reduce:

`H × W`

while still learning the transformation used for downsampling.

---

## 1️⃣1️⃣ What happens to output channels when stride changes?

Stride does not directly determine output channels.

For example:

Filters = `64`

Stride = `1`

or:

Stride = `2`

Both still produce:

> **64 output channels**

because:

`Number of Filters = C_out`

Stride mainly affects:

> `H_out` and `W_out`

---

## 1️⃣2️⃣ For a 5×5 input and 3×3 kernel with no padding, what happens with stride 1 vs stride 2?

### Stride = 1

Valid starting positions:

`3 × 3`

So output:

`3 × 3`

---

### Stride = 2

Valid starting positions:

`2 × 2`

So output:

`2 × 2`

Therefore:

> Increasing stride reduces the number of valid filter positions.

---

## 1️⃣3️⃣ Why is stride 2 commonly used instead of very large strides?

Because very large strides can skip too much spatial information.

A stride of 2 often gives a practical balance between:

* Downsampling
* Reduced computation
* Retaining useful spatial structure

A very large stride can cause aggressive information loss.

---

## 1️⃣4️⃣ How is strided convolution different from pooling?

Both can reduce spatial resolution.

### Strided Convolution

Uses:

> **learned filter weights**

The downsampling operation itself is learned.

### Pooling

Uses a fixed aggregation rule such as:

* Maximum
* Average

So strided convolution performs:

> **learned downsampling**

while pooling performs:

> **fixed downsampling**

---

## 1️⃣5️⃣ Why can increasing stride affect model accuracy?

Because larger stride samples fewer positions.

That can remove:

* Small features
* Fine spatial details
* Precise localization information

So although larger stride improves efficiency, too much downsampling may reduce predictive performance.

---

# ⭐ Staff Engineer Challenge

## Scenario

You are reviewing two convolutional layers.

### Layer A

Input:

`64 × 64 × 32`

Configuration:

* `3 × 3` kernel
* `64 filters`
* `stride = 1`

### Layer B

Same input and filters, but:

`stride = 2`

A developer says:

> "Layer B should produce fewer output channels because it skips more locations."

---

### Question 1

Is the developer correct?

### ✅ Answer

No.

The number of output channels is controlled by:

> **the number of filters**

Both layers have:

`64 filters`

So both produce:

> **64 output channels**

Stride only affects the spatial dimensions.

---

### Question 2

Which layer produces the smaller spatial feature map?

### ✅ Answer

> **Layer B**

because:

`stride = 2`

causes the filter to visit fewer spatial positions.

---

### Question 3

Which layer requires fewer convolution operations?

### ✅ Answer

> **Layer B**

because fewer positions are evaluated.

---

### Question 4

Which layer is more likely to preserve fine spatial details?

### ✅ Answer

> **Layer A**

because:

`stride = 1`

samples the input more densely.

---

### Question 5

Why might you intentionally choose Layer B?

### ✅ Answer

Layer B may be preferred when you want to:

* Downsample the feature map
* Reduce memory usage
* Reduce computation
* Increase efficiency in deeper layers

But this comes with reduced spatial resolution.

---

# 🎯 30-Second Interview Answer

> **Stride is the number of pixels a convolutional filter moves after each operation. A stride of 1 scans densely, while larger strides visit fewer locations and produce smaller feature maps. This reduces computation and can perform downsampling, but it also lowers spatial resolution and may remove fine-grained information. Stride affects output height and width, while the number of filters determines output channels.**

---

# 🧠 Interview Mental Model

Filter Position
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
Fewer Positions
↓
Smaller Feature Map
↓
Lower Computation
↓
Less Spatial Detail

---

# ⭐ Golden Rule

> **Stride controls spatial movement, not filter count: larger stride reduces output height and width by sampling fewer locations, while output channels still depend on the number of filters.**
