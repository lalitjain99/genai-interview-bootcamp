# 💡 Assignment — Pooling in CNNs

> **Focus:** Practice how max pooling, average pooling, output dimensions, channel preservation, information loss, backpropagation, strided convolution, and Global Average Pooling work.

---

# 🧩 Scenario 1 — Basic Max Pooling

Input feature map:

```text
1  3
5  6
```

Apply:

`2 × 2 Max Pooling`

### Question

What is the output?

### ✅ Answer

The largest value is:

`6`

Therefore:

> **Output = 6**

---

# ➗ Scenario 2 — Basic Average Pooling

Input:

```text
1  3
5  7
```

Apply:

`2 × 2 Average Pooling`

### Question

What is the output?

### ✅ Answer

Average:

`(1 + 3 + 5 + 7) / 4`

`= 16 / 4`

`= 4`

Therefore:

> **Output = 4**

---

# 🔝 Scenario 3 — Full Max Pooling

Input:

```text
1  3  2  4
5  6  1  2
7  2  8  1
3  4  6  5
```

Apply:

* Pool size = `2 × 2`
* Stride = `2`

### Question

What is the output feature map?

### ✅ Answer

Region 1:

```text
1  3
5  6
```

Max:

`6`

Region 2:

```text
2  4
1  2
```

Max:

`4`

Region 3:

```text
7  2
3  4
```

Max:

`7`

Region 4:

```text
8  1
6  5
```

Max:

`8`

Therefore:

```text
6  4
7  8
```

So:

> **4 × 4 → 2 × 2**

---

# 📐 Scenario 4 — Pooling Output Dimension

Input:

`32 × 32`

Pool size:

`2 × 2`

Stride:

`2`

Padding:

`0`

### Question

What is the output spatial size?

### ✅ Answer

Use:

`Output = floor((Input + 2P - K) / S) + 1`

So:

`floor((32 - 2)/2) + 1`

`= 15 + 1`

`= 16`

Therefore:

> **Output = 16 × 16**

---

# 📦 Scenario 5 — What Happens to Channels?

Input:

`32 × 32 × 64`

Apply:

`2 × 2 MaxPool, stride 2`

### Question

What is the complete output shape?

### ✅ Answer

Spatial dimensions:

`32 × 32 → 16 × 16`

Standard pooling operates independently on each channel.

So channels remain:

`64`

Therefore:

> **Output = 16 × 16 × 64**

---

# 🎛️ Scenario 6 — Does Pooling Mix Channels?

Input has:

`128 feature-map channels`

A developer says:

> "Max pooling compares values from all 128 channels and keeps the largest channel response."

### Is this correct?

### ✅ Answer

No.

Standard spatial max pooling usually operates:

> **independently within each channel**

So each channel gets its own pooled feature map.

Pooling does not normally compare channel 1 against channel 2, etc.

Therefore:

> **128 input channels → 128 pooled output channels**

---

# ⚡ Scenario 7 — Reduction in Activations

Input:

`64 × 64 × 128`

Apply:

`2 × 2 pooling, stride 2`

### Question

By approximately what factor does the total number of activations decrease?

### ✅ Answer

Before:

`64 × 64 × 128`

`= 524,288`

After:

`32 × 32 × 128`

`= 131,072`

Ratio:

`524,288 / 131,072`

`= 4`

Therefore:

> **The activation count decreases by approximately 4×**

---

# 🗺️ Scenario 8 — Information Loss

Input region:

```text
0.1  0.2
0.9  0.3
```

Apply max pooling.

### Question

What information is preserved and what is lost?

### ✅ Answer

Preserved:

> **Strongest activation = 0.9**

Lost:

* exact position of `0.9`
* values `0.1`, `0.2`, `0.3`
* detailed spatial structure inside the region

Therefore:

> **Pooling is lossy**

---

# 📍 Scenario 9 — Small Spatial Shift

Original region:

```text
0.9  0.2
0.1  0.3
```

Shifted region:

```text
0.1  0.9
0.2  0.3
```

### Question

What does max pooling produce in both cases?

### ✅ Answer

Original:

`max = 0.9`

Shifted:

`max = 0.9`

So the pooled result does not change.

This illustrates:

> **some local robustness to small spatial shifts**

But it does not imply perfect translation invariance.

---

# ⚠️ Scenario 10 — Negative Values

Input region:

```text
-4  -1
-7  -3
```

Apply max pooling.

### Question

What is the result?

### ✅ Answer

Maximum:

`-1`

Therefore:

> **Output = -1**

Max pooling selects the largest value.

It does not automatically replace negative values with zero.

---

# 🔄 Scenario 11 — Backpropagation Through Max Pooling

Input:

```text
1  8
3  4
```

Max pooling outputs:

`8`

Suppose the upstream gradient is:

`g`

### Question

How is the gradient propagated to the input region?

### ✅ Answer

The gradient is routed to the location that produced the maximum:

```text
0  g
0  0
```

Therefore:

> **Only the max position receives the gradient from this pooled output**

---

# ➗ Scenario 12 — Backpropagation Through Average Pooling

A:

`2 × 2`

average-pooling region receives upstream gradient:

`8`

### Question

How much gradient does each input position receive?

### ✅ Answer

There are:

`4 positions`

So each receives:

`8 / 4`

`= 2`

Gradient map:

```text
2  2
2  2
```

Therefore:

> **Average pooling distributes the gradient equally**

---

# 🧮 Scenario 13 — Trainable Parameters

Input:

`64 × 64 × 256`

Apply:

`2 × 2 MaxPool`

### Question

How many trainable parameters are added?

### ✅ Answer

Standard max pooling contains:

> **0 trainable parameters**

It performs a fixed maximum operation.

---

# 🔄 Scenario 14 — Pooling vs Convolution

A developer says:

> "Pooling is basically convolution with fixed weights."

### Is this a good description?

### ✅ Answer

Not really.

Standard convolution:

* multiplies by learned weights
* sums values
* usually combines input channels
* creates learned feature transformations

Pooling:

* uses a fixed summary operation such as max or average
* usually processes channels independently
* has no learned filters

So:

> **Pooling and convolution may both use local windows, but their operations and roles are different**

---

# 🚀 Scenario 15 — Pooling vs Strided Convolution

Input:

`64 × 64 × 128`

Two options are considered.

### Option A

`2×2 MaxPool, stride 2`

### Option B

`3×3 Conv, padding 1, stride 2, 128 filters`

### Question

What output shape does each produce?

### ✅ Answer

Option A:

`64 × 64 × 128`

↓

> **32 × 32 × 128**

Option B:

Spatial dimension:

`floor((64 + 2 - 3)/2) + 1`

`= 32`

Filters:

`128`

So:

> **32 × 32 × 128**

Both have the same output shape.

But:

* MaxPool is fixed and has no parameters
* Strided convolution learns the downsampling transformation

Therefore:

> **Same shape does not mean same operation**

---

# 🌍 Scenario 16 — Global Average Pooling

Input:

`7 × 7 × 512`

Apply:

Global Average Pooling

### Question

What is the output shape?

### ✅ Answer

Each:

`7 × 7`

feature map becomes:

`1 value`

There are:

`512 channels`

Therefore:

> **Output = 1 × 1 × 512**

or conceptually:

> **512-dimensional vector**

---

# 🌍 Scenario 17 — Local vs Global Average Pooling

Input:

`8 × 8 × 64`

### Option A

`2×2 Average Pool, stride 2`

### Option B

Global Average Pooling

### Question

What are their output shapes?

### ✅ Answer

### Option A

`8 × 8 × 64`

↓

> **4 × 4 × 64**

### Option B

The entire `8×8` area of each channel is averaged.

Output:

> **1 × 1 × 64**

So:

> **Local pooling preserves a smaller spatial map, while GAP removes the spatial dimensions entirely**

---

# ✅ True / False

| #  | Statement                                                           | Answer  |
| -- | ------------------------------------------------------------------- | ------- |
| 1  | Max pooling keeps the largest activation in each window.            | ✅ True  |
| 2  | Average pooling keeps the sum of activations.                       | ❌ False |
| 3  | Standard pooling usually reduces H and W.                           | ✅ True  |
| 4  | Standard pooling usually changes the number of channels.            | ❌ False |
| 5  | Max pooling contains learned filter weights.                        | ❌ False |
| 6  | Pooling is generally lossy.                                         | ✅ True  |
| 7  | Max pooling guarantees complete translation invariance.             | ❌ False |
| 8  | Standard pooling usually operates independently per channel.        | ✅ True  |
| 9  | Gradients cannot flow through pooling because it has no parameters. | ❌ False |
| 10 | Global Average Pooling produces one value per channel.              | ✅ True  |
| 11 | Strided convolution can perform learned downsampling.               | ✅ True  |
| 12 | Max pooling always outputs non-negative values.                     | ❌ False |

---

# ⭐ Staff Engineer Challenge

## Scenario

You are designing a vision model for:

> **pixel-level semantic segmentation**

The current encoder repeatedly uses:

`2×2 MaxPool, stride 2`

Starting input:

`512 × 512`

After six pooling operations:

```text
512
↓
256
↓
128
↓
64
↓
32
↓
16
↓
8
```

An engineer proposes adding two more pooling layers because:

> "Smaller maps always make the model faster, so more pooling is better."

---

## Question 1

Is the argument valid?

### ✅ Answer

Only partially.

Smaller feature maps do reduce:

* computation
* activation memory

But aggressive downsampling also removes:

> **spatial detail**

For semantic segmentation, the model must eventually predict:

> **which class belongs to each pixel**

So excessive pooling can make precise localization much harder.

---

## Question 2

What information is lost through repeated pooling?

### ✅ Answer

Repeated pooling progressively loses:

* exact feature positions
* fine boundaries
* small-object details
* high-resolution spatial structure

This can hurt:

> **pixel-level prediction**

---

## Question 3

Does increasing channel count fully recover the lost spatial information?

### ✅ Answer

No.

Increasing channels can increase:

> **feature representation capacity**

but it does not automatically recover exact spatial positions that were discarded by downsampling.

For example:

`8 × 8 × 1024`

has many feature channels, but its spatial grid is still only:

`8 × 8`

---

## Question 4

What could an architecture do instead of blindly adding more pooling?

### ✅ Answer

Possible design approaches include:

* reducing the number of downsampling stages
* using carefully selected strided convolutions
* keeping higher-resolution feature maps
* combining multiple resolutions
* using skip connections between encoder and decoder stages

The key principle is:

> **balance representation, efficiency, and spatial precision**

---

## Question 5

Why isn't strided convolution automatically guaranteed to solve the problem?

### ✅ Answer

Although strided convolution learns the downsampling operation, it still:

> **reduces spatial resolution**

So spatial information can still be lost.

The difference is that strided convolution learns:

> **how to transform information during downsampling**

whereas pooling uses a fixed summary rule.

---

## Question 6

What design trade-off should a Staff Engineer think about?

### ✅ Answer

The trade-off is:

```text
More Downsampling
↓
Lower Compute
Lower Memory
Larger Effective Context
BUT
Less Spatial Precision
```

For localization-heavy tasks, the architecture must preserve enough high-resolution information while still keeping computation manageable.

---

# 🎯 Final Exercise

Complete the comparison:

| Property                      | Convolution | Pooling |
| ----------------------------- | ----------- | ------- |
| Trainable weights             | ?           | ?       |
| Usually combines channels     | ?           | ?       |
| Downsampling possible         | ?           | ?       |
| Learns feature transformation | ?           | ?       |
| Standard parameter count      | ?           | ?       |

### ✅ Answer

| Property                      | Convolution | Pooling |
| ----------------------------- | ----------- | ------- |
| Trainable weights             | ✅ Yes       | ❌ No    |
| Usually combines channels     | ✅ Yes       | ❌ No    |
| Downsampling possible         | ✅ Yes       | ✅ Yes   |
| Learns feature transformation | ✅ Yes       | ❌ No    |
| Standard parameter count      | `> 0`       | `0`     |

---

# 🧠 Final Mental Model

Feature Maps
↓
Choose Local Window
↓

Max Pooling:

`Keep Strongest Response`

or

Average Pooling:

`Keep Average Response`

↓
Move Window Across H/W
↓
Smaller Spatial Feature Maps
↓
Same Number of Channels
↓
Less Computation + Less Spatial Detail

---

# ⭐ Golden Rule

> **Pooling reduces spatial resolution by summarizing each feature map independently: it saves computation and can add local shift robustness, but every downsampling decision trades away some spatial information.**
