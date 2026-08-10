# 💡 Assignment — Image Representation in CNNs

> **Focus:** Apply the concepts of image shape, channels, batches, filters, and feature maps.

---

# 🧩 Scenario 1 — Grayscale Image

You have a grayscale image with:

- Height = `28`
- Width = `28`

### Question 1

How many pixel values does the image contain?

### ✅ Answer

`28 × 28 = 784`

So the image contains:

> **784 pixel values**

---

### Question 2

What is the image shape if we explicitly include the channel dimension?

### ✅ Answer

`28 × 28 × 1`

because grayscale has:

> **1 channel**

---

# 🌈 Scenario 2 — RGB Image

You have an RGB image with shape:

`64 × 64 × 3`

### Question 1

What does each dimension represent?

### ✅ Answer

- Height = `64`
- Width = `64`
- Channels = `3`

The 3 channels are:

- 🔴 Red
- 🟢 Green
- 🔵 Blue

---

### Question 2

How many total values are present?

### ✅ Answer

`64 × 64 × 3`

= `12,288`

---

# 📦 Scenario 3 — Batch Representation

A training batch contains:

`32 RGB images`

Each image has size:

`224 × 224`

### Question

What could the batch shape look like in PyTorch?

### ✅ Answer

PyTorch commonly uses:

`N × C × H × W`

So:

`32 × 3 × 224 × 224`

where:

- `N = 32`
- `C = 3`
- `H = 224`
- `W = 224`

---

# 🔄 Scenario 4 — Reading Tensor Shapes

You receive this tensor:

`16 × 64 × 32 × 32`

The framework uses:

`N × C × H × W`

### Question

Interpret the dimensions.

### ✅ Answer

- Batch Size = `16`
- Channels = `64`
- Height = `32`
- Width = `32`

Since there are 64 channels, this is most likely:

> **an intermediate CNN representation**

rather than a raw RGB image.

---

# 🎛️ Scenario 5 — Filters and Output Channels

Input:

`128 × 128 × 3`

A convolutional layer contains:

`24 filters`

### Question 1

How many feature maps will be produced?

### ✅ Answer

`24 feature maps`

because:

> **1 filter → 1 feature map**

---

### Question 2

How many output channels will the layer have?

### ✅ Answer

`24 output channels`

So:

`Number of Filters = Number of Output Channels`

---

# 🔗 Scenario 6 — Next Layer Input

Suppose:

Conv Layer 1 receives:

`3 input channels`

and has:

`32 filters`

### Question 1

How many output feature maps are produced?

### ✅ Answer

`32`

---

### Question 2

How many input channels does Conv Layer 2 receive?

### ✅ Answer

`32`

because:

`32 feature maps`

become:

> **32 input channels for the next layer**

---

# 🧠 Scenario 7 — RGB or Learned Features?

A tensor entering a deeper CNN layer has:

`128 channels`

A developer says:

> "Those 128 channels must represent 128 different colors."

### Is this correct?

### ✅ Answer

No.

Only the original image channels may directly represent physical components like:

- Red
- Green
- Blue

Deeper CNN channels represent:

> **learned feature maps**

They may respond to patterns such as:

- Edges
- Textures
- Shapes
- Higher-level features

---

# 🗺️ Scenario 8 — Pixel vs Feature-Map Value

Suppose a feature map contains:

5   2   0  
9   8   1  
1   0   3

### Question

What might the value `9` represent?

### ✅ Answer

It is an:

> **activation produced by a filter at that spatial location**

A larger value may mean the filter responded strongly to the local input pattern at that position.

It is not a raw image pixel value.

---

# ⚠️ Scenario 9 — Channels vs Filters

Input image:

`3 channels`

CNN layer:

`64 filters`

A developer says:

> "Since the image has 3 channels, the CNN can only create 3 feature maps."

### Is this correct?

### ✅ Answer

No.

The number of output feature maps is determined by:

> **Number of filters**

So:

`64 filters → 64 feature maps`

The input channel count and output channel count are different concepts.

---

# 🧱 Scenario 10 — Why Keep Spatial Shape?

You have two choices:

### Option A

Keep the image as:

`H × W × C`

### Option B

Immediately flatten it into:

`H × W × C values`

### Which is more suitable for convolution?

### ✅ Answer

Option A.

CNNs preserve:

- 📍 Spatial location
- 🧩 Neighborhood relationships
- 🌈 Channel information

This allows filters to operate on local regions.

---

# ✅ True / False

| # | Statement | Answer |
|---|---|---|
| 1 | A grayscale image usually has one channel. | ✅ True |
| 2 | RGB images usually have three channels. | ✅ True |
| 3 | `H × W × C` represents Height, Width, Channels. | ✅ True |
| 4 | PyTorch commonly uses `N × C × H × W`. | ✅ True |
| 5 | One filter produces one feature map. | ✅ True |
| 6 | Number of input channels always equals number of output channels. | ❌ False |
| 7 | Feature maps from one layer become channels for the next layer. | ✅ True |
| 8 | Deeper CNN channels always represent colors. | ❌ False |
| 9 | A feature-map value is the same thing as a raw pixel value. | ❌ False |
| 10 | Batching adds another tensor dimension. | ✅ True |

---

# ⭐ Staff Engineer Challenge

## Scenario

You are reviewing this CNN pipeline:

Input Batch:

`64 × 3 × 128 × 128`

Conv Layer 1:

`32 filters`

Conv Layer 2:

`64 filters`

Assume PyTorch layout:

`N × C × H × W`

Ignore changes in height and width for now.

---

### Question 1

How many channels enter Conv Layer 1?

### ✅ Answer

`3`

because the original images are RGB.

---

### Question 2

How many channels come out of Conv Layer 1?

### ✅ Answer

`32`

because:

`32 filters → 32 feature maps`

---

### Question 3

How many channels enter Conv Layer 2?

### ✅ Answer

`32`

because Conv Layer 1's feature maps become Conv Layer 2's input channels.

---

### Question 4

How many channels come out of Conv Layer 2?

### ✅ Answer

`64`

because:

`64 filters → 64 feature maps`

---

### Final Flow

RGB Input  
`3 channels`  
↓  
Conv 1  
`32 filters`  
↓  
`32 feature channels`  
↓  
Conv 2  
`64 filters`  
↓  
`64 feature channels`

---

# 🎯 Final Exercise

Complete the following:

### 1.

RGB image:

`H × W × ____`

### 2.

PyTorch batch layout:

`____ × ____ × ____ × ____`

### 3.

One filter produces:

____________________

### 4.

32 filters produce:

____________________

### 5.

Output feature maps become:

____________________

for the next convolutional layer.

---

# ✅ Answers

### 1.

`3`

### 2.

`N × C × H × W`

### 3.

One feature map

### 4.

32 feature maps / 32 output channels

### 5.

Input channels

---

# ⭐ Golden Rule

> **The number of input channels tells us what enters a convolutional layer; the number of filters tells us how many feature channels come out.**