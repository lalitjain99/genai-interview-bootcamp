```markdown
# ⚡ Cheat Sheet — Image Representation in CNNs

> **Core Idea:** Images enter CNNs as tensors. Filters transform input channels into feature maps, and those feature maps become the channels for the next layer.

---

# 🖼️ Image Shape

### Grayscale

`H × W × 1`

### RGB

`H × W × 3`

where:

- `H` = Height
- `W` = Width
- `C` = Channels

---

# 🌈 RGB Channels

RGB image contains:

- 🔴 Red
- 🟢 Green
- 🔵 Blue

Example:

`224 × 224 × 3`

Total values:

`224 × 224 × 3 = 150,528`

---

# 🧱 Image = Tensor

A tensor is a multi-dimensional array.

- Scalar → 0D
- Vector → 1D
- Matrix → 2D
- RGB Image → 3D

So:

`H × W × C`

is a 3D image tensor.

---

# 📦 Batch Shape

Multiple images add a batch dimension.

Common layouts:

### Channels Last

`N × H × W × C`

### Channels First

`N × C × H × W`

PyTorch commonly uses:

`N × C × H × W`

Example:

`32 × 3 × 224 × 224`

---

# 🎛️ Channel vs Filter

### Channel

Describes existing input/feature information.

Example:

RGB input:

`3 channels`

### Filter

A learned pattern detector.

Example:

`32 filters`

produce:

`32 feature maps`

---

# 🗺️ Filter → Feature Map

Remember:

> **1 filter → 1 feature map**

At one location:

Local Region  
×  
Filter  
↓  
One Output Value

Across the whole input:

Same Filter  
↓  
Many Output Values  
↓  
One Feature Map

---

# 🔗 Feature Maps Become Channels

Example:

Input:

`3 channels`

Conv Layer:

`32 filters`

Output:

`32 feature maps`

Therefore:

Next Layer Input:

`32 channels`

Flow:

RGB Image  
`3 channels`  
↓  
32 Filters  
↓  
32 Feature Maps  
↓  
32 Channels for Next Layer

---

# 🧠 Raw vs Learned Channels

### First Layer

Channels may represent:

🔴 Red  
🟢 Green  
🔵 Blue

### Deeper Layers

Channels represent:

- Learned edges
- Textures
- Shapes
- Higher-level features

So:

Raw Channels  
→ Input information

Learned Channels  
→ Feature representations

---

# 📐 Quick Example

Input:

`64 × 64 × 3`

Conv layer:

`16 filters`

Output:

`H_out × W_out × 16`

So:

- Input Channels = `3`
- Filters = `16`
- Output Channels = `16`

---

# ⚠️ Common Confusions

### ❌ Wrong

`3 input channels → only 3 output feature maps`

### ✅ Correct

Output feature maps depend on:

> **Number of filters**

---

### ❌ Wrong

Feature map = one convolution calculation

### ✅ Correct

One location produces:

> One scalar value

Same filter across all locations produces:

> One feature map

---

### ❌ Wrong

Deeper channels still represent RGB

### ✅ Correct

Deeper channels represent learned features.

---

# ⚡ Formula Memory

### Total RGB Values

`H × W × 3`

### Number of Feature Maps

`Number of Feature Maps = Number of Filters`

### Output Channels

`Output Channels = Number of Filters`

### Next Layer Input Channels

`Next Input Channels = Previous Output Channels`

---

# 🧠 Mental Model

Image  
↓  
`H × W × C`  
↓  
Filters  
↓  
Feature Maps  
↓  
Stack Feature Maps  
↓  
Output Channels  
↓  
Input Channels for Next Layer

---

# 🎤 Interview One-Liner

> **An image is represented as a tensor of pixel values. In CNNs, filters operate on the input channels, each filter produces one feature map, and those feature maps become the output channels passed to the next convolutional layer.**

---

# ⭐ Remember These 5 Things

1. 🖼️ Grayscale → 1 channel
2. 🌈 RGB → 3 channels
3. 📦 PyTorch commonly uses `N × C × H × W`
4. 🎛️ One filter → one feature map
5. 🔗 Feature maps become channels for the next layer

---

# ⭐ Golden Rule

> **Channels describe what enters a convolutional layer; filters determine how many learned feature channels come out.**
```
