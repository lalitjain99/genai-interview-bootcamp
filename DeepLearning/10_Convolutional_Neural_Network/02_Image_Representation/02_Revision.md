# 📝 Revision — Image Representation in CNNs

> **Core Idea:** CNNs process images as multi-dimensional tensors while preserving spatial and channel information.

---

# 🖼️ 1. Image = Numbers

A neural network does not see:

> cat, dog, car

It sees:

> **pixel values**

A grayscale image is a 2D grid of intensity values.

Example:

10   20   30  
40   50   60  
70   80   90

---

# ⚫ 2. Grayscale Image

A grayscale image has one intensity value per pixel.

Typical range:

`0 → black`

`255 → white`

Shape:

`Height × Width`

or explicitly with channel dimension:

`Height × Width × 1`

---

# 🌈 3. RGB Image

RGB images have three channels:

- 🔴 Red
- 🟢 Green
- 🔵 Blue

Shape:

`Height × Width × 3`

Example:

`224 × 224 × 3`

---

# 📐 4. Total Number of Values

For an RGB image:

`Total Values = Height × Width × Channels`

Example:

`224 × 224 × 3`

= `150,528`

---

# 🧱 5. Image as a Tensor

A tensor is a multi-dimensional array.

Examples:

Scalar  
→ 0D

Vector  
→ 1D

Matrix  
→ 2D

RGB Image  
→ 3D tensor

So:

`H × W × C`

represents:

Height × Width × Channels

---

# 📦 6. Batch Dimension

During training, multiple images are usually processed together.

If:

Batch Size = 32

Image Shape:

`224 × 224 × 3`

then one possible batch layout is:

`32 × 224 × 224 × 3`

The batch adds one extra dimension.

---

# 🔄 7. Common Tensor Layouts

Different frameworks may store dimensions differently.

### Channels Last

`N × H × W × C`

### Channels First

`N × C × H × W`

PyTorch commonly uses:

`N × C × H × W`

Example:

`32 × 3 × 224 × 224`

---

# 🎛️ 8. Channels vs Filters

Do not confuse them.

### Input Channels

Represent existing input information.

For RGB:

`3 channels`

### Filters

Are learned pattern detectors.

Example:

Input:

`224 × 224 × 3`

CNN Layer:

`32 filters`

Output:

`32 feature maps`

So:

> **Input channels ≠ Number of filters**

---

# 🗺️ 9. One Filter → One Feature Map

One filter scans the complete input.

At each position:

Local Region  
×  
Same Filter Weights  
↓  
One Output Value

All output values together form:

> **One Feature Map**

So:

`1 filter → 1 feature map`

`32 filters → 32 feature maps`

---

# 🔗 10. Feature Maps Become Channels

Suppose:

Input has:

`3 channels`

The convolutional layer has:

`32 filters`

Then:

`32 filters`

produce:

`32 feature maps`

Those feature maps are stacked together.

So the next layer receives:

`32 channels`

Conceptually:

RGB Image  
`3 channels`  
↓  
Conv Layer  
`32 filters`  
↓  
`32 feature maps`  
↓  
Next Layer Input  
`32 channels`

---

# 🧠 11. Raw Channels vs Learned Channels

At the beginning:

Channels may represent:

🔴 Red  
🟢 Green  
🔵 Blue

After convolution:

Channels represent:

- Learned Feature 1
- Learned Feature 2
- Learned Feature 3
- ...

So deeper CNN channels are:

> **Learned feature representations**

not necessarily physical colors.

---

# 🔍 12. Why CNN Keeps Spatial Structure

CNNs preserve:

- 📍 Location
- 🧩 Neighborhood relationships
- 🌈 Channel information

Instead of immediately flattening:

Image  
↓  
`H × W × C`

CNN keeps the tensor structure so filters can inspect local regions.

---

# ⚠️ Key Distinctions

| Concept | Meaning |
|---|---|
| Pixel | Numerical image value |
| Channel | One component of input/feature representation |
| Filter | Learned pattern detector |
| Feature Map | Output produced by one filter |
| Output Channels | Number of feature maps |
| Batch | Multiple images processed together |

---

# 🧠 Mental Model

Raw Image  
↓  
`H × W × C`  
↓  
CNN Filters  
↓  
Feature Maps  
↓  
Stack Feature Maps  
↓  
New Channels  
↓  
Input to Next CNN Layer

---

# 🎤 30-Second Interview Answer

> **Images are represented as tensors of pixel values. A grayscale image usually has one channel, while an RGB image has three. The image can be represented as height × width × channels, and a batch adds another dimension. In a CNN, each filter produces one feature map, and multiple feature maps become the output channels that are passed to the next convolutional layer.**

---

# ⭐ Remember These 6 Points

1. 🖼️ Image = numerical tensor
2. ⚫ Grayscale = 1 channel
3. 🌈 RGB = 3 channels
4. 📦 Batch adds another dimension
5. 🎛️ One filter produces one feature map
6. 🔗 Feature maps become channels for the next layer

---

# ⭐ Golden Rule

> **Raw image channels describe the input; learned feature-map channels describe what the CNN has discovered.**