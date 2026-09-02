# 📝 Revision — Filters and Kernels in CNNs

> **Core Idea:** A filter/kernel is a small set of trainable weights that learns to respond to specific local patterns in the input.

---

# 🎛️ 1. What Is a Filter / Kernel?

A filter is a small weight matrix.

Example:

1   0  -1  
1   0  -1  
1   0  -1

For a `3 × 3` filter:

`3 × 3 = 9 weights`

These weights are trainable parameters.

---

# 🧠 2. What Do Filter Weights Represent?

The arrangement of filter weights determines what type of local pattern produces a strong response.

For example:

1   0  -1  
1   0  -1  
1   0  -1

compares the left and right sides of a local region.

It may respond strongly to:

> **vertical intensity changes**

Different weight patterns produce different responses.

---

# 🔁 3. Filters Are Learned

In real CNNs, filters are usually not manually designed.

Training flow:

Initialize Filter Weights  
↓  
Forward Pass  
↓  
Loss  
↓  
Backpropagation  
↓  
Optimizer Updates Filter Weights

So:

> **The network learns useful filters automatically.**

---

# 🗺️ 4. One Filter → One Feature Map

One filter scans the complete input.

Its responses at all locations form:

> **One Feature Map**

So:

`1 filter → 1 feature map`

`32 filters → 32 feature maps`

---

# 🎛️ 5. Why Multiple Filters?

Images contain many different useful patterns.

Different filters can learn different responses such as:

- Edges
- Textures
- Curves
- Color combinations
- More complex local structures

Therefore, one convolutional layer usually contains multiple filters.

---

# 📦 6. Number of Filters = Output Channels

Suppose:

Input Channels = `3`

Number of Filters = `64`

Then:

`64 filters`

produce:

`64 feature maps`

Therefore:

> **Output Channels = 64**

Remember:

Input Channels  
→ What enters the layer

Filters  
→ Determine what comes out

---

# 📏 7. Filter Size vs Number of Filters

These are different concepts.

Example:

`32 filters of size 3 × 3`

means:

### `3 × 3`

Spatial size of each filter

### `32`

Number of different filters

So:

> **Filter Size ≠ Number of Filters**

---

# 🌈 8. Filters Span All Input Channels

For a grayscale image:

Input Channels = `1`

A `3 × 3` filter has:

`3 × 3 × 1`

weights.

For RGB:

Input Channels = `3`

A spatial `3 × 3` filter actually has:

`3 × 3 × 3`

weights.

So:

> **A filter spans all input channels.**

---

# 🧮 9. Parameter Count Intuition

For RGB input:

Filter Size = `3 × 3`

Input Channels = `3`

Weights per filter:

`3 × 3 × 3 = 27`

If the layer has:

`16 filters`

Total filter weights:

`27 × 16`

= `432`

plus biases.

---

# 🔁 10. Parameter Sharing

Within one filter:

> The same weights are reused across all spatial locations.

But different filters have different weights.

So:

Filter 1  
→ its own weights

Filter 2  
→ different weights

Filter 3  
→ different weights

Within each filter:

> weights are shared spatially

---

# 🏗️ 11. Filters Become More Complex in Deeper Layers

Early CNN layers often learn simpler patterns such as:

- Edges
- Color transitions
- Textures

Deeper layers operate on learned feature maps and can respond to:

- Shapes
- Feature combinations
- Object parts
- More abstract patterns

Conceptually:

Pixels  
↓  
Edges  
↓  
Textures / Shapes  
↓  
Object Parts  
↓  
Higher-Level Features

---

# ⚠️ 12. Filter vs Feature Map

Do not confuse them.

### 🎛️ Filter

Trainable weights

### 🗺️ Feature Map

Activations produced by the filter

So:

Filter  
→ Parameter

Feature Map  
→ Output

---

# 🔤 13. Filter vs Kernel

In CNN discussions:

> **Filter** and **Kernel** are often used interchangeably.

For our learning:

`Filter ≈ Kernel`

Both refer to the learned convolution weights.

---

# ⚡ Quick Recall

| Concept | Meaning |
|---|---|
| Filter / Kernel | Small trainable weight tensor |
| Filter Size | Spatial size such as `3 × 3` |
| Number of Filters | Number of learned detectors |
| One Filter | Produces one feature map |
| Output Channels | Equal to number of filters |
| Input Channels | Channels entering the layer |
| Parameter Sharing | Same filter reused spatially |
| Feature Map | Activations produced by a filter |

---

# 🧠 Mental Model

Input  
↓  
Multiple Filters  
↓  
Each Filter Has Different Learned Weights  
↓  
Each Filter Scans the Input  
↓  
Each Produces One Feature Map  
↓  
Multiple Feature Maps  
↓  
Output Channels

---

# 🎤 30-Second Interview Answer

> **A CNN filter or kernel is a small set of trainable weights applied across local regions of the input. Different filters learn different pattern responses. Each filter spans all input channels, is reused across spatial locations through parameter sharing, and produces one feature map. Therefore, the number of filters determines the number of output channels.**

---

# ⭐ Remember These 5 Things

1. 🎛️ Filter = trainable weights
2. 🧠 Different weights → different pattern response
3. 🗺️ One filter → one feature map
4. 📦 Number of filters = output channels
5. 🌈 Each filter spans all input channels

---

# ⭐ Golden Rule

> **A filter learns what local pattern to respond to, and each filter creates one learned feature channel.**
