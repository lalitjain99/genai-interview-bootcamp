# ⚡ Cheat Sheet — Filters and Kernels in CNNs

> **Core Idea:** A filter/kernel is a small trainable weight tensor that learns a local pattern response and produces one feature map.

---

# 🎛️ What Is a Filter?

A filter/kernel is:

> **A small set of trainable weights**

Example for a single-channel input:

1   0  -1  
1   0  -1  
1   0  -1

For a `3 × 3` filter:

`9 weights`

---

# 🧠 What Does a Filter Learn?

The arrangement of weights determines what local pattern produces a strong response.

Different filters can learn different responses such as:

- Edges
- Textures
- Curves
- Color transitions
- Feature combinations

Remember:

> Filters are usually learned, not manually designed.

---

# 🔁 How Filters Learn

Initialize Weights  
↓  
Forward Pass  
↓  
Loss  
↓  
Backpropagation  
↓  
Optimizer Update  
↓  
Useful Filter Patterns Emerge

---

# 🗺️ One Filter → One Feature Map

A single filter scans the complete input.

Its spatial responses form:

> **One Feature Map**

So:

`1 filter → 1 feature map`

`32 filters → 32 feature maps`

---

# 📦 Filters and Output Channels

The number of filters determines the number of output channels.

Example:

Input Channels = `3`

Filters = `64`

Output Channels = `64`

Therefore:

> **C_out = Number of Filters**

---

# 🌈 Filter Depth

A standard convolutional filter spans:

> **all input channels**

General filter shape:

`K_h × K_w × C_in`

Example:

RGB input:

`C_in = 3`

Spatial kernel:

`3 × 3`

One filter contains:

`3 × 3 × 3 = 27 weights`

---

# 🧮 Parameter Count Formula

For a standard convolutional layer:

`Weights = K_h × K_w × C_in × C_out`

If one bias is used per filter:

`Total Parameters = (K_h × K_w × C_in × C_out) + C_out`

---

# 🧮 Example

Given:

- Kernel = `3 × 3`
- Input Channels = `3`
- Filters = `16`

Weights:

`3 × 3 × 3 × 16`

= `432`

Biases:

`16`

Total:

`432 + 16`

= `448 parameters`

---

# 📏 Filter Size vs Number of Filters

These are NOT the same.

Example:

`64 filters of size 3 × 3`

### `3 × 3`

= spatial kernel size

### `64`

= number of filters

= number of output channels

---

# 🔁 Parameter Sharing

Within one filter:

> Same weights are reused across spatial locations.

Across different filters:

> Different filters have different weights.

So:

Filter A  
→ Shared spatially

Filter B  
→ Shared spatially

But:

`Weights_A ≠ Weights_B`

---

# 🏗️ Filters Across CNN Depth

### Early Layers

Often respond to simpler patterns:

- Edges
- Color transitions
- Textures

### Deeper Layers

Can respond to combinations such as:

- Shapes
- Object parts
- Higher-level representations

Mental hierarchy:

Pixels  
↓  
Edges  
↓  
Textures / Shapes  
↓  
Parts  
↓  
Higher-Level Features

---

# ⚠️ Filter vs Feature Map

### 🎛️ Filter

Trainable parameter

### 🗺️ Feature Map

Computed activation output

So:

Filter  
→ learns

Feature Map  
→ gets produced

---

# 🔤 Filter vs Kernel

In common CNN terminology:

> **Filter ≈ Kernel**

They are often used interchangeably.

---

# ⚡ Quick Recall Table

| Concept | Quick Meaning |
|---|---|
| Filter / Kernel | Trainable local weight tensor |
| Filter Size | `K_h × K_w` |
| Filter Depth | Equal to `C_in` |
| One Filter | Produces one feature map |
| Number of Filters | Equals `C_out` |
| Parameter Sharing | Same filter reused spatially |
| Different Filters | Learn different weights |
| Feature Map | Output activations |
| Filter Training | Backpropagation + optimizer |
| Deep Filters | Learn more abstract responses |

---

# 🧠 Shape Mental Model

Input:

`H × W × C_in`

One Filter:

`K_h × K_w × C_in`

Multiple Filters:

`C_out filters`

Output:

`H_out × W_out × C_out`

For now:

> **Number of filters controls output depth.**

---

# 🎤 Interview One-Liner

> **A CNN filter is a small trainable weight tensor that spans all input channels, shares its weights across spatial locations, and produces one feature map. Multiple filters learn different responses, so the number of filters determines the number of output channels.**

---

# ⭐ Remember These 6 Things

1. 🎛️ Filter = trainable weights
2. 🌈 Filter spans all input channels
3. 🗺️ One filter → one feature map
4. 📦 Number of filters = output channels
5. 🔁 Same filter weights are shared across space
6. 🧠 Different filters learn different pattern responses

---

# ⭐ Golden Rule

> **One filter spans all input channels, shares its weights across space, and creates one output feature map.**