# 📝 Revision — Multiple Filters and Channels in CNNs

> **Core Idea:** In a standard Conv2D layer, each filter spans **all input channels**, combines their contributions, and produces **one feature map**. Multiple filters therefore produce multiple output channels.

---

# 🌈 1. Input Channels

An input can have multiple channels.

Example:

RGB image:

`32 × 32 × 3`

where:

* `32` = height
* `32` = width
* `3` = input channels

So:

> `C_in = 3`

---

# 🎛️ 2. One Filter Spans All Input Channels

If the kernel size is:

`3 × 3`

and:

`C_in = 3`

then one full filter has shape:

`3 × 3 × 3`

For input with:

`64 channels`

one filter has shape:

`3 × 3 × 64`

So:

> **Filter depth = C_in**

---

# 🧩 3. One Filter Contains Channel Slices

For RGB input, one filter can be viewed as:

Red-channel slice
+
Green-channel slice
+
Blue-channel slice

Each slice has its own weights.

Together they form:

> **one complete filter**

They are not three separate output filters.

---

# ➕ 4. What Happens at One Spatial Position?

At one location:

Channel 1 contribution
+
Channel 2 contribution
+
...
+
Channel `C_in` contribution
+
Bias
↓
One Scalar

So:

> **All input-channel contributions are summed into one output value.**

---

# 🔢 5. One Filter Still Produces One Scalar

Even when the input has many channels:

One Full Filter
+
One Spatial Location
↓
One Scalar

Not:

`C_in` separate final scalars.

This is one of the most important concepts to remember.

---

# 🗺️ 6. One Filter → One Feature Map

The filter produces one scalar at every spatial location.

All those values arranged spatially form:

> **One Feature Map**

Therefore:

`1 full filter → 1 feature map`

---

# 🎛️ 7. Multiple Filters

Suppose the layer contains:

`64 filters`

Each filter has its own learned weights and spans all input channels.

So:

Filter 1
→ Feature Map 1

Filter 2
→ Feature Map 2

...

Filter 64
→ Feature Map 64

Therefore:

> **64 filters → 64 feature maps**

---

# 📦 8. Number of Filters = Output Channels

If:

`C_out = 64`

then the layer contains:

`64 filters`

and produces:

`64 output channels`

So:

> **C_out = Number of Filters**

---

# 🧠 9. C_in vs C_out

### `C_in`

Number of channels entering the layer

### `C_out`

Number of channels produced by the layer

Example:

Input:

`32 × 32 × 3`

Filters:

`64`

Then:

`C_in = 3`

`C_out = 64`

---

# 🔄 10. Output Channels Become Next Layer Input Channels

Suppose Layer 1 outputs:

`32 × 32 × 64`

Then Layer 2 receives:

> **64 input channels**

So if Layer 2 uses:

`3 × 3`

filters, each full filter has shape:

`3 × 3 × 64`

If Layer 2 has:

`128 filters`

then it produces:

> **128 output channels**

---

# 🏗️ 11. Deeper Filters Combine Learned Features

The first layer may combine:

* Red
* Green
* Blue

Deeper layers combine:

> **learned feature maps**

For example:

Edge-like feature
+
Texture-like feature
+
Shape-like feature
↓
New complex feature

This supports:

> **hierarchical feature learning**

---

# 🧮 12. Parameter Count for One Filter

General filter shape:

`K_h × K_w × C_in`

So weights per filter:

`K_h × K_w × C_in`

Example:

Kernel:

`3 × 3`

Input channels:

`3`

Weights:

`3 × 3 × 3`

= `27`

If bias is used:

`27 + 1`

= `28 parameters`

---

# 🧮 13. Parameter Count for Full Layer

General weight count:

`K_h × K_w × C_in × C_out`

If one bias is used per filter:

`Total Parameters = K_h × K_w × C_in × C_out + C_out`

---

# 🧮 14. Example — RGB Input

Input:

`32 × 32 × 3`

Conv:

* Kernel = `3 × 3`
* Filters = `64`

Weights:

`3 × 3 × 3 × 64`

= `1,728`

Biases:

`64`

Total parameters:

> **1,792**

---

# 🧮 15. Example — Deeper Layer

Input:

`32 × 32 × 64`

Conv:

* Kernel = `3 × 3`
* Filters = `128`

Weights:

`3 × 3 × 64 × 128`

= `73,728`

Biases:

`128`

Total:

> **73,856 parameters**

---

# 🎛️ 16. Filter Bank Shape

The full convolutional weight tensor can be thought of conceptually as:

`K_h × K_w × C_in × C_out`

Example:

`3 × 3 × 64 × 128`

means:

* spatial kernel = `3 × 3`
* input channels = `64`
* output filters = `128`

---

# 🔁 17. Parameter Sharing

Each complete filter is reused:

> **across spatial locations**

For example:

Filter 1
→ same full weights everywhere

Filter 2
→ its own full weights everywhere

So:

> **parameter sharing is spatial**

Different filters still learn different parameters.

---

# ⚠️ 18. Important Misconception

### Wrong

> One filter produces one feature map per input channel.

### Correct

One standard filter spans all input channels.

Their contributions are summed.

So:

> **One full filter produces one final output feature map.**

---

# ⚠️ 19. Input Channels Do NOT Equal Number of Filters

Example:

Input:

`32 × 32 × 3`

Filters:

`64`

Correct:

* Input channels = `3`
* Filters = `64`
* Output channels = `64`

So:

> `C_in` and `C_out` are independent architecture dimensions.

---

# ⚖️ 20. More Channels Mean More Cost

Parameter count is proportional to:

`C_in × C_out`

So increasing either:

* input channels
* output filters

increases:

* parameters
* computation
* memory requirements

---

# 🧠 21. Full Shape Mental Model

Input:

`H × W × C_in`

One filter:

`K_h × K_w × C_in`

At one location:

All Channel Contributions
↓
One Scalar

Across space:

One Feature Map

Use:

`C_out` filters

↓

`C_out` feature maps

Final output:

`H_out × W_out × C_out`

---

# ⚡ Quick Recall Table

| Concept                    | Meaning                    |
| -------------------------- | -------------------------- |
| `C_in`                     | Input channels             |
| Filter Depth               | `C_in`                     |
| One Filter Shape           | `K_h × K_w × C_in`         |
| One Filter at One Position | One scalar                 |
| One Full Filter            | One feature map            |
| `C_out`                    | Number of filters          |
| Output Channels            | `C_out`                    |
| Weight Count               | `K_h × K_w × C_in × C_out` |
| Bias Count                 | Usually `C_out`            |
| Next Layer `C_in`          | Previous layer `C_out`     |

---

# 🧠 Mental Model

Multiple Input Channels
↓
One Filter Spans All Channels
↓
Channel Contributions Are Summed
↓
One Scalar Per Location
↓
One Feature Map
↓
Multiple Filters
↓
Multiple Feature Maps
↓
Output Channels

---

# 🎤 30-Second Interview Answer

> **In a standard Conv2D layer, each filter spans all input channels. At each spatial location, it computes contributions from every input channel and sums them to produce one scalar. Scanning across the input creates one feature map. Therefore, one filter produces one output channel, and multiple filters produce multiple output channels. The parameter count is `K_h × K_w × C_in × C_out`, plus `C_out` biases if used.**

---

# ⭐ Remember These 6 Things

1. 🌈 Input can contain many channels
2. 🎛️ One filter spans all `C_in` channels
3. ➕ Channel contributions are summed
4. 🗺️ One full filter → one feature map
5. 📦 Number of filters = `C_out`
6. 🧮 Weights = `K_h × K_w × C_in × C_out`

---

# ⭐ Golden Rule

> **One standard convolution filter spans all input channels, combines them into one response, and produces exactly one output feature map.**
