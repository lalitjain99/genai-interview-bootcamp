# ⚡ Cheat Sheet — Multiple Filters and Channels in CNNs

> **Core Idea:** One standard convolution filter spans **all input channels** and produces **one output feature map**. Multiple filters produce multiple output channels.

---

# 🌈 Input Channels

Input shape:

`H × W × C_in`

Example:

`32 × 32 × 3`

means:

* Height = `32`
* Width = `32`
* Input channels = `3`

For RGB:

`C_in = 3`

---

# 🎛️ One Filter Shape

For spatial kernel:

`K_h × K_w`

and:

`C_in` input channels

one full filter has shape:

`K_h × K_w × C_in`

Example:

RGB input + `3 × 3` kernel:

> `3 × 3 × 3`

If input has `64` channels:

> `3 × 3 × 64`

---

# 🧩 Filter Channel Slices

A filter can be thought of as:

Channel 1 slice
+
Channel 2 slice
+
...
+
Channel `C_in` slice

But these slices together form:

> **one complete filter**

They are not separate final filters.

---

# ➕ What Happens at One Location?

Each input channel contributes a value:

`Contribution_1`

`Contribution_2`

...

`Contribution_Cin`

Then:

`Output = Σ Channel Contributions + Bias`

Result:

> **One Scalar**

---

# 🔢 Most Important Rule

Even with many input channels:

> **One full filter at one position → one scalar**

Not:

`C_in` final scalars.

---

# 🗺️ One Filter → One Feature Map

One filter produces:

`1 scalar`

at every spatial position.

Collect all those scalars:

> **One Feature Map**

So:

`1 Filter → 1 Feature Map → 1 Output Channel`

---

# 🎛️ Multiple Filters

If convolution has:

`64 filters`

then:

`64 filters → 64 feature maps`

Therefore:

> `C_out = 64`

General rule:

> **Number of Filters = Number of Output Channels**

---

# 🧠 C_in vs C_out

| Symbol  | Meaning                             |
| ------- | ----------------------------------- |
| `C_in`  | Number of input channels            |
| `C_out` | Number of filters / output channels |

Example:

Input:

`32 × 32 × 3`

Filters:

`64`

Then:

`C_in = 3`

`C_out = 64`

---

# 🔄 Channel Flow Between Layers

Previous layer:

`H × W × 64`

Next layer receives:

> `C_in = 64`

If next layer has:

`128 filters`

then:

> `C_out = 128`

So:

`Previous C_out = Next C_in`

Example:

`3 → 64 → 128 → 256`

---

# 🧮 Weight Count — One Filter

One filter:

`K_h × K_w × C_in`

So:

`Weights per Filter = K_h × K_w × C_in`

Example:

`3 × 3 × 3`

= `27 weights`

---

# 🧮 Weight Count — Full Layer

General formula:

`Weights = K_h × K_w × C_in × C_out`

If bias is used:

`Biases = C_out`

Therefore:

> `Total Parameters = K_h × K_w × C_in × C_out + C_out`

---

# 📐 Example 1 — RGB Input

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

Total:

> **1,792 parameters**

---

# 📐 Example 2 — Deeper Layer

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

# 📦 Complete Output Shape

Input:

`H × W × C_in`

Conv has:

`C_out filters`

Output:

`H_out × W_out × C_out`

Remember:

* `H_out`, `W_out` → kernel, padding, stride
* `C_out` → number of filters

---

# 🏗️ Example

Input:

`32 × 32 × 3`

Conv:

* Kernel = `3 × 3`
* Padding = `1`
* Stride = `1`
* Filters = `64`

Output:

> **32 × 32 × 64**

---

# 🔁 Parameter Sharing

The same complete filter:

`K_h × K_w × C_in`

is reused:

> **across spatial locations**

Different filters:

> have different weights

So parameter sharing means:

`Same Filter → Different H/W Locations`

---

# ⚠️ Parameter Sharing Does NOT Mean

Wrong:

> Same 2D kernel weights must be used for every input channel.

Correct:

Each channel slice can have different weights.

The:

> **complete multi-channel filter**

is what gets reused spatially.

---

# 🧠 Deeper-Layer Intuition

First layer may combine:

RGB channels

↓

Deeper layers combine:

learned feature channels

For example:

Edge-like feature
+
Texture-like feature
+
Shape-like feature
↓
New Learned Feature

This supports:

> **hierarchical feature learning**

---

# ⚖️ Cost of More Channels

Parameter count:

`∝ C_in × C_out`

So increasing:

`C_in`

or:

`C_out`

increases:

* parameters
* computation
* memory use

---

# 🚫 Common Traps

### ❌ One filter per input channel

Wrong.

> One standard filter spans all input channels.

---

### ❌ One filter produces C_in feature maps

Wrong.

> One full filter produces one feature map.

---

### ❌ C_in must equal C_out

Wrong.

They are independent.

Example:

`3 → 64`

is perfectly valid.

---

### ❌ Output channels depend on input channels

Wrong.

> Output channels depend on number of filters.

---

### ❌ Input channels affect H/W output formula

Wrong.

`C_in` affects:

* filter depth
* parameter count
* computation

not the direct spatial-dimension formula.

---

# ⚡ Quick Recall Table

| Concept                    | Rule                       |
| -------------------------- | -------------------------- |
| Input Shape                | `H × W × C_in`             |
| One Filter Shape           | `K_h × K_w × C_in`         |
| One Filter at One Position | One scalar                 |
| One Filter Across Space    | One feature map            |
| Number of Filters          | `C_out`                    |
| Output Shape               | `H_out × W_out × C_out`    |
| Weights                    | `K_h × K_w × C_in × C_out` |
| Biases                     | Usually `C_out`            |
| Next Layer `C_in`          | Previous `C_out`           |
| Parameter Sharing          | Across spatial positions   |

---

# 🧠 Mental Model

Input:

`H × W × C_in`

↓

One Filter:

`K_h × K_w × C_in`

↓

All Channel Contributions

↓

Sum

↓

One Scalar

↓

Scan Across Space

↓

One Feature Map

↓

Use `C_out` Filters

↓

`C_out` Feature Maps

↓

Output:

`H_out × W_out × C_out`

---

# 🎤 Interview One-Liner

> **In standard Conv2D, one filter spans all input channels, sums their contributions into one response, and produces one feature map. Therefore, the number of filters equals the number of output channels, and the weight count is `K_h × K_w × C_in × C_out`.**

---

# ⭐ Remember These 6 Things

1. 🌈 One filter spans all `C_in` channels
2. ➕ Channel contributions are summed
3. 🔢 One filter at one location → one scalar
4. 🗺️ One full filter → one feature map
5. 📦 Number of filters = `C_out`
6. 🧮 Parameters depend on `C_in × C_out`

---

# ⭐ Golden Rule

> **One full multi-channel filter produces exactly one output feature map; multiple filters create multiple output channels.**
