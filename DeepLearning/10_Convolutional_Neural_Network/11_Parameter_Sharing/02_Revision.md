# 📝 Revision — Parameter Sharing in CNNs

> **Core Idea:** Parameter sharing means the **same convolutional filter weights are reused across all spatial locations**. This dramatically reduces parameter count and allows the same learned pattern to be detected anywhere in the input.

---

# 🔁 1. What Is Parameter Sharing?

Suppose a CNN learns a `3 × 3` filter:

```text
w1  w2  w3
w4  w5  w6
w7  w8  w9
```

When the filter moves across the image:

> **the same `w1 ... w9` are reused at every location**

So:

```text
Location 1 → same weights
Location 2 → same weights
Location 3 → same weights
...
```

This is:

> **Parameter Sharing**

---

# 🧠 2. Why Share Parameters?

A useful visual pattern can appear anywhere.

Examples:

* edges
* corners
* textures
* curves
* object parts

Instead of learning:

> edge detector for top-left

and another:

> edge detector for bottom-right

CNNs learn:

> **one detector and reuse it everywhere**

---

# 🎛️ 3. One Filter, Many Locations

A filter may be applied:

* 10 times
* 100 times
* 10,000 times

but it still stores:

> **only one set of parameters**

For a grayscale `3×3` filter:

`9 weights`

Even if used at 900 locations:

> still only `9` learned weights

---

# 🧮 4. Why Parameter Sharing Reduces Parameters

Suppose:

Input:

`32 × 32`

Kernel:

`3 × 3`

Stride:

`1`

Padding:

`0`

Output:

`30 × 30`

So:

`900 spatial positions`

Without sharing:

`900 × 9`

= `8,100 weights`

With sharing:

> **9 weights**

The same 9 weights are reused at all 900 positions.

---

# 🌈 5. Multi-Channel Parameter Sharing

For RGB input:

`C_in = 3`

A `3×3` filter has shape:

`3 × 3 × 3`

That means:

`27 weights`

The entire:

`3 × 3 × 3`

filter is reused across spatial locations.

So:

> **the full multi-channel filter is shared spatially**

---

# ⚠️ 6. Sharing Does NOT Mean Same Weights Across Channels

Suppose one filter has:

`3 × 3 × 64`

weights.

Different channel slices can have:

> **different learned values**

Parameter sharing does NOT mean:

> identical `3×3` weights for every channel

It means:

> the complete `3×3×64` filter is reused across H and W.

---

# 🎛️ 7. Multiple Filters Still Have Different Parameters

Suppose a layer contains:

`64 filters`

Then:

```text
Filter 1 → its own weights
Filter 2 → different weights
...
Filter 64 → different weights
```

But each individual filter is:

> **shared across spatial positions**

So:

> **Parameter sharing is within a filter across space, not across different filters.**

---

# 🗺️ 8. Parameter Sharing Creates Feature Maps

One shared filter is evaluated at many locations:

```text
Location 1 → response
Location 2 → response
Location 3 → response
...
```

Collect those responses spatially:

> **Feature Map**

So a feature map tells us:

> **where and how strongly one shared detector responded**

---

# 🧮 9. Convolution Parameter Formula

For a standard convolution:

`Weights = K_h × K_w × C_in × C_out`

If bias is used:

`Total Parameters = K_h × K_w × C_in × C_out + C_out`

Important:

> **H_out and W_out do not appear in the parameter formula.**

Why?

Because spatial positions reuse the same filters.

---

# 📐 10. Same Conv, Different Input Sizes

Suppose:

* Kernel = `3 × 3`
* `C_in = 3`
* `C_out = 64`

Parameters:

`3 × 3 × 3 × 64 + 64`

= `1,792`

Apply it to:

`32 × 32 × 3`

→ `1,792 parameters`

Apply it to:

`224 × 224 × 3`

→ `1,792 parameters`

Apply it to:

`512 × 512 × 3`

→ `1,792 parameters`

So:

> **input H/W changes computation, not filter parameter count**

---

# ⚡ 11. Parameters vs Computation

Larger image:

> more spatial positions

Therefore:

* more filter applications
* more activations
* more computation
* more memory

But:

> **not more independent convolution weights**

This distinction is very important.

---

# 🏗️ 12. Fully Connected vs Convolution

Fully connected layers generally learn:

> separate connections for different input positions

Convolution instead uses:

> local connectivity + parameter sharing

Example:

Flattened:

`224 × 224 × 3`

= `150,528 inputs`

Connected to:

`1,000 neurons`

requires:

`150,528,000 weights`

before biases.

A convolution with:

* `3×3`
* 3 input channels
* 64 filters

uses only:

`3 × 3 × 3 × 64`

= `1,728 weights`

plus biases.

These layers perform different roles, but this illustrates the parameter efficiency of convolution.

---

# 🧠 13. Local Connectivity + Parameter Sharing

Two key CNN ideas work together:

### 📍 Local Connectivity

Each activation sees only:

> a local input region

### 🔁 Parameter Sharing

The same local filter is:

> reused across spatial locations

Together:

> **far fewer parameters than dense connectivity**

---

# 📍 14. Connection to Translation Equivariance

Because the same filter operates everywhere:

if an input feature moves:

> the corresponding feature-map response tends to move too.

Conceptually:

```text
Input Pattern Moves
↓
Feature Response Moves
```

This property is called:

> **Translation Equivariance**

---

# ⚠️ 15. Equivariance ≠ Invariance

### Translation Equivariance

Input shifts
↓
Output response shifts correspondingly

### Translation Invariance

Input shifts
↓
Output stays unchanged

Convolution is naturally associated with:

> **equivariance**

not perfect invariance.

---

# ⚠️ 16. Practical CNNs Are Not Perfectly Equivariant

Ideal convolution has translation-equivariant structure.

But practical behavior can be affected by:

* stride
* pooling
* padding
* boundaries
* sampling

So avoid saying:

> **Every CNN is perfectly translation-equivariant.**

---

# 🧠 17. Parameter Sharing Is an Inductive Bias

CNNs assume:

> **a useful pattern in one location may also be useful elsewhere**

This assumption is built into the architecture.

That makes parameter sharing an:

> **inductive bias**

It can improve parameter and data efficiency for image-like data.

---

# 📊 18. Shared Parameters ≠ Shared Activations

Same filter:

`W`

can produce different responses.

Example:

```text
Location 1 → 8.2
Location 2 → 0.1
Location 3 → -4.5
```

Why?

Because:

> the local input patches are different.

So:

> **Same weights do not imply same outputs.**

---

# 🔄 19. What Happens During Backpropagation?

The same filter parameter may be used at many spatial locations.

Each use can contribute to its gradient:

```text
Gradient from Location 1
+
Gradient from Location 2
+
Gradient from Location 3
+
...
↓
Gradient for Shared Filter
```

The optimizer then updates:

> **one shared parameter set**

---

# 🧠 20. Shared Across Space, Usually Not Across Layers

Suppose:

Conv1
→ filters A

Conv2
→ filters B

Conv3
→ filters C

Normally:

> **each layer learns its own filters**

Parameter sharing usually means:

> reuse across spatial positions within a convolution layer

not:

> automatically reuse the same filters across different layers.

---

# 🗺️ 21. Parameter Sharing Does Not Remove Location

Same detector is used everywhere.

But its responses remain spatially arranged in the feature map.

So:

Feature in top-left
→ response near top-left

Feature in bottom-right
→ response near bottom-right

Therefore:

> **sharing the detector does not mean losing all position information**

---

# 🧮 22. Critical Parameter-Count Trap

Wrong:

`K_h × K_w × C_in × C_out × H_out × W_out`

Why wrong?

Because:

`H_out × W_out`

describes:

> **how many times filters are applied**

not:

> how many independent parameters exist.

Correct:

> `K_h × K_w × C_in × C_out`

---

# 📦 23. Parameters vs Activations

Example:

Input:

`32 × 32 × 3`

Conv:

* `3×3`
* 64 filters
* stride 1
* padding 1

Output:

`32 × 32 × 64`

### Weights

`3 × 3 × 3 × 64`

= `1,728`

### Output activations

`32 × 32 × 64`

= `65,536`

These are different concepts:

> **Parameters are learned values. Activations are data-dependent outputs.**

---

# ⚡ Quick Recall Table

| Concept                  | Key Rule                                 |
| ------------------------ | ---------------------------------------- |
| Parameter Sharing        | Same filter reused across H/W            |
| Shared Object            | Full `K_h × K_w × C_in` filter           |
| Different Filters        | Have different weights                   |
| Different Channels       | Can have different filter slices         |
| H/W in Parameter Formula | ❌ No                                     |
| Larger H/W               | More compute, not more filter parameters |
| Feature Map              | Responses of one shared filter           |
| Main Benefit             | Parameter efficiency                     |
| Spatial Benefit          | Same pattern detected in many locations  |
| Translation Property     | Equivariance                             |
| Across Layers            | Usually separate parameters              |

---

# 🧠 Mental Model

```text
Learn One Filter
↓
Reuse at Location 1
↓
Reuse at Location 2
↓
Reuse at Location 3
↓
...
↓
Collect Responses
↓
Feature Map
```

Multiple filters:

```text
Filter 1 shared across space → Map 1
Filter 2 shared across space → Map 2
...
Filter C_out shared across space → Map C_out
```

---

# 🎤 30-Second Interview Answer

> **Parameter sharing in CNNs means that the same convolutional filter weights are reused across all spatial positions. This lets one learned detector recognize the same local pattern anywhere in the input, drastically reduces parameter count, and gives convolution its translation-equivariant structure. The number of convolution weights is `K_h × K_w × C_in × C_out`, independent of the number of spatial output positions, although larger feature maps still increase computation.**

---

# ⭐ Remember These 7 Things

1. 🔁 Same filter is reused across H/W
2. 🎛️ Different filters still learn different weights
3. 🌈 The complete multi-channel filter is shared
4. 🧮 H/W do not directly affect convolution parameter count
5. ⚡ Larger H/W increases computation
6. 📍 Parameter sharing supports translation equivariance
7. 🗺️ Feature maps still preserve spatial response locations

---

# ⭐ Golden Rule

> **Learn one detector once and reuse it everywhere: spatial positions share the filter parameters, while their activations remain different and location-dependent.**
