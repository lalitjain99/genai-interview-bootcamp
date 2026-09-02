# ⚡ Cheat Sheet — Parameter Sharing in CNNs

> **Core Idea:** Parameter sharing means the **same convolution filter weights are reused across spatial locations**. This reduces parameter count, lets the same pattern be detected anywhere, and gives convolution a translation-equivariant structure.

---

# 🔁 Parameter Sharing

Same filter:

`W`

is used at:

```text
Location 1 → W
Location 2 → W
Location 3 → W
...
```

Not:

```text
Location 1 → W1
Location 2 → W2
Location 3 → W3
```

So:

> **One learned filter → many spatial applications**

---

# 🎛️ What Exactly Is Shared?

For standard convolution:

One filter shape:

`K_h × K_w × C_in`

The entire multi-channel filter is shared across:

> **H and W positions**

Example:

`3 × 3 × 64`

The same full filter is reused at every valid spatial location.

---

# ⚠️ What Is NOT Shared?

Parameter sharing does NOT mean:

* all filters have identical weights
* all input channels use identical 2D slices
* all CNN layers use the same filters

Instead:

> **Each filter has its own weights, and that filter is reused spatially.**

---

# 🗺️ One Shared Filter → One Feature Map

One filter scans all spatial positions:

```text
Position 1 → response
Position 2 → response
Position 3 → response
...
```

Collect the responses:

> **Feature Map**

So a feature map tells us:

> where and how strongly the same learned detector responded.

---

# 🧠 Why Share Parameters?

Visual patterns can occur anywhere:

* edges
* corners
* textures
* curves
* object parts

Instead of learning:

```text
Edge detector for top-left
Edge detector for center
Edge detector for bottom-right
```

CNN learns:

> **one detector and reuses it everywhere**

---

# 🧮 Parameter Count Formula

For standard convolution:

`Weights = K_h × K_w × C_in × C_out`

If biases are used:

`Total Parameters = K_h × K_w × C_in × C_out + C_out`

Important:

> **H_out and W_out do NOT multiply the parameter count.**

---

# 🚫 Critical Parameter Trap

Wrong:

`K_h × K_w × C_in × C_out × H_out × W_out`

Correct:

> `K_h × K_w × C_in × C_out`

Why?

Because:

`H_out × W_out`

represents:

> **how many times the filters are applied**

not:

> how many different filter weights exist.

---

# 📐 Example

Conv configuration:

* Kernel = `3 × 3`
* `C_in = 3`
* `C_out = 64`

Weights:

`3 × 3 × 3 × 64`

`= 1,728`

Biases:

`64`

Total:

> **1,792 parameters**

Whether input is:

* `32 × 32 × 3`
* `224 × 224 × 3`
* `512 × 512 × 3`

the convolution still has:

> **1,792 trainable parameters**

if kernel and channel counts stay unchanged.

---

# ⚡ Parameters vs Computation

Larger input H/W:

### Parameters

Usually:

> **unchanged**

### Computation

Usually:

> **increases**

because the filter is applied at more spatial positions.

So:

```text
More H/W
↓
More Filter Applications
↓
More Compute + More Activations
```

but not necessarily:

```text
More H/W
↓
More Trainable Filter Weights
```

---

# 🏗️ CNN vs Fully Connected

## Fully Connected

Different spatial inputs generally have:

> separate learned connections

Parameter count can depend heavily on input size.

---

## Convolution

Uses:

* 📍 local connectivity
* 🔁 parameter sharing

Same local detector is reused spatially.

This gives CNNs much better parameter efficiency for image-like data.

---

# 📍 Local Connectivity + Parameter Sharing

Two core CNN ideas:

### Local Connectivity

Look at:

> a small local region

### Parameter Sharing

Reuse:

> the same local detector everywhere

Together:

```text
Local Window
+
Shared Weights
↓
Efficient Spatial Feature Extraction
```

---

# 📍 Translation Equivariance

Because the same detector operates everywhere:

if the input feature shifts:

> the feature response tends to shift as well

Conceptually:

```text
Input Pattern Moves Right
↓
Feature Map Response Moves Right
```

This is:

> **Translation Equivariance**

---

# ⚠️ Equivariance ≠ Invariance

### Equivariance

Input shifts
↓
Output shifts correspondingly

### Invariance

Input shifts
↓
Output remains unchanged

Convolution is naturally associated with:

> **equivariance**

not perfect invariance.

---

# 🧠 Why Sharing Is an Inductive Bias

CNN architecture assumes:

> **a useful local pattern in one location may also be useful elsewhere**

This assumption is built into the model.

Therefore parameter sharing is an:

> **inductive bias**

It can improve:

* parameter efficiency
* data efficiency
* spatial generalization

---

# 📊 Shared Parameters ≠ Shared Activations

Same filter:

`W`

Different local patches:

```text
Location 1 → 8.2
Location 2 → 0.1
Location 3 → -4.5
```

Same weights.

Different outputs.

So:

> **Shared parameters do not mean identical activations.**

---

# 🔄 Backpropagation Through Shared Weights

Suppose one filter weight is used at many positions.

Gradient contributions:

```text
Position 1 gradient
+
Position 2 gradient
+
Position 3 gradient
+
...
↓
Gradient for Shared Weight
```

Then optimizer updates:

> **one stored parameter**

So:

> many spatial uses contribute to one shared weight update.

---

# 🎛️ Multiple Filters

Suppose layer has:

`64 filters`

Then:

```text
Filter 1 → shared across space
Filter 2 → shared across space
...
Filter 64 → shared across space
```

Different filters:

> **different parameters**

Same filter across locations:

> **same parameters**

---

# 🌈 Multi-Channel Sharing

For:

`C_in = 64`

one `3×3` filter has:

`3 × 3 × 64`

weights.

The entire:

> **3×3×64 filter**

is reused across spatial positions.

Do NOT interpret parameter sharing as:

> identical `3×3` kernel slice for every input channel.

---

# 🏗️ Sharing Across Layers?

Normally:

```text
Conv1 → own filters
Conv2 → own filters
Conv3 → own filters
```

Parameter sharing usually refers to:

> **spatial reuse within a convolution layer**

not automatic sharing:

> across different CNN layers.

---

# 🗺️ Does Sharing Remove Position?

No.

The detector is shared, but feature-map coordinates still indicate:

> **where the response occurred**

Example:

Pattern top-left
→ activation top-left

Pattern bottom-right
→ activation bottom-right

So:

> **same detector ≠ lost spatial location**

---

# ⚖️ Limitation of Parameter Sharing

Sharing assumes:

> the same transformation should be useful at different spatial locations.

This may be less ideal when:

* regions have different statistics
* sensor behavior depends on position
* location itself has unique meaning

So parameter sharing is:

> **powerful, but still an architectural assumption**

---

# ⚡ Quick Recall Table

| Concept           | Rule                              |
| ----------------- | --------------------------------- |
| Parameter Sharing | Same filter reused across H/W     |
| Shared Object     | Full `K_h × K_w × C_in` filter    |
| Different Filters | Different weights                 |
| Across Channels   | Slices can differ                 |
| Across Layers     | Usually not shared                |
| Parameter Formula | `K_h × K_w × C_in × C_out`        |
| H/W in Params     | ❌ No                              |
| Larger H/W        | More compute                      |
| Feature Map       | Responses of one shared filter    |
| Spatial Property  | Translation equivariance          |
| Main Benefit      | Parameter efficiency              |
| Shared Weights    | Can produce different activations |

---

# 🚫 Interview Traps

### ❌ Every location has its own filter

No.

> Same filter is reused spatially.

---

### ❌ All filters are identical

No.

> Different filters learn different patterns.

---

### ❌ H_out × W_out multiplies parameter count

No.

> It multiplies filter applications, not independent parameters.

---

### ❌ Larger image always means more convolution parameters

No.

> It usually means more computation.

---

### ❌ Parameter sharing means same weights across channels

No.

> Complete multi-channel filter is shared spatially.

---

### ❌ Sharing means all layers use the same filter

No.

> Different layers usually have different weights.

---

### ❌ Parameter sharing gives translation invariance

Not exactly.

> It supports translation-equivariant behavior.

---

# 🎤 Interview One-Liner

> **Parameter sharing means the same convolutional filter weights are reused across spatial positions. It reduces parameter count, allows the same pattern detector to work anywhere in the input, and contributes to convolution's translation-equivariant structure.**

---

# 🧠 Mental Model

```text
Learn One Filter
↓
Reuse Everywhere Spatially
↓
Different Input Patches
↓
Different Responses
↓
Feature Map
```

During training:

```text
Many Spatial Uses
↓
Gradient Contributions Accumulate
↓
One Shared Filter Update
```

---

# ⭐ Remember These 7 Things

1. 🔁 Same filter is reused across H/W
2. 🎛️ Different filters have different weights
3. 🌈 Full multi-channel filter is shared
4. 🧮 H/W do not directly increase parameter count
5. ⚡ Larger H/W increases computation
6. 📍 Parameter sharing supports translation equivariance
7. 🔄 Gradients from many positions update the same weights

---

# ⭐ Golden Rule

> **Learn the detector once, reuse it everywhere: spatial positions share convolutional parameters, while activations remain location-dependent.**
