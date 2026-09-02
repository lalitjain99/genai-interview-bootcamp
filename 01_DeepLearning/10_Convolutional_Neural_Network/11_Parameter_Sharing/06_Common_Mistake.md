# 🚫 Common Mistakes — Parameter Sharing in CNNs

> **Core Idea:** Most mistakes come from confusing **where parameters are shared, what remains unique, how parameter count is calculated, and how sharing relates to translation behavior**.

---

## ❌ Mistake 1: Thinking Every Spatial Position Has Its Own Filter

### Wrong

> The filter at the top-left has different weights from the filter at the center.

### Correct

The same convolutional filter is reused across spatial positions.

So:

```text
Location 1 → W
Location 2 → W
Location 3 → W
...
```

Parameter sharing means:

> **one learned filter, many spatial applications**

---

## ❌ Mistake 2: Thinking Parameter Sharing Means All Filters Are Identical

Suppose a layer has:

`64 filters`

### Wrong

> All 64 filters must share the same weights.

### Correct

Each filter learns its own parameters.

So:

`Filter 1 ≠ Filter 2 ≠ ... ≠ Filter 64`

What is shared is:

> **each individual filter across spatial locations**

---

## ❌ Mistake 3: Thinking Sharing Happens Across Input Channels

Suppose one filter has shape:

`3 × 3 × 64`

### Wrong

> Every input channel must use the same `3×3` weights.

### Correct

Different channel slices can have different learned weights.

The entire:

`3 × 3 × 64`

filter is what gets reused spatially.

So:

> **sharing is mainly across H and W, not across channels**

---

## ❌ Mistake 4: Multiplying Parameter Count by H_out × W_out

### Wrong

`Parameters = K_h × K_w × C_in × C_out × H_out × W_out`

### Correct

Spatial positions reuse the same filter parameters.

So:

`Weights = K_h × K_w × C_in × C_out`

and, if bias is used:

`Total = K_h × K_w × C_in × C_out + C_out`

`H_out × W_out`

counts:

> **filter applications**

not independent parameters.

---

## ❌ Mistake 5: Thinking Larger Images Always Mean More Conv Parameters

### Wrong

> A `224×224` image must require more convolution weights than a `32×32` image.

### Correct

If:

* kernel size
* `C_in`
* `C_out`

stay unchanged, the number of convolution parameters stays unchanged.

Larger H/W mainly increase:

* computation
* activations
* memory usage

not:

> **the number of learned filter weights**

---

## ❌ Mistake 6: Confusing Parameters with Activations

Example:

Output:

`32 × 32 × 64`

### Wrong

> The convolution has `65,536` parameters.

### Correct

`32 × 32 × 64`

describes:

> **output activations**

Parameter count depends on:

`K_h × K_w × C_in × C_out`

These are different quantities.

---

## ❌ Mistake 7: Thinking Shared Parameters Produce Identical Outputs

### Wrong

> Same weights mean every spatial position gives the same activation.

### Correct

The same filter sees:

> **different local input patches**

So:

```text
Same Filter + Patch A → 8.2
Same Filter + Patch B → 0.1
Same Filter + Patch C → -4.5
```

Therefore:

> **shared weights do not imply shared activations**

---

## ❌ Mistake 8: Thinking Parameter Sharing Removes Spatial Location

### Wrong

> If the same detector is used everywhere, the network cannot know where a feature occurred.

### Correct

Feature-map coordinates still preserve approximate spatial location.

So:

Feature at top-left
→ response near top-left

Feature at bottom-right
→ response near bottom-right

Parameter sharing reuses:

> **the detector**

not:

> the activation position.

---

## ❌ Mistake 9: Calling Parameter Sharing Translation Invariance

### Wrong

> Parameter sharing makes convolution translation-invariant.

### Correct

Convolution is more naturally associated with:

> **translation equivariance**

Meaning:

Input feature shifts
↓
Feature-map response tends to shift correspondingly

Invariance would mean:

> the output stays unchanged.

These are not the same.

---

## ❌ Mistake 10: Saying Every Practical CNN Is Perfectly Translation-Equivariant

### Wrong

> CNNs remain perfectly equivariant under any image translation.

### Correct

Practical behavior can be affected by:

* stride
* pooling
* padding
* boundaries
* discrete sampling
* other architecture components

So a more precise statement is:

> **Convolution provides a translation-equivariant structural bias, but an entire CNN is not guaranteed to be perfectly equivariant.**

---

## ❌ Mistake 11: Thinking Parameter Sharing Is Only a Memory Optimization

### Wrong

> Sharing is just an implementation trick to save RAM.

### Correct

Parameter sharing is also an:

> **architectural inductive bias**

It assumes:

> the same local pattern detector can be useful at different locations.

This affects:

* generalization
* parameter efficiency
* data efficiency
* translation behavior

So it is much more than a storage optimization.

---

## ❌ Mistake 12: Thinking Parameter Sharing Automatically Means Better Performance

### Wrong

> Sharing must always outperform location-specific weights.

### Correct

Sharing is a useful assumption for many image problems, but not always ideal.

If:

* regions have different statistics
* position has unique meaning
* sensor behavior varies spatially

then strict sharing may be restrictive.

So:

> **parameter sharing is a useful bias, not a universal law**

---

## ❌ Mistake 13: Thinking Different CNN Layers Share the Same Filters

### Wrong

> Conv1, Conv2, and Conv3 all reuse the same weights because CNNs use parameter sharing.

### Correct

Different layers usually learn:

> **different parameter sets**

Parameter sharing normally refers to:

> reuse of a filter across spatial positions within a layer.

---

## ❌ Mistake 14: Thinking Shared Weights Are Updated Separately at Every Location

Suppose one weight:

`w`

is used at 100 spatial positions.

### Wrong

> There are 100 separate copies of `w`, each updated independently.

### Correct

There is:

> **one stored parameter**

During backpropagation, gradient contributions from all uses contribute to:

`dL/dw`

Then the optimizer updates:

> **that one shared weight**

---

## ❌ Mistake 15: Thinking Only One Spatial Position Contributes to a Shared Weight's Gradient

### Wrong

> The filter update is based only on the last location where it was applied.

### Correct

All relevant spatial uses can contribute.

Conceptually:

```text
Gradient from Position 1
+
Gradient from Position 2
+
Gradient from Position 3
+
...
↓
Gradient for Shared Weight
```

So one filter learns from patterns occurring throughout the feature map.

---

## ❌ Mistake 16: Forgetting That Local Connectivity and Sharing Are Different Concepts

### Wrong

> Local connectivity and parameter sharing mean the same thing.

### Correct

They are related but distinct.

### Local Connectivity

Each output looks at:

> a small local input region

### Parameter Sharing

The same local weights are:

> reused across different regions

Together they give CNNs their parameter efficiency.

---

## ❌ Mistake 17: Thinking More Spatial Positions Mean More Filters

### Wrong

> If the output is `64×64`, the network needs 4,096 filters.

### Correct

Spatial positions and filters are different dimensions.

For example:

`64 × 64 × 128`

means:

* `64 × 64` spatial positions
* `128` output feature maps / filters

The same 128 filters are reused across the entire spatial grid.

---

## ❌ Mistake 18: Thinking Removing Sharing Is Always a Free Capacity Upgrade

### Wrong

> Give every location separate weights and the model can only improve.

### Correct

Removing sharing can cause:

* huge parameter growth
* more optimizer state
* larger memory requirements
* greater data requirements
* more overfitting risk
* weaker spatial generalization

More flexibility is not automatically:

> **better inductive bias**

---

# ⚡ Quick Trap Table

| Wrong Idea                            | Correct Idea                                   |
| ------------------------------------- | ---------------------------------------------- |
| Each location has its own filter      | Same filter is reused spatially                |
| All filters share identical weights   | Different filters learn different weights      |
| Sharing happens across channels       | Full multi-channel filter is shared across H/W |
| Multiply params by `H_out×W_out`      | Spatial positions reuse parameters             |
| Larger image = more conv weights      | Usually more compute, same weights             |
| Output activations = parameters       | They are different quantities                  |
| Same weights = same outputs           | Different patches give different outputs       |
| Sharing removes location              | Feature maps still preserve response location  |
| Sharing gives invariance              | More directly supports equivariance            |
| CNNs are perfectly equivariant        | Practical components can break it              |
| Sharing is only memory optimization   | It is an inductive bias                        |
| Sharing always improves every task    | It can be restrictive                          |
| Filters shared across layers          | Layers usually have separate weights           |
| Each spatial use gets separate update | Gradients accumulate into shared weights       |
| Local connectivity = sharing          | They are distinct ideas                        |
| More positions = more filters         | Positions reuse the same filters               |
| More parameters always better         | Architecture trade-off matters                 |

---

# 🧠 Final Mental Model

```text
Learn Filter W
↓
Apply W at Position 1
↓
Apply Same W at Position 2
↓
Apply Same W at Position 3
↓
...
↓
Different Responses
↓
Feature Map
```

During training:

```text
Many Uses of W
↓
Many Gradient Contributions
↓
One Gradient for W
↓
One Shared Weight Update
```

---

# ⭐ Golden Rule

> **Parameter sharing means reusing one learned convolutional filter across spatial locations—not sharing all filters, not sharing all channels, and not multiplying parameter count by the number of output positions.**
