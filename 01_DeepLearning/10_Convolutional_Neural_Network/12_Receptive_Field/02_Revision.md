# 📝 Revision — Receptive Field in CNNs

> **Core Idea:** The receptive field of a CNN activation is the region of the original input that can influence that activation. As layers are stacked, receptive fields usually grow, allowing deeper activations to use broader spatial context.

---

# 👁️ 1. What Is a Receptive Field?

The receptive field answers:

> **How much of the original input can influence one activation?**

Example:

A first-layer:

`3 × 3`

convolution sees:

> **a 3×3 input region**

So its receptive field is:

`3 × 3`

---

# 🧩 2. Receptive Field Starts Small

At the input:

`RF = 1`

After one:

`3 × 3, stride 1`

convolution:

`RF = 3`

After another:

`3 × 3`

convolution:

`RF = 5`

After a third:

`RF = 7`

So:

```text
Input       RF = 1
Conv 3×3    RF = 3
Conv 3×3    RF = 5
Conv 3×3    RF = 7
```

---

# 🧠 3. Why Does RF Grow?

A deeper convolution does not operate directly on raw pixels.

It operates on:

> **previous-layer activations**

Those activations already depend on regions of the original input.

So dependencies accumulate through depth.

That is why:

> **small kernels can eventually cover large input regions**

---

# ⭐ 4. Stride-1 Shortcut

If:

* stride = `1`
* dilation = `1`

then every layer adds:

`K - 1`

to receptive-field size.

For a `3×3` convolution:

`K - 1 = 2`

So:

`1 → 3 → 5 → 7 → 9`

---

# 🎛️ 5. Effect of Kernel Size

Larger kernels directly grow receptive field faster.

First-layer examples:

| Kernel |    RF |
| ------ | ----: |
| `3×3`  | `3×3` |
| `5×5`  | `5×5` |
| `7×7`  | `7×7` |

But larger kernels generally mean:

* more parameters
* more computation

---

# 🏗️ 6. Stacking Small Kernels

Two:

`3×3`

stride-1 convolutions give:

> **5×5 receptive-field coverage**

Three:

`3×3`

convolutions give:

> **7×7 receptive-field coverage**

This is one reason stacking small kernels is common.

It can provide:

* larger receptive field
* more nonlinear depth
* parameter efficiency in many architectures

---

# ⚠️ 7. Same RF Does Not Mean Same Network

Two `3×3` layers may have similar RF coverage to one `5×5` layer.

But they are not equivalent.

Two layers can include:

* two learned transformations
* nonlinear activation between them
* different intermediate channels

So:

> **Same receptive-field size ≠ same representation**

---

# 🦘 8. The Jump Concept

To calculate RF through stride or pooling, track another quantity:

> **Jump `j`**

Jump means:

> how far apart neighboring feature-map activations are in original-input coordinates.

Initially:

`r = 1`

`j = 1`

---

# 🧮 9. Core Formulas

For a layer with:

* kernel = `K`
* stride = `S`

use:

`r_new = r_old + (K - 1) × j_old`

and:

`j_new = j_old × S`

These are the main receptive-field formulas.

---

# 🧮 10. Example — Two Stride-1 Convolutions

Start:

`r = 1`

`j = 1`

### Conv 1

`K=3, S=1`

`r = 1 + 2×1 = 3`

`j = 1`

### Conv 2

`K=3, S=1`

`r = 3 + 2×1 = 5`

`j = 1`

Final:

> **RF = 5×5**

---

# ➡️ 11. Effect of Stride

Stride makes later receptive fields grow faster.

Example:

### Conv 1

`K=3, S=2`

Start:

`r=1, j=1`

Then:

`r = 3`

`j = 2`

### Conv 2

`K=3, S=1`

`r = 3 + 2×2`

`= 7`

So:

> **RF = 7×7**

Compare:

Two stride-1 `3×3` convolutions:

> `5×5`

---

# 🏊 12. Pooling Also Increases RF

Pooling is also a local operation.

Example:

Start:

`r=1, j=1`

### Conv `3×3, S=1`

`r=3`

`j=1`

### Pool `2×2, S=2`

`r = 3 + 1×1`

`= 4`

`j = 2`

So pooling:

> **reduces spatial resolution and increases receptive field**

---

# 🧮 13. Conv + Pool + Conv Example

Continue from:

`r=4`

`j=2`

Apply:

`3×3 Conv, S=1`

Then:

`r = 4 + 2×2`

`= 8`

So:

```text
Input       RF = 1
Conv 3×3    RF = 3
Pool 2×2    RF = 4
Conv 3×3    RF = 8
```

---

# 📍 14. Receptive Field ≠ Feature Map Size

These are different concepts.

### Feature Map Size

Example:

`56 × 56 × 256`

means:

> how many activation positions exist.

### Receptive Field

Example:

`27 × 27`

means:

> how much of the original input influences each activation.

So:

> **Feature-map size tells quantity of outputs; RF tells context per output.**

---

# 🧱 15. Padding and Receptive Field

Padding changes:

* output geometry
* boundary behavior

But it does not add real image information.

Near an edge:

part of an activation's theoretical receptive field may include:

> padded values

rather than real input pixels.

---

# 🌍 16. Why Deeper Layers Need Larger RF

Early layers often use small context to learn:

* edges
* local textures
* simple patterns

Deeper layers can combine these into:

* larger structures
* object parts
* broad context

So depth can produce:

```text
Local Pattern
↓
Pattern Combination
↓
Larger Structure
↓
Broader Context
```

---

# 🎯 17. Theoretical vs Effective Receptive Field

### Theoretical RF

All input positions that:

> **could influence an activation**

through the computational graph.

### Effective RF

The region that:

> **strongly influences the activation in practice**

These are not always the same.

---

# ⚠️ 18. Large RF Does Not Mean Equal Influence

If theoretical RF is:

`31 × 31`

do not assume:

> all 961 positions contribute equally.

Influence depends on:

* learned weights
* nonlinearities
* path structure
* data

So RF represents:

> **possible dependency**, not equal importance.

---

# 📈 19. What Makes RF Grow Faster?

Receptive field grows with:

* 🏗️ more layers
* 🎛️ larger kernels
* ➡️ larger stride
* 🏊 pooling/downsampling
* 🕳️ dilation

---

# 🕳️ 20. Dilation Quick Recall

For dilation `D`:

`Effective Kernel Span = D × (K - 1) + 1`

Example:

`K=3`

`D=2`

Then:

`2 × (3-1) + 1`

`= 5`

So a dilated `3×3` convolution spans:

> **5×5 spatial area**

while still using 9 spatial kernel weights per input-output channel pair.

---

# ⚠️ 21. RF Does Not Equal Kernel Size in Deep Layers

A deep activation may have:

`RF = 31×31`

while its current convolution kernel is still:

`3×3`

The larger RF comes from:

> **accumulated dependencies across previous layers**

---

# 🔄 22. Receptive Field vs Parameter Sharing

Do not confuse them.

### Parameter Sharing

> Are the same weights reused across spatial positions?

### Receptive Field

> How much of the original input affects one activation?

They describe different properties.

---

# 🏊 23. Receptive Field vs Pooling

Pooling causes:

```text
Feature Map Resolution ↓
Receptive Field ↑
```

These two effects are related, but not identical.

---

# 🧮 24. Full Example

Architecture:

```text
Conv 3×3, S=1
Conv 3×3, S=1
Pool 2×2, S=2
Conv 3×3, S=1
```

Start:

`r=1, j=1`

### Conv 1

`r=3`

`j=1`

### Conv 2

`r=5`

`j=1`

### Pool

`r=6`

`j=2`

### Conv 3

`r=6 + 2×2`

`=10`

Final:

> **RF = 10×10**

---

# ⚡ Quick Recall Table

| Concept          | Key Rule                                |
| ---------------- | --------------------------------------- |
| Receptive Field  | Input region influencing one activation |
| Initial RF       | `1`                                     |
| One `3×3` Conv   | `3`                                     |
| Two `3×3` S1     | `5`                                     |
| Three `3×3` S1   | `7`                                     |
| RF Formula       | `r_new = r_old + (K-1)j_old`            |
| Jump Formula     | `j_new = j_old × S`                     |
| Larger Kernel    | Faster RF growth                        |
| Larger Stride    | Faster later RF growth                  |
| Pooling          | Increases RF                            |
| Feature Map Size | Number of output positions              |
| RF Size          | Context per activation                  |
| Theoretical RF   | Possible influence                      |
| Effective RF     | Strong practical influence              |
| Dilation         | Expands kernel span                     |

---

# 🧠 Mental Model

```text
Early Layer
↓
Small Local Context
↓
Deeper Layer
↓
Combines Previous Local Features
↓
Larger Context
↓
Broader Representation
```

For calculations:

```text
Start:
r = 1
j = 1

For Each Layer:
r_new = r_old + (K-1) × j_old
j_new = j_old × S
```

---

# 🎤 30-Second Interview Answer

> **The receptive field of a CNN activation is the region of the original input that can influence it. Receptive fields grow with depth because each layer operates on activations that already depend on previous input regions. For general calculations, I track receptive field and jump using `r_new = r_old + (K-1)×j_old` and `j_new = j_old×S`. Stride and pooling make later receptive fields grow faster. Receptive field is different from feature-map size, and the theoretical receptive field only tells us what can influence an activation, not that every input position contributes equally.**

---

# ⭐ Remember These 7 Things

1. 👁️ RF = input context affecting one activation
2. 🏗️ RF grows with depth
3. 🎛️ Larger kernels grow RF faster
4. 🦘 Track jump when stride/pooling is involved
5. 🏊 Pooling increases RF while reducing resolution
6. 📍 RF size is different from feature-map size
7. 🎯 Theoretical RF is not the same as effective influence

---

# ⭐ Golden Rule

> **Small local kernels can create large global context because receptive fields accumulate through depth, stride, and downsampling.**
