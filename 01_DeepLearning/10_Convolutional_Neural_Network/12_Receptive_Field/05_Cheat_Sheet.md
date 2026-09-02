# ⚡ Cheat Sheet — Receptive Field in CNNs

> **Core Idea:** Receptive field tells us **how much of the original input can influence one activation**. It grows with depth, and stride/pooling make later receptive fields grow faster.

---

# 👁️ Receptive Field

Receptive Field:

> **Region of the original input that can influence one activation**

Example:

`3 × 3 Conv`

in the first layer:

> **RF = 3 × 3**

---

# 🧠 Basic Growth

For stride `1` and dilation `1`:

```text
Input       RF = 1
Conv 3×3    RF = 3
Conv 3×3    RF = 5
Conv 3×3    RF = 7
```

So:

* 1 Conv `3×3` → RF `3`
* 2 Conv `3×3` → RF `5`
* 3 Conv `3×3` → RF `7`

---

# ⭐ Stride-1 Shortcut

If:

`S = 1`

and:

`Dilation = 1`

then each layer adds:

`K - 1`

to receptive field.

For `3×3`:

`+2`

For `5×5`:

`+4`

---

# 🦘 Jump

To calculate RF when stride or pooling is involved, also track:

> **Jump `j`**

Jump means:

> distance between neighboring feature-map activations in original-input coordinates.

Initially:

`r = 1`

`j = 1`

---

# 🧮 Core Formulas

For kernel:

`K`

and stride:

`S`

use:

`r_new = r_old + (K - 1) × j_old`

`j_new = j_old × S`

These formulas assume:

> **dilation = 1**

---

# 🧮 Example — Two 3×3 Conv

Start:

`r=1, j=1`

### Conv 1

`K=3, S=1`

`r = 1 + 2×1 = 3`

`j = 1`

### Conv 2

`r = 3 + 2×1 = 5`

Final:

> **RF = 5 × 5**

---

# ➡️ Stride Example

Architecture:

```text
Conv 3×3, S=2
Conv 3×3, S=1
```

Start:

`r=1, j=1`

After first:

`r=3`

`j=2`

After second:

`r = 3 + 2×2`

`=7`

Final:

> **RF = 7 × 7**

---

# 🧠 Why Stride Matters

Stride increases:

> **jump**

Example:

After stride 2:

`j = 2`

A later `3×3` Conv adds:

`(3-1) × 2`

`=4`

instead of:

`2`

So:

> **downsampling accelerates later RF growth**

---

# 🏊 Pooling

Pooling also increases receptive field.

Example:

```text
Conv 3×3, S=1
Pool 2×2, S=2
```

Start:

`r=1, j=1`

After Conv:

`r=3, j=1`

After Pool:

`r = 3 + 1×1`

`=4`

`j=2`

So:

> **RF = 4**

---

# 🏗️ Conv + Pool + Conv

```text
Conv 3×3, S=1
Pool 2×2, S=2
Conv 3×3, S=1
```

After Conv:

`r=3, j=1`

After Pool:

`r=4, j=2`

After final Conv:

`r = 4 + 2×2`

`=8`

Final:

> **RF = 8 × 8**

---

# 🎛️ Effect of Kernel Size

Larger kernels grow RF faster.

| Kernel | First-Layer RF |
| ------ | -------------: |
| `3×3`  |          `3×3` |
| `5×5`  |          `5×5` |
| `7×7`  |          `7×7` |

But larger kernels usually increase:

* parameters
* computation

---

# 🏗️ Small Kernels Can Build Large RF

Two:

`3×3`

stride-1 convolutions:

> **RF = 5×5**

Three:

`3×3`

convolutions:

> **RF = 7×7**

So:

> **Small local operations compose into larger context**

---

# ⚠️ Same RF ≠ Same Network

Two `3×3` layers and one `5×5` layer can have similar RF coverage.

But they are not equivalent.

Two layers can provide:

* extra nonlinearity
* deeper representation
* different intermediate channels

So:

> **Same receptive field does not imply same computation or behavior**

---

# 📍 RF vs Feature Map Size

Do NOT confuse them.

### Feature Map Size

Example:

`56 × 56 × 256`

means:

> number of activation positions

### Receptive Field

Example:

`27 × 27`

means:

> original-input context available to one activation

So:

> **Feature-map size = quantity of outputs**

> **RF = context per output**

---

# 🧱 Padding

Padding affects:

* output dimensions
* boundary behavior
* RF alignment

But it does NOT create:

> **new real image information**

At boundaries:

part of the theoretical RF may contain:

> padding values

instead of real input pixels.

---

# 🎯 Theoretical vs Effective RF

## Theoretical RF

All input positions that:

> **could influence an activation**

## Effective RF

Input region that:

> **strongly influences the activation in practice**

So:

> **Theoretical RF does not mean all positions contribute equally**

---

# 🕳️ Dilation

Effective kernel span:

`K_eff = D × (K - 1) + 1`

Example:

`K = 3`

`D = 2`

Then:

`K_eff = 2 × (3-1) + 1`

`=5`

So:

> **Dilated 3×3 spans 5×5**

while still using:

`9 spatial weights`

per input-output channel pair.

---

# 📈 What Increases RF?

Receptive field grows with:

* 🏗️ more layers
* 🎛️ larger kernels
* ➡️ stride
* 🏊 pooling
* 🕳️ dilation

---

# ⚠️ RF Does NOT Equal Kernel Size

A deep layer may use:

`3×3 kernel`

while having:

`31×31 RF`

Why?

Because:

> **receptive field accumulates across previous layers**

---

# 🔄 RF vs Parameter Sharing

### Parameter Sharing

Question:

> Are the same filter weights reused spatially?

### Receptive Field

Question:

> How much original input affects one activation?

Different concepts.

---

# 🏊 RF vs Pooling

Pooling typically causes:

```text
Feature Map Resolution ↓
Receptive Field ↑
Jump ↑
```

So:

> **smaller feature maps can correspond to larger input context per activation**

---

# 🧮 Full Calculation Example

Architecture:

```text
Conv 3×3, S=1
Conv 3×3, S=1
Pool 2×2, S=2
Conv 3×3, S=1
```

Start:

`r=1, j=1`

| Layer  |  K |  S | RF | Jump |
| ------ | -: | -: | -: | ---: |
| Input  |  — |  — |  1 |    1 |
| Conv 1 |  3 |  1 |  3 |    1 |
| Conv 2 |  3 |  1 |  5 |    1 |
| Pool   |  2 |  2 |  6 |    2 |
| Conv 3 |  3 |  1 | 10 |    2 |

Final:

> **RF = 10 × 10**

---

# ⚡ Quick Recall Table

| Concept          | Rule                                    |
| ---------------- | --------------------------------------- |
| Receptive Field  | Input region influencing one activation |
| Initial RF       | `1`                                     |
| One `3×3` Conv   | `3`                                     |
| Two `3×3`, S1    | `5`                                     |
| Three `3×3`, S1  | `7`                                     |
| RF Formula       | `r_new = r_old + (K-1)j_old`            |
| Jump Formula     | `j_new = j_old × S`                     |
| Larger Kernel    | Faster RF growth                        |
| Stride           | Makes future RF grow faster             |
| Pooling          | RF ↑, jump ↑                            |
| Feature Map Size | Number of outputs                       |
| RF Size          | Context per activation                  |
| Theoretical RF   | Possible influence                      |
| Effective RF     | Strong practical influence              |
| Dilation         | Expands spatial span                    |

---

# 🚫 Interview Traps

### ❌ Two 3×3 Conv = 6×6 RF

Wrong.

> **RF = 5×5**

---

### ❌ RF equals feature-map size

Wrong.

They represent different things.

---

### ❌ Padding adds real input context

Wrong.

> Padding adds artificial boundary values.

---

### ❌ A 31×31 RF means a 31×31 kernel

Wrong.

> Large RF may come from stacked small kernels.

---

### ❌ Every pixel inside theoretical RF contributes equally

Wrong.

> Effective influence can vary significantly.

---

### ❌ Stride only changes output size

Wrong.

> It also changes jump and therefore later RF growth.

---

### ❌ Pooling only shrinks feature maps

Wrong.

> It also increases RF and jump.

---

# 🧠 Mental Model

```text
Input
↓
Small Local Kernel
↓
Small RF
↓
Stack Layers
↓
RF Expands
↓
Stride / Pooling Increase Jump
↓
Later RF Grows Faster
↓
Broader Context
```

---

# 🎤 Interview One-Liner

> **The receptive field is the region of the original input that can influence one CNN activation. I calculate it using `r_new = r_old + (K-1)×j_old` and track jump with `j_new = j_old×S`; depth expands receptive field, while stride and pooling make later growth faster.**

---

# ⭐ Remember These 7 Things

1. 👁️ RF = original-input context per activation
2. 🏗️ Depth increases RF
3. 🎛️ Larger kernels increase RF faster
4. 🦘 Track jump when stride/pooling exists
5. 🏊 Pooling increases RF and reduces resolution
6. 📍 Feature-map size ≠ RF size
7. 🎯 Theoretical RF ≠ equal practical influence

---

# ⭐ Golden Rule

> **Track `r` and `j`: layers accumulate context, while downsampling increases the input spacing between activations and makes receptive fields grow faster.**
