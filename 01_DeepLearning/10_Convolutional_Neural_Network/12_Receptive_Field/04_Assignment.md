# 💡 Assignment — Receptive Field in CNNs

> **Focus:** Practice calculating receptive field, tracking jump, understanding the effect of stride and pooling, and distinguishing receptive field from feature-map size.

---

# 🧩 Scenario 1 — One Convolution

Input
↓
`Conv 3×3, stride 1`

### Question

What is the receptive field of one output activation?

### ✅ Answer

Start with:

`r = 1`

`j = 1`

Then:

`r_new = r_old + (K - 1) × j_old`

`= 1 + (3 - 1) × 1`

`= 3`

Therefore:

> **Receptive Field = 3 × 3**

---

# 🧩 Scenario 2 — Two 3×3 Convolutions

Architecture:

```text
Input
↓
Conv 3×3, S=1
↓
Conv 3×3, S=1
```

### Question

What is the final receptive field?

### ✅ Answer

Start:

`r = 1`

`j = 1`

### Conv 1

`r = 1 + 2×1`

`= 3`

### Conv 2

`r = 3 + 2×1`

`= 5`

Therefore:

> **Final RF = 5 × 5**

---

# 🧩 Scenario 3 — Three 3×3 Convolutions

Architecture:

```text
Conv 3×3, S=1
Conv 3×3, S=1
Conv 3×3, S=1
```

### Question

What is the final receptive field?

### ✅ Answer

For stride 1:

each `3×3` layer adds:

`2`

So:

`1 → 3 → 5 → 7`

Therefore:

> **Final RF = 7 × 7**

---

# 🧠 Scenario 4 — Spot the Mistake

A developer says:

> "Two 3×3 convolutions have a 6×6 receptive field because 3 + 3 = 6."

### Is this correct?

### ✅ Answer

No.

Neighboring convolution windows overlap.

For stride 1:

each additional `3×3` layer adds:

`K - 1`

`= 2`

So:

`3 + 2 = 5`

Therefore:

> **Two 3×3 convolutions produce a 5×5 receptive field**

---

# 🎛️ Scenario 5 — Larger Kernel

Architecture:

`Conv 5×5, stride 1`

### Question

What is the receptive field of the first layer?

### ✅ Answer

A first-layer `5×5` convolution directly sees:

> **5 × 5**

of the input.

Therefore:

> **RF = 5 × 5**

---

# 🦘 Scenario 6 — Understanding Jump

Start with:

`j = 1`

Apply:

`Conv 3×3, stride 2`

### Question

What is the new jump?

### ✅ Answer

Use:

`j_new = j_old × S`

So:

`j_new = 1 × 2`

`= 2`

Therefore:

> **Neighboring activations are now 2 input pixels apart**

in original-input coordinates.

---

# ➡️ Scenario 7 — Stride Changes Future RF Growth

Architecture:

```text
Conv 3×3, S=2
Conv 3×3, S=1
```

### Question

Calculate the final receptive field.

### ✅ Answer

Start:

`r=1`

`j=1`

### Conv 1

`r = 1 + 2×1`

`=3`

`j = 1×2`

`=2`

### Conv 2

`r = 3 + 2×2`

`=7`

Therefore:

> **Final RF = 7 × 7**

---

# 🏊 Scenario 8 — Convolution + Pooling

Architecture:

```text
Conv 3×3, S=1
MaxPool 2×2, S=2
```

### Question

What are the final receptive field and jump?

### ✅ Answer

Start:

`r=1`

`j=1`

### Conv

`r = 1 + 2×1`

`=3`

`j=1`

### Pool

`r = 3 + (2-1)×1`

`=4`

`j = 1×2`

`=2`

Therefore:

> **RF = 4 × 4**

and:

> **Jump = 2**

---

# 🏗️ Scenario 9 — Conv + Pool + Conv

Continue from:

```text
Conv 3×3, S=1
Pool 2×2, S=2
Conv 3×3, S=1
```

### Question

What is the final receptive field?

### ✅ Answer

After first Conv:

`r=3`

`j=1`

After Pool:

`r=4`

`j=2`

Final Conv:

`r = 4 + 2×2`

`=8`

Therefore:

> **Final RF = 8 × 8**

---

# 📐 Scenario 10 — Receptive Field vs Feature Map Size

Suppose a layer outputs:

`56 × 56 × 128`

and each activation has:

`RF = 27 × 27`

### Question

What do these two numbers mean?

### ✅ Answer

`56 × 56`

means:

> **there are 56×56 spatial activation positions**

`27 × 27`

means:

> **each one of those activations can depend on a 27×27 region of the original input**

So:

> **Feature-map size and receptive-field size are different quantities**

---

# 🧱 Scenario 11 — Padding

A first-layer:

`3×3 convolution`

uses padding 1.

A developer says:

> "Padding increases the receptive field from 3×3 to 5×5."

### Is this correct?

### ✅ Answer

No.

The convolution still has an architectural receptive-field span of:

> **3×3**

Padding affects:

* output geometry
* boundary behavior

Near image borders, part of the receptive field may contain:

> padded values

instead of real image pixels.

Padding does not create additional real image information.

---

# 🎯 Scenario 12 — Theoretical vs Effective RF

A deep activation has theoretical receptive field:

`31 × 31`

### Question

Does this mean all 961 input positions influence the activation equally?

### ✅ Answer

No.

The theoretical receptive field describes:

> **which positions could influence the activation**

But actual influence depends on:

* learned weights
* nonlinearities
* paths through the network
* data

So the:

> **effective receptive field may be more concentrated**

than the full theoretical RF.

---

# 🕳️ Scenario 13 — Dilation

A convolution uses:

* Kernel = `3`
* Dilation = `2`

### Question

What is its effective kernel span?

### ✅ Answer

Use:

`K_eff = D × (K - 1) + 1`

So:

`K_eff = 2 × (3 - 1) + 1`

`= 5`

Therefore:

> **A dilated 3×3 kernel spans a 5×5 spatial region**

while still using 9 spatial kernel weights per channel pair.

---

# 🧠 Scenario 14 — Same RF, Different Architecture

Compare:

### Option A

One:

`5×5 convolution`

### Option B

Two:

`3×3 stride-1 convolutions`

### Question

Do they have similar receptive-field coverage?

### ✅ Answer

Yes.

Two `3×3` convolutions produce:

`RF = 5×5`

But the architectures are not equivalent.

Option B may include:

* two learned transformations
* an activation between layers
* different intermediate channels

So:

> **Same RF size does not mean same computation or representation**

---

# 🧮 Scenario 15 — Full Calculation

Architecture:

```text
Conv 3×3, S=1
Conv 3×3, S=1
Pool 2×2, S=2
Conv 3×3, S=1
```

### Question

Calculate the final RF and jump.

### ✅ Answer

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

`r = 6 + 2×2`

`=10`

`j=2`

Therefore:

> **Final RF = 10 × 10**

> **Final Jump = 2**

---

# ✅ True / False

| #  | Statement                                                                            | Answer  |
| -- | ------------------------------------------------------------------------------------ | ------- |
| 1  | Receptive field tells how much original input can influence one activation.          | ✅ True  |
| 2  | Two 3×3 stride-1 convolutions produce a 6×6 RF.                                      | ❌ False |
| 3  | Three 3×3 stride-1 convolutions produce a 7×7 RF.                                    | ✅ True  |
| 4  | Feature-map size and receptive-field size mean the same thing.                       | ❌ False |
| 5  | Larger kernels can grow RF faster.                                                   | ✅ True  |
| 6  | Stride can make later receptive fields grow faster.                                  | ✅ True  |
| 7  | Pooling can increase receptive field.                                                | ✅ True  |
| 8  | Padding creates additional real image context.                                       | ❌ False |
| 9  | A deep layer can use a 3×3 kernel while having a much larger RF.                     | ✅ True  |
| 10 | Every position inside the theoretical RF contributes equally.                        | ❌ False |
| 11 | Dilation can increase spatial span without proportionally increasing kernel weights. | ✅ True  |
| 12 | Larger receptive field is always better.                                             | ❌ False |

---

# ⭐ Staff Engineer Challenge

## Scenario

Consider:

```text
Input
↓
Conv 3×3, S=1
↓
Conv 3×3, S=2
↓
Conv 5×5, S=1
↓
MaxPool 2×2, S=2
↓
Conv 3×3, S=1
```

An engineer calculates the receptive field by simply adding:

`2 + 2 + 4 + 1 + 2`

to the initial RF.

They conclude:

`RF = 12`

---

## Question 1

Why is that approach wrong?

### ✅ Answer

Because after a layer with stride greater than 1:

> **the jump changes**

Later kernels operate on activations whose centers are farther apart in original-input coordinates.

So we must track both:

`r`

and:

`j`

using:

`r_new = r_old + (K-1)×j_old`

`j_new = j_old×S`

---

## Question 2

What happens after Conv 1?

Start:

`r=1`

`j=1`

Layer:

`K=3, S=1`

### ✅ Answer

`r = 1 + 2×1`

`=3`

`j=1`

So:

> `r=3, j=1`

---

## Question 3

What happens after Conv 2?

Layer:

`K=3, S=2`

### ✅ Answer

`r = 3 + 2×1`

`=5`

`j = 1×2`

`=2`

So:

> `r=5, j=2`

---

## Question 4

What happens after the 5×5 convolution?

Layer:

`K=5, S=1`

### ✅ Answer

`r = 5 + (5-1)×2`

`=5 + 8`

`=13`

Jump stays:

`j=2`

So:

> `r=13, j=2`

---

## Question 5

What happens after MaxPool?

Layer:

`K=2, S=2`

### ✅ Answer

`r = 13 + (2-1)×2`

`=15`

`j = 2×2`

`=4`

So:

> `r=15, j=4`

---

## Question 6

What happens after the final 3×3 convolution?

Layer:

`K=3, S=1`

### ✅ Answer

`r = 15 + 2×4`

`=23`

Jump stays:

`j=4`

Therefore:

> **Final RF = 23 × 23**

---

## Question 7

Why did the final 3×3 convolution add 8 to RF?

### ✅ Answer

Because:

`j = 4`

at that stage.

So:

`(K-1) × j`

`= 2 × 4`

`=8`

That convolution covers three neighboring positions whose centers are:

> **4 original-input pixels apart**

---

## Question 8

Would making the receptive field even larger automatically improve the model?

### ✅ Answer

No.

A larger RF provides broader context, but architecture design also needs:

* local detail
* spatial resolution
* reasonable computation
* preservation of small-object information

So:

> **Receptive-field size is a design trade-off, not a metric to maximize blindly**

---

# 🎯 Final Exercise

Fill in the formulas:

### Receptive Field

`r_new = ________________________`

### Jump

`j_new = ________________________`

### ✅ Answer

`r_new = r_old + (K - 1) × j_old`

`j_new = j_old × S`

---

# 🧠 Final Mental Model

```text
Small Kernel
↓
Local Input Context
↓
Stack More Layers
↓
Dependencies Accumulate
↓
Stride / Pooling Increase Jump
↓
Later Layers Cover Input Faster
↓
Large Receptive Field
```

But always remember:

```text
Large Context
≠
Equal Influence
≠
Always Better
```

---

# ⭐ Golden Rule

> **Track both receptive field and jump: depth expands context, while stride and pooling increase the spacing between activations and make later receptive fields grow much faster.**
