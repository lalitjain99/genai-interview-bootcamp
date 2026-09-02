# 💡 Assignment — Parameter Sharing in CNNs

> **Focus:** Practice why CNNs reuse filters across space, how sharing reduces parameters, how it affects computation and translation behavior, and how shared weights receive gradients.

---

# 🧩 Scenario 1 — Basic Sharing

A grayscale image uses one:

`3 × 3`

filter.

The output contains:

`100 spatial positions`

### Question

How many sets of filter weights are learned?

### ✅ Answer

Only:

> **one set**

A `3×3` grayscale filter has:

`9 weights`

Those same 9 weights are reused at all 100 positions.

So:

> **Parameter sharing means many applications, one parameter set.**

---

# 🧮 Scenario 2 — With vs Without Sharing

Input:

`32 × 32`

Kernel:

`3 × 3`

Padding:

`0`

Stride:

`1`

Output:

`30 × 30`

So:

`900 positions`

### Question

How many weights are required for one feature detector:

1. without spatial sharing?
2. with spatial sharing?

### ✅ Answer

### Without sharing

Each position gets its own 9 weights:

`900 × 9`

`= 8,100`

### With sharing

Only one filter:

`3 × 3`

`= 9`

Therefore:

> **Without sharing = 8,100 weights**

> **With sharing = 9 weights**

---

# 🌈 Scenario 3 — Multi-Channel Sharing

Input:

`64 × 64 × 32`

Kernel:

`3 × 3`

### Question

What exactly is shared across spatial positions?

### ✅ Answer

One complete filter has shape:

`3 × 3 × 32`

The entire:

> **3 × 3 × 32 parameter set**

is reused across spatial positions.

It is not just one `3×3` channel slice.

---

# 🎛️ Scenario 4 — Multiple Filters

A layer has:

* `C_in = 3`
* kernel = `3 × 3`
* `64 filters`

A developer says:

> "Because CNNs use parameter sharing, all 64 filters must have the same weights."

### Is this correct?

### ✅ Answer

No.

Each filter learns its own parameter set.

So:

```text
Filter 1 ≠ Filter 2 ≠ ... ≠ Filter 64
```

But:

> **each individual filter is reused across spatial positions**

Sharing occurs across space, not across different filters.

---

# 🧮 Scenario 5 — Parameter Count

Input channels:

`3`

Kernel:

`3 × 3`

Output channels:

`64`

### Question

How many parameters does the convolution contain if biases are enabled?

### ✅ Answer

Weights:

`3 × 3 × 3 × 64`

`= 1,728`

Biases:

`64`

Total:

`1,728 + 64`

`= 1,792`

Therefore:

> **Total parameters = 1,792**

---

# 📐 Scenario 6 — Does Image Size Change Parameters?

Consider the same convolution:

* kernel = `3 × 3`
* `C_in = 3`
* `C_out = 64`

Apply it first to:

`32 × 32 × 3`

and then to:

`224 × 224 × 3`

### Question

Does the number of trainable parameters change?

### ✅ Answer

No.

For both:

`3 × 3 × 3 × 64 + 64`

`= 1,792`

So:

> **Input H/W do not directly affect standard convolution parameter count.**

The larger image does require more computation because the filters are applied at more locations.

---

# ⚡ Scenario 7 — Parameters vs Computation

A developer says:

> "If a larger image does not add convolution parameters, then it does not make the layer more expensive."

### Is this correct?

### ✅ Answer

No.

A larger feature map means:

> **more spatial filter applications**

So it can increase:

* multiply-add operations
* activation memory
* runtime

even though the number of stored trainable weights remains unchanged.

So:

> **Parameter count and computation are different quantities.**

---

# 🚫 Scenario 8 — Incorrect Parameter Formula

A developer calculates convolution weights using:

`K_h × K_w × C_in × C_out × H_out × W_out`

### Question

Why is this wrong?

### ✅ Answer

Because:

`H_out × W_out`

counts:

> **how many spatial locations use the filter**

Those locations do not have separate filter parameters.

Correct weight count:

`K_h × K_w × C_in × C_out`

So:

> **filter applications are not independent parameter sets.**

---

# 🗺️ Scenario 9 — Feature Map Interpretation

A shared filter produces responses:

```text
Location A → 7.2
Location B → 0.4
Location C → -2.1
```

### Question

If the parameters are shared, why are the outputs different?

### ✅ Answer

Because the same filter is applied to:

> **different local input patches**

Shared parameters do not mean shared activations.

So:

`Same Filter + Different Input`

can produce:

> **Different Responses**

---

# 📍 Scenario 10 — Translation Equivariance

A vertical-edge-like pattern moves from the left side of an image to the right side.

### Question

What would we expect from a shared convolutional filter that detects this pattern?

### ✅ Answer

The strong feature-map response should tend to:

> **move to the corresponding new location**

Conceptually:

```text
Input Pattern Shifts
↓
Feature Response Shifts
```

This is the intuition behind:

> **translation equivariance**

---

# ⚠️ Scenario 11 — Equivariance vs Invariance

A developer says:

> "Since convolution shares parameters, moving an object should leave the feature map exactly unchanged."

### Is this correct?

### ✅ Answer

No.

That describes:

> **translation invariance**

Convolution is more naturally associated with:

> **translation equivariance**

Meaning:

Input shifts
↓
Output response tends to shift correspondingly

not:

Input shifts
↓
Output remains identical

---

# 🔄 Scenario 12 — Backpropagation Through Shared Weights

Suppose one filter weight:

`w`

is used at:

`100 spatial locations`

### Question

How is its gradient computed?

### ✅ Answer

All relevant uses of `w` can contribute to:

`dL/dw`

Conceptually:

```text
Gradient Contribution 1
+
Gradient Contribution 2
+
...
+
Gradient Contribution 100
↓
Total Gradient for w
```

The optimizer then updates:

> **one shared weight**

---

# 🧠 Scenario 13 — Sharing Across Layers

Suppose a CNN has:

* Conv1
* Conv2
* Conv3

A developer says:

> "Parameter sharing means Conv1, Conv2, and Conv3 should all use the same filters."

### Is this correct?

### ✅ Answer

No.

Parameter sharing usually means:

> **reuse across spatial positions within a convolution operation**

Different layers normally learn:

> **different filters**

because they operate on different levels of representation.

---

# 🏗️ Scenario 14 — Why Sharing Is an Inductive Bias

A CNN assumes:

> a useful pattern in one image region may also be useful elsewhere.

### Question

Why is this called an inductive bias?

### ✅ Answer

Because the assumption is built directly into:

> **the architecture**

The model is not free to learn completely unrelated local transformations at every position.

Instead it is biased toward:

> **location-independent local pattern reuse**

This often improves parameter and data efficiency for image-like data.

---

# ⚖️ Scenario 15 — When Sharing Can Be Restrictive

Suppose a specialized sensor always behaves differently in the top half and bottom half of an image.

### Question

Why might strict spatial parameter sharing be less ideal?

### ✅ Answer

Because parameter sharing assumes:

> **the same local transformation should be useful everywhere**

If different locations have fundamentally different statistics or meanings, position-specific transformations might be useful.

So parameter sharing is:

> **a powerful assumption, but still an assumption**

---

# ✅ True / False

| #  | Statement                                                                           | Answer  |
| -- | ----------------------------------------------------------------------------------- | ------- |
| 1  | Parameter sharing means one filter is reused across spatial locations.              | ✅ True  |
| 2  | Every spatial position learns its own convolution filter.                           | ❌ False |
| 3  | Different filters in the same layer must share identical weights.                   | ❌ False |
| 4  | The entire multi-channel filter is shared spatially.                                | ✅ True  |
| 5  | `H_out × W_out` should multiply the convolution parameter count.                    | ❌ False |
| 6  | Increasing input H/W can increase computation without increasing filter parameters. | ✅ True  |
| 7  | Shared parameters always produce identical activations.                             | ❌ False |
| 8  | Parameter sharing contributes to translation-equivariant structure.                 | ✅ True  |
| 9  | Equivariance and invariance mean the same thing.                                    | ❌ False |
| 10 | Gradients from multiple spatial uses can contribute to one shared weight update.    | ✅ True  |
| 11 | Parameter sharing normally means all convolution layers use the same filters.       | ❌ False |
| 12 | Parameter sharing is an architectural inductive bias.                               | ✅ True  |

---

# ⭐ Staff Engineer Challenge

## Scenario

You are reviewing a model that receives:

`128 × 128 × 64`

Input.

A layer uses:

* Kernel = `3 × 3`
* Padding = `1`
* Stride = `1`
* Output channels = `128`

An engineer proposes replacing standard convolution with a locally connected version where:

> **every output position has independent weights**

because they believe it will always improve accuracy.

---

## Question 1

How many weights does the standard shared convolution use?

### ✅ Answer

`3 × 3 × 64 × 128`

`= 73,728`

So:

> **Shared convolution = 73,728 weights**

before biases.

---

## Question 2

How many spatial output positions are there?

### ✅ Answer

Padding and stride preserve:

`128 × 128`

So:

`128 × 128`

`= 16,384`

spatial positions.

---

## Question 3

Approximately how many weights would the unshared version use?

### ✅ Answer

Each spatial position would need its own:

`3 × 3 × 64 × 128`

weights.

So:

`16,384 × 73,728`

`= 1,207,959,552`

weights.

That is over:

> **1.2 billion weights**

before biases.

---

## Question 4

How many times larger is this than the shared convolution?

### ✅ Answer

Since every one of the:

`16,384`

spatial positions now gets its own parameter set:

> **16,384× more weights**

---

## Question 5

Why might the unshared model overfit more easily?

### ✅ Answer

Because it has dramatically more degrees of freedom.

Instead of learning:

> one detector useful across locations

it can memorize:

> position-specific patterns

This increases:

* parameter count
* data requirements
* overfitting risk

---

## Question 6

What useful property is weakened by removing sharing?

### ✅ Answer

The model loses the strong assumption that:

> **the same pattern should be recognized similarly at different locations**

This weakens:

* spatial generalization
* parameter efficiency
* convolution's translation-equivariant structure

---

## Question 7

Could unshared local weights ever be useful?

### ✅ Answer

Potentially, yes.

If the problem has strongly position-dependent behavior, locally connected or position-aware transformations may be useful.

But the trade-off must be justified because they sacrifice:

> **the efficiency and inductive bias provided by sharing**

So the correct engineering principle is:

> **Do not remove parameter sharing merely to add capacity; remove or modify it only when the task provides a reason for location-specific transformations.**

---

# 🎯 Final Exercise

Complete the statements:

### 1.

Convolution weight count:

`__________`

### 2.

Spatial positions:

`__________`

the same filter weights.

### 3.

Larger H/W increases:

`__________`

but does not directly increase:

`__________`

### 4.

Input translation tends to cause:

`__________`

### ✅ Answer

1.

`K_h × K_w × C_in × C_out`

2.

> **reuse**

3.

Larger H/W increases:

> **computation and activations**

but does not directly increase:

> **convolution filter parameter count**

4.

Input translation tends to cause:

> **a corresponding translation of the convolutional response**

---

# 🧠 Final Mental Model

```text
Learn Filter Once
↓
Reuse at Many Spatial Locations
↓
Different Local Inputs
↓
Different Activations
↓
One Feature Map
```

During training:

```text
Many Spatial Uses
↓
Many Gradient Contributions
↓
One Shared Parameter Gradient
↓
One Weight Update
```

---

# ⭐ Golden Rule

> **Parameter sharing means many spatial computations reuse one learned filter: this reduces parameter count, lets the same feature detector work across locations, and gives convolution its powerful spatial inductive bias.**
