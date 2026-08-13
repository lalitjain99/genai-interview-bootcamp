# 💡 Assignment — Feature Maps in CNNs

> **Focus:** Practice how feature maps are formed, how to interpret activations, how channels flow between CNN layers, and how feature maps differ from filters and parameters.

---

# 🧩 Scenario 1 — One Filter, One Feature Map

A convolutional layer uses:

`1 filter`

and scans the complete input.

### Question

How many feature maps are produced?

### ✅ Answer

One filter produces:

> **One feature map**

So:

`1 filter → 1 feature map`

---

# 🎛️ Scenario 2 — Multiple Filters

A convolutional layer contains:

`32 filters`

### Question 1

How many feature maps are produced?

### ✅ Answer

`32`

because:

`1 filter → 1 feature map`

So:

`32 filters → 32 feature maps`

---

### Question 2

How many output channels does the layer have?

### ✅ Answer

> **32 output channels**

Because:

`Number of Feature Maps = Number of Output Channels`

---

# 📈 Scenario 3 — Interpreting an Activation

Suppose one feature map is:

0   1   2
1   8   0
0   2   1

### Question

What does the value `8` mean?

### ✅ Answer

It means:

> **The filter produced a strong response near that spatial location.**

The value `8` is:

> **an activation**

It is not:

* a filter weight
* a new trainable parameter
* necessarily an original pixel value

---

# 📍 Scenario 4 — Spatial Information

Suppose a filter responds strongly near the top-right region of an image.

Its feature map contains a high activation near the top-right.

### Question

What does this tell us?

### ✅ Answer

It tells us that:

> **The feature map preserves the relative spatial location of the filter response.**

So feature maps capture both:

* 📍 Where the response happened
* 📈 How strong the response was

---

# ⚠️ Scenario 5 — Exact Pixel Mapping

Input size:

`32 × 32`

Feature map size:

`16 × 16`

A developer says:

> "Feature-map coordinate `(10, 10)` must correspond exactly to original pixel `(10, 10)`."

### Is this correct?

### ✅ Answer

No.

Feature maps preserve:

> **relative spatial structure**

but not necessarily exact one-to-one pixel coordinates.

Operations such as:

* Kernel size
* Stride
* Padding
* Pooling

can change the spatial resolution.

---

# 🎛️ Scenario 6 — Filter vs Feature Map

A developer says:

> "The numbers inside the feature map are the learned filter weights."

### Is this correct?

### ✅ Answer

No.

### Filter

Contains:

> **trainable weights**

### Feature Map

Contains:

> **computed activations**

So:

Filter
→ learned parameter

Feature Map
→ output generated from the input

---

# 🔄 Scenario 7 — Next Layer Input

Suppose:

Conv Layer 1 has:

`16 filters`

Therefore it produces:

`16 feature maps`

### Question

How many input channels does Conv Layer 2 receive?

### ✅ Answer

Conv Layer 2 receives:

> **16 input channels**

because the previous layer's feature maps become the next layer's input channels.

So:

Conv Layer 1
↓
16 Feature Maps
↓
Conv Layer 2 Input = 16 Channels

---

# 🌈 Scenario 8 — Are Deeper Channels RGB?

Suppose the original input is:

`64 × 64 × 3`

After the first convolution:

`64 × 64 × 32`

A developer says:

> "Those 32 channels are just expanded RGB channels."

### Is this correct?

### ✅ Answer

No.

The original 3 channels are:

* Red
* Green
* Blue

But the 32 output channels are:

> **learned feature maps**

They may represent responses to:

* Edges
* Textures
* Color combinations
* Other learned local patterns

---

# 🏗️ Scenario 9 — Feature Complexity with Depth

Consider:

Layer 1
→ simple feature maps

Layer 2
→ more complex feature maps

Layer 3
→ even more abstract feature maps

### Question

Why can deeper feature maps represent more complex patterns?

### ✅ Answer

Because deeper layers operate on:

> **previously learned feature maps**

rather than directly on raw pixels.

So they can combine simpler features into more complex representations.

Conceptually:

Edges
↓
Shapes
↓
Object Parts
↓
Higher-Level Features

---

# 🔁 Scenario 10 — Same Filter, Different Image

Suppose the same trained filter processes:

Image A
and
Image B

### Question

Will it necessarily produce the same feature map?

### ✅ Answer

No.

The filter weights are the same, but:

> **feature maps depend on the input**

So:

Same Filter
+
Different Input
↓
Different Activations
↓
Different Feature Map

---

# 🔥 Scenario 11 — Pre-Activation vs Post-Activation

Suppose convolution produces:

-3   2   5
1  -4   7

ReLU is then applied.

### Question 1

What is the post-ReLU feature map?

### ✅ Answer

ReLU:

`ReLU(x) = max(0, x)`

So:

0   2   5
1   0   7

---

### Question 2

Can both the before-ReLU and after-ReLU tensors be called feature maps?

### ✅ Answer

Yes, depending on context.

More precisely:

Before ReLU
→ Pre-activation feature map

After ReLU
→ Post-activation feature map

---

# ⚠️ Scenario 12 — Strong Activation

A feature map contains a very large activation.

A developer says:

> "This proves the image definitely contains a dog."

### Is this correct?

### ✅ Answer

No.

A large activation means:

> **the filter responded strongly to a local pattern**

It does not automatically prove:

* a specific object exists
* the model is certain
* the filter has a simple semantic meaning

CNN predictions depend on:

> many feature maps and many layers working together.

---

# ✅ True / False

| #  | Statement                                                             | Answer  |
| -- | --------------------------------------------------------------------- | ------- |
| 1  | One filter produces one feature map.                                  | ✅ True  |
| 2  | Feature-map values are trainable parameters.                          | ❌ False |
| 3  | Feature maps preserve relative spatial information.                   | ✅ True  |
| 4  | Number of feature maps equals number of output channels.              | ✅ True  |
| 5  | A feature map is the same thing as a filter.                          | ❌ False |
| 6  | Deeper feature maps must still represent RGB colors.                  | ❌ False |
| 7  | Previous feature maps become input channels for the next layer.       | ✅ True  |
| 8  | The same filter always produces the same feature map for every image. | ❌ False |
| 9  | High activation means strong filter response.                         | ✅ True  |
| 10 | One high activation guarantees an object is present.                  | ❌ False |

---

# ⭐ Staff Engineer Challenge

## Scenario

You are given this CNN:

Input:

`224 × 224 × 3`

Conv Layer 1:

`64 filters`

Output:

`H₁ × W₁ × 64`

Conv Layer 2:

`128 filters`

Output:

`H₂ × W₂ × 128`

A developer says:

> "Layer 1 creates 64 learned filters, but Layer 2 only needs to look at the strongest Layer 1 feature map."

---

### Question 1

Is that how standard convolution works?

### ✅ Answer

No.

A standard Layer 2 filter spans:

> **all 64 input channels**

So it can combine information from multiple Layer 1 feature maps.

---

### Question 2

How many feature maps does Layer 1 produce?

### ✅ Answer

`64`

because:

`64 filters → 64 feature maps`

---

### Question 3

How many input channels does Layer 2 receive?

### ✅ Answer

`64`

because:

> Layer 1's 64 feature maps become Layer 2's input channels.

---

### Question 4

How many feature maps does Layer 2 produce?

### ✅ Answer

`128`

because:

`128 filters → 128 feature maps`

---

### Question 5

Why is combining multiple previous feature maps important?

### ✅ Answer

Because deeper layers can learn relationships between simpler features.

For example:

Feature A
+
Feature B
+
Feature C
↓
More Complex Learned Pattern

This is a major reason CNN representations become more abstract with depth.

---

# 🎯 Final Exercise

Complete the flow:

One Filter
↓
Produces One __________ at Each Location
↓
All Values Arranged Spatially
↓
One __________
↓
N Filters Produce N __________
↓
These Become __________ for the Next Layer

### ✅ Answer

One Filter
↓
Produces One **Activation** at Each Location
↓
All Values Arranged Spatially
↓
One **Feature Map**
↓
N Filters Produce N **Feature Maps**
↓
These Become **Input Channels** for the Next Layer

---

# 🧠 Final Mental Model

Filter
↓
Local Responses
↓
Spatial Activation Grid
↓
Feature Map
↓
Multiple Feature Maps
↓
Output Channels
↓
Next Layer Input Channels
↓
More Complex Representations

---

# ⭐ Golden Rule

> **A feature map is the input-dependent spatial activation produced by one filter, and those maps become the learned channels used by deeper CNN layers.**
