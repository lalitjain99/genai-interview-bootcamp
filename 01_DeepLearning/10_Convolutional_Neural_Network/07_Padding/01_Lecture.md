# 📘 Lecture — Padding in CNNs

> **Core Idea:** Padding adds extra values around the input boundary so the filter can operate near the edges and so we can control how much the spatial dimensions shrink.

---

# 🎯 Learning Objectives

By the end of this lecture, you should understand:

* 🧱 What padding is
* 📉 Why convolution can shrink feature maps
* ⚠️ Why border pixels are used less often without padding
* 0️⃣ What zero padding means
* ✅ What `valid` convolution means
* 🟰 What `same`-style padding means
* 📐 How kernel size relates to padding
* ➡️ How padding interacts with stride
* 🗺️ How padding helps preserve spatial structure
* 🧠 Why padding is useful in deep CNN architectures
* ⚠️ Why padding does not create new learned information

---

# 🧭 1. The Boundary Problem

We already know how convolution works:

Filter
↓
Placed on a local region
↓
Multiply + Sum
↓
One Activation
↓
Move Filter
↓
Repeat

But eventually the filter reaches the edge of the image.

Suppose the input is:

`5 × 5`

and the filter is:

`3 × 3`

The filter cannot move beyond the image boundary unless we somehow provide values outside the original input.

So without any extra treatment:

> **The filter only operates where it fully fits inside the input.**

This creates two important effects:

1. The output becomes smaller.
2. Border pixels participate in fewer convolution operations.

---

# 📉 2. Why Does Convolution Shrink the Output?

Consider:

Input:

`5 × 5`

Kernel:

`3 × 3`

Stride:

`1`

No padding.

The filter can start only where the full `3 × 3` region fits.

Horizontally:

3 valid positions

Vertically:

3 valid positions

So output becomes:

`3 × 3`

Therefore:

`5 × 5`

↓ convolution

`3 × 3`

The spatial dimensions have shrunk.

---

# ⚠️ 3. Border Pixels Are Seen Less Often

Consider a pixel near the center of the image.

Many overlapping `3 × 3` windows may include that pixel.

But a corner pixel can appear in far fewer windows.

For example:

Top-left corner pixel

may participate in only:

> **one local convolution region**

while a center pixel may participate in several.

So without padding:

> **central pixels influence more convolution calculations than border pixels.**

This is sometimes called the:

> **boundary or edge effect**

---

# 🧱 4. What Is Padding?

Padding means:

> **adding extra values around the border of the input**

Suppose the original input is:

a b c
d e f
g h i

With one layer of padding:

0 0 0 0 0
0 a b c 0
0 d e f 0
0 g h i 0
0 0 0 0 0

The original:

`3 × 3`

input becomes:

`5 × 5`

after adding:

`1`

value around every side.

That extra border is:

> **Padding**

---

# 0️⃣ 5. Zero Padding

The most common form is:

> **Zero Padding**

This means the added boundary values are:

`0`

Example:

Original:

1 2 3
4 5 6
7 8 9

Padding = 1:

0 0 0 0 0
0 1 2 3 0
0 4 5 6 0
0 7 8 9 0
0 0 0 0 0

Now the filter can be centered closer to the original image boundaries.

---

# 🗺️ 6. Why Padding Helps at the Borders

Without padding:

The filter cannot extend beyond the image.

With padding:

The filter can partially overlap the original border while the remaining kernel positions interact with padded values.

So:

Original Border Pixel
↓
Can Participate in More Convolution Windows

This gives boundary information:

> **more opportunity to influence the learned representation**

---

# 📐 7. A 5×5 Example

Suppose:

Input:

`5 × 5`

Kernel:

`3 × 3`

Stride:

`1`

---

## Without Padding

Padding:

`0`

Output:

`3 × 3`

---

## With Padding = 1

Padding adds one row/column around all sides.

Effective input becomes:

`7 × 7`

Now the `3 × 3` filter can occupy:

`5 × 5`

valid positions.

So output becomes:

`5 × 5`

Therefore:

> **Padding 1 preserves the spatial size for a 3×3 kernel with stride 1.**

---

# 🟰 8. Same-Style Padding Intuition

A common goal is:

# Input Height/Width

Output Height/Width

This is commonly called:

> **same padding**

For example:

Input:

`32 × 32`

Kernel:

`3 × 3`

Stride:

`1`

Padding:

`1`

Output:

`32 × 32`

So spatial size is preserved.

---

# ✅ 9. What Does “Valid” Mean?

You may hear:

> `padding = valid`

This usually means:

> **no padding**

The filter is applied only where it completely fits inside the original input.

So:

Valid
→ No extra border
→ Output usually shrinks

---

# 🟰 10. What Does “Same” Mean?

You may also hear:

> `padding = same`

The general intention is:

> **choose padding so the output size follows a same-size style rule**

For stride `1`, this commonly means:

# Input spatial size

Output spatial size

Example:

`5 × 5`

↓ 3×3 kernel, stride 1, padding 1

`5 × 5`

---

# ⚠️ 11. “Same” Does Not Always Mean Exactly Same for Larger Stride

This is an important nuance.

When:

`stride = 1`

same-style padding can preserve input height and width.

But when:

`stride > 1`

the output is still typically smaller.

For example:

Input:

`32 × 32`

Stride:

`2`

Even with same-style padding, the output may be around:

`16 × 16`

So:

> **Padding cannot cancel the downsampling effect of a larger stride in the usual same-padding convention.**

We will make this exact in the output-dimension lecture.

---

# 📏 12. How Much Padding Do We Need?

For common odd-sized kernels and stride `1`:

### Kernel = 3 × 3

Padding often:

`1`

### Kernel = 5 × 5

Padding often:

`2`

### Kernel = 7 × 7

Padding often:

`3`

The pattern is:

> **Padding roughly extends half the kernel size on each side**

for odd kernels when we want to preserve spatial size.

Conceptually:

`P = (K - 1) / 2`

for stride 1 and odd kernel size.

We will derive the general formula later.

---

# 🧩 13. Why Does a 3×3 Kernel Need Padding 1?

A `3 × 3` kernel extends:

`1 pixel`

away from its center in every direction.

So if we want the filter to be centered over an original border pixel:

we need:

`1 extra position`

outside the image.

Therefore:

`3 × 3 kernel`

commonly pairs with:

`padding = 1`

---

# 🧩 14. Why Does a 5×5 Kernel Need Padding 2?

A `5 × 5` kernel extends:

`2 pixels`

away from its center.

So to preserve spatial size with stride 1:

we typically need:

`2 padded positions`

around each side.

Therefore:

`5 × 5 kernel`

commonly pairs with:

`padding = 2`

---

# 🧠 15. Padding Does NOT Add Real Image Information

Suppose we add zeros around an image.

Those zeros are not:

* newly observed pixels
* learned features
* real scene information

They are simply:

> **artificial boundary values used to define the convolution operation**

So padding helps with:

* boundary handling
* output-size control

but does not create new semantic information.

---

# ⚠️ 16. Can Zero Padding Affect Activations Near Borders?

Yes.

A filter near the boundary now interacts partly with:

> padded zeros

instead of only real input values.

Therefore, edge activations can be influenced by the padding strategy.

This is one reason different padding methods exist.

---

# 🧱 17. Zero Padding Is Not the Only Padding Type

Although zero padding is very common, other strategies exist.

Examples include:

* Reflection padding
* Replication padding
* Circular padding

For example:

### Reflection Padding

The border is extended using reflected input values.

### Replication Padding

Border values are repeated outward.

These methods may be useful when zeros create undesirable boundary artifacts.

But for standard CNN interviews:

> **Zero padding is the most important one to understand.**

---

# 📍 18. Padding and Spatial Information

Without padding, every convolutional layer can reduce:

`H × W`

If many layers are stacked:

Large Input
↓
Conv
↓
Smaller
↓
Conv
↓
Even Smaller
↓
Conv
↓
Much Smaller

Spatial information can disappear quickly.

Padding lets us build deep networks while controlling:

> **how fast spatial resolution changes**

---

# 🏗️ 19. Why Padding Matters in Deep CNNs

Imagine stacking ten `3 × 3` convolutions with:

* stride = 1
* no padding

Each layer would shrink the spatial dimensions.

For one dimension:

`100`

↓ Conv

`98`

↓ Conv

`96`

↓ Conv

...

This can rapidly reduce feature-map size.

With:

`padding = 1`

each `3 × 3`, stride-1 convolution can preserve:

> **the same spatial height and width**

This makes deep architectures much easier to design.

---

# 🔗 20. Padding and Residual Connections

Remember residual connections:

`y = F(x) + x`

For simple element-wise addition:

`F(x)`

and:

`x`

need compatible shapes.

If convolution unintentionally changes height and width:

the addition may no longer work directly.

Padding can help preserve:

`H × W`

inside residual blocks.

So padding is also important for:

> **shape compatibility**

in architectures such as ResNets.

---

# ➡️ 21. Padding and Stride Work Together

Padding and stride affect spatial size in different ways.

### Padding

Can:

> preserve or enlarge the effective boundary

### Stride

Can:

> reduce the number of sampled positions

Example:

Input:

`32 × 32`

Kernel:

`3 × 3`

Padding:

`1`

Stride:

`1`

Output:

`32 × 32`

But with:

Stride:

`2`

the output becomes much smaller.

So:

> **Padding and stride must be considered together when determining output size.**

---

# 📦 22. Padding Does Not Determine Output Channels

Suppose:

Input:

`32 × 32 × 3`

Filters:

`64`

Kernel:

`3 × 3`

Padding:

`1`

Output might be:

`32 × 32 × 64`

The:

`64`

comes from:

> **number of filters**

Padding affects:

> spatial dimensions

not:

> number of output channels

---

# 🎛️ 23. Padding Does Not Change Filter Size

If we have:

Kernel:

`3 × 3`

Padding:

`1`

the kernel is still:

`3 × 3`

Padding changes:

> the effective input boundary

It does NOT make the kernel:

`5 × 5`

So:

Padding
≠
Kernel Size

---

# 🧠 24. Padding vs Stride vs Kernel Size

Keep these three concepts separate.

### 🎛️ Kernel Size

> How large a local region the filter examines

### ➡️ Stride

> How far the filter moves

### 🧱 Padding

> What extra boundary is added around the input

All three influence:

> `H_out` and `W_out`

but for different reasons.

---

# 📊 25. Quick Comparison

| Setting           | What It Controls       |
| ----------------- | ---------------------- |
| Kernel Size       | Local receptive window |
| Stride            | Filter movement        |
| Padding           | Boundary extension     |
| Number of Filters | Output channels        |

---

# 🧠 26. Complete Mental Story

Without Padding:

Input
↓
Filter Must Stay Fully Inside
↓
Border Positions Limited
↓
Output Shrinks

With Padding:

Input
↓
Add Boundary Values
↓
Filter Can Operate Near Borders
↓
More Valid Positions
↓
Better Control of Output Size

---

# 🎤 30-Second Interview Answer

> **Padding adds extra values, usually zeros, around the boundary of an input before convolution. Without padding, filters can only operate where they fully fit, causing spatial dimensions to shrink and border pixels to participate in fewer convolution windows. Padding gives the filter room to operate near boundaries and lets us control output size. For example, with stride 1, a 3×3 kernel with padding 1 can preserve the input height and width.**

---

# 📌 Key Takeaways

* 🧱 Padding adds values around input boundaries
* 0️⃣ Zero padding is the most common form
* 📉 No padding usually shrinks spatial dimensions
* ⚠️ Border pixels participate in fewer windows without padding
* ✅ `valid` usually means no padding
* 🟰 `same` aims to preserve same-style spatial dimensions
* `3 × 3` + stride 1 + padding 1 → often same H/W
* `5 × 5` + stride 1 + padding 2 → often same H/W
* ➡️ Stride and padding must be considered together
* 📦 Padding does not control output channels
* 🎛️ Padding does not change kernel weights or kernel size
* 🧠 Padding does not create new image information
* 🏗️ Padding helps prevent spatial dimensions shrinking too quickly in deep CNNs

---

# ⭐ Golden Rule

> **Padding gives the filter room to operate at the boundaries and gives the architecture control over how much spatial resolution is preserved.**

---

# ➡️ Next Topic

## 📘 Output Dimension

Next we will combine:

* Input size
* Kernel size
* Padding
* Stride

into one exact formula:

`Output = floor((Input + 2P - K) / S) + 1`

and learn how to calculate CNN shapes confidently during interviews.
