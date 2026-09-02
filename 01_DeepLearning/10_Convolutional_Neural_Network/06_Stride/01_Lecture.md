# 📘 Lecture — Stride in CNNs

> **Core Idea:** Stride tells us how many pixels the filter moves after each convolution step. A larger stride means the filter visits fewer positions, producing a smaller feature map.

---

# 🎯 Learning Objectives

By the end of this lecture, you should understand:

* ➡️ What stride means
* 🧭 How a filter moves across an input
* 🔢 Difference between stride `1`, `2`, and larger values
* 📉 Why larger stride reduces feature-map size
* ⚡ Why larger stride can reduce computation
* 🔍 The trade-off between efficiency and spatial detail
* ↔️ Horizontal and vertical stride
* 🧠 How stride can perform downsampling
* ⚠️ Difference between stride, kernel size, and padding

---

# 🧭 1. Where Stride Enters the Convolution Story

So far, we know this:

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

But there is one important question:

> **How far does the filter move each time?**

That movement distance is called:

> **Stride**

---

# ➡️ 2. What Is Stride?

Stride is the number of pixels the filter shifts after one convolution calculation.

If:

`Stride = 1`

the filter moves:

> **1 pixel at a time**

If:

`Stride = 2`

the filter moves:

> **2 pixels at a time**

If:

`Stride = 3`

the filter moves:

> **3 pixels at a time**

So:

> **Stride controls the step size of the filter movement.**

---

# 🔍 3. Stride = 1

Suppose a filter begins at the left side of an image.

With:

`Stride = 1`

the movement is:

Position 1
↓ move 1 pixel
Position 2
↓ move 1 pixel
Position 3
↓ move 1 pixel
Position 4

The filter examines many neighboring regions.

So:

> **Stride 1 gives dense spatial scanning.**

---

# 🧩 4. A Simple 1D Intuition

Imagine an input:

`[1, 2, 3, 4, 5, 6]`

and a kernel of size:

`3`

With stride:

`1`

the regions are:

`[1, 2, 3]`

then:

`[2, 3, 4]`

then:

`[3, 4, 5]`

then:

`[4, 5, 6]`

Notice how much the regions overlap.

That happens because:

> **the filter only moves one position at a time.**

---

# 🖼️ 5. Stride = 1 in 2D

Suppose we have a `3 × 3` filter.

At the first position:

A A A . .
A A A . .
A A A . .
. . . . .
. . . . .

Move one pixel to the right:

. A A A .
. A A A .
. A A A .
. . . . .
. . . . .

The neighboring regions overlap heavily.

This allows the CNN to capture:

> **fine spatial detail**

---

# ⏩ 6. Stride = 2

Now suppose:

`Stride = 2`

The filter moves two pixels at a time.

Conceptually:

Position 1
↓ move 2 pixels
Position 2
↓ move 2 pixels
Position 3

So the filter skips some intermediate starting positions.

---

# 🧩 7. 1D Example with Stride = 2

Input:

`[1, 2, 3, 4, 5, 6, 7]`

Kernel size:

`3`

Stride:

`2`

The regions become:

`[1, 2, 3]`

then:

`[3, 4, 5]`

then:

`[5, 6, 7]`

Compare this with stride `1`.

Stride 1 visits more positions.

Stride 2 visits fewer positions.

Therefore:

> **larger stride produces fewer output activations.**

---

# 📉 8. Why Larger Stride Produces a Smaller Feature Map

Remember:

One valid filter position
→ One activation

So:

More filter positions
→ Larger feature map

Fewer filter positions
→ Smaller feature map

If we increase stride:

Filter moves farther
↓
Visits fewer locations
↓
Produces fewer activations
↓
Feature map becomes smaller

That is the main relationship.

---

# 🗺️ 9. Visual Intuition

Imagine scanning a floor.

### Stride = 1

You check:

Every nearby tile

Result:

> detailed map

### Stride = 2

You check:

Every second tile

Result:

> fewer measurements

So:

> **larger stride samples the spatial input more sparsely.**

---

# 🔍 10. Stride and Spatial Detail

Stride affects how densely we sample the input.

### Smaller Stride

For example:

`Stride = 1`

Benefits:

* More spatial positions examined
* More detailed output
* Better spatial resolution

But:

* More computation
* Larger feature maps

---

### Larger Stride

For example:

`Stride = 2`

Benefits:

* Smaller feature maps
* Less computation
* Useful for downsampling

But:

* Fewer spatial locations are represented
* Some fine-grained information may be lost

So stride creates a trade-off:

> **Spatial Detail ↔ Computational Efficiency**

---

# ⚡ 11. Why Does Larger Stride Reduce Computation?

At every filter position, the network performs a convolution calculation.

If stride is larger:

Fewer Positions
↓
Fewer Convolution Operations
↓
Less Computation

It also produces smaller output tensors, which means:

> later CNN layers may also require less computation.

---

# 📉 12. Stride as Downsampling

Downsampling means:

> **Reducing the spatial resolution of a representation**

For example:

Large Feature Map
↓
Strided Convolution
↓
Smaller Feature Map

A convolution with:

`Stride = 2`

is often used to reduce spatial size.

Conceptually:

`H × W`

may become approximately:

`H/2 × W/2`

depending on kernel size and padding.

We will calculate this precisely later.

---

# 🧠 13. Why Would We Want Downsampling?

At first, reducing resolution may sound harmful.

But CNNs often do not need full spatial resolution at every depth.

Early layers may benefit from detailed spatial information.

Later layers may focus more on:

* larger patterns
* feature combinations
* semantic structure

So many architectures gradually reduce:

`Height × Width`

while increasing:

> **the number of feature channels**

A common pattern is:

Large Spatial Size

* Few Channels

↓

Smaller Spatial Size

* More Channels

---

# ↔️ 14. Horizontal and Vertical Stride

Stride does not technically have to be the same in both directions.

We can define:

`Stride = (S_h, S_w)`

where:

`S_h`

= vertical movement

and:

`S_w`

= horizontal movement

For example:

`Stride = (2, 1)`

means:

* move 2 positions vertically
* move 1 position horizontally

In many CNN architectures, we commonly use:

`(1,1)`

or:

`(2,2)`

which are often simply written as:

`stride = 1`

or:

`stride = 2`

---

# ⚠️ 15. Stride Does NOT Change Filter Weights

Suppose:

Filter:

1   0  -1
1   0  -1
1   0  -1

Changing:

`Stride = 1`

to:

`Stride = 2`

does NOT change the filter weights.

Stride only changes:

> **where the filter is applied**

So:

Filter Weights
→ Same

Movement Step
→ Different

Output Size
→ Different

---

# ⚠️ 16. Stride Is NOT Kernel Size

Do not confuse these two.

Suppose:

Kernel Size = `3 × 3`

Stride = `2`

### Kernel Size

Tells us:

> how large the local region is

### Stride

Tells us:

> how far the filter moves

So:

`Kernel Size ≠ Stride`

---

# 🧱 17. Stride Is NOT Padding

These concepts affect convolution differently.

### Kernel Size

Controls:

> size of local region examined

### Stride

Controls:

> movement step

### Padding

Controls:

> what happens around the input boundaries

All three can affect:

> **output spatial dimensions**

But they do so for different reasons.

---

# 📐 18. A Small 2D Example

Suppose we have:

Input:

`5 × 5`

Kernel:

`3 × 3`

No padding.

---

## Stride = 1

The filter can begin at:

3 horizontal positions

and:

3 vertical positions

So the output is:

`3 × 3`

---

## Stride = 2

Now the filter jumps farther.

Valid starting positions become fewer.

The output becomes:

`2 × 2`

So:

Stride 1
→ 9 activations

Stride 2
→ 4 activations

This directly shows:

> **Increasing stride reduces the number of filter positions.**

---

# 🧮 19. Why Does the 5×5 Example Give 2×2 for Stride 2?

Consider one dimension.

Input positions:

`1 2 3 4 5`

Kernel width:

`3`

First position covers:

`1 2 3`

Stride = `2`

Next position starts at:

`3`

and covers:

`3 4 5`

There is no valid next full position.

So there are:

`2 positions`

in that dimension.

Since the same happens vertically:

`2 × 2`

output.

---

# 🔗 20. Stride and Feature Maps

Remember:

A feature map stores one activation per filter location.

Therefore:

Stride controls:

> **how densely activations appear in the feature map**

Stride 1:

Dense feature map

Stride 2:

More spatially compressed feature map

So stride directly influences:

> **feature-map spatial resolution**

---

# 🎛️ 21. Stride with Multiple Filters

Suppose a convolutional layer has:

`64 filters`

Stride affects the:

> height and width

of every feature map.

But it does NOT change:

> the number of feature maps

For example:

64 Filters
+
Stride 2

may produce:

`H_out × W_out × 64`

The `64` comes from:

> number of filters

The reduced `H_out` and `W_out` come partly from:

> stride

---

# 🧠 22. Important Shape Distinction

Different CNN settings control different output dimensions.

### Number of Filters

Controls:

`C_out`

### Stride

Influences:

`H_out` and `W_out`

So if we have:

`H × W × C`

Stride mainly affects:

> spatial dimensions

while filter count determines:

> output channels

---

# 🏗️ 23. Where Is Strided Convolution Used?

Strided convolution is commonly used when a network wants to:

* reduce feature-map resolution
* reduce computational cost
* progressively compress spatial information
* replace or complement pooling

For example:

Feature Map
`64 × 64 × 64`

↓

Conv with Stride 2

↓

Feature Map
approximately
`32 × 32 × 128`

The exact spatial size depends on kernel size and padding.

---

# ⚠️ 24. Is Larger Stride Always Better?

No.

Suppose stride becomes too large.

Then the filter may skip many useful positions.

That can cause:

> loss of important spatial detail

So we do not simply maximize stride for speed.

Architecture design requires balancing:

* accuracy
* spatial information
* computation
* memory

---

# 🎯 25. Stride = 1 vs Stride = 2

| Property          | Stride 1   | Stride 2  |
| ----------------- | ---------- | --------- |
| Filter Movement   | 1 pixel    | 2 pixels  |
| Positions Visited | More       | Fewer     |
| Feature Map       | Larger     | Smaller   |
| Spatial Detail    | Higher     | Lower     |
| Computation       | Higher     | Lower     |
| Downsampling      | Usually No | Often Yes |

---

# 🧠 26. Complete Mental Story

Input
↓
Place Filter
↓
Compute Activation
↓
Move Filter by `Stride` Pixels
↓
Compute Next Activation
↓
Repeat
↓
Feature Map

If stride increases:

Movement Becomes Larger
↓
Fewer Positions Visited
↓
Fewer Activations
↓
Smaller Feature Map
↓
Lower Computation
↓
Potential Loss of Spatial Detail

---

# 🎤 30-Second Interview Answer

> **Stride is the number of pixels a convolutional filter moves after each operation. With stride 1, the filter moves one pixel at a time and produces a relatively dense feature map. Increasing stride causes the filter to skip more positions, reducing the spatial dimensions of the output and lowering computation. Because fewer locations are sampled, larger stride also reduces spatial resolution and can lose fine-grained information.**

---

# 📌 Key Takeaways

* ➡️ Stride = filter movement step
* `stride = 1` → move one position at a time
* `stride = 2` → move two positions at a time
* 📉 Larger stride → fewer filter positions
* 🗺️ Fewer positions → smaller feature map
* ⚡ Smaller feature map → less computation
* 🔍 Larger stride may lose spatial detail
* 📉 Strided convolution can perform downsampling
* ↔️ Stride can have separate height and width values
* 🎛️ Stride does not change filter weights
* 📏 Kernel size and stride are different concepts
* 📦 Stride affects spatial dimensions, not the number of output channels

---

# ⭐ Golden Rule

> **Stride controls how far the filter moves: larger stride means fewer convolution positions, a smaller feature map, less computation, and lower spatial resolution.**

---

# ➡️ Next Topic

## 📘 Padding

Next we will study:

* 🧱 Why convolution loses information near image boundaries
* ➕ What padding adds around the input
* 🗺️ How padding helps control feature-map size
* ⚖️ Valid vs same-style convolution
* 📐 How padding interacts with kernel size and stride
