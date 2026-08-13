# 🚫 Common Mistakes — Stride in CNNs

> **Core Idea:** Most stride mistakes come from confusing movement distance with kernel size, output channels, filter weights, or assuming larger stride is always better.

---

## ❌ Mistake 1: Thinking stride is the size of the filter

### Wrong

> `stride = 3` means the filter is `3 × 3`.

### Correct

Stride controls:

> **how far the filter moves**

Kernel size controls:

> **how large the local region is**

So:

`Kernel Size ≠ Stride`

---

## ❌ Mistake 2: Thinking stride changes filter weights

### Wrong

> If stride changes from 1 to 2, the filter learns a different set of weights.

### Correct

Stride only changes:

> **where the filter is applied**

The filter weights themselves are unchanged by the stride setting.

---

## ❌ Mistake 3: Thinking larger stride produces a larger feature map

### Wrong

> Moving farther should cover more space and therefore produce a larger output.

### Correct

Larger stride means:

Fewer Starting Positions
↓
Fewer Activations
↓
Smaller Feature Map

So:

> **Larger stride usually reduces spatial output size.**

---

## ❌ Mistake 4: Thinking stride changes the number of output channels

### Wrong

> Increasing stride from 1 to 2 reduces the number of output channels.

### Correct

Stride mainly affects:

`H_out` and `W_out`

The number of filters determines:

`C_out`

Example:

`64 filters + stride 1 → 64 output channels`

`64 filters + stride 2 → 64 output channels`

---

## ❌ Mistake 5: Thinking stride 2 means every second input pixel is completely ignored

### Wrong

> Stride 2 simply throws away every alternate pixel.

### Correct

Stride controls:

> **the starting positions of the convolution windows**

Neighboring windows may still overlap.

Example:

Kernel size = `3`

Stride = `2`

Regions:

`[1,2,3]`

`[3,4,5]`

So some input values can still participate in multiple windows.

---

## ❌ Mistake 6: Thinking stride 2 always halves the spatial size exactly

### Wrong

> `stride = 2` always converts `H × W` into exactly `H/2 × W/2`.

### Correct

That is often an approximation.

Exact output size also depends on:

* Input size
* Kernel size
* Padding
* Stride

So:

> **Stride 2 often roughly halves resolution, but not always exactly.**

---

## ❌ Mistake 7: Thinking larger stride always improves the model

### Wrong

> Larger stride is better because it makes the network faster.

### Correct

Larger stride reduces computation, but may also lose:

* Fine details
* Small features
* Precise localization information

So stride involves a trade-off:

> **Efficiency ↔ Spatial Detail**

---

## ❌ Mistake 8: Thinking stride 1 means no overlap

### Wrong

> With stride 1, each convolution window sees a completely separate region.

### Correct

For common kernels larger than `1 × 1`, stride 1 usually creates:

> **heavily overlapping local regions**

Example:

Kernel size = `3`

Stride = `1`

`[1,2,3]`

`[2,3,4]`

The windows clearly overlap.

---

## ❌ Mistake 9: Thinking stride and padding do the same thing

### Wrong

> Both are just different ways to resize the output.

### Correct

They affect convolution differently.

### Stride

Controls:

> movement step

### Padding

Controls:

> values added around the input boundaries

Both can influence output size, but they represent different operations.

---

## ❌ Mistake 10: Thinking strided convolution is identical to pooling

### Wrong

> Stride 2 convolution and pooling are exactly the same operation.

### Correct

Both can downsample, but they work differently.

### Strided Convolution

Uses:

> **learned filter weights**

### Pooling

Typically uses a fixed rule such as:

* maximum
* average

So:

> **Strided convolution performs learned downsampling.**

---

## ❌ Mistake 11: Thinking horizontal and vertical stride must always be equal

### Wrong

> Stride can only be a single number.

### Correct

Stride can be:

`(S_h, S_w)`

Example:

`stride = (2,1)`

means:

* move 2 positions vertically
* move 1 position horizontally

Equal strides are common, but not required.

---

## ❌ Mistake 12: Thinking stride only affects the current layer

### Wrong

> A smaller output only saves computation in that one convolution.

### Correct

A smaller feature map is passed to later layers.

So larger stride can also reduce:

* later-layer computation
* memory usage
* total activation size

Its effect can propagate through the network.

---

## ❌ Mistake 13: Thinking larger stride means the filter itself becomes larger

### Wrong

> If stride increases, the filter covers a larger local region.

### Correct

Stride changes:

> **how far the filter moves**

It does not change:

> **the kernel dimensions**

A `3 × 3` kernel remains `3 × 3` whether stride is 1 or 2.

---

## ❌ Mistake 14: Thinking downsampling is always harmful

### Wrong

> Reducing spatial resolution should always be avoided.

### Correct

Downsampling is often useful.

It can:

* Reduce computation
* Increase efficiency
* Compress redundant spatial information
* Help deeper layers focus on higher-level patterns

The important question is:

> **when and how aggressively to downsample**

---

## ❌ Mistake 15: Thinking stride has no effect on information retention

### Wrong

> Stride only changes computation, not the information represented.

### Correct

Larger stride samples fewer positions.

Therefore, aggressive stride may reduce:

> **fine-grained spatial information**

That is especially important for tasks involving:

* Small objects
* Segmentation
* Fine localization
* Medical-image details

---

# ⚡ Quick Trap Table

| Wrong Idea                              | Correct Idea                           |
| --------------------------------------- | -------------------------------------- |
| Stride = kernel size                    | Stride = movement step                 |
| Stride changes filter weights           | It only changes application positions  |
| Larger stride → larger output           | Larger stride → smaller spatial output |
| Stride changes output channels          | Filter count determines channels       |
| Stride 2 throws away every second pixel | It skips window starting positions     |
| Stride 2 always exactly halves size     | Exact size depends on more factors     |
| Larger stride is always better          | It trades detail for efficiency        |
| Stride 1 means no overlap               | Windows often overlap heavily          |
| Stride = padding                        | They are different concepts            |
| Strided conv = pooling                  | Learned vs fixed downsampling          |
| Stride must be equal in both directions | `(S_h, S_w)` can differ                |
| Stride affects only current computation | Smaller outputs reduce later cost too  |
| Larger stride means larger kernel       | Kernel size stays unchanged            |
| Downsampling is always bad              | It is often intentional                |
| Stride does not affect information      | Large stride can lose detail           |

---

# 🧠 Final Mental Model

Stride
↓
Controls Movement Step
↓
Controls Number of Visited Positions
↓
Controls Number of Activations
↓
Influences Spatial Output Size
↓
Affects Computation and Spatial Detail

---

# ⭐ Golden Rule

> **Stride changes where the same filter is applied, not what the filter is; increasing stride samples fewer positions, reducing spatial resolution and computation while potentially losing detail.**
