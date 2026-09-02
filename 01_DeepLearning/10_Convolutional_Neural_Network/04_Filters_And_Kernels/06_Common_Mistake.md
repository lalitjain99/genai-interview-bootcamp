# 🚫 Common Mistakes — Filters and Kernels in CNNs

> **Core Idea:** Most mistakes come from confusing filter size, filter depth, number of filters, feature maps, and parameter sharing.

---

## ❌ Mistake 1: Thinking a `3 × 3` filter always has only 9 weights

### Wrong

> A `3 × 3` filter always contains 9 trainable weights.

### Correct

That is only true for:

> **single-channel input**

For RGB input:

`3 × 3 × 3 = 27 weights`

For input with 32 channels:

`3 × 3 × 32 = 288 weights`

So:

> **A standard filter spans all input channels.**

---

## ❌ Mistake 2: Confusing filter size with number of filters

### Wrong

> `64 filters of size 3 × 3` means the filter size is 64.

### Correct

These are different:

`3 × 3`  
→ spatial size of each filter

`64`  
→ number of different filters

So:

> **Filter Size ≠ Number of Filters**

---

## ❌ Mistake 3: Thinking one filter produces multiple final feature maps

### Wrong

> One filter creates one feature map per input channel.

### Correct

In a standard convolution, one complete filter spans all input channels and combines their contributions.

Then:

> **1 filter → 1 output feature map**

So:

`64 filters → 64 output feature maps`

---

## ❌ Mistake 4: Thinking input channels must equal output channels

### Wrong

> If the input has 3 channels, the output must also have 3 channels.

### Correct

Output channels are determined by:

> **the number of filters**

Example:

Input Channels = `3`

Filters = `64`

Output Channels = `64`

---

## ❌ Mistake 5: Thinking different filters share the same weights

### Wrong

> CNN parameter sharing means all filters use identical weights.

### Correct

Parameter sharing happens:

> **across spatial locations within the same filter**

Different filters normally learn different weights.

So:

Filter A  
→ shared across space

Filter B  
→ shared across space

But:

`Weights_A ≠ Weights_B`

---

## ❌ Mistake 6: Thinking filter weights change depending on location

### Wrong

> The filter uses one set of weights at the top of the image and another set at the bottom.

### Correct

The same filter weights are reused at every spatial location.

Only the:

> **input region**

changes.

This is one of the main ideas behind convolution.

---

## ❌ Mistake 7: Thinking filters are usually manually designed

### Wrong

> We manually create vertical-edge, horizontal-edge, and texture filters.

### Correct

In modern CNNs, filter weights are usually:

> **learned through backpropagation**

They start from an initialization and are updated using gradients.

Hand-crafted edge filters are mainly used for intuition.

---

## ❌ Mistake 8: Assuming every filter has an obvious human-readable meaning

### Wrong

> Every CNN filter can be labeled as edge, circle, eye, texture, etc.

### Correct

Some filters may be easy to visualize.

Others may learn complex combinations that do not have a simple human-readable interpretation.

The network optimizes:

> **task performance**

not interpretability of each filter.

---

## ❌ Mistake 9: Thinking a filter and a feature map are the same thing

### Wrong

> The feature map is the learned filter.

### Correct

They are different.

### 🎛️ Filter

Trainable parameters

### 🗺️ Feature Map

Activations produced by applying the filter

So:

Filter  
→ Learned Weight Tensor

Feature Map  
→ Computed Output

---

## ❌ Mistake 10: Forgetting input channels in parameter-count calculations

### Wrong

For:

- Kernel = `3 × 3`
- Input Channels = `32`
- Filters = `64`

calculate:

`3 × 3 × 64`

### Correct

The full formula is:

`K_h × K_w × C_in × C_out`

So:

`3 × 3 × 32 × 64`

= `18,432 weights`

If bias is used:

`+64 biases`

---

## ❌ Mistake 11: Thinking deeper filters still operate directly on RGB pixels

### Wrong

> Every convolutional layer scans the original RGB image.

### Correct

Only the first layer receives the original image directly.

Later layers receive:

> **feature maps from previous layers**

So deeper filters operate on learned representations, not directly on raw RGB pixels.

---

## ❌ Mistake 12: Assuming early filters always learn edges

### Wrong

> The first CNN layer must learn edge detectors.

### Correct

Edge-like filters are common and useful for intuition, but not guaranteed.

The network learns whatever local patterns help reduce the loss.

So it is safer to say:

> **Early layers often learn simpler local features.**

---

## ❌ Mistake 13: Thinking larger filters are always better

### Wrong

> A bigger kernel always captures more information and therefore performs better.

### Correct

Larger filters:

- Use more parameters
- Increase computation
- May weaken the benefit of highly local processing

Small kernels such as `3 × 3` are common because deeper layers can combine local features progressively.

So:

> **Bigger is not automatically better.**

---

## ❌ Mistake 14: Thinking “filter” and “kernel” must always mean completely different things

### Wrong

> If someone says kernel instead of filter, they are describing a completely different CNN component.

### Correct

In everyday CNN discussions:

> **Filter and kernel are often used interchangeably.**

There can be more precise terminology in specialized contexts, but for standard CNN learning:

`Filter ≈ Kernel`

is a useful working convention.

---

# ⚡ Quick Trap Table

| Wrong Idea | Correct Idea |
|---|---|
| `3 × 3` always means 9 weights | Multiply by input channels |
| Filter size = number of filters | They are different concepts |
| One filter → map per input channel | One full filter → one output map |
| Input channels = output channels | Output channels = number of filters |
| All filters share weights | Sharing is within each filter across space |
| Filter weights change by location | Same weights reused spatially |
| Filters are manually designed | Usually learned with backpropagation |
| Every filter is easily interpretable | Many learned filters are abstract |
| Filter = feature map | Filter is parameter, map is activation |
| Ignore `C_in` in parameter count | Always include input channels |
| Deep filters see RGB pixels | They see previous feature maps |
| Early layers must learn edges | They often learn simple local features |
| Bigger kernels are always better | Kernel size is a design tradeoff |
| Filter and kernel are unrelated | Often used interchangeably |

---

# 🧠 Final Mental Model

Input with `C_in` Channels  
↓  
Each Filter Spans All `C_in` Channels  
↓  
Same Filter Weights Shared Across Space  
↓  
Different Filters Have Different Weights  
↓  
Each Filter Produces One Feature Map  
↓  
Number of Filters = `C_out`

---

# ⭐ Golden Rule

> **Do not confuse spatial kernel size, input-channel depth, number of filters, and output channels: one complete filter spans all input channels and produces one output feature map.**