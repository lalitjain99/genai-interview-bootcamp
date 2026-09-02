```markdown id="cnn-filters-kernels-assignment"
# 💡 Assignment — Filters and Kernels in CNNs

> **Focus:** Practice filter shapes, parameter counts, output channels, parameter sharing, and how filters learn useful local patterns.

---

# 🧩 Scenario 1 — Filter Size

A CNN layer uses a spatial filter of:

`3 × 3`

on a grayscale image.

### Question

How many weights are present in one filter, ignoring bias?

### ✅ Answer

A grayscale image has:

`1 input channel`

So:

`3 × 3 × 1 = 9`

Therefore:

> **One filter has 9 weights**

---

# 🌈 Scenario 2 — RGB Filter

Now the same `3 × 3` filter is applied to an RGB image.

### Question

How many weights does one filter contain?

### ✅ Answer

RGB has:

`3 input channels`

So:

`3 × 3 × 3`

= `27`

Therefore:

> **One filter has 27 weights**

The filter spans all three input channels.

---

# 🎛️ Scenario 3 — Multiple Filters

A convolutional layer receives an RGB image and uses:

`32 filters`

Each filter has spatial size:

`3 × 3`

### Question 1

How many feature maps are produced?

### ✅ Answer

Each filter produces one feature map.

So:

`32 filters → 32 feature maps`

---

### Question 2

How many output channels are produced?

### ✅ Answer

> **32 output channels**

Because:

`Number of Filters = Number of Output Channels`

---

# 🧮 Scenario 4 — Parameter Count

A convolutional layer has:

- Kernel size = `3 × 3`
- Input channels = `3`
- Number of filters = `16`

### Question

Calculate the number of trainable parameters if each filter has one bias.

### ✅ Answer

Weights per filter:

`3 × 3 × 3`

= `27`

For 16 filters:

`27 × 16`

= `432`

Biases:

`16`

Total parameters:

`432 + 16`

= `448`

Therefore:

> **Total trainable parameters = 448**

---

# 🔁 Scenario 5 — Parameter Sharing

A developer says:

> "Since a filter moves across many image locations, it should have different weights at every location."

### Is this correct?

### ✅ Answer

No.

The same filter weights are reused across all spatial locations.

This is:

> **Parameter Sharing**

So:

Same Filter  
↓  
Same Weights  
↓  
Different Input Regions  
↓  
Different Activations

---

# 🎯 Scenario 6 — Different Filters

A layer contains:

Filter A  
Filter B  
Filter C

A developer says:

> "All three filters should share the exact same weights because CNNs use parameter sharing."

### Is this correct?

### ✅ Answer

No.

Parameter sharing happens:

> **within the same filter across spatial locations**

Different filters usually have:

> **different learned weights**

Otherwise, all filters would perform essentially the same transformation.

---

# 🧠 Scenario 7 — What Does a Filter Learn?

Suppose one learned filter responds strongly to:

- vertical intensity changes

and another responds strongly to:

- texture-like patterns

### Question

Were these roles manually assigned?

### ✅ Answer

Usually no.

The filters are learned through training.

Training flow:

Initialize  
↓  
Forward Pass  
↓  
Loss  
↓  
Backpropagation  
↓  
Optimizer Update

The model discovers useful filter weights automatically.

---

# 🏗️ Scenario 8 — Early vs Deep Filters

Suppose a CNN has many convolutional layers.

### Question

Which is more likely?

### Option A

All filters in every layer detect simple edges.

### Option B

Earlier filters may learn simpler patterns while deeper filters may respond to more complex feature combinations.

### ✅ Answer

> **Option B**

Conceptually:

Pixels  
↓  
Edges / Simple Patterns  
↓  
Textures / Shapes  
↓  
Object Parts  
↓  
Higher-Level Features

---

# 📏 Scenario 9 — Filter Size vs Number of Filters

A layer is described as:

`64 filters of size 3 × 3`

### Question

What does `64` represent?

### ✅ Answer

`64` represents:

> **the number of filters**

and therefore:

> **the number of output channels**

---

### Question

What does `3 × 3` represent?

### ✅ Answer

It represents:

> **the spatial size of each filter**

So:

Filter Size  
≠  
Number of Filters

---

# 📦 Scenario 10 — Multi-Channel Input

Input shape:

`64 × 64 × 32`

Convolution layer:

- Kernel size = `3 × 3`
- Number of filters = `128`

### Question 1

What is the conceptual shape of one filter?

### ✅ Answer

Each filter spans all:

`32 input channels`

So one filter has shape:

`3 × 3 × 32`

---

### Question 2

How many weights are in one filter?

### ✅ Answer

`3 × 3 × 32`

= `288`

---

### Question 3

How many feature maps are produced?

### ✅ Answer

`128`

because:

`128 filters → 128 feature maps`

---

# ✅ True / False

| # | Statement | Answer |
|---|---|---|
| 1 | A filter contains trainable weights. | ✅ True |
| 2 | A `3 × 3` filter always has exactly 9 weights. | ❌ False |
| 3 | A standard filter spans all input channels. | ✅ True |
| 4 | One filter produces one feature map. | ✅ True |
| 5 | Number of filters determines output channels. | ✅ True |
| 6 | Different filters must have identical weights. | ❌ False |
| 7 | The same filter weights are reused across spatial locations. | ✅ True |
| 8 | Filters are usually manually designed in modern CNNs. | ❌ False |
| 9 | Filter size and number of filters mean the same thing. | ❌ False |
| 10 | Deeper filters may learn more abstract representations. | ✅ True |

---

# ⭐ Staff Engineer Challenge

## Scenario

You are given:

Input:

`128 × 128 × 64`

Convolutional layer:

- Kernel size = `3 × 3`
- Number of filters = `256`
- One bias per filter

A developer estimates:

> "There are only `3 × 3 × 256` weights because each filter is 3 × 3."

### Question 1

What is wrong with this calculation?

### ✅ Answer

It ignores the:

> **64 input channels**

Each filter must span all input channels.

So one filter has:

`3 × 3 × 64`

weights.

---

### Question 2

How many weights are in one filter?

### ✅ Answer

`3 × 3 × 64`

= `576`

---

### Question 3

How many filter weights are in the entire layer?

### ✅ Answer

`576 × 256`

= `147,456`

---

### Question 4

How many biases are there?

### ✅ Answer

One bias per output filter:

`256 biases`

---

### Question 5

What is the total trainable parameter count?

### ✅ Answer

`147,456 + 256`

= `147,712`

Therefore:

> **Total trainable parameters = 147,712**

---

### Question 6

How many output channels does the layer produce?

### ✅ Answer

`256`

because:

> **Each filter produces one feature map**

---

# 🎯 Final Exercise

Complete the flow:

Input with `C_in` Channels  
↓  
Each Filter Has Shape  
________________________  
↓  
One Filter Produces  
________________________  
↓  
`N` Filters Produce  
________________________  
↓  
Output Channels =  
________________________

### ✅ Answer

Input with `C_in` Channels  
↓  
Each Filter Has Shape  
`K_h × K_w × C_in`  
↓  
One Filter Produces  
One Feature Map  
↓  
`N` Filters Produce  
`N` Feature Maps  
↓  
Output Channels =  
`N`

---

# 🧠 Final Mental Model

Input Channels  
↓  
Each Filter Spans All Input Channels  
↓  
Different Filters Learn Different Weights  
↓  
Each Filter Is Shared Across Space  
↓  
Each Filter Produces One Feature Map  
↓  
Number of Filters = Output Channels

---

# ⭐ Golden Rule

> **A CNN filter spans all input channels, shares its own weights across space, and produces exactly one output feature map.**
```
