# 💡 Assignment — Multiple Filters and Channels in CNNs

> **Focus:** Practice how multi-channel convolution works, how one filter spans all input channels, how output channels are formed, and how to calculate parameter counts.

---

# 🧩 Scenario 1 — RGB Input

Input:

`32 × 32 × 3`

Kernel size:

`3 × 3`

### Question

What is the shape of one standard convolution filter?

### ✅ Answer

The filter must span all input channels.

So:

`3 × 3 × 3`

Therefore:

> **One filter shape = 3 × 3 × 3**

---

# 🌈 Scenario 2 — More Input Channels

Input:

`64 × 64 × 32`

Kernel size:

`3 × 3`

### Question

What is the shape of one filter?

### ✅ Answer

The input has:

`C_in = 32`

So one filter has shape:

`3 × 3 × 32`

Therefore:

> **Filter depth = input channels**

---

# ➕ Scenario 3 — One Spatial Location

A filter spans:

`3 input channels`

At one spatial position, the three channel-wise contributions are:

`4`

`-2`

`5`

Bias:

`1`

### Question

What is the final output at this location?

### ✅ Answer

Sum the channel contributions and bias:

`4 + (-2) + 5 + 1`

`= 8`

Therefore:

> **One scalar output = 8**

---

# ⚠️ Scenario 4 — One Filter, Multiple Channels

A developer says:

> "The input has 3 channels, so one filter should produce 3 final feature maps."

### Is this correct?

### ✅ Answer

No.

A standard filter spans all 3 input channels.

The channel-wise contributions are:

> **summed together**

So:

One Filter
↓
One Scalar Per Location
↓
One Feature Map

Therefore:

> **One full filter produces one final feature map**

---

# 🎛️ Scenario 5 — Multiple Filters

Input:

`32 × 32 × 3`

Conv layer:

`64 filters`

### Question 1

How many output feature maps are produced?

### ✅ Answer

`64`

because:

`1 filter → 1 feature map`

So:

> **64 filters → 64 feature maps**

---

### Question 2

How many output channels are produced?

### ✅ Answer

> **64 output channels**

because:

`C_out = Number of Filters`

---

# 🔄 Scenario 6 — Next Layer Input

Layer 1 produces:

`32 × 32 × 64`

### Question

How many input channels does Layer 2 receive?

### ✅ Answer

Layer 2 receives:

> **64 input channels**

because the previous layer's output channels become the next layer's input channels.

So:

`Previous C_out = Next C_in`

---

# 🧮 Scenario 7 — Parameter Count for One Filter

Input channels:

`3`

Kernel:

`3 × 3`

### Question

How many weights are in one filter?

### ✅ Answer

`3 × 3 × 3`

`= 27`

So:

> **One filter has 27 weights**

If one bias is used:

`27 + 1 = 28 parameters`

---

# 🧮 Scenario 8 — Parameter Count for Whole Layer

Input channels:

`3`

Kernel:

`3 × 3`

Filters:

`64`

### Question

Calculate the total trainable parameters, including one bias per filter.

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

# 🧮 Scenario 9 — Deeper Layer Parameter Count

Input:

`32 × 32 × 64`

Conv:

* Kernel = `3 × 3`
* Filters = `128`

### Question

Calculate the number of parameters including biases.

### ✅ Answer

Weights:

`3 × 3 × 64 × 128`

`= 73,728`

Biases:

`128`

Total:

`73,728 + 128`

`= 73,856`

Therefore:

> **Total parameters = 73,856**

---

# 📦 Scenario 10 — Complete Output Shape

Input:

`32 × 32 × 3`

Conv:

* Kernel = `3 × 3`
* Padding = `1`
* Stride = `1`
* Filters = `64`

### Question

What is the full output shape?

### ✅ Answer

Spatial dimensions are preserved:

`32 × 32`

Number of output channels:

`64`

Therefore:

> **Output = 32 × 32 × 64**

---

# 🧠 Scenario 11 — Input Channels vs Output Channels

Input:

`28 × 28 × 16`

Filters:

`128`

A developer says:

> "Since there are 16 input channels, there can only be 16 output channels."

### Is this correct?

### ✅ Answer

No.

Input channels and output channels are different architecture dimensions.

Here:

`C_in = 16`

`C_out = 128`

because:

> **Number of filters determines output channels**

---

# 🔁 Scenario 12 — Parameter Sharing

A filter has shape:

`3 × 3 × 64`

A developer says:

> "Because parameter sharing is used, every input channel must use exactly the same 3×3 weights."

### Is this correct?

### ✅ Answer

No.

Different channel slices inside one filter can have different weights.

What is shared is:

> **the complete `3 × 3 × 64` filter across spatial locations**

So:

Parameter Sharing
→ Across H and W

Not:

→ Forced identical weights across channels

---

# 🏗️ Scenario 13 — Deeper Feature Combination

Suppose a deeper CNN layer receives:

`64 feature-map channels`

One filter in this layer spans all:

`64 channels`

### Question

Why is this useful?

### ✅ Answer

Because the filter can combine information from many previously learned features.

For example:

Edge-like response
+
Texture response
+
Shape response
↓
New higher-level feature

This enables:

> **hierarchical feature learning**

---

# ⚖️ Scenario 14 — Cost of More Channels

Suppose:

Kernel size stays the same.

You increase:

`C_out: 64 → 128`

### Question

What happens to parameter count?

### ✅ Answer

It approximately doubles.

Because:

`Parameters ∝ C_in × C_out`

If all other values stay fixed:

doubling `C_out`

→ doubles the weight count.

---

# ✅ True / False

| #  | Statement                                                              | Answer  |
| -- | ---------------------------------------------------------------------- | ------- |
| 1  | A standard filter spans all input channels.                            | ✅ True  |
| 2  | One filter produces one final feature map per input channel.           | ❌ False |
| 3  | Channel-wise contributions are summed into one scalar.                 | ✅ True  |
| 4  | Number of filters determines output channels.                          | ✅ True  |
| 5  | `C_in` must always equal `C_out`.                                      | ❌ False |
| 6  | One `3×3` filter on RGB input has 27 weights.                          | ✅ True  |
| 7  | Parameter count includes both `C_in` and `C_out`.                      | ✅ True  |
| 8  | Parameter sharing means all input-channel kernel slices are identical. | ❌ False |
| 9  | Previous layer output channels become next layer input channels.       | ✅ True  |
| 10 | Increasing channel count can increase computation and memory cost.     | ✅ True  |

---

# ⭐ Staff Engineer Challenge

## Scenario

You have:

Input:

`56 × 56 × 64`

Convolution:

* Kernel = `3 × 3`
* Padding = `1`
* Stride = `1`
* Filters = `128`

A developer proposes this interpretation:

> "There are 64 input channels and 128 filters, so the convolution creates `64 × 128 = 8,192` feature maps."

---

### Question 1

What is wrong with this reasoning?

### ✅ Answer

It treats each channel-wise calculation as a separate final feature map.

But in standard convolution:

One Filter
↓
Spans All 64 Input Channels
↓
64 Channel Contributions
↓
Summed Together
↓
One Scalar Per Location
↓
One Feature Map

So:

> **128 filters produce 128 feature maps, not 8,192**

---

### Question 2

What is the shape of one filter?

### ✅ Answer

Kernel:

`3 × 3`

Input channels:

`64`

So:

> **One filter = 3 × 3 × 64**

---

### Question 3

How many weights does one filter contain?

### ✅ Answer

`3 × 3 × 64`

`= 576`

Therefore:

> **576 weights per filter**

---

### Question 4

How many total weights are in the layer?

### ✅ Answer

`576 × 128`

`= 73,728`

So:

> **Total filter weights = 73,728**

---

### Question 5

What is the total parameter count if biases are used?

### ✅ Answer

Biases:

`128`

Total:

`73,728 + 128`

`= 73,856`

Therefore:

> **Total parameters = 73,856**

---

### Question 6

What is the output shape?

### ✅ Answer

Because:

* Kernel = `3 × 3`
* Padding = `1`
* Stride = `1`

the spatial dimensions remain:

`56 × 56`

Number of filters:

`128`

So:

> **Output = 56 × 56 × 128**

---

# 🎯 Final Exercise

Complete the flow:

Input:

`H × W × C_in`

↓

One Filter:

`__________`

↓

At One Spatial Location:

Contributions from `__________` are summed

↓

Produces:

`__________`

↓

Across Space:

`__________`

↓

Using `C_out` Filters:

`__________`

↓

Final Output:

`__________`

### ✅ Answer

Input:

`H × W × C_in`

↓

One Filter:

`K_h × K_w × C_in`

↓

At One Spatial Location:

Contributions from **all input channels** are summed

↓

Produces:

**One Scalar**

↓

Across Space:

**One Feature Map**

↓

Using `C_out` Filters:

**C_out Feature Maps**

↓

Final Output:

`H_out × W_out × C_out`

---

# 🧠 Final Mental Model

Multiple Input Channels
↓
One Filter Spans All Channels
↓
Each Channel Contributes
↓
Contributions Are Summed
↓
One Scalar
↓
Repeat Across Space
↓
One Feature Map
↓
Multiple Filters
↓
Multiple Output Channels

---

# ⭐ Golden Rule

> **One standard convolution filter spans every input channel, combines them into one response, and produces exactly one feature map; the number of filters determines the number of output channels.**
