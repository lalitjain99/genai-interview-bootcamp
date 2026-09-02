# 🎤 Interview — Multiple Filters and Channels in CNNs

> **Focus:** Understand how standard Conv2D handles multiple input channels, how one filter combines channel-wise information, how output channels are formed, and how parameter count is calculated.

---

# 🟢 Beginner

## 1️⃣ If an RGB image has 3 channels, what is the depth of one convolutional filter?

For an RGB image:

`C_in = 3`

If the spatial kernel size is:

`3 × 3`

then one full filter has shape:

`3 × 3 × 3`

So:

> **Filter depth = number of input channels**

---

## 2️⃣ Does one filter operate on only one input channel?

No.

In a standard convolution:

> **One filter spans all input channels**

For RGB:

Red contribution
+
Green contribution
+
Blue contribution
↓
One scalar output

---

## 3️⃣ How many output feature maps does one full filter produce?

One full filter produces:

> **One output feature map**

So:

`1 filter → 1 feature map`

This remains true even if the input has many channels.

---

## 4️⃣ What determines the number of output channels?

The number of:

> **filters**

For example:

`64 filters → 64 output channels`

So:

`C_out = Number of Filters`

---

# 🟡 Intermediate

## 5️⃣ How are multiple input channels combined at one spatial location?

Each input channel interacts with its corresponding filter slice.

For example:

Channel 1
→ contribution 1

Channel 2
→ contribution 2

...

Channel `C_in`
→ contribution `C_in`

Then all contributions are summed:

`Output = Contribution₁ + Contribution₂ + ... + Contribution_Cin + Bias`

The result is:

> **one scalar**

---

## 6️⃣ Why does one filter not produce one final feature map per input channel?

Because the per-channel calculations are only intermediate contributions.

They are summed together before producing the final activation.

So:

Multiple Channel Contributions
↓
Sum
↓
One Scalar Per Location
↓
One Feature Map

Therefore:

> **One complete filter produces one final output feature map.**

---

## 7️⃣ What is the difference between `C_in` and `C_out`?

### `C_in`

Number of channels entering the convolutional layer.

### `C_out`

Number of filters in the layer.

Therefore:

> `C_out = number of output feature maps`

Example:

Input:

`32 × 32 × 3`

Filters:

`64`

Then:

`C_in = 3`

`C_out = 64`

---

## 8️⃣ If a layer receives 64 input channels and uses a 3×3 kernel, what is the shape of one filter?

One filter must span all:

`64 input channels`

So its shape is:

`3 × 3 × 64`

It still produces:

> **one output feature map**

---

## 9️⃣ What happens to output channels when a layer has 128 filters?

It produces:

`128 feature maps`

Therefore:

> **Output channels = 128**

The input could have 3, 32, 64, or another number of channels.

---

# 🔴 Advanced

## 🔟 What is the parameter-count formula for a standard convolutional layer?

Weight count:

`K_h × K_w × C_in × C_out`

If one bias is used for each output filter:

`Total Parameters = K_h × K_w × C_in × C_out + C_out`

---

## 1️⃣1️⃣ Calculate the parameters for a 3×3 convolution with 3 input channels and 64 filters.

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

## 1️⃣2️⃣ Calculate the parameters for a deeper convolution with 64 input channels and 128 filters using a 3×3 kernel.

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

## 1️⃣3️⃣ How does parameter sharing work with multi-channel convolution?

Each full filter has shape:

`K_h × K_w × C_in`

That entire filter is reused across:

> **all spatial locations**

Different filters have different weights.

So:

Filter 1
→ same full weights across space

Filter 2
→ its own full weights across space

Parameter sharing is:

> **spatial sharing of the complete filter**

---

## 1️⃣4️⃣ Are the same kernel weights necessarily reused across different input channels?

No.

Within one standard filter, different input channels can have different kernel slices.

For example:

Channel 1 slice
≠
Channel 2 slice

What is shared is:

> **the full filter across spatial positions**

not the same 2D kernel slice across every channel.

---

## 1️⃣5️⃣ Why do deeper convolutional layers become expensive as channel counts increase?

Because parameter count is proportional to:

`C_in × C_out`

So increasing either:

* input channels
* output filters

increases:

* parameter count
* computation
* memory requirements

For standard convolution:

> **Channel width has a direct computational cost.**

---

# ⭐ Staff Engineer Challenge

## Scenario

You have this convolutional layer:

Input:

`56 × 56 × 64`

Configuration:

* Kernel = `3 × 3`
* Padding = `1`
* Stride = `1`
* Filters = `128`

A developer says:

> "Since the input has 64 channels, each of the 128 filters should produce 64 separate feature maps. That means the output should have 8,192 channels."

---

### Question 1

Is the developer correct?

### ✅ Answer

No.

One full filter spans all:

`64 input channels`

and combines all of their contributions.

At one spatial location:

64 channel contributions
↓
Sum
↓
One scalar

Across the image:

> **one feature map**

So:

`128 filters → 128 feature maps`

not:

`128 × 64`

---

### Question 2

What is the correct output shape?

### ✅ Answer

Because:

* `3 × 3`
* padding = `1`
* stride = `1`

the spatial dimensions stay:

`56 × 56`

There are:

`128 filters`

So:

> **Output = 56 × 56 × 128**

---

### Question 3

What is the shape of one filter?

### ✅ Answer

Each filter spans all 64 input channels.

So:

> **One filter = 3 × 3 × 64**

---

### Question 4

How many weights are in one filter?

### ✅ Answer

`3 × 3 × 64`

`= 576`

So:

> **576 weights per filter**

before bias.

---

### Question 5

How many trainable parameters are in the whole layer?

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

### Question 6

Why is it useful for one filter to span all 64 input channels instead of looking at only one?

### ✅ Answer

Because deeper input channels represent different learned features.

A filter can combine information across channels such as:

Feature A
+
Feature B
+
Feature C
↓
New Feature

This allows deeper layers to learn:

> **relationships and combinations among previously learned representations**

---

# 🎯 30-Second Interview Answer

> **In a standard Conv2D layer, each filter spans all input channels. At each spatial location, it computes a contribution from every input channel and sums them into one scalar. Scanning across the input gives one feature map, so one filter produces one output channel. Therefore, `C_out` equals the number of filters. The weight count is `K_h × K_w × C_in × C_out`, plus one bias per output channel if biases are used.**

---

# 🧠 Interview Mental Model

Input:

`H × W × C_in`

↓

One Filter:

`K_h × K_w × C_in`

↓

At One Position:

All Channel Contributions
↓
Sum
↓
One Scalar

↓

Across Space:

One Feature Map

↓

`C_out` Filters

↓

`C_out` Feature Maps

↓

Output:

`H_out × W_out × C_out`

---

# ⭐ Golden Rule

> **One standard convolution filter spans every input channel, sums their contributions into one response, and therefore creates exactly one output feature map.**
