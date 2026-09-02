# ⚡ Cheat Sheet — Convolution Operation

> **Core Idea:** A filter scans local regions of the input, performs multiply + sum, and creates a feature map.

---

# 🎛️ Convolution at One Location

Local Input Region  
×  
Filter / Kernel  
↓  
Element-wise Multiply  
↓  
Sum  
↓  
One Scalar Output

---

# 🧮 Example

Input Region:

1   2   3  
4   5   6  
7   8   9

Filter:

1   0  -1  
1   0  -1  
1   0  -1

Calculation:

`1×1 + 2×0 + 3×(-1)`  
`+ 4×1 + 5×0 + 6×(-1)`  
`+ 7×1 + 8×0 + 9×(-1)`

Result:

`-6`

So:

> One filter position → one scalar

---

# ➡️ Sliding the Filter

After one calculation:

Move Filter  
↓  
Next Local Region  
↓  
Another Scalar  
↓  
Repeat

The filter weights remain the same.

---

# 🔁 Parameter Sharing

Same Filter Weights  
↓  
Used at Every Spatial Location

This is:

> **Parameter Sharing**

What changes:

- Local input
- Output activation

What does NOT change:

- Filter weights

---

# 🗺️ Feature Map

All scalar outputs from one filter are arranged spatially.

So:

`1 filter → 1 feature map`

Remember:

One Position  
→ One Scalar

Whole Input  
→ One Feature Map

---

# 📈 Activation Meaning

### Large Positive Value

Strong response to the learned pattern

### Near Zero

Weak response

### Large Negative Value

Strong opposite interaction with the filter

---

# 🔥 Convolution + ReLU

Typical flow:

Convolution  
↓  
Feature Map  
↓  
ReLU  
↓  
Activated Feature Map

Example:

Before ReLU:

-5   2   7  
 1  -3   4

After ReLU:

0   2   7  
1   0   4

---

# 🧠 Convolution vs Fully Connected

Both perform weighted sums.

### Fully Connected

Weighted sum over many/all inputs

### Convolution

Weighted sum over a small local region

plus:

> **same weights reused across locations**

---

# 🎓 Filters Are Learned

Filter weights are trainable.

Initialize  
↓  
Forward Pass  
↓  
Loss  
↓  
Backpropagation  
↓  
Update Filter Weights

So CNN filters are usually learned automatically.

---

# ⚠️ Convolution vs Cross-Correlation

Strict mathematical convolution:

→ Kernel is flipped

Deep-learning libraries:

→ Kernel usually not flipped

Technically this is:

> **Cross-correlation**

But deep learning convention calls it:

> **Convolution**

---

# ⚡ Quick Recall Table

| Concept | Meaning |
|---|---|
| Filter / Kernel | Small trainable weight matrix |
| Local Region | Input area currently examined |
| One Position | Produces one scalar |
| Sliding | Move filter across input |
| Parameter Sharing | Same weights reused |
| Feature Map | Complete output of one filter |
| High Activation | Strong filter response |
| ReLU | Often applied after convolution |

---

# 🧠 Mental Model

🖼️ Input  
↓  
🔍 Local Region  
↓  
🎛️ Same Filter  
↓  
✖️ Multiply  
↓  
➕ Sum  
↓  
🔢 One Activation  
↓  
➡️ Move  
↓  
Repeat  
↓  
🗺️ Feature Map

---

# 🎤 Interview One-Liner

> **Convolution applies a small learned filter to local input regions, multiplies corresponding values, sums them into one activation, and reuses the same filter across locations to build a feature map.**

---

# ⭐ Remember These 5 Things

1. 🎛️ Filter = trainable weights  
2. 🔍 Operates on a local region  
3. 🧮 Multiply + sum  
4. 🔢 One location → one scalar  
5. 🗺️ One filter across input → one feature map

---

# ⭐ Golden Rule

> **The filter stays the same, the local region changes, and the resulting activations form the feature map.**