```markdown id="cnn-convolution-operation-revision"
# 📝 Revision — Convolution Operation

> **Core Idea:** Convolution applies the same small filter across local regions of an input, producing one output value at each position. All these values together form a feature map.

---

# 🖼️ 1. What Convolution Does

At one location:

Local Input Region  
×  
Filter  
↓  
Element-wise Multiplication  
↓  
Sum  
↓  
One Output Value

Then the filter moves to the next location.

---

# 🎛️ 2. Filter / Kernel

A filter is a small matrix of trainable weights.

Example:

1   0  -1  
1   0  -1  
1   0  -1

A `3 × 3` filter has:

`9 weights`

The same filter is reused across the input.

---

# 🔍 3. Local Region

A convolution does not initially look at the entire image.

Instead, the filter examines a small local region.

Example:

Image Region:

1   2   3  
4   5   6  
7   8   9

Filter:

1   0  -1  
1   0  -1  
1   0  -1

---

# 🧮 4. Convolution Calculation

Multiply corresponding values:

`1×1 + 2×0 + 3×(-1)`

`+ 4×1 + 5×0 + 6×(-1)`

`+ 7×1 + 8×0 + 9×(-1)`

Result:

`1 - 3 + 4 - 6 + 7 - 9`

= `-6`

So:

> **One filter position → one scalar output**

---

# ➡️ 5. Sliding the Filter

After calculating one output:

Filter moves  
↓  
Next local region  
↓  
Another output value

This continues across the image.

The filter weights do NOT change while moving.

---

# 🗺️ 6. Feature Map

All output values from the same filter are arranged spatially.

Example:

1   3   2  
0   4   5  
1   2   3

This complete grid is:

> **One Feature Map**

Remember:

`1 filter → 1 feature map`

---

# 🔁 7. Parameter Sharing

The same filter weights are used at every spatial location.

So:

Same Filter  
+  
Different Local Regions  
↓  
Different Output Activations

This is called:

> **Parameter Sharing**

---

# 📈 8. High vs Low Activation

If a local region strongly matches what the filter responds to:

→ large activation

If it does not:

→ small activation

A negative activation may indicate an opposite response.

So feature-map values tell us:

- 📈 response strength
- 📍 spatial location

---

# 🔥 9. Activation After Convolution

Convolution is often followed by an activation such as ReLU.

Example:

Before ReLU:

-6   2   5  
 1  -3   4

After ReLU:

0   2   5  
1   0   4

because:

`ReLU(x) = max(0, x)`

---

# 🧠 10. Connection to a Normal Neuron

A normal neuron performs:

`w₁x₁ + w₂x₂ + ...`

Convolution also performs a weighted sum.

The difference is:

### Fully Connected

Weighted sum over many/all inputs

### Convolution

Weighted sum over a small local region

So convolution is still built from familiar neural-network mathematics.

---

# 🎓 11. How Filters Are Learned

Filters are trainable parameters.

Training flow:

Initialize Filter Weights  
↓  
Forward Pass  
↓  
Loss  
↓  
Backpropagation  
↓  
Update Filter Weights

Over time, filters learn patterns useful for the task.

---

# ⚠️ 12. Convolution vs Feature Map

Do not confuse them.

### Convolution

The local operation:

Multiply + Sum

### Feature Map

The complete spatial output created after applying the same filter across the input.

So:

> **Convolution is the operation; feature map is the resulting grid.**

---

# 🎯 13. Important Terminology Detail

In strict mathematics, convolution normally flips the kernel.

Deep-learning libraries typically do not flip it.

That operation is technically:

> **Cross-correlation**

But in deep learning, it is conventionally called:

> **Convolution**

---

# 🧠 Mental Model

🖼️ Input  
↓  
🔍 Pick Local Region  
↓  
🎛️ Apply Filter  
↓  
✖️ Multiply  
↓  
➕ Sum  
↓  
🔢 One Activation  
↓  
➡️ Move Filter  
↓  
Repeat  
↓  
🗺️ Feature Map

---

# ⚡ Quick Recall

| Concept | Meaning |
|---|---|
| Filter / Kernel | Small matrix of trainable weights |
| Local Region | Portion of input currently inspected |
| One Position | Produces one scalar |
| Sliding | Move filter across spatial locations |
| Parameter Sharing | Same filter weights reused |
| Feature Map | Complete output of one filter |
| High Activation | Strong filter response |
| ReLU | Often applied after convolution |

---

# 🎤 30-Second Interview Answer

> **Convolution applies a small learned filter to local regions of an input. At each location, corresponding input and filter values are multiplied and summed to produce one output activation. The same filter is reused across spatial locations, and all resulting activations form one feature map.**

---

# ⭐ Remember These 5 Things

1. 🎛️ Filter = small trainable weight matrix
2. 🔍 Filter works on a local region
3. 🧮 Multiply corresponding values and sum
4. 🔢 One location produces one scalar
5. 🗺️ Same filter across the input produces one feature map

---

# ⭐ Golden Rule

> **One local region gives one value; sliding the same filter across all regions builds the feature map.**
```
