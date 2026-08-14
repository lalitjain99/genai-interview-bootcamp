# ⚡ Cheat Sheet — Padding in CNNs

> **Core Idea:** Padding adds extra boundary values around the input so the filter can operate near the edges and so we can control spatial output size.

---

# 🧱 What Is Padding?

Padding means:

> **Adding extra values around the input boundary**

Most commonly:

> **Zero Padding**

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

---

# 📉 Why Padding Is Needed

Without padding:

* Filter must stay fully inside the input
* Border pixels participate in fewer windows
* Output spatial dimensions usually shrink

So padding helps with:

> **Boundary Handling + Spatial-Size Control**

---

# ✅ Valid Padding

`valid`

usually means:

> **No padding**

So:

`P = 0`

Result:

> Output usually shrinks.

---

# 🟰 Same Padding

`same`

aims to follow a same-style spatial-size rule.

For stride `1`:

Input H/W
≈
Output H/W

Example:

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

---

# ⚠️ Same + Stride > 1

Do NOT assume:

> same padding always preserves exact input size.

If:

`stride = 2`

the output is still downsampled.

Example:

`32 × 32`

may become approximately:

`16 × 16`

depending on the exact padding rule.

---

# 📏 Common Padding Values

For stride `1` and odd kernels:

| Kernel  | Common Padding |
| ------- | -------------: |
| `3 × 3` |            `1` |
| `5 × 5` |            `2` |
| `7 × 7` |            `3` |

Quick rule:

`P = (K - 1) / 2`

for odd kernel sizes when preserving H/W with stride 1.

---

# 📐 Quick Example

Input:

`5 × 5`

Kernel:

`3 × 3`

Stride:

`1`

### Padding = 0

Output:

`3 × 3`

### Padding = 1

Output:

`5 × 5`

So:

> **Padding can prevent spatial shrinkage.**

---

# ⚠️ Border Effect

Without padding:

Center pixels
→ participate in many windows

Border pixels
→ participate in fewer windows

Padding allows the filter to operate around the boundary.

So border information gets:

> **more opportunity to influence the output**

---

# 0️⃣ Zero Padding

Most common padding type:

`0`

values added around the input.

Benefits:

* Simple
* Easy to implement
* Helps preserve spatial size
* Handles boundaries

But zeros are:

> **artificial boundary values**

not real image information.

---

# 🧠 Padding Does NOT Add New Information

Padding values are not:

* Real pixels
* Learned features
* Semantic information

Padding only changes:

> **how convolution behaves near the boundary**

---

# 🎛️ Padding Does NOT Change the Kernel

Example:

Kernel:

`3 × 3`

Padding:

`1`

Kernel remains:

`3 × 3`

Padding changes:

> effective input boundary

not:

> kernel dimensions

---

# ➡️ Padding vs Stride

### Padding

Controls:

> boundary extension

### Stride

Controls:

> movement step

Both affect:

`H_out` and `W_out`

but in different ways.

---

# 📦 Padding vs Output Channels

Padding does NOT determine:

`C_out`

Output channels depend on:

> **Number of Filters**

Example:

64 filters
→ 64 output channels

regardless of whether padding is `0` or `1`.

---

# 🏗️ Padding in Deep CNNs

Without padding:

`100 → 98 → 96 → 94 → ...`

for repeated `3 × 3`, stride-1 convolutions.

With:

`padding = 1`

spatial size can stay:

`100 → 100 → 100 → ...`

So padding helps prevent:

> **rapid spatial shrinkage**

---

# 🔗 Padding and Residual Connections

Residual addition:

`y = F(x) + x`

usually requires compatible shapes.

For a stride-1 `3 × 3` convolution:

`padding = 1`

can preserve:

`H × W`

which helps maintain shape compatibility.

---

# 🧱 Other Padding Types

Besides zero padding:

* Reflection padding
* Replication padding
* Circular padding

For standard interviews:

> **Zero padding is the most important one to know.**

---

# ⚡ Quick Recall Table

| Concept         | Quick Meaning                    |
| --------------- | -------------------------------- |
| Padding         | Extra boundary values            |
| Zero Padding    | Add zeros                        |
| `valid`         | Usually no padding               |
| `same`          | Same-style output sizing         |
| `3×3, S=1, P=1` | Preserves H/W                    |
| `5×5, S=1, P=2` | Preserves H/W                    |
| Padding         | Affects spatial dimensions       |
| Filters         | Determine output channels        |
| Border Pixels   | Get more convolution opportunity |
| Padding Values  | Artificial, not learned          |

---

# 🧠 Shape Mental Model

Input
↓
Add Padding
↓
Effective Input Becomes Larger
↓
More Valid Filter Positions
↓
Better Boundary Handling
+
Better Control of Output Size

---

# 🎤 Interview One-Liner

> **Padding adds extra values, usually zeros, around the input boundary so the convolution filter can operate near the edges and so the network can control spatial output size. For example, a 3×3 kernel with stride 1 and padding 1 can preserve input height and width.**

---

# ⭐ Remember These 6 Things

1. 🧱 Padding = extra boundary
2. 0️⃣ Zero padding is most common
3. ✅ `valid` = usually no padding
4. 🟰 `same` = same-style spatial sizing
5. 📐 `3×3 + S=1 + P=1` preserves H/W
6. 📦 Padding affects spatial size, not channel count

---

# ⭐ Golden Rule

> **Padding gives the filter room to work at the boundaries and gives us control over how much spatial resolution is preserved.**
