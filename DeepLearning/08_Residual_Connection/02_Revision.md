# 📝 Revision — Residual Connections

> **Core Idea:** Residual Connections help very deep networks train more easily by adding a shortcut path around one or more layers.

---

# 1️⃣ Why Residual Connections?

As networks become deeper:

More Layers  
↓  
Harder Optimization  
↓  
Training Error Can Increase

This is called the:

> **Degradation Problem**

It is different from overfitting.

---

# 2️⃣ Residual Equation

The basic equation is:

`y = F(x) + x`

where:

- `x` = input to the current residual block
- `F(x)` = transformation learned by the block
- `y` = block output

---

# 3️⃣ What Does `x` Mean?

`x` can be:

- Raw input to the network, or
- Output/representation from previous layers

It simply means:

> **Input to the current residual block**

---

# 4️⃣ What Is Residual Learning?

Suppose the ideal mapping is:

`H(x)`

Residual learning rewrites it as:

`H(x) = F(x) + x`

So:

`F(x) = H(x) - x`

Important:

> We do not know `H(x)` explicitly.

The network learns `F(x)` indirectly through the final loss and backpropagation.

---

# 5️⃣ Main Intuition

Instead of forcing the block to learn the complete transformation:

`x → H(x)`

we allow it to learn:

`x + correction`

So:

Original Representation  
+  
Learned Change  
↓  
Better Representation

---

# 6️⃣ What If No Change Is Needed?

If the existing representation is already useful:

`H(x) ≈ x`

then:

`F(x) ≈ 0`

Therefore:

`y = x + F(x)`

becomes approximately:

`y ≈ x`

So the block can easily behave like an identity mapping.

---

# 7️⃣ Why Does This Help Deep Networks?

Without residual connection:

x  
↓  
Layer  
↓  
Layer  
↓  
Output

With residual connection:

x ────────────────────┐
│                     │
↓                     │
Layers                 │
↓                     │
F(x)                   │
│                     │
└────── Add x ◄────────┘
          ↓
       F(x) + x

The shortcut preserves useful information and makes optimization easier.

---

# 8️⃣ Gradient Flow

For:

`y = F(x) + x`

the derivative is:

`dy/dx = dF(x)/dx + 1`

The `+1` comes from the identity shortcut.

So gradients can flow:

- Through the residual branch
- Through the shortcut path

This reduces the difficulty of training very deep networks.

---

# 9️⃣ Degradation vs Overfitting

### Degradation

More depth  
↓  
Training error becomes worse

Main issue:

> Optimization

### Overfitting

Training performance is good  
↓  
Validation performance is poor

Main issue:

> Generalization

---

# 🔟 Identity Shortcut

If:

`Shape(x) = Shape(F(x))`

we can directly use:

`y = F(x) + x`

This is called an:

> **Identity Shortcut**

It usually has no trainable parameters.

---

# 1️⃣1️⃣ Projection Shortcut

If:

`Shape(x) ≠ Shape(F(x))`

direct addition is not possible.

Then we transform `x`:

`y = F(x) + Wₛx`

where:

`Wₛ`

changes the shortcut representation to the required shape.

This is called a:

> **Projection Shortcut**

---

# 1️⃣2️⃣ Addition vs Concatenation

Residual Connections normally use:

`F(x) + x`

This is **addition**.

It is different from:

`[F(x), x]`

which is concatenation.

---

# 1️⃣3️⃣ Residual Block

A Residual Block contains:

Input x  
↓  
Transformation Layers  
↓  
F(x)  
↓  
Add Shortcut x  
↓  
Output

The number of layers and width of the block are architecture-design choices.

---

# 1️⃣4️⃣ What Does Training Decide?

We decide:

- Number of residual blocks
- Number of layers per block
- Feature dimensions

Training decides:

> How much each block should actually modify its input.

A block may learn:

`F(x) ≈ 0`

or:

`F(x) = significant transformation`

depending on what reduces the final loss.

---

# 1️⃣5️⃣ Residual Connections Do NOT

Residual Connections do not:

- Guarantee zero vanishing gradients
- Guarantee zero exploding gradients
- Prevent overfitting
- Reduce model parameters automatically
- Mean that skipped layers are ignored

They mainly improve:

> **Optimization, information flow, and gradient flow**

---

# ⚡ Quick Comparison

| Plain Network | Residual Network |
|---|---|
| `y = F(x)` | `y = F(x) + x` |
| Learns full mapping | Learns correction |
| Harder identity mapping | Easy identity path |
| Gradient only through layers | Gradient also through shortcut |
| Deep optimization harder | Deep optimization easier |

---

# 🧠 Mental Model

Very Deep Network  
↓  
Optimization Gets Difficult  
↓  
Add Shortcut  
↓  
`y = F(x) + x`  
↓  
Preserve Existing Information  
+  
Learn Useful Correction  
↓  
Better Gradient Flow  
↓  
Easier Deep-Network Training

---

# 🎤 30-Second Interview Answer

> **A Residual Connection adds the input of a block directly to its transformed output, usually as `y = F(x) + x`. This allows the block to learn only the required correction instead of the full mapping. The shortcut also provides a direct path for information and gradients, making very deep networks easier to optimize and helping address the degradation problem. If dimensions differ, a projection shortcut such as `Wₛx` can be used.**

---

# ⭐ Golden Rule

> **Keep what is already useful and learn only what needs to change.**

`y = F(x) + x`