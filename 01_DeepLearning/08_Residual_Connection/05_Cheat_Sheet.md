# ⚡ Cheat Sheet — Residual Connections

> **Core Idea:** Keep the existing representation and learn only the useful correction.

---

# 🔹 Basic Equation

`y = F(x) + x`

where:

- `x` = input to current residual block
- `F(x)` = learned transformation
- `y` = block output

---

# 🔹 Why Residual Connections?

Very deep plain networks can suffer from:

> **Degradation Problem**

Meaning:

More Depth  
↓  
Harder Optimization  
↓  
Worse Training Performance

Residual Connections make deep networks easier to optimize.

---

# 🔹 Residual Learning

Conceptually:

`H(x) = F(x) + x`

So:

`F(x) = H(x) - x`

Important:

> `H(x)` is not a known intermediate target.

The network learns `F(x)` through final loss and backpropagation.

---

# 🔹 Main Intuition

Instead of:

Learn complete new representation

Residual block learns:

Existing Representation  
+  
Useful Correction

So:

`x + F(x)`

---

# 🔹 Identity Case

If no major change is required:

`F(x) ≈ 0`

Then:

`y ≈ x`

So the block can easily preserve the existing representation.

---

# 🔹 Gradient Flow

For:

`y = F(x) + x`

Derivative:

`dy/dx = dF(x)/dx + 1`

The `+1` comes from the identity shortcut.

This gives gradients an additional direct path.

---

# 🔹 Identity Shortcut

When:

`Shape(x) = Shape(F(x))`

Use:

`y = F(x) + x`

Properties:

- No extra transformation
- Usually no trainable parameters

---

# 🔹 Projection Shortcut

When:

`Shape(x) ≠ Shape(F(x))`

Use:

`y = F(x) + Wₛx`

`Wₛx` transforms the shortcut to a compatible shape.

---

# 🔹 Example

Input:

`x → 64 features`

Residual output:

`F(x) → 128 features`

Direct addition:

❌ Not possible

Use:

`64 → Wₛ → 128`

Then:

`y = F(x) + Wₛx`

---

# 🔹 Degradation vs Overfitting

| Degradation | Overfitting |
|---|---|
| Training performance worsens | Training good, validation poor |
| Optimization problem | Generalization problem |
| Residual Connections help | Regularization helps |

---

# 🔹 Addition vs Concatenation

Residual Connection:

`F(x) + x`

✅ Addition

Not:

`[F(x), x]`

❌ Concatenation

---

# 🔹 Residual Block

Input x  
│
├───────────────┐
│               │
↓               │
Layers          │
↓               │
F(x)            │
│               │
└──── Add x ◄───┘
        ↓
      Output

---

# 🔹 What Do We Decide vs Training?

### Architecture Designer Decides

- Number of blocks
- Layers per block
- Feature dimensions
- Identity vs projection shortcut

### Training Decides

- How much each `F(x)` should modify its input

---

# 🔹 Residual Connections Do NOT Guarantee

- No vanishing gradients
- No exploding gradients
- No overfitting
- Fewer parameters

Their main benefit is:

> **Better optimization, information flow, and gradient flow**

---

# ⚡ Quick Comparison

| Plain Block | Residual Block |
|---|---|
| `y = F(x)` | `y = F(x) + x` |
| Learns full transformation | Learns correction |
| No shortcut | Shortcut path |
| Harder identity mapping | Easy identity mapping |
| Harder deep optimization | Easier deep optimization |

---

# 🧠 Remember These 5 Points

1. Residual equation → `y = F(x) + x`
2. `x` = input to current block
3. `F(x)` = learned correction
4. Same shape → Identity Shortcut
5. Different shape → Projection Shortcut

---

# 🎤 Interview One-Liner

> **A Residual Connection adds a block's input to its learned transformation, `y = F(x) + x`, providing an easy identity path and better gradient flow, which makes very deep networks easier to optimize.**

---

# ⭐ Golden Rule

> **Preserve what is already useful and learn only what needs to change.**