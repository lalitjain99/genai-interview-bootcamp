# 💡 Assignment — Residual Connections

> **Focus:** Apply Residual Connection concepts to practical scenarios rather than repeating theory.

---

# 📌 Scenario 1 — Is This Really a Residual Block?

A block receives input:

`x`

and computes:

`F(x)`

The developer defines the output as:

`y = F(x)`

### Question

Is this a Residual Block?

### ✅ Answer

No.

A basic Residual Block requires a shortcut connection:

`y = F(x) + x`

Without the shortcut:

`y = F(x)`

it is just a normal transformation block.

---

# 📌 Scenario 2 — Identity Behavior

Suppose a Residual Block receives:

`x = 20`

After training, the residual branch learns:

`F(x) ≈ 0`

### Question

What will the block output approximately?

### ✅ Answer

Using:

`y = F(x) + x`

we get:

`y ≈ 0 + 20`

So:

`y ≈ 20`

The block behaves approximately like an:

> **Identity Mapping**

This means the existing representation is preserved.

---

# 📌 Scenario 3 — Learning a Correction

Suppose:

`x = 15`

and the residual branch learns:

`F(x) = -2`

### Question

What is the block output?

### ✅ Answer

`y = F(x) + x`

So:

`y = -2 + 15`

`y = 13`

This shows that a residual branch can learn either:

- Positive correction
- Negative correction
- Near-zero correction

It does not have to be small or positive.

---

# 📌 Scenario 4 — Shape Mismatch

You have:

`x shape = [batch, 64]`

and:

`F(x) shape = [batch, 128]`

A developer writes:

`y = F(x) + x`

### Question

What is wrong?

### ✅ Answer

The shapes are incompatible:

`64 ≠ 128`

So direct addition cannot be performed.

We need a projection shortcut:

`y = F(x) + Wₛx`

where:

`Wₛx`

transforms:

`64 → 128`

---

# 📌 Scenario 5 — Identity or Projection Shortcut?

For each case, choose the correct shortcut.

### Case A

`x shape = [batch, 256]`

`F(x) shape = [batch, 256]`

### ✅ Answer

Use an:

> **Identity Shortcut**

Equation:

`y = F(x) + x`

---

### Case B

`x shape = [batch, 128]`

`F(x) shape = [batch, 256]`

### ✅ Answer

Use a:

> **Projection Shortcut**

Equation:

`y = F(x) + Wₛx`

---

# 📌 Scenario 6 — Degradation or Overfitting?

Consider two models.

### Model A

Training Accuracy = 96%  
Validation Accuracy = 75%

### Model B

After increasing depth:

Training Accuracy drops from 94% to 88%

### Question

Which problem is shown in each case?

### ✅ Answer

### Model A

Training is strong but validation is much worse.

This indicates:

> **Overfitting**

Main issue:

Generalization

---

### Model B

Increasing depth makes the training performance itself worse.

This indicates:

> **Degradation**

Main issue:

Optimization

---

# 📌 Scenario 7 — Gradient Path

Suppose:

`y = F(x) + x`

and:

`dF(x)/dx = 0.02`

### Question

What is:

`dy/dx`?

### ✅ Answer

`dy/dx = dF(x)/dx + 1`

So:

`dy/dx = 0.02 + 1`

`dy/dx = 1.02`

The key idea is that the shortcut adds a direct identity derivative path.

---

# 📌 Scenario 8 — Does Residual Mean Skip the Layers?

A developer says:

> "The shortcut means the network can simply ignore the layers inside the block."

### Is this fully correct?

### ✅ Answer

Not exactly.

The residual branch is still computed:

`F(x)`

and then combined with:

`x`

The block output is:

`x + F(x)`

Training decides whether:

`F(x)`

should be:

- Large
- Small
- Near zero

So the shortcut does not mean the layers are literally removed or skipped during computation.

---

# 📌 Scenario 9 — Block Design

You are designing a deep network.

One stage contains:

`128-dimensional representations`

You want three Residual Blocks while keeping the same dimension.

### Question

What kind of shortcut can normally be used?

### ✅ Answer

Since input and output dimensions remain:

`128 → 128`

the blocks can normally use:

> **Identity Shortcuts**

Conceptually:

128  
↓  
Residual Block  
↓  
128  
↓  
Residual Block  
↓  
128  
↓  
Residual Block  
↓  
128

No projection is required if the shapes remain compatible.

---

# 📌 Scenario 10 — Why Not Concatenate?

Suppose:

`x shape = [batch, 64]`

and:

`F(x) shape = [batch, 64]`

Compare:

### Option A

`F(x) + x`

### Option B

`[F(x), x]`

### Question

Do both produce the same representation size?

### ✅ Answer

No.

### Addition

`F(x) + x`

produces:

`[batch, 64]`

### Concatenation

`[F(x), x]`

produces approximately:

`[batch, 128]`

So residual learning normally uses addition because it represents:

Original Representation  
+  
Learned Correction

without automatically doubling the feature dimension.

---

# ✅ True / False

| # | Statement | Answer |
|---|---|---|
| 1 | `x` always means the raw input of the entire network. | ❌ False |
| 2 | `x` means the input to the current Residual Block. | ✅ True |
| 3 | Residual Connections normally use addition. | ✅ True |
| 4 | A projection shortcut is needed every time. | ❌ False |
| 5 | Identity shortcuts usually have no trainable parameters. | ✅ True |
| 6 | Residual Connections guarantee that gradients can never vanish. | ❌ False |
| 7 | `F(x)` can learn a negative correction. | ✅ True |
| 8 | If `F(x) ≈ 0`, the block behaves approximately like identity. | ✅ True |
| 9 | Degradation and overfitting mean the same thing. | ❌ False |
| 10 | Residual Connections mainly help deep-network optimization. | ✅ True |

---

# ⭐ Staff Engineer Challenge

## Scenario

You are reviewing this architecture:

Input  
↓  
Residual Block 1  
`128 → 128`  
↓  
Residual Block 2  
`128 → 128`  
↓  
Residual Block 3  
`128 → 256`  
↓  
Residual Block 4  
`256 → 256`

A developer proposes using a projection shortcut in **every block**.

### Question 1

Is that necessary?

### ✅ Answer

No.

Projection shortcuts are mainly needed when dimensions do not match.

So:

### Block 1

`128 → 128`

Use:

Identity Shortcut

### Block 2

`128 → 128`

Use:

Identity Shortcut

### Block 3

`128 → 256`

Use:

Projection Shortcut

### Block 4

`256 → 256`

Use:

Identity Shortcut

---

### Question 2

Why avoid unnecessary projections?

### ✅ Answer

An identity shortcut is simpler and has no extra learned parameters.

Unnecessary projections add:

- Parameters
- Computation
- Additional transformation on the shortcut path

So we normally preserve the clean identity path whenever dimensions allow it.

---

### Question 3

Suppose Block 4 eventually learns:

`F(x) ≈ 0`

What does that mean?

### ✅ Answer

It means the block has learned that little or no additional transformation is useful.

So:

`y = x + F(x)`

becomes approximately:

`y ≈ x`

The representation from Block 3 passes through almost unchanged.

---

# 🎯 Final Practical Exercise

Complete the following:

### 1.

`Shape(x) = Shape(F(x))`

Use:

________________________

### 2.

`Shape(x) ≠ Shape(F(x))`

Use:

________________________

### 3.

If no useful change is required:

`F(x) ≈ ______`

### 4.

Residual output:

`y = ______ + ______`

### 5.

Residual Connections mainly address:

________________________

---

# ✅ Answers

### 1.

Identity Shortcut

### 2.

Projection Shortcut

### 3.

`0`

### 4.

`F(x) + x`

### 5.

Deep-network optimization / degradation / gradient and information flow

---

# 🧠 Final Mental Model

Input to Block = x  
↓  
Residual Branch learns F(x)  
↓  
Shortcut preserves x  
↓  
Add Both  
↓  
`y = F(x) + x`

If:

`F(x) ≈ 0`

→ Preserve representation

If:

`F(x)` learns useful change

→ Improve representation

If shapes differ:

`y = F(x) + Wₛx`

---

# ⭐ Golden Rule

> **Use the shortcut to preserve an easy information path, and let the residual branch learn only the transformation that actually helps reduce the final loss.**