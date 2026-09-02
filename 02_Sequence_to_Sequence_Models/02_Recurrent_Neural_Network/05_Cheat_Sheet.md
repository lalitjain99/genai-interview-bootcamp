# ⚡ Cheat Sheet — Recurrent Neural Network

> **Goal:** Fast revision of the most important RNN concepts, equations, dimensions, strengths, limitations, and interview traps.

---

# 🌟 1. RNN in One Line

> **RNN = Current Input + Previous Hidden State → New Hidden State**

Conceptually:

```text
What do I see NOW?
+
What do I remember from BEFORE?
↓
Update my understanding
↓
Carry it FORWARD
```

---

# 🧠 2. Why RNN?

Sequence data has:

* order
* context
* temporal dependency

Example:

```text
I → love → deep → learning
```

The meaning at the current position can depend on:

> previous sequence elements.

A feed-forward network does not naturally maintain:

> persistent sequence context.

RNN introduces:

> **hidden state**

---

# 🔄 3. Core RNN Flow

```text
x1 → RNN → h1
             ↓
x2 → RNN → h2
             ↓
x3 → RNN → h3
```

At time step `t`:

```text
x_t
+
h_{t-1}
↓
RNN Cell
↓
h_t
```

---

# 🧠 4. Hidden State

`h_t` is:

> a learned fixed-dimensional representation of sequence context up to time `t`.

Important:

```text
Hidden State
≠
Exact Sequence Memory
```

Better mental model:

```text
Past Context
↓
Learned Compression
↓
h_t
```

---

# 📐 5. Vanilla RNN Equation

Pre-activation:

`a_t = W_xh x_t + W_hh h_{t-1} + b_h`

Hidden state:

# `h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h)`

Optional output:

`y_t = W_hy h_t + b_y`

---

# 🧠 6. Equation Meaning

### `x_t`

Current input.

### `h_{t-1}`

Previous hidden state / context.

### `W_xh`

Input-to-hidden transformation.

### `W_hh`

Hidden-to-hidden recurrent transformation.

### `b_h`

Bias.

### `tanh`

Nonlinearity producing the new hidden state.

---

# ⚡ 7. Equation Intuition

```text
Current Input Contribution
+
Previous Context Contribution
+
Bias
↓
tanh
↓
New Hidden State
```

Remember:

> **new context = current information + previous context**

---

# 🔄 8. Why Is It Called Recurrent?

Because:

> the hidden state from one time step becomes input to the next recurrent computation.

```text
h1 → h2 → h3 → h4
```

---

# 🔓 9. Unrolled RNN

Compact view:

```text
       ┌───────────┐
       │           │
x_t → RNN → h_t ───┘
```

Unrolled:

```text
h0 → [RNN] → h1 → [RNN] → h2 → [RNN] → h3
      ↑             ↑             ↑
     x1            x2            x3
```

Important:

> each `[RNN]` is the same recurrent cell.

---

# 🔁 10. Parameter Sharing Across Time

The same:

* `W_xh`
* `W_hh`
* `b_h`

are reused at every time step.

```text
t1 → SAME weights
t2 → SAME weights
t3 → SAME weights
```

Remember:

```text
Hidden State
→ changes across time

Weights
→ shared across time
```

---

# 🔗 11. CNN vs RNN Parameter Sharing

```text
CNN
→ Same kernel across SPACE

RNN
→ Same recurrent weights across TIME
```

This is one of the most important conceptual bridges.

---

# 📐 12. Matrix Shapes

If:

`x_t ∈ R^D`

and:

`h_t ∈ R^H`

then:

```text
W_xh → H × D
W_hh → H × H
b_h  → H
```

Optional output for `O` classes:

```text
W_hy → O × H
b_y  → O
```

---

# 🧮 13. Recurrent Parameter Count

Ignoring output layer:

`HD + H² + H`

Where:

* `D` = input dimension
* `H` = hidden dimension

---

# 🧮 14. Example

If:

```text
D = 300
H = 128
```

then:

`W_xh = 128×300 = 38,400`

`W_hh = 128×128 = 16,384`

`b_h = 128`

Total:

# `54,912`

---

# ⭐ 15. Sequence Length vs Parameters

Suppose:

```text
T = 10
T = 100
T = 1000
```

Parameter count remains:

> the same.

Why?

Because:

> recurrent weights are shared across time.

But:

```text
Sequence Length ↑
Compute ↑
```

So:

```text
Parameters
→ independent of T

Computation
→ grows with T
```

---

# 🧠 16. Initial Hidden State

At the first time step:

`h0`

is commonly:

`0`

Conceptually:

> no previous context yet.

Important:

```text
Zero h0
≠
Zero trainable weights
```

Zero hidden-state initialization is normal.

---

# 📤 17. Hidden State vs Output

Hidden state:

`h_t`

is internal sequence context.

Output:

`y_t`

is task-specific prediction.

```text
h_t
├──→ Next Time Step
└──→ Output Layer → y_t
```

Remember:

```text
Hidden State
≠
Task Output
```

---

# 🎯 18. Many-to-One

Sequence:

```text
This → movie → was → excellent
```

Output:

```text
Positive
```

Structure:

```text
x1 → h1
x2 → h2
x3 → h3
x4 → h4
         ↓
      Output
```

Typical use:

> sentiment classification.

---

# 🏷️ 19. Many-to-Many

Example:

```text
John   lives   in   London
 ↓       ↓      ↓      ↓
PER      O      O     LOC
```

Each hidden state produces an output.

Typical use:

* NER
* POS tagging
* sequence labeling

---

# 🔗 20. Recursive Dependency

`h_t` directly depends on:

* `x_t`
* `h_{t-1}`

But:

`h_{t-1}`

already depends on earlier inputs.

So:

```text
x1
↓
h1
↓
h2
↓
h3
...
↓
h_t
```

Therefore:

> `h_t` can be influenced by all earlier sequence elements.

---

# ⚠️ 21. Influence ≠ Perfect Memory

Do not say:

> “`h_t` stores all previous inputs.”

Better:

> “`h_t` is influenced by previous inputs through recurrence.”

Why?

Because hidden state is:

> compressed and imperfect.

---

# 🧠 22. Fixed-Size Hidden-State Bottleneck

Suppose:

```text
H = 128
```

Whether sequence length is:

```text
5
50
500
```

hidden state still has:

> `128` dimensions.

So:

```text
More History
↓
Same-Sized Representation
```

This can make long-range information difficult to preserve.

---

# ⏳ 23. Long-Term Dependency Problem

Information from:

`x1`

may need to influence:

`x1000`.

It must survive:

```text
h1 → h2 → h3 → ... → h1000
```

Repeated transformations can weaken earlier information.

This is a major vanilla RNN limitation.

---

# 🔄 24. Training an RNN

Forward:

```text
Sequence
↓
Hidden States
↓
Prediction
↓
Loss
```

Backward:

```text
Loss
↓
h_T
↓
h_{T-1}
↓
...
↓
h_1
```

This is:

> **Backpropagation Through Time — BPTT**

---

# 🧠 25. Shared Parameter Gradient

`W_hh` is reused at many time steps.

Therefore during backward:

```text
Gradient Contribution at t1
+
Gradient Contribution at t2
+
Gradient Contribution at t3
+
...
↓
One dL/dW_hh
```

Same principle as CNN shared-filter gradients.

---

# 🚨 26. Vanishing Gradient

During BPTT, gradients are repeatedly multiplied through recurrent transformations.

If these products repeatedly reduce magnitude:

```text
Gradient
↓
smaller
↓
smaller
↓
≈ 0
```

then:

> early time steps receive almost no useful learning signal.

This is:

> **Vanishing Gradient**

---

# 💥 27. Exploding Gradient

If repeated transformations amplify gradients:

```text
Gradient
↓
larger
↓
larger
↓
huge
```

training can become unstable.

This is:

> **Exploding Gradient**

---

# 🧠 28. Why Vanilla RNN Struggles With Long Dependencies

Two related reasons:

```text
Forward
→ early information can fade through repeated state updates

Backward
→ gradients can vanish/explode across many steps
```

This motivates:

> LSTM and GRU.

---

# ⚡ 29. Sequential Computation

To calculate:

`h3`

we first need:

`h2`.

To calculate:

`h2`

we first need:

`h1`.

So:

```text
h1 → h2 → h3 → h4
```

must be computed sequentially.

This limits:

> time-step parallelism.

---

# 🧠 30. Batch vs Time Parallelism

```text
Across Batch
→ can be parallel

Across Time
→ sequential dependency
```

This is one reason Transformers are easier to parallelize during training.

---

# 🌊 31. Streaming Strength

RNN naturally supports incremental processing.

If we already have:

`h_t`

and receive:

`x_{t+1}`

we compute:

`h_{t+1} = f(x_{t+1}, h_t)`

No need to replay:

`x1 ... x_t`

every time.

Useful for:

* sensors
* streaming events
* live signals
* transaction sequences

---

# 🧠 32. Model Parameters vs Sequence State

Very important for production.

### Model Parameters

```text
W_xh
W_hh
b_h
```

Shared globally.

### Hidden State

```text
h_t
```

belongs to:

> an individual sequence/entity.

Example:

```text
Same RNN Weights

User A → h_A
User B → h_B
User C → h_C
```

---

# ⚠️ 33. Hidden-State Leakage

If two independent sequences accidentally share the same hidden state:

```text
Sequence A
↓
h_A
↓
Sequence B
```

Sequence B may inherit context from A.

So independent sequences require:

> separate/reset hidden states.

---

# ⏱️ 34. Standard Forward RNN Is Causal

At time `t`, a forward RNN has access to:

```text
x1 ... x_t
```

but not:

```text
x_{t+1} ... x_T
```

So it naturally uses:

> past context.

To use both past and future context:

> Bidirectional RNN.

---

# 🔗 35. RNN vs Feed-Forward Network

| Feed-Forward                 | RNN                        |
| ---------------------------- | -------------------------- |
| No persistent sequence state | Hidden state               |
| No built-in temporal context | Carries previous context   |
| Fixed mapping                | Recurrent update           |
| Strong parallelism           | Sequential across time     |
| No time weight sharing       | Shared weights across time |

---

# 🔗 36. RNN vs CNN

| CNN                       | RNN                   |
| ------------------------- | --------------------- |
| Spatial structure         | Sequential structure  |
| Sharing across space      | Sharing across time   |
| Feature maps              | Hidden states         |
| Spatial locality          | Temporal context      |
| Highly parallel spatially | Sequential temporally |

Shortcut:

```text
CNN
→ What pattern exists HERE?

RNN
→ Given what I knew BEFORE,
  what do I know NOW?
```

---

# 🚫 37. High-Yield Interview Traps

### ❌ Hidden state = exact memory

No.

> It is compressed learned context.

---

### ❌ Hidden state and weights are the same thing

No.

```text
h_t
→ dynamic state

W_xh / W_hh
→ learned shared parameters
```

---

### ❌ Different time steps have different weights

No.

> standard RNN shares weights across time.

---

### ❌ Longer sequence means more parameters

No.

> more compute, same recurrent parameters.

---

### ❌ Hidden state is always the output

No.

> output can be generated from hidden state.

---

### ❌ Same weights mean same hidden state

No.

Inputs and previous states differ.

---

### ❌ `h_t` directly sees all previous tokens

No.

It sees them:

> indirectly through recurrence.

---

### ❌ Vanilla RNN perfectly solves long-term dependencies

No.

That is one of its main weaknesses.

---

### ❌ All RNN time steps can be processed together

No.

`h_t` depends on:

`h_{t-1}`.

---

### ❌ Zero `h0` causes neural-network symmetry problems

No.

`h0` is a state, not the shared trainable weight matrix.

---

# 🧩 38. If You Remove `W_hh`

Original:

`h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h)`

Remove:

`W_hh h_{t-1}`

Then:

`h_t = tanh(W_xh x_t + b_h)`

The model loses:

> direct recurrent context.

So:

> recurrence effectively disappears.

---

# 📐 39. Shape Sanity Check

Always verify:

```text
W_xh x_t
→ H-dimensional

W_hh h_{t-1}
→ H-dimensional

b_h
→ H-dimensional
```

Then they can be added.

Result:

`h_t ∈ R^H`

---

# 🧮 40. Fast Parameter Formula

For vanilla hidden-state computation:

# `H(D + H + 1)`

Equivalent to:

`HD + H² + H`

Useful for quick interviews.

---

# 🎤 41. 30-Second Interview Answer

> **A Recurrent Neural Network is designed for sequential data by maintaining a hidden state that carries learned context from previous time steps. At each step, it combines the current input with the previous hidden state using shared recurrent parameters, typically `h_t = tanh(W_xh x_t + W_hh h_{t-1} + b)`. Because the same parameters are reused across time, model size does not depend on sequence length. However, recurrence is sequential and vanilla RNNs struggle with long-range dependencies because information and gradients must pass through many recurrent transformations, leading to vanishing or exploding gradients.**

---

# 🧠 42. Five Things You Must Know Cold

1. **RNN = current input + previous hidden state**
2. **Hidden state = compressed sequence context**
3. **Same weights are shared across time**
4. **Parameter count does not depend on sequence length**
5. **Vanilla RNN struggles with long-term dependencies**

---

# ⚡ 43. Formula Flash Card

```text
Hidden State:

h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h)
```

```text
Optional Output:

y_t = W_hy h_t + b_y
```

```text
Shapes:

W_xh = H × D
W_hh = H × H
b_h  = H
```

```text
Recurrent Parameters:

HD + H² + H
```

---

# 🧠 44. RNN Story in One Diagram

```text
Sequential Data
↓
Previous Context Matters
↓
Need Memory
↓
Hidden State
↓
Current Input + Previous State
↓
Shared Recurrent Transformation
↓
New Hidden State
↓
Repeat Across Time
```

Then:

```text
Long Sequence
↓
Long Recurrent Chain
↓
BPTT
↓
Vanishing / Exploding Gradients
↓
Long-Term Dependency Problem
↓
LSTM / GRU
```

---

# ⭐ Final Mental Model

```text
x_t
+
h_{t-1}
↓
W_xh + W_hh
↓
tanh
↓
h_t
├──→ next time step
└──→ optional output
```

Think:

> **What do I see now + what do I remember from before = what should I remember next?**

---

# ⭐ Golden Rule

> **RNN = one shared recurrent update rule applied repeatedly across time, carrying a learned hidden state from the past into the present.**
