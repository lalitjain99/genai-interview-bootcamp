# 📝 Revision — Recurrent Neural Network

> **Goal:** Revise the core RNN concepts quickly without losing the story behind why RNNs exist.

---

# 🌟 1. Why RNN?

Sequence data has a special property:

> **current meaning can depend on previous elements.**

Example:

```text
I → love → deep → learning
```

When processing `"learning"`, previous words matter.

A normal feed-forward network does not naturally carry:

> previous sequence context.

So we need:

```text
Current Input
+
Previous Context
↓
Updated Context
```

That leads to:

> **Recurrent Neural Networks**

---

# 🧠 2. Core RNN Idea

At time step `t`, RNN receives:

* current input `x_t`
* previous hidden state `h_{t-1}`

and produces:

* new hidden state `h_t`

Conceptually:

`h_t = f(x_t, h_{t-1})`

Mental model:

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

# 🧠 3. Hidden State

Hidden state is:

> **a learned fixed-dimensional representation of previous sequence context.**

Example:

```text
"I"
↓
h1

"love" + h1
↓
h2

"AI" + h2
↓
h3
```

Important:

> `h_t` does not literally store all previous tokens.

It is a:

> **compressed learned representation**

---

# 🔄 4. Why “Recurrent”?

Because the hidden state produced at one step is reused at the next step.

```text
x1 → RNN → h1
             ↓
x2 → RNN → h2
             ↓
x3 → RNN → h3
```

So information keeps recurring through time.

---

# 🔓 5. Unrolled RNN

Compact loop:

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

> These are not separate networks.

They represent:

> **the same RNN cell reused at different time steps.**

---

# 🔄 6. Parameter Sharing Across Time

RNN uses the same parameters at:

```text
t1
t2
t3
...
tT
```

This is similar to CNNs.

```text
CNN
→ same filter across SPACE

RNN
→ same recurrent weights across TIME
```

Key consequence:

> **parameter count does not depend on sequence length.**

---

# 📐 7. Vanilla RNN Equation

Pre-activation:

`a_t = W_xh x_t + W_hh h_{t-1} + b_h`

Hidden state:

# `h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h)`

Optional output:

`y_t = W_hy h_t + b_y`

---

# 🧠 8. Equation Intuition

### `W_xh x_t`

Represents:

> contribution from the current input.

### `W_hh h_{t-1}`

Represents:

> contribution from previous context.

### `b_h`

Bias.

### `tanh`

Adds:

> nonlinearity.

So:

```text
Current Information
+
Previous Information
↓
Learned Combination
↓
New Hidden State
```

---

# 📐 9. Dimensions

If:

`x_t ∈ R^D`

and:

`h_t ∈ R^H`

then:

```text
W_xh → H × D
W_hh → H × H
b_h  → H
h_t  → H
```

Where:

* `D` = input dimension
* `H` = hidden size

---

# 🧮 10. Parameter Count

For the recurrent hidden-state computation:

### Input-to-hidden

`H × D`

### Hidden-to-hidden

`H × H`

### Bias

`H`

Total:

# `HD + H² + H`

---

# 🧮 11. Example

Suppose:

```text
D = 300
H = 128
```

Then:

`W_xh = 128 × 300 = 38,400`

`W_hh = 128 × 128 = 16,384`

`b_h = 128`

Total:

`38,400 + 16,384 + 128`

`= 54,912`

So:

> **54,912 recurrent parameters**

---

# ⭐ 12. Sequence Length vs Parameter Count

If sequence length changes:

```text
T = 10
T = 100
T = 1000
```

parameter count remains:

> unchanged.

Why?

Because:

> the same recurrent weights are reused.

But compute increases because:

```text
Longer Sequence
→ More Time Steps
→ More Recurrent Computation
```

Remember:

```text
Sequence Length ↑
Parameters → Same
Compute ↑
```

---

# 🧠 13. Initial Hidden State `h0`

At the first time step there is no previous state.

So commonly:

`h0 = 0`

This means:

> no previous context is available yet.

Important:

```text
Zero h0
≠
Zero Weight Initialization
```

Zero hidden state is normal.

---

# 📤 14. Hidden State vs Output

Hidden state:

`h_t`

is primarily:

> internal sequence context.

Task output may be:

`y_t`

computed from:

`h_t`.

Example:

`y_t = W_hy h_t + b_y`

So:

```text
h_t
├──→ next time step
└──→ output layer → y_t
```

Therefore:

> **Hidden State ≠ Task Output**

---

# 🎯 15. Many-to-One RNN

Example:

> sentiment classification

```text
This → movie → was → excellent
 ↓       ↓       ↓        ↓
h1      h2      h3       h4
                           ↓
                       Classifier
                           ↓
                        Positive
```

Many inputs:

> one output.

Usually the final hidden state is used as:

> sequence representation.

---

# 🏷️ 16. Many-to-Many RNN

Example:

```text
John   lives   in   London
 ↓       ↓      ↓      ↓
PER      O      O     LOC
```

Each hidden state can produce:

> one output.

Useful for:

* Named Entity Recognition
* POS tagging
* sequence labeling

---

# 🧠 17. Recursive Dependency

Example:

`h3 = f(x3, h2)`

But:

`h2 = f(x2, h1)`

and:

`h1 = f(x1, h0)`

Therefore:

`h3`

is indirectly influenced by:

```text
x1
x2
x3
```

In general:

> `h_t` can carry information from earlier inputs.

---

# ⚠️ 18. Hidden State Is a Compression Bottleneck

Suppose hidden size:

`H = 128`

For:

```text
5-token sequence
```

hidden state size:

`128`

For:

```text
500-token sequence
```

hidden state size:

still:

`128`

So repeatedly:

```text
Previous History
+
New Input
↓
Compress
↓
Fixed-Size Hidden State
```

This creates difficulty for:

> long-range dependencies.

---

# ⏳ 19. Short vs Long Dependencies

### Short Dependency

```text
The sky is dark, so it may ...
```

Recent context may predict:

`rain`.

### Long Dependency

```text
She grew up in Japan ...
many sentences later ...
the language she speaks at home is ...
```

Prediction may depend on:

`Japan`

from much earlier.

Vanilla RNNs can struggle to preserve:

> such distant information.

---

# ⚠️ 20. Why Long-Term Memory Is Difficult

Information may need to survive:

```text
h1
↓
h2
↓
h3
↓
...
↓
h100
```

At each step:

> state is transformed again.

Vanilla RNN has no explicit mechanism saying:

```text
Keep this information for 80 steps.
Forget this information.
```

That motivates:

> LSTM and GRU.

---

# 🔄 21. How Does RNN Learn?

Training still uses:

```text
Forward Pass
↓
Prediction
↓
Loss
↓
Backpropagation
↓
Gradient
↓
Optimizer
↓
Parameter Update
```

But RNN has a special complication:

> the same recurrent weights are reused across many time steps.

So gradients must flow backward through:

> the unrolled recurrent graph.

This is:

# **Backpropagation Through Time — BPTT**

---

# 🔗 22. Connection With CNN Backpropagation

CNN:

```text
Same Filter
used at many spatial positions
↓
Gradient contributions combine
↓
One filter gradient
```

RNN:

```text
Same W_hh
used at many time steps
↓
Gradient contributions combine
↓
One dL/dW_hh
```

Same core principle:

> **shared parameter → accumulate gradients from all uses**

---

# 🚨 23. Vanishing / Exploding Gradient Preview

During BPTT:

```text
Loss
↓
h_T
↓
h_{T-1}
↓
h_{T-2}
↓
...
```

Gradients repeatedly pass through:

* `W_hh`
* activation derivatives

Repeated multiplication can make gradients:

### Very Small

> vanishing gradients

or:

### Very Large

> exploding gradients

This is one reason vanilla RNNs struggle with:

> long sequences.

---

# ⚡ 24. Sequential Computation

To calculate:

`h3`

we need:

`h2`.

To calculate:

`h2`

we need:

`h1`.

So:

```text
h1 → h2 → h3 → h4
```

must be processed sequentially.

Therefore:

> standard RNNs have limited parallelism across time.

---

# 🧠 25. Batch Parallelism vs Time Parallelism

RNN can process multiple examples together:

```text
Batch Dimension
→ Parallel
```

But within one sequence:

```text
t1 → t2 → t3
```

remains sequential.

So:

```text
Across Batch
→ Parallel

Across Time
→ Sequential
```

---

# 🌊 26. Streaming Advantage

RNN naturally supports incremental processing.

```text
x1 arrives
↓
h1

x2 arrives
↓
h2

x3 arrives
↓
h3
```

Only the current input and previous state are needed.

This makes recurrent models conceptually useful for:

> streaming sequence data.

---

# 🔗 27. RNN vs Feed-Forward Network

| Feed-Forward              | RNN                        |
| ------------------------- | -------------------------- |
| No recurrent state        | Hidden state               |
| Inputs mostly independent | Ordered dependency         |
| No built-in history       | Carries previous context   |
| No temporal sharing       | Shared weights across time |
| High parallelism          | Sequential time dependency |

---

# 🔗 28. RNN vs CNN

| CNN                        | RNN                            |
| -------------------------- | ------------------------------ |
| Spatial structure          | Sequential structure           |
| Sharing across space       | Sharing across time            |
| Feature maps               | Hidden states                  |
| Local spatial context      | Running sequence context       |
| Strong spatial parallelism | Sequential temporal processing |

Shortcut:

```text
CNN
→ What pattern exists HERE?

RNN
→ Given what I knew BEFORE,
  what do I know NOW?
```

---

# ⚠️ 29. High-Yield Misconceptions

### ❌ Hidden state stores every previous token exactly

No.

> It is compressed learned context.

---

### ❌ Different time steps use different weights

No.

> Same recurrent parameters are reused.

---

### ❌ Hidden state is always the prediction

No.

> It may feed a separate output layer.

---

### ❌ RNN solves long-term dependencies completely

No.

> Vanilla RNNs struggle with them.

---

### ❌ Longer sequence means more model parameters

No.

> It means more computation.

---

### ❌ RNN time steps are fully parallel

No.

> `h_t` depends on `h_{t-1}`.

---

# 📐 30. Formula Sheet

### Pre-Activation

`a_t = W_xh x_t + W_hh h_{t-1} + b_h`

### Hidden State

`h_t = tanh(a_t)`

### Optional Output

`y_t = W_hy h_t + b_y`

### Parameter Count

`HD + H² + H`

---

# 🧠 31. Five Things to Remember Cold

1. `x_t` = current input.
2. `h_{t-1}` = previous context.
3. `h_t` = updated context.
4. Same weights are reused across time.
5. Vanilla RNN struggles with long-term dependencies.

---

# 🎤 32. 30-Second Interview Revision

> **An RNN is a neural network for sequential data that maintains a hidden state representing previous context. At each time step it combines the current input `x_t` with the previous hidden state `h_{t-1}` to produce a new hidden state, typically using `h_t = tanh(W_xh x_t + W_hh h_{t-1} + b)`. The same recurrent parameters are shared across all time steps, so parameter count does not depend on sequence length. However, long recurrent chains make vanilla RNNs difficult to train for long-range dependencies because gradients can vanish or explode during Backpropagation Through Time.**

---

# 🧠 Final Mental Model

```text
Sequence Needs Context
↓
Carry Previous Information
↓
Hidden State
↓
Current Input + Previous Hidden State
↓
Shared Recurrent Transformation
↓
New Hidden State
↓
Carry Forward
```

Then:

```text
Long Sequence
↓
Many Recurrent Transformations
↓
BPTT
↓
Vanishing / Exploding Gradients
↓
LSTM / GRU
```

---

# ⭐ Golden Rule

> **RNN = current input + previous hidden context → updated hidden context, using the same learned transformation at every time step.**
