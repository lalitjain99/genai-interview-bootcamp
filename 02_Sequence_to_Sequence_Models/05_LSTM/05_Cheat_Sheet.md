# ⚡ Cheat Sheet — Long Short-Term Memory (LSTM)

> **Goal:** Fast revision of why LSTM exists, how its gates work, how cell state differs from hidden state, how candidate memory is used, and why LSTM improves long-term gradient flow.

---

# 🌟 1. LSTM in One Line

> **LSTM = RNN + controlled memory path + gates**

It was introduced mainly to improve:

* long-term dependency learning
* vanishing-gradient behavior
* selective memory retention

---

# 🧠 2. Why Vanilla RNN Struggles

Vanilla RNN:

`h_t = tanh(W_xh x_t + W_hh h_(t-1) + b_h)`

At every time step:

```text id="lstm-cheat-1"
Old State
+
New Input
↓
Transform Again
↓
New State
```

For long sequences:

```text id="lstm-cheat-2"
Repeated transformations
↓
Information may fade
+
Gradients may vanish/explode
```

LSTM introduces:

> a dedicated controlled memory pathway.

---

# ⭐ 3. Two States in LSTM

### Cell State

`c_t`

> controlled internal memory pathway

### Hidden State

`h_t`

> current exposed representation

Shortcut:

```text id="lstm-cheat-3"
c_t
→ what I keep internally

h_t
→ what I expose right now
```

Do **not** say:

> hidden state stores only the last few states.

---

# 🔄 4. Inputs and Outputs at Time `t`

Inputs:

```text id="lstm-cheat-4"
x_t
→ current input

h_(t-1)
→ previous hidden state

c_(t-1)
→ previous cell state
```

Outputs:

```text id="lstm-cheat-5"
h_t
c_t
```

---

# 🚪 5. LSTM Decisions

At each time step, LSTM asks:

```text id="lstm-cheat-6"
What old information should I keep?

What new information could I write?

How much of that new information should I write?

What part of memory should I expose?
```

These correspond to:

```text id="lstm-cheat-7"
Forget Gate
Candidate Memory
Input Gate
Output Gate
```

---

# 🔐 6. Forget Gate

Equation:

`f_t = σ(W_f[h_(t-1), x_t] + b_f)`

Purpose:

> control how much previous cell-state information is retained

Applied as:

`f_t ⊙ c_(t-1)`

Interpretation:

```text id="lstm-cheat-8"
f_t ≈ 1
→ keep

f_t ≈ 0
→ forget
```

---

# 🧠 7. Why Forget Gate Uses Current Input `x_t`

Forget decision depends on:

> what new information just arrived.

Example:

```text id="lstm-cheat-9"
Old memory:
London

Current input:
Moved to Paris
```

The current input tells the LSTM that some old memory may now be outdated.

So:

```text id="lstm-cheat-10"
x_t + h_(t-1)
→ decide how much to retain

c_(t-1)
→ actual memory being filtered
```

---

# ✍️ 8. Input Gate

Equation:

`i_t = σ(W_i[h_(t-1), x_t] + b_i)`

Purpose:

> control how much new candidate information gets written

Think:

```text id="lstm-cheat-11"
i_t
→ write permission
```

---

# 📝 9. Candidate Memory

Equation:

`g_t = tanh(W_g[h_(t-1), x_t] + b_g)`

Also written as:

`c̃_t`

Purpose:

> proposed new content that could be added to memory

Best distinction:

```text id="lstm-cheat-12"
Candidate g_t
→ WHAT could be written

Input Gate i_t
→ HOW MUCH should be written
```

---

# ⭐ 10. Candidate + Input Gate

Actual new-memory contribution:

# `i_t ⊙ g_t`

Example:

```text id="lstm-cheat-13"
g_t = [0.8, -0.5]

i_t = [1.0, 0.2]
```

Then:

```text id="lstm-cheat-14"
i_t ⊙ g_t
=
[0.8, -0.1]
```

---

# 🔥 11. Why `tanh` for Candidate?

Candidate is:

> CONTENT

It should be able to represent signed information.

`tanh` gives:

`[-1, 1]`

So candidate values can be:

* positive
* negative
* near zero

It is also bounded.

---

# 🚪 12. Why Sigmoid for Gates?

Sigmoid gives:

`[0,1]`

That is ideal for:

> soft control

```text id="lstm-cheat-15"
0
→ block

0.5
→ partially allow

1
→ strongly allow
```

Shortcut:

```text id="lstm-cheat-16"
sigmoid
→ CONTROL

tanh
→ CONTENT
```

---

# 📐 13. Most Important Equation

# `c_t = f_t ⊙ c_(t-1) + i_t ⊙ g_t`

Read it as:

# **New Memory = KEEP + WRITE**

Where:

```text id="lstm-cheat-17"
KEEP
= f_t ⊙ c_(t-1)
```

and:

```text id="lstm-cheat-18"
WRITE
= i_t ⊙ g_t
```

---

# 🧠 14. Cell-State Update in Words

```text id="lstm-cheat-19"
Take old memory
↓
Keep useful parts
↓
Add selected new information
↓
Updated memory
```

That is the core of LSTM.

---

# 📤 15. Output Gate

Equation:

`o_t = σ(W_o[h_(t-1), x_t] + b_o)`

Purpose:

> decide how much of the updated memory should be exposed

Hidden state:

# `h_t = o_t ⊙ tanh(c_t)`

---

# ⭐ 16. Memory vs Exposure

```text id="lstm-cheat-20"
c_t
→ memory retained internally

h_t
→ exposed representation
```

If:

`o_t ≈ 0`

then:

`h_t`

may be suppressed while:

`c_t`

still retains information.

Remember:

> **Not exposed ≠ forgotten**

---

# 🧩 17. Full LSTM Equations

### Forget

`f_t = σ(W_f[h_(t-1), x_t] + b_f)`

### Input

`i_t = σ(W_i[h_(t-1), x_t] + b_i)`

### Candidate

`g_t = tanh(W_g[h_(t-1), x_t] + b_g)`

### Cell State

`c_t = f_t ⊙ c_(t-1) + i_t ⊙ g_t`

### Output

`o_t = σ(W_o[h_(t-1), x_t] + b_o)`

### Hidden

`h_t = o_t ⊙ tanh(c_t)`

---

# ⚡ 18. Gate Shortcut

```text id="lstm-cheat-21"
f_t
→ KEEP?

g_t
→ WHAT new content?

i_t
→ HOW MUCH to write?

c_t
→ MEMORY

o_t
→ HOW MUCH to expose?

h_t
→ EXPOSED STATE
```

---

# 🔍 19. Important Special Cases

## If `f_t = 1`

Old memory is fully retained.

---

## If `f_t = 0`

Old memory is removed.

---

## If `i_t = 0`

No new candidate is written.

---

## If `f_t = 1` and `i_t = 0`

# `c_t = c_(t-1)`

Memory passes forward unchanged.

---

## If `o_t = 0`

Hidden state is suppressed, but cell memory can remain.

---

# 🧮 20. Quick Numerical Example

Suppose:

```text id="lstm-cheat-22"
c_(t-1) = 0.8
f_t     = 0.9
i_t     = 0.5
g_t     = 0.4
```

Then:

`c_t = 0.9×0.8 + 0.5×0.4`

`= 0.72 + 0.20`

# `= 0.92`

Interpretation:

> preserve most old memory + add some new information.

---

# ⭐ 21. Why LSTM Helps Vanishing Gradients

Vanilla RNN gradient path repeatedly involves:

```text id="lstm-cheat-23"
W_hh
×
activation derivative
```

LSTM cell-state path:

`c_t = f_t ⊙ c_(t-1) + ...`

has a direct derivative containing:

`f_t`.

So across time:

```text id="lstm-cheat-24"
f_t × f_(t-1) × f_(t-2) × ...
```

If relevant forget gates remain near:

`1`

gradient can survive much better.

---

# ⚠️ 22. LSTM Does Not Completely Solve Vanishing Gradients

If:

```text id="lstm-cheat-25"
f_t = 0.5
```

for many steps:

`0.5^T`

still becomes tiny.

So say:

> **LSTM mitigates vanishing gradients and improves long-range gradient flow.**

Do not say:

> **LSTM completely eliminates vanishing gradients.**

---

# 🔗 23. Connection to Residual Connections

Residual connection:

`y = x + F(x)`

LSTM cell update:

`c_t = retained old state + new contribution`

Both use:

> additive information paths

which can improve:

* information preservation
* gradient flow

They are not the same architecture, but share a useful design principle.

---

# 📐 24. Dimensions

If:

* input size = `D`
* hidden size = `H`

then:

`[h_(t-1), x_t]`

has dimension:

`H + D`

Each gate/candidate produces:

`H`

values.

Therefore:

```text id="lstm-cheat-26"
W_f → H × (H+D)
W_i → H × (H+D)
W_g → H × (H+D)
W_o → H × (H+D)
```

Bias:

`H`

for each transformation.

---

# 🧮 25. Parameter Count

Standard LSTM cell:

# `4H(H + D + 1)`

Why `4`?

Because of:

```text id="lstm-cheat-27"
Forget
Input
Candidate
Output
```

---

# 🧮 26. Parameter Example

Given:

```text id="lstm-cheat-28"
D = 64
H = 128
```

Then:

`4 × 128 × (128 + 64 + 1)`

`= 4 × 128 × 193`

# `= 98,816`

---

# 🔗 27. Vanilla RNN vs LSTM

| Vanilla RNN                    | LSTM                             |
| ------------------------------ | -------------------------------- |
| Mainly hidden state            | Hidden + cell state              |
| Simple recurrent update        | Gated memory update              |
| Rewrites state repeatedly      | Selectively edits memory         |
| Fewer parameters               | More parameters                  |
| Lower compute                  | Higher compute                   |
| Poorer long-term gradient flow | Better long-term gradient flow   |
| Short dependencies easier      | Long dependencies handled better |

---

# 🧠 28. Cell State vs Hidden State

Best answer:

```text id="lstm-cheat-29"
c_t
→ controlled memory pathway

h_t
→ current exposed representation
```

Avoid:

> “hidden state stores the last few states.”

Avoid:

> “cell state stores the entire sequence.”

Both are learned vector representations.

---

# ⚠️ 29. Candidate Memory Is Not Final Memory

```text id="lstm-cheat-30"
g_t
→ proposed content

i_t ⊙ g_t
→ accepted new content

c_t
→ actual updated memory
```

---

# ⚠️ 30. Input Gate Is Not Candidate Memory

```text id="lstm-cheat-31"
Input Gate
→ CONTROL

Candidate
→ CONTENT
```

This distinction is one of the most important LSTM interview points.

---

# ⚠️ 31. Output Gate Does Not Forget

Output gate controls:

> what becomes visible in `h_t`.

Forget gate controls:

> what old information remains in `c_t`.

Do not swap them.

---

# ⚠️ 32. Gates Are Not Binary

Gate values usually look like:

```text id="lstm-cheat-32"
0.17
0.62
0.94
```

They are:

> soft differentiable controls.

---

# ⚠️ 33. Gates Are Learned

Do not think:

```text id="lstm-cheat-33"
if important:
    gate = 1
```

Instead:

```text id="lstm-cheat-34"
x_t + h_(t-1)
↓
Learned Transformation
↓
sigmoid
↓
Gate Values
```

---

# 🔄 34. LSTM Still Uses BPTT

LSTM is trained using:

> **Backpropagation Through Time**

```text id="lstm-cheat-35"
Forward LSTM
↓
Loss
↓
BPTT
↓
Gate + State Gradients
↓
Optimizer
```

LSTM changes the architecture.

It does not replace BPTT.

---

# ⏳ 35. LSTM Is Still Sequential

At time `t`, it needs:

```text id="lstm-cheat-36"
h_(t-1)
c_(t-1)
```

Therefore:

```text id="lstm-cheat-37"
t1 → t2 → t3 → ...
```

remains sequential.

So LSTM improves:

> memory and gradient flow

but not:

> time-step parallelism.

---

# 🌊 36. Streaming Strength

For streaming inference, maintain:

```text id="lstm-cheat-38"
h_t
c_t
```

Then:

```text id="lstm-cheat-39"
x_(t+1)
+
h_t
+
c_t
↓
LSTM
↓
h_(t+1), c_(t+1)
```

No need to replay all earlier inputs.

---

# 👥 37. Shared Model, Separate States

Shared across users:

> trained LSTM parameters.

Separate per sequence:

```text id="lstm-cheat-40"
User A → h_A, c_A
User B → h_B, c_B
User C → h_C, c_C
```

Do not share recurrent state between unrelated sequences.

---

# 💰 38. LSTM Cost

Compared with vanilla RNN:

```text id="lstm-cheat-41"
More Gates
↓
More Matrix Multiplications
↓
More Parameters
↓
More Compute + Memory
```

This later motivates:

> **GRU**

---

# 🚫 39. High-Yield Interview Traps

### ❌ LSTM stores the whole sequence exactly

No.

---

### ❌ Hidden state = last few states

No.

---

### ❌ Candidate = final memory

No.

---

### ❌ Input gate creates content

No.

Candidate creates/proposes content.

---

### ❌ Candidate uses sigmoid

Typically no.

It uses `tanh`.

---

### ❌ Forget gate does not need current input

Wrong.

Current input helps determine whether old information is still relevant.

---

### ❌ Output gate forgets memory

No.

It controls exposure.

---

### ❌ `o_t = 0` deletes cell memory

No.

---

### ❌ Gates are binary

No.

---

### ❌ LSTM eliminates vanishing gradients completely

No.

---

### ❌ LSTM does not need BPTT

False.

---

### ❌ LSTM is fully parallel across time

False.

---

# 📐 40. Formula Flash Card

```text id="lstm-cheat-42"
f_t = σ(W_f[h_(t-1),x_t] + b_f)

i_t = σ(W_i[h_(t-1),x_t] + b_i)

g_t = tanh(W_g[h_(t-1),x_t] + b_g)

c_t = f_t ⊙ c_(t-1) + i_t ⊙ g_t

o_t = σ(W_o[h_(t-1),x_t] + b_o)

h_t = o_t ⊙ tanh(c_t)
```

Parameter count:

# `4H(H+D+1)`

---

# 🎤 41. 30-Second Interview Answer

> **LSTM is a gated recurrent architecture designed to improve long-term dependency learning. It maintains a cell state as a controlled memory pathway. The forget gate decides how much old memory to retain, candidate memory proposes new content, the input gate decides how much of that content to write, and the output gate controls how much of the updated cell state is exposed through the hidden state. Its additive cell-state update provides a more favorable information and gradient path than the repeated nonlinear state replacement of a vanilla RNN.**

---

# 🧠 42. Seven Things You Must Know Cold

1. **LSTM = gated recurrent memory architecture**
2. **`c_t` = controlled memory pathway**
3. **`h_t` = exposed representation**
4. **Forget gate = how much old memory stays**
5. **Candidate = WHAT; Input gate = HOW MUCH**
6. **Cell state = KEEP + WRITE**
7. **Additive memory path improves long-term gradient flow**

---

# 🧠 43. Story in One Diagram

```text id="lstm-cheat-43"
Vanilla RNN
↓
Long-Term Dependency Problem
↓
Need Stable Memory Path
↓
Cell State
↓
Need Controlled Memory
```

Then:

```text id="lstm-cheat-44"
Forget Gate
→ retain old information

Candidate
→ propose new information

Input Gate
→ accept/reject candidate

Cell State
→ updated memory

Output Gate
→ expose relevant memory

Hidden State
→ current representation
```

---

# ⭐ Final Mental Model

```text id="lstm-cheat-45"
OLD MEMORY
c_(t-1)
   │
   ▼
Forget Gate
   │
   ▼
KEEP
   │
   ├──────────────┐
   │              │
   │       Candidate g_t
   │              │
   │       × Input Gate
   │              │
   │           WRITE
   │              │
   └─────── + ────┘
           │
           ▼
          c_t
           │
         tanh
           │
      × Output Gate
           │
           ▼
          h_t
```

Think:

```text id="lstm-cheat-46"
f_t → KEEP old?
g_t → WHAT new?
i_t → HOW MUCH new?
c_t → MEMORY
o_t → HOW MUCH show?
h_t → SHOW now
```

---

# ⭐ Golden Rule

> **LSTM works by separating memory from exposure: it selectively keeps old information, selectively writes new candidate content, and selectively exposes the resulting memory while preserving a better path for long-range information and gradients.**
