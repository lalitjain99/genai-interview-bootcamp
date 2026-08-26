# 📝 Revision — Long Short-Term Memory (LSTM)

> **Goal:** Quickly revise why LSTM was needed, how its gates work, how cell state differs from hidden state, how candidate memory works, and why LSTM improves long-term gradient flow.

---

# 🌟 1. Why Was LSTM Needed?

Vanilla RNN:

`h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h)`

At every time step:

```text
Previous Hidden State
+
Current Input
↓
New Nonlinear Transformation
↓
New Hidden State
```

For long sequences, information and gradients must pass through:

> many repeated recurrent transformations.

This can cause:

* vanishing gradients
* exploding gradients
* poor long-term dependency learning

So we needed:

> a more stable and controllable memory path.

That led to:

# **LSTM — Long Short-Term Memory**

---

# 🧠 2. LSTM Core Idea

LSTM introduces:

> **a dedicated cell state for memory**

and:

> **gates that control what should be forgotten, written, and exposed.**

Instead of repeatedly replacing the entire memory:

```text
Old State
↓
Transform Everything
↓
New State
```

LSTM performs:

```text
Old Memory
↓
Keep Selected Parts
+
Add Selected New Information
↓
Updated Memory
```

---

# ⭐ 3. LSTM Has Two States

## Cell State

`c_t`

Think:

> **controlled internal memory**

It carries selected information across time.

---

## Hidden State

`h_t`

Think:

> **current exposed representation**

It is used by:

* the next LSTM step
* downstream prediction layers

---

### Best Mental Model

```text
c_t
→ what I keep in memory

h_t
→ what I expose/use right now
```

Important:

> `h_t` does NOT simply store the last few hidden states.

It is a learned representation derived from the current cell state and gating decisions.

---

# 🔄 4. What Enters an LSTM at Time `t`?

The LSTM receives:

```text
x_t
→ current input

h_(t-1)
→ previous hidden state

c_(t-1)
→ previous cell state
```

Then it produces:

```text
h_t
→ new hidden state

c_t
→ new cell state
```

---

# 🚪 5. Why Gates?

Once we have memory, we need control over it.

The LSTM must decide:

```text
What old information should I forget?

What new information should I write?

What part of memory should I expose?
```

These decisions are handled by:

* forget gate
* input gate
* output gate

The gates use:

> **sigmoid**

because sigmoid outputs values between:

`0` and `1`.

---

# 🔐 6. Forget Gate

Equation:

`f_t = σ(W_f [h_(t-1), x_t] + b_f)`

Purpose:

> **decide how much old cell-state information to retain**

Old memory contribution:

`f_t ⊙ c_(t-1)`

Interpretation:

```text
f_t ≈ 1
→ keep

f_t ≈ 0
→ forget

0 < f_t < 1
→ partially retain
```

---

# ✍️ 7. Input Gate

Equation:

`i_t = σ(W_i [h_(t-1), x_t] + b_i)`

Purpose:

> **decide how much new candidate information should be written into memory**

Think:

```text
i_t
→ write permission
```

Important:

> the input gate does NOT contain the new content itself.

It only controls:

> how much of that content is accepted.

---

# 📝 8. Candidate Memory

Equation:

`g_t = tanh(W_g [h_(t-1), x_t] + b_g)`

You may also see:

`c̃_t`

Candidate memory means:

> **the new information the LSTM is proposing to write into the cell state.**

Best distinction:

```text
g_t
→ WHAT could be written

i_t
→ HOW MUCH should be written
```

Actual new memory contribution:

`i_t ⊙ g_t`

---

# ⭐ 9. Candidate Memory vs Input Gate

This distinction is extremely important.

Suppose:

```text
g_t = [0.8, -0.5, 0.3]

i_t = [1.0, 0.2, 0.0]
```

Then:

`i_t ⊙ g_t`

becomes:

```text
[0.8, -0.1, 0]
```

So:

```text
Candidate
→ proposed content

Input Gate
→ controls acceptance

Product
→ actual information written
```

---

# 🧠 10. Why `tanh` for Candidate Memory?

Candidate memory represents:

> content

not:

> a gate.

`tanh` outputs:

`[-1, 1]`

So candidate values can be:

* positive
* negative
* zero

This allows the model to propose:

> signed changes in the learned memory representation.

Example:

```text
+0.8
→ strong positive candidate contribution

-0.6
→ strong contribution in the opposite learned direction

0
→ no candidate contribution
```

Also, `tanh` keeps candidate values bounded.

---

# ⭐ 11. Why Sigmoid for Gates but `tanh` for Candidate?

Best shortcut:

```text
Sigmoid
→ CONTROL
→ 0 to 1
→ how much?

tanh
→ CONTENT
→ -1 to +1
→ what?
```

Therefore:

```text
Forget Gate  → sigmoid
Input Gate   → sigmoid
Output Gate  → sigmoid

Candidate    → tanh
```

---

# 📐 12. Cell-State Update

The most important LSTM equation:

# `c_t = f_t ⊙ c_(t-1) + i_t ⊙ g_t`

Read it as:

```text
New Memory
=
Old Memory I Keep
+
New Information I Write
```

Or:

```text
c_t
=
KEEP
+
WRITE
```

Where:

```text
KEEP
= f_t ⊙ c_(t-1)

WRITE
= i_t ⊙ g_t
```

---

# 🧠 13. Why the Cell-State Equation Matters

Vanilla RNN repeatedly performs something like:

```text
Old State
↓
Matrix Transformation
↓
Nonlinearity
↓
Completely New State
```

LSTM instead gives old cell state a more direct path:

```text
c_(t-1)
↓
× f_t
↓
+
↓
c_t
```

This additive update makes information and gradients:

> easier to preserve across long time spans.

---

# 📤 14. Output Gate

Equation:

`o_t = σ(W_o [h_(t-1), x_t] + b_o)`

Purpose:

> **decide how much of the updated memory should be exposed as the hidden state**

The hidden state is:

# `h_t = o_t ⊙ tanh(c_t)`

So:

```text
Cell State
↓
tanh
↓
Output Gate
↓
Hidden State
```

---

# 🧠 15. Memory vs Exposure

Suppose:

`c_t`

contains useful information.

But:

`o_t ≈ 0`

Then that information may remain in the cell state while being weakly exposed through:

`h_t`.

So:

```text
Not Exposed
≠
Forgotten
```

This is one of the most useful LSTM intuitions.

---

# 🧩 16. Complete LSTM Equations

### Forget Gate

`f_t = σ(W_f [h_(t-1), x_t] + b_f)`

### Input Gate

`i_t = σ(W_i [h_(t-1), x_t] + b_i)`

### Candidate Memory

`g_t = tanh(W_g [h_(t-1), x_t] + b_g)`

### Cell State

`c_t = f_t ⊙ c_(t-1) + i_t ⊙ g_t`

### Output Gate

`o_t = σ(W_o [h_(t-1), x_t] + b_o)`

### Hidden State

`h_t = o_t ⊙ tanh(c_t)`

---

# ⚡ 17. Five Questions Behind the Equations

Instead of memorizing formulas, remember:

```text
f_t
→ What should I FORGET?

i_t
→ How much should I WRITE?

g_t
→ WHAT new information is available?

c_t
→ What is my UPDATED MEMORY?

o_t
→ What should I SHOW?

h_t
→ What am I exposing NOW?
```

---

# 🧮 18. Simple Cell-State Example

Suppose:

```text
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

* most old memory was preserved
* some new information was added

---

# 🔍 19. What If Forget Gate = 1?

If:

`f_t = 1`

then old memory is fully retained before the new write:

`c_t = c_(t-1) + i_t ⊙ g_t`

This strongly favors:

> memory preservation.

---

# 🔍 20. What If Forget Gate = 0?

If:

`f_t = 0`

then old cell-state contribution becomes:

`0`

So:

`c_t = i_t ⊙ g_t`

The old memory is removed from that dimension.

---

# 🔍 21. What If Input Gate = 0?

If:

`i_t = 0`

then:

`i_t ⊙ g_t = 0`

No candidate information is written.

So:

`c_t = f_t ⊙ c_(t-1)`

The cell only:

> keeps/forgets existing memory.

---

# 🔍 22. What If Forget = 1 and Input = 0?

Then:

`c_t = c_(t-1)`

So:

```text
c_(t-1)
────────────→
c_t
```

The memory is copied forward unchanged.

This is the key intuition behind LSTM's ability to preserve information over many time steps.

---

# 🔍 23. What If Output Gate = 0?

Since:

`h_t = o_t ⊙ tanh(c_t)`

if:

`o_t = 0`

then:

`h_t ≈ 0`

for that component.

But:

`c_t`

can still contain information.

Again:

```text
Hidden from output
≠
removed from memory
```

---

# ⭐ 24. Why LSTM Helps Vanishing Gradients

Cell update:

`c_t = f_t ⊙ c_(t-1) + ...`

The direct derivative along the cell-state path contains:

`f_t`

So:

`∂c_t/∂c_(t-1)`

contains approximately:

`f_t`

Across multiple steps:

```text
f_t × f_(t-1) × f_(t-2) × ...
```

If important-memory forget gates stay near:

`1`

then the gradient can survive much better.

Example:

```text
1 × 1 × 1 × 1
≈ 1
```

Compare with vanilla RNN, where long gradient flow repeatedly involves:

```text
W_hh
×
tanh'
```

---

# ⚠️ 25. LSTM Does Not Completely Eliminate Vanishing Gradients

Do not say:

> “LSTM solves vanishing gradients completely.”

If:

`f_t = 0.5`

for many steps:

```text
0.5 × 0.5 × 0.5 × ...
```

can still vanish.

Better statement:

> **LSTM provides a much more favorable controlled path for long-term gradient and information flow.**

---

# ⭐ 26. Why the Forget Gate Matters for Gradients

Forget gate controls:

> forward memory retention

and also influences:

> gradient flow through the cell-state path.

So:

```text
Forget Gate Near 1
↓
Memory preserved better
+
Gradient preserved better
```

This is a key interview point.

---

# 🔗 27. Connection to Residual Connections

Residual network:

`y = x + F(x)`

provides a relatively direct information and gradient route.

LSTM's additive cell update:

`c_t = old contribution + new contribution`

uses a related high-level principle:

> avoid forcing all information through a completely new nonlinear transformation at every step.

LSTM is not the same as ResNet, but the intuition is related.

---

# 📐 28. Dimensions

If:

```text
Input Dimension  = D
Hidden Dimension = H
```

then:

`[h_(t-1), x_t]`

has dimension:

`H + D`

Each gate/candidate produces:

`H`

values.

Therefore:

```text
W_f → H × (H+D)

W_i → H × (H+D)

W_g → H × (H+D)

W_o → H × (H+D)
```

Biases:

`H`

each.

---

# 🧮 29. LSTM Parameter Count

One transformation contains:

`H(H+D) + H`

parameters.

LSTM has four:

* forget
* input
* candidate
* output

Therefore:

# `4H(H + D + 1)`

This excludes any downstream output layer.

---

# 🧮 30. Parameter Example

Given:

```text
D = 100
H = 256
```

Parameter count:

`4 × 256 × (256 + 100 + 1)`

`= 4 × 256 × 357`

# `= 365,568`

---

# 🔗 31. Vanilla RNN vs LSTM

| Vanilla RNN                            | LSTM                           |
| -------------------------------------- | ------------------------------ |
| Mainly hidden state                    | Hidden + cell state            |
| One main recurrent transform           | Multiple gated transforms      |
| Rewrites context each step             | Selectively edits memory       |
| Poorer long-term gradient flow         | Better long-term gradient flow |
| Fewer parameters                       | More parameters                |
| Lower compute                          | Higher compute                 |
| Stronger for short/simple dependencies | Better for longer dependencies |

---

# 🧠 32. Cell State vs Hidden State

### Cell State `c_t`

> controlled memory pathway

### Hidden State `h_t`

> current exposed representation

Shortcut:

```text
c_t → what I remember internally

h_t → what I expose right now
```

But avoid saying:

> hidden state stores only the last few states.

That is not how it works.

---

# ⚠️ 33. Cell State Is Not Permanent Memory

Cell state can be:

* preserved
* modified
* partially forgotten
* overwritten

through:

```text
Forget Gate
+
Input Gate / Candidate
```

So:

> `c_t` is controlled memory, not immutable storage.

---

# ⚠️ 34. Candidate Memory Is Not the Cell State

Candidate:

`g_t`

means:

> proposed new information.

Cell state:

`c_t`

means:

> updated memory after combining retained old information and accepted candidate information.

So:

```text
Candidate
≠
Memory

Candidate × Input Gate
→ new contribution

Old Memory + New Contribution
→ updated memory
```

---

# ⚠️ 35. Input Gate Is Not “Input Memory”

There is no standard separate concept called:

> input memory.

The standard LSTM concepts are:

* input gate
* candidate memory
* cell state

Remember:

```text
Input Gate
→ control

Candidate
→ content

Cell State
→ memory
```

---

# ⚠️ 36. Gates Are Not Binary

Sigmoid produces continuous values such as:

```text
0.12
0.51
0.96
```

So gates perform:

> soft differentiable filtering.

They are not normally strict:

`0 / 1`

switches.

---

# ⚠️ 37. Gates Are Learned, Not Programmed

We do not manually write:

```text
if token == "France":
    remember = True
```

Instead:

```text
x_t + h_(t-1)
↓
Learned Weights
↓
Sigmoid / tanh
↓
Gate + Candidate Values
```

The network learns these behaviors from the training objective.

---

# 🔄 38. How Is LSTM Trained?

LSTM still uses:

> **Backpropagation Through Time**

So:

```text
Forward Through LSTM Sequence
↓
Compute Loss
↓
BPTT
↓
Gradients Through Gates + Cell State
↓
Optimizer
↓
Update Parameters
```

LSTM does not replace BPTT.

It gives BPTT:

> a better recurrent architecture to work with.

---

# ⏳ 39. LSTM Still Processes Time Sequentially

At time `t`, LSTM needs:

```text
h_(t-1)
c_(t-1)
```

So:

```text
t1 → t2 → t3 → ...
```

cannot be fully parallelized across time.

Therefore LSTM improves long-term memory but does not solve:

> recurrent sequential computation.

---

# 🌊 40. Streaming Strength

For streaming inference, preserve:

```text
h_t
c_t
```

Then when:

`x_(t+1)`

arrives:

```text
x_(t+1)
+
h_t
+
c_t
↓
Next LSTM Step
```

No need to replay the entire sequence.

---

# 🧠 41. Independent Sequences Need Separate States

Model parameters can be shared.

States should be sequence-specific.

```text
Shared LSTM Weights

User A → h_A, c_A
User B → h_B, c_B
User C → h_C, c_C
```

Do not leak:

`h` or `c`

between unrelated sequences.

---

# 📈 42. LSTM Trade-Off

LSTM gains:

* better long-term memory
* better gradient flow
* adaptive forgetting/writing

But costs:

* more parameters
* more computation
* more memory
* more latency

This motivates the later:

> **GRU**

---

# 🚫 43. High-Yield Interview Traps

### ❌ Cell state = hidden state

No.

---

### ❌ Hidden state stores the last few states

No.

---

### ❌ Candidate memory is already stored memory

No.

It is:

> proposed new content.

---

### ❌ Input gate determines what content exists

No.

Candidate generates content.

Input gate controls:

> how much enters memory.

---

### ❌ Candidate uses sigmoid

Typically no.

Candidate uses:

> `tanh`

because it represents signed content.

---

### ❌ Forget gate deletes entire memory at once

Not necessarily.

It operates element-wise.

---

### ❌ Output gate forgets memory

No.

It controls:

> exposure into hidden state.

---

### ❌ LSTM completely eliminates vanishing gradient

No.

---

### ❌ LSTM does not use BPTT

False.

---

### ❌ Gates are manually defined

False.

They are learned.

---

# 📐 44. Formula Flash Card

```text
Forget:

f_t = σ(W_f[h_(t-1), x_t] + b_f)
```

```text
Input:

i_t = σ(W_i[h_(t-1), x_t] + b_i)
```

```text
Candidate:

g_t = tanh(W_g[h_(t-1), x_t] + b_g)
```

```text
Memory:

c_t = f_t ⊙ c_(t-1) + i_t ⊙ g_t
```

```text
Output:

o_t = σ(W_o[h_(t-1), x_t] + b_o)
```

```text
Hidden:

h_t = o_t ⊙ tanh(c_t)
```

---

# 🎤 45. 30-Second Interview Answer

> **LSTM is a gated recurrent architecture designed to improve long-term dependency learning compared with vanilla RNNs. It introduces a cell state that acts as a controlled memory pathway. The forget gate determines how much old memory to retain, the candidate proposes new content, the input gate determines how much of that candidate to write, and the output gate controls how much of the updated cell state is exposed as the hidden state. Because the cell state uses an additive gated update, important information and gradients can travel through time more effectively than in a vanilla RNN.**

---

# 🧠 46. Seven Things You Must Know Cold

1. **LSTM was motivated by vanilla RNN long-term dependency problems.**
2. **`c_t` = controlled memory pathway.**
3. **`h_t` = current exposed representation.**
4. **Forget gate decides what old memory remains.**
5. **Candidate says WHAT; input gate says HOW MUCH.**
6. **Cell update = KEEP + WRITE.**
7. **Additive cell-state path improves long-range gradient flow.**

---

# 🧠 47. LSTM Story in One Diagram

```text
Vanilla RNN
↓
Repeatedly Rewrites Hidden State
↓
Long-Term Information + Gradient Problems
↓
Need Better Memory Path
↓
Cell State
↓
Need Memory Control
```

Then:

```text
Forget Gate
→ what old memory stays

Candidate
→ what new content is proposed

Input Gate
→ how much gets written

Cell State
→ updated memory

Output Gate
→ what gets exposed

Hidden State
→ current visible representation
```

---

# ⭐ Final Mental Model

```text
OLD MEMORY
c_(t-1)
   │
   ▼
Forget Gate
   │
   ▼
KEEP
   │
   ├─────────────┐
   │             │
   │      Candidate g_t
   │             │
   │      × Input Gate
   │             │
   │          WRITE
   │             │
   └────── + ────┘
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

Remember:

```text
f_t → KEEP?
g_t → WHAT new content?
i_t → HOW MUCH to write?
c_t → MEMORY
o_t → HOW MUCH to expose?
h_t → EXPOSED STATE
```

---

# 🔗 Where This Leads

LSTM gives us powerful gating and a dedicated cell-state memory path.

But it also introduces:

* four learned transformations
* two recurrent states
* more parameters
* more computation

So the natural next question becomes:

> **Can we simplify this architecture while preserving most of the gating benefits?**

That leads to:

# `06_GRU`

---

# ⭐ Golden Rule

> **LSTM manages memory by keeping selected old information, writing selected candidate information, and exposing only the part currently needed — all while providing a more stable path for long-term information and gradients.**
