# 💡 Assignment — Long Short-Term Memory (LSTM)

> **Goal:** Apply LSTM concepts through state tracing, gate reasoning, numerical updates, debugging, and practical architecture decisions.
> Focus on **using the mechanism**, not repeating interview definitions.

---

# 🧩 Task 1 — Trace the LSTM Inputs and Outputs

At time step `t`, an LSTM receives:

```text
x_t
h_(t-1)
c_(t-1)
```

### Questions

1. Which of these represents the current input?
2. Which carries the previous exposed representation?
3. Which carries the previous controlled memory?
4. What two recurrent states does the LSTM produce?

### ✅ Answer

```text
x_t
→ current input
```

```text
h_(t-1)
→ previous hidden/exposed state
```

```text
c_(t-1)
→ previous cell-state memory
```

The LSTM produces:

```text
h_t
→ new hidden state
```

and:

```text
c_t
→ new cell state
```

So:

```text
x_t + h_(t-1) + c_(t-1)
↓
LSTM
↓
h_t + c_t
```

---

# 🧠 Task 2 — Why Does the Forget Gate Need `x_t`?

Suppose the cell state currently represents information similar to:

```text
Current city = London
```

Now the model receives:

```text
x_t = "He moved to Paris"
```

### Question

Why should the forget gate look at the current input `x_t` when deciding what old information to retain?

### ✅ Answer

Because whether old memory is still relevant depends on:

> what new information has just arrived.

The forget gate computes:

`f_t = σ(W_f[h_(t-1), x_t] + b_f)`

Conceptually:

```text
Previous context
+
Current input
↓
Decide whether old memory is still useful
```

The current input `"moved to Paris"` provides evidence that:

> the old London-related memory may need to be weakened.

So:

```text
x_t + h_(t-1)
→ decide HOW MUCH to forget

c_(t-1)
→ actual memory being filtered
```

---

# 🔐 Task 3 — Apply a Forget Gate

Suppose:

```text
c_(t-1) = [0.8, -0.4, 0.6]
```

and:

```text
f_t = [1.0, 0.25, 0.0]
```

### Question

What is the retained old-memory contribution?

### ✅ Answer

Compute:

`f_t ⊙ c_(t-1)`

So:

```text
[1.0×0.8,
 0.25×(-0.4),
 0×0.6]
```

Result:

```text
[0.8, -0.1, 0]
```

Interpretation:

* first memory dimension fully retained
* second mostly forgotten
* third completely removed

---

# ✍️ Task 4 — Candidate vs Input Gate

Suppose:

```text
g_t = [0.7, -0.8, 0.2]
```

and:

```text
i_t = [1.0, 0.5, 0.0]
```

### Questions

1. What does `g_t` represent?
2. What does `i_t` represent?
3. What new information is actually written?

### ✅ Answer

### Candidate `g_t`

Represents:

> proposed new memory content.

### Input Gate `i_t`

Represents:

> how much of each candidate component should be written.

Actual write:

`i_t ⊙ g_t`

```text
[1×0.7,
 0.5×(-0.8),
 0×0.2]
```

Result:

```text
[0.7, -0.4, 0]
```

Remember:

```text
Candidate
→ WHAT

Input Gate
→ HOW MUCH
```

---

# 🧮 Task 5 — Compute the New Cell State

Suppose:

```text
c_(t-1) = [0.6, 0.2]
f_t     = [0.8, 0.5]
i_t     = [0.7, 1.0]
g_t     = [0.4, -0.3]
```

### Question

Calculate:

`c_t`

### ✅ Answer

Cell-state update:

`c_t = f_t ⊙ c_(t-1) + i_t ⊙ g_t`

First, retained old memory:

```text
f_t ⊙ c_(t-1)
=
[0.8×0.6,
 0.5×0.2]
=
[0.48, 0.10]
```

New write:

```text
i_t ⊙ g_t
=
[0.7×0.4,
 1.0×(-0.3)]
=
[0.28, -0.30]
```

Add them:

```text
c_t
=
[0.48 + 0.28,
 0.10 - 0.30]
```

# `c_t = [0.76, -0.20]`

---

# 🧠 Task 6 — Interpret the Cell-State Equation

Given:

`c_t = f_t ⊙ c_(t-1) + i_t ⊙ g_t`

### Question

Explain this equation without using mathematical terminology.

### ✅ Answer

A good explanation is:

> **Take the old memory, keep only the parts still useful, then add the useful parts of the new candidate information.**

Or:

```text
New Memory
=
Old Memory I Keep
+
New Information I Write
```

Shortcut:

# `MEMORY = KEEP + WRITE`

---

# 🔍 Task 7 — What If `f_t = 1` and `i_t = 0`?

Suppose:

```text
f_t = 1
i_t = 0
```

for a particular cell dimension.

### Question

What happens to the cell state?

### ✅ Answer

Cell update:

`c_t = 1×c_(t-1) + 0×g_t`

Therefore:

# `c_t = c_(t-1)`

So:

```text
Keep old memory
✅

Write new information
❌
```

This creates an almost direct memory path.

---

# 🔍 Task 8 — What If `f_t = 0` and `i_t = 1`?

### Question

What happens?

### ✅ Answer

Cell update:

`c_t = 0×c_(t-1) + 1×g_t`

Therefore:

# `c_t = g_t`

Conceptually:

```text
Forget old memory
+
Fully accept new candidate
```

This is effectively:

> replacing that memory component with new candidate information.

---

# 🔍 Task 9 — What If the Output Gate Is Zero?

Suppose:

```text
o_t = 0
```

but:

```text
c_t ≠ 0
```

### Questions

1. What happens to `h_t`?
2. Is the cell-state memory deleted?

### ✅ Answer

Hidden state:

`h_t = o_t ⊙ tanh(c_t)`

So:

`h_t = 0`

for that component.

But:

> the cell state itself can still retain the information.

Therefore:

```text
Not exposed
≠
Forgotten
```

This distinction is fundamental in LSTM.

---

# 🧠 Task 10 — Why `tanh` for Candidate Memory?

A developer proposes replacing candidate memory with:

```text
g_t = sigmoid(...)
```

### Question

Why is `tanh` generally more suitable for candidate content?

### ✅ Answer

Candidate memory represents:

> content

rather than:

> a gate.

`tanh` outputs:

`[-1, 1]`

which allows signed candidate information.

The candidate can contribute:

* positively
* negatively
* approximately zero

Sigmoid only gives:

`[0,1]`

which is better suited to:

> controlling how much information passes.

Shortcut:

```text
sigmoid
→ CONTROL

tanh
→ CONTENT
```

---

# 🧠 Task 11 — Diagnose an Incorrect LSTM Explanation

A developer says:

> “The input gate generates the new information, and candidate memory decides how much of it to store.”

### Question

What is wrong?

### ✅ Answer

The roles are reversed.

Correct:

```text
Candidate g_t
→ generates/proposes WHAT new content could be stored
```

```text
Input Gate i_t
→ decides HOW MUCH of that content should be written
```

So:

`i_t ⊙ g_t`

is the actual new-memory contribution.

---

# 🧩 Task 12 — Hidden State vs Cell State

Consider a sequence-processing system.

At time `t`, the LSTM has:

```text
c_t
h_t
```

### Match each statement.

### A.

“Contains the main controlled memory pathway.”

### B.

“Represents what the LSTM is currently exposing to the next computation/output.”

### C.

“Can preserve information even when the output gate suppresses it.”

### D.

“Is computed as `o_t ⊙ tanh(c_t)`.”

### ✅ Answer

```text
A → c_t
B → h_t
C → c_t
D → h_t
```

---

# ⚠️ Task 13 — Fix the “Short-Term Memory” Claim

A developer says:

> “The cell state stores all long-term information, while hidden state stores only the previous 2–3 states.”

### Is this correct?

### ✅ Answer

No.

The first part is a useful rough intuition, but the second part is incorrect.

The hidden state does not literally store:

> the last few hidden states.

A better statement is:

> **The cell state provides the primary controlled memory pathway, while the hidden state is the current exposed representation derived from the cell state and current gating decisions.**

---

# 📖 Task 14 — Story-Based Gate Reasoning

Suppose the model processes:

```text
Alice lives in London.
...
Alice moved to Berlin.
```

At the second sentence, what should ideally happen to a memory dimension representing Alice's current city?

### ✅ Answer

Conceptually:

### Forget gate

Should reduce the old `"London"` representation.

```text
f_t → relatively small
```

### Candidate

Should propose information related to:

```text
Berlin
```

### Input gate

Should allow that useful new candidate to be written.

```text
i_t → relatively high
```

Then:

```text
Old London memory
↓ weaken

New Berlin candidate
↓ write

Updated cell state
→ Berlin-related representation
```

The exact gate values are learned, not manually specified.

---

# ⭐ Task 15 — Why Does LSTM Help Vanishing Gradients?

Suppose information should survive:

`100`

steps.

### Question

Why can the LSTM cell-state path help compared with vanilla RNN?

### ✅ Answer

Vanilla RNN repeatedly propagates gradient through terms involving:

```text
W_hh
×
activation derivative
```

LSTM cell state has:

`c_t = f_t ⊙ c_(t-1) + ...`

Along the direct memory path:

`∂c_t/∂c_(t-1)`

contains:

`f_t`.

If relevant forget-gate values remain near:

`1`:

```text
1 × 1 × 1 × ...
```

the gradient can survive much more effectively.

So LSTM creates:

> a more favorable long-term gradient path.

---

# 🔢 Task 16 — Compare Gradient Preservation

Consider a simplified cell-state path.

### Model A

Forget gate remains:

`f_t = 0.99`

for `100` steps.

### Model B

Forget gate remains:

`f_t = 0.5`

for `100` steps.

### Question

Which model preserves the direct cell-state gradient better?

### ✅ Answer

Model A.

Approximate path multiplier:

### Model A

`0.99^100 ≈ 0.366`

A substantial part remains.

### Model B

`0.5^100`

is effectively:

> almost zero.

So LSTM can preserve long-term gradients when the model learns:

> forget gates close to `1` for important information.

---

# ⚠️ Task 17 — Does LSTM Eliminate Vanishing Gradients?

A developer says:

> “LSTM cannot have vanishing gradients because it has a cell state.”

### Correct or incorrect?

### ✅ Answer

Incorrect.

LSTM:

> mitigates the problem

but does not guarantee perfect gradient preservation.

If forget gates are repeatedly small:

```text
f_t × f_(t-1) × ...
```

can still shrink dramatically.

Other paths and nonlinearities can also experience gradient problems.

Better statement:

> **LSTM provides a more stable and controllable gradient pathway than vanilla RNN.**

---

# 🔗 Task 18 — Connect LSTM to Residual Connections

What common design principle do LSTM cell-state updates and residual connections share?

### ✅ Answer

Residual connection:

`y = x + F(x)`

Cell-state update:

`c_t = retained old memory + new contribution`

Both use an:

> additive route

that allows information and gradients to avoid being forced entirely through a new nonlinear transformation at every step.

They are not the same architecture.

But they share the principle:

> **create easier information and gradient pathways.**

---

# 📐 Task 19 — Matrix Shapes

Suppose:

```text
Input dimension D = 64
Hidden dimension H = 128
```

### Questions

1. What is the size of `[h_(t-1), x_t]`?
2. What is the shape of `W_f`?
3. What is the shape of `W_i`?
4. What is the shape of `W_g`?
5. What is the shape of `W_o`?

### ✅ Answer

Concatenated dimension:

`H + D`

`= 128 + 64`

`= 192`

Each gate outputs:

`H = 128`

values.

Therefore:

```text
W_f = 128 × 192
W_i = 128 × 192
W_g = 128 × 192
W_o = 128 × 192
```

---

# 🧮 Task 20 — Parameter Count

Given:

```text
D = 64
H = 128
```

calculate the LSTM parameter count excluding the downstream output layer.

### ✅ Answer

Formula:

`4H(H + D + 1)`

Substitute:

`4 × 128 × (128 + 64 + 1)`

`= 4 × 128 × 193`

# `= 98,816`

---

# 🔧 Task 21 — Debug a Memory Leak Between Users

A production system uses one LSTM fraud model for many customers.

The system shares:

```text
h_t
c_t
```

between all users.

### Question

What is wrong?

### ✅ Answer

Model parameters can be shared across users solving the same task.

But recurrent state must be:

> sequence-specific.

Correct:

```text
Shared:
LSTM weights

Separate:
User A → h_A, c_A
User B → h_B, c_B
User C → h_C, c_C
```

Sharing states can cause:

* context contamination
* incorrect predictions
* data leakage

---

# 🌊 Task 22 — Streaming Inference

A sensor sends a new observation every second.

At time:

`t = 100`

the system already has:

```text
h_100
c_100
```

When `x_101` arrives, does it need to replay all 100 previous observations?

### ✅ Answer

No.

It can compute:

```text
x_101
+
h_100
+
c_100
↓
LSTM
↓
h_101, c_101
```

This makes LSTM naturally useful for:

> streaming inference.

---

# ⚠️ Task 23 — Does LSTM Solve Time Parallelism?

Suppose you have:

```text
x1, x2, x3, ..., x100
```

Can all LSTM time steps be fully computed simultaneously?

### ✅ Answer

No.

Time `t` requires:

```text
h_(t-1)
c_(t-1)
```

So:

```text
t1 → t2 → t3 → ...
```

remains sequential.

LSTM solves:

> memory and gradient-flow problems better

but not:

> recurrent time-step dependency.

---

# 🔍 Task 24 — Find the Incorrect Statement

Which statement is incorrect?

### A.

Candidate memory proposes new content.

### B.

Forget gate controls old-memory retention.

### C.

Output gate deletes information from the cell state.

### D.

Input gate controls how much candidate enters memory.

### ✅ Answer

❌ **C is incorrect.**

Output gate controls:

> exposure through `h_t`

not:

> deletion from `c_t`.

---

# 🧠 Task 25 — Gate or Content?

Classify each as either:

> CONTROL

or:

> CONTENT

### `f_t`

### `i_t`

### `g_t`

### `o_t`

### ✅ Answer

```text
f_t → CONTROL
i_t → CONTROL
g_t → CONTENT
o_t → CONTROL
```

This is why:

```text
f_t, i_t, o_t
→ sigmoid

g_t
→ tanh
```

---

# ✅ True / False — Reasoning Traps

| # | Statement | Answer |
|---|---|
| 1 | LSTM introduces a cell state in addition to hidden state. | ✅ True |
| 2 | The hidden state literally stores the previous few hidden states. | ❌ False |
| 3 | Candidate memory is proposed new content. | ✅ True |
| 4 | Input gate decides how much candidate content is written. | ✅ True |
| 5 | Forget gate usually considers current input when deciding what old memory to retain. | ✅ True |
| 6 | Forget gate directly receives `c_(t-1)` in the standard formulation used here. | ❌ False |
| 7 | `c_(t-1)` is the memory actually filtered by the forget gate. | ✅ True |
| 8 | Output gate controls what is exposed from the updated memory. | ✅ True |
| 9 | Output gate deletes information from `c_t`. | ❌ False |
| 10 | LSTM gates are normally binary. | ❌ False |
| 11 | Candidate memory commonly uses `tanh`. | ✅ True |
| 12 | Sigmoid is useful for gates because its values lie between 0 and 1. | ✅ True |
| 13 | LSTM completely eliminates vanishing gradients. | ❌ False |
| 14 | LSTM is still trained using BPTT. | ✅ True |
| 15 | LSTM removes sequential dependency across time. | ❌ False |

---

# 🧮 Calculation Challenge

Consider one scalar LSTM memory dimension.

Given:

```text
c_(t-1) = 0.7
f_t     = 0.8
i_t     = 0.6
g_t     = -0.5
o_t     = 0.9
```

### Question 1 — Retained Old Memory

### ✅ Answer

`f_t × c_(t-1)`

`= 0.8 × 0.7`

# `= 0.56`

---

### Question 2 — New Memory Contribution

### ✅ Answer

`i_t × g_t`

`= 0.6 × -0.5`

# `= -0.30`

---

### Question 3 — Updated Cell State

### ✅ Answer

`c_t = 0.56 - 0.30`

# `= 0.26`

---

### Question 4 — Hidden State

`h_t = o_t × tanh(c_t)`

`tanh(0.26) ≈ 0.254`

Therefore:

`h_t ≈ 0.9 × 0.254`

# `≈ 0.229`

---

# ⭐ Staff Engineer Challenge — Stateful LSTM Service

## Scenario

You are building an LSTM-based real-time fraud engine.

There are:

`3 million active customers`

Each customer generates an independent transaction sequence.

The system must maintain:

```text
h_t
c_t
```

for each active customer.

Events can:

* arrive late
* arrive twice
* arrive out of order
* be processed by different application replicas

---

## Question 1 — What Is Shared and What Is Per User?

### ✅ Answer

Shared globally:

> trained LSTM parameters.

Per user:

```text
h_t
c_t
```

So:

```text
Global Model
↓
same weights

Customer A → h_A, c_A
Customer B → h_B, c_B
Customer C → h_C, c_C
```

---

## Question 2 — Why Does Event Ordering Matter?

### ✅ Answer

LSTM state evolves recursively.

```text
Event A
↓
Update h,c
↓
Event B
↓
Update h,c
```

If processing order changes:

```text
B → A
```

the resulting:

`h_t, c_t`

trajectory can be different.

Therefore ordered sequence processing matters.

---

## Question 3 — What Problem Can Duplicate Events Cause?

### ✅ Answer

If the same event is processed twice:

```text
Event
↓
Update state

Same Event Again
↓
Update state again
```

the memory may be modified twice.

This can create incorrect state.

A production system may need:

* event IDs
* deduplication
* idempotency handling

---

## Question 4 — What Happens If Two Replicas Update the Same User Concurrently?

### ✅ Answer

Both may read the same old:

```text
h_t, c_t
```

then independently create conflicting next states.

This creates:

> race conditions / lost state updates.

The architecture may require:

* per-user ordering
* partitioning by user
* locking or optimistic concurrency
* single-writer semantics

depending on the system.

---

## Question 5 — Should We Store LSTM State Forever?

### ✅ Answer

Not necessarily.

For millions of users, state retention has infrastructure cost.

We need policies for:

* inactive-user expiration
* persistence
* recovery
* serialization
* state versioning
* model-version compatibility

So recurrent-model choice creates:

> distributed state-management requirements.

---

## Question 6 — When Might GRU Be Attractive?

### ✅ Answer

If LSTM accuracy is good but:

* inference latency is high
* state size matters
* compute is constrained

GRU may be attractive because it simplifies the gated architecture.

It uses:

* fewer gates
* one recurrent state

while retaining much of the benefit of gated recurrence.

This motivates our next lecture.

---

# 🎯 Final Reasoning Exercise

Complete the chain:

```text
Vanilla RNN
↓
Long-term information and gradient problems
↓
Need a __________ memory path
↓
LSTM introduces __________
↓
Old memory retention is controlled by __________
↓
New content is proposed by __________
↓
How much new content is written is controlled by __________
↓
Memory update = __________ + __________
↓
What gets exposed is controlled by __________
↓
Exposed representation = __________
```

### ✅ Answer

```text
Vanilla RNN
↓
Long-term information and gradient problems
↓
Need a CONTROLLED / MORE STABLE memory path
↓
LSTM introduces CELL STATE
↓
Old memory retention is controlled by FORGET GATE
↓
New content is proposed by CANDIDATE MEMORY
↓
How much new content is written is controlled by INPUT GATE
↓
Memory update = KEEP + WRITE
↓
What gets exposed is controlled by OUTPUT GATE
↓
Exposed representation = HIDDEN STATE
```

---

# 🧠 Final Assignment Mental Model

```text
Previous Memory c_(t-1)
↓
Forget Gate
↓
KEEP
```

At the same time:

```text
Current Input x_t
+
Previous Hidden h_(t-1)
↓
Candidate
→ WHAT new content?

Input Gate
→ HOW MUCH to accept?
↓
WRITE
```

Then:

```text
KEEP
+
WRITE
↓
c_t
```

Finally:

```text
c_t
↓
tanh
↓
Output Gate
↓
h_t
```

---

# ⭐ Golden Rule

> **The easiest way to reason about an LSTM is to separate three jobs: decide what old memory remains, decide what new candidate content gets written, and decide what part of the resulting memory is exposed right now.**
