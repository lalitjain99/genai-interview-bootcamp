# 📘 Lecture — Gated Recurrent Unit (GRU)

## 🌟 Introduction

We just learned why LSTM was created.

Vanilla RNN had a major problem:

```text id="gru-intro-1"
Long Sequence
↓
Repeated Hidden-State Transformations
↓
Vanishing / Exploding Gradients
↓
Poor Long-Term Dependency Learning
```

LSTM improved this by introducing:

```text id="gru-intro-2"
Cell State
+
Forget Gate
+
Input Gate
+
Candidate Memory
+
Output Gate
```

That gave the network much better control over memory.

But LSTM introduced another issue:

> **It became relatively complex.**

At every time step, LSTM has to compute multiple gates and maintain:

```text id="gru-intro-3"
h_t
+
c_t
```

So researchers naturally asked:

# **Can we keep the benefits of gating, but make the recurrent unit simpler?**

Can we:

* use fewer gates?
* use fewer parameters?
* maintain only one recurrent state?
* still learn long-term dependencies reasonably well?

This leads us to:

# **GRU — Gated Recurrent Unit**

---

# 🎯 Learning Objectives

By the end of this lecture, you should be able to:

## 🧠 Conceptual Understanding

* Explain why GRU was introduced after LSTM.
* Understand how GRU simplifies LSTM.
* Explain the purpose of:

  * update gate
  * reset gate
  * candidate hidden state
* Explain why GRU does not use a separate cell state.
* Understand how GRU combines memory and hidden representation into one state.

## 📐 Mathematical Understanding

* Write the GRU equations.
* Understand the update-gate equation.
* Understand the reset-gate equation.
* Understand candidate hidden-state computation.
* Understand the final hidden-state interpolation.
* Explain how gradients can propagate through the direct hidden-state path.

## 🔗 Comparative Understanding

* Compare:

  * vanilla RNN
  * LSTM
  * GRU
* Understand which LSTM gates are conceptually simplified in GRU.
* Understand GRU's parameter and computational advantages.

## 🧮 Applied Understanding

* Trace a GRU numerically.
* Interpret high and low gate values.
* Understand how GRU preserves or replaces memory.
* Explain how GRU can work in streaming systems.

---

# 📖 Part 1 — What Problem Are We Solving Now?

LSTM already improved the long-term dependency problem.

So why do we need GRU?

Because LSTM has several moving parts:

```text id="gru-part1-1"
Previous Hidden State h_(t-1)
Previous Cell State c_(t-1)
Current Input x_t

↓

Forget Gate
Input Gate
Candidate Memory
Output Gate

↓

New Cell State c_t
New Hidden State h_t
```

This works well.

But it means:

* more parameters
* more matrix operations
* two recurrent states
* more implementation complexity

So the new question becomes:

> **Can we simplify the memory mechanism without returning to vanilla RNN's weaknesses?**

---

# 🧠 Part 2 — What Should We Keep From LSTM?

Before inventing GRU, let's identify the useful ideas from LSTM.

LSTM taught us that a good recurrent model should be able to:

### 1. Preserve useful old information

### 2. Forget outdated information

### 3. Add useful new information

### 4. Create easier gradient paths through time

So GRU keeps the idea of:

> **learned gating**

but simplifies the structure.

---

# ⭐ Part 3 — The Biggest Simplification

LSTM has:

```text id="gru-part3-1"
Cell State c_t
+
Hidden State h_t
```

GRU says:

> **Why maintain two separate recurrent states?**

Instead, GRU uses only:

# `h_t`

as its recurrent state.

So:

```text id="gru-part3-2"
LSTM
→ c_t + h_t

GRU
→ h_t only
```

This hidden state must now do both jobs:

* carry useful memory
* represent the current hidden representation

But unlike vanilla RNN, GRU does not blindly overwrite it.

It uses gates.

---

# 📖 Part 4 — What Decisions Does GRU Need?

If we only have one state:

`h_(t-1)`

we need to answer two major questions.

### Question 1

> How much of the old hidden state should we keep?

This leads to:

# **Update Gate**

### Question 2

> When creating new candidate information, how much of the old hidden state should we use?

This leads to:

# **Reset Gate**

These are the two primary GRU gates.

---

# ⭐ Part 5 — GRU Components

A standard GRU contains:

```text id="gru-part5-1"
Update Gate z_t
Reset Gate r_t
Candidate Hidden State h̃_t
Final Hidden State h_t
```

A simple mental model:

```text id="gru-part5-2"
Update Gate
→ How much old memory should survive?

Reset Gate
→ How much old context should influence the new candidate?

Candidate
→ What new state could replace the old one?

Final Hidden State
→ Blend old state and candidate
```

---

# 🔄 Part 6 — Inputs to a GRU

At time step `t`, GRU receives:

```text id="gru-part6-1"
x_t
→ current input

h_(t-1)
→ previous hidden state
```

It produces:

```text id="gru-part6-2"
h_t
→ new hidden state
```

There is:

> **no separate cell state `c_t`**

in the standard GRU.

---

# 🚪 Part 7 — Update Gate

The update gate answers:

> **How much of the existing hidden state should be preserved versus replaced by new information?**

A common equation is:

# `z_t = σ(W_z x_t + U_z h_(t-1) + b_z)`

or equivalently:

`z_t = σ(W_z [h_(t-1), x_t] + b_z)`

depending on notation.

Because it uses sigmoid:

`z_t ∈ [0,1]`

---

# 🧠 Part 8 — Update Gate Intuition

Suppose:

```text id="gru-part8-1"
z_t = 0.9
```

Conceptually, this means:

> preserve a large amount from one side of the old/new blend.

But here we must be careful.

There are **two common GRU conventions**.

One common form is:

`h_t = z_t ⊙ h_(t-1) + (1-z_t) ⊙ h̃_t`

In this convention:

```text id="gru-part8-2"
z_t ≈ 1
→ keep more OLD state

z_t ≈ 0
→ use more NEW candidate
```

Some books/framework descriptions swap the roles of `z_t` and `1-z_t`.

The architecture is conceptually the same.

For this lecture, we will use:

# `h_t = z_t ⊙ h_(t-1) + (1-z_t) ⊙ h̃_t`

So remember:

> **High `z_t` = preserve more old state**

---

# ⭐ Part 9 — Why Is It Called the Update Gate?

Because it controls:

> how much the hidden state should be updated with new information.

You can think of it as a blending controller:

```text id="gru-part9-1"
Old Hidden State
       │
       │ z_t
       ▼
     KEEP
       │
       ├─────────────┐
       │             │
       │       New Candidate
       │             │
       │      × (1-z_t)
       │             │
       └────── + ────┘
              │
              ▼
             h_t
```

---

# 🔁 Part 10 — Why Do We Need Another Gate?

Suppose we now want to create a new candidate state:

`h̃_t`

Should the new candidate always depend heavily on the entire previous hidden state?

Not necessarily.

Sometimes the model encounters something that signals:

> “Ignore much of the old context when creating the new representation.”

So GRU introduces another gate:

# **Reset Gate**

---

# 🚪 Part 11 — Reset Gate

The reset gate is:

# `r_t = σ(W_r x_t + U_r h_(t-1) + b_r)`

It decides:

> how much of the previous hidden state should influence the candidate hidden state.

So:

```text id="gru-part11-1"
r_t ≈ 1
→ use more previous context when constructing candidate

r_t ≈ 0
→ largely ignore previous context when constructing candidate
```

---

# 🧠 Part 12 — Reset Gate Intuition

Suppose the model has been processing one topic:

```text id="gru-part12-1"
The meeting is about the sales report...
```

Then suddenly:

```text id="gru-part12-2"
New topic: system outage...
```

To construct a representation of the new situation, the GRU may want:

> less influence from the previous context.

So:

```text id="gru-part12-3"
Reset Gate
↓
Reduce old hidden-state contribution
↓
Build candidate more strongly from current input
```

Important:

> Reset does not literally erase the final hidden state.

It controls:

> how previous hidden information is used when creating the candidate.

---

# ⚠️ Part 13 — Update Gate vs Reset Gate

This is the most important GRU distinction.

### Update Gate

Controls:

> old state vs new candidate in the final hidden state.

### Reset Gate

Controls:

> how much old state participates in creating the candidate.

So:

```text id="gru-part13-1"
Reset Gate
→ candidate construction

Update Gate
→ final old/new blending
```

Do not mix them.

---

# 🧠 Part 14 — Candidate Hidden State

Now we can build:

> proposed new hidden information.

A common equation is:

# `h̃_t = tanh(W_h x_t + U_h(r_t ⊙ h_(t-1)) + b_h)`

Read it carefully.

First:

`r_t ⊙ h_(t-1)`

means:

> filter the previous hidden state using the reset gate.

Then combine that with:

`x_t`

and create:

`h̃_t`

using `tanh`.

---

# 📖 Part 15 — Candidate Hidden State in Words

Read:

`h̃_t = tanh(W_h x_t + U_h(r_t ⊙ h_(t-1)) + b_h)`

as:

```text id="gru-part15-1"
Current Input
+
Selected Previous Context
↓
Learned Transformation
↓
tanh
↓
Proposed New Hidden State
```

So:

> `h̃_t` is not yet the final hidden state.

It is only:

> a candidate replacement/update.

---

# 🔥 Part 16 — Why `tanh` for Candidate?

Same core reasoning as LSTM candidate memory.

Candidate represents:

> content.

`tanh` outputs:

`[-1,1]`

so the candidate can contain:

* positive values
* negative values
* near-zero values

while keeping the representation bounded.

So again:

```text id="gru-part16-1"
Sigmoid
→ CONTROL

tanh
→ CONTENT
```

---

# ⭐ Part 17 — Final Hidden-State Update

Now we have:

* old state `h_(t-1)`
* candidate `h̃_t`
* update gate `z_t`

Final state:

# `h_t = z_t ⊙ h_(t-1) + (1-z_t) ⊙ h̃_t`

Read this as:

> **new hidden state = retained old state + accepted candidate state**

Or:

# `h_t = KEEP OLD + USE NEW`

---

# 🧠 Part 18 — This Equation Is the Heart of GRU

Compare:

LSTM:

`c_t = f_t ⊙ c_(t-1) + i_t ⊙ g_t`

GRU:

`h_t = z_t ⊙ h_(t-1) + (1-z_t) ⊙ h̃_t`

Notice the similarity.

Both use:

> **additive gated blending**

This creates a more direct path for:

* information
* gradients

through time.

---

# 🔗 Part 19 — Connection Between LSTM and GRU

Conceptually:

```text id="gru-part19-1"
LSTM

Forget Gate
+
Input Gate
↓
Control old-vs-new memory
```

GRU simplifies this idea with:

```text id="gru-part19-2"
Update Gate
↓
Control old-vs-new hidden state
```

This is not an exact one-to-one equivalence.

But conceptually, the GRU update gate combines much of the role of:

* retention
* writing

into one old-vs-new blending mechanism.

---

# 🧮 Part 20 — Simple Numerical Example

Suppose:

```text id="gru-part20-1"
h_(t-1) = 0.8

z_t = 0.9

h̃_t = 0.2
```

Then:

`h_t = 0.9×0.8 + 0.1×0.2`

`= 0.72 + 0.02`

# `= 0.74`

Interpretation:

> preserve most of the old state and incorporate only a little of the candidate.

---

# 🧮 Part 21 — Strong Update Example

Suppose:

```text id="gru-part21-1"
h_(t-1) = 0.8

z_t = 0.1

h̃_t = -0.4
```

Then:

`h_t = 0.1×0.8 + 0.9×(-0.4)`

`= 0.08 - 0.36`

# `= -0.28`

Now:

> most of the old state was replaced by the candidate.

---

# 🔍 Part 22 — What If `z_t = 1`?

Using our convention:

`h_t = z_t h_(t-1) + (1-z_t)h̃_t`

If:

`z_t = 1`

then:

# `h_t = h_(t-1)`

So:

```text id="gru-part22-1"
Keep old state
✅

Use candidate
❌
```

This gives GRU a nearly direct memory path.

---

# 🔍 Part 23 — What If `z_t = 0`?

Then:

# `h_t = h̃_t`

So:

> completely use the new candidate.

Conceptually:

```text id="gru-part23-1"
Old State
❌

Candidate
✅
```

---

# 🔍 Part 24 — What If `r_t = 0`?

Candidate equation:

`h̃_t = tanh(W_h x_t + U_h(r_t ⊙ h_(t-1)) + b_h)`

If:

`r_t = 0`

then:

`r_t ⊙ h_(t-1) = 0`

So candidate becomes approximately:

`h̃_t = tanh(W_h x_t + b_h)`

Meaning:

> previous hidden-state information is largely ignored while constructing the candidate.

---

# 🔍 Part 25 — What If `r_t = 1`?

Then:

`r_t ⊙ h_(t-1) = h_(t-1)`

So candidate uses:

> the full previous hidden-state representation.

---

# 🧠 Part 26 — Important Reset-Gate Insight

Reset gate affects:

# **candidate creation**

not directly:

# **final memory preservation**

This is extremely important.

For example:

even if:

`r_t = 0`

the old hidden state can still survive through:

`z_t ⊙ h_(t-1)`

if:

`z_t`

is high.

So:

```text id="gru-part26-1"
Reset = 0
does NOT mean
forget everything
```

---

# ⭐ Part 27 — Full GRU Equations

Using our convention:

### Update Gate

`z_t = σ(W_z x_t + U_z h_(t-1) + b_z)`

### Reset Gate

`r_t = σ(W_r x_t + U_r h_(t-1) + b_r)`

### Candidate Hidden State

`h̃_t = tanh(W_h x_t + U_h(r_t ⊙ h_(t-1)) + b_h)`

### Final Hidden State

# `h_t = z_t ⊙ h_(t-1) + (1-z_t) ⊙ h̃_t`

---

# ⚡ Part 28 — GRU in Four Questions

```text id="gru-part28-1"
r_t
→ How much OLD context should candidate creation use?

h̃_t
→ What NEW hidden state could I create?

z_t
→ How much OLD state should survive?

h_t
→ What is my final blended state?
```

---

# 🧠 Part 29 — GRU Memory Story

Imagine maintaining a working document.

You already have:

```text id="gru-part29-1"
Old Version
```

New information arrives.

First ask:

> Do I need the old document context when drafting the new version?

That is like:

`r_t`

Then you create:

> a proposed new version

That is:

`h̃_t`

Then ask:

> Should I mostly keep the old version or mostly adopt the new one?

That is:

`z_t`

Finally:

> blend them into the new state

That gives:

`h_t`

---

# 🔗 Part 30 — GRU vs Vanilla RNN

Vanilla RNN:

```text id="gru-part30-1"
Old State + Input
↓
Completely New Nonlinear State
```

GRU:

```text id="gru-part30-2"
Old State
──────────────┐
              │
              ├→ controlled blend → h_t
              │
Candidate
──────────────┘
```

So GRU gives the old state:

> a direct gated path into the new state.

This helps long-range information preservation.

---

# ⭐ Part 31 — Why GRU Helps Vanishing Gradients

Final state:

`h_t = z_t ⊙ h_(t-1) + (1-z_t) ⊙ h̃_t`

There is a direct path from:

`h_(t-1)`

to:

`h_t`

through:

`z_t`.

If:

`z_t ≈ 1`

then:

```text id="gru-part31-1"
h_(t-1)
────────────→
h_t
```

can be nearly preserved.

During backward propagation, this provides:

> a more favorable gradient route than vanilla RNN's repeated full nonlinear replacement.

---

# ⚠️ Part 32 — Does GRU Completely Eliminate Vanishing Gradients?

No.

Just like LSTM:

> GRU mitigates the problem.

It does not guarantee:

* perfect long-term memory
* gradient preservation forever
* stable training under all conditions

Gate values, sequence length, optimization, and parameter dynamics still matter.

---

# 🔗 Part 33 — LSTM vs GRU Memory Design

LSTM:

```text id="gru-part33-1"
c_t
→ primary memory path

h_t
→ exposed representation
```

GRU:

```text id="gru-part33-2"
h_t
→ memory + current representation combined
```

So GRU removes:

> the explicit separation between cell state and hidden state.

That is one of its biggest simplifications.

---

# 📊 Part 34 — LSTM vs GRU Gates

### LSTM

```text id="gru-part34-1"
Forget Gate
Input Gate
Output Gate
Candidate
```

### GRU

```text id="gru-part34-2"
Update Gate
Reset Gate
Candidate
```

No separate:

> output gate.

No separate:

> cell state.

---

# 🧠 Part 35 — Where Did the Output Gate Go?

In LSTM:

`o_t`

controls how much cell state becomes visible through:

`h_t`.

But GRU has:

> only one recurrent state.

So there is no separate internal cell memory that needs a separate exposure mechanism.

Therefore GRU has:

> no standard output gate.

---

# 🧠 Part 36 — Where Did Forget and Input Gates Go?

GRU does not literally have:

* forget gate
* input gate

Instead, update gate controls the balance between:

```text id="gru-part36-1"
Old Hidden State
vs
New Candidate
```

Using:

`z_t`

and:

`1-z_t`

So if one side gets more weight:

> the other automatically gets less.

This is a major simplification.

---

# ⭐ Part 37 — Why GRU Can Use One Gate for Old vs New

LSTM independently controls:

```text id="gru-part37-1"
How much old memory to retain
+
How much new content to write
```

GRU couples these decisions:

```text id="gru-part37-2"
More old
→ less candidate

Less old
→ more candidate
```

because:

`z_t + (1-z_t) = 1`

This makes GRU:

> simpler but slightly less independently flexible.

---

# 📐 Part 38 — Dimensions

Let:

* input dimension = `D`
* hidden dimension = `H`

Then:

```text id="gru-part38-1"
x_t ∈ R^D

h_(t-1) ∈ R^H

z_t ∈ R^H

r_t ∈ R^H

h̃_t ∈ R^H

h_t ∈ R^H
```

For separate matrix notation:

```text id="gru-part38-2"
W_z, W_r, W_h
→ H × D

U_z, U_r, U_h
→ H × H
```

Biases:

`H`

each.

---

# 🧮 Part 39 — GRU Parameter Count

There are three learned transformations:

```text id="gru-part39-1"
Update
Reset
Candidate
```

For each:

`HD + H² + H`

Therefore:

# `3(HD + H² + H)`

or:

# `3H(D + H + 1)`

under the simplified single-bias formulation.

---

# 🔗 Part 40 — Compare Parameter Count

Vanilla RNN:

# `H(D + H + 1)`

GRU:

# `3H(D + H + 1)`

LSTM:

# `4H(D + H + 1)`

Approximately:

```text id="gru-part40-1"
Vanilla RNN
→ 1×

GRU
→ 3×

LSTM
→ 4×
```

relative to the main recurrent transformation count.

Actual framework parameter counts can vary slightly because of implementation details such as multiple bias vectors.

---

# 🧮 Part 41 — Parameter Example

Suppose:

```text id="gru-part41-1"
D = 100
H = 256
```

Then:

`3 × 256 × (100 + 256 + 1)`

`= 3 × 256 × 357`

`= 3 × 91,392`

# `= 274,176`

Compare simplified LSTM:

`365,568`

So GRU uses fewer parameters.

---

# 🔥 Part 42 — Why GRU Can Be Faster

GRU typically computes:

```text id="gru-part42-1"
3 learned transformations
```

versus LSTM:

```text id="gru-part42-2"
4 learned transformations
```

And GRU maintains:

> one recurrent state instead of two conceptual states.

This can mean:

* fewer parameters
* lower compute
* potentially lower memory usage
* potentially faster training/inference

But actual speed depends on:

* hardware
* framework kernels
* sequence length
* batch size

---

# 🔗 Part 43 — GRU vs LSTM

| LSTM                         | GRU                     |
| ---------------------------- | ----------------------- |
| Cell state + hidden state    | Hidden state only       |
| Forget gate                  | No separate forget gate |
| Input gate                   | No separate input gate  |
| Output gate                  | No output gate          |
| Candidate memory             | Candidate hidden state  |
| More parameters              | Fewer parameters        |
| More flexible memory control | Simpler coupled control |
| Often higher compute         | Often lower compute     |

---

# 🧠 Part 44 — Is GRU Always Better Than LSTM?

No.

There is no universal winner.

GRU may work better when:

* dataset is smaller
* compute is constrained
* latency matters
* simpler architecture is preferred

LSTM may work better when:

* richer memory control is useful
* separate cell and hidden states help
* task benefits from independent forget/write/expose decisions

The best model is usually:

> an empirical choice.

---

# 📖 Part 45 — Example: Topic Change

Suppose sequence:

```text id="gru-part45-1"
The user was discussing football...
...
Now let's talk about quantum computing.
```

At the topic switch:

### Reset Gate

May become smaller for some dimensions:

```text id="gru-part45-2"
r_t ↓
```

so candidate creation relies less on old football context.

### Candidate

Creates new quantum-related representation.

### Update Gate

Controls how strongly the model transitions from:

> old context

to:

> new candidate.

---

# 🧠 Part 46 — Reset Gate Does Not Mean “Forget State”

This deserves repetition.

Suppose:

```text id="gru-part46-1"
r_t = 0
```

That only means:

> ignore old state while creating candidate.

But final state is:

`h_t = z_t h_(t-1) + (1-z_t)h̃_t`

If:

`z_t = 0.9`

then old state may still strongly survive.

Therefore:

> reset and update perform different jobs.

---

# ⚠️ Part 47 — Naming Convention Warning

You may see GRU final update written as:

`h_t = (1-z_t) ⊙ h_(t-1) + z_t ⊙ h̃_t`

instead of:

`h_t = z_t ⊙ h_(t-1) + (1-z_t) ⊙ h̃_t`

Both conventions exist.

The important concept is:

> one gate controls interpolation between old state and candidate state.

Always check:

> how the source defines `z_t`.

Do not memorize only:

> “high z means old” or “high z means new”

without checking the convention.

---

# 🎤 Part 48 — 30-Second Interview Answer

> **GRU, or Gated Recurrent Unit, is a gated recurrent architecture that simplifies LSTM. It uses a single hidden state instead of separate hidden and cell states, and mainly uses two gates: an update gate and a reset gate. The reset gate controls how much previous context is used when creating the candidate hidden state, while the update gate controls the blend between the previous hidden state and the new candidate. This gated additive update helps preserve information and gradients over longer sequences while typically using fewer parameters and less computation than LSTM.**

---

# 🎤 Part 49 — Update Gate Interview Answer

> **The update gate controls the trade-off between preserving the previous hidden state and incorporating the candidate hidden state. In the convention `h_t = z_t h_(t-1) + (1-z_t)h̃_t`, a value of `z_t` near 1 preserves more old state, while a value near 0 uses more of the new candidate.**

---

# 🎤 Part 50 — Reset Gate Interview Answer

> **The reset gate controls how much of the previous hidden state participates in constructing the candidate hidden state. A small reset-gate value reduces the influence of old context during candidate creation, while a large value allows more previous information to contribute.**

---

# 🎤 Part 51 — GRU vs LSTM Interview Answer

> **LSTM maintains separate cell and hidden states and uses forget, input, and output gates, whereas GRU uses a single hidden state and mainly update and reset gates. GRU couples old-memory retention and new-state writing through one update mechanism, making it simpler and typically cheaper, while LSTM provides more independent memory control.**

---

# ⚠️ Part 52 — High-Yield Interview Traps

### ❌ GRU has a cell state just like LSTM

No.

Standard GRU uses:

`h_t`

only.

---

### ❌ Reset gate directly deletes the old state

No.

It controls old-state influence during candidate creation.

---

### ❌ Update gate and reset gate do the same thing

No.

---

### ❌ Candidate is the final hidden state

No.

It is blended with previous hidden state.

---

### ❌ GRU completely solves vanishing gradients

No.

It mitigates the issue.

---

### ❌ GRU is always faster than LSTM

Not guaranteed.

Usually fewer operations/parameters, but real speed depends on implementation.

---

### ❌ GRU is always better than LSTM

No.

Task-dependent.

---

### ❌ High update gate always means new information

Not under every notation.

Check the equation convention.

---

# 🔄 Part 53 — GRU Is Still Trained With BPTT

Just like:

* vanilla RNN
* LSTM

GRU is recurrent.

Therefore training still uses:

> **Backpropagation Through Time**

```text id="gru-part53-1"
GRU Forward
↓
Loss
↓
BPTT
↓
Gradients Through Gates
↓
Optimizer
```

GRU improves the recurrent state pathway.

It does not replace BPTT.

---

# ⏳ Part 54 — GRU Is Still Sequential Across Time

At time `t`, GRU needs:

`h_(t-1)`

So:

```text id="gru-part54-1"
h_1
↓
h_2
↓
h_3
↓
...
```

must still be processed sequentially across time.

So GRU does not solve:

> recurrent parallelization limitations.

---

# 🌊 Part 55 — Streaming Strength

Like RNN and LSTM, GRU is naturally suitable for streaming.

Maintain:

`h_t`

Then when:

`x_(t+1)`

arrives:

```text id="gru-part55-1"
x_(t+1)
+
h_t
↓
GRU
↓
h_(t+1)
```

Compared with LSTM, state management is simpler because we maintain:

> one recurrent state instead of `h_t + c_t`.

---

# 👥 Part 56 — Production State Management

Model weights can be shared across users.

Hidden states cannot.

```text id="gru-part56-1"
Shared GRU Weights

User A → h_A
User B → h_B
User C → h_C
```

Each independent sequence needs:

> its own hidden state.

Otherwise:

* context leakage
* incorrect predictions
* sequence contamination

can occur.

---

# 🔗 Part 57 — Evolution So Far

We can now see the architecture evolution clearly.

```text id="gru-part57-1"
Vanilla RNN
↓
Simple recurrence
↓
Long-term dependency problem
```

Then:

```text id="gru-part57-2"
LSTM
↓
Cell State + Multiple Gates
↓
Better long-term memory
↓
More complexity
```

Then:

```text id="gru-part57-3"
GRU
↓
Simplified gating
↓
Single recurrent state
↓
Fewer parameters
```

---

# 🚀 Part 58 — But We Still Have Another Problem

Both LSTM and GRU improve:

> how information flows through time.

But both still process a sequence like:

```text id="gru-part58-1"
x1
↓
x2
↓
x3
↓
x4
```

One direction:

> past → future

What if the prediction at time `t` needs:

> information from both earlier and later tokens?

Example:

```text id="gru-part58-2"
He went to the bank to deposit money.
```

The meaning of:

`bank`

becomes clearer from later words:

`deposit money`

A normal left-to-right recurrent network does not see future context at that time step.

So the next question becomes:

# **Can an RNN process the sequence in both directions?**

That leads to:

# `07_Bidirectional_RNN`

---

# 📐 Formula Flash Card

### Update Gate

`z_t = σ(W_z x_t + U_z h_(t-1) + b_z)`

### Reset Gate

`r_t = σ(W_r x_t + U_r h_(t-1) + b_r)`

### Candidate

`h̃_t = tanh(W_h x_t + U_h(r_t ⊙ h_(t-1)) + b_h)`

### Final Hidden State

# `h_t = z_t ⊙ h_(t-1) + (1-z_t) ⊙ h̃_t`

Using the convention adopted in this lecture.

### Parameters

# `3H(D + H + 1)`

under the simplified formulation.

---

# 🧠 Final Mental Model

```text id="gru-final-1"
Current Input x_t
+
Previous Hidden h_(t-1)
       │
       ├───────────────→ Update Gate z_t
       │
       │
       └───────────────→ Reset Gate r_t
                              │
                              ▼
                    Filter Old Context
                              │
                              +
                             x_t
                              │
                              ▼
                       Candidate h̃_t
```

Then:

```text id="gru-final-2"
Previous Hidden h_(t-1)
       │
       │ × z_t
       ▼
      KEEP
       │
       ├─────────────┐
       │             │
       │      Candidate h̃_t
       │             │
       │      × (1-z_t)
       │             │
       └────── + ────┘
              │
              ▼
             h_t
```

Remember:

```text id="gru-final-3"
Reset Gate
→ How much OLD context helps build NEW candidate?

Candidate
→ What NEW state could I create?

Update Gate
→ How much OLD vs NEW should survive?

Hidden State
→ Final blended memory/representation
```

---

# ⭐ Golden Rule

> **GRU simplifies LSTM by using a single hidden state and two main gates: the reset gate controls how much old context is used to create the candidate, while the update gate controls how much of the old state versus the new candidate becomes the next hidden state.**
