# 📝 Revision — Gated Recurrent Unit (GRU)

> **Goal:** Quickly revise why GRU was introduced, how it simplifies LSTM, how the update and reset gates work, how the candidate hidden state is formed, and how GRU preserves long-term information with fewer parameters.

---

# 🌟 1. Why Was GRU Introduced?

LSTM improved vanilla RNN by introducing:

```text
Cell State
+
Forget Gate
+
Input Gate
+
Output Gate
+
Candidate Memory
```

This gave much better memory control.

But LSTM also became:

* more complex
* more parameter-heavy
* more computationally expensive
* dependent on two recurrent states: `h_t` and `c_t`

So the next question was:

> **Can we keep the benefits of gating while simplifying the recurrent architecture?**

That led to:

# **GRU — Gated Recurrent Unit**

---

# 🧠 2. Core Idea of GRU

GRU keeps the idea of:

> learned gates

but simplifies LSTM.

Instead of:

```text
LSTM
→ h_t + c_t
```

GRU uses only:

# `h_t`

So:

```text
GRU
→ one recurrent state
→ hidden state = memory + current representation
```

There is:

> no separate cell state in standard GRU.

---

# ⭐ 3. Main Components of GRU

A GRU mainly contains:

```text
Update Gate z_t
Reset Gate r_t
Candidate Hidden State h̃_t
Final Hidden State h_t
```

Shortcut:

```text
Reset Gate
→ how much old context helps build candidate?

Candidate
→ what new hidden state could be created?

Update Gate
→ how much old state vs candidate should survive?

Hidden State
→ final blended state
```

---

# 🔄 4. Inputs and Output

At time step `t`, GRU receives:

```text
x_t
→ current input

h_(t-1)
→ previous hidden state
```

It produces:

```text
h_t
→ new hidden state
```

Unlike LSTM, there is no:

`c_t`

---

# 🚪 5. Update Gate

Equation:

`z_t = σ(W_z x_t + U_z h_(t-1) + b_z)`

The update gate controls:

> the balance between old hidden state and new candidate hidden state.

In this revision, we use:

# `h_t = z_t ⊙ h_(t-1) + (1-z_t) ⊙ h̃_t`

Therefore:

```text
z_t ≈ 1
→ keep more OLD state

z_t ≈ 0
→ use more NEW candidate
```

---

# ⚠️ 6. Update-Gate Convention Warning

Some books use:

`h_t = (1-z_t) ⊙ h_(t-1) + z_t ⊙ h̃_t`

So in those sources:

```text
z_t ≈ 1
→ more NEW state
```

Therefore:

> **Always check the equation before interpreting `z_t`.**

Do not memorize:

> high `z_t` always means old

without checking the notation.

---

# 🧠 7. Update Gate Intuition

Think of the update gate as:

> an old-vs-new blending controller.

Example:

```text
Old state = 0.8
Candidate = 0.2
z_t = 0.9
```

Then:

`h_t = 0.9×0.8 + 0.1×0.2`

`= 0.72 + 0.02`

# `= 0.74`

So GRU keeps mostly the old state.

---

# 🔁 8. Reset Gate

Equation:

`r_t = σ(W_r x_t + U_r h_(t-1) + b_r)`

The reset gate controls:

> how much of the previous hidden state should influence candidate creation.

Interpretation:

```text
r_t ≈ 1
→ use more old context

r_t ≈ 0
→ ignore much of old context
```

---

# 🧠 9. Reset Gate Is NOT a Forget Gate

This is a very important distinction.

Reset gate does not directly decide:

> whether the old hidden state survives in `h_t`.

It decides:

> how much old hidden-state information is used while constructing the candidate.

So:

```text
Reset Gate
→ candidate construction

Update Gate
→ final old/new blending
```

---

# 📝 10. Candidate Hidden State

Equation:

`h̃_t = tanh(W_h x_t + U_h(r_t ⊙ h_(t-1)) + b_h)`

Read it as:

```text
Current Input
+
Selected Previous Context
↓
Learned Transformation
↓
tanh
↓
Candidate Hidden State
```

The candidate is:

> proposed new hidden content.

It is not yet:

> the final hidden state.

---

# 🔥 11. Why `tanh` for Candidate?

Same reasoning as LSTM candidate memory.

Candidate represents:

> CONTENT

not:

> CONTROL

`tanh` gives:

`[-1,1]`

so candidate values can be:

* positive
* negative
* near zero

and remain bounded.

Shortcut:

```text
sigmoid
→ CONTROL

tanh
→ CONTENT
```

---

# ⭐ 12. Final GRU Hidden-State Equation

# `h_t = z_t ⊙ h_(t-1) + (1-z_t) ⊙ h̃_t`

Read it as:

```text
New Hidden State
=
Old State I Keep
+
New Candidate I Accept
```

Or simply:

# `h_t = KEEP OLD + USE NEW`

---

# 🧠 13. Why This Equation Is Important

Vanilla RNN:

```text
Old State
↓
Complete Nonlinear Transformation
↓
New State
```

GRU:

```text
Old State
───────────────┐
               │
               ├→ Blend → h_t
               │
Candidate
───────────────┘
```

So the previous state has:

> a more direct route into the next state.

This helps:

* information preservation
* gradient flow
* long-term dependency learning

---

# 🔍 14. What If `z_t = 1`?

Using our convention:

`h_t = h_(t-1)`

So:

```text
Keep old state
✅

Use candidate
❌
```

This creates a direct memory path.

---

# 🔍 15. What If `z_t = 0`?

Then:

`h_t = h̃_t`

So:

```text
Old state
❌

Candidate
✅
```

The hidden state is replaced by the candidate.

---

# 🔍 16. What If `r_t = 0`?

Candidate equation:

`h̃_t = tanh(W_h x_t + U_h(r_t ⊙ h_(t-1)) + b_h)`

If:

`r_t = 0`

then:

`r_t ⊙ h_(t-1) = 0`

So candidate creation largely ignores:

> previous hidden context.

But this does **not** mean old state disappears from final `h_t`.

If `z_t` is high, old state can still survive.

---

# 🔍 17. What If `r_t = 1`?

Then:

`r_t ⊙ h_(t-1) = h_(t-1)`

So candidate creation uses:

> the full previous hidden representation.

---

# ⚠️ 18. Reset Gate = 0 Does NOT Mean “Forget Everything”

This is a common mistake.

Even if:

`r_t = 0`

final hidden state is still:

`h_t = z_t h_(t-1) + (1-z_t)h̃_t`

So if:

`z_t = 0.9`

then the old hidden state still strongly contributes.

Therefore:

```text
Reset Gate
≠
Final Retention Gate
```

---

# ⭐ 19. Full GRU Equations

### Update Gate

`z_t = σ(W_z x_t + U_z h_(t-1) + b_z)`

### Reset Gate

`r_t = σ(W_r x_t + U_r h_(t-1) + b_r)`

### Candidate Hidden State

`h̃_t = tanh(W_h x_t + U_h(r_t ⊙ h_(t-1)) + b_h)`

### Final Hidden State

# `h_t = z_t ⊙ h_(t-1) + (1-z_t) ⊙ h̃_t`

---

# ⚡ 20. Four-Question Mental Model

```text
r_t
→ How much old context should candidate creation use?

h̃_t
→ What new state could I create?

z_t
→ How much old state should survive?

h_t
→ What is the final blended state?
```

---

# 🔗 21. GRU vs LSTM Core Difference

### LSTM

```text
c_t
→ memory

h_t
→ exposed representation
```

### GRU

```text
h_t
→ memory + representation combined
```

So GRU removes:

> the explicit cell-state / hidden-state separation.

---

# 🔗 22. Where Did LSTM's Forget and Input Gates Go?

LSTM separately controls:

```text
Forget old memory
+
Write new memory
```

GRU couples these decisions through:

`z_t`

and:

`1-z_t`

So:

```text
More OLD
→ Less NEW

Less OLD
→ More NEW
```

This makes GRU simpler.

---

# 🧠 23. Why Is This a Simplification?

LSTM can independently choose:

```text
retain 90% old
+
write 60% new
```

GRU's update mechanism couples the two weights:

```text
z_t + (1-z_t) = 1
```

So if one increases:

> the other decreases.

This gives:

* fewer gates
* fewer parameters
* simpler control

but also:

> slightly less independent flexibility.

---

# 🧠 24. Where Did the Output Gate Go?

LSTM has separate:

```text
Cell State
+
Hidden State
```

so it needs an output gate to decide:

> what part of cell memory should be exposed.

GRU has only:

`h_t`

So there is no separate internal cell memory that requires:

> an exposure gate.

Therefore standard GRU has:

> no output gate.

---

# 🔗 25. LSTM vs GRU Gates

| LSTM                     | GRU                     |
| ------------------------ | ----------------------- |
| Forget Gate              | No separate forget gate |
| Input Gate               | No separate input gate  |
| Output Gate              | No output gate          |
| Candidate Memory         | Candidate Hidden State  |
| Cell State               | No separate cell state  |
| Hidden State             | Hidden State            |
| More independent control | Simpler coupled control |

---

# ⭐ 26. Why GRU Helps Vanishing Gradients

Final update:

`h_t = z_t ⊙ h_(t-1) + ...`

contains a direct path from:

`h_(t-1)`

to:

`h_t`.

If:

`z_t ≈ 1`

then:

```text
h_(t-1)
────────────→
h_t
```

can be nearly preserved.

During BPTT, this gives:

> a more favorable long-range gradient path.

---

# ⚠️ 27. GRU Does Not Completely Eliminate Vanishing Gradients

Do not say:

> “GRU solves vanishing gradients completely.”

Better:

> **GRU mitigates long-range gradient problems through gated additive state updates.**

Long-term behavior still depends on:

* gate values
* sequence length
* optimization
* parameter dynamics
* task complexity

---

# 🔄 28. GRU Still Uses BPTT

GRU remains recurrent.

So training still uses:

> **Backpropagation Through Time**

```text
GRU Forward
↓
Loss
↓
BPTT
↓
Gradients Through Gates + State
↓
Optimizer
```

GRU changes the recurrent architecture.

It does not replace BPTT.

---

# ⏳ 29. GRU Is Still Sequential

At time `t`, GRU needs:

`h_(t-1)`

So:

```text
t1 → t2 → t3 → ...
```

still has sequential dependency.

GRU improves:

> memory behavior

but not:

> time-step parallelism.

---

# 🌊 30. GRU and Streaming

GRU is naturally suitable for streaming.

Maintain:

`h_t`

Then:

```text
x_(t+1)
+
h_t
↓
GRU
↓
h_(t+1)
```

Compared with LSTM, production state management is simpler because we store:

> one recurrent state instead of two.

---

# 👥 31. Shared Parameters vs Per-Sequence State

Shared:

> GRU model parameters.

Separate per independent sequence:

```text
User A → h_A
User B → h_B
User C → h_C
```

Do not share hidden state across unrelated users or sequences.

---

# 📐 32. Dimensions

Let:

* input dimension = `D`
* hidden dimension = `H`

Then:

```text
x_t ∈ R^D
h_(t-1) ∈ R^H

z_t ∈ R^H
r_t ∈ R^H
h̃_t ∈ R^H
h_t ∈ R^H
```

Matrices:

```text
W_z, W_r, W_h
→ H × D

U_z, U_r, U_h
→ H × H
```

Biases:

`H`

each.

---

# 🧮 33. GRU Parameter Count

GRU has three main transformations:

```text
Update
Reset
Candidate
```

Each contains:

`HD + H² + H`

So:

# `3H(D + H + 1)`

under the simplified single-bias formulation.

---

# 🧮 34. Parameter Count Comparison

### Vanilla RNN

`H(D + H + 1)`

### GRU

`3H(D + H + 1)`

### LSTM

`4H(D + H + 1)`

Approximate transformation count:

```text
Vanilla RNN
→ 1×

GRU
→ 3×

LSTM
→ 4×
```

---

# 🧮 35. Numerical Parameter Example

Given:

```text
D = 100
H = 256
```

GRU:

`3 × 256 × (100 + 256 + 1)`

`= 3 × 256 × 357`

# `= 274,176`

Simplified LSTM:

# `365,568`

So GRU has fewer parameters.

---

# 🚀 36. Why GRU Can Be Faster Than LSTM

GRU generally has:

* fewer gates
* fewer transformations
* fewer parameters
* one recurrent state

So it can often require:

* less compute
* less memory
* lower latency

But:

> actual runtime depends on hardware and framework implementation.

Do not say:

> GRU is always faster.

---

# 🔗 37. Vanilla RNN vs GRU

| Vanilla RNN             | GRU                      |
| ----------------------- | ------------------------ |
| Simple recurrent update | Gated recurrent update   |
| One hidden state        | One hidden state         |
| No gates                | Update + reset gates     |
| Rewrites state directly | Blends old and candidate |
| More vanishing issues   | Better long-range flow   |
| Fewer parameters        | More than vanilla RNN    |
| Simpler                 | More expressive          |

---

# 🔗 38. LSTM vs GRU

| LSTM                     | GRU                  |
| ------------------------ | -------------------- |
| `c_t + h_t`              | `h_t` only           |
| 3 gates + candidate      | 2 gates + candidate  |
| Separate memory/exposure | Combined state       |
| More parameters          | Fewer parameters     |
| More independent control | More coupled control |
| Often higher compute     | Often lower compute  |

---

# ⚠️ 39. GRU Is Not Always Better Than LSTM

There is no universal winner.

GRU can be attractive when:

* lower compute matters
* smaller datasets
* lower latency
* simpler architecture

LSTM can be attractive when:

* richer memory control helps
* separate cell/hidden states are useful
* independent retain/write/expose decisions matter

Final choice should be:

> empirical.

---

# 🚫 40. High-Yield Interview Traps

### ❌ GRU has a separate cell state

No.

---

### ❌ Reset gate deletes old memory

No.

It affects candidate construction.

---

### ❌ Update gate and reset gate do the same job

No.

---

### ❌ Candidate is the final state

No.

It is blended with old state.

---

### ❌ `r_t = 0` means old state disappears

No.

Old state can still survive through `z_t`.

---

### ❌ High `z_t` always means “more new”

Depends on notation.

Check the equation.

---

### ❌ GRU completely eliminates vanishing gradients

No.

---

### ❌ GRU does not use BPTT

False.

---

### ❌ GRU is parallel across time

False.

---

### ❌ GRU is always better/faster than LSTM

False.

---

# 📐 41. Formula Flash Card

```text
Update Gate:

z_t = σ(W_z x_t + U_z h_(t-1) + b_z)
```

```text
Reset Gate:

r_t = σ(W_r x_t + U_r h_(t-1) + b_r)
```

```text
Candidate:

h̃_t = tanh(
    W_h x_t
    +
    U_h(r_t ⊙ h_(t-1))
    +
    b_h
)
```

```text
Final Hidden State:

h_t =
z_t ⊙ h_(t-1)
+
(1-z_t) ⊙ h̃_t
```

Parameter count:

# `3H(D + H + 1)`

---

# 🎤 42. 30-Second Interview Answer

> **GRU is a gated recurrent architecture that simplifies LSTM by using a single hidden state and two primary gates: an update gate and a reset gate. The reset gate controls how much previous context is used when creating the candidate hidden state, while the update gate controls the blend between the old hidden state and the candidate. This additive gated update helps preserve information and gradients over longer sequences while generally using fewer parameters and less computation than LSTM.**

---

# 🎤 43. Update Gate Answer

> **The update gate controls the old-vs-new balance in the final hidden state. In the convention `h_t = z_t h_(t-1) + (1-z_t)h̃_t`, a larger `z_t` preserves more old state and a smaller `z_t` uses more candidate information.**

---

# 🎤 44. Reset Gate Answer

> **The reset gate controls how much of the previous hidden state participates in candidate creation. A smaller reset value reduces the influence of old context while constructing the candidate, while a larger value allows more previous context to contribute.**

---

# 🧠 45. Seven Things You Must Know Cold

1. **GRU simplifies LSTM.**
2. **GRU has no separate cell state.**
3. **Update gate = final old-vs-new blend.**
4. **Reset gate = old-context usage during candidate creation.**
5. **Candidate = proposed new hidden content.**
6. **GRU uses additive gated state updates.**
7. **GRU usually has fewer parameters than LSTM.**

---

# 🧠 46. Evolution Story

```text
Vanilla RNN
↓
Simple but poor long-term gradient flow
↓
LSTM
↓
Better memory through cell state + gates
↓
More complexity
↓
GRU
↓
Simplified gated recurrence
```

---

# 🧠 47. Final Mental Model

```text
Previous Hidden h_(t-1)
        │
        ├────────────→ Update Gate z_t
        │
        └────────────→ Reset Gate r_t
                            │
                            ▼
                  Select Previous Context
                            │
                            +
                           x_t
                            │
                            ▼
                      Candidate h̃_t
```

Then:

```text
Old State h_(t-1)
       │
       │ × z_t
       ▼
      KEEP
       │
       ├──────────────┐
       │              │
       │      Candidate h̃_t
       │              │
       │      × (1-z_t)
       │              │
       └────── + ─────┘
              │
              ▼
             h_t
```

Remember:

```text
r_t
→ OLD context for candidate?

h̃_t
→ WHAT new state?

z_t
→ OLD vs NEW?

h_t
→ FINAL state
```

---

# ⭐ Golden Rule

> **GRU simplifies LSTM by combining memory and representation into one hidden state: the reset gate controls how much old context is used to build the candidate, and the update gate controls how much old state versus new candidate becomes the next hidden state.**
