# ⚡ Cheat Sheet — Gated Recurrent Unit (GRU)

> **Goal:** Fast revision of why GRU exists, how update/reset gates work, how the candidate hidden state is created, and how GRU differs from LSTM.

---

# 🌟 1. GRU in One Line

> **GRU = simpler gated RNN with one recurrent state**

It was introduced to keep much of LSTM's long-term memory benefit while using:

* fewer gates
* fewer parameters
* one recurrent state

---

# 🧠 2. Why GRU Was Needed

LSTM solved many vanilla-RNN memory problems, but it introduced:

```text
Cell State c_t
+
Hidden State h_t
+
Forget Gate
+
Input Gate
+
Output Gate
+
Candidate
```

GRU asks:

> Can we simplify this?

So GRU uses:

```text
Hidden State h_t
+
Update Gate
+
Reset Gate
+
Candidate Hidden State
```

---

# ⭐ 3. GRU Has One Recurrent State

LSTM:

```text
c_t
→ cell memory

h_t
→ exposed hidden state
```

GRU:

```text
h_t
→ recurrent memory + representation
```

So standard GRU has:

> **no separate cell state**

---

# 🚪 4. Main GRU Components

```text
Update Gate z_t
Reset Gate r_t
Candidate Hidden State h̃_t
Final Hidden State h_t
```

Shortcut:

```text
r_t
→ How much OLD context helps build candidate?

h̃_t
→ What NEW state could be created?

z_t
→ How much OLD vs NEW should survive?

h_t
→ Final state
```

---

# 🔄 5. Inputs and Output

Inputs:

```text
x_t
→ current input

h_(t-1)
→ previous hidden state
```

Output:

```text
h_t
→ new hidden state
```

No:

`c_t`

---

# 🚪 6. Update Gate

Equation:

`z_t = σ(W_z x_t + U_z h_(t-1) + b_z)`

Purpose:

> control the final blend between old state and candidate.

Using our convention:

# `h_t = z_t ⊙ h_(t-1) + (1-z_t) ⊙ h̃_t`

So:

```text
z_t ≈ 1
→ preserve more OLD state

z_t ≈ 0
→ use more NEW candidate
```

---

# ⚠️ 7. Update-Gate Convention Warning

Some sources use:

`h_t = (1-z_t) ⊙ h_(t-1) + z_t ⊙ h̃_t`

So always check the formula.

Do not blindly memorize:

> high `z_t` means old

or

> high `z_t` means new

without checking the source convention.

---

# 🔁 8. Reset Gate

Equation:

`r_t = σ(W_r x_t + U_r h_(t-1) + b_r)`

Purpose:

> control how much previous hidden state is used while creating the candidate.

Interpretation:

```text
r_t ≈ 1
→ use more old context

r_t ≈ 0
→ use less old context
```

---

# ⭐ 9. Reset vs Update

This distinction must be crystal clear.

```text
Reset Gate
→ candidate construction

Update Gate
→ final old/new blending
```

Do not confuse them.

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

Candidate means:

> proposed new hidden content.

It is not yet the final state.

---

# 🧠 11. Meaning of `W_h` and `U_h`

In:

`h̃_t = tanh(W_h x_t + U_h(r_t ⊙ h_(t-1)) + b_h)`

```text
W_h
→ transforms current input x_t

U_h
→ transforms selected previous hidden state
```

Shortcut:

```text
W_h
→ CURRENT input contribution

U_h
→ PAST context contribution
```

---

# 🔥 12. Why `tanh` for Candidate?

Candidate represents:

> CONTENT

`tanh` gives:

`[-1,1]`

So it can represent:

* positive values
* negative values
* near-zero values

while remaining bounded.

Shortcut:

```text
sigmoid
→ CONTROL

tanh
→ CONTENT
```

---

# 📐 13. Final Hidden-State Equation

# `h_t = z_t ⊙ h_(t-1) + (1-z_t) ⊙ h̃_t`

Read it as:

# **Final State = KEEP OLD + USE NEW**

---

# 🔍 14. Special Cases

## If `z_t = 1`

`h_t = h_(t-1)`

Old state is preserved.

---

## If `z_t = 0`

`h_t = h̃_t`

Candidate fully replaces old state.

---

## If `r_t = 0`

Candidate creation largely ignores old context.

But old state can still survive through:

`z_t ⊙ h_(t-1)`

---

## If `r_t = 1`

Candidate uses the full previous hidden state.

---

# ⚠️ 15. Reset Gate = 0 Does NOT Mean Forget Everything

If:

```text
r_t = 0
z_t = 0.9
```

then candidate ignores old context, but final state can still be:

```text
90% old state
+
10% candidate
```

So:

# **Reset ≠ Final Retention**

---

# 🧮 16. Quick Numerical Example

Suppose:

```text
h_(t-1) = 0.8
h̃_t     = 0.2
z_t      = 0.75
```

Then:

`h_t = 0.75×0.8 + 0.25×0.2`

`= 0.60 + 0.05`

# `= 0.65`

Interpretation:

> mostly preserve old state.

---

# ⭐ 17. Why GRU Helps Long-Term Gradients

Final state contains:

`z_t ⊙ h_(t-1)`

So if:

`z_t ≈ 1`

there is a relatively direct path:

```text
h_(t-1)
────────────→
h_t
```

This helps:

* preserve information
* preserve gradient flow
* reduce long-term dependency issues

---

# ⚠️ 18. GRU Does Not Completely Eliminate Vanishing Gradients

GRU:

> mitigates the problem

but does not guarantee perfect memory.

Gradient behavior still depends on:

* gate values
* sequence length
* optimization
* parameters
* task complexity

---

# 🔗 19. GRU vs Vanilla RNN

Vanilla RNN:

```text
Old State
+
Input
↓
Completely New Nonlinear State
```

GRU:

```text
Old State
─────────┐
         ├→ gated blend → h_t
Candidate
─────────┘
```

So GRU gives the old state:

> a direct gated route forward.

---

# 🔗 20. GRU vs LSTM

| LSTM                            | GRU                     |
| ------------------------------- | ----------------------- |
| `c_t + h_t`                     | `h_t` only              |
| Forget gate                     | No separate forget gate |
| Input gate                      | No separate input gate  |
| Output gate                     | No output gate          |
| Candidate memory                | Candidate hidden state  |
| More independent memory control | More coupled control    |
| More parameters                 | Fewer parameters        |
| Often more compute              | Often less compute      |

---

# 🧠 21. Where Did LSTM Forget + Input Gates Go?

LSTM:

```text
Forget
→ how much OLD to keep

Input
→ how much NEW to write
```

GRU combines much of this using:

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

This is simpler, but less independently flexible.

---

# 🧠 22. Where Did the Output Gate Go?

LSTM needs an output gate because it separates:

```text
internal cell memory
from
exposed hidden state
```

GRU does not have this separation.

It only has:

`h_t`

Therefore:

> no standard output gate.

---

# 📐 23. Dimensions

If:

* input size = `D`
* hidden size = `H`

then:

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

# 🧮 24. GRU Parameter Count

GRU has three main transformations:

```text
Update
Reset
Candidate
```

Simplified count:

# `3H(D + H + 1)`

---

# 🧮 25. Parameter Comparison

### Vanilla RNN

`H(D+H+1)`

### GRU

`3H(D+H+1)`

### LSTM

`4H(D+H+1)`

Approximate transformation count:

```text
Vanilla
→ 1×

GRU
→ 3×

LSTM
→ 4×
```

---

# 🌊 26. Streaming Strength

For streaming inference:

```text
h_t
+
x_(t+1)
↓
GRU
↓
h_(t+1)
```

No need to replay the full sequence.

Compared with LSTM, state management is simpler because only:

`h_t`

needs to be maintained.

---

# 👥 27. Shared Weights, Separate State

Shared:

> GRU model parameters.

Separate per independent sequence:

```text
User A → h_A
User B → h_B
User C → h_C
```

Never share hidden state between unrelated sequences.

---

# 🔄 28. GRU Still Uses BPTT

GRU remains recurrent.

So training still uses:

> **Backpropagation Through Time**

```text
Forward GRU
↓
Loss
↓
BPTT
↓
Gradient
↓
Optimizer
```

---

# ⏳ 29. GRU Is Still Sequential

At time `t`, GRU needs:

`h_(t-1)`

So:

```text
t1 → t2 → t3 → ...
```

still has temporal dependency.

GRU improves:

> memory behavior

not:

> full time-step parallelization.

---

# 🚫 30. High-Yield Traps

### ❌ GRU has a cell state

No.

---

### ❌ Reset gate decides final retention

No.

It affects candidate construction.

---

### ❌ Update gate creates the candidate

No.

It blends old and new.

---

### ❌ Candidate is final hidden state

No.

---

### ❌ `r_t = 0` means old state disappears

No.

---

### ❌ GRU has an output gate

No.

---

### ❌ GRU completely eliminates vanishing gradients

No.

---

### ❌ GRU does not use BPTT

False.

---

### ❌ GRU is fully parallel across time

False.

---

### ❌ GRU is always better than LSTM

False.

---

# 📐 31. Formula Flash Card

```text
Update:
z_t = σ(W_z x_t + U_z h_(t-1) + b_z)
```

```text
Reset:
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
Final:
h_t =
z_t ⊙ h_(t-1)
+
(1-z_t) ⊙ h̃_t
```

Simplified parameters:

# `3H(D + H + 1)`

---

# 🎤 32. 30-Second Interview Answer

> **GRU is a gated recurrent architecture that simplifies LSTM by using one hidden state and two main gates. The reset gate controls how much previous context is used to build the candidate hidden state, while the update gate controls the blend between the old hidden state and the new candidate. This gated additive state update improves long-range information and gradient flow compared with vanilla RNN while generally using fewer parameters than LSTM.**

---

# 🧠 33. Six Things You Must Know Cold

1. **GRU has one recurrent state: `h_t`.**
2. **Reset gate affects candidate creation.**
3. **Update gate affects final old-vs-new blending.**
4. **`W_h` handles current input; `U_h` handles selected past context.**
5. **Candidate is proposed new hidden content.**
6. **GRU is simpler than LSTM but still recurrent and trained with BPTT.**

---

# 🧠 Final Mental Model

```text
Previous Hidden h_(t-1)
        │
        ├────────────→ Reset Gate r_t
        │                   │
        │                   ▼
        │          Select Past Context
        │                   │
x_t ────┴───────────────────┤
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
      ├─────────────┐
      │             │
      │     Candidate h̃_t
      │             │
      │     × (1-z_t)
      │             │
      └────── + ────┘
             │
             ▼
            h_t
```

Remember:

# **Reset = old context for candidate**

# **Candidate = proposed new state**

# **Update = old vs new**

# **Hidden State = final recurrent state**

---

# ⭐ Golden Rule

> **GRU simplifies gated recurrence by using one hidden state: the reset gate controls how much past context contributes to the candidate, while the update gate controls how much old state versus new candidate becomes the next state.**
