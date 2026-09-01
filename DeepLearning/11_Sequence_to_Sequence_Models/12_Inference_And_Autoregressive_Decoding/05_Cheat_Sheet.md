# ⚡ Cheat Sheet — Inference and Autoregressive Decoding

> **Goal:** Fast revision of how Seq2Seq inference works, how autoregressive decoding feeds generated tokens back, how logits/softmax/selection differ, why `<EOS>` and `max_length` matter, and why greedy decoding can fail at the sequence level.

---

# 🌟 1. Autoregressive Decoding in One Line

> **Generate one token, feed it back, then generate the next token.**

```text id="inf-cheat-1"
<BOS>
↓
ŷ1
↓
ŷ2
↓
ŷ3
↓
...
↓
<EOS>
```

Core loop:

# **Predict → Select → Feed Back → Repeat**

---

# 🧠 2. Training vs Inference

### Teacher-Forced Training

```text id="inf-cheat-2"
predict y1
↓
feed TRUE y1
↓
predict y2
```

### Inference

```text id="inf-cheat-3"
predict ŷ1
↓
feed ŷ1
↓
predict ŷ2
```

Key point:

# **Inference has no ground-truth target history.**

---

# 🚀 3. How Decoding Starts

The decoder starts with:

# `<BOS>` / `<SOS>`

because there is no previous target token yet.

```text id="inf-cheat-4"
Source
↓
Encoder
↓
Source Representation

<BOS>
↓
Decoder
```

---

# 📐 4. Decoder State → Token Distribution

Decoder hidden state:

`d_t`

Output projection:

# `z_t = W_o d_t + b_o`

Then:

# `P(y_t | y_<t, X) = softmax(z_t)`

Flow:

```text id="inf-cheat-5"
Hidden State
↓
Logits
↓
Softmax
↓
Vocabulary Probabilities
```

---

# 🚫 5. Softmax Does Not Select the Token

Softmax gives:

> a probability distribution.

Example:

```text id="inf-cheat-6"
A → 0.60
B → 0.25
C → 0.10
<EOS> → 0.05
```

Then a:

# **decoding strategy**

selects the token.

---

# ⭐ 6. Greedy Decoding

Greedy selects:

# `ŷ_t = argmax_y P(y | y_<t, X)`

Example:

```text id="inf-cheat-7"
A → 0.60
B → 0.25
```

Choose:

`A`

Pros:

* simple
* fast
* deterministic
* low memory

---

# 🔄 7. Feedback Makes It Autoregressive

After selecting:

`A`

the decoder uses:

```text id="inf-cheat-8"
A
↓
Embedding
↓
Next Decoder Step
```

So:

> previous generated outputs influence future outputs.

---

# 📐 8. Decoder Equation During Inference

# `d_t = f_dec(e(ŷ_(t-1)), d_(t-1), c)`

where:

* `ŷ_(t-1)` = previous generated token
* `d_(t-1)` = previous hidden state
* `c` = source context

Then:

`z_t = W_o d_t + b_o`

and:

`softmax(z_t)`.

---

# 🛑 9. `<EOS>` Terminates Generation

When:

# `<EOS>`

is selected:

```text id="inf-cheat-9"
Generation
↓
STOP
```

This provides:

> learned variable-length termination.

---

# 🚨 10. Why `max_length` Is Still Needed

The model may never emit `<EOS>`.

So practical systems stop when:

```text id="inf-cheat-10"
token == <EOS>
OR
length >= max_length
```

Remember:

```text id="inf-cheat-11"
<EOS>
→ semantic stop

max_length
→ safety stop
```

---

# ⚠️ 11. `<EOS>` vs `PAD`

### `<EOS>`

> actual end of generated sequence

### `PAD`

> artificial batching placeholder

Example:

```text id="inf-cheat-12"
A B <EOS> PAD PAD
```

Actual output:

```text id="inf-cheat-13"
A B
```

---

# 🚨 12. Error Propagation

Suppose correct history is:

```text id="inf-cheat-14"
A B
```

but generated history becomes:

```text id="inf-cheat-15"
A X
```

Then future distribution changes:

```text id="inf-cheat-16"
Wrong Token
↓
Different History
↓
Different Hidden State
↓
Different Next Probabilities
↓
Possible More Errors
```

This is:

# **autoregressive error propagation**

---

# 🔗 13. Exposure Bias vs Error Propagation

### Exposure Bias

```text id="inf-cheat-17"
Training
→ ground-truth history

Inference
→ generated history
```

### Error Propagation

```text id="inf-cheat-18"
Wrong generated token
→ affects later predictions
```

Related, but not identical.

---

# 🚨 14. Greedy Is Locally Optimal Only

Suppose:

```text id="inf-cheat-19"
A → 0.55
B → 0.45
```

Greedy chooses:

`A`.

But later:

```text id="inf-cheat-20"
A continuation → 0.20
B continuation → 0.90
```

Then:

```text id="inf-cheat-21"
A-path = 0.55 × 0.20 = 0.11
B-path = 0.45 × 0.90 = 0.405
```

So:

# **Best token now ≠ best full sequence**

This motivates:

# **Beam Search**

---

# 📐 15. Sequence Probability

# `P(Y|X) = Π_t P(y_t | y_<t, X)`

Because many probabilities are multiplied, long-sequence values become tiny.

Use log scores:

# `log P(Y|X) = Σ_t log P(y_t | y_<t, X)`

This is numerically more stable.

---

# 🧠 16. Why Long Outputs Cost More

Autoregressive generation is sequential:

```text id="inf-cheat-22"
Step 1
↓
Step 2
↓
Step 3
↓
...
```

So:

```text id="inf-cheat-23"
10 output tokens
→ ~10 decoding steps

500 output tokens
→ ~500 decoding steps
```

Longer outputs usually mean:

> higher latency.

---

# 🔁 17. Common Generation Failures

### Repetition

```text id="inf-cheat-24"
B
↓
B fed back
↓
B predicted again
↓
loop
```

### Early Termination

```text id="inf-cheat-25"
A <EOS>
```

when output should be longer.

### No Termination

```text id="inf-cheat-26"
A B C C C C ...
```

until `max_length`.

These are model/decoding behaviors, not just loop bugs.

---

# 🎲 18. Greedy vs Sampling vs Beam Search

```text id="inf-cheat-27"
Model Probabilities
↓
Decoding Strategy
```

### Greedy

Choose highest probability.

### Sampling

Randomly sample according to distribution.

### Beam Search

Keep multiple high-scoring partial sequences.

Same model.

Different decoding behavior.

---

# 🔗 19. Problem Taxonomy

```text id="inf-cheat-28"
Teacher Forcing
→ training history strategy

Exposure Bias
→ train/inference mismatch

Autoregressive Decoding
→ feedback generation loop

Greedy Decoding
→ local token selection

Beam Search
→ multi-candidate sequence search

Context Bottleneck
→ source-access problem
```

Do not mix them.

---

# 🚫 20. High-Yield Traps

### ❌ Inference uses ground-truth previous tokens

No.

### ❌ Decoder hidden state is already a word

No.

### ❌ Logits are probabilities

No.

### ❌ Softmax selects the token

No.

### ❌ Autoregressive decoding generates all tokens simultaneously

No.

### ❌ `<EOS>` = `PAD`

No.

### ❌ `max_length` predicts correct output length

No.

### ❌ One wrong token affects only one position

No.

### ❌ Greedy finds the globally best sequence

No.

### ❌ Beam Search changes the model probabilities

No.

---

# 📐 21. Formula Flash Card

### Decoder

# `d_t = f_dec(e(ŷ_(t-1)), d_(t-1), c)`

### Logits

# `z_t = W_o d_t + b_o`

### Probabilities

# `P(y_t | y_<t, X) = softmax(z_t)`

### Greedy

# `ŷ_t = argmax_y P(y | y_<t, X)`

### Sequence Probability

# `P(Y|X) = Π_t P(y_t | y_<t, X)`

### Log Score

# `log P(Y|X) = Σ_t log P(y_t | y_<t, X)`

### Stop

```text id="inf-cheat-29"
<EOS>
OR
max_length
```

---

# 🎤 22. 30-Second Interview Answer

> **Autoregressive decoding is the inference process where a sequence model generates one token at a time and feeds each selected token back into the decoder to condition the next prediction. Generation begins with `<BOS>`. The decoder produces a hidden state, that state is projected to vocabulary logits, softmax produces token probabilities, and a decoding strategy such as greedy argmax selects the next token. The loop continues until `<EOS>` or a maximum-length safeguard. Because each selected token changes future history, errors can propagate, and greedy decoding may miss the best complete sequence because it only optimizes locally.**

---

# 🧠 10 Things You Must Know Cold

1. **Inference has no ground-truth target.**
2. **`<BOS>` / `<SOS>` starts decoding.**
3. **Decoder state → logits → probabilities → selection.**
4. **Softmax does not choose the token.**
5. **Generated tokens are fed back.**
6. **That feedback makes decoding autoregressive.**
7. **`<EOS>` provides learned termination.**
8. **`max_length` is a safety guard.**
9. **Errors can affect the whole future trajectory.**
10. **Greedy is locally optimal, not sequence-optimal.**

---

# 🧠 Final Mental Model

```text id="inf-cheat-30"
Previous Token
      │
      ▼
   Embedding
      │
      ▼
    Decoder
      │
      ▼
 Hidden State
      │
      ▼
    Logits
      │
      ▼
    Softmax
      │
      ▼
 Probabilities
      │
      ▼
Decoding Strategy
      │
      ▼
 Selected Token
      │
      └───────────────┐
                      │
                      ▼
               Next Decoder Step
```

Repeat until:

```text id="inf-cheat-31"
<EOS>
```

or:

```text id="inf-cheat-32"
max_length
```

The core loop:

# **Predict → Select → Feed Back → Repeat**

And the limitation that matters next:

# **Best Token Now ≠ Best Complete Sequence**

That leads directly to:

# `13_Beam_Search`

---

# ⭐ Golden Rule

> **The model provides probabilities; the decoding strategy chooses the path. In autoregressive generation, every selected token becomes part of the future input, so one local decoding choice can influence the entire remaining sequence.**
