# 📝 Revision — Inference and Autoregressive Decoding

> **Goal:** Quickly revise how Seq2Seq inference works when the correct target is unknown, how autoregressive decoding feeds predictions back into the decoder, why `<BOS>` and `<EOS>` matter, how greedy decoding works, and why local token choices can hurt full-sequence quality.

---

# 🌟 1. What Is Inference?

Inference means:

> using the trained model to generate an output for a new input.

For Seq2Seq:

```text id="inf-rev-1"
Source Sequence
↓
Encoder
↓
Decoder
↓
Generated Target Sequence
```

At inference:

> the correct target sequence is not available.

---

# 🧠 2. Training vs Inference

### Training with Teacher Forcing

```text id="inf-rev-2"
<SOS>
↓
predict y1
↓
feed TRUE y1
↓
predict y2
```

### Inference

```text id="inf-rev-3"
<SOS>
↓
predict ŷ1
↓
feed ŷ1
↓
predict ŷ2
```

So:

# **Training History ≠ Inference History**

---

# ⭐ 3. What Is Autoregressive Decoding?

Autoregressive decoding means:

> generate one token at a time and use previously generated tokens to predict the next token.

Flow:

```text id="inf-rev-4"
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

Each new token depends on:

> source + generated target history.

---

# 📐 4. Core Probability

At target step `t`:

# `P(y_t | y_<t, X)`

The whole sequence probability is:

# `P(Y|X) = Π_t P(y_t | y_<t, X)`

This is the mathematical basis of autoregressive generation.

---

# 🚀 5. First Decoder Step

At inference:

```text id="inf-rev-5"
Source
↓
Encoder
↓
Source Representation

<BOS>
↓
Decoder
↓
d1
```

The first decoder input is:

# `<BOS>` / `<SOS>`

because no previous generated target token exists yet.

---

# 🧠 6. Decoder State Is Not a Token

The decoder produces:

`d_t`

which is a hidden representation.

Then:

# `z_t = W_o d_t + b_o`

creates vocabulary logits.

So:

```text id="inf-rev-6"
Decoder State
↓
Output Projection
↓
Logits
```

---

# 📐 7. Logits → Probabilities

Apply:

# `softmax(z_t)`

to obtain:

# `P(y_t | y_<t, X)`

Example:

```text id="inf-rev-7"
A → 0.60
B → 0.25
C → 0.10
<EOS> → 0.05
```

Important:

> softmax produces probabilities.

It does **not** choose the token.

---

# ⭐ 8. Decoding Strategy Chooses the Token

Once probabilities are available, a decoding strategy chooses the next token.

Examples:

* Greedy decoding
* Sampling
* Beam Search

For greedy:

# `ŷ_t = argmax_y P(y | y_<t, X)`

---

# 🔄 9. Feedback Loop

Suppose greedy chooses:

`A`

Then:

```text id="inf-rev-8"
A
↓
Embedding
↓
Next Decoder Step
```

So the complete loop is:

# **Predict → Select → Feed Back → Predict Again**

That is the essence of autoregressive decoding.

---

# 📐 10. Decoder Equation During Inference

A simplified recurrent decoder:

# `d_t = f_dec(e(ŷ_(t-1)), d_(t-1), c)`

where:

* `ŷ_(t-1)` = previous generated token
* `d_(t-1)` = previous decoder state
* `c` = source context

Then:

`z_t = W_o d_t + b_o`

and:

`softmax(z_t)`

creates the next-token distribution.

---

# 🛑 11. Why `<EOS>` Is Needed

The model usually does not know output length in advance.

So it learns to predict:

# `<EOS>`

When selected:

```text id="inf-rev-9"
<EOS>
↓
STOP
```

This provides:

> learned variable-length termination.

---

# ⚠️ 12. `<EOS>` vs `PAD`

### `<EOS>`

Means:

> sequence finished.

### `PAD`

Means:

> artificial batching placeholder.

Example:

```text id="inf-rev-10"
A B <EOS> PAD PAD
```

Actual generated sequence:

```text id="inf-rev-11"
A B
```

---

# 🚨 13. Maximum-Length Safeguard

What if the model never predicts `<EOS>`?

Use:

# `max_length`

Stop when:

```text id="inf-rev-12"
token == <EOS>
OR
generated_length >= max_length
```

So we have:

```text id="inf-rev-13"
<EOS>
→ semantic stop

max_length
→ safety stop
```

---

# ⭐ 14. Greedy Decoding

Greedy decoding chooses:

> the highest-probability token at every step.

Example:

```text id="inf-rev-14"
A → 0.55
B → 0.45
```

Greedy chooses:

`A`.

Advantages:

* simple
* fast
* deterministic
* low memory

---

# 🚨 15. Greedy Is Only Locally Optimal

Suppose:

```text id="inf-rev-15"
A → 0.55
B → 0.45
```

Later:

```text id="inf-rev-16"
Best continuation after A = 0.20
Best continuation after B = 0.90
```

Scores:

```text id="inf-rev-17"
A path = 0.55 × 0.20 = 0.11
B path = 0.45 × 0.90 = 0.405
```

So:

# **Best Token Now ≠ Best Full Sequence**

This motivates:

> Beam Search.

---

# 🧠 16. Error Propagation

Suppose correct output is:

```text id="inf-rev-18"
A B C
```

but model generates:

```text id="inf-rev-19"
A X
```

Then future predictions are conditioned on:

`A, X`

instead of:

`A, B`.

So:

```text id="inf-rev-20"
Wrong Token
↓
Different Next Input
↓
Different Decoder State
↓
Different Probability Distribution
↓
Possible More Errors
```

---

# 🔗 17. Exposure Bias vs Error Propagation

### Exposure Bias

Training/inference mismatch:

```text id="inf-rev-21"
Training
→ ground-truth history

Inference
→ generated history
```

### Error Propagation

Actual inference effect:

```text id="inf-rev-22"
wrong token
→ future decoding changes
```

They are related but not identical.

---

# 📐 18. Sequence Probability in Log Space

Raw probability:

# `P(Y|X) = Π_t P(y_t | y_<t, X)`

For long sequences, this becomes very small.

So use:

# `log P(Y|X) = Σ_t log P(y_t | y_<t, X)`

This is numerically more stable.

---

# 🧠 19. Why Longer Sequences Get Lower Raw Probability

If average token probability is:

`0.8`

then:

```text id="inf-rev-23"
2 tokens:
0.8² = 0.64

10 tokens:
0.8¹⁰ ≈ 0.107

50 tokens:
0.8⁵⁰ ≈ 0.000014
```

So raw sequence probability naturally decreases as sequence length grows.

This becomes important for:

> Beam Search scoring.

---

# 🔄 20. Autoregressive Generation Is Sequential

To generate token `t+1`, the model needs:

> token `t`.

For an RNN decoder:

```text id="inf-rev-24"
d1
↓
d2
↓
d3
↓
d4
```

So inference cannot generate all target tokens simultaneously.

---

# ⚡ 21. Output Length Affects Latency

If output has:

```text id="inf-rev-25"
10 tokens
```

roughly 10 decoding steps are needed.

If:

```text id="inf-rev-26"
500 tokens
```

roughly 500 sequential decoding steps are needed.

So:

> longer output → higher inference latency.

---

# 🧠 22. RNN Training vs Inference Parallelism

Teacher forcing may make all target tokens known during training.

But RNN states remain:

```text id="inf-rev-27"
d1 → d2 → d3
```

sequential.

So teacher forcing does not remove:

> recurrent state dependency.

---

# 🔁 23. Repetition Failure

The model may generate:

```text id="inf-rev-28"
A B B B B B...
```

Possible loop:

```text id="inf-rev-29"
B generated
↓
B fed back
↓
state predicts B again
↓
B fed back
```

This can create:

> self-reinforcing repetition.

---

# 🚨 24. Max Length Does Not Fix Repetition

A limit such as:

`max_length = 100`

only guarantees:

> generation eventually stops.

It does not fix:

> why the model is repeating.

---

# 🛑 25. Early `<EOS>` Failure

Suppose expected output is long, but model generates:

```text id="inf-rev-30"
A <EOS>
```

The loop itself is behaving correctly.

The issue is that:

> `<EOS>` received too high a model score.

Potential contributors include:

* training quality
* decoder state
* bad history
* length bias
* data distribution

---

# 🎲 26. Greedy vs Sampling

Distribution:

```text id="inf-rev-31"
A → 0.50
B → 0.30
C → 0.20
```

### Greedy

Always:

`A`

### Sampling

Could select:

* A
* B
* C

according to their probabilities.

Greedy:

> deterministic.

Sampling:

> stochastic.

---

# 🧠 27. Decoding Strategy Depends on the Task

Translation:

> fidelity is important.

Common strategies:

* Greedy
* Beam Search

Creative generation:

> diversity may matter.

Possible strategies:

* Sampling
* Top-k
* Top-p

The model probabilities stay the same.

The:

> token-selection strategy changes.

---

# 🚫 28. Softmax vs Decoding

Do not say:

> “Softmax predicts the word.”

Better:

```text id="inf-rev-32"
Logits
↓
Softmax
↓
Probability Distribution
↓
Decoding Strategy
↓
Selected Token
```

Softmax scores possibilities.

The decoding strategy chooses one.

---

# 📐 29. Output Projection Cost

If:

```text id="inf-rev-33"
Decoder hidden size = H
Vocabulary size = V
```

then:

# `W_o ∈ R^(V × H)`

At every decoder step, the system generates:

`V`

logits.

So large vocabularies can make the output layer computationally significant.

---

# 📦 30. Batch Inference

Multiple sequences can often decode together.

But sequences may finish at different times:

```text id="inf-rev-34"
A → <EOS>
B → continues
C → continues
```

Finished sequences need to be:

> masked / marked complete.

Otherwise they may accidentally continue generating.

---

# 🔗 31. Inference vs Evaluation

### Inference

> Generate output.

### Evaluation

> Measure how good the output is.

Evaluation may use:

* reference targets
* BLEU
* ROUGE
* exact match
* human evaluation
* other task metrics

Do not confuse:

> producing an answer

with:

> scoring the answer.

---

# 🧠 32. Modern LLM Connection

Modern LLM generation follows the same loop.

Prompt:

```text id="inf-rev-35"
"The capital of France is"
```

Model predicts:

```text id="inf-rev-36"
Paris
```

Then the next context includes:

```text id="inf-rev-37"
"The capital of France is Paris"
```

and generation continues.

So:

# **LLM generation is autoregressive decoding too.**

---

# 🔗 33. Problem Taxonomy

Keep these concepts separate:

```text id="inf-rev-38"
Context Bottleneck
→ source access

Teacher Forcing
→ training history

Exposure Bias
→ train/inference mismatch

Autoregressive Decoding
→ generation feedback loop

Greedy Decoding
→ local selection

Beam Search
→ multi-candidate sequence search
```

---

# 🚫 34. High-Yield Traps

### ❌ Inference has ground-truth target tokens

No.

### ❌ Decoder hidden state is already the output token

No.

### ❌ Logits are probabilities

No.

### ❌ Softmax selects the token

No.

### ❌ Autoregressive decoding generates all tokens at once

No.

### ❌ `<EOS>` = `PAD`

No.

### ❌ Maximum length gives the correct output length

No.

### ❌ Greedy gives the globally best sequence

No.

### ❌ One wrong token affects only one position

No.

### ❌ Beam Search changes model probabilities

No.

### ❌ Inference = evaluation

No.

---

# 📐 35. Formula Flash Card

### Decoder

# `d_t = f_dec(e(ŷ_(t-1)), d_(t-1), c)`

### Logits

# `z_t = W_o d_t + b_o`

### Probability Distribution

# `P(y_t | y_<t, X) = softmax(z_t)`

### Greedy Selection

# `ŷ_t = argmax_y P(y | y_<t, X)`

### Sequence Probability

# `P(Y|X) = Π_t P(y_t | y_<t, X)`

### Log Score

# `log P(Y|X) = Σ_t log P(y_t | y_<t, X)`

### Stop

```text id="inf-rev-39"
<EOS>
OR
max_length reached
```

---

# 🎤 36. 30-Second Interview Answer

> **Autoregressive decoding is the inference process where a Seq2Seq decoder generates one token at a time and feeds the selected token back as input for the next step. Generation begins with `<BOS>`, the decoder produces a hidden state, that state is projected to vocabulary logits, softmax converts logits to a probability distribution, and a decoding strategy such as greedy argmax chooses the next token. This loop continues until `<EOS>` or a maximum-length limit. Because every selected token becomes part of future history, errors can propagate, and greedy decoding can be suboptimal because locally best token choices do not guarantee the best complete sequence.**

---

# 🧠 10 Things You Must Know Cold

1. **Inference has no ground-truth target.**
2. **`<BOS>` starts decoding.**
3. **Decoder state → logits → probabilities → token selection.**
4. **Softmax does not choose the token.**
5. **Generated tokens are fed back into the decoder.**
6. **That feedback makes decoding autoregressive.**
7. **`<EOS>` provides learned stopping.**
8. **Maximum length is a safety guard.**
9. **Errors can propagate through later decoder states.**
10. **Greedy is locally optimal, not sequence-optimal.**

---

# 🧠 Final Mental Model

```text id="inf-rev-40"
<BOS>
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
Probability Distribution
  │
  ▼
Decoding Strategy
  │
  ▼
Selected Token
  │
  └──────────────┐
                 │
                 ▼
          Next Decoder Step
```

Repeat until:

```text id="inf-rev-41"
<EOS>
```

or:

```text id="inf-rev-42"
max_length
```

The core loop is:

# **Predict → Select → Feed Back → Repeat**

And the key problem:

# **Best Token Now ≠ Best Sequence Overall**

That naturally leads to:

# `13_Beam_Search`

---

# ⭐ Golden Rule

> **Autoregressive inference is a sequential feedback process: every selected token becomes part of the model's future context, so decoding strategy matters because one local decision can influence the entire remainder of the generated sequence.**
