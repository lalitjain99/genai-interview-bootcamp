# 🎤 Interview Questions — Inference and Autoregressive Decoding

> **Goal:** Test whether you understand how Seq2Seq inference actually works, how autoregressive decoding feeds generated tokens back into the decoder, how logits/softmax/selection differ, why `<EOS>` and max-length limits matter, and why greedy decoding can fail at the sequence level.

---

# 🟢 Beginner

## Q1. What is autoregressive decoding?

### ✅ Answer

Autoregressive decoding is the process of generating:

> one token at a time

where each new token is conditioned on:

* the source sequence
* previously generated target tokens

Conceptually:

```text id="inf-int-1"
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

The selected token at one step becomes part of the input/history for the next step.

---

## Q2. How does inference differ from teacher-forced training?

### ✅ Answer

During teacher-forced training:

```text id="inf-int-2"
predict y1
↓
feed TRUE y1
↓
predict y2
```

During inference:

```text id="inf-int-3"
predict ŷ1
↓
feed ŷ1
↓
predict ŷ2
```

The key difference is:

> inference has no ground-truth target sequence.

The model must use its own predictions.

---

## Q3. Why do we need `<BOS>` or `<SOS>`?

### ✅ Answer

At the first decoding step, there is:

> no previous generated target token.

So the decoder needs a special start token:

# `<BOS>` / `<SOS>`

Conceptually:

```text id="inf-int-4"
<BOS>
↓
Decoder
↓
First Target Prediction
```

---

## Q4. Why do we need `<EOS>`?

### ✅ Answer

The target length is usually unknown in advance.

So the decoder learns to generate:

# `<EOS>`

when the sequence is complete.

When `<EOS>` is selected:

> decoding stops.

This enables variable-length generation.

---

# 🟡 Intermediate

## Q5. What are the steps from decoder hidden state to the next generated token?

### ✅ Answer

The sequence is:

```text id="inf-int-5"
Decoder Hidden State
↓
Output Projection
↓
Logits
↓
Softmax
↓
Vocabulary Probability Distribution
↓
Decoding Strategy
↓
Selected Token
```

Mathematically:

`z_t = W_o d_t + b_o`

then:

`P(y_t | y_<t, X) = softmax(z_t)`

Then a decoding strategy such as greedy decoding selects the token.

---

## Q6. Does softmax choose the next token?

### ✅ Answer

No.

Softmax only converts logits into:

> a normalized probability distribution.

For example:

```text id="inf-int-6"
A → 0.60
B → 0.25
C → 0.10
<EOS> → 0.05
```

A separate decoding strategy decides what to do with those probabilities.

For greedy decoding:

# `argmax`

selects the next token.

---

## Q7. What is greedy decoding?

### ✅ Answer

Greedy decoding chooses:

> the highest-probability token at the current timestep.

Mathematically:

# `ŷ_t = argmax_y P(y | y_<t, X)`

Example:

```text id="inf-int-7"
A → 0.62
B → 0.23
C → 0.15
```

Greedy selects:

`A`.

It is:

* simple
* fast
* deterministic

but not guaranteed to find the best complete sequence.

---

## Q8. Why is a maximum decoding length still needed if we have `<EOS>`?

### ✅ Answer

Because the model may fail to generate `<EOS>`.

For example:

```text id="inf-int-8"
A B B B B B B ...
```

A production system therefore usually stops when either:

```text id="inf-int-9"
token == <EOS>
```

or:

```text id="inf-int-10"
generated_length >= max_length
```

So:

* `<EOS>` = learned semantic termination
* `max_length` = safety safeguard

---

## Q9. What is the difference between `<EOS>` and `PAD`?

### ✅ Answer

`<EOS>` means:

> the sequence has ended.

`PAD` means:

> this position is artificial and exists only for batching/alignment.

Example:

```text id="inf-int-11"
A B <EOS> PAD PAD
```

The actual sequence is:

```text id="inf-int-12"
A B
```

These tokens serve completely different purposes.

---

## Q10. Why can one wrong token affect many later predictions?

### ✅ Answer

Because the generated token is fed back into the decoder.

Suppose the correct history is:

```text id="inf-int-13"
A B
```

but the model generates:

```text id="inf-int-14"
A X
```

Then future prediction becomes:

`P(y_3 | A, X, X_source)`

instead of:

`P(y_3 | A, B, X_source)`

So:

```text id="inf-int-15"
Wrong Token
↓
Different Next Input
↓
Different Decoder State
↓
Different Future Distribution
```

This is autoregressive error propagation.

---

# 🔴 Advanced

## Q11. What is the mathematical basis of autoregressive decoding?

### ✅ Answer

The full sequence probability is factorized as:

# `P(Y|X) = Π_t P(y_t | y_<t, X)`

Each target token is conditioned on:

* the source `X`
* all previous target tokens

During inference, those previous target tokens are:

> model-generated tokens.

This factorization is what makes token-by-token decoding possible.

---

## Q12. Why is greedy decoding not guaranteed to find the highest-probability complete sequence?

### ✅ Answer

Greedy decoding optimizes:

> the best next token

rather than:

> the best complete sequence.

Example:

```text id="inf-int-16"
Step 1:
A → 0.55
B → 0.45
```

Greedy picks:

`A`.

But suppose:

```text id="inf-int-17"
Best continuation after A = 0.20
Best continuation after B = 0.90
```

Then:

```text id="inf-int-18"
A-path = 0.55 × 0.20 = 0.11
B-path = 0.45 × 0.90 = 0.405
```

So the lower-probability first token produces the better overall sequence.

---

## Q13. Why do decoding algorithms often use log probabilities instead of raw probabilities?

### ✅ Answer

Sequence probability multiplies many values smaller than 1:

`P(Y|X) = Π_t p_t`

For long sequences, this becomes numerically tiny.

Using logs:

# `log P(Y|X) = Σ_t log p_t`

gives:

* better numerical stability
* easier accumulation
* simpler sequence scoring

This is especially important for beam search.

---

## Q14. Why does autoregressive inference have sequential latency?

### ✅ Answer

At timestep `t+1`, the decoder usually needs:

* previous generated token `ŷ_t`
* previous decoder state `d_t`

So:

```text id="inf-int-19"
Step 1
↓
Step 2
↓
Step 3
↓
Step 4
```

cannot be collapsed into one simultaneous operation.

For an RNN decoder:

`d_t` explicitly depends on `d_(t-1)`.

Therefore:

> longer output sequences require more sequential decoding steps.

---

## Q15. What is the difference between model probabilities and decoding policy?

### ✅ Answer

The model produces:

> a probability distribution.

The decoding policy decides:

> how to turn that distribution into an actual sequence.

Examples:

```text id="inf-int-20"
Model:
A → 0.50
B → 0.30
C → 0.20
```

Possible policies:

* Greedy → select A
* Sampling → sample A/B/C
* Beam Search → maintain multiple candidate sequences

The underlying model probabilities may be identical while output behavior differs because:

> decoding strategy changed.

---

# ⭐ Staff Engineer Challenge

## Scenario

You have deployed a recurrent Seq2Seq summarization system.

Inference pipeline:

```text id="inf-int-21"
Encoder
↓
Decoder
↓
Greedy Decoding
```

Observed production behavior:

* short outputs work well
* long outputs frequently repeat phrases
* some sequences terminate too early
* some never emit `<EOS>`
* latency grows sharply with output length
* one bad token often causes the rest of the summary to deteriorate

---

## Question 1. Why does latency grow with output length?

### ✅ Answer

Because decoding is autoregressive.

For each new token:

```text id="inf-int-22"
previous token
+
previous state
↓
next decoder state
↓
next token
```

So a 200-token output requires roughly:

> 200 sequential generation steps.

This dependency limits inference parallelism.

---

## Question 2. Why might repetition become self-reinforcing?

### ✅ Answer

Suppose the model generates:

`B`.

That token becomes the next decoder input:

```text id="inf-int-23"
B
↓
decoder state
↓
high probability for B again
↓
B
↓
repeat
```

The autoregressive feedback loop can reinforce a bad local pattern.

A max-length limit prevents infinite generation but does not fix the underlying distribution.

---

## Question 3. Why might the model terminate too early?

### ✅ Answer

If `<EOS>` gets a high enough score at an early step, the decoding strategy selects it.

The loop is behaving correctly.

Potential causes may include:

* model calibration
* training data
* decoding history
* length bias
* exposure bias
* weak source representation

So the key issue is usually:

> why `<EOS>` became too probable,

not whether the decoder should obey `<EOS>`.

---

## Question 4. Why is max length not a quality solution?

### ✅ Answer

Max length only provides:

> a hard safety boundary.

If the model repeats:

```text id="inf-int-24"
B B B B ...
```

a limit of 100 simply gives:

```text id="inf-int-25"
B B B ... until 100
```

It stops runaway generation but does not improve:

* semantic quality
* repetition
* sequence selection

---

## Question 5. Would teacher forcing solve these inference failures?

### ✅ Answer

Not directly.

Teacher forcing is a:

> training strategy.

At inference, ground-truth target tokens are unavailable.

However, the training/inference mismatch created by heavy teacher forcing may contribute to poor robustness.

So teacher forcing may be part of the diagnosis, but it is not an inference-time decoding strategy.

---

## Question 6. Why might replacing greedy decoding help?

### ✅ Answer

Greedy permanently commits to:

> one locally best token.

A better sequence might begin with a slightly lower-probability token.

A search strategy that retains multiple candidates can explore alternatives rather than immediately discarding them.

This leads to:

# **Beam Search**

---

## Question 7. What production metrics would you monitor?

### ✅ Answer

I would monitor:

* output length
* decoding latency
* tokens generated per second
* `<EOS>` position
* max-length termination rate
* repetition rate
* sequence quality metric
* error rate vs output length
* frequency of early termination
* candidate score / log probability where useful

The goal is to separate:

```text id="inf-int-26"
model-quality problems
from
decoding-strategy problems
from
latency/system problems
```

---

# ⚡ Rapid-Fire Traps

### ❌ “Inference uses the correct previous target.”

False.

Normal free-running inference uses model-generated tokens.

---

### ❌ “The decoder directly outputs a word.”

Incomplete.

It outputs a hidden representation that is projected to logits.

---

### ❌ “Logits are probabilities.”

False.

---

### ❌ “Softmax chooses the token.”

False.

---

### ❌ “Greedy decoding changes the model.”

False.

It changes how model probabilities are converted into outputs.

---

### ❌ “`<EOS>` and `PAD` are equivalent.”

False.

---

### ❌ “Max length predicts the correct target length.”

False.

Safety guard only.

---

### ❌ “One wrong token cannot affect later tokens.”

False.

---

### ❌ “Greedy finds the globally best sequence.”

False.

---

### ❌ “Autoregressive generation can generate every target position simultaneously.”

False.

---

### ❌ “Beam Search fixes the model probabilities.”

False.

It searches them differently.

---

### ❌ “Inference and evaluation mean the same thing.”

False.

---

# 📐 Formula Flash Card

### Decoder Recurrence

# `d_t = f_dec(e(ŷ_(t-1)), d_(t-1), c)`

### Logits

# `z_t = W_o d_t + b_o`

### Token Distribution

# `P(y_t | y_<t, X) = softmax(z_t)`

### Greedy Selection

# `ŷ_t = argmax_y P(y | y_<t, X)`

### Sequence Probability

# `P(Y|X) = Π_t P(y_t | y_<t, X)`

### Log Sequence Score

# `log P(Y|X) = Σ_t log P(y_t | y_<t, X)`

### Stop Condition

```text id="inf-int-27"
selected token == <EOS>
OR
generated length >= max_length
```

---

# 🎤 30-Second Interview Answer

> **Autoregressive decoding is the inference process where a sequence model generates one token at a time and feeds each selected token back into the decoder to condition the next prediction. Generation starts with `<BOS>`. The decoder produces a hidden state, the output layer converts it into vocabulary logits, softmax produces token probabilities, and a decoding strategy such as greedy argmax chooses the next token. The process repeats until `<EOS>` or a maximum-length safeguard is reached. Because every selected token changes future history, errors can propagate, and greedy decoding may miss the best complete sequence because it only optimizes locally.**

---

# 🧠 Final Interview Mental Model

Think in five stages:

```text id="inf-int-28"
1. START

<BOS>
```

```text id="inf-int-29"
2. SCORE

Decoder
↓
Hidden State
↓
Logits
↓
Probabilities
```

```text id="inf-int-30"
3. SELECT

Probability Distribution
↓
Decoding Strategy
↓
Token
```

```text id="inf-int-31"
4. FEEDBACK

Selected Token
↓
Next Decoder Input
```

```text id="inf-int-32"
5. REPEAT

until <EOS>
or max_length
```

The core loop:

# **Predict → Select → Feed Back → Repeat**

And the core limitation of greedy search:

# **Locally Best Token ≠ Globally Best Sequence**

That is exactly why the next lecture is:

# `13_Beam_Search`

---

# ⭐ Golden Rule

> **The model defines probabilities; the decoding algorithm defines the path through those probabilities. In autoregressive inference, every chosen token becomes part of the future input, so a decoding decision is not isolated—it changes the remainder of the sequence.**
