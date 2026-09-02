# 💡 Assignment — Inference and Autoregressive Decoding

> **Goal:** Apply the full inference loop by tracing decoder states, logits, probabilities, token selection, feedback, stopping conditions, sequence scores, error propagation, and greedy-decoding limitations.

---

# 🧩 Task 1 — Put the Inference Pipeline in Order

Arrange these steps correctly:

* Softmax
* Selected token
* Decoder hidden state
* Previous token embedding
* Output projection
* Logits
* Feed selected token back

### ✅ Answer

```text id="inf-ass-1"
Previous Token
↓
Embedding
↓
Decoder Hidden State
↓
Output Projection
↓
Logits
↓
Softmax
↓
Probability Distribution
↓
Selected Token
↓
Feed Back
```

---

# 🚀 Task 2 — First Decoder Input

At inference, the target sequence is unknown.

What token is typically used to start generation?

### ✅ Answer

# `<BOS>` / `<SOS>`

Because there is no previous generated token yet.

---

# 📐 Task 3 — Hidden State to Logits

Suppose:

```text id="inf-ass-2"
decoder hidden size H = 256
target vocabulary V = 20,000
```

What are the shapes of:

* `d_t`
* `W_o`
* `b_o`
* `z_t`

### ✅ Answer

```text id="inf-ass-3"
d_t ∈ R^256

W_o ∈ R^(20000 × 256)

b_o ∈ R^20000

z_t ∈ R^20000
```

So each decoding step produces:

> one logit per vocabulary token.

---

# 🧠 Task 4 — Logits or Probabilities?

Suppose decoder output projection gives:

```text id="inf-ass-4"
A      → 3.4
B      → 1.2
C      → -0.5
<EOS>  → 0.3
```

Are these probabilities?

### ✅ Answer

No.

These are:

# **logits**

They do not need to:

* lie between 0 and 1
* sum to 1

Softmax converts them into probabilities.

---

# ⭐ Task 5 — Who Chooses the Token?

A developer says:

> “Softmax selects the next token.”

Correct?

### ✅ Answer

No.

Softmax produces:

> a probability distribution.

A decoding strategy selects the token.

For greedy decoding:

# `argmax`

performs the selection.

---

# 🧮 Task 6 — Greedy Selection

Suppose:

```text id="inf-ass-5"
A      → 0.50
B      → 0.31
C      → 0.14
<EOS>  → 0.05
```

What token does greedy decoding choose?

### ✅ Answer

# `A`

because it has the highest probability.

---

# 🔄 Task 7 — What Happens Next?

After greedy decoding selects:

`A`

what happens at the next decoder step?

### ✅ Answer

`A` becomes the next decoder input.

Conceptually:

```text id="inf-ass-6"
A
↓
Embedding
↓
Decoder
↓
Next Hidden State
```

This feedback is what makes decoding:

# **autoregressive**

---

# 📐 Task 8 — Complete the Decoder Equation

Fill the blank:

`d_t = f_dec(e(__________), d_(t-1), c)`

during free-running inference.

### ✅ Answer

# `ŷ_(t-1)`

So:

`d_t = f_dec(e(ŷ_(t-1)), d_(t-1), c)`

where:

`ŷ_(t-1)`

is the model-selected previous token.

---

# 🧠 Task 9 — Why Is It Called Autoregressive?

Choose the best answer.

### A

Because every output token is predicted independently.

### B

Because the model uses its own previous outputs to generate future outputs.

### ✅ Answer

# B

Autoregressive generation means:

> generated history conditions future generation.

---

# 🧮 Task 10 — Full Sequence Probability

Suppose the generated sequence is:

```text id="inf-ass-7"
A B <EOS>
```

with:

```text id="inf-ass-8"
P(A|X) = 0.8
P(B|A,X) = 0.6
P(<EOS>|A,B,X) = 0.9
```

Calculate:

`P(Y|X)`.

### ✅ Answer

`0.8 × 0.6 × 0.9`

# `= 0.432`

---

# 📐 Task 11 — Probability Factorization

Complete:

# `P(Y|X) = __________________________`

### ✅ Answer

# `Π_t P(y_t | y_<t, X)`

This is the core autoregressive factorization.

---

# 🧠 Task 12 — Why Use Log Probabilities?

Why do decoding algorithms often calculate:

`Σ log p_t`

instead of:

`Π p_t`?

### ✅ Answer

Because multiplying many probabilities produces:

> extremely small numbers.

Logs convert multiplication into addition:

# `log P(Y|X) = Σ_t log P(y_t | y_<t, X)`

This improves numerical stability.

---

# 🧮 Task 13 — Compare Sequence Scores

Sequence A:

```text id="inf-ass-9"
0.8 × 0.5
```

Sequence B:

```text id="inf-ass-10"
0.6 × 0.9
```

Which has higher total probability?

### ✅ Answer

Sequence A:

`0.4`

Sequence B:

`0.54`

Therefore:

# Sequence B

has higher overall probability.

---

# 🚨 Task 14 — Why Greedy Can Fail

At step 1:

```text id="inf-ass-11"
A → 0.60
B → 0.40
```

Best next-step continuation:

```text id="inf-ass-12"
after A → 0.20
after B → 0.90
```

Which path is better overall?

### ✅ Answer

A:

`0.60 × 0.20 = 0.12`

B:

`0.40 × 0.90 = 0.36`

So:

# B-path is better.

Greedy would have chosen A and missed it.

This demonstrates:

# **best local token ≠ best complete sequence**

---

# 🛑 Task 15 — Stopping Conditions

List the two common stopping conditions during inference.

### ✅ Answer

1. Model selects:

# `<EOS>`

2. Generated length reaches:

# `max_length`

So:

```text id="inf-ass-13"
if token == <EOS>
→ stop

if length >= max_length
→ stop
```

---

# 🧠 Task 16 — Why Is `max_length` Needed?

A developer says:

> “We already have `<EOS>`, so max length is unnecessary.”

Why is this unsafe?

### ✅ Answer

The model may fail to produce `<EOS>`.

For example:

```text id="inf-ass-14"
A B B B B B B ...
```

Without a hard limit, decoding could continue excessively.

`max_length` is:

> a safety guard, not a learned stopping rule.

---

# 🚫 Task 17 — `<EOS>` vs `PAD`

Classify:

### `<EOS>`

### `PAD`

### ✅ Answer

```text id="inf-ass-15"
<EOS>
→ semantic end of sequence

PAD
→ artificial batching placeholder
```

They are not interchangeable.

---

# 🚨 Task 18 — Diagnose Early Termination

Expected output:

```text id="inf-ass-16"
A B C D
```

Generated output:

```text id="inf-ass-17"
A <EOS>
```

Is the inference loop broken?

### ✅ Answer

Not necessarily.

The decoding loop is correctly obeying:

`<EOS>`.

The issue is that:

> the model assigned too much probability to `<EOS>` too early.

Possible causes include:

* model training
* decoder history
* length bias
* data quality
* exposure bias

---

# 🔁 Task 19 — Diagnose Repetition

Generated output:

```text id="inf-ass-18"
A B B B B B B
```

Explain one autoregressive mechanism that can produce this.

### ✅ Answer

```text id="inf-ass-19"
B generated
↓
B fed back
↓
new decoder state strongly favors B
↓
B selected again
↓
loop reinforces itself
```

This is a self-reinforcing decoding pattern.

---

# 🚨 Task 20 — Does Max Length Fix Repetition?

### ✅ Answer

No.

It only limits how long the repetition can continue.

Example:

```text id="inf-ass-20"
max_length = 20
```

might produce:

```text id="inf-ass-21"
B B B B ... until token 20
```

The underlying quality problem remains.

---

# 🧠 Task 21 — Error Propagation

Correct sequence:

```text id="inf-ass-22"
A B C D
```

Model generates:

```text id="inf-ass-23"
A X
```

What conditional distribution is used next?

### ✅ Answer

The model now predicts based on:

# `P(y_3 | A, X, X_source)`

instead of:

# `P(y_3 | A, B, X_source)`

The wrong token changes the future conditioning context.

---

# 🔗 Task 22 — Exposure Bias or Error Propagation?

Classify:

### A

The model trains on ground-truth histories but infers on generated histories.

### B

One wrong generated token changes later decoder states.

### ✅ Answer

```text id="inf-ass-24"
A → Exposure Bias
B → Error Propagation
```

---

# 🧠 Task 23 — Training vs Inference

Fill the blanks:

### Teacher-forced training

Previous decoder token:

`__________`

### Free-running inference

Previous decoder token:

`__________`

### ✅ Answer

Training:

# ground-truth previous token

Inference:

# model-generated previous token

---

# 🚨 Task 24 — Is Inference Fully Parallel?

Can a recurrent autoregressive decoder generate:

```text id="inf-ass-25"
y1, y2, y3, y4
```

all at the same time?

### ✅ Answer

No.

Because:

```text id="inf-ass-26"
d2 depends on d1 and ŷ1
d3 depends on d2 and ŷ2
```

So recurrent inference is:

> sequential across target time.

---

# 🧮 Task 25 — Latency Reasoning

Suppose average decoder time per step is:

`4 ms`

Ignoring other overhead, approximately how long would 100 autoregressive decoding steps take?

### ✅ Answer

`100 × 4 ms`

# `= 400 ms`

This illustrates why output length directly affects autoregressive latency.

---

# 📐 Task 26 — Vocabulary Projection Cost

Suppose:

```text id="inf-ass-27"
H = 512
V = 50,000
```

How many weight values are in:

`W_o`

ignoring bias?

### ✅ Answer

`50,000 × 512`

# `= 25,600,000`

parameters.

So the output projection can be substantial.

---

# 🎲 Task 27 — Greedy vs Sampling

Distribution:

```text id="inf-ass-28"
A → 0.55
B → 0.30
C → 0.15
```

Which statement is correct?

### A

Greedy can choose B.

### B

Sampling can choose B.

### ✅ Answer

# B

Greedy always chooses:

`A`.

Sampling may choose:

* A
* B
* C

according to the sampling rule.

---

# 🧠 Task 28 — Model vs Decoding Strategy

Two systems use exactly the same model probabilities.

System A uses greedy.

System B uses sampling.

Can their outputs differ?

### ✅ Answer

Yes.

The model defines:

> probabilities.

The decoding strategy defines:

> how tokens are selected from those probabilities.

So:

```text id="inf-ass-29"
Same Model
+
Different Decoding Policy
=
Different Generated Behavior
```

---

# 🚫 Task 29 — Inference vs Evaluation

Classify:

### A

Generate a French translation from an English sentence.

### B

Compare the generated translation with a reference using BLEU.

### ✅ Answer

```text id="inf-ass-30"
A → Inference
B → Evaluation
```

Inference generates.

Evaluation measures quality.

---

# 📦 Task 30 — Batched Decoding

Three sequences are decoding together:

```text id="inf-ass-31"
A → emits <EOS>
B → continues
C → continues
```

What should happen to A?

### ✅ Answer

A should be:

> marked finished / masked.

It should not keep accumulating new generated tokens while B and C continue.

---

# ✅ True / False

| #  | Statement                                                                     | Answer  |
| -- | ----------------------------------------------------------------------------- | ------- |
| 1  | Inference normally has access to the correct target sequence.                 | ❌ False |
| 2  | `<BOS>` can start autoregressive decoding.                                    | ✅ True  |
| 3  | Decoder hidden state is already a token probability distribution.             | ❌ False |
| 4  | Output projection produces vocabulary logits.                                 | ✅ True  |
| 5  | Softmax itself chooses the next token.                                        | ❌ False |
| 6  | Greedy decoding uses argmax.                                                  | ✅ True  |
| 7  | Generated tokens are fed back during autoregressive inference.                | ✅ True  |
| 8  | `<EOS>` and `PAD` serve the same purpose.                                     | ❌ False |
| 9  | A max-length limit is useful even when `<EOS>` exists.                        | ✅ True  |
| 10 | Greedy decoding guarantees the best complete sequence.                        | ❌ False |
| 11 | A wrong generated token can change future distributions.                      | ✅ True  |
| 12 | Raw sequence probabilities generally shrink as length grows.                  | ✅ True  |
| 13 | Log probabilities are useful for sequence scoring.                            | ✅ True  |
| 14 | Autoregressive RNN decoding is sequential across target steps.                | ✅ True  |
| 15 | Changing decoding strategy can change outputs without changing model weights. | ✅ True  |

---

# ⭐ Staff Engineer Challenge — Production Autoregressive Service

## Scenario

You operate a summarization service:

```text id="inf-ass-32"
Encoder
↓
RNN Decoder
↓
Greedy Search
```

Production metrics show:

```text id="inf-ass-33"
Average output = 70 tokens
P95 output = 250 tokens

Some outputs:
→ repeat phrases

Some:
→ terminate after 5–10 tokens

Others:
→ hit max_length

P95 latency:
→ much worse than P50
```

---

## Question 1 — Why is P95 latency likely correlated with output length?

### ✅ Answer

Because generation is autoregressive.

Each output token requires another sequential decoder step.

So roughly:

```text id="inf-ass-34"
Longer Generation
↓
More Sequential Steps
↓
Higher Latency
```

---

## Question 2 — What metrics would you add?

### ✅ Answer

Useful metrics include:

* generated token count
* time per generated token
* tokens/sec
* `<EOS>` position
* percentage hitting max length
* repetition rate
* early termination rate
* output quality vs generated length

This helps distinguish:

> latency problems from quality problems.

---

## Question 3 — 15% of requests hit max length. What does this tell you?

### ✅ Answer

It indicates a significant fraction of generations are:

> failing to terminate naturally before the safeguard.

Potential reasons include:

* repetition
* poorly learned `<EOS>`
* decoding behavior
* model calibration
* training distribution

It does not mean max length itself is too low automatically.

---

## Question 4 — Would increasing max length fix the issue?

### ✅ Answer

Not necessarily.

It may simply produce:

> longer bad outputs

and increase latency.

The correct investigation is:

> why the model is failing to terminate naturally.

---

## Question 5 — Why might greedy decoding contribute to quality problems?

### ✅ Answer

Greedy commits immediately to:

> one locally best token.

Once selected, alternatives disappear.

Because each selection changes future history, one poor local choice can lead the sequence into:

> a lower-quality trajectory.

---

## Question 6 — What search change would you investigate next?

### ✅ Answer

Retain several promising candidate sequences instead of one.

Conceptually:

```text id="inf-ass-35"
Step 1
→ keep several candidates

Step 2
→ expand each

Score complete partial sequences
↓
retain strongest candidates
```

This leads directly to:

# **Beam Search**

---

## Question 7 — What trade-off do you expect from Beam Search?

### ✅ Answer

Potential benefit:

> better sequence-level search.

Costs:

* more compute
* more memory
* higher latency

So production evaluation should compare:

```text id="inf-ass-36"
quality gain
vs
latency / compute cost
```

---

# 🎯 Final Applied Exercise

Complete the inference story:

```text id="inf-ass-37"
Source
↓
Encoder
↓
Decoder starts with __________
↓
Produces __________
↓
Output projection produces __________
↓
Softmax produces __________
↓
__________ chooses token
↓
Selected token is __________
↓
Repeat until __________ or __________
```

### ✅ Answer

```text id="inf-ass-38"
Source
↓
Encoder
↓
Decoder starts with <BOS>/<SOS>
↓
Produces HIDDEN STATE
↓
Output projection produces LOGITS
↓
Softmax produces TOKEN PROBABILITY DISTRIBUTION
↓
DECODING STRATEGY chooses token
↓
Selected token is FED BACK AS NEXT INPUT
↓
Repeat until <EOS> or MAX_LENGTH
```

---

# 🧠 Final Assignment Mental Model

The core inference loop:

```text id="inf-ass-39"
Previous Token
↓
Embedding
↓
Decoder
↓
Hidden State
↓
Logits
↓
Softmax
↓
Distribution
↓
Selection
↓
Generated Token
└──────────────┐
               │
               ▼
        Next Decoder Step
```

The architectural consequence:

```text id="inf-ass-40"
Generated Token
↓
Future History
↓
Future State
↓
Future Predictions
```

Therefore:

# **every decoding choice can change the entire future trajectory.**

Greedy decoding does:

```text id="inf-ass-41"
Best token now
↓
commit immediately
```

But sequence generation really wants:

```text id="inf-ass-42"
Best complete sequence
```

That mismatch leads naturally to:

# `13_Beam_Search`

---

# ⭐ Golden Rule

> **When debugging autoregressive inference, trace the complete loop—not just the model output: hidden state → logits → probabilities → selection → feedback → next state. Most generation failures become easier to reason about once you identify exactly where that loop is going wrong.**
