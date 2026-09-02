# 📘 Lecture — Inference and Autoregressive Decoding

## 🌟 Introduction

In the previous lecture, we studied **Teacher Forcing**.

During training, we often give the decoder:

> the correct previous target token.

For example:

```text
<SOS>
↓
predict y1
↓
feed TRUE y1
↓
predict y2
↓
feed TRUE y2
```

But at inference time:

> the correct target does not exist yet.

There is no teacher.

So the decoder must generate the sequence by using:

> its own previous predictions.

That gives us:

# **Autoregressive Decoding**

This lecture answers a very practical question:

# **How does a Seq2Seq decoder actually generate a sequence when the target is unknown?**

We will carefully trace:

* `<BOS>` / `<SOS>`
* decoder state
* logits
* softmax probabilities
* token selection
* feedback
* `<EOS>`
* maximum generation length
* greedy decoding
* error propagation
* training vs inference
* decoding latency
* why sequence generation is fundamentally different from classification

---

# 🎯 Learning Objectives

By the end of this lecture, you should be able to:

## 🧠 Conceptual Understanding

* Explain what inference means in Seq2Seq.
* Explain autoregressive decoding.
* Understand why generated tokens must be fed back.
* Distinguish model probability computation from token selection.
* Explain the role of `<BOS>` and `<EOS>`.
* Understand why inference can accumulate errors.

## 📐 Mathematical Understanding

* Understand:

`P(y_t | y_<t, X)`

* Understand sequence factorization:

`P(Y|X) = Π_t P(y_t | y_<t, X)`

* Understand logits → softmax → decoding decision.
* Understand why probability distributions and chosen tokens are different objects.

## 🧮 Practical Understanding

* Trace a complete decoding loop.
* Understand greedy token selection.
* Understand termination conditions.
* Understand maximum-length safeguards.
* Diagnose repetition, early termination, and runaway generation.

## 🔗 Architectural Understanding

* Connect inference with:

  * Teacher Forcing
  * Exposure Bias
  * Seq2Seq
  * Greedy Decoding
  * Beam Search

---

# 📖 Part 1 — Training Had an Advantage

Suppose target is:

```text
A B C <EOS>
```

During training with teacher forcing:

```text
Decoder Input:
<SOS> A B C

Target:
A B C <EOS>
```

The decoder always knows:

> the correct previous token.

But now imagine deployment.

User provides only:

```text
Source X
```

The target sequence is unknown.

So we cannot feed:

```text
A
B
C
```

because those are precisely what the model must generate.

This changes everything.

---

# 🧠 Part 2 — What Information Exists at Inference?

At inference, we have:

```text
Source Sequence X
```

The encoder processes it.

Then the decoder receives:

* encoder context/state
* initial decoder state
* `<SOS>` / `<BOS>`

That is enough to produce:

> the first target-token probability distribution.

After that, the model must continue on its own.

---

# ⭐ Part 3 — The First Decoder Step

Suppose:

```text
Source:
"I love AI"
```

The encoder creates source representation:

`c`

The decoder begins with:

`<SOS>`

So:

```text
Source
↓
Encoder
↓
Context c

<SOS>
+
Initial Decoder State
↓
Decoder
↓
d1
```

Now `d1` is a hidden state.

It is not yet:

> a token.

---

# 📐 Part 4 — Hidden State → Logits

The decoder state:

`d_t`

is projected to vocabulary logits:

# `z_t = W_o d_t + b_o`

If target vocabulary size is:

`V`

then:

`z_t ∈ R^V`

Example:

```text
Token A → 3.2
Token B → 1.1
Token C → -0.7
<EOS>   → 0.4
...
```

These are:

# **logits**

They are not probabilities.

---

# ⭐ Part 5 — Logits → Probability Distribution

Apply softmax:

# `p_t = softmax(z_t)`

Now:

```text
A      → 0.61
B      → 0.18
C      → 0.04
<EOS>  → 0.03
...
```

This represents:

# `P(y_t | y_<t, X)`

Very important:

> **softmax does not choose the token.**

Softmax only produces:

> a probability distribution.

---

# 🧠 Part 6 — Probability Distribution vs Decoding Strategy

Once we have:

```text
A → 0.61
B → 0.18
C → 0.04
...
```

we need a rule for selecting the next token.

That rule is:

# **the decoding strategy**

Examples include:

* Greedy decoding
* Sampling
* Beam Search
* Top-k sampling
* Top-p sampling

For our recurrent Seq2Seq story, we first focus on:

# **Greedy Decoding**

---

# ⭐ Part 7 — Greedy Selection

Greedy decoding chooses:

# `ŷ_t = argmax_y P(y | y_<t, X)`

Example:

```text
A → 0.61
B → 0.18
C → 0.04
```

Greedy selects:

# `A`

because it has the highest probability.

---

# 🔄 Part 8 — Feed the Prediction Back

Now comes the defining autoregressive step.

The generated token:

`A`

becomes the decoder input for the next step.

So:

```text
<SOS>
↓
predict A
↓
feed A
↓
predict B
↓
feed B
↓
predict C
```

This is:

# **autoregressive decoding**

---

# 🧠 Part 9 — What Does Autoregressive Mean?

Autoregressive means:

> future outputs are generated conditioned on previously generated outputs.

Mathematically:

# `P(y_t | y_1, ..., y_(t-1), X)`

or compactly:

# `P(y_t | y_<t, X)`

The generated history becomes part of the model's future input.

---

# 📐 Part 10 — Sequence Probability

For target:

`Y = (y_1, y_2, ..., y_T)`

the full sequence probability is:

# `P(Y|X) = Π_t P(y_t | y_<t, X)`

Example:

```text
P(A|X) = 0.8

P(B|A,X) = 0.7

P(<EOS>|A,B,X) = 0.9
```

Then:

`P(Y|X) = 0.8 × 0.7 × 0.9`

# `= 0.504`

Autoregressive decoding operationalizes this factorization one step at a time.

---

# 📖 Part 11 — Complete Inference Example

Suppose target vocabulary contains:

```text
A
B
C
<EOS>
```

### Step 0

Input:

`<SOS>`

Decoder produces:

```text
A → 0.70
B → 0.15
C → 0.10
<EOS> → 0.05
```

Greedy chooses:

`A`

---

### Step 1

Feed:

`A`

Decoder produces:

```text
A → 0.08
B → 0.75
C → 0.12
<EOS> → 0.05
```

Choose:

`B`

---

### Step 2

Feed:

`B`

Decoder produces:

```text
C → 0.72
<EOS> → 0.20
...
```

Choose:

`C`

---

### Step 3

Feed:

`C`

Decoder produces:

```text
<EOS> → 0.84
...
```

Choose:

`<EOS>`

Stop.

Final sequence:

```text
A B C
```

---

# 🛑 Part 12 — Why `<EOS>` Is Critical

The decoder does not generally know output length in advance.

So it needs to learn:

> when the sequence is complete.

That is the job of:

# `<EOS>`

When the decoder predicts `<EOS>`:

```text
Generation
↓
STOP
```

Without this, variable-length generation would be much harder to control.

---

# 🚨 Part 13 — What If `<EOS>` Never Appears?

A poorly trained model might keep generating:

```text
A B C C C C C C ...
```

forever.

Production systems therefore usually include:

# **maximum decoding length**

For example:

```text
max_length = 100
```

Stop when either:

```text
<EOS> predicted
```

or:

```text
generated_length >= max_length
```

---

# ⭐ Part 14 — Two Termination Conditions

A practical decoding loop often uses:

```text
if token == <EOS>:
    stop
```

plus:

```text
if timestep == max_length:
    stop
```

So:

# **semantic termination + safety termination**

`<EOS>`:

> model-decided stopping.

`max_length`:

> system safeguard.

---

# 🧠 Part 15 — `<EOS>` vs `PAD`

Do not confuse them.

### `<EOS>`

Means:

> sequence is finished.

### `PAD`

Means:

> artificial placeholder used for batching.

So:

```text
A B <EOS> PAD PAD
```

means:

```text
A B
```

is the actual sequence.

---

# 🔄 Part 16 — Training vs Inference Revisited

### Training with Teacher Forcing

```text
<SOS>
↓
predict y1
↓
feed TRUE y1
↓
predict y2
```

### Inference

```text
<SOS>
↓
predict ŷ1
↓
feed ŷ1
↓
predict ŷ2
```

That difference is exactly why:

> exposure bias exists.

---

# 🚨 Part 17 — One Wrong Token Changes the Future

Suppose correct output is:

```text
A B C D
```

But inference generates:

```text
A X
```

Now the model's next conditional distribution becomes:

# `P(y_3 | A, X, X_source)`

instead of:

# `P(y_3 | A, B, X_source)`

These are different distributions.

So an early mistake changes:

> the future generation trajectory.

---

# 🧠 Part 18 — Error Propagation

The process is:

```text
Wrong Token
↓
Wrong / Different Next Input
↓
Different Decoder State
↓
Different Probability Distribution
↓
Possible New Error
```

This is:

# **autoregressive error propagation**

It is a consequence of:

> feeding generated outputs back into the model.

---

# ⭐ Part 19 — Exposure Bias vs Error Propagation

Important distinction:

### Exposure Bias

Training problem:

```text
training sees ground-truth history
inference sees generated history
```

### Error Propagation

Inference behavior:

```text
one generated error affects future predictions
```

So:

```text
Exposure Bias
→ model may be poorly prepared for bad histories

Error Propagation
→ bad history actually changes future decoding
```

---

# 📖 Part 20 — Why Greedy Decoding Seems Reasonable

At each step, greedy decoding asks:

> Which token is most likely right now?

So:

```text
ŷ_t = argmax P(y_t | history, X)
```

This is:

* simple
* fast
* deterministic
* low memory

That makes it a natural baseline.

---

# 🚨 Part 21 — But Greedy Makes a Local Decision

Suppose first step:

```text
A → 0.55
B → 0.45
```

Greedy chooses:

`A`

But perhaps:

```text
Best continuation after A = 0.20
```

while:

```text
Best continuation after B = 0.90
```

Then sequence scores:

```text
A path:
0.55 × 0.20 = 0.11
```

```text
B path:
0.45 × 0.90 = 0.405
```

So:

> the best first token did not lead to the best full sequence.

This is the key limitation of greedy decoding.

---

# ⭐ Part 22 — Local Optimum vs Sequence Optimum

Greedy solves:

# `best token now`

But sequence generation wants:

# `best complete sequence`

Those objectives are not identical.

So:

```text
Locally Best Choice
≠
Globally Best Sequence
```

This will motivate:

# **Beam Search**

in the next chapter.

---

# 📐 Part 23 — Sequence Scores in Log Space

Sequence probability multiplies many numbers less than 1:

`0.8 × 0.7 × 0.9 × ...`

For long sequences, this becomes very small.

So decoding algorithms usually work with:

# **log probabilities**

Because:

`log(ab) = log(a) + log(b)`

So:

# `log P(Y|X) = Σ_t log P(y_t | y_<t, X)`

This is numerically more stable.

---

# 🧠 Part 24 — Why Longer Sequences Naturally Have Lower Raw Probability

Suppose each generated token has average probability:

`0.8`

Then:

### 2 tokens

`0.8² = 0.64`

### 10 tokens

`0.8¹⁰ ≈ 0.107`

### 50 tokens

`0.8⁵⁰ ≈ 0.000014`

So raw sequence probability:

> naturally shrinks with sequence length.

This becomes important for:

> Beam Search length normalization

later.

---

# ⭐ Part 25 — Greedy Decoder Pseudocode

Conceptually:

```text
encode source

token = <BOS>
state = decoder_initial_state

for t in 1 ... max_length:

    embedding = embed(token)

    state = decoder(embedding, state, context)

    logits = output_projection(state)

    probabilities = softmax(logits)

    token = argmax(probabilities)

    if token == <EOS>:
        break

    append token to output
```

This is the classic autoregressive inference loop.

---

# 🧠 Part 26 — Important Order of Operations

Remember:

```text
Previous Token
↓
Embedding
↓
Decoder
↓
Hidden State
↓
Linear Projection
↓
Logits
↓
Softmax
↓
Probability Distribution
↓
Decoding Strategy
↓
Selected Token
↓
Feedback
```

Do not collapse these into:

> “decoder gives a word.”

There are several distinct transformations.

---

# 🚫 Part 27 — Softmax Does Not Decode

A very common interview mistake:

> “Softmax selects the next token.”

No.

Softmax computes:

> probability distribution.

Then a decoding strategy selects.

For greedy:

```text
softmax
↓
argmax
```

For sampling:

```text
softmax
↓
sample
```

For beam search:

```text
scores
↓
expand multiple candidates
```

---

# ⭐ Part 28 — Argmax vs Sampling

Suppose:

```text
A → 0.50
B → 0.30
C → 0.20
```

### Greedy

Always picks:

`A`

### Sampling

May choose:

* A
* B
* C

according to probabilities.

Greedy is deterministic.

Sampling introduces:

> stochasticity.

For classic translation systems, greedy or beam search were common.

---

# 🧠 Part 29 — Why Sampling Is Useful in Generative Tasks

For tasks with many valid outputs:

```text
Story generation
Dialogue
Creative writing
```

always choosing argmax can produce:

> repetitive or overly predictable output.

Sampling can increase:

> diversity.

But it may also increase:

> errors or incoherence.

This becomes more important in modern LLM decoding.

---

# 🔗 Part 30 — Translation vs Open-Ended Generation

For translation:

> fidelity matters strongly.

So historically:

* Greedy
* Beam Search

are natural choices.

For creative generation:

> diversity may matter more.

So sampling strategies become more useful.

The decoding strategy should match:

> the task objective.

---

# 🧠 Part 31 — Decoder State vs Generated History

In an RNN decoder, history exists in two ways:

### Explicit Token History

Previous generated tokens.

### Hidden State

Compressed recurrent summary of prior decoding steps.

At step `t`:

```text
previous token
+
previous hidden state
+
source context
↓
new decoder state
```

So generated history affects future predictions both:

> directly through the token input

and:

> indirectly through recurrent state.

---

# 📐 Part 32 — Decoder Equation During Inference

A simplified recurrent decoder:

# `d_t = f_dec(e(ŷ_(t-1)), d_(t-1), c)`

where:

`ŷ_(t-1)`

is:

> the token selected by the model.

Then:

`z_t = W_o d_t + b_o`

and:

`P(y_t | ...) = softmax(z_t)`

Then decoding strategy chooses:

`ŷ_t`.

---

# ⭐ Part 33 — Why Inference Cannot Be Fully Parallelized in an Autoregressive RNN

To compute:

`d_2`

we need:

`d_1`

and usually:

`ŷ_1`

To compute:

`d_3`

we need:

`d_2`

and:

`ŷ_2`

So:

```text
Step 1
↓
Step 2
↓
Step 3
↓
Step 4
```

must occur sequentially.

This creates:

> inference latency proportional to generated length.

---

# 🧠 Part 34 — Long Output = More Sequential Steps

If output has:

```text
10 tokens
```

decoder performs roughly:

```text
10 recurrent generation steps
```

If output has:

```text
500 tokens
```

it performs roughly:

```text
500 sequential generation steps
```

So autoregressive generation has a fundamental:

> sequential dependency.

---

# ⭐ Part 35 — Training vs Inference Parallelism Nuance

During teacher-forced training, the entire target sequence is known.

For Transformers, this allows many target positions to be computed in parallel under causal masking.

But for an RNN decoder:

```text
d_t depends on d_(t-1)
```

so recurrent states remain sequential.

At inference, both architectures are autoregressive when generating one token at a time.

This distinction will matter greatly when we reach Transformers.

---

# 🚨 Part 36 — Early `<EOS>` Problem

Suppose model outputs:

```text
A <EOS>
```

when expected output is much longer.

Possible reasons include:

* model learned overly high `<EOS>` probability
* poor training
* decoder history errors
* length bias
* data issues

The decoding loop itself is behaving correctly:

> it stops because `<EOS>` was selected.

The problem is:

> the model's predicted distribution.

---

# 🧠 Part 37 — Repetition Problem

A model may produce:

```text
A B B B B B...
```

Why?

Once `B` is generated:

```text
B
↓
fed back
↓
state predicts B again
↓
B fed back again
```

A self-reinforcing loop can emerge.

This is especially common in poorly calibrated autoregressive models.

---

# ⭐ Part 38 — Maximum-Length Safeguard Does Not Fix Repetition

If:

```text
max_length = 100
```

then repetition may simply become:

```text
B B B B ... until step 100
```

The safeguard prevents infinite decoding.

It does not improve:

> generation quality.

---

# 🧠 Part 39 — Invalid Token Masking

In some tasks, certain tokens may be impossible or illegal at a given step.

Before selection, systems can modify logits:

```text
invalid token
→ score = -∞
```

After softmax:

> probability becomes effectively 0.

This is called:

> constrained decoding / masking.

Conceptually, it controls:

> which tokens the decoder is allowed to select.

---

# 🔗 Part 40 — Why This Matters Beyond Translation

Autoregressive decoding appears in:

* machine translation
* summarization
* speech recognition
* image captioning
* code generation
* chatbots
* modern LLMs

The core loop remains:

```text
History
↓
Predict Next Token
↓
Select Token
↓
Append to History
↓
Repeat
```

---

# ⭐ Part 41 — LLM Connection

Modern decoder-only LLM generation follows the same high-level idea.

Given prompt:

```text
"The capital of France is"
```

model produces:

```text
P(next token | prompt)
```

Select:

`Paris`

Then new context becomes:

```text
"The capital of France is Paris"
```

Now predict the next token.

So:

# **modern LLM generation is autoregressive decoding at massive scale.**

---

# 🧠 Part 42 — Training vs Generation in LLMs

Training:

```text
correct prefix
→ predict next token
```

Generation:

```text
generated prefix
→ predict next token
```

Same fundamental distinction we saw with Teacher Forcing.

This is why these early Seq2Seq concepts are foundational for understanding LLMs.

---

# 📐 Part 43 — Greedy Decoding Complexity Intuition

Suppose:

* target length = `T'`
* vocabulary size = `V`

At each generation step:

> model produces scores over `V` tokens.

Very roughly, output selection work grows with:

# `T' × V`

plus the model's hidden-state computation.

This helps explain why:

* large vocabularies
* long outputs

increase inference cost.

---

# ⭐ Part 44 — Output Projection Can Be Expensive

If:

```text
H = 512
V = 50,000
```

then:

`W_o ∈ R^(50000 × 512)`

Every decoder step needs to produce:

> 50,000 logits.

So vocabulary projection can be a significant component of decoding cost.

This becomes even more important in large language models.

---

# 🧠 Part 45 — Batch Inference

Can multiple sequences be decoded together?

Yes.

For example:

```text
Sequence A → currently step 4
Sequence B → currently step 4
Sequence C → currently step 4
```

can often be batched.

But sequences may finish at different times:

```text
A → <EOS>
B → continues
C → continues
```

So production systems need to manage:

> variable completion lengths.

---

# 🚨 Part 46 — Finished Sequences Should Not Keep Generating

Once sequence A emits:

`<EOS>`

it should be marked:

> finished.

Otherwise later batched decoding might accidentally append extra tokens.

So batching needs:

> completion masks.

---

# 🧠 Part 47 — Greedy Decoding Is Deterministic Only if the Model Is Deterministic

Given fixed:

* model weights
* inputs
* numerical environment

greedy decoding normally produces the same argmax sequence.

Sampling may produce different sequences because of:

> randomness.

This is useful when discussing reproducibility.

---

# ⭐ Part 48 — Inference Has No Ground-Truth Loss

During production inference:

> we usually do not know the correct target.

So we typically do not compute:

```text
cross-entropy against the correct sequence
```

during generation.

Instead we produce:

> output.

Evaluation is a separate process where references may exist.

---

# 🧠 Part 49 — Inference vs Evaluation

### Inference

Generate output for an input.

### Evaluation

Measure output quality using:

* reference target
* task metric
* human evaluation
* model-based evaluation

Do not confuse:

> generating predictions

with:

> calculating their quality.

---

# 🔗 Part 50 — Problem Map So Far

We can now distinguish:

```text
Context Bottleneck
→ source information access

Teacher Forcing
→ training history strategy

Exposure Bias
→ train/inference mismatch

Autoregressive Decoding
→ inference feedback process

Greedy Decoding
→ local token-selection strategy

Beam Search
→ multi-candidate sequence search
```

These are separate pieces of the Seq2Seq system.

---

# 🎤 Part 51 — 30-Second Interview Answer

> **Autoregressive decoding is the inference process where a sequence model generates one target token at a time, conditioning each new prediction on the source and the previously generated target tokens. The decoder starts with a `<BOS>` token, produces vocabulary logits, converts them to probabilities, uses a decoding strategy such as greedy argmax to select a token, feeds that token back as the next input, and repeats until `<EOS>` or a maximum-length limit is reached. Because each generated token affects future decoder states, errors can propagate, and inference is inherently sequential.**

---

# 🎤 Part 52 — What Is the Difference Between Softmax and Greedy Decoding?

Strong answer:

> **Softmax converts logits into a probability distribution over the vocabulary. Greedy decoding is a separate decision rule that selects the highest-probability token from that distribution. Softmax scores possibilities; greedy decoding chooses one.**

---

# 🎤 Part 53 — Why Is Autoregressive Inference Sequential?

Strong answer:

> **Because the next decoder step depends on the previous decoder state and usually the previously generated token. The model cannot know token `t+1` until token `t` has been selected and fed back, so generation proceeds step by step.**

---

# 🎤 Part 54 — Why Can Greedy Decoding Be Suboptimal?

Strong answer:

> **Greedy decoding maximizes the probability of the next token locally, but the locally best token may lead to a low-probability continuation. Therefore it does not guarantee the highest-probability complete sequence, which motivates sequence-level search methods such as beam search.**

---

# 🚫 Part 55 — High-Yield Traps

### ❌ Softmax chooses the token

No.

Softmax produces probabilities.

---

### ❌ Decoder hidden state is the generated token

No.

It must be projected to logits first.

---

### ❌ Inference uses ground-truth previous targets

No.

Not in normal free-running generation.

---

### ❌ Autoregressive decoding predicts all tokens simultaneously

No.

Token-by-token.

---

### ❌ `<EOS>` and `PAD` are the same

No.

---

### ❌ Maximum length tells the model what the correct target length is

No.

It is a safety limit.

---

### ❌ Greedy decoding finds the globally best sequence

No.

---

### ❌ One prediction error affects only one timestep

No.

It can alter future history and states.

---

### ❌ Beam Search changes the model probabilities

No.

It changes how candidate sequences are searched.

---

### ❌ Inference and evaluation are the same thing

No.

---

# 📐 Formula Flash Card

### Decoder Recurrence

# `d_t = f_dec(e(ŷ_(t-1)), d_(t-1), c)`

### Output Logits

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

```text
token == <EOS>
OR
t >= max_length
```

---

# 🧠 Part 56 — 10 Things You Must Know Cold

1. **Inference starts without the correct target sequence.**
2. **`<BOS>` / `<SOS>` starts decoding.**
3. **Decoder state is projected into vocabulary logits.**
4. **Softmax creates probabilities; it does not choose a token.**
5. **A decoding strategy selects the next token.**
6. **Autoregressive decoding feeds generated tokens back.**
7. **`<EOS>` provides learned sequence termination.**
8. **Maximum length is a safety guard.**
9. **Errors can propagate through future decoder states.**
10. **Greedy decoding is locally optimal, not globally optimal.**

---

# 🧠 Final Mental Model

Inference begins with:

```text
SOURCE
↓
ENCODER
↓
SOURCE REPRESENTATION
```

Then:

```text
<BOS>
  │
  ▼
DECODER
  │
  ▼
HIDDEN STATE
  │
  ▼
LOGITS
  │
  ▼
SOFTMAX
  │
  ▼
TOKEN DISTRIBUTION
  │
  ▼
DECODING STRATEGY
  │
  ▼
SELECTED TOKEN
  │
  └───────────────┐
                  │
                  ▼
          NEXT DECODER STEP
```

Continue until:

```text
<EOS>
```

or:

```text
max_length
```

The fundamental feedback loop is:

# **Predict → Select → Feed Back → Predict Again**

And the key weakness of greedy generation is:

```text
Best Token Now
≠
Best Complete Sequence
```

---

# 🚀 Where We Go Next

We now know exactly how free-running inference works.

But we discovered a problem.

Suppose at the first step:

```text
A → 0.51
B → 0.49
```

Greedy immediately commits to:

`A`

and permanently discards:

`B`.

But what if the sequence beginning with `B` becomes much better later?

Instead of keeping only:

> one candidate sequence,

what if we keep:

> several promising sequences

at every decoding step?

That leads directly to our next lecture:

# `13_Beam_Search`

The progression is:

```text
Teacher Forcing
↓
Inference Has No Teacher
↓
Autoregressive Decoding
↓
Greedy Token Selection
↓
Local Decisions Can Be Suboptimal
↓
Beam Search ← NEXT
```

---

# ⭐ Golden Rule

> **Autoregressive decoding is a feedback loop: the model predicts a distribution, a decoding strategy selects one token, that token becomes part of the next input, and the process repeats until termination—so every decoding decision can influence the entire remainder of the sequence.**
