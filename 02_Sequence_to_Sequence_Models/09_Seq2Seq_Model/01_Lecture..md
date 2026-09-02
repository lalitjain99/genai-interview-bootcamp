# 📘 Lecture — Sequence-to-Sequence (Seq2Seq) Model

## 🌟 Introduction

In the previous lecture, we understood the **Encoder–Decoder architecture**.

We saw the basic idea:

```text
Input Sequence
↓
Encoder
↓
Context / Representation
↓
Decoder
↓
Output Sequence
```

But that was still a fairly high-level architectural view.

Now we want to answer the next question:

# **How does a complete Seq2Seq model actually work from input tokens to output tokens?**

For example:

```text
English:
"I love AI"

↓

French:
"J'aime l'IA"
```

We now need to connect all the pieces:

* source tokens
* embeddings
* encoder hidden states
* context vector
* decoder initial state
* target tokens
* `<SOS>`
* `<EOS>`
* decoder hidden states
* output logits
* softmax probabilities
* training flow
* inference flow

This complete pipeline is what we usually mean by a:

# **Seq2Seq Model**

---

# 🎯 Learning Objectives

By the end of this lecture, you should be able to:

## 🧠 Conceptual Understanding

* Explain what a Seq2Seq model is.
* Explain how it differs from the broader Encoder–Decoder idea.
* Understand the complete information flow from source sequence to target sequence.
* Explain how encoder and decoder cooperate.
* Understand why target generation is autoregressive.

## 📐 Mathematical Understanding

* Represent source and target sequences mathematically.
* Write encoder recurrence.
* Write decoder recurrence.
* Understand decoder token probabilities.
* Understand the factorization:

`P(Y|X) = Π_t P(y_t | y_<t, X)`

## 🧮 Practical Understanding

* Trace a translation example end-to-end.
* Understand source embeddings and target embeddings.
* Understand decoder initialization.
* Understand how `<SOS>` and `<EOS>` are used.
* Distinguish training-time decoding from inference-time decoding.

## 🔗 Architectural Understanding

* Connect Seq2Seq with:

  * RNN
  * LSTM
  * GRU
  * BiRNN
  * Encoder–Decoder
* Understand why the original Seq2Seq model eventually struggles with long sequences.
* Prepare for the next topic:

  * **Context Vector Bottleneck**

---

# 📖 Part 1 — What Exactly Is Seq2Seq?

Seq2Seq stands for:

# **Sequence-to-Sequence**

It is a model designed to transform:

```text
one sequence
↓
another sequence
```

Examples:

```text
English
→ French
```

```text
Article
→ Summary
```

```text
Audio Sequence
→ Text Sequence
```

```text
Question
→ Answer Sequence
```

The key point is:

> both input and output are sequences.

---

# 🧠 Part 2 — Seq2Seq vs Encoder–Decoder

These terms are related, but it helps to keep the distinction clear.

## Encoder–Decoder

This is the architectural pattern:

```text
Encoder
→ Representation
→ Decoder
```

It tells us:

> separate source processing from target generation.

## Seq2Seq

This usually refers to:

> using that architecture specifically to map one sequence to another.

So:

```text
Encoder–Decoder
→ architecture pattern

Seq2Seq
→ sequence-transformation model built using that pattern
```

In practice, people often use the terms interchangeably.

But conceptually:

> Seq2Seq is the concrete sequence-to-sequence use of Encoder–Decoder.

---

# 📖 Part 3 — A Full Example

Suppose we want to translate:

```text
"I love AI"
```

into:

```text
"J'aime l'IA"
```

Let the source tokens be:

```text
x1 = "I"
x2 = "love"
x3 = "AI"
```

Target tokens:

```text
y1 = "J'aime"
y2 = "l'IA"
```

With special target tokens:

```text
<SOS>
J'aime
l'IA
<EOS>
```

Now let us trace everything.

---

# ⭐ Part 4 — Step 1: Tokenization

Before an RNN can process text, words/tokens must become numerical IDs.

Example source vocabulary:

```text
"I"     → 12
"love"  → 47
"AI"    → 205
```

So:

```text
"I love AI"
```

becomes:

```text
[12, 47, 205]
```

But these IDs themselves are not semantic vectors.

We need embeddings.

---

# 🧠 Part 5 — Step 2: Source Embeddings

Each source token ID is mapped to a dense vector.

Suppose embedding dimension:

`D = 4`

Then:

```text
"I"
→ [0.2, -0.4, 0.8, 0.1]
```

```text
"love"
→ [0.7, 0.3, -0.2, 0.5]
```

```text
"AI"
→ [-0.1, 0.9, 0.4, 0.6]
```

These become:

`x_1, x_2, x_3`

for the encoder.

Important:

> the encoder consumes embeddings, not raw words.

---

# 📐 Part 6 — Encoder Recurrence

For a simple recurrent encoder:

`h_t = f_enc(x_t, h_(t-1))`

So:

```text
x1 + h0
↓
h1
```

```text
x2 + h1
↓
h2
```

```text
x3 + h2
↓
h3
```

The encoder processes the source left-to-right.

At the end:

# `c = h_3`

in the original fixed-context Seq2Seq design.

---

# 🧠 Part 7 — What Does the Final Encoder State Represent?

The final state:

`h_3`

is meant to summarize:

```text
"I love AI"
```

into one learned numerical representation.

Conceptually:

```text
Source Tokens
↓
Embeddings
↓
Encoder Recurrence
↓
Final Hidden State
↓
Context Vector
```

That vector is then handed to the decoder.

---

# ⭐ Part 8 — Step 3: Initialize the Decoder

A simple setup is:

# `d_0 = c`

So:

```text
Encoder Final State
↓
Decoder Initial State
```

This is the handoff point between:

> source understanding

and:

> target generation.

If dimensions differ, use:

`d_0 = W_c c + b_c`

---

# 📖 Part 9 — Step 4: Give the Decoder `<SOS>`

The decoder needs something to start from.

So the first decoder input is:

# `<SOS>`

That token is also embedded.

Suppose:

```text
<SOS>
→ target embedding vector
```

Then decoder uses:

```text
<SOS> embedding
+
d_0
↓
Decoder
↓
d_1
```

From `d_1`, the model predicts the first target token.

---

# 📐 Part 10 — Decoder Recurrence

A simplified decoder recurrence is:

`d_t = f_dec(e(y_(t-1)), d_(t-1), c)`

where:

* `e(y_(t-1))` = embedding of previous target token
* `d_(t-1)` = previous decoder hidden state
* `c` = source context

Some implementations inject `c` only through initialization.

Others may condition every decoder step on `c`.

The exact formulation can vary.

---

# ⭐ Part 11 — Decoder State Is Not Yet a Word

Suppose decoder computes:

`d_1`

That is a hidden vector.

We need to convert it into:

> one score per target-vocabulary token.

So:

`z_1 = W_o d_1 + b_o`

If target vocabulary size is:

`V`

then:

`z_1 ∈ R^V`

These values are:

> logits.

---

# 📐 Part 12 — Softmax Produces Token Probabilities

Apply:

`softmax(z_1)`

Then we obtain:

`P(y_1 | X)`

Example:

```text
J'aime  → 0.72
Je      → 0.10
Tu      → 0.05
l'IA    → 0.03
...
```

The model now has a probability distribution over the target vocabulary.

---

# 🧮 Part 13 — Select the First Token

With simple greedy decoding:

```text
argmax probability
```

selects:

`J'aime`

So:

```text
<SOS>
↓
Decoder
↓
J'aime
```

Now the decoder moves to the next step.

---

# ⭐ Part 14 — Generate the Second Token

At the next step, decoder uses:

```text
previous token = "J'aime"
previous state = d_1
source context = c
```

Then:

`d_2 = f_dec(e("J'aime"), d_1, c)`

Then:

`z_2 = W_o d_2 + b_o`

Then:

`softmax(z_2)`

Suppose highest probability becomes:

`l'IA`

So:

```text
J'aime
↓
Decoder
↓
l'IA
```

---

# ⭐ Part 15 — Generate `<EOS>`

Next:

```text
previous token = l'IA
```

Decoder computes:

`d_3`

and distribution:

`P(y_3 | y_1, y_2, X)`

Suppose:

```text
<EOS> → 0.91
```

Then decoder generates:

# `<EOS>`

Generation stops.

Final output:

```text
J'aime l'IA
```

---

# 🧠 Part 16 — Complete End-to-End Flow

```text
SOURCE
"I love AI"

↓ Tokenize

[I, love, AI]

↓ Source Embedding

x1  x2  x3

↓ Encoder

h1 → h2 → h3

↓
c = h3

↓
Decoder Initial State

d0

↓
<SOS>

↓
d1
↓
P(y1)
↓
"J'aime"

↓
d2
↓
P(y2)
↓
"l'IA"

↓
d3
↓
P(y3)
↓
<EOS>
```

That is a basic recurrent Seq2Seq model.

---

# 🧠 Part 17 — Source Vocabulary vs Target Vocabulary

In translation, source and target languages differ.

Therefore we often have:

```text
Source Vocabulary
≠
Target Vocabulary
```

Example:

```text
English vocabulary
→ Encoder embedding table

French vocabulary
→ Decoder embedding table
```

So there may be:

> two different embedding matrices.

---

# 📐 Part 18 — Source Embedding Matrix

If source vocabulary size is:

`V_src`

and source embedding dimension is:

`D_src`

then:

`E_src ∈ R^(V_src × D_src)`

Each source token ID selects one row.

So:

```text
token id
↓
E_src
↓
source embedding vector
```

---

# 📐 Part 19 — Target Embedding Matrix

Similarly:

`E_tgt ∈ R^(V_tgt × D_tgt)`

Target token:

```text
previous generated token
↓
target embedding lookup
↓
decoder input
```

Source and target embeddings:

> do not need to be the same.

---

# 🧠 Part 20 — Why Does the Decoder Need Target Embeddings?

Because the previous token ID itself is just:

> an integer identifier.

For example:

```text
"J'aime" → token ID 1532
```

The decoder needs a learned vector representation:

```text
ID 1532
↓
Target Embedding
↓
e("J'aime")
```

That embedding becomes part of the next decoder step.

---

# ⭐ Part 21 — Training: We Already Know the Target

Now the most important training distinction.

During training, suppose target is:

```text
<SOS> J'aime l'IA <EOS>
```

We already know the correct sequence.

So training can feed:

```text
<SOS>
```

to predict:

```text
J'aime
```

Then feed true:

```text
J'aime
```

to predict:

```text
l'IA
```

Then feed true:

```text
l'IA
```

to predict:

```text
<EOS>
```

This is:

> teacher forcing.

We will study it separately later.

---

# 📖 Part 22 — Training Input and Target Shift

This is extremely important.

Suppose target sequence is:

```text
J'aime l'IA <EOS>
```

Decoder input becomes:

```text
<SOS> J'aime l'IA
```

Expected outputs become:

```text
J'aime l'IA <EOS>
```

Notice:

> the target is shifted by one position.

So:

```text
Decoder Input:
<SOS>   J'aime   l'IA

Expected:
J'aime  l'IA     <EOS>
```

This is a core Seq2Seq training pattern.

---

# 🧠 Part 23 — Why Shift the Target?

Because at every step, the decoder learns:

> given previous target tokens, predict the next target token.

So:

```text
Given <SOS>
→ predict first token

Given first token
→ predict second token

Given second token
→ predict <EOS>
```

This directly matches the autoregressive probability factorization.

---

# 📐 Part 24 — Conditional Probability

Let:

`X = (x_1, ..., x_T)`

and:

`Y = (y_1, ..., y_T')`

Seq2Seq models:

# `P(Y | X)`

Using the chain rule:

# `P(Y|X) = Π_t P(y_t | y_1,...,y_(t-1), X)`

This equation is central.

It means:

> target sequence probability is built one token at a time.

---

# 🧮 Part 25 — Small Probability Example

Suppose target is:

```text
A B <EOS>
```

Model probabilities:

```text
P(A | X) = 0.8
```

```text
P(B | A, X) = 0.7
```

```text
P(<EOS> | A,B,X) = 0.9
```

Then:

`P(Y|X) = 0.8 × 0.7 × 0.9`

`= 0.504`

So the probability of the entire output sequence is:

# `0.504`

---

# ⭐ Part 26 — Training Objective

During training we want the correct target sequence to receive high probability.

Equivalent goal:

> maximize `P(Y|X)`

or minimize:

# **negative log-likelihood / cross-entropy loss**

For each target token:

`L_t = -log P(y_t^correct)`

Total sequence loss:

`L = Σ_t L_t`

Possibly averaged or masked depending on batching.

---

# 🧠 Part 27 — Why Use Log Probability?

Multiplying many probabilities:

```text
0.8 × 0.7 × 0.9 × ...
```

can become numerically very small.

Using logs:

`log(ab) = log(a) + log(b)`

So:

`log P(Y|X) = Σ_t log P(y_t | ...)`

This makes optimization more convenient.

---

# 📐 Part 28 — Cross-Entropy at Each Decoder Step

Suppose correct token is:

`J'aime`

and model predicts:

```text
J'aime → 0.8
Je     → 0.1
Tu     → 0.1
```

Loss:

`-log(0.8)`

If model gives correct token:

`0.01`

then:

`-log(0.01)`

is much larger.

So training pushes the model toward:

> higher probability for the correct next token.

---

# 🔗 Part 29 — Does the Encoder Receive Gradients?

Yes.

Even though loss is calculated at decoder outputs:

```text
Decoder Loss
↓
Decoder Parameters
↓
Decoder Initial State / Context
↓
Encoder
```

gradients flow backward through:

> the encoder-to-decoder connection.

So the encoder learns representations useful for target generation.

---

# 🧠 Part 30 — End-to-End Learning

This is crucial.

We do not separately tell the encoder:

> “store subject here, verb there.”

Instead:

```text
Translation Loss
↓
Backpropagation
↓
Decoder learns generation
+
Encoder learns useful source representation
```

The entire system is trained:

> end-to-end.

---

# 🔄 Part 31 — BPTT Exists on Both Sides

Because encoder and decoder are recurrent:

```text
Encoder
→ BPTT through source sequence

Decoder
→ BPTT through target sequence
```

Gradients also flow across:

```text
Encoder Context
→ Decoder
```

So training graph covers:

> both recurrent chains.

---

# ⭐ Part 32 — Training Flow

For source:

```text
I love AI
```

Target:

```text
J'aime l'IA
```

Training:

```text
SOURCE
↓
Encoder
↓
Context

TARGET INPUT
<SOS> J'aime l'IA
↓
Decoder
↓
Predictions
J'aime l'IA <EOS>

↓
Cross-Entropy
↓
Backpropagation
```

---

# ⚠️ Part 33 — Inference Is Different

During inference:

> correct target sequence is unknown.

We only have:

```text
Source Sequence
```

So:

```text
Source
↓
Encoder
↓
Context
↓
<SOS>
↓
Predict y1
↓
feed y1 back
↓
Predict y2
↓
...
↓
<EOS>
```

This means the decoder must:

> rely on its own predictions.

---

# 🧠 Part 34 — Training vs Inference

## Training

Often:

```text
True previous target
→ next prediction
```

## Inference

```text
Predicted previous target
→ next prediction
```

So:

```text
Training Context
≠
Inference Context
```

This difference will become important in:

# **Teacher Forcing**

---

# ⚠️ Part 35 — Error Propagation During Inference

Suppose correct output should be:

```text
I am happy
```

But decoder predicts:

```text
I is
```

Now:

`is`

becomes part of the decoder's future history.

So later predictions may also degrade.

Conceptually:

```text
Wrong Early Token
↓
Wrong Next Input
↓
Changed Hidden State
↓
Potential Later Errors
```

---

# ⭐ Part 36 — Greedy Decoding

The simplest inference strategy is:

> choose the highest-probability token at every step.

Example:

```text
A → 0.60
B → 0.30
C → 0.10
```

Choose:

`A`

Then continue.

This is:

# **Greedy Decoding**

---

# ⚠️ Part 37 — Why Greedy May Be Suboptimal

Suppose at first step:

```text
A → 0.55
B → 0.45
```

Greedy chooses:

`A`

But perhaps:

```text
A → later sequence probability = very low
```

while:

```text
B → later sequence probability = much higher
```

So choosing the best local token does not always produce:

> the best full sequence.

This eventually motivates:

# **Beam Search**

which we will study later.

---

# 🧠 Part 38 — Seq2Seq with LSTM

A classic setup:

```text
LSTM Encoder
↓
(h_T, c_T)
↓
LSTM Decoder
```

Decoder may initialize:

`h_0^dec = h_T^enc`

`c_0^dec = c_T^enc`

Then target generation proceeds recurrently.

---

# 🧠 Part 39 — Seq2Seq with GRU

Similarly:

```text
GRU Encoder
↓
h_T
↓
GRU Decoder
```

Simple state handoff:

`h_0^dec = h_T^enc`

This gives a simpler state interface.

---

# 🔗 Part 40 — Seq2Seq with Bidirectional Encoder

A common architecture can be:

```text
BiLSTM Encoder
↓
Source Representation
↓
Unidirectional LSTM Decoder
```

Why?

Encoder source:

> fully available.

Decoder future target:

> not available.

So:

```text
Encoder
→ can use both source directions

Decoder
→ must remain causal
```

---

# ⚠️ Part 41 — Bidirectional Encoder State Combination

Suppose encoder uses:

```text
forward final state →h_T
backward final state ←h_1
```

We may combine them:

```text
c = [→h_T ; ←h_1]
```

If each is `H` dimensional:

`c ∈ R^(2H)`

The decoder may then require a projection:

`d_0 = W_c c + b_c`

---

# 🧠 Part 42 — Batch Training and Padding

Real training uses multiple sequences in a batch.

But sequences have different lengths.

Example:

```text
Sentence A → 5 tokens
Sentence B → 8 tokens
Sentence C → 3 tokens
```

We often pad them to a common length.

Example:

```text
A: x x x x x PAD PAD PAD
B: x x x x x x   x   x
C: x x x PAD PAD PAD PAD PAD
```

---

# ⭐ Part 43 — Why Mask Padding?

Padding tokens are artificial.

We do not want the model to treat them as real sequence information or count them as real target loss.

So we use:

> masks.

For target loss:

```text
Real token
→ include loss

PAD token
→ ignore loss
```

This is important in practical Seq2Seq training.

---

# 📐 Part 44 — Target Loss Mask

Suppose:

```text
Target:
A B <EOS> PAD PAD
```

Loss might be:

```text
A      → calculate
B      → calculate
<EOS>  → calculate
PAD    → ignore
PAD    → ignore
```

Otherwise the model spends effort learning:

> meaningless padding positions.

---

# 🧠 Part 45 — Source and Target Have Separate Lengths

For each example we may track:

```text
source_length
target_length
```

because:

`T_src ≠ T_tgt`

This affects:

* padding
* masking
* recurrent processing
* loss computation
* batching

---

# ⭐ Part 46 — A Compact Computational Graph

```text
SOURCE IDS
↓
Source Embedding
↓
Encoder RNN/LSTM/GRU
↓
Context

TARGET IDS shifted right
↓
Target Embedding
↓
Decoder RNN/LSTM/GRU
↓
Decoder States
↓
Output Projection
↓
Logits
↓
Softmax / Cross-Entropy
↓
Training Loss
```

This is the essential Seq2Seq training pipeline.

---

# 🧠 Part 47 — What Exactly Is Learned?

The model learns:

### Source Embeddings

How source tokens should be represented.

### Encoder Parameters

How source sequence information should be combined.

### Target Embeddings

How target-history tokens should be represented.

### Decoder Parameters

How to generate conditioned on source and history.

### Output Projection

How decoder hidden states map to target-token scores.

Everything can be trained:

> jointly end-to-end.

---

# 🔗 Part 48 — Seq2Seq Is Conditional Generation

Normal language model:

```text
P(Y)
```

Seq2Seq:

# `P(Y | X)`

because target generation is conditioned on:

> source sequence `X`.

Example:

```text
Translation:
P(French Sentence | English Sentence)
```

This is a fundamental distinction.

---

# 🧠 Part 49 — Why the Context Matters at Every Output Step

Suppose source is:

```text
I love AI
```

When generating:

`J'aime`

the decoder needs source meaning.

When generating:

`l'IA`

it still needs source meaning.

So target generation must remain conditioned on:

> the encoded source information.

In the original architecture this conditioning comes through:

* decoder initialization
* and/or repeatedly injecting context

depending on implementation.

---

# ⚠️ Part 50 — Context Initialization vs Context at Every Step

Two conceptual variants:

## Variant A

Context initializes decoder:

```text
c
↓
d0
```

Then decoder evolves from there.

## Variant B

Context is also explicitly fed at every step:

`d_t = f(y_(t-1), d_(t-1), c)`

Both are possible formulations.

Do not assume:

> every implementation uses the context identically.

---

# ⭐ Part 51 — The Core Limitation Reappears

Everything we have built works well conceptually.

But look carefully:

```text
x1
x2
x3
...
xT
↓
Encoder
↓
ONE Context c
↓
Decoder
↓
y1 y2 y3 ...
```

The entire source still has to pass through:

> one fixed-dimensional context representation.

This works better for:

> shorter sequences

than:

> very long or information-rich sequences.

---

# 🧠 Part 52 — Why Long Sequences Are Hard

Suppose source has:

```text
5 tokens
```

and context size:

`256`

Now source has:

```text
100 tokens
```

context size remains:

`256`.

Now:

```text
500 tokens
```

still:

`256`.

The decoder must recover everything it needs from the same compressed channel.

That is increasingly difficult.

---

# 📖 Part 53 — Example: Long Translation

Suppose source:

```text
The scientist who moved to Germany after studying in India
published an important paper about renewable energy last year.
```

When decoder reaches the end of translation, it may need:

> "last year"

But that phrase appeared long before in the encoder computation.

The decoder has no direct way to say:

> “Show me the encoder state corresponding to 'last year'.”

It only has:

`c`

That is the central weakness.

---

# ⭐ Part 54 — The Next Problem

We now have enough understanding to isolate the next question:

# **Can one fixed vector really carry everything the decoder needs from a long input sequence?**

This deserves its own lecture.

Our next topic:

# `10_Context_Vector_Bottleneck`

There we will study:

* why information compression becomes difficult
* why early source information can become hard to recover
* why simply making hidden state larger is not enough
* why decoder needs more flexible source access
* how this pressure naturally leads to attention

---

# 🔗 Part 55 — Evolution So Far

```text
Sequence Data
↓
RNN
↓
Long-Term Dependency Problem
↓
LSTM / GRU
↓
Need Both-Side Source Context
↓
Bidirectional RNN
↓
Need Sequence → Sequence Mapping
↓
Encoder–Decoder
↓
Complete Conditional Generation
↓
Seq2Seq
↓
Fixed Context Bottleneck
```

And soon:

```text
Fixed Context Bottleneck
↓
Need Dynamic Source Access
↓
Attention
```

---

# 🎤 Part 56 — 30-Second Interview Answer

> **A Seq2Seq model maps one variable-length sequence to another using an encoder and decoder. The encoder converts source-token embeddings into a source representation, traditionally the final recurrent hidden state. The decoder is initialized or conditioned with that representation and generates target tokens autoregressively from a start token until an end token is produced. During training, target tokens are typically shifted so the decoder learns next-token prediction, while during inference it feeds back its own predictions. The model is trained end-to-end using token-level cross-entropy, but the original design suffers from a fixed-context-vector bottleneck for long sequences.**

---

# 🎤 Part 57 — What Makes Seq2Seq Different From a Normal Language Model?

Strong answer:

> **A standard language model learns `P(Y)`, while a Seq2Seq model learns the conditional distribution `P(Y|X)`. The target sequence is generated not only from its previous tokens but also from an encoded source sequence.**

---

# 🎤 Part 58 — Why Shift the Target During Training?

Strong answer:

> **Because the decoder is trained to predict the next target token given previous target tokens. Therefore `<SOS>` plus all target tokens except the last are used as decoder inputs, while the actual target tokens plus `<EOS>` are used as expected outputs.**

---

# 🎤 Part 59 — Why Does the Decoder Need `<EOS>`?

Strong answer:

> **Because output sequences may have different lengths. `<EOS>` gives the decoder a learned termination condition instead of requiring a fixed output length.**

---

# 🚫 Part 60 — High-Yield Traps

### ❌ Seq2Seq means input and output lengths are equal

No.

---

### ❌ Encoder receives target tokens

No.

Encoder receives the source sequence.

---

### ❌ Decoder uses source embeddings directly in the basic model

Not usually.

It receives source information through the encoder representation.

---

### ❌ Token IDs are directly fed as semantic vectors

No.

They are typically mapped through embeddings.

---

### ❌ Decoder hidden state is a vocabulary probability

No.

Need projection + softmax.

---

### ❌ Training and inference are identical

No.

During inference, true future target tokens are unavailable.

---

### ❌ Greedy decoding always gives the highest-probability sequence

No.

It gives locally best tokens.

---

### ❌ Padding should contribute to target loss

No.

Mask it.

---

### ❌ Seq2Seq solves the context bottleneck

No.

Classic Seq2Seq actually exposes it.

---

# 📐 Formula Flash Card

### Source

`X = (x_1, ..., x_T)`

### Target

`Y = (y_1, ..., y_T')`

### Encoder

`h_t = f_enc(x_t, h_(t-1))`

### Original Context

# `c = h_T`

### Decoder

`d_t = f_dec(e(y_(t-1)), d_(t-1), c)`

### Output Logits

`z_t = W_o d_t + b_o`

### Token Distribution

`P(y_t | y_<t, X) = softmax(z_t)`

### Sequence Probability

# `P(Y|X) = Π_t P(y_t | y_<t, X)`

### Token Loss

`L_t = -log P(y_t^correct)`

### Sequence Loss

# `L = Σ_t L_t`

ignoring padded target positions.

---

# 🧠 Part 61 — Ten Things You Must Know Cold

1. **Seq2Seq = sequence → sequence**
2. **Usually built with Encoder–Decoder**
3. **Source and target vocabularies may differ**
4. **Both sides typically use embeddings**
5. **Encoder final state traditionally becomes context**
6. **`<SOS>` starts target generation**
7. **`<EOS>` ends target generation**
8. **Target is shifted during training**
9. **Decoder is autoregressive**
10. **Fixed context becomes the main bottleneck**

---

# 🧠 Final Mental Model

Think of Seq2Seq as a complete conditional-generation pipeline:

```text
SOURCE TOKENS
      │
      ▼
Source Embeddings
      │
      ▼
    ENCODER
      │
      ▼
    CONTEXT
      │
      ▼
    DECODER
      │
      ▼
Target Distribution
      │
      ▼
Target Token
      │
      └─────────┐
                │
                ▼
         Feed Back Next Step
```

During training:

```text
Correct Previous Target
→ predict next target
```

During inference:

```text
Model's Previous Prediction
→ predict next target
```

And the probability view is:

# `Source X → model P(Target Y | X)`

---

# 🚀 Where We Go Next

We now understand the complete classic Seq2Seq pipeline.

But one uncomfortable question remains:

```text
Entire Source Sequence
↓
ONE Fixed Context Vector
↓
Entire Target Generation
```

How much information can one fixed vector realistically preserve?

Why does performance deteriorate as source sequences grow?

Why can't the decoder simply retrieve the exact source information it needs?

That is our next lecture:

# `10_Context_Vector_Bottleneck`

And that problem will become the direct bridge toward:

# **Attention**

---

# ⭐ Golden Rule

> **A Seq2Seq model learns `P(Y|X)` by encoding the source into a representation and autoregressively decoding the target one token at a time; its power comes from end-to-end conditional generation, while its classic weakness comes from forcing all source information through one fixed context vector.**
s