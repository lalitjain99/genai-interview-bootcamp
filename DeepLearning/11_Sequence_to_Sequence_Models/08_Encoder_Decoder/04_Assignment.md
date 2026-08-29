# 💡 Assignment — Encoder–Decoder Architecture

> **Goal:** Apply Encoder–Decoder concepts through sequence tracing, dimensional reasoning, state transfer, autoregressive decoding, training-vs-inference analysis, and bottleneck diagnosis.

---

# 🧩 Task 1 — Identify the Roles

Consider:

```text
English Sentence
↓
?
↓
Context Representation
↓
?
↓
French Sentence
```

### Questions

1. Which component reads the English sentence?
2. Which component generates the French sentence?
3. What connects the two?

### ✅ Answer

```text
English Sentence
↓
Encoder
↓
Context Representation
↓
Decoder
↓
French Sentence
```

So:

```text
Encoder
→ source understanding

Context
→ source representation

Decoder
→ target generation
```

---

# 🧠 Task 2 — Why Not Just Use Sequence Classification?

Suppose the task is:

```text
Input:
"I love AI"

Output:
"J'aime l'IA"
```

Why is a normal many-to-one classifier insufficient?

### ✅ Answer

A classifier typically produces:

```text
Sequence
→ Single Label
```

But translation requires:

```text
Sequence
→ Another Sequence
```

The output:

* contains multiple tokens
* may have different length
* must preserve ordering
* must be generated step-by-step

Therefore we need a sequence-generation mechanism.

---

# 🔍 Task 3 — Trace the Encoder

Suppose the source sequence is:

```text
x1 = "I"
x2 = "love"
x3 = "AI"
```

The encoder follows:

`h_t = f_enc(x_t, h_(t-1))`

### Fill the flow

```text
x1 + h0
↓
_____

x2 + _____
↓
_____

x3 + _____
↓
_____
```

### ✅ Answer

```text
x1 + h0
↓
h1

x2 + h1
↓
h2

x3 + h2
↓
h3
```

In the original recurrent Encoder–Decoder:

# `c = h3`

---

# 🧠 Task 4 — What Does the Context Vector Actually Contain?

A developer says:

> “The context vector stores all source tokens exactly.”

Is that correct?

### ✅ Answer

No.

The context vector is:

> a learned compressed numerical representation.

It does not literally store:

```text
token 1
token 2
token 3
...
```

as an exact copy.

A better description is:

> it tries to preserve information useful for the target task.

---

# 📐 Task 5 — Decoder Initialization

Suppose:

```text
Encoder final hidden state:
h_T ∈ R^256

Decoder hidden state:
d_0 ∈ R^256
```

Can we directly initialize:

`d_0 = h_T`?

### ✅ Answer

Yes, dimensionally this is possible because:

```text
h_T ∈ R^256
d_0 ∈ R^256
```

A simple Encoder–Decoder may directly pass the encoder final state to the decoder.

---

# 📐 Task 6 — Hidden Sizes Do Not Match

Now suppose:

```text
Encoder hidden size = 512
Decoder hidden size = 256
```

Can we directly use:

`d_0 = h_T`?

### ✅ Answer

No.

The dimensions differ:

```text
h_T ∈ R^512
d_0 ∈ R^256
```

Use a learned projection:

# `d_0 = W_c h_T + b_c`

where:

```text
W_c ∈ R^(256×512)

b_c ∈ R^256
```

Then:

```text
R^512
↓
projection
↓
R^256
```

---

# 🧩 Task 7 — LSTM State Transfer

An LSTM encoder ends with:

```text
h_T^enc
c_T^enc
```

What might be passed to an LSTM decoder?

### ✅ Answer

A common setup is:

```text
h_0^dec = h_T^enc
c_0^dec = c_T^enc
```

or learned projections of them.

Remember:

```text
LSTM
→ hidden state h
→ cell state c
```

So both may participate in encoder-to-decoder state transfer.

---

# 🧩 Task 8 — GRU State Transfer

What changes if both encoder and decoder use GRU?

### ✅ Answer

GRU has only one recurrent state:

`h_t`

So a simple transfer can be:

# `h_0^dec = h_T^enc`

or a projected version.

There is:

> no separate cell state `c_t`.

---

# ⭐ Task 9 — Start Token

Why does the decoder need:

`<SOS>`

or:

`<BOS>`?

### ✅ Answer

Before generating the first target token, there is no previous target word.

So `<SOS>` provides:

> a known initial decoder input.

Example:

```text
<SOS>
↓
Decoder
↓
"Je"
```

Without a start signal, we need some other mechanism to initiate decoding.

---

# ⭐ Task 10 — End Token

Suppose the decoder generates:

```text
Je
suis
heureux
<EOS>
```

What should happen when `<EOS>` is generated?

### ✅ Answer

Decoding should stop.

`<EOS>` tells the model:

> the target sequence is complete.

This allows the decoder to generate variable-length outputs.

---

# 🧠 Task 11 — Variable-Length Reasoning

Suppose:

```text
Input length = 8 tokens

Output length = 5 tokens
```

Is this valid in Encoder–Decoder?

### ✅ Answer

Yes.

There is no requirement that:

# `T_input = T_output`

Encoder and decoder operate over separate sequence lengths.

This is important for:

* translation
* summarization
* speech recognition

---

# 🔄 Task 12 — Autoregressive Decoding

Suppose target is:

```text
<SOS> I am ready <EOS>
```

Fill in the inference flow.

### ✅ Answer

```text
<SOS>
↓
predict "I"

"I"
↓
predict "am"

"am"
↓
predict "ready"

"ready"
↓
predict <EOS>
```

Each prediction becomes the next decoder input.

This is:

# **autoregressive decoding**

---

# 📐 Task 13 — Sequence Probability

Complete:

`P(Y|X) = __________________________`

### ✅ Answer

# `P(Y|X) = Π_t P(y_t | y_<t, X)`

Meaning:

> the probability of the full target sequence is decomposed into next-token probabilities conditioned on the source and previous target tokens.

---

# 🧮 Task 14 — From Decoder State to Vocabulary Probabilities

Suppose:

```text
d_t ∈ R^128

Vocabulary size V = 10,000
```

The output projection is:

`z_t = W_o d_t + b_o`

What are the shapes?

### ✅ Answer

```text
d_t ∈ R^128
```

To produce one logit per vocabulary token:

```text
W_o ∈ R^(10000×128)

b_o ∈ R^10000

z_t ∈ R^10000
```

Then:

```text
softmax(z_t)
```

produces:

> 10,000 token probabilities.

---

# 🧮 Task 15 — Greedy Decoding

Suppose the decoder outputs:

```text
Token A → 0.10
Token B → 0.65
Token C → 0.20
<EOS>   → 0.05
```

What does greedy decoding select?

### ✅ Answer

It selects:

# `Token B`

because it has the highest probability:

`0.65`

Greedy decoding chooses:

`argmax`

at each generation step.

---

# 🔥 Task 16 — Detect a Decoder Error Cascade

Suppose the correct sequence is:

```text
I am going home
```

but decoder predicts:

```text
I is ...
```

Why can this affect later predictions?

### ✅ Answer

During inference, the model's previous prediction becomes part of the next-step input/history.

So:

```text
wrong token at t
↓
different decoder input/state at t+1
↓
different future predictions
```

An early error can therefore:

> propagate through the generated sequence.

This motivates later discussions around:

* teacher forcing
* exposure bias
* decoding strategies

---

# 🧠 Task 17 — Training vs Inference

For target:

```text
<SOS> Je suis prêt <EOS>
```

During training, what token might be supplied when predicting:

`suis`?

### ✅ Answer

The correct previous token:

`Je`

may be supplied.

During inference, however, the model uses:

> whatever it previously predicted.

So:

```text
Training
→ true previous token can be used

Inference
→ predicted previous token is used
```

This difference leads to:

# **Teacher Forcing**

---

# ⚠️ Task 18 — Can the Decoder Be Bidirectional?

Suppose an engineer proposes a BiLSTM decoder for real-time translation generation.

What is the problem?

### ✅ Answer

To generate:

`y_t`

a backward decoder would require:

```text
y_(t+1), y_(t+2), ...
```

But those tokens do not exist yet during inference.

That would violate:

> causal autoregressive generation.

So a standard autoregressive decoder should be:

> unidirectional / causal.

---

# ✅ Task 19 — Can the Encoder Be Bidirectional?

Same translation system.

Can the source encoder use BiLSTM?

### ✅ Answer

Usually yes.

The entire source sentence may already be available before decoding begins.

So:

```text
Source sequence
→ fully known
→ forward + backward encoding valid
```

Thus:

```text
BiLSTM Encoder
+
Causal Decoder
```

is a valid architecture.

---

# 🔍 Task 20 — Encoder vs Decoder State

Classify each statement.

### A

> Represents information about the source sequence.

### B

> Tracks target-generation history.

### C

> May be used to initialize generation.

### D

> Evolves while output tokens are produced.

### ✅ Answer

| Statement | State                                          |
| --------- | ---------------------------------------------- |
| A         | Encoder state                                  |
| B         | Decoder state                                  |
| C         | Encoder state/context helps initialize decoder |
| D         | Decoder state                                  |

Mental model:

```text
Encoder
→ What did I read?

Decoder
→ What have I generated and what comes next?
```

---

# 🔗 Task 21 — Many-to-Many vs Encoder–Decoder

Which architecture better fits each task?

### A. POS tagging

```text
The   cat   runs
DET   NOUN  VERB
```

### B. English → French translation

### ✅ Answer

### POS tagging

Simple aligned many-to-many architecture may work because:

```text
x1 ↔ y1
x2 ↔ y2
x3 ↔ y3
```

### Translation

Encoder–Decoder is more natural because:

* lengths may differ
* target ordering may differ
* strict one-to-one alignment is not required

---

# 🚨 Task 22 — Diagnose the Bottleneck

Suppose an Encoder–Decoder translates:

```text
5-token sentences
```

very well, but performance falls dramatically for:

```text
80-token sentences
```

The decoder only receives:

`h_T`

from the encoder.

What architectural problem would you suspect?

### ✅ Answer

# **Fixed context vector bottleneck**

The entire 80-token sequence must be compressed into:

> one fixed-dimensional vector.

Important source details can become difficult to preserve.

---

# 🧠 Task 23 — Would Increasing Hidden Size Fully Fix It?

A teammate says:

> “Let's increase the context vector from 256 to 4096 dimensions. Then the bottleneck is solved.”

Is this fully correct?

### ✅ Answer

No.

A larger vector increases capacity and may improve performance.

But structurally, the architecture still does:

```text
All Encoder Information
↓
One Fixed Vector
↓
Decoder
```

The decoder still cannot directly access specific source positions.

So:

> larger context may help, but it does not remove the architectural bottleneck.

---

# ⭐ Task 24 — Derive Attention From the Problem

If one fixed vector is the problem, what architectural idea would improve it?

### ✅ Answer

Instead of giving the decoder only:

`h_T`

retain:

```text
h1, h2, h3, ..., hT
```

Then at each decoder step allow the model to ask:

> Which encoder states are most relevant right now?

Conceptually:

```text
Encoder States
h1 h2 h3 ... hT
│  │  │      │
└──┴──┴──────┴──→ Decoder
```

This leads toward:

# **Attention**

---

# 🧩 Task 25 — Architecture Debugging

A developer builds:

```text
Input Sentence
↓
Encoder GRU
↓
h_T
↓
Dense Layer
↓
Entire translated sentence emitted in one operation
```

They call this a recurrent Encoder–Decoder.

What is missing?

### ✅ Answer

The architecture lacks:

> a recurrent/autoregressive decoder.

For a recurrent Seq2Seq model, target generation should conceptually be:

```text
Decoder State
↓
y1
↓
Decoder State
↓
y2
↓
...
```

A single Dense layer does not model the target sequence recurrently in the same way.

---

# ✅ True / False

| #  | Statement                                                                | Answer  |
| -- | ------------------------------------------------------------------------ | ------- |
| 1  | Encoder–Decoder can map sequences of different lengths.                  | ✅ True  |
| 2  | The encoder always generates target tokens.                              | ❌ False |
| 3  | The original context vector is often based on the final encoder state.   | ✅ True  |
| 4  | The context vector is an exact copy of the input sequence.               | ❌ False |
| 5  | `<SOS>` can initialize decoding.                                         | ✅ True  |
| 6  | `<EOS>` signals generation termination.                                  | ✅ True  |
| 7  | An autoregressive decoder can use future target tokens during inference. | ❌ False |
| 8  | A source encoder can often be bidirectional.                             | ✅ True  |
| 9  | Encoder and decoder hidden dimensions must always match.                 | ❌ False |
| 10 | A projection can connect different hidden dimensions.                    | ✅ True  |
| 11 | GRU encoder transfers a separate cell state to the decoder.              | ❌ False |
| 12 | LSTM state transfer may involve both `h` and `c`.                        | ✅ True  |
| 13 | Decoder hidden state is directly a probability distribution over words.  | ❌ False |
| 14 | Softmax can convert decoder logits into token probabilities.             | ✅ True  |
| 15 | Fixed context compression can hurt long-sequence performance.            | ✅ True  |

---

# ⭐ Staff Engineer Challenge — Long-Document Summarization

## Scenario

Your team builds a recurrent summarization service:

```text
Long Document
↓
BiLSTM Encoder
↓
Final Encoder State
↓
LSTM Decoder
↓
Summary
```

Results:

```text
Documents under 100 tokens
→ good summaries

Documents around 800 tokens
→ important facts frequently disappear
```

---

## Question 1 — What is the first architectural issue you would investigate?

### ✅ Answer

The:

# **fixed context vector bottleneck**

The complete 800-token document must be represented by:

> one fixed-size final encoder state.

The decoder has no direct way to revisit the individual encoder states representing important earlier passages.

---

## Question 2 — Why does using BiLSTM not solve this problem?

### ✅ Answer

BiLSTM improves:

> contextual encoding from both source directions.

But if the final architecture still reduces everything to:

```text
one final context representation
```

then the compression bottleneck remains.

So:

```text
Bidirectionality
→ richer source encoding

but

Fixed final vector
→ still bottlenecked
```

---

## Question 3 — Why might simply adding more LSTM layers not be enough?

### ✅ Answer

More layers increase modeling capacity.

But the decoder still receives the source through:

> the same restricted communication channel.

The structural problem is not just:

> encoder capacity.

It is:

> decoder access to source information.

---

## Question 4 — What architecture would you propose next?

### ✅ Answer

Preserve the full encoder-state sequence:

```text
h1, h2, ..., hT
```

and allow each decoder step to build a context based on:

> the encoder positions most relevant to that output step.

This is the direction toward:

# **attention-based Encoder–Decoder**

---

## Question 5 — Would you necessarily keep the encoder bidirectional?

### ✅ Answer

For offline summarization of a fully uploaded document:

> yes, bidirectionality can be valid.

The complete source document is available before generation begins.

The decoder still needs to remain causal because:

> the future summary tokens have not yet been generated.

---

## Question 6 — Suppose the decoder hidden size is 512 and vocabulary size is 50,000. What is the basic output projection shape?

### ✅ Answer

Decoder state:

`d_t ∈ R^512`

Need:

`50,000`

logits.

So:

```text
W_o ∈ R^(50000×512)

b_o ∈ R^50000
```

Then:

`z_t ∈ R^50000`

and:

```text
softmax(z_t)
```

produces the vocabulary distribution.

---

## Question 7 — What production trade-offs remain even after improving the architecture?

### ✅ Answer

You would still consider:

* sequence length
* decoder latency
* autoregressive generation cost
* memory usage
* beam-search cost
* model size
* batching
* maximum output length
* termination behavior

A better representation architecture does not remove:

> all production constraints of autoregressive decoding.

---

# 🎯 Final Applied Exercise

Complete the evolution:

```text
Need Sequence → Sequence
↓
Separate __________ from __________
↓
Encoder reads __________
↓
Creates __________
↓
Decoder starts with __________
↓
Generates output __________
↓
Stops at __________
↓
Long inputs cause __________
↓
Need decoder access to multiple __________
↓
Eventually leads to __________
```

### ✅ Answer

```text
Need Sequence → Sequence
↓
Separate UNDERSTANDING from GENERATION
↓
Encoder reads SOURCE SEQUENCE
↓
Creates CONTEXT / REPRESENTATION
↓
Decoder starts with <SOS> / <BOS>
↓
Generates output AUTOREGRESSIVELY
↓
Stops at <EOS>
↓
Long inputs cause FIXED-CONTEXT BOTTLENECK
↓
Need decoder access to multiple ENCODER STATES
↓
Eventually leads to ATTENTION
```

---

# 🧠 Final Assignment Mental Model

```text
SOURCE TOKENS
x1 x2 x3 ... xT
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
    <SOS>
      │
      ▼
     y1
      │
      ▼
     y2
      │
      ▼
     ...
      │
      ▼
    <EOS>
```

Ask yourself four questions:

```text
1. What is the encoder representing?
2. How does that information reach the decoder?
3. What does the decoder condition each next token on?
4. Is one fixed context vector enough for long inputs?
```

---

# ⭐ Golden Rule

> **Encoder–Decoder solves variable-length sequence transformation by separating source representation from target generation, but if all source information must pass through one fixed-dimensional context vector, that communication channel eventually becomes the architecture's central bottleneck.**
