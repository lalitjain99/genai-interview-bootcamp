# 📘 Lecture — Bidirectional RNN

## 🌟 Introduction

So far, every recurrent model we have studied processes information in one direction:

```text
x1 → x2 → x3 → x4 → ...
```

That means at time step `t`, the model knows:

> what came before

but not:

> what comes after.

This is completely fine for tasks where the future is unavailable.

For example:

```text
Real-time sensor stream
Stock price stream
Live speech input
Next-token prediction
```

But many sequence-understanding problems are different.

Suppose we read:

```text
He went to the bank to deposit money.
```

When we first encounter:

`bank`

there are multiple possible meanings:

```text
bank
→ financial institution

bank
→ side of a river
```

The earlier words:

```text
He went to the
```

do not completely resolve the meaning.

But later words:

```text
to deposit money
```

make it obvious.

So a natural question appears:

# **Why should a sequence model use only the past when the entire sequence is already available?**

If both earlier and later context are available, perhaps we should process:

```text
Past → Present
```

and also:

```text
Future → Present
```

That leads to:

# **Bidirectional RNN**

---

# 🎯 Learning Objectives

By the end of this lecture, you should be able to:

## 🧠 Conceptual Understanding

* Explain why a standard one-directional RNN can miss useful future context.
* Explain why bidirectional processing is useful.
* Understand that a Bidirectional RNN contains:

  * one forward recurrent network
  * one backward recurrent network
* Understand how both directions create a richer representation of each sequence position.
* Explain when Bidirectional RNNs can and cannot be used.

## 📐 Mathematical Understanding

* Write forward hidden-state equations.
* Write backward hidden-state equations.
* Understand how the two hidden states are combined.
* Understand output dimensionality after concatenation.
* Understand parameter-count implications.

## 🔗 Architectural Understanding

* Connect Bidirectional RNNs to:

  * vanilla RNN
  * LSTM
  * GRU
* Understand that bidirectionality is independent of the recurrent cell type.
* Understand why Bidirectional RNN is not suitable for causal autoregressive generation.

## 🧮 Applied Understanding

* Trace a small sentence in both directions.
* Understand how future context improves token representations.
* Reason about inference constraints.
* Compare online/streaming vs offline sequence processing.

---

# 📖 Part 1 — The Limitation of a Normal RNN

Consider a normal left-to-right RNN:

```text
x1 → x2 → x3 → x4 → x5
```

At time step `t=3`, hidden state:

`h_3`

depends on:

```text
x1
x2
x3
```

but not:

```text
x4
x5
```

Mathematically:

`h_t = f(x_t, h_(t-1))`

Therefore:

```text
h_3
← x1, x2, x3

but

h_3
✗ does not know x4, x5
```

This creates a limitation when the meaning of the current element depends on:

> future context.

---

# 🧠 Part 2 — A Language Example

Consider:

```text
The crane flew over the lake.
```

At the word:

`crane`

you may wonder:

```text
crane
→ bird?

crane
→ construction machine?
```

Later context:

```text
flew over the lake
```

makes the bird meaning much more likely.

A normal left-to-right RNN representing the word `crane` at that position has not yet seen:

```text
flew over the lake
```

So its representation is based only on:

> previous context.

---

# 📖 Part 3 — What If We Read the Sentence Backward Too?

Suppose we process the same sequence twice.

### Forward

```text
The → crane → flew → over → the → lake
```

### Backward

```text
lake → the → over → flew → crane → The
```

Now, for the word:

`crane`

we have two representations.

Forward representation:

> what the past tells us about `crane`

Backward representation:

> what the future tells us about `crane`

Combine them and we get:

> a context representation informed by both directions.

This is the core idea of:

# **Bidirectional RNN**

---

# ⭐ Part 4 — The Architecture

A Bidirectional RNN consists of two separate recurrent networks.

```text
Forward RNN
→ processes left to right

Backward RNN
→ processes right to left
```

For the sequence:

```text
x1   x2   x3   x4
```

Forward network:

```text
x1 → x2 → x3 → x4
↓    ↓    ↓    ↓
→h1  →h2  →h3  →h4
```

Backward network:

```text
x1 ← x2 ← x3 ← x4
↓    ↓    ↓    ↓
←h1  ←h2  ←h3  ←h4
```

At every position `t`, we now have:

```text
forward hidden state  →h_t

backward hidden state ←h_t
```

---

# 🧠 Part 5 — What Does Each Direction Know?

At position `t`:

### Forward hidden state

`→h_t`

contains information from:

```text
x1, x2, ..., x_t
```

### Backward hidden state

`←h_t`

contains information from:

```text
x_T, x_(T-1), ..., x_t
```

So together:

```text
→h_t + ←h_t
```

provide information from:

> both sides of the sequence.

---

# 📐 Part 6 — Forward RNN Equation

Forward direction behaves like an ordinary RNN:

`→h_t = f(W_x→ x_t + W_h→ →h_(t-1) + b_→)`

For a vanilla RNN:

`→h_t = tanh(W_x→ x_t + W_h→ →h_(t-1) + b_→)`

Conceptually:

```text
Previous Forward State
+
Current Input
↓
Forward Hidden State
```

---

# 📐 Part 7 — Backward RNN Equation

Backward direction moves from the end toward the beginning.

So:

`←h_t = f(W_x← x_t + W_h← ←h_(t+1) + b_←)`

Notice the important difference.

Forward:

`→h_t` depends on `→h_(t-1)`

Backward:

`←h_t` depends on `←h_(t+1)`

because the backward network is traversing:

```text
T → T-1 → T-2 → ...
```

---

# ⭐ Part 8 — The Key Representation

At position `t`, we combine:

`→h_t`

and:

`←h_t`

A very common method is:

# `h_t = [→h_t ; ←h_t]`

where:

`;`

means:

> concatenation.

So if:

```text
→h_t ∈ R^H
←h_t ∈ R^H
```

then:

# `h_t ∈ R^(2H)`

---

# 🧠 Part 9 — Why Concatenation?

Suppose:

```text
→h_t = [0.3, 0.7]

←h_t = [-0.4, 0.9]
```

Concatenation gives:

```text
h_t = [0.3, 0.7, -0.4, 0.9]
```

This preserves:

* forward information
* backward information

separately.

The downstream layer can then learn:

> how much each direction matters.

---

# ⚠️ Part 10 — Must We Concatenate?

No.

Possible combination methods include:

* concatenation
* addition
* averaging
* learned combination

But concatenation is extremely common because it preserves both representations without immediately mixing them.

---

# 📖 Part 11 — Worked Sentence Example

Consider:

```text
He sat on the bank of the river.
```

Suppose we are representing:

`bank`

### Forward RNN sees:

```text
He
He sat
He sat on
He sat on the
He sat on the bank
```

The context is still ambiguous.

### Backward RNN effectively sees:

```text
river
the river
of the river
bank of the river
```

Now the future context strongly suggests:

> river bank

So at `bank`:

```text
→h_bank
→ past context

←h_bank
→ future context
```

Combined:

```text
[→h_bank ; ←h_bank]
```

creates a much richer representation.

---

# ⭐ Part 12 — Bidirectional Does NOT Mean One RNN Changes Direction

This is a common misunderstanding.

A Bidirectional RNN is not:

```text
one RNN
→ goes forward
→ turns around
→ comes backward
```

Instead, there are:

> **two separate recurrent networks**

with separate learned parameters.

```text
Forward RNN
+
Backward RNN
```

Their outputs are later combined.

---

# 🧠 Part 13 — Are the Forward and Backward Weights Shared?

Typically:

> No.

Forward network has its own parameters:

```text
W_x→
W_h→
b_→
```

Backward network has separate parameters:

```text
W_x←
W_h←
b_←
```

Why?

Because:

> processing past-to-future and future-to-past may require different learned transformations.

---

# 🔗 Part 14 — Bidirectional RNN Is Not a New Cell Type

This is important.

Bidirectionality describes:

> how the sequence is processed.

It does not describe:

> the recurrent cell itself.

So we can have:

```text
Bidirectional Vanilla RNN
```

```text
Bidirectional LSTM
```

```text
Bidirectional GRU
```

In practice, you will often encounter:

# **BiLSTM**

and:

# **BiGRU**

---

# 🧠 Part 15 — Bidirectional LSTM

A BiLSTM contains:

```text
Forward LSTM
+
Backward LSTM
```

Each LSTM has its own:

* cell state
* hidden state
* gates
* parameters

At every position:

```text
→h_t
←h_t
```

are combined.

So:

```text
LSTM
→ improves long-term recurrence

Bidirectionality
→ provides past + future context
```

These solve:

> different problems.

---

# 🔗 Part 16 — Bidirectional GRU

Similarly:

```text
Forward GRU
+
Backward GRU
```

Each direction has its own:

* update gate
* reset gate
* hidden state
* parameters

Then:

`[→h_t ; ←h_t]`

forms the bidirectional representation.

---

# ⭐ Part 17 — What Problem Does Bidirectionality Solve?

It solves:

> incomplete contextual information caused by processing only one direction.

Normal RNN:

```text
Past → Current
```

Bidirectional RNN:

```text
Past → Current ← Future
```

So each position can use:

> left context + right context.

---

# ⚠️ Part 18 — What Problem Does It NOT Solve?

Bidirectional processing does not fundamentally solve:

* sequential computation
* long-sequence scalability
* all vanishing-gradient problems

If the underlying cell is vanilla RNN:

> it can still suffer from long-range gradient issues.

Using:

* BiLSTM
* BiGRU

helps with that aspect.

But the sequence is still recurrent.

---

# 📖 Part 19 — Where Is Bidirectional Processing Useful?

It is especially useful when:

> the entire sequence is already available before inference.

Examples include:

### Named Entity Recognition

```text
Apple released a new device.
```

Context around `Apple` helps determine whether it means:

* company
* fruit

---

### Part-of-Speech Tagging

Future words can help determine the grammatical role of the current token.

---

### Sentiment Analysis

Both earlier and later phrases can contribute to interpretation.

---

### Speech Recognition

When processing recorded audio, later acoustic information can improve interpretation of earlier frames.

---

### Sequence Labeling

Tasks where each token needs a context-rich representation.

---

# 🚫 Part 20 — Where Is Bidirectional Processing Problematic?

Suppose you are predicting:

> the next token.

Input:

```text
The cat sat on the
```

You are trying to predict:

`mat`

You cannot use:

`mat`

or later words to create the representation used to predict `mat`.

That would be:

> future-information leakage.

So for causal generation:

```text
Past
↓
Predict Future
```

we cannot use future tokens.

---

# ⭐ Part 21 — Bidirectional RNN and Causality

A causal model obeys:

> prediction at time `t` can depend only on information available up to `t`.

Bidirectional RNN violates this if used for real-time causal prediction because:

`←h_t`

depends on future sequence elements.

Therefore:

```text
Offline understanding
→ Bidirectional can be useful

Online causal prediction
→ Bidirectional often impossible
```

---

# 🧠 Part 22 — Streaming Example

Suppose sensor data arrives:

```text
t1
t2
t3
```

At `t3`, future:

```text
t4
t5
...
```

has not arrived yet.

So a backward RNN cannot compute:

`←h_3`

using future observations.

Therefore a true Bidirectional RNN generally requires:

> access to the complete sequence, or at least future lookahead.

---

# ⚠️ Part 23 — Real-Time Does Not Automatically Mean Impossible

Sometimes systems use:

> limited lookahead.

For example, speech recognition might wait:

`500 ms`

before making a prediction.

Then it can use a small amount of future context.

But full bidirectional recurrence requires:

> full future sequence availability.

This introduces:

* latency
* buffering
* operational trade-offs.

---

# 🧮 Part 24 — Output Dimensions

Suppose each direction has hidden size:

`H = 128`

Then:

```text
→h_t ∈ R^128

←h_t ∈ R^128
```

If concatenated:

`h_t = [→h_t ; ←h_t]`

then:

# `h_t ∈ R^256`

This is important when connecting a BiRNN to:

* Dense layers
* classifiers
* CRFs
* attention layers

---

# 🧮 Part 25 — Parameter Count

Suppose one RNN direction has:

`P`

parameters.

A bidirectional version has approximately:

# `2P`

because we have:

```text
Forward parameters
+
Backward parameters
```

If using a vanilla RNN with input dimension `D` and hidden dimension `H`:

One direction:

`H(D + H + 1)`

Bidirectional:

# `2H(D + H + 1)`

excluding downstream output layers.

---

# 🔗 Part 26 — BiLSTM Parameter Count

A simplified LSTM direction has:

`4H(D+H+1)`

Bidirectional LSTM therefore has approximately:

# `8H(D+H+1)`

again excluding downstream layers and framework-specific bias conventions.

So bidirectionality improves context but increases:

* parameters
* computation
* memory

---

# 🔗 Part 27 — BiGRU Parameter Count

Simplified one-direction GRU:

`3H(D+H+1)`

Bidirectional GRU:

# `6H(D+H+1)`

So:

```text
Bidirectional
≈ twice the recurrent computation
```

because the sequence is processed:

> once forward and once backward.

---

# 🧠 Part 28 — What Happens During Training?

Suppose the sequence is:

```text
x1 x2 x3 x4
```

Forward network computes:

```text
→h1 → →h2 → →h3 → →h4
```

Backward network computes:

```text
←h4 → ←h3 → ←h2 → ←h1
```

Then outputs at corresponding positions are aligned:

```text
Position 1 → [→h1 ; ←h1]
Position 2 → [→h2 ; ←h2]
Position 3 → [→h3 ; ←h3]
Position 4 → [→h4 ; ←h4]
```

These combined representations are passed to the task head.

---

# 🔄 Part 29 — Does Bidirectional RNN Still Use BPTT?

Yes.

Each direction is recurrent.

So we perform BPTT through:

* forward recurrent network
* backward recurrent network

Each network learns its own parameters.

Bidirectionality:

> does not replace BPTT.

---

# ⭐ Part 30 — Example: Named Entity Recognition

Sentence:

```text
I bought an Apple phone yesterday.
```

At:

`Apple`

forward context provides:

```text
I bought an Apple
```

Backward context provides information like:

```text
Apple phone yesterday
```

The word:

`phone`

strongly suggests:

> Apple = company/entity

rather than fruit.

So combining both directions improves token classification.

---

# 📖 Part 31 — Example: Ambiguous Word

Sentence:

```text
The fisherman sat on the bank and watched the water.
```

At:

`bank`

forward context:

```text
The fisherman sat on the bank
```

provides some clue.

Backward context:

```text
bank and watched the water
```

provides additional evidence.

Together:

> river-bank interpretation becomes much clearer.

---

# 🧠 Part 32 — Sequence-Level Classification

Suppose we want one output for an entire sentence:

> positive or negative sentiment.

A bidirectional recurrent model can create context-aware states across the entire sequence.

Then we may use:

* final states
* pooled states
* attention over states

to classify the sequence.

One possible representation is:

```text
[→h_T ; ←h_1]
```

because:

* `→h_T` has processed the complete sequence forward
* `←h_1` has processed the complete sequence backward

---

# ⚠️ Part 33 — Final Hidden State Can Be Confusing

For a forward RNN:

`→h_T`

is the final forward state.

For a backward RNN:

the state that has consumed the entire reversed sequence aligns with the beginning of the original sequence:

`←h_1`

So for whole-sequence summaries, be careful with:

> what “final backward state” means.

Framework APIs may return direction-specific final states in their own layout.

---

# 🧠 Part 34 — Bidirectional Does Not Mean Future Prediction

A Bidirectional RNN does not predict the future merely because it reads backward.

It simply uses:

> known future sequence elements as context.

For example:

```text
Full sentence already available
↓
process both directions
↓
better representation
```

This is fundamentally different from:

> forecasting an unknown future.

---

# 🔗 Part 35 — Bidirectional vs Forecasting

Suppose you have time series:

```text
x1, x2, x3, x4
```

and you want to predict:

`x5`

At inference time:

`x5`

is unknown.

So you cannot use future information beyond `x4`.

A bidirectional model might still be useful for:

> offline representation of historical windows

but not by peeking into observations that would be unavailable at prediction time.

---

# 🚨 Part 36 — Data Leakage Risk

This is extremely important in production and interviews.

Suppose we train a bidirectional model to predict:

> whether a machine will fail at time `t`.

If backward context includes:

> events occurring after time `t`

then the model may effectively know:

> what happens after the prediction point.

This creates:

# **future leakage**

The model may look excellent offline but fail in production.

---

# 🧠 Part 37 — Always Ask This Question

Before using bidirectionality:

> **Will this future information actually exist when I make the prediction in production?**

If:

### Yes

Bidirectionality may be valid.

### No

Using it may introduce leakage.

This is one of the most important architectural questions.

---

# 🔗 Part 38 — Offline vs Online Tasks

### Offline / Full-Sequence Tasks

Examples:

```text
Document classification
NER
POS tagging
Recorded speech transcription
Sentence understanding
```

Bidirectional processing:

> often useful.

---

### Online / Causal Tasks

Examples:

```text
Next-token generation
Real-time forecasting
Live anomaly detection
Streaming sensor prediction
```

Full bidirectional processing:

> usually inappropriate.

---

# 🧠 Part 39 — Relationship to What Comes Later

Bidirectional RNN teaches us an important idea:

> representations improve when a token can access context from both sides.

But the recurrent architecture still requires:

```text
t1 → t2 → t3 → ...
```

and:

```text
tT → tT-1 → tT-2 → ...
```

So context propagation remains:

> sequential.

Later, attention-based architectures will ask:

> Instead of carrying context step-by-step, can each position directly look at other relevant positions?

That will eventually lead us toward:

> Attention and Transformers.

But before that, we need another major concept:

# **Encoder–Decoder Architecture**

---

# 🔗 Part 40 — Our Evolution So Far

```text
Sequential Data
↓
Need Memory
↓
Vanilla RNN
↓
Long-Term Dependency Problem
↓
LSTM
↓
Better Controlled Memory
↓
GRU
↓
Simpler Gated Memory
↓
One-Directional Context Limitation
↓
Bidirectional RNN
↓
Past + Future Context
```

---

# 🎤 Part 41 — 30-Second Interview Answer

> **A Bidirectional RNN processes a sequence in both forward and backward directions using two separate recurrent networks. The forward network captures context from the past, while the backward network captures context from future sequence positions. Their hidden states are typically concatenated to create a representation informed by both sides of the sequence. Bidirectional RNNs are useful for full-sequence understanding tasks such as NER and sequence labeling, but they are generally unsuitable for causal or streaming prediction when future information is unavailable.**

---

# 🎤 Part 42 — Why Use Bidirectional RNN?

Strong answer:

> **A normal RNN representation at position `t` only has access to previous sequence elements. For tasks where the complete sequence is available, future elements may also provide important context. A Bidirectional RNN processes the sequence from both directions and combines the two hidden states, allowing each position to use both past and future information.**

---

# 🎤 Part 43 — Why Not Use It for Next-Token Prediction?

Strong answer:

> **Because the backward hidden state uses future sequence elements. During next-token prediction those future tokens are not yet available. Using them would leak the answer into the model and violate the causal prediction requirement.**

---

# ⚠️ Part 44 — High-Yield Interview Traps

### ❌ Bidirectional RNN is one RNN run forward and then backward

No.

It uses:

> two recurrent networks.

---

### ❌ Forward and backward directions usually share parameters

No.

They typically have separate learned parameters.

---

### ❌ Bidirectional RNN predicts unknown future tokens

No.

It uses future inputs only when they are already available.

---

### ❌ Bidirectional RNN is always better than one-directional RNN

No.

It depends on whether future context is available and valid.

---

### ❌ Bidirectional RNN solves vanishing gradients

Not by itself.

Cell type still matters.

---

### ❌ BiLSTM and Bidirectional RNN are exactly the same term

BiLSTM is:

> a Bidirectional RNN specifically using LSTM cells.

---

### ❌ Backward state at `t` contains only `x_(t+1)`

No.

It can summarize context from:

`x_t ... x_T`

through backward recurrence.

---

### ❌ Bidirectional model is safe for any time-series prediction

No.

Check for future leakage.

---

# 📐 Formula Flash Card

## Forward

`→h_t = f(W_x→ x_t + W_h→ →h_(t-1) + b_→)`

## Backward

`←h_t = f(W_x← x_t + W_h← ←h_(t+1) + b_←)`

## Combine

# `h_t = [→h_t ; ←h_t]`

If each direction has hidden size:

`H`

then:

# `combined size = 2H`

---

# 🧠 Part 45 — Five Things You Must Know Cold

1. **Bidirectional RNN = forward RNN + backward RNN.**
2. **Forward state sees past context; backward state sees future context.**
3. **The two directions usually have separate parameters.**
4. **Their states are commonly concatenated.**
5. **Bidirectional models require future context, so they are unsuitable for strictly causal prediction.**

---

# 🧠 Final Mental Model

For each position:

```text
Past Context
    ↓
Forward RNN
    ↓
  →h_t
    │
    │
    ├─────────────┐
    │             │
                  ↓
             [→h_t ; ←h_t]
                  ↑
    │             │
    └─────────────┘
    │
  ←h_t
    ↑
Backward RNN
    ↑
Future Context
```

Think:

```text
→h_t
→ what the PAST tells me about position t

←h_t
→ what the FUTURE tells me about position t

[→h_t ; ←h_t]
→ what BOTH sides tell me
```

---

# 🚀 Where We Go Next

Bidirectional RNN solves one important problem:

> **using context from both directions**

But now suppose our problem is not:

```text
one input sequence
→ representation/classification
```

Instead we want:

```text
one sequence
→ another sequence
```

For example:

```text
English sentence
→ French sentence
```

or:

```text
Long document
→ Summary
```

Now we face a new architectural problem:

> How do we convert an entire input sequence into another output sequence?

That leads us to:

# `08_Encoder_Decoder`

We will derive it from first principles:

```text
Input Sequence
↓
Need to Understand/Compress It
↓
Encoder
↓
Internal Representation
↓
Need to Generate Another Sequence
↓
Decoder
```

---

# ⭐ Golden Rule

> **A Bidirectional RNN creates context-aware representations by processing the same sequence in two directions: the forward network captures information from the past, the backward network captures information from the future, and both representations are combined at each position.**
