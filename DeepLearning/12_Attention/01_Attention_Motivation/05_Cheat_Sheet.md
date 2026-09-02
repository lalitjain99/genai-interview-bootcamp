# ⚡ Cheat Sheet — Attention Motivation

> **Goal:** Fast revision of why attention was needed, how it addresses the fixed context bottleneck, why dynamic source access matters, and how this idea leads toward the full attention mechanism and later Transformers.

---

# 🌟 1. Attention Motivation in One Line

> **Different decoder steps need different source information, so the decoder should dynamically access encoder states instead of relying on one fixed source summary.**

Core shift:

# **Fixed Compression → Dynamic Retrieval**

---

# 🧠 2. Classic Seq2Seq Problem

Classic encoder-decoder:

```text
Source
↓
Encoder
↓
One Fixed Context c
↓
Decoder
```

Often:

# `c = h_T`

All source information must pass through:

> one fixed-size representation.

This creates the:

# **Context Vector Bottleneck**

---

# 📈 3. Why Long Sequences Make It Worse

Source length grows:

```text
5 tokens
↓
50 tokens
↓
500 tokens
```

but context size may stay:

```text
512 dimensions
```

So:

```text
Source Information ↑
while
Communication Interface = Fixed
```

Result:

> increasing compression pressure.

---

# ⭐ 4. Capacity vs Accessibility

### Capacity

> How much information can the representation potentially store?

### Accessibility

> Can the decoder retrieve the exact source detail it needs right now?

Increasing hidden size mainly improves:

# **capacity**

Attention mainly improves:

# **accessibility**

---

# 🚫 5. Bigger Context Vector ≠ Attention

Increasing:

```text
512 → 1024 → 2048
```

creates a larger fixed representation.

But the decoder still asks:

> where inside this one vector is the information I need?

Attention instead changes:

> how source information is accessed.

---

# 🧠 6. Different Target Steps Need Different Source Parts

Source:

```text
I gave the red book to John yesterday
```

Different output steps may need:

```text
I
red book
John
yesterday
```

So why should every decoder step use:

> exactly the same source summary?

That question motivates attention.

---

# ⭐ 7. Keep All Encoder States

Instead of depending only on:

# `h_T`

retain:

# `h_1, h_2, ..., h_T`

Think:

```text
h1
h2
h3
...
hT
```

as a:

> source representation bank.

---

# 🔄 8. Fixed Context → Dynamic Context

Without attention:

# `c`

for every decoder step.

With attention:

```text
Step 1 → c1
Step 2 → c2
Step 3 → c3
```

So:

# `c_t`

depends on the current decoder step.

---

# 📐 9. Core Attention Equation

# `c_t = Σ_i α_(t,i) h_i`

where:

* `h_i` = encoder state at source position `i`
* `α_(t,i)` = relevance weight for source position `i`
* `c_t` = dynamic context at decoder step `t`

This is the single most important equation for this chapter.

---

# 🧠 10. Attention Weight Meaning

Example:

```text
α_t = [0.05, 0.10, 0.75, 0.10]
```

Interpretation:

> source position 3 matters most at this decoder step.

But the context still uses:

> all source states.

This is:

# **soft attention**

---

# 📐 11. Typical Weight Properties

Usually:

# `α_(t,i) ≥ 0`

and:

# `Σ_i α_(t,i) = 1`

This lets us interpret attention weights as:

> normalized source relevance.

---

# ⭐ 12. Source–Target Alignment

Attention learns:

> which source positions matter for each target position.

Conceptually:

```text
Target y1 → mostly source x2
Target y2 → mostly source x4
Target y3 → source x3 + x4
```

This is:

# **alignment**

---

# 🚫 13. Alignment Is Not Always One-to-One

Do not assume:

```text
one source token
↔
one target token
```

Possible mappings include:

* one-to-many
* many-to-one
* reordered phrases
* soft multi-position relevance

So attention learns:

> soft relevance, not rigid matching.

---

# 🔦 14. Searchlight Mental Model

Classic Seq2Seq:

> read the source, memorize it, turn away, generate from memory.

Attention:

> keep the source visible and shine a searchlight on the relevant part at each step.

That is a strong mental model for:

# **dynamic source access**

---

# 🧠 15. Attention Is Not Just Better Memory

### LSTM

Improves:

> recurrent memory flow.

### Attention

Improves:

> source information access.

So:

```text
LSTM
→ memory mechanism

Attention
→ retrieval/access mechanism
```

Historically, they were often used together.

---

# 🔍 16. Conceptual Attention Pipeline

At decoder step `t`:

```text
Decoder State / Need
+
Encoder States
↓
Relevance Scores
↓
Softmax
↓
Attention Weights
↓
Weighted Sum
↓
c_t
```

Then:

```text
c_t
+
decoder information
↓
next target prediction
```

---

# 📐 17. Relevance Score Preview

Conceptually:

# `e_(t,i) = score(decoder_state, h_i)`

Then:

# `α_(t,i) = softmax(e_(t,i))`

across source positions.

We have not yet derived the exact score function.

That belongs to the:

# **Attention Mechanism**

lecture.

---

# ⭐ 18. Why Softmax Fits Attention

Softmax converts raw relevance scores into weights that are:

* non-negative
* normalized
* differentiable

Example:

```text
Scores:
1.0  4.0  0.5

↓ softmax

Weights:
0.05  0.92  0.03
```

Now they can be used in:

# `c_t = Σ_i α_(t,i)h_i`

---

# 🧠 19. Soft Attention Is Differentiable

Instead of:

```text
choose h3 only
```

use:

```text
0.1h2 + 0.8h3 + 0.1h4
```

This weighted computation allows attention to be learned:

> end-to-end with gradient descent.

---

# ⚡ 20. Attention Creates Shorter Information Paths

Without attention:

```text
early source token
↓
many recurrent transitions
↓
final source state
↓
decoder
```

With attention:

```text
early encoder state
──────────────► decoder step
```

This improves:

* accessibility
* long-range detail preservation
* learning signal flow

---

# 🚫 21. Attention ≠ Vanishing Gradient Fix

Vanishing gradients:

> optimization / gradient-flow problem.

Context bottleneck:

> source communication/access problem.

Attention primarily solves:

# **source accessibility**

though shorter paths may also help optimization.

---

# ⭐ 22. Attention Did Not Immediately Replace LSTM

Early systems often used:

```text
BiRNN / LSTM Encoder
↓
Attention
↓
RNN / LSTM Decoder
```

Attention initially improved:

> encoder-decoder communication.

Only later did self-attention architectures remove recurrence.

---

# 🧠 23. Attention vs Bidirectional RNN

### BiRNN

Makes source states richer:

> left + right context.

### Attention

Lets decoder dynamically access:

> whichever source states matter.

So:

```text
BiRNN
→ richer representations

Attention
→ dynamic retrieval
```

---

# 🔍 24. Attention vs Beam Search

### Attention

Asks:

> Which source information should I use now?

### Beam Search

Asks:

> Which target hypothesis should I keep?

So:

```text
Attention
→ source-side access

Beam Search
→ target-side search
```

---

# 🔄 25. Attention vs Teacher Forcing

### Teacher Forcing

Controls:

> previous target-token input during training.

### Attention

Controls:

> source information available to decoder.

Different problems.

---

# 🔗 26. Attention vs Autoregressive Decoding

Autoregressive decoding:

```text
previous generated token
↓
next token
```

Attention adds:

```text
source states
↓
dynamic c_t
```

A decoder can use both:

```text
previous target token
+
decoder state
+
c_t
↓
next prediction
```

---

# 📐 27. Decoder Before and After Attention

Without attention:

# `d_t = f(y_(t-1), d_(t-1), c)`

with fixed:

`c`

With attention:

# `d_t = f(y_(t-1), d_(t-1), c_t)`

where:

# `c_t = Σ_i α_(t,i)h_i`

---

# ⭐ 28. Attention Matrix

If:

* source length = `T`
* target length = `T'`

attention weights can form roughly a:

# `T' × T`

matrix.

Each row:

> one target step.

Each column:

> one source position.

This can be visualized as an:

# **alignment map**

---

# ⚠️ 29. Attention Weights ≠ Perfect Explanation

Attention maps can show:

> which positions received larger weights.

But:

# **attention weight ≠ complete causal explanation**

Do not overstate interpretability.

---

# 🧠 30. Variable-Length Source Interface

Classic:

```text
T increases
but
decoder still gets one fixed c
```

Attention:

```text
T increases
→ h1...hT remain accessible
```

So the amount of directly accessible source representation:

> grows with source length.

This reduces fixed-summary pressure.

---

# ⚡ 31. Computational Cost

For:

* source length = `T`
* target length = `T'`

recurrent encoder-decoder attention conceptually evaluates roughly:

# `T × T'`

source-target relevance relationships.

So:

```text
Better Access
↔
More Computation
```

---

# 🧠 32. Why Attention Was Worth the Cost

Attention improves:

* long-sequence handling
* name preservation
* number/date preservation
* source-target alignment
* early-source accessibility
* detail retention

That made it a major architectural breakthrough.

---

# 🔗 33. Attention → Self-Attention

Current idea:

```text
Decoder
↓
attends to
↓
Encoder States
```

Next major idea:

```text
Sequence Representations
↓
attend to
↓
Other Representations
in the same sequence
```

This becomes:

# **Self-Attention**

and eventually:

# **Transformers**

---

# 🧠 34. Problem Taxonomy

```text
Vanishing Gradients
→ gradient flow

Context Bottleneck
→ fixed source communication

Teacher Forcing
→ training target history

Exposure Bias
→ train/inference mismatch

Beam Search
→ target sequence search

Attention
→ dynamic source access
```

Keep these separate.

---

# 🚫 35. High-Yield Traps

### ❌ Attention is just a bigger context vector

No.

### ❌ Attention uses the same `c` everywhere

No.

### ❌ Attention selects exactly one token

Usually no.

### ❌ Alignment must be one-to-one

No.

### ❌ Attention was only invented for vanishing gradients

No.

### ❌ Attention immediately replaced LSTM

No.

### ❌ Attention = Beam Search

No.

### ❌ Attention fixes exposure bias

No.

### ❌ Bigger hidden state completely solves the bottleneck

No.

### ❌ Attention weights perfectly explain reasoning

No.

---

# 📐 36. Formula Flash Card

### Classic Context

# `c = h_T`

### Dynamic Context

# `c_t = Σ_i α_(t,i)h_i`

### Conceptual Score

# `e_(t,i) = score(decoder_state, h_i)`

### Normalized Weights

# `α_(t,i) = softmax(e_(t,i))`

### Weight Constraints

# `α_(t,i) ≥ 0`

# `Σ_i α_(t,i) = 1`

---

# 🎤 37. 30-Second Interview Answer

> **Attention was introduced to address the fixed-context bottleneck in classic Seq2Seq models. Instead of compressing the entire source into one fixed vector and making every decoder step depend on the same summary, attention keeps all encoder hidden states accessible and constructs a step-specific context `c_t = Σ_i α_(t,i)h_i`. The weights indicate how relevant each source position is to the current decoder step. This changes the architecture from fixed compression to dynamic source retrieval, improving long-sequence handling, alignment, and detail preservation and setting the foundation for self-attention and Transformers.**

---

# 🧠 10 Things You Must Know Cold

1. **Classic Seq2Seq often uses one fixed source context.**
2. **Longer sources increase compression pressure.**
3. **Capacity and accessibility are different problems.**
4. **Attention primarily improves accessibility.**
5. **All encoder states remain available.**
6. **Each decoder step can get a different `c_t`.**
7. **`c_t` is a weighted sum of encoder states.**
8. **Attention weights represent soft relevance/alignment.**
9. **Attention solves source access, not target search.**
10. **Attention is the conceptual bridge to self-attention and Transformers.**

---

# 🧠 Final Mental Model

Before attention:

```text
Source
↓
Compress Everything
↓
ONE Fixed Vector
↓
Decoder Must Recover Everything
```

After attention:

```text
h1  h2  h3  ...  hT
 \   |   |       /
  \  |   |      /
 Current Decoder Need
          │
          ▼
   Relevance Scores
          │
          ▼
   Attention Weights
          │
          ▼
 c_t = Σ_i α_(t,i)h_i
          │
          ▼
 Current Target Prediction
```

The architectural shift:

# **Remember Everything in One Place**

becomes:

# **Retrieve What Matters Right Now**

And the next question is:

# **How do we actually compute `e_(t,i)` and `α_(t,i)`?**

That takes us into the full:

# **Attention Mechanism**

---

# ⭐ Golden Rule

> **Attention is fundamentally an information-access mechanism: rather than asking one fixed vector to preserve every source detail for the entire decoding process, it allows each target step to construct the source view it needs at that moment.**
