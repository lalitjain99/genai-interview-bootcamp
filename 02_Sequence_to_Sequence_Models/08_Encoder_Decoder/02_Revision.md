# 📝 Revision — Encoder–Decoder Architecture

> **Goal:** Quickly revise why Encoder–Decoder architecture was needed, how encoder and decoder work together, how variable-length sequence generation happens, and why the original fixed context vector became a bottleneck.

---

# 🌟 1. Why Encoder–Decoder Was Needed

Earlier sequence models often solved problems like:

```text
Sequence
→ Single Label
```

Examples:

```text
Review
→ Positive / Negative

Sensor Sequence
→ Anomaly / Normal
```

But some tasks require:

# **Sequence → Sequence**

Examples:

```text
English Sentence
→ French Sentence

Document
→ Summary

Audio Frames
→ Text
```

This creates a new architectural problem:

> How do we read one sequence and generate another sequence, possibly of a different length?

That leads to:

# **Encoder–Decoder**

---

# 🧠 2. Core Idea

Split the problem into two jobs.

```text
Input Sequence
↓
ENCODER
↓
Representation
↓
DECODER
↓
Output Sequence
```

Think:

```text
Encoder
→ Understand the input

Decoder
→ Generate output from that understanding
```

---

# ⭐ 3. Encoder Role

The encoder reads:

```text
x1, x2, ..., xT
```

and updates its hidden state:

`h_t = f_enc(x_t, h_(t-1))`

At the end:

`h_T`

contains a learned representation of the input.

In the original architecture:

# `Context = h_T`

---

# 🧠 4. Context Vector

The context vector is:

> a learned compressed representation of the complete input sequence.

Important:

```text
Context Vector
≠ exact copy of the input
```

It is a:

> fixed-dimensional learned summary.

Mental model:

```text
Input Sequence
↓
Encoder
↓
Compress
↓
Context Vector
```

---

# ⚠️ 5. Context Vector Does Not Store Every Token Exactly

Do not say:

> “The context vector contains every word.”

Better:

> **The context vector contains a learned task-relevant representation of the source sequence.**

It may preserve:

* semantic information
* sequence relationships
* relevant features

but it is still a compressed vector.

---

# 🔗 6. Which Cells Can the Encoder Use?

Encoder–Decoder is an architecture pattern.

The encoder can use:

```text
Vanilla RNN
LSTM
GRU
BiLSTM
BiGRU
```

So:

> Encoder–Decoder is not tied to one recurrent cell.

---

# ⭐ 7. Why Can the Encoder Be Bidirectional?

If the complete source sequence is already available:

```text
Source Sentence
→ fully known
```

then encoder can process:

```text
left → right
+
right → left
```

So BiLSTM/BiGRU encoders can be useful.

But this logic does **not** automatically apply to the decoder.

---

# 🚫 8. Why Decoder Is Usually Not Bidirectional

The decoder generates:

```text
y1 → y2 → y3 → ...
```

When generating:

`y_t`

future outputs:

```text
y_(t+1), y_(t+2), ...
```

do not exist yet.

So a normal autoregressive decoder must be:

> causal / unidirectional.

Otherwise it would use future-answer information.

---

# 🧠 9. Decoder Role

The decoder uses:

* encoder context
* previous decoder state
* previous output token

to generate the next token.

Conceptually:

```text
Context
+
Previous Decoder State
+
Previous Output
↓
Decoder
↓
Next Token
```

---

# 📐 10. Decoder State

Let decoder hidden state be:

`d_t`

A simplified recurrence:

`d_t = f_dec(y_(t-1), d_(t-1), context)`

Then:

```text
d_t
↓
Linear Layer
↓
Logits
↓
Softmax
↓
Probability Over Vocabulary
```

---

# ⭐ 11. How Does Decoder Start?

The decoder needs an initial token.

So we use:

# `<SOS>`

or:

# `<BOS>`

Meaning:

> Start / Beginning Of Sequence

Example:

```text
<SOS>
↓
Decoder
↓
"Je"
```

---

# ⭐ 12. How Does Decoder Stop?

The decoder learns to produce:

# `<EOS>`

Meaning:

> End Of Sequence

When `<EOS>` is generated:

> decoding stops.

This allows:

> variable-length output generation.

---

# 🧠 13. Why Input and Output Lengths Can Differ

Encoder reads:

```text
x1 x2 x3 x4 x5
```

Decoder may generate:

```text
y1 y2 y3 <EOS>
```

So:

# `T_input ≠ T_output`

There is no requirement for one-to-one token alignment.

This is why Encoder–Decoder works well for:

* translation
* summarization
* transcription
* response generation

---

# 📖 14. Tiny Translation Example

Input:

```text
I am happy
```

Output:

```text
Je suis heureux
```

Encoder:

```text
I
↓
h1

am
↓
h2

happy
↓
h3
```

Context:

`c = h3`

Decoder starts:

```text
<SOS>
↓
Je
↓
suis
↓
heureux
↓
<EOS>
```

---

# 🔗 15. Encoder State Becomes Decoder Starting Knowledge

A common simple setup:

# `d_0 = h_T`

So:

```text
Encoder Final State
↓
Decoder Initial State
```

Think:

```text
Encoder
→ What did I understand?

Decoder
→ Start generating from that understanding
```

---

# ⚠️ 16. Encoder and Decoder Hidden Sizes Need Not Match

Suppose:

```text
Encoder hidden = 512
Decoder hidden = 256
```

Then direct assignment:

`d_0 = h_T`

does not work dimensionally.

Use a projection:

`d_0 = W_c h_T + b_c`

So the two sides can have:

> different state dimensions.

---

# 🧠 17. LSTM Encoder–Decoder State Transfer

LSTM has:

```text
h_T
c_T
```

So a decoder LSTM may initialize:

```text
h_0^dec = h_T^enc
c_0^dec = c_T^enc
```

or projected versions of them.

---

# 🧠 18. GRU Encoder–Decoder State Transfer

GRU has only:

`h_T`

So:

```text
h_0^dec = h_T^enc
```

or a transformed version.

This makes state transfer simpler than with LSTM.

---

# ⭐ 19. From Decoder State to Token Probability

Decoder hidden state is not itself a word.

We compute:

`z_t = W_o d_t + b_o`

Then:

`P(y_t | ...) = softmax(z_t)`

So:

```text
Decoder State
↓
Linear Layer
↓
Logits
↓
Softmax
↓
Vocabulary Probabilities
```

---

# 🧮 20. Example Probability Distribution

Suppose vocabulary:

```text
Je
Tu
Nous
Il
```

Softmax output:

```text
Je    → 0.75
Tu    → 0.10
Nous  → 0.10
Il    → 0.05
```

With greedy decoding:

> choose `Je`.

Later, beam search improves upon purely greedy selection.

---

# 🧠 21. Decoder Is Autoregressive

The decoder generates one token at a time.

```text
y1
↓
y2
↓
y3
↓
...
```

Each next token depends on:

> previously generated tokens.

Probability:

# `P(Y|X) = Π_t P(y_t | y_<t, X)`

This is the core autoregressive idea.

---

# ⭐ 22. Training vs Inference

This distinction is very important.

## Training

Correct target sequence is known.

So previous true target token may be fed to the decoder.

```text
True previous token
↓
Predict next token
```

## Inference

True target does not exist.

So decoder must use:

> its own previous prediction.

```text
Previous predicted token
↓
Predict next token
```

This later leads to:

# **Teacher Forcing**

---

# ⚠️ 23. Early Inference Errors Can Propagate

Suppose decoder makes a wrong prediction at:

`t=2`

Then that wrong token may become input at:

`t=3`.

So:

```text
Early Error
↓
Changes Decoder State
↓
Changes Later Predictions
```

This is one of the challenges of autoregressive decoding.

---

# 🔗 24. Encoder–Decoder vs Simple Many-to-Many RNN

Aligned many-to-many:

```text
x1 → y1
x2 → y2
x3 → y3
```

Example:

> POS tagging

Encoder–Decoder:

```text
x1 x2 x3 x4
↓
Encoder
↓
Context
↓
Decoder
↓
y1 y2 y3
```

No direct input-output positional alignment is required.

---

# ⭐ 25. Classic Use Cases

### Machine Translation

```text
English
→ French
```

### Summarization

```text
Long Text
→ Short Summary
```

### Speech Recognition

```text
Audio Frames
→ Text
```

### Response Generation

```text
Input Sequence
→ Output Sequence
```

---

# 📐 26. Mathematical Encoder View

Input:

`X = (x_1, ..., x_T)`

Encoder:

`h_t = f_enc(x_t, h_(t-1))`

Original context:

# `c = h_T`

So:

# `c = Encoder(X)`

---

# 📐 27. Mathematical Decoder View

Output:

`Y = (y_1, ..., y_T')`

Decoder:

`d_t = f_dec(y_(t-1), d_(t-1), c)`

Then:

`P(y_t | y_<t, c) = softmax(W_o d_t + b_o)`

---

# 🧠 28. What Does Decoder Learn?

Decoder learns:

### Source Conditioning

> What output should correspond to this input?

and:

### Target Sequence Structure

> What should come next given what I already generated?

For translation:

```text
Encoder Context
→ source meaning

Previous Target Tokens
→ target-language structure
```

---

# 🔐 29. Encoder and Decoder Usually Have Separate Parameters

Encoder:

> learns source representation.

Decoder:

> learns target generation.

So usually:

```text
Encoder Weights
≠
Decoder Weights
```

Even if both use the same cell type.

---

# ⭐ 30. Major Strength

Encoder–Decoder supports:

# **Variable-Length Input → Variable-Length Output**

This was a major advance for neural sequence transformation.

---

# 🚨 31. Major Weakness: Fixed Context Bottleneck

Original architecture compresses:

```text
x1 x2 ... xT
```

into:

```text
one fixed-size vector c
```

No matter whether the input has:

```text
5 tokens
50 tokens
500 tokens
```

the decoder gets the same-sized context representation.

This creates:

# **Context Vector Bottleneck**

---

# 🧠 32. Why Is This a Bottleneck?

Think of:

```text
Short Paragraph
→ one sticky note
```

Maybe manageable.

But:

```text
Entire Book
→ same one sticky note
```

Important details may be lost.

Likewise:

> long source sequences are difficult to compress perfectly into one fixed-dimensional vector.

---

# ⚠️ 33. Decoder Cannot Directly Revisit Earlier Encoder States

Original architecture:

```text
h1 h2 h3 ... hT
             │
             ▼
             c
             │
             ▼
          Decoder
```

The decoder relies mainly on:

> `c`

rather than directly selecting:

```text
h1
h2
h3
...
```

This becomes problematic for long inputs.

---

# ⭐ 34. The Question That Leads to Attention

Instead of forcing everything through:

`c`

what if decoder could ask:

> “Which encoder states are relevant to the token I am generating now?”

Conceptually:

```text
h1 h2 h3 h4
│  │  │  │
└──┴──┴──┴──→ Decoder selectively looks at them
```

This is the core idea that eventually leads to:

# **Attention**

---

# 🔗 35. Encoder–Decoder vs Seq2Seq

### Encoder–Decoder

Architecture pattern:

```text
Encoder
→ Representation
→ Decoder
```

### Seq2Seq

Usually refers to using this architecture for:

```text
Sequence
→ Sequence
```

They are closely related, but not conceptually identical.

---

# 🚫 36. High-Yield Interview Traps

### ❌ Encoder generates target tokens

No.

Decoder generates target tokens.

---

### ❌ Context vector stores exact input

No.

It is compressed representation.

---

### ❌ Input/output lengths must match

No.

---

### ❌ Decoder outputs all tokens independently

No.

It is autoregressive in recurrent Seq2Seq.

---

### ❌ Decoder can always be bidirectional

No.

Future target tokens are unavailable during autoregressive inference.

---

### ❌ `<EOS>` is just padding

No.

It means:

> generation ends.

---

### ❌ Encoder and decoder must use the same cell type

No.

---

### ❌ Encoder–Decoder completely solves long-sequence problems

No.

The fixed context vector creates a bottleneck.

---

# 📐 37. Formula Flash Card

### Encoder

`h_t = f_enc(x_t, h_(t-1))`

### Original Context

# `c = h_T`

### Decoder

`d_t = f_dec(y_(t-1), d_(t-1), c)`

### Output Logits

`z_t = W_o d_t + b_o`

### Token Probability

`P(y_t | y_<t, c) = softmax(z_t)`

### Sequence Probability

# `P(Y|X) = Π_t P(y_t | y_<t, X)`

---

# 🎤 38. 30-Second Interview Answer

> **Encoder–Decoder is an architecture for mapping one variable-length sequence to another. The encoder processes the source sequence and converts it into a learned representation, traditionally the final encoder hidden state. The decoder is initialized or conditioned using that representation and generates the target sequence autoregressively, one token at a time, until an end-of-sequence token is produced. The original architecture's key limitation is that the entire input must be compressed into one fixed-size context vector, which becomes a bottleneck for long sequences and eventually motivates attention.**

---

# 🧠 39. Seven Things You Must Know Cold

1. **Encoder = reads/represents input**
2. **Decoder = generates output**
3. **Original context = usually final encoder state**
4. **`<SOS>` starts decoding**
5. **`<EOS>` stops decoding**
6. **Input/output lengths can differ**
7. **Fixed context vector eventually becomes a bottleneck**

---

# 🧠 Final Mental Model

```text
SOURCE SEQUENCE
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
TARGET SEQUENCE
```

Think:

```text
Encoder
→ What did I read?

Context
→ What did I understand?

Decoder
→ What should I generate?
```

And remember the coming limitation:

```text
Whole Input
↓
One Fixed Vector
↓
Too Much Compression
↓
Context Bottleneck
↓
Attention
```

---

# ⭐ Golden Rule

> **Encoder–Decoder separates understanding from generation: the encoder compresses the source sequence into a learned representation, and the decoder uses that representation plus its own generation history to produce the target sequence one token at a time.**
