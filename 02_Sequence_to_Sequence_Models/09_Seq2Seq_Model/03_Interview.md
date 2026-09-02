# 🎤 Interview Questions — Sequence-to-Sequence (Seq2Seq) Model

> **Goal:** Test whether you truly understand the full Seq2Seq pipeline: source embeddings, encoder state, context transfer, decoder inputs, autoregressive generation, target shifting, training vs inference, loss computation, and the fixed-context bottleneck.

---

# 🟢 Beginner

## Q1. What is a Seq2Seq model?

### ✅ Answer

Seq2Seq stands for:

# **Sequence-to-Sequence**

It maps:

```text
Input Sequence
↓
Output Sequence
```

Usually through an:

```text
Encoder
↓
Context / Representation
↓
Decoder
```

Examples include:

* translation
* summarization
* speech-to-text
* sequence transformation

The key point is:

> both the input and output are sequences, and their lengths do not have to match.

---

## Q2. What is the difference between Seq2Seq and Encoder–Decoder?

### ✅ Answer

Encoder–Decoder describes the architecture pattern:

```text
Encoder
→ Representation
→ Decoder
```

Seq2Seq describes the task/model:

```text
Sequence
→ Sequence
```

A classic Seq2Seq model is commonly implemented using an Encoder–Decoder architecture.

So:

> **Encoder–Decoder = architectural pattern**
> **Seq2Seq = sequence transformation using that pattern**

In practice, the terms are often used interchangeably.

---

## Q3. What are the main components of a classic Seq2Seq model?

### ✅ Answer

A classic recurrent Seq2Seq model contains:

```text
Source Tokens
↓
Source Embedding
↓
Encoder
↓
Context Vector
↓
Decoder
↓
Output Projection
↓
Softmax
↓
Target Tokens
```

Important components include:

* source vocabulary
* source embedding layer
* encoder RNN/LSTM/GRU
* context representation
* target embedding layer
* decoder RNN/LSTM/GRU
* output projection
* softmax
* `<SOS>` and `<EOS>`

---

## Q4. Can source and target sequence lengths be different?

### ✅ Answer

Yes.

For example:

```text
Source:
x1 x2 x3 x4 x5

Target:
y1 y2 y3
```

There is no requirement that:

`T_src = T_tgt`

The decoder generates tokens until:

`<EOS>`

is produced.

This is one of the main strengths of Seq2Seq models.

---

# 🟡 Intermediate

## Q5. Why do source tokens need an embedding layer?

### ✅ Answer

Token IDs are only identifiers.

For example:

```text
"I" → 12
"love" → 47
"AI" → 205
```

The numbers:

`12, 47, 205`

do not directly encode semantic meaning.

So an embedding layer maps each token ID to a dense learned vector:

```text
Token ID
↓
Embedding Lookup
↓
Dense Vector
```

These vectors become the inputs to the encoder.

---

## Q6. Can source and target vocabularies have different sizes?

### ✅ Answer

Yes.

This is common in translation.

For example:

```text
English Vocabulary
→ Encoder

French Vocabulary
→ Decoder
```

So:

`V_src ≠ V_tgt`

is perfectly valid.

They may also use separate embedding matrices:

`E_src`

and:

`E_tgt`.

---

## Q7. How does the encoder work in a classic recurrent Seq2Seq model?

### ✅ Answer

For source sequence:

`X = (x_1, x_2, ..., x_T)`

the encoder computes:

`h_t = f_enc(x_t, h_(t-1))`

For example:

```text
x1 + h0 → h1
x2 + h1 → h2
x3 + h2 → h3
...
```

In the original fixed-context Seq2Seq model:

# `c = h_T`

The final encoder state acts as the source context representation.

---

## Q8. How does the decoder start generation?

### ✅ Answer

The decoder usually starts with:

`<SOS>` or `<BOS>`

and an initial state derived from the encoder.

A simple setup is:

`d_0 = c`

or:

`d_0 = W_c c + b_c`

Then:

```text
<SOS>
↓
Decoder
↓
First Target Token
```

The `<SOS>` token provides the decoder with a first input before any target token has been generated.

---

## Q9. Why do we shift the target sequence during training?

### ✅ Answer

Because the decoder is learning:

> given previous target tokens, predict the next target token.

Suppose target is:

```text
J'aime l'IA <EOS>
```

Decoder input:

```text
<SOS> J'aime l'IA
```

Expected output:

```text
J'aime l'IA <EOS>
```

So:

```text
Input:   <SOS>    J'aime    l'IA
Target:  J'aime   l'IA      <EOS>
```

The shift directly implements next-token prediction.

---

## Q10. How is a decoder hidden state converted into a target-token prediction?

### ✅ Answer

Suppose decoder state is:

`d_t`

First compute logits:

`z_t = W_o d_t + b_o`

Then apply:

`softmax(z_t)`

to obtain:

`P(y_t | y_<t, X)`

So:

```text
Decoder State
↓
Linear Projection
↓
Logits
↓
Softmax
↓
Target-Vocabulary Distribution
```

The decoder state itself is not a word or probability distribution.

---

# 🔴 Advanced

## Q11. What does autoregressive generation mean in Seq2Seq?

### ✅ Answer

Autoregressive generation means:

> each target token is generated conditioned on previously generated target tokens and the source sequence.

Mathematically:

# `P(Y|X) = Π_t P(y_t | y_<t, X)`

For example:

```text
<SOS>
↓
y1
↓
y2
↓
y3
↓
<EOS>
```

Each new prediction becomes part of the history used to generate the next token.

---

## Q12. What is the difference between Seq2Seq training and inference?

### ✅ Answer

During training, the correct target sequence is known.

So the decoder can often receive:

> the true previous target token.

For example:

```text
True y1
↓
Predict y2
```

During inference, the true target is unknown.

Therefore the decoder must use:

> its own previous prediction.

```text
Predicted y1
↓
Predict y2
```

This difference is central to:

> teacher forcing and exposure bias.

---

## Q13. How is the Seq2Seq model trained end-to-end?

### ✅ Answer

At each decoder step, a token-level loss is computed.

For the correct target token:

`L_t = -log P(y_t^correct)`

Total sequence loss:

`L = Σ_t L_t`

Gradients flow:

```text
Loss
↓
Output Projection
↓
Decoder
↓
Context / State Transfer
↓
Encoder
```

So both:

* encoder
* decoder

are trained jointly.

Because both sides are recurrent, BPTT is used through:

> source and target recurrent chains.

---

## Q14. Why must padded target positions be masked during training?

### ✅ Answer

Sequences in a batch often have different lengths.

For example:

```text
A B <EOS> PAD PAD
```

The `PAD` tokens are artificial and should not contribute to the training objective.

Correct:

```text
A      → loss
B      → loss
<EOS>  → loss
PAD    → ignore
PAD    → ignore
```

Otherwise the model learns to optimize meaningless padding positions and the loss becomes distorted.

---

## Q15. What is the main limitation of the original Seq2Seq model?

### ✅ Answer

The main limitation is:

# **the fixed context vector bottleneck**

The entire source sequence:

```text
x1 x2 x3 ... xT
```

is compressed into one fixed-dimensional representation:

`c`

and the decoder must generate the entire target from that representation.

For long or information-rich sources, this becomes difficult because:

* important details may be compressed away
* decoder cannot directly revisit a particular source position
* the same-size context must represent arbitrarily different sequence lengths

This limitation directly motivates:

# **Attention**

---

# ⭐ Staff Engineer Challenge

## Scenario

You are building a recurrent Seq2Seq translation system.

Architecture:

```text
BiGRU Encoder
↓
Final Context Vector
↓
GRU Decoder
↓
Target Sentence
```

The system behaves like this:

```text
Short sentences
→ BLEU / translation quality is good

Long sentences
→ entities disappear
→ dates are mistranslated
→ earlier clauses are omitted
```

---

## Question 1. What architectural issue would you investigate first?

### ✅ Answer

The first issue is the:

# **fixed context vector bottleneck**

Even though the encoder is bidirectional and gated, all source information is eventually compressed into:

> one fixed-dimensional context representation.

As the source grows longer, the decoder has difficulty recovering precise details such as:

* entity names
* dates
* earlier clauses

---

## Question 2. Why doesn't using a BiGRU encoder automatically solve this?

### ✅ Answer

BiGRU improves the quality of source encoding because each source representation can use:

> past + future source context.

GRU also provides gated recurrence.

But if the architecture still ends with:

```text
All Source Information
↓
One Context Vector
↓
Decoder
```

the communication bottleneck remains.

So:

```text
BiGRU
→ better encoder

but

one fixed context
→ same structural limitation
```

---

## Question 3. What would happen if you simply increased the context size from 256 to 2048?

### ✅ Answer

It may improve capacity.

However, it does not fundamentally remove the architectural problem.

The decoder still receives:

> one fixed representation for the entire source.

It still cannot directly request:

```text
Give me the encoder information for source position 17.
```

So larger hidden size may help, but it does not solve:

> selective source retrieval.

---

## Question 4. Your target vocabulary is 40,000 and decoder hidden size is 512. What is the output projection shape?

### ✅ Answer

Decoder state:

`d_t ∈ R^512`

Need:

`40,000`

logits.

Therefore:

`W_o ∈ R^(40000×512)`

`b_o ∈ R^40000`

and:

`z_t ∈ R^40000`

Then softmax creates:

> 40,000 target-token probabilities.

---

## Question 5. What should happen to target padding tokens during loss computation?

### ✅ Answer

They should be:

> masked out.

For example:

```text
Target:
A B C <EOS> PAD PAD
```

Loss should be calculated for:

```text
A
B
C
<EOS>
```

but not:

```text
PAD
PAD
```

This is especially important when batching variable-length target sequences.

---

## Question 6. Why might offline teacher-forced accuracy look better than actual inference quality?

### ✅ Answer

During teacher-forced training/evaluation, the decoder receives:

> correct previous target tokens.

During actual inference, it receives:

> its own previous predictions.

If it makes an error:

```text
Wrong token
↓
wrong next decoder input
↓
changed state
↓
future predictions may degrade
```

So teacher-forced token accuracy does not fully reproduce autoregressive inference conditions.

---

## Question 7. What architectural change would you propose next?

### ✅ Answer

Retain the complete encoder state sequence:

```text
h1, h2, h3, ..., hT
```

and allow every decoder step to determine:

> which encoder positions are relevant to the current output token.

Instead of:

```text
Whole Source
↓
one fixed context
↓
all decoder steps
```

we want:

```text
Encoder States
↓
dynamic context per decoder step
```

This leads to:

# **Attention**

---

# ⚡ Rapid-Fire Traps

### ❌ “Seq2Seq requires the same source and target vocabulary.”

False.

---

### ❌ “Source token IDs can be fed directly as semantic representations.”

Not usually. They are typically embedded first.

---

### ❌ “The decoder gets the whole correct target during inference.”

False.

It generates incrementally.

---

### ❌ “Target shifting means deleting the first target word.”

No.

It means aligning:

```text
<SOS> y1 y2 ...
```

with:

```text
y1 y2 y3 ...
```

---

### ❌ “Softmax updates the decoder hidden state.”

No.

Softmax converts output logits into probabilities.

---

### ❌ “The sequence loss is only calculated at the final token.”

No.

Classic Seq2Seq typically accumulates token-level losses across target positions.

---

### ❌ “Padding should receive the same loss as real target tokens.”

No.

Mask it.

---

### ❌ “Greedy decoding always finds the highest probability output sequence.”

False.

It finds the highest-probability token locally at each step.

---

### ❌ “A BiGRU decoder is always better.”

No.

An autoregressive decoder cannot use unknown future target tokens.

---

### ❌ “Classic Seq2Seq solved the long-source problem.”

No.

Its fixed-context design exposed that limitation.

---

# 📐 Formula Flash Card

### Source

`X = (x_1, ..., x_T)`

### Target

`Y = (y_1, ..., y_T')`

### Source Embedding

`x_t = E_src[token_t]`

### Encoder

`h_t = f_enc(x_t, h_(t-1))`

### Classic Context

# `c = h_T`

### Decoder

`d_t = f_dec(e(y_(t-1)), d_(t-1), c)`

### Output

`z_t = W_o d_t + b_o`

### Probability

`P(y_t | y_<t, X) = softmax(z_t)`

### Full Sequence Probability

# `P(Y|X) = Π_t P(y_t | y_<t, X)`

### Token Loss

`L_t = -log P(y_t^correct)`

### Sequence Loss

# `L = Σ_t L_t`

with padding positions masked.

---

# 🎤 30-Second Interview Answer

> **A classic Seq2Seq model maps a variable-length source sequence to a variable-length target sequence using an encoder and autoregressive decoder. Source and target tokens are embedded separately, the encoder converts the source into a context representation, and the decoder starts from `<SOS>` and predicts one target token at a time until `<EOS>`. During training, the target is shifted so each decoder step learns next-token prediction, often using teacher forcing, and token-level cross-entropy is backpropagated through both decoder and encoder. The key limitation is that classic Seq2Seq compresses the entire source into one fixed-dimensional context vector, which becomes a bottleneck for long sequences.**

---

# 🧠 Final Interview Mental Model

Think of the model in four stages:

```text
1. REPRESENT SOURCE
Source IDs
↓
Source Embeddings
↓
Encoder
```

```text
2. TRANSFER SOURCE INFORMATION
Encoder
↓
Context c
↓
Decoder Initial State
```

```text
3. GENERATE TARGET
<SOS>
↓
y1
↓
y2
↓
...
↓
<EOS>
```

```text
4. LEARN
Target Shift
↓
Cross-Entropy
↓
Backpropagation Through Decoder + Encoder
```

And keep the central probability equation in mind:

# `P(Y|X) = Π_t P(y_t | y_<t, X)`

Finally:

```text
Classic Seq2Seq
↓
One Context Vector
↓
Long-Sequence Bottleneck
↓
Need Dynamic Source Access
↓
Attention
```

---

# ⭐ Golden Rule

> **A Seq2Seq model is conditional autoregressive generation: encode the source, condition the decoder on it, predict the target one token at a time, and train the complete source-to-target pipeline end-to-end.**
