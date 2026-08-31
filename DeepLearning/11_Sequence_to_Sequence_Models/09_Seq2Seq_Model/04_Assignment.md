# 💡 Assignment — Sequence-to-Sequence (Seq2Seq) Model

> **Goal:** Apply Seq2Seq concepts by tracing the complete source-to-target pipeline, reasoning about embeddings, state transfer, target shifting, autoregressive decoding, loss, masking, and the fixed-context bottleneck.

---

# 🧩 Task 1 — Identify the Full Seq2Seq Pipeline

Fill the blanks:

```text
Source Tokens
↓
__________
↓
Encoder
↓
__________
↓
Decoder
↓
__________
↓
Softmax
↓
Target Tokens
```

### ✅ Answer

```text
Source Tokens
↓
Source Embeddings
↓
Encoder
↓
Context Vector
↓
Decoder
↓
Logits
↓
Softmax
↓
Target Tokens
```

---

# 🧠 Task 2 — Seq2Seq or Not?

Classify the following tasks.

### A. Review → Positive/Negative

### B. English sentence → French sentence

### C. Audio frames → Transcript

### D. Image → Cat/Dog

### E. Article → Summary

### ✅ Answer

```text
A → not inherently Seq2Seq
B → Seq2Seq
C → Seq2Seq
D → not Seq2Seq
E → Seq2Seq
```

Seq2Seq means:

> both input and output are sequences.

---

# 🔢 Task 3 — Why Are Token IDs Not Enough?

Suppose:

```text
"I"    → 12
"love" → 47
"AI"   → 205
```

A developer says:

> “We can directly feed 12, 47, and 205 into the RNN because these numbers represent the words.”

What is wrong?

### ✅ Answer

Token IDs are:

> identifiers, not semantic numerical values.

For example:

`205`

does not mean that `"AI"` is somehow semantically 193 units larger than `"I"`.

Instead:

```text
Token ID
↓
Embedding Lookup
↓
Dense Learned Vector
```

The embedding provides a meaningful learnable representation.

---

# 📐 Task 4 — Source Embedding Shape

Suppose:

```text
Source vocabulary = 20,000
Source embedding dimension = 300
```

What is the source embedding matrix shape?

### ✅ Answer

# `E_src ∈ R^(20000 × 300)`

Each source token selects:

> one 300-dimensional embedding vector.

---

# 📐 Task 5 — Target Embedding Shape

Suppose:

```text
Target vocabulary = 30,000
Target embedding dimension = 256
```

What is:

`E_tgt`

shape?

### ✅ Answer

# `E_tgt ∈ R^(30000 × 256)`

There is no requirement that:

```text
V_src = V_tgt
```

or:

```text
D_src = D_tgt
```

---

# 🔄 Task 6 — Trace the Encoder

Source sequence:

```text
x1 = "I"
x2 = "love"
x3 = "AI"
```

Using:

`h_t = f_enc(x_t, h_(t-1))`

complete:

```text
x1 + h0
↓
____

x2 + ____
↓
____

x3 + ____
↓
____
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

Classic Seq2Seq context:

# `c = h3`

---

# 🧠 Task 7 — What Is the Context Vector?

Which description is better?

### Option A

> The context vector contains an exact copy of every source token.

### Option B

> The context vector is a learned compressed representation of the source sequence.

### ✅ Answer

# Option B

The context vector attempts to preserve:

> information useful for target generation.

It is not literal token storage.

---

# 📐 Task 8 — Decoder Initialization

Suppose:

```text
c ∈ R^256
decoder hidden state ∈ R^256
```

Can we use:

`d_0 = c`

### ✅ Answer

Yes.

The dimensions match.

---

# 📐 Task 9 — Decoder Dimension Mismatch

Now suppose:

```text
c ∈ R^512
d_0 ∈ R^256
```

Can we directly use:

`d_0 = c`?

### ✅ Answer

No.

We can use:

# `d_0 = W_c c + b_c`

with:

```text
W_c ∈ R^(256 × 512)

b_c ∈ R^256
```

Then the source context is projected into decoder-state space.

---

# 🚀 Task 10 — Why `<SOS>`?

Before the decoder generates the first target token, what previous target token exists?

### ✅ Answer

None.

So we provide:

# `<SOS>` / `<BOS>`

as the initial decoder input.

Conceptually:

```text
<SOS>
↓
Decoder
↓
y1
```

---

# 🛑 Task 11 — Why `<EOS>`?

Suppose output lengths vary.

How does decoder know when generation is finished?

### ✅ Answer

It learns to predict:

# `<EOS>`

When `<EOS>` is produced:

> generation terminates.

This avoids requiring a fixed target length.

---

# 🔄 Task 12 — Target Shifting

Suppose target sequence is:

```text
A B C <EOS>
```

Create:

1. decoder input
2. decoder expected output

### ✅ Answer

Decoder input:

```text
<SOS> A B C
```

Expected output:

```text
A B C <EOS>
```

Alignment:

```text
Input:   <SOS>   A   B   C
Target:    A     B   C  <EOS>
```

---

# 🧠 Task 13 — Why Shift the Target?

Why not use:

```text
Input:
A B C <EOS>

Target:
A B C <EOS>
```

at the same positions?

### ✅ Answer

Because the decoder should learn:

> predict the next token from previous tokens.

We want:

```text
<SOS>
→ predict A

A
→ predict B

B
→ predict C

C
→ predict <EOS>
```

This matches:

# `P(y_t | y_<t, X)`

---

# 📐 Task 14 — Output Projection Shape

Suppose:

```text
Decoder hidden size H = 256
Target vocabulary size V = 50,000
```

For:

`z_t = W_o d_t + b_o`

find the shapes.

### ✅ Answer

```text
d_t ∈ R^256

W_o ∈ R^(50000 × 256)

b_o ∈ R^50000

z_t ∈ R^50000
```

Softmax then creates:

> 50,000 target-token probabilities.

---

# 🧮 Task 15 — Greedy Token Selection

At one decoder step:

```text
A      → 0.15
B      → 0.62
C      → 0.13
<EOS>  → 0.10
```

What does greedy decoding select?

### ✅ Answer

# `B`

because:

`0.62`

is the maximum probability.

---

# 🧮 Task 16 — Calculate Sequence Probability

Suppose the model generates:

```text
A B <EOS>
```

with:

```text
P(A|X) = 0.8
P(B|A,X) = 0.6
P(<EOS>|A,B,X) = 0.9
```

Calculate:

`P(Y|X)`

### ✅ Answer

`P(Y|X) = 0.8 × 0.6 × 0.9`

# `= 0.432`

This follows:

# `P(Y|X) = Π_t P(y_t | y_<t, X)`

---

# 🧮 Task 17 — Compare Two Candidate Sequences

Suppose:

### Sequence A

```text
P(A|X) = 0.7
P(B|A,X) = 0.5
```

Total:

`0.7 × 0.5 = 0.35`

### Sequence B

```text
P(C|X) = 0.6
P(D|C,X) = 0.9
```

What is Sequence B probability, and which sequence is more probable?

### ✅ Answer

Sequence B:

`0.6 × 0.9`

# `= 0.54`

So:

```text
Sequence A = 0.35
Sequence B = 0.54
```

Sequence B is more probable overall.

This shows why:

> the locally highest first token does not necessarily produce the globally best sequence.

---

# 🔥 Task 18 — Why Greedy Decoding Can Fail

At step 1:

```text
A → 0.7
B → 0.6
```

Greedy chooses:

`A`

Later:

```text
best continuation after A = 0.1

best continuation after B = 0.9
```

Compare:

### Path A

`0.7 × 0.1`

### Path B

`0.6 × 0.9`

### ✅ Answer

Path A:

`0.07`

Path B:

`0.54`

So despite:

`A`

having the best first-step probability,

the sequence beginning with:

`B`

is much better overall.

This motivates:

> Beam Search.

---

# 🎓 Task 19 — Training vs Inference

Suppose correct target is:

```text
A B C
```

### During training

When predicting:

`B`

what may be fed as previous token?

### During inference

What is fed?

### ✅ Answer

Training:

> the true `A` may be fed.

Inference:

> whatever the model predicted at the previous step is fed.

So:

```text
Training
→ ground-truth previous token may be available

Inference
→ model's previous prediction
```

---

# 🚨 Task 20 — Error Propagation

Correct sequence:

```text
A B C
```

Model predicts:

```text
A X ...
```

Explain why the error at `X` can affect later predictions.

### ✅ Answer

During inference:

```text
X
↓
becomes next decoder input/history
↓
changes decoder state
↓
changes future distributions
```

So one early mistake can propagate.

This is one of the important consequences of:

> autoregressive decoding.

---

# 📐 Task 21 — Cross-Entropy Loss

Suppose correct token probability is:

`0.8`

Token loss:

`L = -log(0.8)`

Now another model assigns:

`0.02`

to the correct token.

Which model gets larger loss?

### ✅ Answer

The second model.

Because:

```text
-log(0.02)
>
-log(0.8)
```

So cross-entropy penalizes:

> low probability assigned to the correct token.

---

# 🧮 Task 22 — Sequence Loss

Suppose correct-token losses for three output positions are:

```text
0.2
0.5
0.1
```

Using sum reduction:

### ✅ Answer

`L = 0.2 + 0.5 + 0.1`

# `= 0.8`

The model may instead average depending on the implementation.

---

# 📦 Task 23 — Padding Mask

Target batch entry:

```text
A B <EOS> PAD PAD
```

Which positions should contribute to token-level loss?

### ✅ Answer

Include:

```text
A
B
<EOS>
```

Ignore:

```text
PAD
PAD
```

So:

```text
Real target tokens
→ include

Padding
→ mask
```

---

# 🚨 Task 24 — What Happens If Padding Is Not Masked?

### ✅ Answer

The model is trained to predict:

> artificial padding positions.

This can:

* distort the loss
* waste capacity
* bias training toward shorter/longer padding patterns
* produce misleading metrics

Padding should usually be excluded from meaningful token loss.

---

# 🔗 Task 25 — Does the Encoder Learn From Decoder Loss?

A developer says:

> “Cross-entropy is calculated at decoder outputs, so only the decoder gets trained.”

Correct?

### ✅ Answer

No.

The computational graph is connected:

```text
Loss
↓
Output Layer
↓
Decoder
↓
Context
↓
Encoder
```

So gradients propagate into:

> encoder parameters as well.

Seq2Seq is trained:

# **end-to-end**

---

# 🔄 Task 26 — Where Does BPTT Happen?

### ✅ Answer

Because both components are recurrent:

```text
Encoder
→ BPTT through source time steps

Decoder
→ BPTT through target time steps
```

The gradient also crosses the encoder-decoder connection.

---

# 🧠 Task 27 — LSTM Seq2Seq

An LSTM encoder returns:

```text
h_T^enc
c_T^enc
```

What can initialize an LSTM decoder?

### ✅ Answer

A simple approach:

```text
h_0^dec = h_T^enc

c_0^dec = c_T^enc
```

or projections of them.

---

# 🧠 Task 28 — GRU Seq2Seq

What changes for GRU?

### ✅ Answer

GRU has one recurrent state.

So:

```text
h_0^dec = h_T^enc
```

or projected version.

There is:

> no separate LSTM-style cell state.

---

# ↔️ Task 29 — Bidirectional Encoder

Suppose each direction of a BiGRU encoder has:

`H = 128`

and final directional states are concatenated.

What is context size?

### ✅ Answer

```text
Forward = 128
Backward = 128
```

Therefore:

# `Context = 256 dimensions`

If decoder hidden size is 128, a projection may be needed.

---

# 🚫 Task 30 — Bidirectional Decoder

Why is a standard bidirectional decoder inappropriate for autoregressive inference?

### ✅ Answer

When predicting:

`y_t`

future target tokens:

```text
y_(t+1), y_(t+2), ...
```

do not yet exist.

A backward target decoder would require:

> information from the future target.

That causes:

> leakage / impossible inference-time dependency.

---

# 🚨 Task 31 — Diagnose Long-Sequence Failure

Your model performs well for:

```text
10–20 source tokens
```

but badly for:

```text
150 source tokens
```

Architecture:

```text
GRU Encoder
↓
h_T
↓
GRU Decoder
```

What should you suspect first?

### ✅ Answer

# **Fixed context vector bottleneck**

All source information must pass through:

`h_T`

regardless of source length.

The decoder cannot directly access earlier encoder states.

---

# 🧠 Task 32 — Why Doesn't a Larger Hidden State Fully Solve It?

Suppose hidden state increases:

```text
256 → 2048
```

Why is the problem not structurally eliminated?

### ✅ Answer

The model still uses:

```text
Entire Source
↓
One Fixed Representation
↓
Decoder
```

The decoder cannot selectively revisit:

```text
h1
h2
h3
...
hT
```

So capacity increased, but:

> the communication pattern did not fundamentally change.

---

# ⭐ Task 33 — Derive the Next Architectural Improvement

If the decoder struggles because it receives only:

`c`

what should we expose instead?

### ✅ Answer

Retain:

```text
h1, h2, ..., hT
```

and let each decoder step determine:

> which source states matter right now.

This gives:

```text
Decoder step 1
→ maybe focus on h2

Decoder step 2
→ maybe focus on h5

Decoder step 3
→ maybe focus on h3
```

That leads to:

# **Attention**

---

# ✅ True / False

| #  | Statement                                                                 | Answer  |
| -- | ------------------------------------------------------------------------- | ------- |
| 1  | Seq2Seq maps one sequence to another.                                     | ✅ True  |
| 2  | Source and target lengths must match.                                     | ❌ False |
| 3  | Source and target vocabularies must match.                                | ❌ False |
| 4  | Token IDs are semantic vectors.                                           | ❌ False |
| 5  | The encoder commonly consumes embeddings.                                 | ✅ True  |
| 6  | `<SOS>` is used to start decoding.                                        | ✅ True  |
| 7  | `<EOS>` indicates the end of generation.                                  | ✅ True  |
| 8  | Decoder targets are typically shifted during training.                    | ✅ True  |
| 9  | Autoregressive decoder outputs are independent of previous target tokens. | ❌ False |
| 10 | Softmax converts logits into probabilities.                               | ✅ True  |
| 11 | Padding should normally contribute to target loss.                        | ❌ False |
| 12 | Decoder loss can train the encoder through backpropagation.               | ✅ True  |
| 13 | Greedy decoding guarantees the globally best sequence.                    | ❌ False |
| 14 | Training and inference decoder inputs are always identical.               | ❌ False |
| 15 | Classic fixed-context Seq2Seq can struggle with long source sequences.    | ✅ True  |

---

# ⭐ Staff Engineer Challenge — Production Translation Service

## Scenario

Your team deploys this model:

```text
Source Sentence
↓
BiLSTM Encoder
↓
512-D Context
↓
LSTM Decoder
↓
Translation
```

Observations:

```text
Short sentences
→ strong quality

Long legal paragraphs
→ names omitted
→ numbers changed
→ phrases repeated

Inference latency
→ also increases significantly with output length
```

---

## Question 1 — Why might names and numbers disappear from long inputs?

### ✅ Answer

The source paragraph must be compressed into:

> one fixed context representation.

Fine-grained information such as:

* names
* numbers
* dates
* specific clauses

may be difficult to preserve and recover.

This is the fixed-context bottleneck.

---

## Question 2 — Why does the BiLSTM encoder not remove that bottleneck?

### ✅ Answer

BiLSTM improves:

> how the source is encoded.

But if only the final combined representation reaches the decoder:

```text
Rich Encoder States
↓
One Context
↓
Decoder
```

the communication channel is still fixed.

---

## Question 3 — Why does decoder latency increase with output length?

### ✅ Answer

The recurrent decoder is autoregressive:

```text
y1 → y2 → y3 → ...
```

Each token depends on the previous decoder state/output.

So target positions cannot all be generated independently in parallel.

Longer outputs require:

> more sequential decoding steps.

---

## Question 4 — Should you report teacher-forced token accuracy as the only production metric?

### ✅ Answer

No.

Teacher forcing gives the decoder correct previous tokens.

Production inference uses:

> its own predictions.

Therefore evaluation should also include actual autoregressive decoding.

Useful metrics may include:

* sequence quality
* task-specific metric
* exact match where appropriate
* latency
* output length behavior
* repetition
* termination reliability

---

## Question 5 — Your output vocabulary increases from 20k to 100k. What component becomes significantly larger?

### ✅ Answer

The output projection:

`W_o ∈ R^(V_tgt × H)`

and output softmax computation.

If decoder hidden size is fixed, increasing vocabulary:

```text
20,000
→
100,000
```

makes the output layer approximately:

> 5× larger in vocabulary dimension.

---

## Question 6 — What immediate architecture improvement would you investigate for source-information loss?

### ✅ Answer

Allow the decoder to access:

```text
h1, h2, ..., hT
```

instead of depending solely on one context vector.

Then create:

> a decoder-step-specific context representation.

This is the architectural direction toward:

# **Attention**

---

## Question 7 — What is the key production lesson?

### ✅ Answer

Seq2Seq quality depends not only on recurrent-cell capacity but also on:

> how information flows from source to decoder.

A strong encoder cannot compensate indefinitely for:

> a restrictive encoder-decoder communication bottleneck.

---

# 🎯 Final Applied Exercise

Complete the full Seq2Seq training story:

```text
Source Tokens
↓
__________
↓
Encoder
↓
__________
↓
Initialize / Condition __________
↓
Target input starts with __________
↓
Decoder predicts __________
↓
Output projection creates __________
↓
Softmax creates __________
↓
Compare with shifted __________
↓
Compute __________
↓
Backpropagate through __________ and __________
```

### ✅ Answer

```text
Source Tokens
↓
SOURCE EMBEDDINGS
↓
Encoder
↓
CONTEXT
↓
Initialize / Condition DECODER
↓
Target input starts with <SOS>
↓
Decoder predicts NEXT TARGET TOKEN
↓
Output projection creates LOGITS
↓
Softmax creates TOKEN PROBABILITIES
↓
Compare with shifted TARGET SEQUENCE
↓
Compute CROSS-ENTROPY LOSS
↓
Backpropagate through DECODER and ENCODER
```

---

# 🧠 Final Assignment Mental Model

```text
SOURCE
  │
  ▼
Embeddings
  │
  ▼
Encoder
  │
  ▼
Context
  │
  ▼
Decoder
  │
  ▼
Logits
  │
  ▼
Softmax
  │
  ▼
Next Token
  │
  └───────────────┐
                  │
                  ▼
          Feed Back Next Step
```

Training:

```text
<SOS> y1 y2
↓
predict
y1    y2 <EOS>
```

Inference:

```text
<SOS>
↓
model prediction
↓
feed prediction back
↓
next prediction
```

And the architectural pressure:

```text
Long Source
↓
One Fixed Context
↓
Information Loss
↓
Need Direct Access to Encoder States
↓
Attention
```

---

# ⭐ Golden Rule

> **To understand Seq2Seq, trace the complete chain: embed the source, encode it, transfer source information to the decoder, shift the target for next-token training, generate autoregressively at inference, and always ask whether one fixed context vector can really preserve everything the decoder needs.**
