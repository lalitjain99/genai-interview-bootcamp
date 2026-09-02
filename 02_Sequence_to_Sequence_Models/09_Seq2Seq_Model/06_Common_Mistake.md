# 🚫 Common Mistakes — Sequence-to-Sequence (Seq2Seq) Model

> **Goal:** Avoid the most common conceptual, training, inference, dimensional, and architectural mistakes in classic Seq2Seq models.

---

## ❌ Mistake 1: Thinking Seq2Seq Requires Equal Input and Output Lengths

### Wrong

`T_src = T_tgt`

must always hold.

### Correct

Seq2Seq supports:

# `T_src ≠ T_tgt`

Example:

```text id="seq-mistake-1"
Source:
x1 x2 x3 x4 x5

Target:
y1 y2 y3
```

This flexibility is one of the main reasons Seq2Seq is useful.

---

## ❌ Mistake 2: Thinking Token IDs Are Semantic Vectors

Example:

```text id="seq-mistake-2"
"I"    → 12
"love" → 47
"AI"   → 205
```

These values are only identifiers.

They should typically go through:

```text id="seq-mistake-3"
Token ID
↓
Embedding Lookup
↓
Dense Vector
```

Do not interpret token IDs numerically.

---

## ❌ Mistake 3: Thinking Source and Target Must Share the Same Vocabulary

Not necessarily.

In translation:

```text id="seq-mistake-4"
English vocabulary
→ Encoder

French vocabulary
→ Decoder
```

So:

`V_src ≠ V_tgt`

is completely normal.

The model may also have separate:

* source embedding matrix
* target embedding matrix

---

## ❌ Mistake 4: Thinking the Encoder Generates Target Tokens

The encoder's role is:

> represent the source sequence.

The decoder's role is:

> generate the target sequence.

Mental model:

```text id="seq-mistake-5"
Encoder
→ What did I read?

Decoder
→ What should I generate?
```

---

## ❌ Mistake 5: Thinking the Context Vector Stores the Source Exactly

In classic Seq2Seq:

`c = h_T`

The context vector is:

> a learned compressed representation.

It is not:

> an exact copy of all source tokens.

This distinction becomes especially important for long sequences.

---

## ❌ Mistake 6: Thinking the Decoder Directly Receives Every Source Token in Classic Seq2Seq

In the basic fixed-context model:

```text id="seq-mistake-6"
Source
↓
Encoder
↓
Context
↓
Decoder
```

The decoder primarily depends on the encoded context.

Direct dynamic access to all encoder states appears later with:

> attention.

---

## ❌ Mistake 7: Thinking `<SOS>` Is Part of the Actual Target Text

`<SOS>` / `<BOS>` is a special control token used to:

> start decoder generation.

It is not normally part of the user-visible target text.

---

## ❌ Mistake 8: Confusing `<EOS>` With Padding

`<EOS>` means:

> the output sequence has ended.

`PAD` means:

> this batch position is artificial.

They serve different purposes.

```text id="seq-mistake-7"
<EOS>
→ semantic termination

PAD
→ batching convenience
```

---

## ❌ Mistake 9: Forgetting to Shift Targets During Training

Suppose the target is:

```text id="seq-mistake-8"
A B C <EOS>
```

Correct decoder input:

```text id="seq-mistake-9"
<SOS> A B C
```

Expected output:

```text id="seq-mistake-10"
A B C <EOS>
```

This shift teaches:

> previous tokens → next token.

---

## ❌ Mistake 10: Thinking Target Shifting Means Dropping the First Target Token

No.

The target is not simply shortened.

Instead:

```text id="seq-mistake-11"
Decoder Input:
<SOS> y1 y2

Expected Output:
y1    y2 <EOS>
```

The two sequences are aligned one step apart.

---

## ❌ Mistake 11: Thinking the Decoder Hidden State Is Already a Token Probability Distribution

The decoder state:

`d_t`

is a hidden representation.

Usually:

```text id="seq-mistake-12"
d_t
↓
Linear Projection
↓
Logits
↓
Softmax
↓
Token Probabilities
```

So:

`d_t ≠ P(y_t)`

---

## ❌ Mistake 12: Confusing Logits With Probabilities

Logits:

`z_t = W_o d_t + b_o`

can contain arbitrary real values.

They are not probabilities.

Only after softmax:

`softmax(z_t)`

do we obtain a normalized distribution.

---

## ❌ Mistake 13: Thinking Training and Inference Use the Same Previous Token

During training, teacher forcing may provide:

> the true previous target token.

During inference:

> the true target is unavailable.

So the decoder must use its own previous prediction.

```text id="seq-mistake-13"
Training
→ Ground Truth History

Inference
→ Model-Generated History
```

---

## ❌ Mistake 14: Ignoring Error Propagation During Inference

If the model makes an early mistake:

```text id="seq-mistake-14"
Wrong y_t
↓
used in future decoder history
↓
different state
↓
possible additional errors
```

So autoregressive inference can accumulate errors over time.

---

## ❌ Mistake 15: Thinking Greedy Decoding Finds the Globally Best Sequence

Greedy decoding chooses:

> the highest-probability token at the current step.

But:

```text id="seq-mistake-15"
Best local token
≠
necessarily best full sequence
```

A slightly lower-probability token now may lead to a much better continuation.

This is why beam search exists.

---

## ❌ Mistake 16: Including `PAD` Tokens in the Target Loss

Wrong:

```text id="seq-mistake-16"
A B <EOS> PAD PAD
           ↑   ↑
        calculate loss
```

Correct:

```text id="seq-mistake-17"
A      → include
B      → include
<EOS>  → include
PAD    → ignore
PAD    → ignore
```

Use masking so artificial padding positions do not distort training.

---

## ❌ Mistake 17: Thinking Only the Decoder Learns Because Loss Is Computed There

Loss is usually calculated at decoder outputs.

But gradients flow:

```text id="seq-mistake-18"
Loss
↓
Decoder
↓
Context / State Transfer
↓
Encoder
```

So Seq2Seq is trained:

> end-to-end.

---

## ❌ Mistake 18: Thinking Seq2Seq Does Not Need BPTT

If encoder and decoder are recurrent, both require backpropagation through time.

```text id="seq-mistake-19"
Encoder
→ BPTT through source

Decoder
→ BPTT through target
```

Bidirectional or gated variants do not remove this requirement.

---

## ❌ Mistake 19: Treating LSTM and GRU State Transfer as Identical

LSTM may transfer:

```text id="seq-mistake-20"
h_T
+
c_T
```

GRU transfers:

```text id="seq-mistake-21"
h_T
```

because GRU has no separate cell state.

Do not invent:

`c_T`

for a GRU.

---

## ❌ Mistake 20: Assuming Encoder and Decoder Hidden Dimensions Must Match

They do not have to.

If:

```text id="seq-mistake-22"
Encoder context = 512
Decoder hidden = 256
```

use a learned projection:

`d_0 = W_c c + b_c`

So dimension mismatch is:

> an interface problem, not an invalid architecture.

---

## ❌ Mistake 21: Thinking a Bidirectional Encoder Means the Decoder Should Also Be Bidirectional

No.

The source sequence may already be fully available.

So:

> bidirectional encoder can be valid.

But during autoregressive generation, future target tokens do not exist.

Therefore the decoder should normally remain:

> causal.

---

## ❌ Mistake 22: Thinking a Bidirectional Encoder Removes the Fixed-Context Bottleneck

A BiLSTM/BiGRU can produce a richer source representation.

But if everything still becomes:

```text id="seq-mistake-23"
Entire Source
↓
One Context Vector
↓
Decoder
```

the bottleneck remains.

Bidirectionality and source-access bottleneck are:

> different problems.

---

## ❌ Mistake 23: Thinking a Larger Context Vector Completely Solves Long-Sequence Problems

Increasing:

```text id="seq-mistake-24"
256 → 1024 → 4096
```

may help capacity.

But the architecture still requires:

> one fixed representation to carry everything.

The decoder still cannot directly request:

```text id="seq-mistake-25"
h_5
h_20
h_97
```

when those states become relevant.

---

## ❌ Mistake 24: Thinking Classic Seq2Seq Already Has Attention

No.

Classic Seq2Seq:

```text id="seq-mistake-26"
Encoder
↓
Fixed Context
↓
Decoder
```

Attention changes this by allowing the decoder to dynamically use:

```text id="seq-mistake-27"
h1, h2, ..., hT
```

Classic Seq2Seq is precisely the architecture whose limitations motivate attention.

---

## ❌ Mistake 25: Thinking `P(Y|X)` Means Output Tokens Are Independent

Seq2Seq factorizes:

`P(Y|X) = Π_t P(y_t | y_<t, X)`

Notice:

`y_<t`

Each output token depends on:

> previous target tokens.

So target tokens are not independently generated.

---

## ❌ Mistake 26: Multiplying Token Probabilities Without Considering Numerical Stability

Full sequence probability is a product:

`Π_t P(y_t | ...)`

For long sequences, this can become extremely small.

Training commonly works with:

> log probabilities

because:

`log(ab) = log(a) + log(b)`

So negative log-likelihood becomes a sum over steps.

---

# ⚡ Quick Trap Table

| Wrong Idea                                | Correct Idea                         |
| ----------------------------------------- | ------------------------------------ |
| Source/target length must match           | They can differ                      |
| Token IDs are embeddings                  | IDs need embedding lookup            |
| Same source/target vocabulary required    | They can differ                      |
| Encoder generates target                  | Decoder generates target             |
| Context stores source exactly             | It is compressed representation      |
| Decoder directly reads every source token | Not in classic fixed-context Seq2Seq |
| `<EOS>` = padding                         | `<EOS>` terminates generation        |
| Target does not need shifting             | Shift for next-token prediction      |
| Decoder state = probabilities             | Projection + softmax needed          |
| Logits = probabilities                    | Softmax converts logits              |
| Training = inference                      | Previous-token source differs        |
| Greedy = globally optimal                 | Only locally optimal                 |
| PAD contributes to loss                   | Mask it                              |
| Only decoder learns                       | Encoder learns through gradient flow |
| Seq2Seq does not use BPTT                 | Recurrent versions do                |
| GRU has an LSTM cell state                | No                                   |
| Hidden dimensions must match              | Projection can connect them          |
| Bidirectional decoder is always fine      | Not for causal decoding              |
| BiEncoder fixes context bottleneck        | No                                   |
| Bigger context fully fixes bottleneck     | No                                   |
| Classic Seq2Seq includes attention        | No                                   |

---

# 🧠 Debugging Checklist

When reviewing a Seq2Seq implementation, check:

```text id="seq-mistake-28"
1. Are source IDs embedded before encoding?
2. Are source and target vocabularies handled separately?
3. What exactly becomes the source context?
4. How is decoder initial state created?
5. Are dimensions compatible or projected?
6. Does decoding start with <SOS>/<BOS>?
7. Does generation stop on <EOS>?
8. Is the target shifted correctly?
9. Are target IDs embedded before decoder input?
10. Are decoder states projected to vocabulary logits?
11. Is softmax/cross-entropy applied correctly?
12. Are PAD positions masked?
13. Is teacher-forcing behavior distinguished from inference?
14. Does inference feed predictions back correctly?
15. Can errors propagate autoregressively?
16. Is greedy decoding being confused with global sequence search?
17. Are encoder gradients flowing through the decoder/context path?
18. Are LSTM vs GRU states handled correctly?
19. Is a bidirectional encoder paired with a causal decoder when appropriate?
20. Is long-sequence failure being traced to the fixed-context bottleneck?
```

---

# 📐 Formula Sanity Check

### Encoder

`h_t = f_enc(x_t, h_(t-1))`

### Classic Context

# `c = h_T`

### Decoder

`d_t = f_dec(e(y_(t-1)), d_(t-1), c)`

### Output Logits

`z_t = W_o d_t + b_o`

### Probability

# `P(y_t | y_<t, X) = softmax(z_t)`

### Sequence Probability

# `P(Y|X) = Π_t P(y_t | y_<t, X)`

### Token Loss

`L_t = -log P(y_t^correct)`

### Sequence Loss

# `L = Σ_t L_t`

with padding positions ignored.

---

# 🧠 Final Mental Model

Keep these stages separate:

```text id="seq-mistake-29"
SOURCE
↓
Embed
↓
Encode
↓
Context
```

```text id="seq-mistake-30"
TARGET HISTORY
↓
Embed
↓
Decode
↓
Logits
↓
Probabilities
↓
Next Token
```

Training:

```text id="seq-mistake-31"
<SOS> y1 y2
↓
predict
y1    y2 <EOS>
```

Inference:

```text id="seq-mistake-32"
<SOS>
↓
prediction
↓
feed prediction back
↓
next prediction
```

And never forget the architectural limitation:

```text id="seq-mistake-33"
Long Source
↓
One Fixed Context
↓
Decoder Must Recover Everything
↓
Information Bottleneck
↓
Need Dynamic Encoder-State Access
↓
Attention
```

---

# ⭐ Golden Rule

> **The biggest classic Seq2Seq mistake is thinking that a powerful encoder makes one fixed context vector sufficient: the encoder may learn rich source information, but if the decoder can only receive that information through one compressed channel, long-sequence information retrieval remains fundamentally difficult.**
