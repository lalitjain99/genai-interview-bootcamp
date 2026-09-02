# 🎤 Interview Questions — Encoder–Decoder Architecture

> **Goal:** Test whether you truly understand why Encoder–Decoder architecture was needed, how source information flows into the decoder, how autoregressive generation works, and why the original fixed-context design eventually led to attention.

---

# 🟢 Beginner

## Q1. What is an Encoder–Decoder architecture?

### ✅ Answer

An Encoder–Decoder architecture separates a sequence-to-sequence problem into two parts:

```text id="ed-int-1"
Input Sequence
↓
Encoder
↓
Representation / Context
↓
Decoder
↓
Output Sequence
```

The encoder reads and represents the source sequence.

The decoder uses that representation to generate the target sequence.

It is commonly used for tasks like:

* machine translation
* summarization
* speech-to-text
* sequence transformation

---

## Q2. Why do we need separate encoder and decoder components?

### ✅ Answer

Because input understanding and output generation are different jobs.

The encoder answers:

> What information is present in the source sequence?

The decoder answers:

> Given that information and what I have generated so far, what should I generate next?

So:

```text id="ed-int-2"
Encoder
→ Understand / represent

Decoder
→ Generate
```

This separation naturally supports:

> sequence → sequence

problems.

---

## Q3. What is the context vector in the original Encoder–Decoder architecture?

### ✅ Answer

In the original recurrent Encoder–Decoder architecture, the final encoder hidden state is often used as the context vector:

`c = h_T`

It is a:

> fixed-dimensional learned representation of the complete source sequence.

It should not be thought of as an exact copy of every input token.

---

## Q4. Can input and output sequence lengths be different?

### ✅ Answer

Yes.

For example:

```text id="ed-int-3"
Input:
x1 x2 x3 x4 x5

Output:
y1 y2 y3
```

Encoder–Decoder does not require:

`T_input = T_output`

The decoder can continue generating until it produces:

`<EOS>`

So variable-length mappings are naturally supported.

---

# 🟡 Intermediate

## Q5. Write the basic encoder recurrence.

### ✅ Answer

For a recurrent encoder:

`h_t = f_enc(x_t, h_(t-1))`

where:

* `x_t` = current source input
* `h_(t-1)` = previous encoder state
* `h_t` = current encoder state

In the original fixed-context architecture:

`c = h_T`

where `T` is the final source position.

---

## Q6. How does decoding begin?

### ✅ Answer

The decoder typically receives:

> a special start token

such as:

`<SOS>` or `<BOS>`

and an initial decoder state derived from the encoder representation.

For example:

`d_0 = h_T`

or:

`d_0 = W_c h_T + b_c`

Then:

```text id="ed-int-4"
<SOS>
↓
Decoder
↓
First Target Token
```

---

## Q7. How does the decoder know when to stop generating?

### ✅ Answer

The decoder learns to produce:

`<EOS>`

meaning:

> End Of Sequence.

Generation continues until:

* `<EOS>` is generated, or
* some external maximum-length condition is reached.

This enables variable-length output.

---

## Q8. How is a decoder hidden state converted into a token prediction?

### ✅ Answer

Suppose decoder state is:

`d_t`

First compute logits:

`z_t = W_o d_t + b_o`

Then apply softmax:

`P(y_t | ...) = softmax(z_t)`

So:

```text id="ed-int-5"
Decoder State
↓
Linear Projection
↓
Logits
↓
Softmax
↓
Vocabulary Distribution
```

The next token is then selected using a decoding strategy such as:

* greedy decoding
* beam search

---

## Q9. What does autoregressive decoding mean?

### ✅ Answer

Autoregressive decoding means:

> each output token is generated conditioned on previous output tokens.

Conceptually:

```text id="ed-int-6"
<SOS>
↓
y1
↓
y2
↓
y3
↓
...
```

The sequence probability is factorized as:

`P(Y|X) = Π_t P(y_t | y_<t, X)`

So later output tokens depend on:

> the previously generated target sequence.

---

## Q10. Can the encoder and decoder use different recurrent cell types or hidden sizes?

### ✅ Answer

Yes.

For example:

```text id="ed-int-7"
Encoder
→ BiLSTM, hidden size 512

Decoder
→ GRU, hidden size 256
```

If their state dimensions differ, a learned transformation can connect them:

`d_0 = W_c h_T + b_c`

So the architecture does not require:

* same cell type
* same hidden dimension

as long as the interface is handled correctly.

---

# 🔴 Advanced

## Q11. What is the difference between Encoder–Decoder and a simple many-to-many RNN?

### ✅ Answer

A simple aligned many-to-many RNN may produce:

```text id="ed-int-8"
x1 → y1
x2 → y2
x3 → y3
```

where input and output positions are closely aligned.

Example:

> POS tagging.

Encoder–Decoder instead separates source processing from target generation:

```text id="ed-int-9"
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

This allows:

* different input/output lengths
* no strict position-to-position correspondence
* general sequence transformation

---

## Q12. Why can an encoder often be bidirectional while an autoregressive decoder usually cannot?

### ✅ Answer

The complete source sequence is often available before decoding starts.

Therefore the encoder can legitimately use:

```text id="ed-int-10"
past + future source context
```

using BiLSTM or BiGRU.

But decoder output tokens are generated one at a time.

When predicting:

`y_t`

future target tokens:

`y_(t+1), y_(t+2), ...`

do not yet exist.

Using them would cause:

> target leakage.

So:

```text id="ed-int-11"
Encoder
→ can often be bidirectional

Autoregressive Decoder
→ normally causal
```

---

## Q13. What is the difference between encoder hidden state and decoder hidden state?

### ✅ Answer

The encoder hidden state primarily represents:

> source sequence information.

The decoder hidden state primarily represents:

> current target-generation state conditioned on the source.

A useful mental model is:

```text id="ed-int-12"
Encoder State
→ What did I read?

Decoder State
→ Given what I read and what I generated so far, what should happen next?
```

They are related but have different roles.

---

## Q14. What is the fixed-context-vector bottleneck?

### ✅ Answer

In the original recurrent Encoder–Decoder architecture:

> the entire input sequence is compressed into one fixed-size vector.

For example:

```text id="ed-int-13"
5-token input
↓
c

50-token input
↓
c

500-token input
↓
c
```

The context-vector dimensionality remains fixed.

As the input becomes longer or more information-rich, the model may struggle to preserve every useful detail in that one vector.

This becomes an:

# **information bottleneck**

---

## Q15. Why did the fixed-context bottleneck motivate attention?

### ✅ Answer

The problem is that the decoder receives one compressed representation rather than directly accessing the individual encoder states.

Instead of forcing:

```text id="ed-int-14"
h1 h2 h3 ... hT
            ↓
            c
            ↓
         Decoder
```

we would like the decoder to access:

```text id="ed-int-15"
h1
h2
h3
...
hT
```

and determine:

> which source positions are most relevant at each output step.

That is the key idea that eventually becomes:

# **Attention**

---

# ⭐ Staff Engineer Challenge

## Scenario

You are building a neural translation service.

Your first design uses:

```text id="ed-int-16"
BiLSTM Encoder
↓
Final Encoder Representation
↓
LSTM Decoder
↓
Translated Sentence
```

The model performs well on short sentences but translation quality falls sharply when input sentences become long.

---

## Question 1. What is the first architectural issue you would suspect?

### ✅ Answer

The likely issue is the:

# **fixed context vector bottleneck**

The complete source sentence must be compressed into a limited-size representation before the decoder generates any output.

Longer sentences contain more:

* entities
* relationships
* clauses
* semantic details

so compressing everything into one vector becomes increasingly difficult.

---

## Question 2. Would simply making the context vector much larger completely solve the problem?

### ✅ Answer

Not necessarily.

Increasing hidden size may increase capacity, but the architecture still requires:

> all source information to pass through one fixed representation.

The decoder still cannot directly revisit individual source positions.

So a larger vector may help, but it does not remove the structural bottleneck.

---

## Question 3. Why is using a BiLSTM encoder valid here?

### ✅ Answer

Because the entire source sentence is already available before translation begins.

Therefore the encoder can use:

> both left and right source context.

There is no source-side future leakage in this setup.

---

## Question 4. Why would a bidirectional decoder be problematic?

### ✅ Answer

At inference time, target tokens are generated sequentially.

When generating:

`y_t`

future target tokens:

`y_(t+1), y_(t+2), ...`

do not yet exist.

A bidirectional decoder would require those future target tokens, causing:

> target-side leakage.

So normal autoregressive decoding must remain causal.

---

## Question 5. The encoder hidden size is 512 but the decoder hidden size is 256. How can you initialize the decoder?

### ✅ Answer

Use a learned projection:

`d_0 = W_c h_T + b_c`

If:

`h_T ∈ R^512`

and:

`d_0 ∈ R^256`

then:

`W_c ∈ R^(256×512)`

and:

`b_c ∈ R^256`

This maps encoder representation into the decoder-state space.

---

## Question 6. If you use an LSTM encoder and an LSTM decoder, what states may need to be transferred?

### ✅ Answer

LSTM has both:

* hidden state `h`
* cell state `c`

So the decoder may be initialized using:

`h_0^dec = h_T^enc`

`c_0^dec = c_T^enc`

or learned transformations of them.

This differs from GRU, which only has:

> one recurrent hidden state.

---

## Question 7. How would you evolve this architecture to improve long-sentence translation?

### ✅ Answer

The next major architectural improvement is to avoid forcing the decoder to depend only on one final source vector.

Instead, expose:

```text id="ed-int-17"
h1, h2, ..., hT
```

to the decoder and let it dynamically determine:

> which encoder positions matter for each generated token.

That leads naturally to:

# **Attention**

---

# ⚡ Rapid-Fire Traps

### ❌ “Encoder–Decoder requires equal input and output lengths.”

False.

---

### ❌ “The context vector stores the original sentence exactly.”

False.

It is a learned compressed representation.

---

### ❌ “The encoder generates the output tokens.”

No.

The decoder generates them.

---

### ❌ “The decoder only needs the encoder context and not its own history.”

False.

Autoregressive decoding depends on previous target outputs/states.

---

### ❌ “The decoder hidden state is already a token probability distribution.”

No.

It must typically pass through:

```text id="ed-int-18"
Linear Projection
↓
Softmax
```

---

### ❌ “Encoder and decoder must have equal hidden sizes.”

False.

Use a projection if needed.

---

### ❌ “An autoregressive decoder can safely be bidirectional.”

False.

Future target tokens are unavailable.

---

### ❌ “A larger context vector completely removes the bottleneck.”

Not necessarily.

The structural compression remains.

---

### ❌ “Encoder–Decoder and attention are the same architecture.”

No.

Attention is a later improvement to how decoder accesses encoder information.

---

### ❌ “`<EOS>` is just padding.”

No.

It signals that generation should stop.

---

# 📐 Formula Flash Card

### Encoder

`h_t = f_enc(x_t, h_(t-1))`

### Original Context

# `c = h_T`

### Decoder

`d_t = f_dec(y_(t-1), d_(t-1), c)`

### Decoder Initialization Example

`d_0 = h_T`

or:

`d_0 = W_c h_T + b_c`

### Output Logits

`z_t = W_o d_t + b_o`

### Token Probability

`P(y_t | y_<t, c) = softmax(z_t)`

### Sequence Probability

# `P(Y|X) = Π_t P(y_t | y_<t, X)`

---

# 🎤 30-Second Interview Answer

> **Encoder–Decoder is an architecture for mapping one variable-length sequence to another. The encoder processes the source sequence and converts it into a learned representation, traditionally the final recurrent hidden state. The decoder is conditioned on that representation and generates the target sequence autoregressively, one token at a time, until it produces an end-of-sequence token. The key limitation of the original design is that the entire source sequence must be compressed into one fixed-dimensional context vector, which becomes a bottleneck for long inputs and motivates attention.**

---

# 🧠 Final Interview Mental Model

```text id="ed-int-19"
SOURCE
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
  ├──→ y1
  │
  ├──→ y2
  │
  ├──→ y3
  │
  └──→ <EOS>
```

Keep the responsibilities separate:

```text id="ed-int-20"
Encoder
→ What did I read?

Context
→ What representation did I create?

Decoder
→ What should I generate next?
```

And remember the architectural pressure:

```text id="ed-int-21"
Whole Source
↓
One Fixed Vector
↓
Information Bottleneck
↓
Need Direct Access to Encoder States
↓
Attention
```

---

# ⭐ Golden Rule

> **Encoder–Decoder separates source representation from target generation, but the original recurrent design forces the full source sequence through a single fixed-dimensional context representation—an elegant solution that eventually becomes its biggest limitation.**
