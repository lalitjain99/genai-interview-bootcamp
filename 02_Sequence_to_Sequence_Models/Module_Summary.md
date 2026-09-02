# ⭐ Module Summary — Sequence-to-Sequence Models

> **Module Goal:** Build the complete conceptual bridge from basic sequence modeling to encoder-decoder systems, Seq2Seq training and inference, search, the fixed-context bottleneck, and finally the motivation for attention.

---

# 🌟 1. The Big Story of This Module

This module was not a collection of disconnected topics.

Each concept appeared because the previous one had a limitation.

The progression was:

```text
Sequential Data
↓
Need Memory
↓
RNN
↓
Long-Term Dependency Problem
↓
Vanishing / Exploding Gradients
↓
LSTM / GRU
↓
Need Better Context in Both Directions
↓
Bidirectional RNN
↓
Need Sequence-to-Sequence Mapping
↓
Encoder–Decoder
↓
Seq2Seq Model
↓
Fixed Context Bottleneck
↓
Teacher Forcing
↓
Inference Has No Teacher
↓
Autoregressive Decoding
↓
Greedy Search Is Myopic
↓
Beam Search
↓
Target Search Improved
↓
Source Access Still Fixed
↓
Attention Motivation
```

This entire module can be understood as one evolving question:

# **How can a neural network read a variable-length sequence, preserve the right information, and generate another variable-length sequence reliably?**

---

# 🧭 2. Module Map

The module covered:

1. `01_Why_Sequence_Models`
2. `02_Recurrent_Neural_Network`
3. `03_Backpropagation_Through_Time`
4. `04_Vanishing_And_Exploding_Gradients`
5. `05_LSTM`
6. `06_GRU`
7. `07_Bidirectional_RNN`
8. `08_Encoder_Decoder`
9. `09_Seq2Seq_Model`
10. `10_Context_Vector_Bottleneck`
11. `11_Teacher_Forcing`
12. `12_Inference_And_Autoregressive_Decoding`
13. `13_Beam_Search`
14. `14_Attention_Motivation`

The next major module naturally begins with:

# **Attention Mechanism**

---

# 📖 3. Why Sequence Models Were Needed

A standard feedforward network treats inputs as:

> independent fixed-size vectors.

But many real-world problems contain:

* order
* temporal dependence
* context
* variable-length input

Examples:

```text
"I love AI"
```

is different from:

```text
"AI love I"
```

even if the same words appear.

So we needed a mechanism that could preserve:

> information from previous positions.

That led to:

# **Recurrent Neural Networks**

---

# 🧠 4. RNN — The First Memory Mechanism

A vanilla RNN introduces a hidden state:

# `h_t`

that carries information from previous timesteps.

Core equations:

# `a_t = W_xh x_t + W_hh h_(t-1) + b_h`

# `h_t = tanh(a_t)`

The hidden state combines:

```text
Current Input
+
Previous Memory
↓
New Memory
```

This gives the model:

> sequential state.

---

# 📐 5. RNN Parameter Sharing

The same parameters are reused at every timestep:

* `W_xh`
* `W_hh`
* `b_h`

So:

```text
x1 → RNN
x2 → same RNN
x3 → same RNN
```

This gives:

* parameter efficiency
* sequence-length independence of parameter count
* ability to process variable-length sequences

But compute still grows with sequence length.

---

# 🧠 6. Hidden State Is Not Exact Memory

The hidden state is not:

> a literal storage of every previous token.

It is:

> a learned compressed representation of relevant history.

That compression becomes important later when discussing:

* long-term dependency
* encoder context
* attention

---

# 🔄 7. How RNNs Learn — BPTT

An RNN unrolled through time behaves like a deep computational graph.

Training uses:

# **Backpropagation Through Time**

The gradient flows backward through:

```text
h_T
↓
h_(T-1)
↓
h_(T-2)
↓
...
```

A long-range dependency contains repeated Jacobian products:

# `∂h_T/∂h_k = Π_t ∂h_t/∂h_(t-1)`

This creates the next major problem.

---

# 🚨 8. Vanishing and Exploding Gradients

Repeated multiplication can cause gradients to:

### Vanish

```text
0.5 × 0.5 × 0.5 × ...
→ 0
```

### Explode

```text
2 × 2 × 2 × ...
→ huge
```

So vanilla RNNs may struggle to learn:

> long-range dependencies.

---

# 📐 9. Why `tanh` Can Contribute to Vanishing

For:

# `h_t = tanh(a_t)`

the derivative satisfies:

# `|tanh'(a_t)| ≤ 1`

Repeated products involving these derivatives can shrink gradients significantly.

So information from very distant timesteps becomes:

> difficult to credit correctly.

---

# ⚡ 10. Exploding Gradient Mitigation

A common mitigation is:

# **Gradient Clipping**

For example, clip by global norm.

Important:

```text
Gradient Clipping
→ handles exploding gradients
```

It does not fundamentally solve:

```text
vanishing gradients
```

---

# 🧠 11. LSTM — Create a Better Memory Path

LSTM introduces a dedicated:

# **cell state `c_t`**

plus gates that control:

* what to forget
* what to write
* what to expose

Core equations:

# `f_t = σ(...)`

# `i_t = σ(...)`

# `g_t = tanh(...)`

# `c_t = f_t ⊙ c_(t-1) + i_t ⊙ g_t`

# `o_t = σ(...)`

# `h_t = o_t ⊙ tanh(c_t)`

---

# ⭐ 12. LSTM Mental Model

Think:

```text
Cell State
=
Notebook
```

Then:

### Forget Gate

> What old information should remain?

### Input Gate

> How much new information should be written?

### Candidate

> What new content could be written?

### Output Gate

> What part of memory should be exposed now?

The key equation:

# `c_t = KEEP + WRITE`

where:

```text
KEEP = f_t ⊙ c_(t-1)
WRITE = i_t ⊙ g_t
```

---

# 🧠 13. Why LSTM Helps Gradient Flow

The cell-state update creates a more direct additive path:

# `c_t = f_t ⊙ c_(t-1) + ...`

So gradients can propagate through the cell state more easily than through repeated nonlinear transformation alone.

LSTM:

> mitigates

but does not magically eliminate:

> all long-term dependency problems.

---

# ⚡ 14. GRU — Simplify the Gating System

GRU removes the separate cell state.

It uses:

* update gate
* reset gate
* candidate hidden state

One common convention:

# `h_t = z_t ⊙ h_(t-1) + (1-z_t) ⊙ h̃_t`

Here:

```text
z_t ≈ 1
→ preserve old state

z_t ≈ 0
→ use new candidate
```

Be careful:

> some sources use the opposite update-gate convention.

Always inspect the equation.

---

# 🧠 15. Reset vs Update Gate

### Reset Gate

Controls how much previous context influences:

> candidate construction.

### Update Gate

Controls how much old vs new information enters:

> final hidden state.

So:

```text
Reset
→ build candidate

Update
→ blend old and new
```

---

# ⚖️ 16. LSTM vs GRU

### LSTM

* cell state + hidden state
* 4 major transformations
* more expressive gating
* more parameters

### GRU

* one main hidden state
* 3 major transformations
* fewer parameters
* simpler architecture

Neither is universally superior.

The right choice depends on:

* data
* model size
* latency
* optimization
* empirical performance

---

# 🔄 17. Bidirectional RNN

A standard RNN at position `t` sees only:

> past context.

But some tasks can use:

> past + future context.

Bidirectional RNN runs:

```text
Forward:
x1 → x2 → ... → xT

Backward:
xT → ... → x2 → x1
```

Then combines both representations.

Often:

# `[→h_t ; ←h_t]`

---

# 🚨 18. Bidirectional Does Not Mean BPTT

These are completely different.

### Bidirectional RNN

> architecture with forward and backward sequence processing.

### BPTT

> training algorithm for recurrent computation.

Do not confuse them.

---

# 🧠 19. When Bidirectional Models Are Appropriate

Suitable when:

> the full input sequence is available before prediction.

Examples:

* classification
* tagging
* source encoding for translation

Not suitable for strict causal prediction where:

> future tokens do not yet exist.

---

# 🌉 20. Why Encoder–Decoder Was Needed

RNN-based models still had a structural limitation:

> many tasks map one sequence to another sequence of different length.

Examples:

```text
English
→ French

Document
→ Summary

Audio
→ Text
```

A single fixed-output classifier is not enough.

So we split the problem:

```text
Encoder
→ understand source

Decoder
→ generate target
```

---

# 🧠 21. Encoder–Decoder Architecture

Classic flow:

```text
Source
↓
Encoder
↓
Context / Representation
↓
Decoder
↓
Target
```

Encoder:

# `h_t = f_enc(x_t, h_(t-1))`

Classic fixed context:

# `c = h_T`

Decoder:

# `d_t = f_dec(y_(t-1), d_(t-1), c)`

---

# 📐 22. Decoder Output

Decoder hidden state:

`d_t`

is projected:

# `z_t = W_o d_t + b_o`

Then:

# `P(y_t | y_<t, X) = softmax(z_t)`

The model therefore predicts:

> one target-token distribution at each decoder step.

---

# ⭐ 23. Encoder–Decoder vs Seq2Seq

### Encoder–Decoder

> architectural pattern.

### Seq2Seq

> complete sequence-to-sequence conditional modeling problem/system.

In practice, the terms are often used interchangeably, but conceptually:

```text
Encoder–Decoder
→ components

Seq2Seq
→ full sequence mapping pipeline
```

---

# 📖 24. Full Seq2Seq Pipeline

```text
Source Tokens
↓
Source Embeddings
↓
Encoder
↓
Source Representation
↓
Decoder
↓
Decoder Hidden States
↓
Output Projection
↓
Vocabulary Logits
↓
Softmax
↓
Target Tokens
```

---

# 📐 25. Seq2Seq Probability

For:

`Y = (y_1, ..., y_T')`

the model factorizes:

# `P(Y|X) = Π_t P(y_t | y_<t, X)`

This equation is foundational.

It later appears again in:

* autoregressive decoding
* Beam Search
* language modeling
* LLM training

---

# 🔄 26. Target Shifting

For target:

```text
A B C <EOS>
```

decoder training input becomes:

```text
<SOS> A B C
```

labels:

```text
A B C <EOS>
```

So:

```text
Previous Target
→ Predict Next Target
```

---

# 🚨 27. The Fixed Context Bottleneck

Classic Seq2Seq may reduce:

```text
h1, h2, ..., hT
```

to:

# `c = h_T`

Then every decoder step relies on:

> that same source representation.

As source length grows:

```text
Information ↑
while
Interface stays fixed
```

This creates:

# **Context Vector Bottleneck**

---

# 🧠 28. Capacity vs Accessibility

A crucial distinction:

### Capacity

How much information can the context potentially encode?

### Accessibility

Can the decoder retrieve:

> the exact source detail needed right now?

Increasing hidden dimension improves capacity.

It does not fundamentally solve:

> step-specific source accessibility.

---

# 🚫 29. Context Bottleneck ≠ Vanishing Gradient

### Vanishing Gradient

> training / credit-assignment problem.

### Context Bottleneck

> source-representation / communication problem.

They can interact.

They are not the same.

---

# 🎓 30. Teacher Forcing

Training an autoregressive decoder creates a problem.

If its early prediction is wrong:

```text
wrong token
↓
fed back
↓
different state
↓
possibly more errors
```

Teacher Forcing avoids this during training by feeding:

> the correct previous target token.

---

# 📐 31. Teacher Forcing Definition

At decoder timestep `t`:

> feed ground-truth `y_(t-1)`

to predict:

> `y_t`.

So:

```text
Teacher Forcing
≠ give current answer

Teacher Forcing
= give correct previous history
```

---

# 🧠 32. Why Teacher Forcing Helps

Without teacher forcing early in training:

```text
weak prediction
↓
bad history
↓
bad state
↓
harder learning
```

With teacher forcing:

```text
correct history
↓
cleaner state
↓
next-token learning
```

So training becomes:

* more stable
* easier
* often faster

---

# 🚨 33. Exposure Bias

Teacher Forcing creates a new problem.

Training:

```text
Ground-Truth History
```

Inference:

```text
Model-Generated History
```

So:

# **Training History ≠ Inference History**

This mismatch is commonly called:

# **Exposure Bias**

---

# 🔥 34. Exposure Bias vs Error Propagation

### Exposure Bias

> model was trained mostly on correct histories.

### Error Propagation

> one wrong generated token changes future decoding.

Related but different.

---

# ⚖️ 35. Teacher Forcing Ratio

Let:

# `p_TF`

be the probability of feeding the ground-truth previous token.

```text
p_TF = 1
→ full teacher forcing

p_TF = 0
→ fully self-fed
```

Trade-off:

```text
More Teacher Forcing
→ easier training
→ larger mismatch

Less Teacher Forcing
→ harder training
→ more inference-like histories
```

---

# 🔄 36. Scheduled Sampling

One possible mitigation:

```text
Early Training
→ high teacher forcing

Later
→ gradually reduce
```

This is:

# **Scheduled Sampling**

It may reduce mismatch but does not guarantee:

> exposure bias is solved.

---

# 🚀 37. Inference — The Teacher Disappears

At inference:

> the correct target does not exist.

So generation starts with:

# `<BOS>` / `<SOS>`

Then:

```text
Predict token
↓
Select token
↓
Feed it back
↓
Predict next token
```

This is:

# **Autoregressive Decoding**

---

# 📐 38. Autoregressive Decoder Equation

During inference:

# `d_t = f_dec(e(ŷ_(t-1)), d_(t-1), c)`

where:

`ŷ_(t-1)`

is:

> the model-selected previous token.

---

# 🧠 39. Full Inference Pipeline

```text
Previous Token
↓
Embedding
↓
Decoder
↓
Hidden State
↓
Output Projection
↓
Logits
↓
Softmax
↓
Token Distribution
↓
Decoding Strategy
↓
Selected Token
↓
Feed Back
```

Memorize:

# **Predict → Select → Feed Back → Repeat**

---

# 🚫 40. Softmax Does Not Choose the Token

Softmax produces:

> probabilities.

A decoding strategy chooses the token.

Examples:

* greedy
* sampling
* Beam Search

So:

```text
Softmax
→ score normalization

Decoding Strategy
→ decision
```

---

# 🛑 41. Sequence Termination

Generation stops when:

# `<EOS>`

is selected.

But a production system also needs:

# `max_length`

So:

```text
<EOS>
→ semantic stop

max_length
→ safety stop
```

---

# 🔥 42. Autoregressive Error Propagation

Suppose correct:

```text
A B C
```

model generates:

```text
A X
```

Then future distribution becomes:

# `P(y_3 | A, X, X_source)`

rather than:

# `P(y_3 | A, B, X_source)`

So:

```text
Wrong Token
↓
Different History
↓
Different State
↓
Different Future Predictions
```

---

# 🐢 43. Why Autoregressive Inference Is Sequential

To generate:

`y_(t+1)`

we need:

`ŷ_t`.

Therefore:

```text
Step 1
↓
Step 2
↓
Step 3
```

must proceed sequentially.

Longer generation therefore usually means:

> greater inference latency.

---

# 🎯 44. Greedy Decoding

Greedy chooses:

# `ŷ_t = argmax_y P(y | y_<t, X)`

Advantages:

* simple
* fast
* deterministic
* low memory

But it optimizes:

> the next token locally.

---

# 🚨 45. Greedy Limitation

Suppose:

```text
A → 0.55
B → 0.45
```

Greedy picks:

`A`.

But:

```text
A continuation = 0.20
B continuation = 0.90
```

Then:

```text
A path = 0.11
B path = 0.405
```

So:

# **Best Token Now ≠ Best Sequence Overall**

---

# 🔦 46. Beam Search

Beam Search delays commitment.

Instead of keeping:

```text
1 hypothesis
```

it keeps:

# `k hypotheses`

at each step.

`k` is:

> beam size / beam width.

---

# ⭐ 47. Beam Size 1

# `Beam(k=1) ≈ Greedy`

Because only one path survives after every pruning step.

---

# 🔄 48. Beam Search Core Loop

Memorize:

# **Expand → Score → Rank → Prune**

```text
Current Beams
↓
Expand Every Beam
↓
Generate Candidate Extensions
↓
Update Cumulative Scores
↓
Rank
↓
Keep Top k
↓
Repeat
```

---

# 📐 49. Beam Sequence Score

Raw:

# `P(Y|X) = Π_t P(y_t | y_<t, X)`

Practically, use:

# `S(Y) = Σ_t log P(y_t | y_<t, X)`

because logs are:

* numerically stable
* easy to accumulate

Remember:

> less negative log score is better.

---

# 🛑 50. Completed Beams

If a hypothesis generates:

```text
A B <EOS>
```

it becomes:

# **completed**

and should not continue normal expansion.

Beam Search usually tracks:

* active hypotheses
* completed hypotheses

---

# 🚨 51. Beam Search Is Approximate

Beam Search prunes paths.

Once removed:

> a path is gone forever.

Therefore:

# **Beam Search does not guarantee the global optimum**

It is a compromise between:

```text
Greedy
and
Exhaustive Search
```

---

# 📏 52. Length Bias

Longer sequences naturally accumulate:

> more negative log probability.

So raw Beam Search scoring can favor:

* short outputs
* early `<EOS>`

This is:

# **length bias**

---

# 📐 53. Length Normalization

Conceptually:

# `S_norm(Y) = S(Y) / LP(Y)`

where:

`LP(Y)`

depends on sequence length.

The purpose is to:

> compensate for raw score bias toward shorter sequences.

But over-correction can cause:

> unnecessarily long outputs.

---

# 🧠 54. Search Error vs Model Error

### Search Error

The model likes the correct sequence, but decoding fails to find it.

Beam Search can help.

### Model Error

The model itself assigns bad probabilities.

Beam Search cannot fix it.

Golden distinction:

# **Better Search ≠ Better Model**

---

# ⚡ 55. Beam Width Trade-Off

```text
Beam Width ↑
↓
More Hypotheses
↓
More Exploration
↓
More Compute
↓
More Memory
↓
More Latency
```

Bigger beam is:

> not automatically better.

---

# 🎯 56. Why Attention Is Needed After Beam Search

By this point we improved:

* recurrent memory
* sequence encoding
* decoder training
* inference
* target search

But the decoder may still depend on:

```text
ONE FIXED SOURCE CONTEXT
```

Beam Search searches:

> target hypotheses.

It does not improve:

> source accessibility.

So we return to the earlier bottleneck.

---

# 🔦 57. Attention Motivation

The key question:

> Why should every target step use the same source summary?

Different target tokens need:

> different source information.

So instead of:

# `c`

use:

# `c_t`

A dynamic context for every decoder step.

---

# 📐 58. Attention Context Equation

Keep all encoder states:

# `h_1, h_2, ..., h_T`

Then construct:

# `c_t = Σ_i α_(t,i)h_i`

where:

`α_(t,i)`

represents:

> relevance of source position `i` for target step `t`.

---

# 🧠 59. Attention Mental Model

Classic Seq2Seq:

> read source, memorize it, turn around, generate.

Attention:

> keep source visible and look at the relevant region whenever needed.

So attention is better understood as:

# **dynamic retrieval**

not just:

> better memory.

---

# ⭐ 60. Soft Alignment

Suppose:

```text
α =
[0.05, 0.10, 0.75, 0.10]
```

Then the decoder focuses mostly on:

> source position 3,

while still using other source states.

This is:

# **soft attention**

---

# 🧠 61. Alignment

Attention creates:

> target-step ↔ source-position relevance.

This is called:

# **source-target alignment**

It does not have to be:

> one-to-one.

---

# 📐 62. Attention Pipeline Preview

Conceptually:

```text
Decoder Need
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

The next module will explain:

> how those relevance scores are actually computed.

---

# ⚡ 63. Attention Computational Trade-Off

For:

* source length `T`
* target length `T'`

recurrent encoder-decoder attention conceptually considers about:

# `T × T'`

source-target relationships.

So:

```text
Better Source Access
↔
More Computation
```

---

# 🚫 64. Attention Is Not the Same as Other Fixes

### LSTM

> recurrent memory

### Teacher Forcing

> training target history

### Beam Search

> target sequence search

### Attention

> dynamic source access

These mechanisms solve:

> different problems.

---

# 🧠 65. Master Problem Taxonomy

| Problem                                         | Mechanism               |
| ----------------------------------------------- | ----------------------- |
| No sequence memory                              | RNN                     |
| Long-term recurrent dependency                  | LSTM / GRU              |
| Need future input context                       | Bidirectional RNN       |
| Variable-length source → variable-length target | Encoder–Decoder         |
| Fixed source communication                      | Attention               |
| Decoder unstable during training                | Teacher Forcing         |
| Train/inference history mismatch                | Exposure Bias           |
| Free-running generation                         | Autoregressive Decoding |
| Greedy local commitment                         | Beam Search             |
| Target search error                             | Wider/better search     |
| Model assigns wrong probabilities               | Better model/training   |
| Source accessibility                            | Attention               |

---

# 📐 66. Core Formula Collection

## RNN

# `h_t = tanh(W_xh x_t + W_hh h_(t-1) + b_h)`

---

## LSTM Cell

# `c_t = f_t ⊙ c_(t-1) + i_t ⊙ g_t`

---

## GRU

# `h_t = z_t ⊙ h_(t-1) + (1-z_t) ⊙ h̃_t`

---

## Seq2Seq

# `P(Y|X) = Π_t P(y_t | y_<t, X)`

---

## Teacher-Forced Loss

# `L = -Σ_t log P(y_t* | y_<t*, X)`

---

## Autoregressive Decoder

# `d_t = f_dec(e(ŷ_(t-1)), d_(t-1), c)`

---

## Greedy

# `ŷ_t = argmax_y P(y | y_<t, X)`

---

## Beam Score

# `S(Y) = Σ_t log P(y_t | y_<t, X)`

---

## Attention Context

# `c_t = Σ_i α_(t,i)h_i`

---

# 🚫 67. Module-Wide High-Yield Mistakes

### ❌ Hidden state stores every past token exactly

No.

It is a learned compressed representation.

### ❌ BPTT = backward RNN

No.

### ❌ Gradient clipping solves vanishing gradients

No.

### ❌ LSTM completely eliminates long-term dependency problems

No.

### ❌ GRU update-gate convention is universal

No.

Check the equation.

### ❌ Bidirectional models are causal

Not in standard full-sequence form.

### ❌ Encoder–Decoder and Seq2Seq are conceptually identical terms

Closely related, but architecture vs full task/model.

### ❌ Teacher Forcing gives the current answer

No.

Gives previous correct target.

### ❌ Teacher Forcing removes autoregression

No.

### ❌ Softmax selects the generated token

No.

### ❌ Greedy finds the globally best sequence

No.

### ❌ Beam Search guarantees global optimum

No.

### ❌ Larger beam is always better

No.

### ❌ Beam Search fixes model error

No.

### ❌ Attention is just a bigger context vector

No.

### ❌ Attention was only invented for vanishing gradients

No.

---

# 🎤 68. Most Important Interview Connections

A strong interview explanation should connect concepts instead of defining them independently.

For example:

> **Vanilla RNNs introduced recurrent hidden state to model sequence history, but BPTT over long sequences caused vanishing and exploding gradients. LSTM and GRU improved long-term information flow using gates. Encoder-decoder architectures then enabled variable-length sequence mapping, but classic Seq2Seq compressed the entire source into a fixed context vector. Teacher Forcing stabilized decoder training, while autoregressive decoding handled inference and Beam Search reduced greedy search errors. None of those solved the source-access bottleneck, which motivated attention: retaining encoder states and dynamically constructing a different source context for every decoder step.**

That answer tells:

> the entire module story.

---

# 🧠 69. 15 Things You Must Know Cold

1. **RNN introduces recurrent state.**
2. **BPTT trains recurrent models through time.**
3. **Repeated Jacobian products cause vanish/explode problems.**
4. **LSTM creates gated cell-state memory.**
5. **GRU simplifies gated recurrence.**
6. **BiRNN uses past and future source context.**
7. **Encoder–Decoder separates source understanding from target generation.**
8. **Seq2Seq factorizes `P(Y|X)` autoregressively.**
9. **Classic fixed context creates a source bottleneck.**
10. **Teacher Forcing gives correct previous target history during training.**
11. **Exposure bias is training vs inference history mismatch.**
12. **Autoregressive inference feeds generated tokens back.**
13. **Greedy optimizes locally.**
14. **Beam Search keeps multiple sequence hypotheses.**
15. **Attention dynamically retrieves source information per decoder step.**

---

# ⭐ 70. Staff-Level Mental Model

A senior engineer should not think of this module as:

```text
RNN
LSTM
GRU
Seq2Seq
Teacher Forcing
Beam Search
Attention
```

Think of it as a sequence of system constraints:

```text
Need temporal state
→ RNN

Need long-range credit assignment
→ gated recurrence

Need variable-length output
→ encoder-decoder

Need stable autoregressive training
→ Teacher Forcing

Need deployment generation
→ autoregressive inference

Need better sequence search
→ Beam Search

Need better source access
→ Attention
```

Each technique changes:

> a different part of the system.

---

# 🧠 Final Module Mental Model

The entire module in one picture:

```text
SOURCE SEQUENCE
      │
      ▼
   ENCODER
      │
      ├── recurrent state
      │
      ├── LSTM / GRU memory
      │
      └── possibly bidirectional
      │
      ▼
SOURCE REPRESENTATION
      │
      ▼
   DECODER
      │
      ├── Training:
      │      Teacher Forcing
      │
      ├── Inference:
      │      Autoregressive Feedback
      │
      ├── Search:
      │      Greedy / Beam Search
      │
      └── Source Access:
             Fixed Context
                  ↓
             BOTTLENECK
                  ↓
              ATTENTION
```

And the architectural evolution:

```text
ONE FIXED SOURCE SUMMARY
↓
Does not scale well with information demand
↓
Keep Encoder States Accessible
↓
Retrieve Relevant Source Information Per Step
↓
ATTENTION
```

---

# 🚀 Where We Go Next

We have now completed the motivation for attention.

We know:

# `c_t = Σ_i α_(t,i)h_i`

But one key question remains unanswered:

# **Where do `α_(t,i)` come from?**

We now need to understand:

1. How the decoder compares itself to an encoder state.
2. What an attention score is.
3. How scores are normalized.
4. How weighted context is produced.
5. Additive/Bahdanau attention.
6. Multiplicative/Luong attention.
7. Why dot products become so important.
8. How these ideas eventually become Query, Key, and Value.

So the next learning progression is:

```text
Attention Motivation
↓
Attention Mechanism
↓
Attention Scores
↓
Softmax Weights
↓
Weighted Context
↓
Self-Attention
↓
Query / Key / Value
↓
Scaled Dot-Product Attention
↓
Multi-Head Attention
↓
Transformer
```

---

# ⭐ Module Golden Rule

> **The evolution of sequence models is a story of removing bottlenecks one by one: RNNs added memory, LSTM and GRU improved long-range information flow, encoder-decoder models enabled sequence transformation, Teacher Forcing stabilized training, autoregressive decoding enabled generation, Beam Search improved target search, and attention finally changed source access from one fixed compressed summary into dynamic retrieval.**
