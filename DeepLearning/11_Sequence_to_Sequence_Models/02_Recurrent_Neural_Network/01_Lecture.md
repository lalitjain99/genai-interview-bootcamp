# 📘 Lecture — Recurrent Neural Network

> **Core Idea:** A Recurrent Neural Network processes a sequence one step at a time while carrying forward a **hidden state** that summarizes previous context.

---

# 🎯 1. Why Do We Need an RNN?

In the previous lecture, we saw that sequence data has a key property:

> **the current element may depend on previous elements.**

For example:

```text
The movie was not ...
```

To understand the next word or final sentiment, the model should remember:

> what came before.

A normal feed-forward network has no persistent state across sequence positions.

So we need something like:

```text
Current Input
+
Previous Context
↓
Current Representation
```

That is exactly what an RNN introduces.

---

# 🧠 2. The Basic RNN Idea

An RNN processes a sequence:

`x1, x2, x3, ..., xT`

one element at a time.

At each time step `t`, it receives:

* current input `x_t`
* previous hidden state `h_{t-1}`

and produces:

* new hidden state `h_t`

Conceptually:

`h_t = f(x_t, h_{t-1})`

This is the central equation of recurrent neural networks.

---

# ⭐ 3. What Is the Hidden State?

The hidden state is:

> **a learned running representation of the sequence seen so far.**

For example:

```text
I
↓
h1

love + h1
↓
h2

deep + h2
↓
h3

learning + h3
↓
h4
```

So `h3` contains a learned summary influenced by:

```text
I → love → deep
```

It does not literally store the words.

It stores:

> a compressed learned representation.

---

# 🔄 4. RNN as a Loop

A compact RNN is often drawn like this:

```text
       ┌──────────────┐
       │              │
       ↓              │
x_t → RNN Cell → h_t ─┘
```

The output hidden state is fed back into the same cell for the next time step.

This recurrence is why it is called:

> **Recurrent Neural Network**

---

# 🧩 5. Unrolling the RNN

The loop becomes easier to understand if we unroll it through time.

```text
h0
↓
x1 → RNN → h1
             ↓
x2 → RNN → h2
             ↓
x3 → RNN → h3
             ↓
x4 → RNN → h4
```

Or more explicitly:

```text
h0 ──→ [RNN] ──→ h1 ──→ [RNN] ──→ h2 ──→ [RNN] ──→ h3
        ↑                 ↑                 ↑
        x1                x2                x3
```

Important:

> these are not different RNN cells with different parameters.

They are:

> **the same recurrent transformation reused across time.**

---

# 🔄 6. Parameter Sharing Across Time

This connects directly to our CNN module.

CNN:

```text
Same Filter
→ reused across spatial positions
```

RNN:

```text
Same Recurrent Weights
→ reused across time steps
```

So:

```text
CNN → parameter sharing across SPACE

RNN → parameter sharing across TIME
```

This is one of the most important parallels between CNNs and RNNs.

---

# 🧮 7. The Vanilla RNN Equation

A common RNN hidden-state equation is:

`h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h)`

Let's break it down.

---

# 🎛️ 8. `W_xh x_t`

This term processes:

> the current input.

`W_xh`

maps:

`x_t`

into the hidden-state space.

So:

```text
Current Input
↓
Input-to-Hidden Transformation
```

---

# 🔁 9. `W_hh h_{t-1}`

This term processes:

> previous hidden state.

`W_hh`

maps previous memory:

`h_{t-1}`

into the new hidden-state computation.

So:

```text
Previous Context
↓
Hidden-to-Hidden Transformation
```

This is what introduces:

> recurrence.

---

# ➕ 10. Bias

Then:

`b_h`

is added.

So before activation:

`a_t = W_xh x_t + W_hh h_{t-1} + b_h`

This is sometimes called the:

> pre-activation.

---

# ⚡ 11. Why `tanh`?

Traditional vanilla RNNs commonly use:

`tanh`

So:

`h_t = tanh(a_t)`

`tanh` maps values roughly into:

`[-1, 1]`

This keeps the hidden representation bounded.

Historically, `tanh` was a common choice for recurrent state updates.

Other recurrent architectures may use different nonlinearities.

---

# 🧠 12. Full RNN Cell

So at one time step:

```text
x_t ─────────────┐
                 ↓
              W_xh
                 ↓
              (+) ──→ tanh ──→ h_t
                 ↑
              W_hh
                 ↑
h_{t-1} ─────────┘
```

Mathematically:

`h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h)`

---

# 🎯 13. Why Combine Current Input and Previous State?

Because the model wants to answer:

> “What should my current representation be, given what I see now and what I already know?”

So:

```text
Current Evidence
+
Previous Context
↓
Updated Context
```

This is the essential RNN mechanism.

---

# 🧠 14. A Simple Example

Suppose we process:

```text
I → love → AI
```

At `t=1`:

```text
x1 = "I"
h0 = initial state

h1 = f(x1, h0)
```

At `t=2`:

```text
x2 = "love"

h2 = f(x2, h1)
```

At `t=3`:

```text
x3 = "AI"

h3 = f(x3, h2)
```

So:

`h3`

depends indirectly on:

* `AI`
* `love`
* `I`

through the recurrent chain.

---

# ⭐ 15. Hidden State Carries Context Forward

This creates:

```text
x1
↓
h1
   ↘
    x2
     ↓
     h2
        ↘
         x3
          ↓
          h3
```

Therefore later states can encode information from:

> earlier sequence positions.

This is what a basic feed-forward network does not naturally provide.

---

# 📐 16. Hidden State Dimensions

Suppose:

`x_t ∈ R^D`

and hidden state:

`h_t ∈ R^H`

Then:

`W_xh`

must map:

`D → H`

So:

`W_xh.shape = H × D`

And:

`W_hh`

maps:

`H → H`

So:

`W_hh.shape = H × H`

Bias:

`b_h.shape = H`

---

# 🧮 17. Parameter Count of a Vanilla RNN

Given:

* input dimension = `D`
* hidden dimension = `H`

RNN hidden-state parameters are:

### Input-to-hidden

`H × D`

### Hidden-to-hidden

`H × H`

### Bias

`H`

Total:

`HD + H² + H`

If there is also an output layer, that adds more parameters.

---

# ⭐ 18. Sequence Length Does NOT Increase RNN Parameter Count

Suppose sequence length is:

`10`

or:

`100`

or:

`1000`

The same:

* `W_xh`
* `W_hh`
* `b_h`

are reused at every step.

So parameter count does NOT scale with:

> sequence length.

This is the recurrent equivalent of CNN parameter sharing.

---

# ⚡ 19. But Compute DOES Increase With Sequence Length

Even though parameters are reused:

> the recurrent cell still has to execute at each time step.

So:

```text
Longer Sequence
↓
More Recurrent Steps
↓
More Compute
```

Again:

> **same parameters ≠ same computation**

---

# 🎯 20. Initial Hidden State `h0`

At the first time step, there is no previous sequence state.

So we need:

`h0`

Commonly:

`h0 = 0`

a zero vector.

But in some systems:

> the initial state may be learned or supplied from another model.

For now:

> assume zero initialization.

---

# 🧠 21. Does Zero `h0` Create the Same Symmetry Problem as Zero Weights?

No.

This is a useful distinction.

We previously learned:

> zero initialization of trainable hidden weights can cause symmetry problems.

But `h0` is typically:

> an activation/state value, not the shared trainable weight matrix.

So initializing the hidden state with zeros is perfectly normal.

---

# 📤 22. RNN Can Also Produce an Output

An RNN may produce:

`y_t`

at each time step.

For example:

`y_t = W_hy h_t + b_y`

Then:

```text
x_t
↓
RNN
↓
h_t
↓
Output Layer
↓
y_t
```

Depending on the task, we may:

* use every `y_t`
* use only the final state/output
* decode another sequence

---

# 🧠 23. Many-to-One RNN

Example:

> sentiment classification

Input:

```text
This → movie → was → excellent
```

We may process the whole sequence and use:

`h_T`

the final hidden state.

```text
x1 → h1
x2 → h2
x3 → h3
x4 → h4
         ↓
      Classifier
         ↓
      Positive
```

This is:

> **many-to-one**

---

# 🎯 24. Why Use Final Hidden State?

The hope is:

> `h_T` summarizes the important information from the sequence.

Then:

```text
h_T
↓
Linear Layer
↓
Class
```

This works reasonably for some shorter/simple sequence tasks.

But it creates a limitation:

> the entire sequence must be represented in one fixed-size vector.

That will matter later.

---

# 🧠 25. Many-to-Many RNN

For sequence tagging:

```text
John → lives → in → London
 ↓       ↓      ↓      ↓
PER      O      O     LOC
```

We need one output per input position.

So:

```text
h1 → y1
h2 → y2
h3 → y3
h4 → y4
```

This is:

> many-to-many with aligned lengths.

---

# 🔄 26. One-to-Many

An RNN can also generate a sequence from one initial input.

Example:

```text
Image Features
↓
RNN
↓
A
↓
dog
↓
runs
↓
...
```

Historically, this idea was used in:

> image captioning.

---

# 🎯 27. Seq2Seq Preview

For translation:

```text
English sequence
↓
Encoder RNN
↓
Context
↓
Decoder RNN
↓
French sequence
```

Both encoder and decoder can be recurrent models.

We will study this later.

For now, understand the basic RNN cell first.

---

# 🧠 28. Does an RNN Output Depend Only on the Current Input?

No.

Because:

`h_t = f(x_t, h_{t-1})`

and:

`h_{t-1}`

depends on:

`h_{t-2}`

and so on.

Therefore:

`h_t`

is indirectly a function of:

`x1, x2, ..., x_t`

Conceptually:

`h_t = f(x_t, x_{t-1}, ..., x_1)`

through recurrence.

---

# 🔗 29. Computational Graph View

For three time steps:

```text
x1      x2      x3
↓       ↓       ↓
h1 →    h2 →    h3
```

But each `h_t` is a computation node.

So when training:

> gradients must travel backward through this chain.

This leads directly to:

> **Backpropagation Through Time**

our next major lecture.

---

# ⚠️ 30. RNN Is Sequential by Nature

To compute:

`h3`

we need:

`h2`.

To compute:

`h2`

we need:

`h1`.

Therefore:

```text
h1
↓
h2
↓
h3
↓
h4
```

cannot be fully parallelized across time in the same way as a Transformer training pass.

This creates:

> a sequential-computation bottleneck.

---

# 🧠 31. Why Sequential Computation Matters

For a sequence of length:

`1000`

the recurrence must conceptually process:

```text
1 → 2 → 3 → ... → 1000
```

This limits:

* training parallelism
* hardware utilization
* scalability to very long sequences

This becomes one major reason:

> Transformers eventually become dominant.

---

# ⏳ 32. RNN Memory Is Compressed

Hidden state has fixed dimension:

`H`

regardless of whether the sequence contains:

* 10 tokens
* 100 tokens
* 1000 tokens

So the RNN repeatedly compresses history into:

> a fixed-size vector.

Conceptually:

```text
x1
↓
h1

x1,x2
↓
h2

x1,x2,x3
↓
h3

...
```

Each new state must retain what is useful from previous history.

---

# ⚠️ 33. Why Long-Term Memory Becomes Difficult

Suppose:

```text
The book that I borrowed from the professor,
who moved here from France many years ago,
was ...
```

A later prediction may depend on something much earlier.

The information has to survive:

```text
h1 → h2 → h3 → ... → hT
```

Repeated transformations can cause earlier information to:

> fade or become distorted.

This is one part of the long-term dependency problem.

---

# 🔄 34. Gradient Problem Preview

During backward propagation, gradients also travel through:

```text
h_T
↓
h_{T-1}
↓
h_{T-2}
↓
...
```

The same recurrent weight matrix appears repeatedly.

This means gradients involve repeated multiplication.

Depending on those values, gradients can:

* shrink dramatically
* grow dramatically

leading to:

> vanishing or exploding gradients.

---

# 🧠 35. Why `tanh` Can Contribute to Vanishing Gradients

`tanh` has derivative:

> less than or equal to 1 in magnitude

and becomes very small in saturated regions.

During many recurrent steps:

```text
small derivative
×
small derivative
×
small derivative
×
...
```

can produce:

> extremely small gradients.

We will derive this carefully in the BPTT and gradient lectures.

---

# 🎯 36. The Long-Term Dependency Problem

Basic RNNs work reasonably for:

> short dependencies.

They struggle when important information must survive over:

> many time steps.

This is exactly what motivated:

* LSTM
* GRU

These architectures introduce:

> gating mechanisms

to manage information flow.

---

# ⭐ 37. Vanilla RNN vs LSTM / GRU Preview

### Vanilla RNN

```text
x_t + h_{t-1}
↓
tanh
↓
h_t
```

### LSTM

Adds mechanisms to decide:

```text
What to Forget?
What to Store?
What to Output?
```

### GRU

Uses a simpler gating structure to decide:

```text
What to Keep?
What to Update?
```

We will get there progressively.

---

# 🧠 38. Hidden State Is Not “Human Memory”

Avoid thinking:

> `h_t` literally remembers sentences the way a person does.

It is simply:

> a learned numerical vector.

Its values do not have fixed predefined meanings like:

```text
h[0] = subject
h[1] = verb
h[2] = sentiment
```

The representation is:

> distributed and learned.

---

# 📐 39. Example Dimensions

Suppose:

`x_t.shape = 300`

and:

`hidden_size = 128`

Then:

```text
W_xh.shape = 128×300
W_hh.shape = 128×128
b_h.shape   = 128
h_t.shape   = 128
```

At every time step:

> the same matrices are reused.

---

# 🧮 40. Parameter Example

Given:

`D = 300`

`H = 128`

Input-to-hidden:

`128 × 300`

`= 38,400`

Hidden-to-hidden:

`128 × 128`

`= 16,384`

Bias:

`128`

Total recurrent parameters:

`38,400 + 16,384 + 128`

`= 54,912`

And this remains:

> 54,912

whether sequence length is:

`10` or `1000`.

---

# 🔁 41. Batch Processing

Suppose batch size:

`B = 32`

sequence length:

`T = 20`

input dimension:

`D = 300`

Conceptually an input tensor may look like:

`B × T × D`

or framework-specific variants such as:

`T × B × D`.

The recurrent cell processes each:

> sequence step

while handling the batch dimension in parallel.

---

# ⚡ 42. Time Is Sequential, Batch Is Parallel

Important distinction.

Within a batch:

> multiple examples can be processed together.

But within one recurrent sequence:

```text
t1 → t2 → t3 → ...
```

hidden-state dependencies remain sequential.

So RNNs have:

> batch parallelism

but limited:

> time-step parallelism.

---

# 🧠 43. Variable-Length Sequences

RNNs can conceptually process sequences of different lengths because:

> the same recurrent cell is reused.

For batching, we often use:

* padding
* masks
* packed sequence utilities

The underlying recurrence itself does not require:

> one globally fixed sequence length.

---

# 🎭 44. Padding Must Not Become Real Context

Suppose:

```text
I love AI <PAD> <PAD>
```

If padding is processed as ordinary meaningful data:

> it can distort hidden states.

So training systems usually use masking or sequence-length information so that:

> padded positions are handled appropriately.

---

# 🎯 45. Causal Nature of a Standard Forward RNN

A standard forward RNN at time `t` has access to:

```text
x1, x2, ..., x_t
```

but not:

```text
x_{t+1}, x_{t+2}, ...
```

Therefore it is naturally:

> forward / causal with respect to sequence order.

For some tasks, future context is also useful.

That motivates:

> Bidirectional RNNs.

---

# 🔁 46. Bidirectional RNN Preview

Instead of only:

```text
left → right
```

we can process:

```text
left → right
+
right → left
```

Then each position gets context from:

> past and future.

Useful for:

* tagging
* sequence understanding
* contextual encoding

But not appropriate in the same way for:

> causal next-token generation.

---

# 🧠 47. RNN vs Feed-Forward Network

| Property                                   | Feed-Forward   | RNN     |
| ------------------------------------------ | -------------- | ------- |
| Persistent state                           | No             | Yes     |
| Sequence order built in                    | Weak/No        | Yes     |
| Parameters shared over time                | No             | Yes     |
| Handles variable sequence length naturally | Limited        | Yes     |
| Current output can depend on history       | Not inherently | Yes     |
| Time-step parallelism                      | High           | Limited |

---

# 🔗 48. RNN vs CNN

| Property         | CNN                      | RNN                    |
| ---------------- | ------------------------ | ---------------------- |
| Main structure   | Spatial                  | Sequential             |
| Sharing          | Across space             | Across time            |
| State recurrence | No                       | Yes                    |
| Locality         | Spatial neighborhoods    | Temporal progression   |
| Main strength    | Spatial patterns         | Ordered dependencies   |
| Parallelism      | High spatial parallelism | Sequential across time |

---

# ⚠️ 49. Common Misconception — RNN Has Different Weights at Every Time Step

Wrong:

```text
W1 for t1
W2 for t2
W3 for t3
```

Standard RNN:

```text
Same W
↓
t1
t2
t3
...
```

This parameter sharing is essential.

---

# ⚠️ 50. Common Misconception — Hidden State Is the Output

Not necessarily.

`h_t`

is:

> internal state / representation.

The model may produce:

`y_t`

from:

`h_t`.

For example:

`y_t = W_hy h_t + b_y`

So:

```text
Hidden State
≠
Task Output
```

although some systems may directly use hidden states as representations.

---

# ⚠️ 51. Common Misconception — Final Hidden State Contains Everything Perfectly

No.

The final state is:

> a fixed-size learned summary.

It can lose information, especially for long sequences.

This limitation becomes central in:

> encoder-decoder Seq2Seq.

---

# ⚠️ 52. Common Misconception — RNN Solves Long-Term Dependencies Completely

No.

Basic RNNs actually struggle with:

> long-range dependencies.

RNN gives us:

> sequential memory.

LSTM/GRU were developed to improve:

> long-term information retention and gradient behavior.

---

# 🔄 53. How an RNN Learns

Training loop still follows:

```text
Forward
↓
Prediction
↓
Loss
↓
Backpropagation
↓
Gradient
↓
Optimizer
↓
Parameter Update
```

The difference:

> recurrent parameters are reused many times in the forward graph.

Therefore backward must account for:

> every time-step use.

This is BPTT.

---

# ⭐ 54. Shared Parameter Gradient Again

Suppose `W_hh` is reused at:

```text
t1
t2
t3
t4
```

During backward:

> gradient contributions from all these uses accumulate into one `dL/dW_hh`.

This is very similar to CNN shared-filter backpropagation:

```text
CNN:
Many spatial uses
→ one filter gradient

RNN:
Many temporal uses
→ one recurrent-weight gradient
```

---

# 🧠 55. RNN Computational Graph

Forward:

```text
h0
↓
[h1]
↓
[h2]
↓
[h3]
↓
Loss
```

Backward:

```text
Loss
↓
h3
↓
h2
↓
h1
↓
shared recurrent parameters
```

Because the graph extends through time:

> training can become challenging for long sequences.

---

# 🎯 56. When Were RNNs Useful?

RNNs historically became important for:

* language modeling
* machine translation
* speech recognition
* handwriting recognition
* time-series prediction
* sequence labeling

They were later enhanced by:

* LSTM
* GRU
* attention

before Transformers became dominant in many NLP tasks.

---

# ⚠️ 57. RNNs Are Not Obsolete in Every Context

Transformers dominate many modern sequence applications, but RNN-like models may still be relevant when:

* models must be small
* streaming is important
* data arrives sequentially
* latency per incremental step matters
* limited compute/memory is available
* specific time-series tasks benefit from recurrent structure

So:

> architecture choice remains task-dependent.

---

# 🧠 58. Streaming Is a Natural RNN Strength

Suppose sensor values arrive:

```text
x1
then x2
then x3
...
```

An RNN can maintain:

`h_t`

and update it as new data arrives.

It does not necessarily need to repeatedly process:

> the entire past sequence.

This makes recurrence conceptually attractive for:

> streaming systems.

---

# 🎯 59. What Should You Remember About Vanilla RNN?

The entire model boils down to:

`h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h)`

And four ideas:

1. current input
2. previous hidden state
3. shared recurrent parameters
4. updated hidden state

Everything else grows from this.

---

# 🧠 60. The Core RNN Story

```text
Sequence Element x_t
+
Previous State h_{t-1}
↓
Shared Recurrent Transformation
↓
New State h_t
↓
Carry Forward
```

Repeat:

```text
t1 → t2 → t3 → ... → tT
```

---

# 🎤 61. Interview Question — What Is an RNN?

A strong answer:

> **A Recurrent Neural Network is a neural architecture designed for sequential data. At each time step it combines the current input with a hidden state from the previous step to produce an updated hidden state. The same recurrent parameters are reused across all time steps, allowing the network to process variable-length sequences while carrying contextual information forward.**

---

# 🎤 62. Interview Question — What Is Hidden State?

> **The hidden state is a learned fixed-dimensional representation that carries information from previous sequence positions into the current computation. It acts as the recurrent model's running context, although it is a compressed representation rather than an exact copy of the entire history.**

---

# 🎤 63. Interview Question — Why Share Weights Across Time?

> **Weight sharing allows the same transformation to process sequence elements regardless of their position, keeps parameter count independent of sequence length, and gives the model a consistent rule for updating its hidden state across time.**

---

# 🎤 64. Interview Question — Why Do Vanilla RNNs Struggle With Long Sequences?

> **Information and gradients must pass through many recurrent transformations. Repeated multiplication through recurrent weights and activation derivatives can cause gradients to shrink or grow dramatically, leading to vanishing or exploding gradients and making long-range dependencies difficult to learn.**

---

# ⚡ 65. Quick Recall Table

| Concept        | Key Idea                            |
| -------------- | ----------------------------------- |
| RNN            | Neural network with recurrent state |
| `x_t`          | Current input                       |
| `h_{t-1}`      | Previous hidden state               |
| `h_t`          | Updated hidden state                |
| Recurrence     | Previous state feeds next step      |
| Weight sharing | Same parameters across time         |
| `W_xh`         | Input → hidden                      |
| `W_hh`         | Hidden → hidden                     |
| `h0`           | Initial hidden state                |
| Many-to-One    | Sequence → one output               |
| Many-to-Many   | Sequence → sequence outputs         |
| BPTT           | Backprop through unrolled time      |
| Limitation     | Long-term dependencies              |
| LSTM/GRU       | Gated recurrent improvements        |

---

# 🧠 66. Key Equation Sheet

### Hidden Pre-Activation

`a_t = W_xh x_t + W_hh h_{t-1} + b_h`

### Hidden State

`h_t = tanh(a_t)`

### Optional Output

`y_t = W_hy h_t + b_y`

### Parameter Count

`HD + H² + H`

for the recurrent hidden-state computation.

---

# 🎯 67. Full Mental Model

```text
                   ┌────────────────────┐
                   │                    │
                   │                    ↓
x_t ──→ Input Transform ──┐         Previous State
                          │              │
                          ↓              ↓
                         SUM ← Hidden Transform
                          ↓
                        tanh
                          ↓
                         h_t
                          │
              ┌───────────┴───────────┐
              ↓                       ↓
        Task Output              Next Time Step
```

---

# 🔗 68. Where This Leads Next

We now understand the forward RNN:

```text
x_t + h_{t-1}
↓
h_t
```

But we have not answered:

> **How does the RNN learn its recurrent weights when the same weights are reused over many time steps?**

To answer that, we need to:

1. unroll the RNN
2. construct the computational graph
3. propagate loss gradients backward through time
4. accumulate gradients for shared recurrent weights

That is:

# **Backpropagation Through Time — BPTT**

---

# ⭐ Golden Rule

> **An RNN processes a sequence by repeatedly combining the current input with a hidden representation of the past, using the same learned parameters at every time step.**
